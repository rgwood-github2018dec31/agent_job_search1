"""Per-run OpenRouter endpoint pin for the single-turn model (extract, rate, classifiers).

Chosen once per run from the tool server's `rank_providers`, which OWNS the ranking rule: drop the
slowest share by measured throughput and the priciest share, then return the cheapest survivor,
followed by the demoted endpoints as last-resort fallbacks. This module keeps only what is the
caller's own — the single-turn token mix, the precision floor, and the per-run pin/failure state —
so it and the scraper's pin no longer carry two drifting copies of the same policy.

A call that fails on the pinned endpoint drops it for the rest of the run and the next-cheapest
survivor takes over; with none left, calls go unpinned (OpenRouter's choice). The failed call itself
is asked once of the new endpoint before its caller falls back (`endpoint_failed` returns True). A failure to reach
the local tool server itself is not the endpoint's and drops nothing: on 2026-10-20 the server
died mid-run and each refused connection dropped one more endpoint, the whole list in 18 s.

Unpinned until `pin_single_turn_endpoints()` runs, so interactive mode and tests route as before.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from http import HTTPStatus

import httpx
from utils_tools_n_agents_common.mcp_client import call_mcp_tool

from agentic_job_search.config import (
    LLM_MCP_SINGLE_TURN_TIMEOUT_SECONDS,
    LLM_OPENROUTER_MCP_URL,
    SINGLE_TURN_CACHED_PROMPT_SHARE,
    SINGLE_TURN_COMPLETION_TOKENS_PER_PROMPT_TOKEN,
    SINGLE_TURN_DROP_PRICIEST_SHARE,
    SINGLE_TURN_DROP_SLOWEST_SHARE,
    SINGLE_TURN_MIN_BITS_PER_WEIGHT,
)
from agentic_job_search.text_budget import snippet

logger = logging.getLogger(__name__)

# `requests` renders a raised-for status as "402 Client Error: Payment Required for url: ...";
# the OpenRouter body itself carries {"error": {"code": 402, ...}}. Both are matched, and the
# match is on the STATUS, never on the reason phrase -- the phrase is upstream's text and can
# change without notice, while the code is the contract.
_HTTP_STATUS_RE = re.compile(r'\b(\d{3})\s+(?:Client|Server)\s+Error\b')
_JSON_STATUS_RE = re.compile(r'"(?:code|status|status_code)"\s*:\s*(\d{3})\b')


def http_status_of(error_text: str) -> int | None:
    '''The HTTP status named in a provider error string, or None if it names none.'''
    for pattern in (_HTTP_STATUS_RE, _JSON_STATUS_RE):
        if match := pattern.search(error_text):
            return int(match.group(1))
    return None


def provider_of(tag: str) -> str:
    '''The provider an endpoint tag belongs to: `inference-net` for `inference-net/us`.'''
    return tag.split('/', 1)[0]


def is_rate_limited(error: str) -> bool:
    '''True for a 429. On a pinned call (fallbacks off) it is the provider's limit, and it holds for
    every endpoint of that provider: on 2026-10-22 InferenceNet's shared pool throttled
    `inference-net/us`, the pin moved to `inference-net/fp8`, and that failed the same way.'''
    return http_status_of(error) == HTTPStatus.TOO_MANY_REQUESTS


def price_text(price: float | None) -> str:
    '''A ranked price for the log lines. The server reports None only when NO endpoint of the
    model publishes a price, which is then not a number to print.'''
    return 'n/a' if price is None else f'${price:.4f}/M'


@dataclass
class EndpointPin:
    model: str
    ranked: list[tuple[float | None, dict]] = field(default_factory=list)

    def routing(self) -> dict:
        '''chat() arguments that serve a call on the pinned endpoint; {} when unpinned.'''
        if not self.ranked:
            return {}
        endpoint = self.ranked[0][1]
        return {'provider': endpoint['tag'], 'quantization': endpoint.get('quantization')}

    def drop(self, tag: str, error: str) -> None:
        if not self.ranked or self.ranked[0][1]['tag'] != tag:
            return   # already moved on, e.g. by an earlier failure
        self.ranked.pop(0)
        skipped = []
        if is_rate_limited(error):
            skipped = [e['tag'] for _, e in self.ranked if provider_of(e['tag']) == provider_of(tag)]
            self.ranked = [(p, e) for p, e in self.ranked if provider_of(e['tag']) != provider_of(tag)]
        if skipped:
            logger.warning(f'Single-turn pin: {provider_of(tag)} is rate-limited, so its other '
                           f'endpoint(s) {", ".join(skipped)} are dropped too')
        after = (f"switched to {self.ranked[0][1]['tag']} (est. {price_text(self.ranked[0][0])})"
                 if self.ranked else 'no endpoint left, so calls go unpinned')
        logger.warning(f'Single-turn pin: {self.model} call on {tag} failed ({snippet(error)}); '
                       f'{after} for the rest of the run')


_pins: dict[str, EndpointPin] = {}


async def pin_single_turn_endpoints(model: str) -> None:
    """Choose this run's endpoint for `model`. A failure to choose is logged and leaves it unpinned."""
    try:
        raw = await call_mcp_tool(LLM_OPENROUTER_MCP_URL, 'rank_providers', {
            'model': model,
            'min_bits': SINGLE_TURN_MIN_BITS_PER_WEIGHT,
            'cached_prompt_share': SINGLE_TURN_CACHED_PROMPT_SHARE,
            'completion_tokens_per_prompt_token': SINGLE_TURN_COMPLETION_TOKENS_PER_PROMPT_TOKEN,
            'drop_slowest_share': SINGLE_TURN_DROP_SLOWEST_SHARE,
            'drop_priciest_share': SINGLE_TURN_DROP_PRICIEST_SHARE,
        }, timeout_seconds=LLM_MCP_SINGLE_TURN_TIMEOUT_SECONDS)
        ranking = json.loads(raw)
        if not isinstance(ranking, dict) or ranking.get('ok') is not True:
            raise RuntimeError(f'rank_providers refused: {snippet(raw)}')
    except Exception as ex:
        _pins.pop(model, None)
        logger.warning(f'Single-turn pin: could not rank providers for {model} ({type(ex).__name__}: {ex}); '
                       f"running unpinned, so the provider and its price are OpenRouter's choice")
        return
    # The server ranked and reported; the pin only walks the list, carrying the price the server
    # ranked each entry by so the pin's estimate and the server's order cannot disagree.
    ranked = [(e['effective_price_per_million'], e) for e in ranking.get('ranked') or []]
    _pins[model] = EndpointPin(model, ranked)
    if ranked:
        logger.info(f'Single-turn pin: {model} -> {ranked[0][1]["tag"]} ({ranking.get("report") or ""})')
    else:
        logger.warning(f'Single-turn pin: no endpoint survived for {model}, running unpinned '
                       f'({ranking.get("report") or ""})')


def routing_for(model: str) -> dict:
    pin = _pins.get(model)
    return pin.routing() if pin else {}


# Failures between this process and the local tool server, which never reached the endpoint.
# A read timeout is NOT here: the server was waiting on the endpoint, so the endpoint was slow.
_LOCAL_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)


def endpoint_failed(model: str, routing: dict, error: str | BaseException) -> bool:
    '''Report a failed call made with `routing`; drops that endpoint when it is still the pin,
    unless `error` is a failure to reach the local tool server.

    Returns True when the endpoint was at fault and the model now routes elsewhere, so the caller
    should ask the same call once more with `routing_for(model)` before falling back. Until
    2026-10-26 the failed call went straight to its fallback: a 429 on the run's first call moved
    the pin to morph, yet that call (a company-name match) was answered by Anthropic instead.'''
    if not routing or not (pin := _pins.get(model)):
        return False
    if isinstance(error, _LOCAL_TRANSPORT_ERRORS):
        logger.info(f"Single-turn pin: {model} kept on {routing['provider']}: the call failed reaching "
                    f'the tool server, not the endpoint ({type(error).__name__}: {snippet(str(error))})')
        return False
    if isinstance(error, BaseException):
        error = f'{type(error).__name__}: {error}'
    pin.drop(routing['provider'], error)
    now = pin.routing()
    if now == routing:
        return False
    logger.warning(f"Single-turn pin: asking the failed {model} call again on "
                   f"{now.get('provider') or 'no pin (OpenRouter chooses)'}")
    return True


def reset_pins() -> None:
    _pins.clear()
