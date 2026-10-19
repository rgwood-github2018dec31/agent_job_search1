"""Per-run OpenRouter endpoint pin for the single-turn model (extract, rate, classifiers).

Chosen once per run from `list_endpoints`: drop the slowest share by measured throughput and the
priciest share by price at the single-turn token mix, each over the same usable set, and pin the
cheapest that is left. A call that fails on the pinned endpoint drops it for the rest of the run
and the next-cheapest survivor takes over; with none left, calls go unpinned (OpenRouter's choice).

Unpinned until `pin_single_turn_endpoints()` runs, so interactive mode and tests route as before.
"""

import json
import logging
from dataclasses import dataclass, field

from utils_tools_n_agents_common.mcp_client import call_mcp_tool

from agentic_job_search.config import (
    LLM_MCP_SINGLE_TURN_TIMEOUT_SECONDS,
    LLM_OPENROUTER_MCP_URL,
    SINGLE_TURN_COMPLETION_TOKENS_PER_PROMPT_TOKEN,
    SINGLE_TURN_DROP_PRICIEST_SHARE,
    SINGLE_TURN_DROP_SLOWEST_SHARE,
    SINGLE_TURN_MIN_BITS_PER_WEIGHT,
)
from agentic_job_search.text_budget import snippet

logger = logging.getLogger(__name__)

PER_MILLION = 1_000_000


def single_turn_price_per_million(endpoint: dict) -> float | None:
    '''Per million PROMPT tokens at the single-turn token mix; None when a price is missing.'''
    prompt, completion = endpoint.get('prompt_price'), endpoint.get('completion_price')
    if prompt is None or completion is None:
        return None
    return (prompt + SINGLE_TURN_COMPLETION_TOKENS_PER_PROMPT_TOKEN * completion) * PER_MILLION


def select_single_turn_endpoints(listing: dict) -> tuple[list[tuple[float, dict]], str]:
    '''Surviving endpoints cheapest first, plus a one-line account of what was dropped and why.

    Usable: up (status 0), tool-calling, at a declared precision of at least
    `SINGLE_TURN_MIN_BITS_PER_WEIGHT`, with a measured throughput and both prices. Unknown bits or
    throughput is excluded, never assumed good. The slowest and priciest cuts are each taken over
    the whole usable set, rounded down, so fewer than three usable endpoints drop nothing.
    '''
    usable: list[tuple[float, dict]] = []
    seen: set[str] = set()
    for endpoint in listing.get('endpoints') or []:
        tag = endpoint.get('tag')
        if not tag or tag in seen:
            continue
        if endpoint.get('status') != 0 or endpoint.get('supports_tools') is not True:
            continue
        bits = endpoint.get('bits')
        if bits is None or bits < SINGLE_TURN_MIN_BITS_PER_WEIGHT or endpoint.get('throughput_p50') is None:
            continue
        price = single_turn_price_per_million(endpoint)
        if price is None:
            continue
        seen.add(tag)
        usable.append((price, endpoint))
    n_slowest = int(len(usable) * SINGLE_TURN_DROP_SLOWEST_SHARE)
    n_priciest = int(len(usable) * SINGLE_TURN_DROP_PRICIEST_SHARE)
    by_speed = sorted(usable, key=lambda pair: pair[1]['throughput_p50'])
    by_price = sorted(usable, key=lambda pair: pair[0], reverse=True)
    slowest = {e['tag'] for _p, e in by_speed[:n_slowest]}
    priciest = {e['tag'] for _p, e in by_price[:n_priciest]}
    kept = sorted(((p, e) for p, e in usable if e['tag'] not in slowest | priciest), key=lambda pair: pair[0])
    report = (f'{len(usable)} usable of {len(listing.get("endpoints") or [])}; dropped slowest '
              f'{sorted(slowest) or "none"}, priciest {sorted(priciest) or "none"}; kept '
              + (', '.join(f"{e['tag']} ${p:.4f}/M {e['throughput_p50']} tok/s" for p, e in kept) or 'none'))
    return kept, report


@dataclass
class EndpointPin:
    model: str
    ranked: list[tuple[float, dict]] = field(default_factory=list)

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
        after = (f"switched to {self.ranked[0][1]['tag']} (est. ${self.ranked[0][0]:.4f}/M)"
                 if self.ranked else 'no endpoint left, so calls go unpinned')
        logger.warning(f'Single-turn pin: {self.model} call on {tag} failed ({snippet(error)}); '
                       f'{after} for the rest of the run')


_pins: dict[str, EndpointPin] = {}


async def pin_single_turn_endpoints(model: str) -> None:
    """Choose this run's endpoint for `model`. A failure to choose is logged and leaves it unpinned."""
    try:
        raw = await call_mcp_tool(LLM_OPENROUTER_MCP_URL, 'list_endpoints', {'model': model},
                                  timeout_seconds=LLM_MCP_SINGLE_TURN_TIMEOUT_SECONDS)
        listing = json.loads(raw)
        if not isinstance(listing, dict) or listing.get('ok') is not True:
            raise RuntimeError(f'list_endpoints refused: {snippet(raw)}')
    except Exception as ex:
        _pins.pop(model, None)
        logger.warning(f'Single-turn pin: could not list endpoints for {model} ({type(ex).__name__}: {ex}); '
                       f"running unpinned, so the provider and its price are OpenRouter's choice")
        return
    ranked, report = select_single_turn_endpoints(listing)
    _pins[model] = EndpointPin(model, ranked)
    if ranked:
        logger.info(f'Single-turn pin: {model} -> {ranked[0][1]["tag"]} ({report})')
    else:
        logger.warning(f'Single-turn pin: no endpoint survived for {model}, running unpinned ({report})')


def routing_for(model: str) -> dict:
    pin = _pins.get(model)
    return pin.routing() if pin else {}


def endpoint_failed(model: str, routing: dict, error: str) -> None:
    '''Report a failed call made with `routing`; drops that endpoint when it is still the pin.'''
    if routing and (pin := _pins.get(model)):
        pin.drop(routing['provider'], error)


def reset_pins() -> None:
    _pins.clear()
