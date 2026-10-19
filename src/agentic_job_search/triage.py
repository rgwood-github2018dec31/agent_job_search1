'''Cheap-LLM helpers for Stage 2: local triage, and non-Anthropic rating calls.

OpenRouter and local Ollama models are reached through the pre-existing FastMCP
HTTP tool servers (tools_llm_remote_openrouter on :8006, tools_llm_local on
:8002), not through their raw HTTP APIs. A server that is down raises, and
callers treat that as a provider failure (fall through / fail open).

The generic MCP client (`call_mcp_tool`, `mcp_session`, `unwrap_exception`)
lives in utils_tools_n_agents_common.mcp_client, shared with agent_meta1 and
agent_stock_trends1.
'''

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from utils_tools_n_agents_common.mcp_client import call_mcp_tool, unwrap_exception

from agentic_job_search.config import (
    LLM_LOCAL_MCP_URL,
    LLM_LOCAL_PURPOSE_RATING,
    LLM_LOCAL_PURPOSE_TRIAGE,
    LLM_LOCAL_SERVICE_NAME,
    LLM_MCP_CALL_TIMEOUT_SECONDS,
    LLM_MCP_SINGLE_TURN_TIMEOUT_SECONDS,
    LLM_OPENROUTER_MCP_URL,
    LOCAL_LLM_TEMPERATURE,
    LOG_SNIPPET_MAX_CHARS,
    MODEL_NAME_SINGLE_TURN,
    OLLAMA_MODEL_NAME_TRIAGE,
    TRIAGE_THRESHOLD,
)
from agentic_job_search.endpoint_pin import endpoint_failed, routing_for
from agentic_job_search.text_budget import quote_for_error

logger = logging.getLogger(__name__)


PROVIDER_UNAVAILABLE_STATUSES = frozenset({401, 402, 403, 429})

# `requests` renders a raised-for status as "402 Client Error: Payment Required for url: ...";
# the OpenRouter body itself carries {"error": {"code": 402, ...}}. Both are matched, and the
# match is on the STATUS, never on the reason phrase -- the phrase is upstream's text and can
# change without notice, while the code is the contract.
_HTTP_STATUS_RE = re.compile(r'\b(\d{3})\s+(?:Client|Server)\s+Error\b')
_JSON_STATUS_RE = re.compile(r'"(?:code|status|status_code)"\s*:\s*(\d{3})\b')


class ProviderUnavailableError(RuntimeError):
    '''The remote LLM provider itself is refusing every call - unpaid, unauthorised, or throttled.

    Distinct from a one-off failure, and the distinction decides control flow. A page that would
    not load breaks ONE query and the next should still run; a 402 breaks EVERY query, so retrying
    per query burns the remaining requests and their human-emulation pacing delays on calls that
    cannot succeed, then reports "0 jobs" as though the searches had simply found nothing. That is
    exactly what happened on 2026-09-02: six queries, six 402s, ~70s of pacing between them, and
    the Anthropic fallback -- which exists precisely for this -- never fired, because `run_scraper`
    swallowed each failure and returned normally.

    Same shape as `AgentApiError` in tools_generic (2026-08-24): a systemic API-level fault must
    abort the loop, while a per-item fault warns and continues.
    '''

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class TruncatedResponseError(RuntimeError):
    '''The model hit its output ceiling mid-answer, so the body is incomplete.

    Distinct from a malformed response, and the distinction is the whole point: a truncated JSON
    body is not a parse bug to be worked around, it is a request that needs re-issuing with more
    room. Before this existed, `chat_openrouter` inspected `finish_reason` only when `content` was
    null, so a truncated NON-empty body went to `extract_json_object` as if it were complete and
    surfaced as "No JSON object found in LLM response" — which reads like a fence or a prose
    wrapper and sent the 2026-09 diagnosis down exactly that dead end. `finish_reason` was in the
    payload the entire time.
    '''

    def __init__(self, message: str, max_tokens: int | None = None):
        super().__init__(message)
        self.max_tokens = max_tokens


def http_status_of(error_text: str) -> int | None:
    '''The HTTP status named in a provider error string, or None if it names none.'''
    for pattern in (_HTTP_STATUS_RE, _JSON_STATUS_RE):
        if match := pattern.search(error_text):
            return int(match.group(1))
    return None


def provider_error(context: str, error_text: str) -> RuntimeError:
    '''Classify an OpenRouter MCP failure as systemic (provider) or one-off.

    Returns the exception rather than raising it, so call sites keep their `raise` visible.
    '''
    status = http_status_of(error_text)
    message = f'{context}: {error_text}'
    if status in PROVIDER_UNAVAILABLE_STATUSES:
        return ProviderUnavailableError(message, status=status)
    return RuntimeError(message)


class LocalModelMissingError(RuntimeError):
    '''OLLAMA_MODEL_NAME_TRIAGE is not installed on the Ollama server behind the local MCP tool server.

    Distinct from the server being down: that is the fail-open case this module is designed
    around, while this is a misconfiguration that silently disables triage on every job.
    '''


def extract_json_object(text: str) -> dict:
    '''Extract the first JSON object from text that may contain prose around it.'''
    decoder = json.JSONDecoder()
    start = text.find('{')
    # A '{' that does not start valid JSON is expected in prose; only the last failure matters,
    # and it is chained onto the final error below rather than logged per candidate.
    last_error: json.JSONDecodeError | None = None
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as ex:
            last_error = ex
        start = text.find('{', start + 1)
    # Head AND tail, deliberately. The head-only message is why this was first misdiagnosed as a
    # ```json fence problem: the fence is at the head and the fault is always in the tail (a
    # truncated body, or `"reasoning":Exceptional` with the opening quote missing). Fences have
    # always parsed here — raw_decode scans from the first '{' and ignores everything around it.
    raise ValueError(
        f'No JSON object found in LLM response ({len(text)} chars). '
        f'Head: {text[:LOG_SNIPPET_MAX_CHARS]!r} ... Tail: {text[-LOG_SNIPPET_MAX_CHARS:]!r}'
        + (f' Last decode error: {last_error}' if last_error else '')
    ) from last_error


_Parsed = TypeVar('_Parsed')


@dataclass(frozen=True)
class ChatReply:
    '''One OpenRouter completion and who served it.

    Returning only (content, cost) is how a provider's "Your quota is exhausted" notice, sent back
    as an ordinary completion, failed later as "No JSON object found" with no record of which
    provider had sent it (2026-10-14). Everything the server reports about the call stays here, so
    a caller whose parse fails can say which model, provider and finish_reason produced the text.
    '''

    content: str
    cost_usd: float
    requested_model: str
    model: str
    provider: str
    served_quantization: str
    finish_reason: str
    usage: dict = field(default_factory=dict)

    def describe(self) -> str:
        '''The call's metadata in one line, for log lines and exception messages.'''
        served = self.model if self.model == self.requested_model else f'{self.model} (requested {self.requested_model})'
        return (f'model={served} provider={self.provider or "?"} quantization={self.served_quantization or "?"} '
                f'finish_reason={self.finish_reason or "?"} tokens={self.usage.get("prompt_tokens")}/'
                f'{self.usage.get("completion_tokens")} cost=${self.cost_usd:.6f}')

    def parse(self, coerce: Callable[[dict], _Parsed] | None = None, what: str = 'reply') -> _Parsed | dict:
        '''The content's JSON object, run through `coerce` when given.

        A parse or validation failure raises ValueError carrying the call's metadata and the
        content: whether it was a model's slip or a provider's notice, the log alone tells.
        '''
        try:
            obj = extract_json_object(self.content)
            return coerce(obj) if coerce is not None else obj
        except (ValueError, KeyError, TypeError) as ex:
            raise ValueError(
                f'OpenRouter {what} unusable ({type(ex).__name__}: {ex}) — {self.describe()}; '
                f'content: {quote_for_error(self.content, f"unusable OpenRouter {what} from {self.model}")!r}'
            ) from ex


def _field(data: dict, key: str, kinds: type | tuple[type, ...], raw: str) -> Any:
    '''data[key] when it has one of `kinds`; raises naming the field, with the whole reply.'''
    value = data.get(key)
    if not isinstance(value, kinds):
        raise ValueError(f'OpenRouter chat reply field {key!r} is {type(value).__name__}, not {kinds}: '
                         f'{quote_for_error(raw, "OpenRouter chat reply")!r}')
    return value


async def chat_openrouter(
    prompt: str, system: str = '', model: str = MODEL_NAME_SINGLE_TURN, max_tokens: int | None = None,
    timeout_seconds: float = LLM_MCP_SINGLE_TURN_TIMEOUT_SECONDS,
) -> ChatReply:
    """One completion via the OpenRouter MCP server, with who served it (ChatReply).

    `max_tokens` defaults to None, i.e. it is NOT SENT and the model's own ceiling applies. There
    is nothing to protect against here: the MCP tool declares it optional and applies no clamp,
    `glm-5.3-flash` allows 131,072 completion tokens, and the two agentic call sites
    (scrape_openrouter, extract_openrouter) have always sent nothing. The former 3000 default was
    chosen by no call site — all eight simply inherited it — and every one of the 13 truncations
    across 2026-09-01..03 was `completion_tokens: 3000` exactly, surviving a model change.

    Removing it should also LOWER cost: a truncated company-match paid for 3000 reasoning tokens,
    returned null, and then paid again for the Anthropic fallback. Finishing the answer is cheaper
    than paying for both halves of a failure.
    """
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    args = {'messages': messages, 'model': model}
    if max_tokens is not None:
        args['max_tokens'] = max_tokens
    routing = routing_for(model)
    args.update(routing)
    try:
        raw = await call_mcp_tool(LLM_OPENROUTER_MCP_URL, 'chat', args, timeout_seconds=timeout_seconds)
    except Exception as ex:
        endpoint_failed(model, routing, f'{type(ex).__name__}: {ex}')
        raise
    try:
        data = json.loads(raw)
    except ValueError as ex:
        raise ValueError(f'OpenRouter chat reply for {model} is not JSON ({ex}): '
                         f'{quote_for_error(raw, "OpenRouter chat reply")!r}') from ex
    if not isinstance(data, dict):
        raise ValueError(f'OpenRouter chat reply for {model} is not an object: {quote_for_error(raw, "OpenRouter chat reply")!r}')
    if data.get('ok') is not True:
        endpoint_failed(model, routing, str(data.get('error') or raw))
        # The whole reply, not just `error`: the server's other fields say who failed.
        raise provider_error(f'OpenRouter chat failed for {model}', quote_for_error(raw, 'OpenRouter chat failure reply'))
    _field(data, 'content', (str, type(None)), raw)
    usage = data.get('usage') or {}
    if data.get('cost_usd') is None:
        logger.warning(f'OpenRouter reported no cost for a {model} call (counted as $0): '
                       f'{quote_for_error(raw, "OpenRouter chat reply with no cost")}')
    reply = ChatReply(
        content=data.get('content') or '', cost_usd=float(data.get('cost_usd') or 0.0),
        requested_model=model, model=str(data.get('model') or model), provider=str(data.get('provider') or ''),
        served_quantization=str(data.get('served_quantization') or ''),
        finish_reason=str(data.get('finish_reason') or ''), usage=usage if isinstance(usage, dict) else {},
    )
    # Checked BEFORE the content test, and regardless of whether content is empty: a truncated
    # non-empty body is the case that used to reach the JSON parser disguised as a parse failure.
    if reply.finish_reason == 'length':
        raise TruncatedResponseError(
            f'OpenRouter response truncated (finish_reason=length): max_tokens={max_tokens}, '
            f'content_chars={len(reply.content)} — {reply.describe()}',
            max_tokens=max_tokens,
        )
    if not reply.content:
        raise RuntimeError(f'OpenRouter chat returned empty content — {reply.describe()}; '
                           f'reply: {quote_for_error(raw, "OpenRouter empty-content reply")!r}')
    return reply


async def list_local_models() -> list[str]:
    '''Names of the models actually installed on the local Ollama server.'''
    raw = await call_mcp_tool(LLM_LOCAL_MCP_URL, 'list_models', {})
    try:
        data = json.loads(raw)
    except ValueError as ex:
        raise ValueError(f'Local list_models reply is not JSON ({ex}): {quote_for_error(raw, "local list_models reply")!r}') from ex
    # An error reply used to read as "no models installed", which then reported the triage model
    # as missing: a false diagnosis that disables triage.
    if not isinstance(data, dict) or not isinstance(data.get('models'), list):
        raise ValueError(f'Local list_models reply has no models list: {quote_for_error(raw, "local list_models reply")!r}')
    return [m['name'] for m in data['models'] if isinstance(m, dict) and m.get('name')]


def _looks_like_missing_model(message: str) -> bool:
    """Whether a failure message is Ollama saying the requested model is not installed.

    Ollama answers /api/generate for an unknown tag with a 404 carrying
    {"error": "model 'x' not found"}, so both halves are checked rather than treating every
    404 as a missing model.
    """
    lowered = message.lower()
    return 'not found' in lowered and ('model' in lowered or '/api/generate' in lowered)


async def _describe_missing_model(model: str) -> str:
    installed = []
    try:
        installed = await list_local_models()
    except Exception as ex:  # listing is best-effort; the missing model is the finding
        logger.warning(f'Could not list installed local models: {unwrap_exception(ex)}')
    listed = ', '.join(installed) if installed else '<could not list installed models>'
    return (
        f'Local model {model!r} is not installed on the Ollama server behind '
        f'{LLM_LOCAL_MCP_URL}. Installed models: {listed}. '
        f'Set OLLAMA_MODEL_NAME_TRIAGE in config.py to one of these, or pull the missing model.'
    )


async def generate_local(
    prompt: str,
    system: str = '',
    model: str = OLLAMA_MODEL_NAME_TRIAGE,
    max_tokens: int | None = None,
    *,
    purpose_slug: str,
) -> str:
    """One completion via the local Ollama MCP server. Returns the response text.

    `purpose_slug` (why this call is made — one of the LLM_LOCAL_PURPOSE_* constants) is required,
    and is sent with LLM_LOCAL_SERVICE_NAME so the server's call log can trace a model loading in
    Ollama back to this program and this call site.

    Raises LocalModelMissingError - naming the installed models - when the tag is gone, so a
    misconfiguration is distinguishable from the server being down. Every other failure is
    re-raised untouched: a down server must keep reading as a down server.
    """
    try:
        args = {
            'prompt': prompt, 'system': system, 'model': model, 'temperature': LOCAL_LLM_TEMPERATURE,
            'service_name': LLM_LOCAL_SERVICE_NAME, 'purpose_slug': purpose_slug,
        }
        if max_tokens is not None:
            args['max_tokens'] = max_tokens
        raw = await call_mcp_tool(LLM_LOCAL_MCP_URL, 'generate', args, timeout_seconds=LLM_MCP_CALL_TIMEOUT_SECONDS)
    except Exception as ex:
        if _looks_like_missing_model(unwrap_exception(ex)):
            raise LocalModelMissingError(await _describe_missing_model(model)) from ex
        raise
    try:
        data = json.loads(raw)
    except ValueError as ex:
        raise ValueError(f'Local generate reply ({model}) is not JSON ({ex}): {quote_for_error(raw, "local generate reply")!r}') from ex
    if not isinstance(data, dict):
        raise ValueError(f'Local generate reply ({model}) is not an object: {quote_for_error(raw, "local generate reply")!r}')
    if data.get('error') or data.get('ok') is False:
        raise RuntimeError(f'Local generate ({model}) reported an error: {quote_for_error(raw, "local generate error reply")!r}')
    done_reason = data.get('done_reason')
    meta = (f'model={data.get("model") or model} done={data.get("done")} done_reason={done_reason} '
            f'tokens={data.get("prompt_eval_count")}/{data.get("eval_count")}')
    if done_reason == 'length':
        raise TruncatedResponseError(
            f'Local LLM response truncated (done_reason=length): max_tokens={max_tokens}, '
            f'response_chars={len(data.get("response") or "")} — {meta}', max_tokens=max_tokens,
        )
    response = data.get('response')
    if not isinstance(response, str) or not response:
        raise RuntimeError(f'Local LLM returned no response text — {meta}; reply: {quote_for_error(raw, "local generate reply")!r}')
    return response


async def preflight_local_model(model: str = OLLAMA_MODEL_NAME_TRIAGE) -> str | None:
    '''None when `model` is installed, otherwise a message naming what IS installed.

    Also None when the server itself is unreachable - that is the already-handled fail-open
    case, not a misconfiguration, and reporting it here would cry wolf on every run made with
    the tool servers stopped.
    '''
    try:
        installed = await list_local_models()
    except Exception as ex:
        logger.info(f'Local LLM server unreachable, skipping model preflight: {unwrap_exception(ex)}')
        return None
    if model in installed:
        return None
    return await _describe_missing_model(model)


TRIAGE_INSTRUCTIONS = '''You are a fast job-fit triage filter. Score how well the job posting below fits the candidate profile on a 1-5 scale:
1 = clear non-fit (wrong field, wrong location, hard deal-breaker)
2 = weak fit (major gaps or mismatches)
3 = plausible fit (worth a closer look)
4 = good fit
5 = excellent fit

Be conservative: when unsure, score HIGHER so a stronger model takes a closer look. Only score 1-2 when the mismatch is obvious.

Respond with ONLY a JSON object:
{"score": <1-5>, "reason": "<one sentence>", "quote": "<words of the job posting that support the score, copied as written>"}'''


async def triage_job_fit(job_text: str, profile_block: str) -> dict | None:
    """Score a job extract 1-5 with the local LLM. Returns {'score', 'reason'},
    or None if the local server is unavailable or the response is unusable (fail open)."""
    prompt = f'{profile_block}\n\n--- JOB POSTING ---\n{job_text}\n--- END JOB POSTING ---'
    try:
        response = await generate_local(prompt, system=TRIAGE_INSTRUCTIONS, purpose_slug=LLM_LOCAL_PURPOSE_TRIAGE)
        result = extract_json_object(response)
        score = int(result['score'])
    except Exception as ex:
        logger.warning(
            f'Triage unavailable, failing open (job goes to full rating): {unwrap_exception(ex)}'
        )
        return None
    return {'score': score, 'reason': str(result.get('reason', '')), 'quote': str(result.get('quote') or '')}


def triage_rejects(triage_result: dict | None) -> bool:
    return triage_result is not None and triage_result['score'] <= TRIAGE_THRESHOLD


RATING_JSON_INSTRUCTIONS = (
    'Respond with ONLY a JSON object: '
    '{"rating": <1-5>, "company": "<hiring company>", "title": "<job title>", '
    '"reasoning": "<2-3 sentences on the fit>", "summary": "<short snake_case-friendly label for the job>", '
    '"pros": [{"text": "<short phrase naming a concrete strength>", "quote": "<words of the posting that show it>"}, ...], '
    '"warnings": [{"text": "<short phrase naming anything conflicting with the requirements>", '
    '"quote": "<words of the posting that show it>"}, ...]}. '
    'Each quote is copied from the job posting text above, as written. Code looks it up there and '
    'drops any pro or warning whose quote it cannot find.'
)


# Room to finish the answer when a model has already proved it wants more than its own default.
# Only ever used on the retry, never on the first attempt.
_TRUNCATION_RETRY_MAX_TOKENS = 32_000


class MalformedRatingReplyError(ValueError):
    '''A rating reply that arrived whole but holds no JSON object with an integer `rating`.'''


def parse_rating_reply(content: str) -> dict:
    '''The rating dict from a model reply; MalformedRatingReplyError names what was wrong with it.'''
    try:
        result = extract_json_object(content)
        result['rating'] = int(result['rating'])
    except (ValueError, KeyError, TypeError) as ex:
        raise MalformedRatingReplyError(f'{type(ex).__name__}: {ex}') from ex
    return result


async def rate_with_openrouter(system_prompt: str, user_prompt: str, model: str = MODEL_NAME_SINGLE_TURN) -> tuple[dict, float]:
    '''Rate a job via OpenRouter. Returns (rating dict, cost_usd).

    Retries once on truncation because losing this call is costly: the job was written to
    processed_jobs/ back in Stage 1b. A posting the model had scored 5 was lost that way on
    2026-09-03.

    A complete reply that is not a usable rating raises MalformedRatingReplyError and is never
    repaired; `rate_job` then asks MODEL_NAME_RATING_FALLBACKS. On 2026-10-08 a reply holding
    `"rating": 5` left its last array unclosed.
    '''
    prompt = f'{user_prompt}\n\n{RATING_JSON_INSTRUCTIONS}'
    try:
        reply = await chat_openrouter(prompt, system=system_prompt, model=model)
    except TruncatedResponseError as ex:
        logger.warning(f'Rating call truncated, retrying once with max_tokens={_TRUNCATION_RETRY_MAX_TOKENS}: {ex}')
        reply = await chat_openrouter(
            prompt, system=system_prompt, model=model, max_tokens=_TRUNCATION_RETRY_MAX_TOKENS
        )
    try:
        return parse_rating_reply(reply.content), reply.cost_usd
    except MalformedRatingReplyError as ex:
        raise MalformedRatingReplyError(
            f'{ex} — {reply.describe()}; content: {quote_for_error(reply.content, f"malformed rating reply from {reply.model}")!r}'
        ) from ex


async def rate_with_ollama(system_prompt: str, user_prompt: str, model: str = OLLAMA_MODEL_NAME_TRIAGE) -> dict:
    '''Rate a job via the local Ollama server. Returns the rating dict (cost is zero).'''
    response = await generate_local(
        f'{user_prompt}\n\n{RATING_JSON_INSTRUCTIONS}', system=system_prompt, model=model,
        purpose_slug=LLM_LOCAL_PURPOSE_RATING,
    )
    return parse_rating_reply(response)
