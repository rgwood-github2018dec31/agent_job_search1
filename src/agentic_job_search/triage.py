"""Cheap-LLM helpers for Stage 2: MCP tool-server client, local triage, and
non-Anthropic rating calls.

OpenRouter and local Ollama models are reached through the pre-existing FastMCP
HTTP tool servers (tools_llm_remote_openrouter on :8006, tools_llm_local on
:8002), not through their raw HTTP APIs. A server that is down raises, and
callers treat that as a provider failure (fall through / fail open).
"""

import json
import logging
import re
from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agentic_job_search.config import (
    LLM_JSON_MAX_TOKENS,
    LLM_LOCAL_MCP_URL,
    LLM_OPENROUTER_MCP_URL,
    LOCAL_MODEL,
    OPENROUTER_MODEL,
    TRIAGE_THRESHOLD,
)

logger = logging.getLogger(__name__)


PROVIDER_UNAVAILABLE_STATUSES = frozenset({401, 402, 403, 429})

# `requests` renders a raised-for status as "402 Client Error: Payment Required for url: ...";
# the OpenRouter body itself carries {"error": {"code": 402, ...}}. Both are matched, and the
# match is on the STATUS, never on the reason phrase -- the phrase is upstream's text and can
# change without notice, while the code is the contract.
_HTTP_STATUS_RE = re.compile(r'\b(\d{3})\s+(?:Client|Server)\s+Error\b')
_JSON_STATUS_RE = re.compile(r'"(?:code|status|status_code)"\s*:\s*(\d{3})\b')


class ProviderUnavailableError(RuntimeError):
    """The remote LLM provider itself is refusing every call - unpaid, unauthorised, or throttled.

    Distinct from a one-off failure, and the distinction decides control flow. A page that would
    not load breaks ONE query and the next should still run; a 402 breaks EVERY query, so retrying
    per query burns the remaining requests and their human-emulation pacing delays on calls that
    cannot succeed, then reports "0 jobs" as though the searches had simply found nothing. That is
    exactly what happened on 2026-09-02: six queries, six 402s, ~70s of pacing between them, and
    the Anthropic fallback -- which exists precisely for this -- never fired, because `run_scraper`
    swallowed each failure and returned normally.

    Same shape as `AgentApiError` in tools_generic (2026-08-24): a systemic API-level fault must
    abort the loop, while a per-item fault warns and continues.
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def http_status_of(error_text: str) -> int | None:
    """The HTTP status named in a provider error string, or None if it names none."""
    for pattern in (_HTTP_STATUS_RE, _JSON_STATUS_RE):
        if match := pattern.search(error_text):
            return int(match.group(1))
    return None


def provider_error(context: str, error_text: str) -> RuntimeError:
    """Classify an OpenRouter MCP failure as systemic (provider) or one-off.

    Returns the exception rather than raising it, so call sites keep their `raise` visible.
    """
    status = http_status_of(error_text)
    message = f'{context}: {error_text}'
    if status in PROVIDER_UNAVAILABLE_STATUSES:
        return ProviderUnavailableError(message, status=status)
    return RuntimeError(message)


class LocalModelMissingError(RuntimeError):
    """LOCAL_MODEL is not installed on the Ollama server behind the local MCP tool server.

    Distinct from the server being down: that is the fail-open case this module is designed
    around, while this is a misconfiguration that silently disables triage on every job.
    """


def unwrap_exception(ex: BaseException) -> str:
    """Flatten an exception's real messages out of any ExceptionGroup / __cause__ nesting.

    Every MCP failure in this codebase arrives wrapped: `streamable_http_client` and
    `ClientSession` each open an anyio task group, so a RuntimeError raised inside `call()`
    surfaces to the caller as an ExceptionGroup whose str() is the useless
    "unhandled errors in a TaskGroup (1 sub-exception)". Logging `ex` directly therefore
    discards the only informative text there is - which is exactly how
    `model 'qwen3.6:latest' not found` stayed invisible across ten runs and 250+ warnings.
    """
    messages: list[str] = []
    seen: set[int] = set()

    def walk(err: BaseException | None) -> None:
        if err is None or id(err) in seen:
            return
        seen.add(id(err))
        children = getattr(err, 'exceptions', None)
        if children:
            for child in children:
                walk(child)
        else:
            text = str(err).strip()
            messages.append(f'{type(err).__name__}: {text}' if text else type(err).__name__)
        walk(err.__cause__)
        walk(err.__context__)

    walk(ex)
    return ' | '.join(messages) if messages else repr(ex)


@asynccontextmanager
async def mcp_session(url: str):
    """Open one MCP session and yield a call(tool_name, args) -> str function.

    Use this when consecutive tool calls must share server-side state (e.g. the
    Playwright browser tab persists within a session, not across sessions)."""
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(tool_name: str, args: dict) -> str:
                result = await session.call_tool(tool_name, args)
                texts = [c.text for c in result.content if getattr(c, 'text', None)]
                joined = '\n'.join(texts)
                if getattr(result, 'isError', False):
                    raise RuntimeError(f'MCP tool {tool_name!r} at {url} returned an error: {joined[:500]}')
                return joined

            async def list_tools() -> list:
                """The server's own tool schemas.

                Used to build the OpenAI-style function definitions for the non-Anthropic loops, so
                the schemas come from the running server instead of being transcribed by hand into a
                constant that nothing keeps in sync when the server changes.
                """
                return list((await session.list_tools()).tools)

            call.list_tools = list_tools
            yield call


async def call_mcp_tool(url: str, tool_name: str, args: dict) -> str:
    """Call a single tool on a streamable-HTTP MCP server and return its text content."""
    async with mcp_session(url) as call:
        return await call(tool_name, args)


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object from text that may contain prose around it."""
    decoder = json.JSONDecoder()
    start = text.find('{')
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find('{', start + 1)
    raise ValueError(f'No JSON object found in LLM response (first 300 chars): {text[:300]!r}')


async def chat_openrouter(
    prompt: str, system: str = '', model: str = OPENROUTER_MODEL, max_tokens: int = LLM_JSON_MAX_TOKENS
) -> tuple[str, float]:
    """One completion via the OpenRouter MCP server. Returns (content, cost_usd)."""
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    raw = await call_mcp_tool(
        LLM_OPENROUTER_MCP_URL, 'chat',
        {'messages': messages, 'model': model, 'max_tokens': max_tokens},
    )
    data = json.loads(raw)
    if not data.get('ok'):
        raise provider_error('OpenRouter chat failed', str(data.get('error') or raw[:300]))
    if not data.get('content'):
        raise RuntimeError(f'OpenRouter chat returned empty content: {raw[:300]}')
    return data['content'], float(data.get('cost_usd') or 0.0)


async def list_local_models() -> list[str]:
    """Names of the models actually installed on the local Ollama server."""
    raw = await call_mcp_tool(LLM_LOCAL_MCP_URL, 'list_models', {})
    data = json.loads(raw)
    return [m['name'] for m in data.get('models', ()) if m.get('name')]


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
        f'Set LOCAL_MODEL in config.py to one of these, or pull the missing model.'
    )


async def generate_local(
    prompt: str, system: str = '', model: str = LOCAL_MODEL, max_tokens: int = LLM_JSON_MAX_TOKENS
) -> str:
    """One completion via the local Ollama MCP server. Returns the response text.

    Raises LocalModelMissingError - naming the installed models - when the tag is gone, so a
    misconfiguration is distinguishable from the server being down. Every other failure is
    re-raised untouched: a down server must keep reading as a down server.
    """
    try:
        raw = await call_mcp_tool(
            LLM_LOCAL_MCP_URL, 'generate',
            {'prompt': prompt, 'system': system, 'model': model, 'max_tokens': max_tokens, 'temperature': 0.2},
        )
    except Exception as ex:
        if _looks_like_missing_model(unwrap_exception(ex)):
            raise LocalModelMissingError(await _describe_missing_model(model)) from ex
        raise
    data = json.loads(raw)
    response = data.get('response', '')
    if not response:
        raise RuntimeError(f'Local LLM returned empty response: {raw[:300]}')
    return response


async def preflight_local_model(model: str = LOCAL_MODEL) -> str | None:
    """None when `model` is installed, otherwise a message naming what IS installed.

    Also None when the server itself is unreachable - that is the already-handled fail-open
    case, not a misconfiguration, and reporting it here would cry wolf on every run made with
    the tool servers stopped.
    """
    try:
        installed = await list_local_models()
    except Exception as ex:
        logger.info(f'Local LLM server unreachable, skipping model preflight: {unwrap_exception(ex)}')
        return None
    if model in installed:
        return None
    return await _describe_missing_model(model)


TRIAGE_INSTRUCTIONS = """You are a fast job-fit triage filter. Score how well the job posting below fits the candidate profile on a 1-5 scale:
1 = clear non-fit (wrong field, wrong location, hard deal-breaker)
2 = weak fit (major gaps or mismatches)
3 = plausible fit (worth a closer look)
4 = good fit
5 = excellent fit

Be conservative: when unsure, score HIGHER so a stronger model takes a closer look. Only score 1-2 when the mismatch is obvious.

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<one sentence>"}"""


async def triage_job_fit(job_text: str, profile_block: str) -> dict | None:
    """Score a job extract 1-5 with the local LLM. Returns {'score', 'reason'},
    or None if the local server is unavailable or the response is unusable (fail open)."""
    prompt = f'{profile_block}\n\n--- JOB POSTING ---\n{job_text}\n--- END JOB POSTING ---'
    try:
        response = await generate_local(prompt, system=TRIAGE_INSTRUCTIONS)
        result = extract_json_object(response)
        score = int(result['score'])
    except Exception as ex:
        logger.warning(
            f'Triage unavailable, failing open (job goes to full rating): {unwrap_exception(ex)}'
        )
        return None
    return {'score': score, 'reason': str(result.get('reason', ''))}


def triage_rejects(triage_result: dict | None) -> bool:
    return triage_result is not None and triage_result['score'] <= TRIAGE_THRESHOLD


RATING_JSON_INSTRUCTIONS = (
    'Respond with ONLY a JSON object: '
    '{"rating": <1-5>, "company": "<hiring company>", "title": "<job title>", '
    '"reasoning": "<2-3 sentences on the fit>", "summary": "<short snake_case-friendly label for the job>", '
    '"pros": ["<short phrase naming a concrete strength>", ...], '
    '"warnings": ["<short phrase naming anything conflicting with the requirements>", ...]}'
)


async def rate_with_openrouter(system_prompt: str, user_prompt: str, model: str = OPENROUTER_MODEL) -> tuple[dict, float]:
    """Rate a job via OpenRouter. Returns (rating dict, cost_usd)."""
    content, cost_usd = await chat_openrouter(
        f'{user_prompt}\n\n{RATING_JSON_INSTRUCTIONS}', system=system_prompt, model=model
    )
    result = extract_json_object(content)
    result['rating'] = int(result['rating'])
    return result, cost_usd


async def rate_with_ollama(system_prompt: str, user_prompt: str, model: str = LOCAL_MODEL) -> dict:
    """Rate a job via the local Ollama server. Returns the rating dict (cost is zero)."""
    response = await generate_local(
        f'{user_prompt}\n\n{RATING_JSON_INSTRUCTIONS}', system=system_prompt, model=model
    )
    result = extract_json_object(response)
    result['rating'] = int(result['rating'])
    return result
