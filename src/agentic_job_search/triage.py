"""Cheap-LLM helpers for Stage 2: MCP tool-server client, local triage, and
non-Anthropic rating calls.

OpenRouter and local Ollama models are reached through the pre-existing FastMCP
HTTP tool servers (tools_llm_remote_openrouter on :8006, tools_llm_local on
:8002), not through their raw HTTP APIs. A server that is down raises, and
callers treat that as a provider failure (fall through / fail open).
"""

import json
import logging
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
    if not data.get('ok') or not data.get('content'):
        raise RuntimeError(f'OpenRouter chat failed or returned empty content: {raw[:300]}')
    return data['content'], float(data.get('cost_usd') or 0.0)


async def generate_local(
    prompt: str, system: str = '', model: str = LOCAL_MODEL, max_tokens: int = LLM_JSON_MAX_TOKENS
) -> str:
    """One completion via the local Ollama MCP server. Returns the response text."""
    raw = await call_mcp_tool(
        LLM_LOCAL_MCP_URL, 'generate',
        {'prompt': prompt, 'system': system, 'model': model, 'max_tokens': max_tokens, 'temperature': 0.2},
    )
    data = json.loads(raw)
    response = data.get('response', '')
    if not response:
        raise RuntimeError(f'Local LLM returned empty response: {raw[:300]}')
    return response


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
        logger.warning(f'Triage unavailable, failing open (job goes to full rating): {ex}')
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
