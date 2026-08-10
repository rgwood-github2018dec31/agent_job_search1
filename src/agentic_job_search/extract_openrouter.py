"""OpenRouter-driven agentic job-page extractor.

Runs a function-calling agent loop: an OpenRouter model (via the tools_llm_remote_openrouter
MCP server's `chat` tool) decides which browser tools to call, this module executes them
against the shared Playwright MCP session, and the loop ends when the model calls
submit_job_extract or the iteration cap is hit. Used when EXTRACTOR_PROVIDER='openrouter';
the deterministic Haiku fallback in agent.py still covers loop failures.
"""

import json
import logging

from agentic_job_search.config import (
    EXTRACTOR_OPENROUTER_MAX_ITERATIONS,
    EXTRACTOR_TOOL_RESULT_MAX_CHARS,
    LLM_OPENROUTER_MCP_URL,
    OPENROUTER_MODEL,
)
from agentic_job_search.triage import call_mcp_tool, mcp_session

logger = logging.getLogger(__name__)

_BROWSER_TOOL_NAMES = {'browser_navigate', 'browser_snapshot', 'browser_click', 'browser_wait_for'}

OPENROUTER_EXTRACT_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'browser_navigate',
            'description': 'Navigate the browser to a URL.',
            'parameters': {
                'type': 'object',
                'properties': {'url': {'type': 'string'}},
                'required': ['url'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'browser_snapshot',
            'description': 'Capture an accessibility snapshot of the current page as text.',
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'browser_click',
            'description': 'Click an element on the page.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'element': {'type': 'string', 'description': 'Human-readable element description'},
                    'target': {'type': 'string', 'description': 'Exact element reference from the page snapshot, e.g. "e611"'},
                },
                'required': ['element', 'target'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'browser_wait_for',
            'description': 'Wait for the given number of seconds (e.g. for dynamic content to render).',
            'parameters': {
                'type': 'object',
                'properties': {'time': {'type': 'number', 'description': 'Seconds to wait'}},
                'required': ['time'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'submit_job_extract',
            'description': (
                'Submit the condensed extract of the job posting. Call exactly once when done. '
                'Set closed=true if the page shows "No longer accepting applications". '
                'Include the workplace type (Remote / Hybrid / On-site) in location, and also pass '
                "workplace_type as exactly 'remote', 'hybrid', or 'onsite'. Any mention of required "
                "days in the office (e.g. '2-3 days onsite') is 'hybrid', not 'remote'. "
                'Pass sponsorship_note with any visa/work-authorization statement, verbatim. '
                'Pass language_requirement with languages explicitly REQUIRED (not nice-to-have), '
                "comma-separated lowercase, e.g. 'english, german'; omit if none stated. "
                'Pass relocation with the country/city if the posting requires relocating to or '
                'residing in a specific place; omit for work-from-anywhere roles. '
                "Pass education_requirement as 'master' or 'phd' ONLY if an advanced degree is a hard "
                "requirement (e.g. 'MSc required', 'PhD is a must'); omit it when the degree is merely "
                'preferred, when equivalent experience is accepted, or when only a Bachelor\'s is required.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'company': {'type': 'string'},
                    'description': {'type': 'string', 'description': 'Condensed posting content (requirements, responsibilities, stack, seniority)'},
                    'location': {'type': 'string'},
                    'workplace_type': {'type': 'string', 'description': "Work arrangement: 'remote', 'hybrid', or 'onsite'"},
                    'date_posted': {'type': 'string'},
                    'closed': {'type': 'boolean'},
                    'salary': {'type': 'string'},
                    'sponsorship_note': {'type': 'string'},
                    'language_requirement': {'type': 'string', 'description': "Explicitly required languages, comma-separated lowercase"},
                    'relocation': {'type': 'string', 'description': 'Location the candidate must relocate to / reside in, if required'},
                    'education_requirement': {'type': 'string', 'description': "'master' or 'phd' ONLY if an advanced degree is a HARD requirement; empty when merely preferred or when equivalent experience is accepted"},
                },
                'required': ['title', 'company', 'description'],
            },
        },
    },
]

_SUBMIT_REQUIRED_FIELDS = ('title', 'company', 'description')


def _extract_from_submit_args(args: dict) -> dict:
    return {
        'title': args['title'], 'company': args['company'], 'description': args['description'],
        'location': args.get('location', ''), 'date_posted': args.get('date_posted', ''),
        'closed': bool(args.get('closed', False)), 'salary': args.get('salary', ''),
        'sponsorship_note': args.get('sponsorship_note', ''),
        'language_requirement': args.get('language_requirement', ''),
        'relocation': args.get('relocation', ''),
        'workplace_type': str(args.get('workplace_type') or '').strip().lower(),
        'education_requirement': str(args.get('education_requirement') or '').strip().lower(),
    }


async def extract_job_page_openrouter(
    candidate: dict, playwright_mcp_url: str, stage_stats: dict, system_prompt: str
) -> dict | None:
    """Extract a job page via an OpenRouter function-calling loop. Returns the extract
    dict, or None on failure (caller falls back to the deterministic Haiku path)."""
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': (
            f"Extract this job posting:\n"
            f"Company: {candidate['company']}\n"
            f"Title: {candidate['title']}\n"
            f"URL: {candidate['url']}\n"
            f"Posted: {candidate['date_posted']}\n"
            f"Snippet: {candidate['snippet']}"
        )},
    ]
    total_cost = 0.0
    async with mcp_session(playwright_mcp_url) as browser_call:
        for iteration in range(EXTRACTOR_OPENROUTER_MAX_ITERATIONS):
            raw = await call_mcp_tool(
                LLM_OPENROUTER_MCP_URL, 'chat',
                {'messages': messages, 'model': OPENROUTER_MODEL, 'tools': OPENROUTER_EXTRACT_TOOLS},
            )
            data = json.loads(raw)
            if not data.get('ok'):
                logger.warning(
                    f"Extract (openrouter): {candidate['company']} — {candidate['title']}: "
                    f"chat failed: {data.get('error', raw[:300])}"
                )
                return None
            usage = data.get('usage') or {}
            stage_stats['cost'] += float(data.get('cost_usd') or 0.0)
            stage_stats['input_tokens'] += usage.get('prompt_tokens', 0)
            stage_stats['output_tokens'] += usage.get('completion_tokens', 0)
            total_cost += float(data.get('cost_usd') or 0.0)

            tool_calls = data.get('tool_calls')
            if not tool_calls:
                logger.warning(
                    f"Extract (openrouter): {candidate['company']} — {candidate['title']}: "
                    f"model stopped without submitting (iteration {iteration + 1}, "
                    f"finish_reason={data.get('finish_reason')})"
                )
                return None

            messages.append({'role': 'assistant', 'content': data.get('content'), 'tool_calls': tool_calls})
            for tool_call in tool_calls:
                name = tool_call['function']['name']
                call_id = tool_call.get('id', '')
                try:
                    args = json.loads(tool_call['function'].get('arguments') or '{}')
                except json.JSONDecodeError as ex:
                    messages.append({
                        'role': 'tool', 'tool_call_id': call_id,
                        'content': f'Error: invalid JSON arguments ({ex}). Retry the call with valid JSON.',
                    })
                    continue

                if name == 'submit_job_extract':
                    missing = [f for f in _SUBMIT_REQUIRED_FIELDS if not args.get(f)]
                    if missing:
                        messages.append({
                            'role': 'tool', 'tool_call_id': call_id,
                            'content': f'Error: missing required field(s) {missing}. Call submit_job_extract again with them.',
                        })
                        continue
                    extract = _extract_from_submit_args(args)
                    logger.info(
                        f"Extract (openrouter): {candidate['company']} — {candidate['title']}: "
                        f"{len(extract['description'])} chars condensed in {iteration + 1} iteration(s), "
                        f"${total_cost:.4f}"
                    )
                    return extract

                if name in _BROWSER_TOOL_NAMES:
                    try:
                        result_text = await browser_call(name, args)
                    except Exception as ex:
                        result_text = f'Error executing {name}: {ex}'
                    logger.debug(
                        f'Extract (openrouter) iteration {iteration + 1}: {name}({args}) '
                        f'→ {len(result_text)} chars: {result_text[:200]!r}'
                    )
                    messages.append({
                        'role': 'tool', 'tool_call_id': call_id,
                        'content': result_text[:EXTRACTOR_TOOL_RESULT_MAX_CHARS],
                    })
                else:
                    messages.append({
                        'role': 'tool', 'tool_call_id': call_id,
                        'content': f'Error: unknown tool {name!r}. Available: '
                                   f'{sorted(_BROWSER_TOOL_NAMES)} and submit_job_extract.',
                    })

    logger.warning(
        f"Extract (openrouter): {candidate['company']} — {candidate['title']}: "
        f"iteration cap ({EXTRACTOR_OPENROUTER_MAX_ITERATIONS}) reached without submit, ${total_cost:.4f} spent"
    )
    return None
