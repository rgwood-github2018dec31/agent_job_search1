"""OpenRouter job-page extractor: one posting's text in, one submit_job_extract call out.

Code reads the page (`linkedin_page.read_job_page`) and hands this loop the text; the model has
exactly one tool, `submit_job_extract`, and no browser. It used to navigate, snapshot and click
for itself (2026-09-24), which let a model decide what to read and gave it a click it did not need:
the description arrives complete without expanding "… more". The loop remains so a submit with
missing fields or invalid JSON can be retried, capped at `EXTRACTOR_OPENROUTER_MAX_ITERATIONS`.
The deterministic Haiku fallback in agent.py covers loop failures.
"""

import json
import logging

from utils_tools_n_agents_common.mcp_client import call_mcp_tool

from agentic_job_search.config import (
    EXTRACTOR_OPENROUTER_MAX_ITERATIONS,
    LLM_MCP_CALL_TIMEOUT_SECONDS,
    LLM_OPENROUTER_MCP_URL,
    MODEL_NAME_EXTRACTOR,
    SALARY_FIELD_DESCRIPTION,
)
from agentic_job_search.text_budget import snippet

logger = logging.getLogger(__name__)

_SUBMIT_TOOL_NAME = 'submit_job_extract'


OPENROUTER_EXTRACT_TOOLS = [
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
                "Pass residency_scope as 'country_only' if the posting requires LIVING IN the "
                "country it is advertised in ('Remote within country', 'must be based in "
                "Germany'), or 'area_wide' if it offers a whole multi-country area ('remote "
                "anywhere in the EU', 'Work from Anywhere', 'any EMEA country'); omit it when "
                "the posting does not say. This is where the HOLDER MUST LIVE, not where the "
                "job is advertised. "
                'Pass posting_language with the language the SOURCE PAGE ITSELF IS WRITTEN IN, '
                "lowercase, e.g. 'english', 'french'. Judge the original page, NOT the condensed "
                'English text you are writing here — you translate as you condense. The original job '
                "title is the clearest tell ('Scientifique principal des données en IA' means "
                "'french'). Omit only if genuinely undeterminable. "
                'Pass relocation with the country/city if the posting requires relocating to or '
                'residing in a specific place; omit for work-from-anywhere roles. '
                "Pass education_requirement as 'master' or 'phd' ONLY if an advanced degree is a hard "
                "requirement (e.g. 'MSc required', 'PhD is a must'); omit it when the degree is merely "
                "preferred, when equivalent experience is accepted, or when only a Bachelor's is required. "
                'Pass is_agency=true if the poster is a staffing firm, recruiting agency, or job '
                'aggregator reposting on behalf of another company rather than the employer that would '
                "actually hire, and pass end_client with that hiring company's name if the posting "
                "names it (agencies usually keep it anonymous, e.g. 'our client, a leading fintech' — "
                'leave end_client empty in that case).'
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
                    'salary': {'type': 'string', 'description': SALARY_FIELD_DESCRIPTION},
                    'sponsorship_note': {'type': 'string'},
                    'language_requirement': {'type': 'string', 'description': 'Explicitly required languages, comma-separated lowercase'},
                    'posting_language': {'type': 'string', 'description': "Language the SOURCE page is written in, lowercase e.g. 'english', 'french' — judge the original page, not your condensed English output"},
                    'residency_scope': {'type': 'string', 'enum': ['country_only', 'area_wide', ''],
                                        'description': "Whether the posting pins residence to the country it is anchored in ('country_only') or offers a whole multi-country area ('area_wide'); empty when the posting does not say"},
                    'relocation': {'type': 'string', 'description': 'Location the candidate must relocate to / reside in, if required'},
                    'education_requirement': {'type': 'string', 'description': "'master' or 'phd' ONLY if an advanced degree is a HARD requirement; empty when merely preferred or when equivalent experience is accepted"},
                    # Must stay in sync with submit_job_extract in tools_generic.py — a field added
                    # to one extractor and not the other silently returns empty for that provider.
                    'is_agency': {
                        'type': 'boolean',
                        'description': (
                            'True if the poster is a staffing firm, recruiting agency, or job aggregator '
                            'reposting on behalf of another company, rather than the employer that '
                            'would actually hire.'
                        ),
                    },
                    'end_client': {
                        'type': 'string',
                        'description': 'The company that would actually hire, if the posting names one; otherwise an empty string',
                    },
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
        'posting_language': str(args.get('posting_language') or '').strip().lower(),
        'relocation': args.get('relocation', ''),
        'residency_scope': str(args.get('residency_scope') or '').strip().lower(),
        'workplace_type': str(args.get('workplace_type') or '').strip().lower(),
        'education_requirement': str(args.get('education_requirement') or '').strip().lower(),
        # None (not False) when the model said nothing, so derive_agency_posting() can tell
        # "judged not an agency" from "did not answer".
        'is_agency': args.get('is_agency'),
        'end_client': str(args.get('end_client') or '').strip(),
    }


async def extract_job_page_openrouter(
    candidate: dict, page_text: str, stage_stats: dict, system_prompt: str
) -> dict | None:
    """Condense one posting's page text via submit_job_extract. Returns the extract dict, or None
    on failure (the caller falls back to the deterministic Haiku path)."""
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': (
            f"Extract this job posting:\n"
            f"Company: {candidate['company']}\n"
            f"Title: {candidate['title']}\n"
            f"URL: {candidate['url']}\n"
            f"Posted: {candidate['date_posted']}\n"
            f"Snippet: {candidate['snippet']}\n\n"
            f"--- PAGE TEXT ---\n{page_text}\n--- END PAGE TEXT ---"
        )},
    ]
    total_cost = 0.0
    for iteration in range(EXTRACTOR_OPENROUTER_MAX_ITERATIONS):
        raw = await call_mcp_tool(
            LLM_OPENROUTER_MCP_URL, 'chat',
            {'messages': messages, 'model': MODEL_NAME_EXTRACTOR, 'tools': OPENROUTER_EXTRACT_TOOLS},
            timeout_seconds=LLM_MCP_CALL_TIMEOUT_SECONDS,
        )
        data = json.loads(raw)
        if not data.get('ok'):
            logger.warning(
                f"Extract (openrouter): {candidate['company']} — {candidate['title']}: "
                f"chat failed: {data.get('error') or snippet(raw)}"
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
                logger.warning(
                    f"Extract (openrouter): {candidate['company']} — {candidate['title']}: "
                    f"{name} called with invalid JSON arguments ({ex}): "
                    f"{snippet(tool_call['function'].get('arguments'))}"
                )
                messages.append({
                    'role': 'tool', 'tool_call_id': call_id,
                    'content': f'Error: invalid JSON arguments ({ex}). Retry the call with valid JSON.',
                })
                continue

            if name != _SUBMIT_TOOL_NAME:
                messages.append({
                    'role': 'tool', 'tool_call_id': call_id,
                    'content': f'Error: unknown tool {name!r}. The page text is in the user message; '
                               f'the only tool is {_SUBMIT_TOOL_NAME}.',
                })
                continue
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
                f"{len(page_text)} page chars → {len(extract['description'])} chars condensed in "
                f"{iteration + 1} iteration(s), ${total_cost:.4f}"
            )
            return extract

    logger.warning(
        f"Extract (openrouter): {candidate['company']} — {candidate['title']}: "
        f"iteration cap ({EXTRACTOR_OPENROUTER_MAX_ITERATIONS}) reached without submit, ${total_cost:.4f} spent"
    )
    return None
