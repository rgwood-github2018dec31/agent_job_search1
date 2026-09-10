"""Stage 1b scraper driven by an OpenRouter function-calling loop instead of the Claude Agent SDK.

Why this exists: Stage 1b reached 86.9% of total run cost ($48.24 of $55.52 over ten runs, peaking
at $10.06). Cost is `sum over turns of context_size`, and ~90% of it was cache traffic. A flash-tier
model with implicit caching prices the same traffic ~9x cheaper -- measured, not projected.

Two design rules hold this file together, both learned the hard way (see CLAUDE.md):

1. **The model chooses ACTIONS; code moves DATA.** No tool here accepts a page or job payload. A
   model relaying data cannot distinguish copying from producing, so when it has none it emits a
   plausible object rather than failing. Removing the argument removes the opportunity.
2. **Nothing in the job results list is ever clicked.** Enforced in `_guard_call` below, not merely
   prompted -- the accessibility tree disguises each card's Dismiss button as the card itself, and
   that already destroyed three real jobs.
"""
import asyncio
import json
import logging
from typing import Any
from collections.abc import Callable

from agentic_job_search import tools_generic
from agentic_job_search.config import (
    LLM_OPENROUTER_MCP_URL,
    MODEL_NAME_SCRAPER,
    SCRAPER_DISALLOWED_BROWSER_TOOLS,
    SCRAPER_OPENROUTER_MAX_ITERATIONS,
    SCRAPER_TOOL_RESULT_MAX_CHARS,
    UI_BLOCK_SIGNATURES,
)
from agentic_job_search.triage import call_mcp_tool, http_status_of, provider_error

logger = logging.getLogger(__name__)

# Gateway/server errors worth one retry of the same chat call. Deliberately disjoint from
# PROVIDER_UNAVAILABLE_STATUSES (401/402/403/429): those break every query and must abort at once.
TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504})
TRANSIENT_RETRY_DELAY_SECONDS = 5

# Substrings that mean a click would land inside the results list. `Dismiss` is the dangerous one:
# it is the only real <button> in a card and removes the job from the user's feed permanently.
FORBIDDEN_CLICK_SUBSTRINGS = ('job-card-component-ref', 'SearchResultsMainContent', 'Dismiss')

# Tools the model may call that are ours rather than the browser's. Every parameter is a DECISION;
# none carries page or job data. Adding one that does reintroduces the fabrication failure mode and
# is guarded by a unit test.
LOCAL_TOOL_DEFS: list[dict] = [
    {'type': 'function', 'function': {
        'name': 'run_ui_contract',
        'description': (
            'Check that the current results page is sound, and report it. Takes NO page data: the '
            'system runs the exact contract JavaScript, reads the result and judges it. Returns '
            '"ok ..." or an instruction telling you what to do. Call this before harvesting.'),
        'parameters': {'type': 'object', 'properties': {
            'region': {'type': 'string', 'description': 'Which configured region this search is for'},
        }, 'required': ['region'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'harvest_listings',
        'description': (
            'Harvest every job card on the current results page. Takes NO arguments. The system runs '
            'the exact harvest JavaScript and KEEPS the listings; you get back only a count. You '
            'never handle job data.'),
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'record_listings',
        'description': (
            'Record everything the most recent harvest_listings call returned. Takes NO arguments: '
            'the system records from its own copy. Deduplication, the age check, the already-applied '
            'blocklist, title screening and queueing all happen in code. Call harvest_listings first.'),
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'report_problem',
        'description': (
            'Say that you cannot proceed, and why. Use this whenever you do not have real data: a '
            'call failed, a result was missing or unreadable, the page is not what you expected, or '
            'you hit a challenge or verification page. Stopping and saying so is a SUCCESSFUL '
            'outcome. Never guess or reconstruct something you did not actually see.'),
        'parameters': {'type': 'object', 'properties': {
            'what_happened': {'type': 'string'},
        }, 'required': ['what_happened'], 'additionalProperties': False}}},
]

LOCAL_TOOL_NAMES = {t['function']['name'] for t in LOCAL_TOOL_DEFS}


def _guard_call(name: str, args: dict) -> str | None:
    """Refuse anything that would touch the job results list. Returns a refusal, or None to allow.

    Code-enforced rather than prompted: to the model a card and its Dismiss button are the same
    node, so wording cannot prevent this (CLAUDE.md, Account safety).
    """
    if name in ('browser_click', 'browser_hover'):
        blob = ' '.join(str(args.get(k, '')) for k in ('target', 'element', 'ref', 'selector'))
        for bad in FORBIDDEN_CLICK_SUBSTRINGS:
            if bad.lower() in blob.lower():
                logger.error(f'Stage 1b: REFUSED {name} naming {bad!r} — that is inside the results list')
                tools_generic._ui_alerts.append({
                    'kind': 'blocked_click', 'query': tools_generic._current_query or '',
                    'region': tools_generic._current_region or '',
                    'detail': f'refused {name} naming {bad!r} (results-list click)'})
                return (f'REFUSED: "{bad}" is inside the job results list. Clicking a result card hits '
                        'its Dismiss button and permanently destroys a real job. Filter chips are ABOVE '
                        'the results — click those, or use harvest_listings to read the list.')
    if name == 'browser_evaluate':
        body = str(args.get('function', ''))
        if '.click(' in body or 'dispatchEvent' in body:
            logger.error('Stage 1b: REFUSED browser_evaluate that drives the page')
            return 'REFUSED: JavaScript may read the page, never drive it. No .click() or dispatchEvent.'
    return None


def browser_tool_defs(mcp_tools: list) -> list[dict]:
    """Convert the Playwright server's own schemas into OpenAI-style function definitions."""
    disallowed = {t.replace('mcp__playwright__', '') for t in SCRAPER_DISALLOWED_BROWSER_TOOLS}
    return [
        {'type': 'function', 'function': {
            'name': t.name, 'description': (t.description or '')[:1024],
            'parameters': t.inputSchema or {'type': 'object', 'properties': {}}}}
        for t in mcp_tools if t.name not in disallowed
    ]


def _extract_json(raw: str) -> dict | None:
    """Pull the object out of a browser_evaluate response.

    playwright-mcp answers with a `### Result` section followed by `### Ran Playwright code`, and
    that trailing block ECHOES THE SCRIPT, braces included. Scanning to the last `}` therefore runs
    past the JSON into the echoed source -- which silently returned None on every real harvest while
    passing against hand-written fixtures. Isolate the Result section, then take the outermost
    balanced object.
    """
    body = raw
    if '### Ran Playwright code' in body:
        body = body.split('### Ran Playwright code', 1)[0]
    if '### Result' in body:
        body = body.split('### Result', 1)[1]
    start = body.find('{')
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(body)):
        ch = body[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1])
                except Exception as ex:
                    logger.warning(f'Stage 1b: evaluate result did not parse as JSON: {ex}')
                    return None
    return None


class ScrapeSession:
    """One query's scraping run: holds the browser, the harvest, and the conversation.

    The harvest lives HERE, in code. It is handed to `do_record_listings` directly and never passes
    through the model in either direction.
    """

    def __init__(self, browser_call: Callable, query: str, model: str | None = None):
        self._browser = browser_call
        self.query = query
        self.model = model or MODEL_NAME_SCRAPER
        self._harvest: list[dict] | None = None
        self.region = ''
        self.problems: list[str] = []
        self.blocked = False
        self.iterations = 0
        self.cost = 0.0
        self.usage = {'prompt': 0, 'completion': 0, 'cached': 0}

    async def _evaluate(self, js: str) -> dict | None:
        return _extract_json(await self._browser('browser_evaluate', {'function': js}))

    async def dispatch_local(self, name: str, args: dict) -> str:
        """Run one of our tools. Every branch obtains its own data."""
        from agentic_job_search.agent import SCRAPER_HARVEST_JS, SCRAPER_UI_CONTRACT_JS

        if name == 'run_ui_contract':
            self.region = str(args.get('region') or self.region)
            tools_generic._current_region = self.region
            report = await self._evaluate(SCRAPER_UI_CONTRACT_JS)
            if report is None:
                logger.warning('Stage 1b: UI contract evaluate returned no readable object')
                return ('could not read the page structure (the contract evaluate returned no '
                        'object). Do NOT guess what the page contained — call report_problem and stop.')
            if any(sig in str(report.get('body_sample', '')).lower() for sig in UI_BLOCK_SIGNATURES):
                self.blocked = True
            result = await tools_generic.report_search.handler(
                {'query': self.query, 'region': self.region, 'report': report})
            return result['content'][0]['text']

        if name == 'harvest_listings':
            harvest = await self._evaluate(SCRAPER_HARVEST_JS)
            if harvest is None:
                self._harvest = None
                return ('the harvest returned no readable object. Do NOT invent listings — call '
                        'report_problem and stop.')
            if harvest.get('error'):
                self._harvest = None
                logger.warning(f'Stage 1b: harvest error: {harvest["error"]}')
                tools_generic._ui_alerts.append({
                    'kind': 'empty_harvest', 'query': self.query, 'region': self.region,
                    'detail': str(harvest['error'])})
                return (f'harvest error: {harvest["error"]}. The markup has changed. Do NOT fall back '
                        'to clicking the results list and do NOT invent listings — call report_problem '
                        'and stop.')
            jobs = [j for j in (harvest.get('jobs') or []) if isinstance(j, dict)]
            self._harvest = jobs
            if not jobs:
                tools_generic._ui_alerts.append({
                    'kind': 'empty_harvest', 'query': self.query, 'region': self.region,
                    'detail': 'harvest returned zero listings'})
                return ('harvested 0 listings. If the page visibly shows results then the markup has '
                        'changed — call report_problem and stop. Do NOT invent listings.')
            logger.info(f'Stage 1b: harvested {len(jobs)} listing(s) for "{self.query}" / {self.region}')
            return f'harvested {len(jobs)} listing(s); the system is holding them. Call record_listings next.'

        if name == 'record_listings':
            if not self._harvest:
                return ('nothing harvested yet, or the last harvest failed. Call harvest_listings '
                        'first — do not describe listings yourself.')
            summary = await tools_generic.do_record_listings(self._harvest, query=self.query)
            self._harvest = None      # consumed, so each search must re-harvest
            return summary

        if name == 'report_problem':
            what = str(args.get('what_happened', ''))[:500]
            self.problems.append(what)
            logger.warning(f'Stage 1b: model reported a problem: {what}')
            tools_generic._ui_alerts.append({
                'kind': 'model_reported', 'query': self.query, 'region': self.region, 'detail': what})
            return 'Understood — recorded. Stopping is the right call; do not continue guessing.'

        return f'ERROR: unknown tool {name}'

    async def run(self, system_prompt: str, user_prompt: str, tools: list[dict]) -> None:
        """Drive the conversation until the model stops, it is blocked, or iterations run out."""
        messages: list[dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        for i in range(SCRAPER_OPENROUTER_MAX_ITERATIONS):
            self.iterations = i + 1
            chat_args = {'messages': messages, 'model': self.model, 'tools': tools}
            raw = await call_mcp_tool(LLM_OPENROUTER_MCP_URL, 'chat', chat_args)
            data = json.loads(raw)
            if not data.get('ok') and http_status_of(str(data.get('error') or '')) in TRANSIENT_HTTP_STATUSES:
                # A gateway hiccup is not this query failing. Without a retry one 502 discarded a
                # query's whole conversation 71 iterations in (2026-09-10), losing its second region.
                logger.warning(f'Stage 1b: transient OpenRouter error, retrying once in '
                               f'{TRANSIENT_RETRY_DELAY_SECONDS}s: {data.get("error")}')
                await asyncio.sleep(TRANSIENT_RETRY_DELAY_SECONDS)
                raw = await call_mcp_tool(LLM_OPENROUTER_MCP_URL, 'chat', chat_args)
                data = json.loads(raw)
            if not data.get('ok'):
                # A 402/401/403/429 is the PROVIDER refusing, not this query failing: every
                # remaining query would fail identically, so this must reach run_scraper as a
                # distinct type rather than be swallowed by its per-query handler.
                raise provider_error('OpenRouter chat failed',
                                     str(data.get('error') or raw[:300]))

            usage = data.get('usage') or {}
            self.cost += float(data.get('cost_usd') or 0.0)
            self.usage['prompt'] += usage.get('prompt_tokens', 0)
            self.usage['completion'] += usage.get('completion_tokens', 0)
            self.usage['cached'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0

            if text := data.get('content'):
                from agentic_job_search.agent import log_agent_text
                log_agent_text('Stage 1b', str(text))

            tool_calls = data.get('tool_calls')
            if not tool_calls:
                logger.info(f'Stage 1b: model finished after {self.iterations} iteration(s) '
                            f'(finish_reason={data.get("finish_reason")})')
                return
            messages.append({'role': 'assistant', 'content': data.get('content'), 'tool_calls': tool_calls})

            for call in tool_calls:
                name = call['function']['name']
                try:
                    args = json.loads(call['function'].get('arguments') or '{}')
                except Exception:
                    args = {}
                if (refusal := _guard_call(name, args)) is not None:
                    out = refusal
                elif name in LOCAL_TOOL_NAMES:
                    out = await self.dispatch_local(name, args)
                else:
                    # `filename` diverts the result to disk, leaving the model with nothing to read
                    # -- the exact condition that produced a fabricated report on 2026-08-21.
                    if name in ('browser_evaluate', 'browser_snapshot') and 'filename' in args:
                        args.pop('filename')
                    try:
                        out = await self._browser(name, args)
                    except Exception as ex:
                        out = f'ERROR calling {name}: {ex}'
                messages.append({'role': 'tool', 'tool_call_id': call.get('id', ''),
                                 'content': out[:SCRAPER_TOOL_RESULT_MAX_CHARS]})

            if self.blocked:
                logger.error('Stage 1b: block signature detected — stopping this query, not retrying')
                return
        logger.warning(f'Stage 1b: hit the {SCRAPER_OPENROUTER_MAX_ITERATIONS}-iteration cap for '
                       f'"{self.query}" — it may not have finished every search')
