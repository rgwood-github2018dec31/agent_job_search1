"""Search the Web: one query through the OpenRouter tool server's `web_search` (2026-11-19).

Only the query leaves the machine; the engine chooses the pages. Used by Applied Organization records
(an organization's relatives) and by Search Organization (an organization nobody could identify).
"""

import json
import logging
from dataclasses import dataclass

from utils_tools_n_agents_common.mcp_client import call_mcp_tool

from agentic_job_search import job_trace
from agentic_job_search.config import (
    LLM_OPENROUTER_MCP_URL,
    MODEL_NAME_WEB_SEARCH,
    WEB_SEARCH_ENGINE,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT_SECONDS,
)
from agentic_job_search.text_budget import quote_for_error

logger = logging.getLogger(__name__)


class WebSearchError(RuntimeError):
    '''The search failed; the message names the query and carries the tool server's reply.'''


@dataclass(frozen=True)
class WebSearchResults:
    query: str
    query_run: str                 # what the model actually searched
    results: list[dict]            # [{'title', 'url', 'content'}], as the engine returned them
    cost_usd: float


async def search(query: str, what: str, stage_stats: dict | None = None) -> WebSearchResults:
    '''Run `query` once; raises WebSearchError naming `what` and the query on any failure. A failure
    to reach the tool server propagates as call_mcp_tool raised it.'''
    args = {'query': query, 'model': MODEL_NAME_WEB_SEARCH, 'max_results': WEB_SEARCH_MAX_RESULTS,
            'engine': WEB_SEARCH_ENGINE}
    raw = await call_mcp_tool(LLM_OPENROUTER_MCP_URL, 'web_search', args, timeout_seconds=WEB_SEARCH_TIMEOUT_SECONDS)
    where = f'{what}: web_search {query!r} via {MODEL_NAME_WEB_SEARCH}/{WEB_SEARCH_ENGINE}'
    try:
        data = json.loads(raw)
    except ValueError as ex:
        raise WebSearchError(f'{where} replied with non-JSON ({ex}): {quote_for_error(raw, "web_search reply")!r}') from ex
    if not isinstance(data, dict) or data.get('ok') is not True:
        raise WebSearchError(f'{where} failed: {quote_for_error(raw, "web_search failure reply")!r}')
    results = data.get('results')
    if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
        raise WebSearchError(f'{where}: "results" is not a list of objects: {quote_for_error(raw, "web_search reply")!r}')
    cost = data.get('cost_usd')
    if cost is None:
        logger.warning(f'{where} reported no cost (counted as $0)')
    cost = float(cost or 0.0)
    if stage_stats is not None:
        stage_stats['cost'] = stage_stats.get('cost', 0.0) + cost
    found = WebSearchResults(query=query, query_run=str(data.get('query_run') or ''), results=results, cost_usd=cost)
    job_trace.record('web_search', what=what, query=query, query_run=found.query_run, engine=WEB_SEARCH_ENGINE,
                     model=MODEL_NAME_WEB_SEARCH, results=[{'title': r.get('title'), 'url': r.get('url')} for r in results],
                     cost_usd=cost)
    logger.info(f'{where}: {len(results)} result(s), ${cost:.4f}')
    return found
