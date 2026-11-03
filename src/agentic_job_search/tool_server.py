'''Wait Out Tool Server Outage: tell a local LLM tool server being down from a failure of the job.

On 2026-11-02 `tools_llm_remote_openrouter` was down for 2.3 minutes while the agent_meta1 daemon
restarted it. In that time 2 Stage 1b queries and the first 7 of 21 Stage 2 jobs failed in about
12 s each. Every one of those failures was about the server, not about the query or the job.
Code now asks the server whether it is back before writing off any more work.
'''

import asyncio
import logging
import re
import time

from agentic_job_search.config import (
    LLM_LOCAL_MCP_URL,
    LLM_OPENROUTER_MCP_URL,
    TOOL_SERVER_PROBE_TIMEOUT_SECONDS,
    TOOL_SERVER_WAIT_MAX_SECONDS,
    TOOL_SERVER_WAIT_POLL_SECONDS,
)
from utils_tools_n_agents_common import mcp_client

logger = logging.getLogger(__name__)

# The LLM tool servers this run depends on. The browser's MCP server is not here: a dead browser has
# its own recovery (BrowserServer), which restarts it rather than waiting.
TOOL_SERVER_URLS = (LLM_OPENROUTER_MCP_URL, LLM_LOCAL_MCP_URL)

# The shared client words every failure to reach a server `MCP <request> at <url> transport failed`
# (mcp_client._Request.transport_error). A timeout is worded `timed out` instead and does not match:
# the server was up and waiting on its provider.
_TRANSPORT_FAILED = re.compile(r'MCP .*? at (\S+) transport failed')


def _chain(ex: BaseException):
    '''`ex` and every exception it was raised from or during, each once.'''
    seen: set[int] = set()
    pending: list[BaseException | None] = [ex]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend((current.__cause__, current.__context__))


def outage_url(ex: BaseException) -> str | None:
    '''The tool server's URL when `ex` is a failure to reach one of TOOL_SERVER_URLS, else None.

    Callers wrap the client's error (gate_fields raises a RuntimeError listing every attempt, and
    rate_job joins the chain into one message), so the whole chain and each message are searched.'''
    for current in _chain(ex):
        for url in _TRANSPORT_FAILED.findall(str(current)):
            if url in TOOL_SERVER_URLS:
                return url
    return None


async def probe(url: str) -> str | None:
    '''None when the server at `url` answers tools/list, else why it did not.

    The failure is returned, not raised: an unreachable server is the expected answer while
    waiting, and wait_until_up logs the last one when it gives up.'''
    try:
        await mcp_client.list_tools(url, timeout_seconds=TOOL_SERVER_PROBE_TIMEOUT_SECONDS)
    except Exception as ex:
        return f'{type(ex).__name__}: {ex}'
    return None


async def wait_until_up(url: str, why: str) -> bool:
    '''Poll `url` until it answers or TOOL_SERVER_WAIT_MAX_SECONDS pass; True when it came back.

    `why` names the work that failed, for the log.'''
    started = time.monotonic()
    logger.warning(f'Tool server {url} is unreachable ({why}); waiting up to '
                   f'{TOOL_SERVER_WAIT_MAX_SECONDS}s for it to come back')
    while True:
        last_error = await probe(url)
        waited = time.monotonic() - started
        if last_error is None:
            logger.warning(f'Tool server {url} is back after {waited:.0f}s; retrying {why}')
            return True
        if waited >= TOOL_SERVER_WAIT_MAX_SECONDS:
            logger.error(f'Tool server {url} still unreachable after {waited:.0f}s; giving up on '
                         f'{why}. Last probe: {last_error}')
            return False
        await asyncio.sleep(TOOL_SERVER_WAIT_POLL_SECONDS)
