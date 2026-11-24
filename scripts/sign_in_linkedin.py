'''Sign In to LinkedIn: open the agent's browser profile on the login page so the user can sign in.

The profile is opened through the agent's own pinned @playwright/mcp (BrowserServer, visible) and
never through a Chromium started by hand. Playwright's browser keeps its cookies under a mock
keychain. On 2026-11-23 a login made in a hand-started Chromium was saved under the real macOS
Keychain, and the agent's next session check still found the profile signed out.

While the user signs in, the script only reads `location.href` and never clicks or types. Once
the page is a signed-in LinkedIn page, it waits LINKEDIN_SIGN_IN_SETTLE_SECONDS for the cookies
to be written, closes the browser, and checks the session headless, the way a run reads it. It
holds the run lock throughout, so no run can start on the profile meanwhile.

Usage:
    uv run python scripts/sign_in_linkedin.py
'''

import asyncio
import logging
import sys
import time

import agentic_job_search.tools_generic as tools
from agentic_job_search.agent import BrowserServer
from agentic_job_search.config import (
    LINKEDIN_SIGN_IN_POLL_SECONDS,
    LINKEDIN_SIGN_IN_SETTLE_SECONDS,
    LINKEDIN_SIGN_IN_URL,
    LINKEDIN_SIGN_IN_WAIT_MAX_SECONDS,
    SECONDS_PER_MINUTE,
)
from agentic_job_search.linkedin_session import check_linkedin_session, is_past_login
from agentic_job_search.scrape_openrouter import parse_evaluate_result
from utils_tools_n_agents_common.mcp_client import mcp_session

logger = logging.getLogger('sign_in_linkedin')

READ_URL_JS = '() => ({url: location.href})'


async def wait_for_sign_in() -> bool:
    '''Open the login page visibly and wait for the user to get past it. True once signed in.'''
    browser = BrowserServer('visible')
    await browser.start()
    try:
        async with mcp_session(browser.mcp['url']) as call:
            await call('browser_navigate', {'url': LINKEDIN_SIGN_IN_URL})
            logger.info(f'Login page open in the agent profile; sign in there '
                        f'(waiting up to {LINKEDIN_SIGN_IN_WAIT_MAX_SECONDS // SECONDS_PER_MINUTE} min)')
            deadline = time.monotonic() + LINKEDIN_SIGN_IN_WAIT_MAX_SECONDS
            while time.monotonic() < deadline:
                await asyncio.sleep(LINKEDIN_SIGN_IN_POLL_SECONDS)
                raw = await call('browser_evaluate', {'function': READ_URL_JS})
                url = (parse_evaluate_result(raw) or {}).get('url', '')
                if is_past_login(url):
                    logger.info(f'Signed in: now at {url}; waiting {LINKEDIN_SIGN_IN_SETTLE_SECONDS}s '
                                f'for the cookies to be saved, then closing the browser')
                    await asyncio.sleep(LINKEDIN_SIGN_IN_SETTLE_SECONDS)
                    return True
            logger.error(f'Timed out after {LINKEDIN_SIGN_IN_WAIT_MAX_SECONDS // SECONDS_PER_MINUTE} min without a sign-in')
            return False
    finally:
        await browser.stop()


async def confirm_headless() -> bool:
    '''The session check a run makes, on a fresh headless browser.'''
    browser = BrowserServer('headless')
    await browser.start()
    try:
        async with mcp_session(browser.mcp['url']) as call:
            check = await check_linkedin_session(call)
    finally:
        await browser.stop()
    return check.signed_in


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if active := tools.acquire_run_lock('sign-in'):
        logger.error(f'A run is using the profile (pid {active["pid"]}, mode {active.get("mode")}, '
                     f'started {active.get("started_at")}); sign in after it finishes')
        return 1
    try:
        if not asyncio.run(wait_for_sign_in()):
            return 1
        if not asyncio.run(confirm_headless()):
            logger.error('The headless session check still finds the profile signed out; sign in again')
            return 1
        logger.info('Signed in: the next run will find the session')
        return 0
    finally:
        tools.release_run_lock()


if __name__ == '__main__':
    sys.exit(main())
