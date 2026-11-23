'''Check LinkedIn Session: one visit to the feed at the start of a run, before any search.

On 2026-11-23 the profile had been signed out since the day before. Nothing in code noticed it:
the scraper model met the sign-in wall on every query, each query was retried once, and the run
spent 11 minutes going back to the wall about 12 times. The health report put it down to
"UI CHANGED". Now code opens the feed once. A signed-in member stays on the feed, and a
signed-out profile is redirected to a login, authwall or checkpoint path or served the guest
page. The run stops at that point, before any query. The same visit, made every run, also keeps
the session in use.

The page is only read (one `browser_evaluate`), never driven. The markers are the URL path and
link text and hrefs, never LinkedIn's hashed class names.
'''

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from agentic_job_search.config import (
    LINKEDIN_SESSION_CHECK_URL,
    LINKEDIN_SIGNED_OUT_LOGIN_PATH_PREFIXES,
)
from agentic_job_search.linkedin_page import BrowserCall
from agentic_job_search.scrape_openrouter import parse_evaluate_result
from agentic_job_search.text_budget import snippet

logger = logging.getLogger(__name__)

# Read-only: the URL the browser settled on, and the sign-in / join links only a guest page shows.
LINKEDIN_SESSION_CHECK_JS = '''() => {
  const links = Array.from(document.querySelectorAll('a'));
  const guestLinks = links.filter(a => {
    const text = (a.innerText || '').trim().toLowerCase();
    const href = a.getAttribute('href') || '';
    return (text === 'sign in' && href.includes('/login'))
        || (text === 'join now' && href.includes('/signup'));
  }).map(a => (a.innerText || '').trim());
  return {url: location.href, title: document.title, guest_links: guestLinks};
}'''


@dataclass(frozen=True)
class SessionCheck:
    signed_in: bool
    url: str
    reason: str


def decide_session(page: dict) -> SessionCheck:
    '''Signed in or not, from the read-only capture of the session-check page.

    Raises ValueError when the capture has no url: an unreadable page decides nothing.
    '''
    url = page.get('url')
    if not isinstance(url, str) or not url:
        raise ValueError(f'session check: the page capture has no url: {page!r}')
    path = urlparse(url).path
    if path.startswith(LINKEDIN_SIGNED_OUT_LOGIN_PATH_PREFIXES):
        return SessionCheck(False, url, f'redirected to {path}')
    guest_links = page.get('guest_links') or []
    if guest_links:
        return SessionCheck(False, url, f'guest page (links: {", ".join(map(str, guest_links))})')
    return SessionCheck(True, url, 'no sign-in redirect or guest links')


async def check_linkedin_session(call: BrowserCall) -> SessionCheck:
    '''Open the feed once and decide whether the profile is signed in. Raises when the page cannot be read.'''
    await call('browser_navigate', {'url': LINKEDIN_SESSION_CHECK_URL})
    raw = await call('browser_evaluate', {'function': LINKEDIN_SESSION_CHECK_JS})
    page = parse_evaluate_result(raw)
    if not isinstance(page, dict):
        raise ValueError(f'session check: browser_evaluate returned no object ({len(raw)} chars): {snippet(raw)!r}')
    check = decide_session(page)
    logger.info(f'LinkedIn session check: {"signed in" if check.signed_in else "SIGNED OUT"} '
                f'at {check.url} ({check.reason}; title {page.get("title")!r})')
    return check
