'''Read a LinkedIn job page and a company page as a signed-out visitor: one plain HTTP GET each.

Why (2026-11-06): two runs in a row (11-04, 11-05) stopped within two minutes of Stage 2 starting,
on a Cloudflare "you have been blocked" page served to the logged-in browser on a company About
page. Both pages are public: a job page (`/jobs/view/<id>/`) and a company page (`/company/<slug>/`)
load without signing in, so reading them needs neither the browser nor the account. This is the
default (`--job-and-company-pages-via`, LINKEDIN_PAGE_READER_DEFAULT); the signed-in browser
remains available for the job page.

What a signed-out visitor sees, measured on 2026-11-06 (Auxo Talent, job 4482929457):

- The job page is LinkedIn's public markup, rendered on the server: title, company and its link,
  location, posting age, the description, and the criteria list (seniority, employment type,
  industries). It has NO workplace badge (Remote / Hybrid / On-site), so `HeaderCard.workplace` is
  '' on this path and the workplace is read from the description. The age is coarser ("1 week
  ago" where the signed-in page says "6 days ago").
- `/company/<slug>/about/` redirects a visitor to the login page; `/company/<slug>/` does not, and
  states the Website. Its "Affiliated pages" do not say which one is the parent, so
  `CompanyAbout.parent_*` stay '' (the model fills what the name says), and every affiliated page
  is kept in `CompanyAbout.affiliated`, each matching as the same organization (Match Applied
  Company).

The public markup is matched on its readable BEM class names and `data-test-id`s (`top-card-layout__
title`, `about-us__website`), which have been stable for years, unlike the signed-in page's hashed
classes, which are never matched. A missing element is a markup change: the read fails with the
page kept, and the caller falls back as for any failed read.

Every signed-out request waits its turn on one pacer (`pacing.signed_out_request_interval_seconds`,
set by the user after a 429 at ~27 requests in 15 minutes, 2026-11-06). At that pace a run's job
pages take about a minute each, so job_page_prefetch.JobPagePrefetcher starts reading them while Stage 1b is still
searching, in the order the search queues them; Stage 2 takes each page as it reaches the job.

A company page is looked up in the Wayback Machine first and read from LinkedIn only when no
recent archived copy parses (COMPANY_PAGE_ARCHIVE_MAX_AGE_DAYS); the archive has its own pacer.

The workplace the public page lacks is taken from the search card, which carries it after the
location ("Warsaw (Remote)").

A challenge stops the run exactly as on the browser path (LinkedInBlockedError): an HTTP status in
`LINKEDIN_SIGNED_OUT_BLOCK_STATUSES`, a redirect to a login or checkpoint page for a job page, or a
UI_BLOCK_SIGNATURES phrase on a page without the expected content. No cookies are ever sent.
'''
import asyncio
import copy
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from http import HTTPStatus
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from agentic_job_search import minimal_archive, preferences, tools_generic
from agentic_job_search.config import (
    COMPANY_PAGE_ARCHIVE_INTERVAL_SECONDS,
    COMPANY_PAGE_ARCHIVE_MAX_AGE_DAYS,
    LINKEDIN_COMPANY_URL,
    LINKEDIN_EMPLOYMENT_TYPES,
    LINKEDIN_JOB_DESCRIPTION_HEADING,
    LINKEDIN_SIGNED_OUT_BLOCK_STATUSES,
    LINKEDIN_SIGNED_OUT_FETCH_TIMEOUT_SECONDS,
    LINKEDIN_SIGNED_OUT_HEADERS,
    LINKEDIN_SIGNED_OUT_LOGIN_PATH_PREFIXES,
    LINKEDIN_WORKPLACE_BADGES,
    WAYBACK_AVAILABILITY_URL,
    WAYBACK_FETCH_TIMEOUT_SECONDS,
    WAYBACK_HEADERS,
    WAYBACK_RAW_SNAPSHOT_URL,
    WAYBACK_REFUSAL_STATUSES,
    WAYBACK_TIMESTAMP_FORMAT,
)
from agentic_job_search.linkedin_page import (
    _COMPANY_SLUG_RE,
    BLOCKED_PAGE_SUFFIX,
    CompanyAbout,
    HeaderCard,
    JobPage,
    LinkedInBlockedError,
    NotAnAboutPageError,
    _keep_unreadable_about_page,
    is_closed_text,
    localize_assets,
    norm,
    raw_html_path,
    save_raw_html,
    strip_js,
    visible_block_match,
)
from agentic_job_search.salary import mentions_pay
from agentic_job_search.text_budget import snippet

logger = logging.getLogger(__name__)

SOURCE_SIGNED_OUT = 'signed_out'   # JobPage.source on this path
_TITLE_SELECTOR = '.top-card-layout__title'
_TOP_CARD_SELECTOR = '.top-card-layout'
_COMPANY_LINK_SELECTOR = '.topcard__org-name-link'
_LOCATION_SELECTOR = '.topcard__flavor--bullet'
_POSTED_SELECTOR = '.posted-time-ago__text'
_SALARY_SELECTOR = '.compensation__salary'
_DESCRIPTION_SELECTOR = '.show-more-less-html__markup'
_CRITERIA_ITEM_SELECTOR = '.description__job-criteria-item'
_CRITERIA_LABEL_SELECTOR = '.description__job-criteria-subheader'
_CRITERIA_VALUE_SELECTOR = '.description__job-criteria-text'
_CRITERIA_EMPLOYMENT_LABEL = 'Employment type'
_CRITERIA_HEADING = 'Job criteria'
_ABOUT_SELECTOR = '[data-test-id="about-us"]'
_WEBSITE_SELECTOR = '[data-test-id="about-us__website"]'
# Outbound links: /redir/redirect?url=… on the public page, /safety/go?url=… on the signed-in one.
_REDIRECT_PATHS = ('/redir/redirect', '/safety/go')
_REDIRECT_TARGET_KEY = 'url'
# Each "Affiliated pages" entry is a link tracked under exactly this name ("…_show_more" is the toggle).
_AFFILIATED_TRACKING_NAME = 'affiliated-pages'


# The search card's location ends with the workplace badge in brackets: 'Lisbon, Portugal (Hybrid)'.
_CARD_WORKPLACE_RE = re.compile(r'\((?P<badge>' + '|'.join(map(re.escape, LINKEDIN_WORKPLACE_BADGES)) + r')\)\s*$')


class SignedOutPageError(RuntimeError):
    '''The public page could not be fetched or did not have the expected content. Not a challenge.'''


class NoArchivedCopyError(RuntimeError):
    '''No usable archived copy of a page: none, too old, or the archive could not be read.'''


class SignedOutPacer:
    '''Lets one request to a site start at a time, each a drawn interval after the one before.

    `interval` gives the (min, max) range, drawn per request. One pacer per site serves the whole
    process: for LinkedIn, `pacing.signed_out_request_interval_seconds`, so the job pages read
    ahead during Stage 1b and the company pages read in Stage 2 share one pace.
    '''

    def __init__(self, interval: Callable[[], tuple[float, float]]) -> None:
        self._interval = interval
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self, what: str) -> None:
        async with self._lock:
            delay = self._next_at - time.monotonic()
            if delay > 0:
                logger.debug(f'Signed-out pace: waiting {delay:.0f}s before {what}')
                await asyncio.sleep(delay)
            self._next_at = time.monotonic() + random.uniform(*self._interval())


pacer = SignedOutPacer(lambda: preferences.signed_out_request_interval_seconds())
archive_pacer = SignedOutPacer(lambda: COMPANY_PAGE_ARCHIVE_INTERVAL_SECONDS)


def card_workplace(card_location: str) -> str:
    '''The workplace the search card states after its location ('remote' for 'Warsaw (Remote)'), or ''.'''
    m = _CARD_WORKPLACE_RE.search(card_location or '')
    return LINKEDIN_WORKPLACE_BADGES[m.group('badge')] if m else ''


def card_place(card_location: str) -> str:
    '''The search card's location without its workplace badge ('Warsaw' for 'Warsaw (Remote)').'''
    return _CARD_WORKPLACE_RE.sub('', card_location or '').strip()


def _get(url: str, what: str) -> requests.Response:
    '''One cookie-free GET. Raises LinkedInBlockedError on a challenge status, SignedOutPageError otherwise.'''
    try:
        resp = requests.get(url, headers=LINKEDIN_SIGNED_OUT_HEADERS, timeout=LINKEDIN_SIGNED_OUT_FETCH_TIMEOUT_SECONDS)
    except requests.RequestException as ex:
        raise SignedOutPageError(f'{what}: GET {url} failed: {type(ex).__name__}: {ex}') from ex
    if resp.status_code in LINKEDIN_SIGNED_OUT_BLOCK_STATUSES:
        raise LinkedInBlockedError(f'{what}: GET {url} returned HTTP {resp.status_code} {resp.reason}, LinkedIn '
                                   f'refusing a signed-out visitor; body: {snippet(resp.text)!r}')
    if resp.status_code != HTTPStatus.OK:
        raise SignedOutPageError(f'{what}: GET {url} returned HTTP {resp.status_code} {resp.reason}; '
                             f'body: {snippet(resp.text)!r}')
    return resp


def _redirected_to_login(resp: requests.Response) -> bool:
    return urlparse(resp.url).path.startswith(LINKEDIN_SIGNED_OUT_LOGIN_PATH_PREFIXES)


def _raise_if_challenge(soup: BeautifulSoup, url: str, what: str, html: str, keep_as: str = '') -> None:
    '''A page without the expected content that shows a UI_BLOCK_SIGNATURES phrase is a challenge.'''
    phrase, context = visible_block_match(soup)
    if phrase:
        kept = _keep_unreadable_about_page(keep_as, html, suffix=BLOCKED_PAGE_SUFFIX) if keep_as else 'not kept'
        raise LinkedInBlockedError(f'{what}: {url} shows "{phrase}" in: {context!r}; page {kept}')


def _text(root: Tag, selector: str) -> str:
    el = root.select_one(selector)
    return norm(el.get_text(' ')) if el is not None else ''


def signed_out_header(soup: BeautifulSoup) -> HeaderCard:
    '''The public top card. Raises SignedOutPageError when it has no title (not a job page, or new markup).'''
    title = _text(soup, _TITLE_SELECTOR)
    if not title:
        raise SignedOutPageError(f'no {_TITLE_SELECTOR} on the page')
    criteria = signed_out_criteria(soup)
    employment = criteria.get(_CRITERIA_EMPLOYMENT_LABEL, '')
    salary = _text(soup, _SALARY_SELECTOR)
    card = soup.select_one(_TOP_CARD_SELECTOR)
    return HeaderCard(
        company=_text(soup, _COMPANY_LINK_SELECTOR), title=title, location=_text(soup, _LOCATION_SELECTOR),
        posted=_text(soup, _POSTED_SELECTOR),
        workplace='',   # the public page has no workplace badge
        employment_type=employment if employment in LINKEDIN_EMPLOYMENT_TYPES else '',
        salary=salary if mentions_pay(salary) else '',
        closed=is_closed_text(card.get_text(' ') if card is not None else ''),
    )


def signed_out_criteria(soup: BeautifulSoup) -> dict[str, str]:
    '''The criteria list under the description: label -> value, as written.'''
    criteria = {}
    for item in soup.select(_CRITERIA_ITEM_SELECTOR):
        label, value = _text(item, _CRITERIA_LABEL_SELECTOR), _text(item, _CRITERIA_VALUE_SELECTOR)
        if label:
            criteria[label] = value
    return criteria


def signed_out_company_slug(soup: BeautifulSoup) -> str:
    link = soup.select_one(_COMPANY_LINK_SELECTOR)
    m = _COMPANY_SLUG_RE.search(str(link.get('href', ''))) if link is not None else None
    return m.group('slug') if m else ''


def reading_document(header: HeaderCard, criteria: dict[str, str], description: Tag) -> BeautifulSoup:
    '''What the extractor reads, in the signed-in page's order: company, title, location and the
    age on their own lines, then the description under LINKEDIN_JOB_DESCRIPTION_HEADING, then the
    criteria. Built from the parsed parts, so no page furniture reaches the model.'''
    doc = BeautifulSoup('<html><body><main></main></body></html>', 'lxml')
    main = doc.main
    for line in (header.company, header.title, header.location, header.posted, header.salary):
        if line:
            p = doc.new_tag('p')
            p.string = line
            main.append(p)
    heading = doc.new_tag('h2')
    heading.string = LINKEDIN_JOB_DESCRIPTION_HEADING
    main.append(heading)
    main.append(copy.copy(description))
    if criteria:
        heading = doc.new_tag('h2')
        heading.string = _CRITERIA_HEADING
        main.append(heading)
        ul = doc.new_tag('ul')
        for label, value in criteria.items():
            li = doc.new_tag('li')
            li.string = f'{label}: {value}'
            ul.append(li)
        main.append(ul)
    return doc


def _fetch_job(url: str, job_id: str) -> tuple[str, str]:
    resp = _get(url, f'job {job_id}')
    if _redirected_to_login(resp):
        raise LinkedInBlockedError(f'job {job_id}: GET {url} was redirected to {resp.url}: LinkedIn wants a '
                                   'signed-in visitor for a public job page')
    return resp.text, resp.url


async def read_job_page_signed_out(candidate: dict) -> JobPage:
    '''Fetch the public job page and return what the extractor reads, archived like the browser path.'''
    job_id = str(candidate.get('job_id') or '')
    url = str(candidate['url'])
    await pacer.wait(f'job page {job_id}')
    html, final_url = await asyncio.to_thread(_fetch_job, url, job_id)
    soup = BeautifulSoup(html, 'lxml')
    description = soup.select_one(_DESCRIPTION_SELECTOR)
    try:
        header = signed_out_header(soup)
        if description is None:
            raise SignedOutPageError(f'no {_DESCRIPTION_SELECTOR} on the page')
    except SignedOutPageError as ex:
        _raise_if_challenge(soup, url, f'job {job_id}', html)
        raise SignedOutPageError(f'job {job_id}: {url} ({final_url}, {len(html):,} chars) is not a readable public '
                             f'job page: {ex}') from ex
    if not header.workplace and (workplace := card_workplace(str(candidate.get('snippet') or ''))):
        header.workplace = workplace   # from the search card: the public page has no badge
    criteria = signed_out_criteria(soup)
    company_slug = signed_out_company_slug(soup)
    company = header.company or str(candidate.get('company') or '')
    doc = reading_document(header, criteria, description)

    strip_js(soup)
    html_path = raw_html_path(candidate)
    asset_counts = localize_assets(soup, None, html_path)   # no logo kept: every image blanked
    archive_meta = {'url': url, 'job_id': job_id, 'company': company, 'source': SOURCE_SIGNED_OUT,
                    'captured': datetime.now().isoformat(timespec='seconds'), 'dom_chars': len(html)}
    save_raw_html(candidate, soup, archive_meta)

    main = doc.main
    text = '\n'.join(line for line in (norm(s) for s in main.get_text('\n').split('\n')) if line)
    markdown = minimal_archive.to_markdown(main)
    tools_generic.save_raw_posting(candidate, markdown, SOURCE_SIGNED_OUT, len(markdown))
    await minimal_archive.save_minimal_archive(doc, html_path, archive_meta, [])
    logger.debug(f'Job page {job_id} (signed out): {len(html):,} chars → {len(text):,} chars of text; '
                 f'assets {asset_counts}')
    return JobPage(text=text, source=SOURCE_SIGNED_OUT, company_slug=company_slug, listed_location=header.location,
                   header=header, posting_company=company, dom_chars=len(html),
                   removed_chars=len(html) - len(text), markdown=markdown)


def parse_company_page_signed_out(soup: BeautifulSoup, slug: str) -> CompanyAbout:
    '''Name and Website from the public company page. Raises NotAnAboutPageError when it is not one.

    A company page is one with the "About us" section; its <h1> is the name.
    '''
    heading = soup.find('h1')
    name = norm(heading.get_text(' ')) if heading is not None else ''
    if not name or soup.select_one(_ABOUT_SELECTOR) is None:
        title = norm(soup.title.get_text(' ')) if soup.title is not None else ''
        raise NotAnAboutPageError(f'LinkedIn company {slug}: the public page has no company name or no '
                                  f'{_ABOUT_SELECTOR} section (title {title!r}, <h1> {name!r})')
    about = CompanyAbout(slug=slug, name=name)
    block = soup.select_one(_WEBSITE_SELECTOR)
    link = block.find('a', href=True) if block is not None else None
    if link is not None:
        target = outbound_target(str(link['href']))
        if target.startswith(('http://', 'https://')):
            about.website = target
    about.affiliated = affiliated_pages(soup)
    return about


def affiliated_pages(soup: BeautifulSoup) -> tuple[str, ...]:
    '''The names on "Affiliated pages", in page order, each once. The page does not say which is the parent.'''
    names: dict[str, None] = {}
    for link in soup.find_all('a', attrs={'data-tracking-control-name': _AFFILIATED_TRACKING_NAME}):
        lines = list(link.stripped_strings)
        if lines and (name := norm(lines[0])):
            names.setdefault(name, None)
    return tuple(names)


def outbound_target(href: str) -> str:
    '''The real target of a LinkedIn outbound redirect link; any other href unchanged.'''
    parsed = urlparse(href)
    if parsed.path.startswith(_REDIRECT_PATHS):
        return parse_qs(parsed.query).get(_REDIRECT_TARGET_KEY, [''])[0]
    return href


class ArchiveRefusedError(NoArchivedCopyError):
    '''The archive refused us (WAYBACK_REFUSAL_STATUSES): no further lookups this run.'''


# Set at the archive's first refusal; from then on every company page goes straight to LinkedIn.
_archive_refused: str = ''


def _raise_if_archive_refuses(resp: requests.Response, what: str) -> None:
    if resp.status_code in WAYBACK_REFUSAL_STATUSES:
        raise ArchiveRefusedError(f'{what}: GET {resp.url} returned HTTP {resp.status_code} {resp.reason}; '
                                  f'body: {snippet(resp.text)!r}')


def _archived_company_html(slug: str) -> tuple[str, str]:
    '''(html, snapshot url) of the newest Wayback copy of the company page. Raises NoArchivedCopyError.'''
    page = LINKEDIN_COMPANY_URL.format(slug=slug)
    what = f'archived company page {slug}'
    try:
        resp = requests.get(WAYBACK_AVAILABILITY_URL, params={'url': page}, headers=WAYBACK_HEADERS,
                            timeout=WAYBACK_FETCH_TIMEOUT_SECONDS)
    except requests.RequestException as ex:
        raise NoArchivedCopyError(f'{what}: GET {WAYBACK_AVAILABILITY_URL}?url={page} failed: '
                                  f'{type(ex).__name__}: {ex}') from ex
    _raise_if_archive_refuses(resp, what)
    if resp.status_code != HTTPStatus.OK:
        raise NoArchivedCopyError(f'{what}: GET {resp.url} returned HTTP {resp.status_code} {resp.reason}; '
                                  f'body: {snippet(resp.text)!r}')
    try:
        closest = (resp.json().get('archived_snapshots') or {}).get('closest') or {}
    except (ValueError, AttributeError) as ex:
        raise NoArchivedCopyError(f'{what}: {resp.url} did not return a JSON object ({type(ex).__name__}: {ex}); '
                                  f'body: {snippet(resp.text)!r}') from ex
    if not closest.get('available') or str(closest.get('status')) != str(int(HTTPStatus.OK)):
        raise NoArchivedCopyError(f'{what}: the archive has no copy ({snippet(closest)!r})')
    timestamp = str(closest.get('timestamp') or '')
    try:
        captured = datetime.strptime(timestamp, WAYBACK_TIMESTAMP_FORMAT)
    except ValueError as ex:
        raise NoArchivedCopyError(f'{what}: unreadable snapshot timestamp {timestamp!r} in {snippet(closest)!r}') from ex
    if datetime.now() - captured > timedelta(days=COMPANY_PAGE_ARCHIVE_MAX_AGE_DAYS):
        raise NoArchivedCopyError(f'{what}: the newest copy is from {captured:%Y-%m-%d}, older than '
                                  f'COMPANY_PAGE_ARCHIVE_MAX_AGE_DAYS')
    raw_url = WAYBACK_RAW_SNAPSHOT_URL.format(timestamp=timestamp, url=page)
    try:
        raw = requests.get(raw_url, headers=WAYBACK_HEADERS, timeout=WAYBACK_FETCH_TIMEOUT_SECONDS)
    except requests.RequestException as ex:
        raise NoArchivedCopyError(f'{what}: GET {raw_url} failed: {type(ex).__name__}: {ex}') from ex
    _raise_if_archive_refuses(raw, what)
    if raw.status_code != HTTPStatus.OK:
        raise NoArchivedCopyError(f'{what}: GET {raw_url} returned HTTP {raw.status_code} {raw.reason}; '
                                  f'body: {snippet(raw.text)!r}')
    return raw.text, raw_url


async def read_archived_company_page(slug: str) -> CompanyAbout:
    '''The company page as the Wayback Machine last captured it. Raises NoArchivedCopyError.'''
    global _archive_refused
    if _archive_refused:
        raise NoArchivedCopyError(f'archived company page {slug}: not asked, the archive refused us earlier '
                                  f'this run ({_archive_refused})')
    await archive_pacer.wait(f'archived company page {slug}')
    try:
        html, raw_url = await asyncio.to_thread(_archived_company_html, slug)
    except ArchiveRefusedError as ex:
        _archive_refused = str(ex)
        logger.warning(f'The archive refused a lookup; company pages are read from LinkedIn for the rest of '
                       f'the run: {ex}')
        raise
    try:
        about = parse_company_page_signed_out(BeautifulSoup(html, 'lxml'), slug)
    except NotAnAboutPageError as ex:
        # The archive keeps whatever LinkedIn served it, a login wall included.
        raise NoArchivedCopyError(f'archived company page {slug} ({raw_url}) is not a company page: {ex}') from ex
    about.page_url = raw_url
    return about


async def read_company_page_signed_out(slug: str) -> CompanyAbout:
    '''The public company page: its name, Website and affiliated pages (never a parent; see the module doc).

    A recent archived copy first (read_archived_company_page), LinkedIn itself otherwise.

    Raises LinkedInBlockedError on a challenge, NotAnAboutPageError when the page is not a company
    page (kept in COMPANY_ABOUT_FAILURES_DIR), SignedOutPageError on a transport or HTTP failure.
    '''
    try:
        about = await read_archived_company_page(slug)
    except NoArchivedCopyError as ex:
        logger.info(f'Company page {slug}: reading LinkedIn, no usable archived copy: {ex}')
    else:
        logger.debug(f'Company page {slug}: read from {about.page_url}')
        return about
    url = LINKEDIN_COMPANY_URL.format(slug=slug)
    await pacer.wait(f'company page {slug}')
    resp = await asyncio.to_thread(_get, url, f'company {slug}')
    if _redirected_to_login(resp):
        raise NotAnAboutPageError(f'LinkedIn company {slug}: {url} was redirected to {resp.url} (a page only '
                                  'signed-in members may see)')
    soup = BeautifulSoup(resp.text, 'lxml')
    try:
        return parse_company_page_signed_out(soup, slug)
    except NotAnAboutPageError as ex:
        _raise_if_challenge(soup, url, f'company {slug}', resp.text, keep_as=slug)
        raise NotAnAboutPageError(f'{ex}; {url}, {len(resp.text)} chars, '
                                  f'{_keep_unreadable_about_page(slug, resp.text)}') from ex

