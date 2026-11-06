'''Read a LinkedIn job page and a company page as a signed-out visitor: one plain HTTP GET each.

Why (2026-11-06): two runs in a row (11-04, 11-05) stopped within two minutes of Stage 2 starting,
on a Cloudflare "you have been blocked" page served to the logged-in browser on a company About
page. Both pages are public: a job page (`/jobs/view/<id>/`) and a company page (`/company/<slug>/`)
load without signing in, so reading them needs neither the browser nor the account. `--page-fetch
http` reads them this way; the default stays the logged-in browser for the job page.

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

A challenge stops the run exactly as on the browser path (LinkedInBlockedError): an HTTP status in
`LINKEDIN_GUEST_BLOCK_STATUSES`, a redirect to a login or checkpoint page for a job page, or a
UI_BLOCK_SIGNATURES phrase on a page without the expected content. No cookies are ever sent.
'''
import asyncio
import copy
import logging
from datetime import datetime
from http import HTTPStatus
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from agentic_job_search import minimal_archive, tools_generic
from agentic_job_search.config import (
    LINKEDIN_COMPANY_URL,
    LINKEDIN_EMPLOYMENT_TYPES,
    LINKEDIN_GUEST_BLOCK_STATUSES,
    LINKEDIN_GUEST_FETCH_TIMEOUT_SECONDS,
    LINKEDIN_GUEST_HEADERS,
    LINKEDIN_GUEST_LOGIN_PATH_PREFIXES,
    LINKEDIN_JOB_DESCRIPTION_HEADING,
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

SOURCE_GUEST = 'guest'   # JobPage.source on this path
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


class GuestPageError(RuntimeError):
    '''The public page could not be fetched or did not have the expected content. Not a challenge.'''


def _get(url: str, what: str) -> requests.Response:
    '''One cookie-free GET. Raises LinkedInBlockedError on a challenge status, GuestPageError otherwise.'''
    try:
        resp = requests.get(url, headers=LINKEDIN_GUEST_HEADERS, timeout=LINKEDIN_GUEST_FETCH_TIMEOUT_SECONDS)
    except requests.RequestException as ex:
        raise GuestPageError(f'{what}: GET {url} failed: {type(ex).__name__}: {ex}') from ex
    if resp.status_code in LINKEDIN_GUEST_BLOCK_STATUSES:
        raise LinkedInBlockedError(f'{what}: GET {url} returned HTTP {resp.status_code} {resp.reason}, LinkedIn '
                                   f'refusing a signed-out visitor; body: {snippet(resp.text)!r}')
    if resp.status_code != HTTPStatus.OK:
        raise GuestPageError(f'{what}: GET {url} returned HTTP {resp.status_code} {resp.reason}; '
                             f'body: {snippet(resp.text)!r}')
    return resp


def _redirected_to_login(resp: requests.Response) -> bool:
    return urlparse(resp.url).path.startswith(LINKEDIN_GUEST_LOGIN_PATH_PREFIXES)


def _raise_if_challenge(soup: BeautifulSoup, url: str, what: str, html: str, keep_as: str = '') -> None:
    '''A page without the expected content that shows a UI_BLOCK_SIGNATURES phrase is a challenge.'''
    phrase, context = visible_block_match(soup)
    if phrase:
        kept = _keep_unreadable_about_page(keep_as, html, suffix=BLOCKED_PAGE_SUFFIX) if keep_as else 'not kept'
        raise LinkedInBlockedError(f'{what}: {url} shows "{phrase}" in: {context!r}; page {kept}')


def _text(root: Tag, selector: str) -> str:
    el = root.select_one(selector)
    return norm(el.get_text(' ')) if el is not None else ''


def guest_header(soup: BeautifulSoup) -> HeaderCard:
    '''The public top card. Raises GuestPageError when it has no title (not a job page, or new markup).'''
    title = _text(soup, _TITLE_SELECTOR)
    if not title:
        raise GuestPageError(f'no {_TITLE_SELECTOR} on the page')
    criteria = guest_criteria(soup)
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


def guest_criteria(soup: BeautifulSoup) -> dict[str, str]:
    '''The criteria list under the description: label -> value, as written.'''
    criteria = {}
    for item in soup.select(_CRITERIA_ITEM_SELECTOR):
        label, value = _text(item, _CRITERIA_LABEL_SELECTOR), _text(item, _CRITERIA_VALUE_SELECTOR)
        if label:
            criteria[label] = value
    return criteria


def guest_company_slug(soup: BeautifulSoup) -> str:
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


async def read_job_page_guest(candidate: dict) -> JobPage:
    '''Fetch the public job page and return what the extractor reads, archived like the browser path.'''
    job_id = str(candidate.get('job_id') or '')
    url = str(candidate['url'])
    html, final_url = await asyncio.to_thread(_fetch_job, url, job_id)
    soup = BeautifulSoup(html, 'lxml')
    description = soup.select_one(_DESCRIPTION_SELECTOR)
    try:
        header = guest_header(soup)
        if description is None:
            raise GuestPageError(f'no {_DESCRIPTION_SELECTOR} on the page')
    except GuestPageError as ex:
        _raise_if_challenge(soup, url, f'job {job_id}', html)
        raise GuestPageError(f'job {job_id}: {url} ({final_url}, {len(html):,} chars) is not a readable public '
                             f'job page: {ex}') from ex
    criteria = guest_criteria(soup)
    company_slug = guest_company_slug(soup)
    company = header.company or str(candidate.get('company') or '')
    doc = reading_document(header, criteria, description)

    strip_js(soup)
    html_path = raw_html_path(candidate)
    asset_counts = localize_assets(soup, None, html_path)   # no logo kept: every image blanked
    archive_meta = {'url': url, 'job_id': job_id, 'company': company, 'source': SOURCE_GUEST,
                    'captured': datetime.now().isoformat(timespec='seconds'), 'dom_chars': len(html)}
    save_raw_html(candidate, soup, archive_meta)

    main = doc.main
    text = '\n'.join(line for line in (norm(s) for s in main.get_text('\n').split('\n')) if line)
    markdown = minimal_archive.to_markdown(main)
    tools_generic.save_raw_posting(candidate, markdown, SOURCE_GUEST, len(markdown))
    await minimal_archive.save_minimal_archive(doc, html_path, archive_meta, [])
    logger.debug(f'Job page {job_id} (signed out): {len(html):,} chars → {len(text):,} chars of text; '
                 f'assets {asset_counts}')
    return JobPage(text=text, source=SOURCE_GUEST, company_slug=company_slug, listed_location=header.location,
                   header=header, posting_company=company, dom_chars=len(html),
                   removed_chars=len(html) - len(text), markdown=markdown)


def parse_company_guest(soup: BeautifulSoup, slug: str) -> CompanyAbout:
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


async def read_company_guest(slug: str) -> CompanyAbout:
    '''Fetch the public company page: its name, Website and affiliated pages (never a parent; see the module doc).

    Raises LinkedInBlockedError on a challenge, NotAnAboutPageError when the page is not a company
    page (kept in COMPANY_ABOUT_FAILURES_DIR), GuestPageError on a transport or HTTP failure.
    '''
    url = LINKEDIN_COMPANY_URL.format(slug=slug)
    resp = await asyncio.to_thread(_get, url, f'company {slug}')
    if _redirected_to_login(resp):
        raise NotAnAboutPageError(f'LinkedIn company {slug}: {url} was redirected to {resp.url} (a page only '
                                  'signed-in members may see)')
    soup = BeautifulSoup(resp.text, 'lxml')
    try:
        return parse_company_guest(soup, slug)
    except NotAnAboutPageError as ex:
        _raise_if_challenge(soup, url, f'company {slug}', resp.text, keep_as=slug)
        raise NotAnAboutPageError(f'{ex}; {url}, {len(resp.text)} chars, '
                                  f'{_keep_unreadable_about_page(slug, resp.text)}') from ex
