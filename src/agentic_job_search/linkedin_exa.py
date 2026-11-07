'''Read a LinkedIn job page and a company page through OpenRouter's web fetch tool, Exa engine.

Why (2026-11-07): our own signed-out GETs (linkedin_signed_out) were refused with a 429 after about
27 requests unpaced and about 53 paced at one a minute, on 2026-11-06. Exa fetches the public page
from its own infrastructure, so nothing reaches LinkedIn from this machine and no LinkedIn limit
applies to us. The default (`--job-and-company-pages-via`, LINKEDIN_PAGE_READER_DEFAULT).

The fetch goes through the OpenRouter tool server's `web_fetch` (allowed_domains PAGE_FETCH_ALLOWED_DOMAINS),
which returns the engine's content verbatim. A model issues the fetch call, because OpenRouter has no
fetch-only call; it predicts nothing about the page and its reply is discarded, so this call is exempt
from Verify Source Reference. Every model reads only the fetched text, as on the other paths.

What Exa returns, measured 2026-11-07:
- A job page is Markdown of LinkedIn's public job page, and its `title` is "<job title> at <company> —
  <location> | LinkedIn Jobs". The body has the heading, the company and location on one line, the
  posting age, the description, the criteria ("### Employment type") and a "## Page Links" list,
  which holds the company page link. Like the signed-out page, it has no workplace badge, so the
  workplace is taken from the search card ("Warsaw (Remote)").
- A company page is Exa's own company profile, not LinkedIn's page: "# <name>", then sections such
  as "## Company Details" with "- Homepage: <domain>" and "- LinkedIn: linkedin.com/company/<slug>".
  It has no "Affiliated pages", so CompanyAbout.affiliated stays empty on this path.
'''
import asyncio
import json
import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from utils_tools_n_agents_common.mcp_client import call_mcp_tool

from agentic_job_search import job_trace, minimal_archive, tools_generic
from agentic_job_search.config import (
    LINKEDIN_COMPANY_URL,
    LINKEDIN_EMPLOYMENT_TYPES,
    LINKEDIN_JOB_DESCRIPTION_HEADING,
    LLM_OPENROUTER_MCP_URL,
    MODEL_NAME_PAGE_FETCH,
    PAGE_FETCH_ALLOWED_DOMAINS,
    PAGE_FETCH_ATTEMPTS,
    PAGE_FETCH_ENGINE,
    PAGE_FETCH_RETRY_DELAY_SECONDS,
    PAGE_FETCH_RETRYABLE_MARKER,
    PAGE_FETCH_TIMEOUT_SECONDS,
)
from agentic_job_search.linkedin_page import (
    _COMPANY_SLUG_RE,
    CompanyAbout,
    HeaderCard,
    JobPage,
    NotAnAboutPageError,
    _keep_unreadable_about_page,
    is_closed_text,
    norm,
    raw_html_path,
)
from agentic_job_search.linkedin_signed_out import card_workplace
from agentic_job_search.text_budget import quote_for_error

logger = logging.getLogger(__name__)

SOURCE_EXA = 'exa_via_openrouter'   # JobPage.source on this path

_JOB_TITLE_SUFFIX = ' | LinkedIn Jobs'
_TITLE_LOCATION_SEPARATOR = ' — '
_TITLE_COMPANY_SEPARATOR = ' at '
_PAGE_LINKS_HEADING = '## Page Links'
_POSTED_RE = re.compile(r'^(?:Reposted\s+)?\d+\s+\w+\s+ago$', re.IGNORECASE)
_CRITERIA_HEADING_RE = re.compile(r'^(?:-\s*)?###\s+(?P<label>.+?)\s*$')
_EMPLOYMENT_TYPE_LABEL = 'Employment type'
# The page's own link to the posting company, preferred over the logo's and any other company link.
_COMPANY_LINK_MARKERS = ('public_jobs_topcard-org-name', 'public_jobs_topcard_logo')
# The lines of the header the closed-posting phrases are looked for in, after the job title heading.
_HEADER_LINES = 8
_COMPANY_HOMEPAGE_RE = re.compile(r'^-\s*Homepage:\s*(?P<site>\S+)\s*$', re.MULTILINE)
_COMPANY_LINKEDIN_RE = re.compile(r'^-\s*LinkedIn:\s*(?P<url>\S+)\s*$', re.MULTILINE)
_COMPANY_NAME_RE = re.compile(r'^#\s+(?P<name>.+?)\s*$', re.MULTILINE)


class ExaPageError(RuntimeError):
    '''The fetch failed, or what came back does not read as the page asked for.'''


async def fetch_page(url: str, what: str, stage_stats: dict | None) -> dict:
    '''The tool server's web_fetch reply for `url` (ok, content, title, http_status, cost_usd).

    Raises ExaPageError naming the request, with the whole reply, on any failure. A failure to reach
    the tool server propagates as call_mcp_tool raised it, so tool_server.outage_url still sees it.
    '''
    args = {'url': url, 'model': MODEL_NAME_PAGE_FETCH, 'allowed_domains': list(PAGE_FETCH_ALLOWED_DOMAINS),
            'engine': PAGE_FETCH_ENGINE}
    for attempt in range(1, PAGE_FETCH_ATTEMPTS + 1):
        raw = await call_mcp_tool(LLM_OPENROUTER_MCP_URL, 'web_fetch', args, timeout_seconds=PAGE_FETCH_TIMEOUT_SECONDS)
        try:
            data = json.loads(raw)
        except ValueError as ex:
            raise ExaPageError(f'{what}: web_fetch {url} via {MODEL_NAME_PAGE_FETCH}/{PAGE_FETCH_ENGINE} replied with '
                               f'non-JSON ({ex}): {quote_for_error(raw, "web_fetch reply")!r}') from ex
        if isinstance(data, dict) and data.get('ok') is True:
            break
        error = str(data.get('error') if isinstance(data, dict) else raw)
        if PAGE_FETCH_RETRYABLE_MARKER in error and attempt < PAGE_FETCH_ATTEMPTS:
            logger.info(f'{what}: Exa had no content for {url} (attempt {attempt} of {PAGE_FETCH_ATTEMPTS}), '
                        f'asking again in {PAGE_FETCH_RETRY_DELAY_SECONDS}s')
            await asyncio.sleep(PAGE_FETCH_RETRY_DELAY_SECONDS)
            continue
        raise ExaPageError(f'{what}: web_fetch {url} via {MODEL_NAME_PAGE_FETCH}/{PAGE_FETCH_ENGINE} failed '
                           f'(attempt {attempt} of {PAGE_FETCH_ATTEMPTS}): '
                           f'{quote_for_error(raw, "web_fetch failure reply")!r}')
    content = data.get('content')
    if not isinstance(content, str) or not content.strip():
        raise ExaPageError(f'{what}: web_fetch {url} returned no content: {quote_for_error(raw, "web_fetch reply")!r}')
    cost = data.get('cost_usd')
    if cost is None:
        logger.warning(f'{what}: web_fetch {url} reported no cost (counted as $0)')
    if stage_stats is not None:
        stage_stats['cost'] = stage_stats.get('cost', 0.0) + float(cost or 0.0)
    job_trace.record('web_fetch', url=url, what=what, engine=PAGE_FETCH_ENGINE, model=MODEL_NAME_PAGE_FETCH,
                     served_model=data.get('model'), http_status=data.get('http_status'),
                     title=data.get('title'), chars=len(content), cost_usd=cost)
    return data


def without_page_links(content: str) -> str:
    '''The content before Exa's "## Page Links" list: the links are not the page's text.'''
    head, _sep, _links = content.partition(f'\n{_PAGE_LINKS_HEADING}')
    return head


def tidy_lines(markdown: str) -> str:
    '''Each line stripped and runs of blank lines folded to one, the text otherwise as Exa wrote it.'''
    lines, blank = [], False
    for line in (raw.strip() for raw in markdown.splitlines()):
        if not line:
            if lines and not blank:
                lines.append('')
            blank = True
            continue
        lines.append(line)
        blank = False
    return '\n'.join(lines).strip()


def job_header(title_field: str, content: str, card_company: str) -> HeaderCard:
    '''The header from Exa's `title` ("<title> at <company> — <location> | LinkedIn Jobs") and the body.

    Raises ExaPageError when the title is not a LinkedIn job page's: a login wall, an expired page, or
    some other page Exa returned instead.
    '''
    if not title_field.endswith(_JOB_TITLE_SUFFIX):
        raise ExaPageError(f'the fetched title {title_field!r} is not a LinkedIn job page (no {_JOB_TITLE_SUFFIX!r})')
    rest = title_field[:-len(_JOB_TITLE_SUFFIX)]
    left, sep, location = rest.rpartition(_TITLE_LOCATION_SEPARATOR)
    if not sep:
        left, location = rest, ''
    company_suffix = f'{_TITLE_COMPANY_SEPARATOR}{card_company}'
    if card_company and left.endswith(company_suffix):
        title, company = left[:-len(company_suffix)], card_company
    else:
        title, sep, company = left.rpartition(_TITLE_COMPANY_SEPARATOR)
        if not sep:
            title, company = left, ''
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    heading_at = next((i for i, line in enumerate(lines) if line.startswith('# ')), 0)
    header_lines = lines[heading_at + 1:heading_at + 1 + _HEADER_LINES]
    criteria = job_criteria(content)
    employment = criteria.get(_EMPLOYMENT_TYPE_LABEL, '')
    return HeaderCard(
        company=norm(company), title=norm(title), location=norm(location),
        posted=next((line for line in header_lines if _POSTED_RE.match(line)), ''),
        workplace='',   # the public page has no workplace badge
        employment_type=employment if employment in LINKEDIN_EMPLOYMENT_TYPES else '',
        closed=is_closed_text(' '.join(header_lines)),
    )


def job_criteria(content: str) -> dict[str, str]:
    '''The criteria list ("- ### Seniority level" then its value on the next non-empty line): label -> value.'''
    criteria, label = {}, ''
    for line in (raw.strip() for raw in content.splitlines()):
        if not line:
            continue
        if m := _CRITERIA_HEADING_RE.match(line):
            label = m.group('label')
            continue
        if label:
            criteria[label] = line
            label = ''
    return criteria


def job_company_slug(content: str) -> str:
    '''The posting company's LinkedIn slug, from the page's own company link in "## Page Links".'''
    links = [line for line in content.partition(f'\n{_PAGE_LINKS_HEADING}')[2].splitlines() if 'company/' in line]
    for marker in _COMPANY_LINK_MARKERS:
        for line in links:
            if marker in line and (m := _COMPANY_SLUG_RE.search(line)):
                return m.group('slug')
    return ''


def reading_document(header: HeaderCard, body_markdown: str) -> BeautifulSoup:
    '''What the extractor reads, in the signed-in page's order: company, title, location and the age on
    their own lines, then the page's body under LINKEDIN_JOB_DESCRIPTION_HEADING, rendered from Exa's Markdown.'''
    doc = BeautifulSoup('<html><body><main></main></body></html>', 'lxml')
    main = doc.main
    for line in (header.company, header.title, header.location, header.posted):
        if line:
            p = doc.new_tag('p')
            p.string = line
            main.append(p)
    heading = doc.new_tag('h2')
    heading.string = LINKEDIN_JOB_DESCRIPTION_HEADING
    main.append(heading)
    body = BeautifulSoup(MarkdownIt().render(body_markdown), 'lxml')
    for element in list((body.body or body).children):
        main.append(element)
    return doc


def body_after_heading(content: str) -> str:
    '''The page's body after its "# <job title>" heading, without the Page Links list.'''
    text = without_page_links(content)
    _before, sep, after = text.partition('\n# ')
    if not sep:
        return text
    return after.partition('\n')[2]


async def read_job_page_exa(candidate: dict, stage_stats: dict | None = None) -> JobPage:
    '''Fetch the public job page through Exa and return what the extractor reads, archived like the other paths.'''
    job_id = str(candidate.get('job_id') or '')
    url = str(candidate['url'])
    data = await fetch_page(url, f'job {job_id}', stage_stats)
    content = str(data['content'])
    try:
        header = job_header(str(data.get('title') or ''), content, str(candidate.get('company') or ''))
    except ExaPageError as ex:
        raise ExaPageError(f'job {job_id}: {url} ({len(content):,} chars) is not a readable job page: {ex}; '
                           f'content: {quote_for_error(content, "Exa job page content")!r}') from ex
    if (workplace := card_workplace(str(candidate.get('snippet') or ''))):
        header.workplace = workplace   # from the search card: the public page has no badge
    company = header.company or str(candidate.get('company') or '')
    doc = reading_document(header, tidy_lines(body_after_heading(content)))

    html_path = raw_html_path(candidate)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fetched_path = html_path.with_suffix('.exa.md')
    fetched_path.write_text(content, encoding='utf-8')   # the fetch as Exa returned it, Page Links included
    archive_meta = {'url': url, 'job_id': job_id, 'company': company, 'source': SOURCE_EXA,
                    'captured': datetime.now().isoformat(timespec='seconds'), 'dom_chars': len(content),
                    'fetched': fetched_path.name}
    main = doc.main
    text = '\n'.join(line for line in (norm(s) for s in main.get_text('\n').split('\n')) if line)
    markdown = minimal_archive.to_markdown(main)
    tools_generic.save_raw_posting(candidate, markdown, SOURCE_EXA, len(markdown))
    await minimal_archive.save_minimal_archive(doc, html_path, archive_meta, [])
    logger.debug(f'Job page {job_id} (Exa): {len(content):,} chars fetched → {len(text):,} chars of text')
    return JobPage(text=text, source=SOURCE_EXA, company_slug=job_company_slug(content),
                   listed_location=header.location, header=header, posting_company=company,
                   dom_chars=len(content), removed_chars=len(content) - len(text), markdown=markdown)


def parse_company_profile(content: str, slug: str) -> CompanyAbout:
    '''Exa's company profile: name ("# <name>"), Website ("- Homepage:"), checked against `slug`.

    Raises NotAnAboutPageError when it has no name, or names another company's LinkedIn page.
    '''
    name = _COMPANY_NAME_RE.search(content)
    if name is None:
        raise NotAnAboutPageError(f'LinkedIn company {slug}: the fetched profile has no "# <name>" heading')
    if (linked := _COMPANY_LINKEDIN_RE.search(content)) is not None:
        m = _COMPANY_SLUG_RE.search(linked.group('url'))
        if m is not None and m.group('slug').casefold() != slug.casefold():
            raise NotAnAboutPageError(f'LinkedIn company {slug}: the fetched profile is of {m.group("slug")!r} '
                                      f'({linked.group("url")})')
    homepage = _COMPANY_HOMEPAGE_RE.search(content)
    return CompanyAbout(slug=slug, name=norm(name.group('name')), website=homepage.group('site') if homepage else '',
                        page_url=LINKEDIN_COMPANY_URL.format(slug=slug))


async def read_company_page_exa(slug: str, stage_stats: dict | None = None) -> CompanyAbout:
    '''The company's name and Website from Exa's profile of its LinkedIn page (no affiliated pages; see the module doc).

    Raises NotAnAboutPageError when the profile does not read as this company's (kept in
    COMPANY_ABOUT_FAILURES_DIR), ExaPageError when the fetch failed.
    '''
    url = LINKEDIN_COMPANY_URL.format(slug=slug)
    data = await fetch_page(url, f'company {slug}', stage_stats)
    content = str(data['content'])
    try:
        return parse_company_profile(content, slug)
    except NotAnAboutPageError as ex:
        raise NotAnAboutPageError(f'{ex}; {url}, {len(content)} chars, '
                                  f'{_keep_unreadable_about_page(slug, content)}') from ex
