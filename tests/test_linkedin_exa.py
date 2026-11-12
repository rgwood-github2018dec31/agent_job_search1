'''Read a LinkedIn job page and a company page through OpenRouter's web fetch tool, Exa engine.

The fixtures mirror what Exa returned on 2026-11-07 for a public job page and a company page.
'''
import json

import pytest
from agentic_job_search import company_domain, job_trace, linkedin_page
from agentic_job_search import linkedin_exa as lx
from agentic_job_search.config import (
    LINKEDIN_JOB_DESCRIPTION_HEADING,
    LLM_OPENROUTER_MCP_URL,
    MODEL_NAME_PAGE_FETCH,
    PAGE_FETCH_ALLOWED_DOMAINS,
    PAGE_FETCH_ENGINE,
)
from agentic_job_search.linkedin_signed_out import NoArchivedCopyError

JOB_URL = 'https://www.linkedin.com/jobs/view/4484344124/'
CANDIDATE = {'site': 'linkedin', 'job_id': '4484344124', 'url': JOB_URL, 'company': 'Acme Data',
             'snippet': 'Rzeszów (Remote)'}
JOB_TITLE = 'Senior Data Scientist – AdTech (f/m/x) at Acme Data — Rzeszów, Podkarpackie, Poland | LinkedIn Jobs'
JOB_CONTENT = '''Senior Data Scientist – AdTech (f/m/x) at Acme Data — Rzeszów, Podkarpackie, Poland | LinkedIn Jobs



# Senior Data Scientist – AdTech (f/m/x)

 Acme Data Rzeszów, Podkarpackie, Poland

 4 days ago

 Be among the first 25 applicants

 Do you want to train ML models that run at a billion-scale? Your tasks

- Measuring existing model performance
- Owning the design of the new methodology

Requirements

- Minimum 5 years of experience in training ML models
- Advanced level of English

- ### Seniority level

 Mid-Senior level
- ### Employment type

 Full-time
- ### Industries

 IT Services and IT Consulting

## Page Links

- https://pl.linkedin.com/company/acme-data?trk=public_jobs_topcard_logo
- https://pl.linkedin.com/company/acme-data?trk=public_jobs_topcard-org-name
- https://www.linkedin.com/company/other-co?trk=public_jobs_similar-jobs
'''
COMPANY_CONTENT = '''# Acme Data

Acme Data is a IT Services and IT Consulting company. Acme employs 6,280 people.

## About

Acme is a technology consulting partner.

## Company Details
- Industry: IT Services and IT Consulting
- Homepage: acmedata.example
- Aliases: Acme Sp. z o. o, acme
- LinkedIn: linkedin.com/company/acme-data

## Page Links

- https://acmedata.example
'''


def reply(content: str, title: str = JOB_TITLE, **overrides) -> str:
    return json.dumps({'ok': True, 'url': JOB_URL, 'title': title, 'content': content, 'http_status': 200,
                       'engine': 'exa', 'model': MODEL_NAME_PAGE_FETCH, 'cost_usd': 0.0012, **overrides})


@pytest.fixture
def fetched(monkeypatch):
    '''call_mcp_tool answering web_fetch with the reply each test sets; records every call.'''
    calls, replies = [], {}

    async def call(url, tool, args, timeout_seconds=None):
        calls.append((url, tool, args))
        return replies[args['url']]

    monkeypatch.setattr(lx, 'call_mcp_tool', call)
    return replies, calls


async def test_a_job_page_is_read_into_the_signed_in_shape(fetched):
    replies, calls = fetched
    replies[JOB_URL] = reply(JOB_CONTENT)
    stats = {'cost': 0.0}
    page = await lx.read_job_page_exa(CANDIDATE, stats)
    assert page.source == lx.SOURCE_EXA
    assert page.header.title == 'Senior Data Scientist – AdTech (f/m/x)' and page.header.company == 'Acme Data'
    assert page.header.location == 'Rzeszów, Podkarpackie, Poland' and page.header.posted == '4 days ago'
    assert page.header.employment_type == 'Full-time'
    assert page.header.workplace == 'remote'   # from the search card
    assert page.company_slug == 'acme-data'    # the page's own company link, not a similar job's
    lines = page.markdown.split('\n')
    assert lines[:4] == ['Acme Data', 'Senior Data Scientist – AdTech (f/m/x)', 'Rzeszów, Podkarpackie, Poland',
                         '4 days ago']
    assert any(line.endswith(LINKEDIN_JOB_DESCRIPTION_HEADING) for line in lines)
    assert 'Minimum 5 years of experience' in page.model_text
    assert 'Page Links' not in page.model_text and 'trk=' not in page.model_text
    assert stats['cost'] == pytest.approx(0.0012)
    assert calls == [(LLM_OPENROUTER_MCP_URL, 'web_fetch', {
        'url': JOB_URL, 'model': MODEL_NAME_PAGE_FETCH, 'allowed_domains': list(PAGE_FETCH_ALLOWED_DOMAINS),
        'engine': PAGE_FETCH_ENGINE})]


async def test_a_prefetched_fetch_is_parsed_without_fetching_again_and_traced(fetched, tmp_path):
    '''Read Job Pages Ahead: the fetch was made during Stage 1b, with no job trace; Stage 2 records it.'''
    replies, calls = fetched
    replies[JOB_URL] = reply(JOB_CONTENT)
    prefetch_stats = {'cost': 0.0}
    fetch = await lx.fetch_job_page(CANDIDATE, prefetch_stats)
    assert prefetch_stats['cost'] == pytest.approx(0.0012) and fetch.attempts == 1
    stage2_stats = {'cost': 0.0}
    trace_path = tmp_path / 'linkedin-4484344124.jsonl'
    _trace, token = job_trace.start(trace_path, 'linkedin', '4484344124')
    try:
        page = await lx.read_job_page_exa(CANDIDATE, stage2_stats, fetch)
    finally:
        job_trace.finish(token)
    assert len(calls) == 1, 'fetched once, during the read-ahead'
    assert stage2_stats['cost'] == 0.0, 'the cost was counted when it was fetched'
    assert page.header.company == 'Acme Data'
    [step] = [s for s in job_trace.load(trace_path) if s['kind'] == 'web_fetch']
    assert step['prefetched'] is True and step['attempts'] == 1 and step['url'] == JOB_URL


async def test_the_fetch_is_kept_as_exa_returned_it(fetched):
    replies, _ = fetched
    replies[JOB_URL] = reply(JOB_CONTENT)
    await lx.read_job_page_exa(CANDIDATE)
    html_path = linkedin_page.raw_html_path(CANDIDATE)
    assert html_path.with_suffix('.exa.md').read_text(encoding='utf-8') == JOB_CONTENT
    assert html_path.with_suffix('.txt').exists()


async def test_a_title_that_is_not_a_job_page_is_an_error_with_the_content(fetched):
    replies, _ = fetched
    replies[JOB_URL] = reply('Sign in to view this page', title='LinkedIn Login, Sign in | LinkedIn')
    with pytest.raises(lx.ExaPageError, match=r'not a readable job page.*Sign in to view'):
        await lx.read_job_page_exa(CANDIDATE)


async def test_a_failed_fetch_names_the_request_and_carries_the_reply(fetched):
    replies, _ = fetched
    replies[JOB_URL] = json.dumps({'ok': False, 'error': 'the fetched page answered HTTP 404'})
    with pytest.raises(lx.ExaPageError, match=r'job 4484344124: web_fetch .*4484344124.*HTTP 404'):
        await lx.read_job_page_exa(CANDIDATE)


async def test_a_non_json_reply_is_an_error(fetched):
    replies, _ = fetched
    replies[JOB_URL] = 'Internal Server Error'
    with pytest.raises(lx.ExaPageError, match=r'non-JSON.*Internal Server Error'):
        await lx.read_job_page_exa(CANDIDATE)


def test_a_closed_posting_is_seen_in_the_header():
    content = JOB_CONTENT.replace('Be among the first 25 applicants', 'No longer accepting applications')
    assert lx.job_header(JOB_TITLE, content, 'Acme Data').closed


def test_a_company_name_with_at_in_it_is_split_on_the_card_company():
    title = 'Data Lead at Work at Home Co — Canada | LinkedIn Jobs'
    header = lx.job_header(title, '# Data Lead\n', 'Work at Home Co')
    assert (header.title, header.company, header.location) == ('Data Lead', 'Work at Home Co', 'Canada')


@pytest.fixture
def archived(monkeypatch):
    '''read_archived_company_page answering from `copies` (slug -> CompanyAbout); no copy otherwise.'''
    copies, asked = {}, []

    async def read(slug):
        asked.append(slug)
        if slug not in copies:
            raise NoArchivedCopyError(f'archived company page {slug}: the archive has no copy')
        return copies[slug]

    monkeypatch.setattr(lx, 'read_archived_company_page', read)
    return copies, asked


async def test_a_company_profile_gives_the_name_and_website(fetched, archived):
    replies, _ = fetched
    url = 'https://www.linkedin.com/company/acme-data/'
    replies[url] = reply(COMPANY_CONTENT, title='Acme Data', url=url)
    stats = {'cost': 0.0}
    about = await lx.read_company_page_exa('acme-data', stats)
    assert (about.name, about.website, about.affiliated) == ('Acme Data', 'acmedata.example', ())
    assert about.source == company_domain.SOURCE_EXA   # Exa's profile, not LinkedIn's page
    assert stats['cost'] == pytest.approx(0.0012)
    assert archived[1] == ['acme-data']   # the archive was asked first


async def test_an_archived_linkedin_page_is_read_before_exa(fetched, archived):
    """2026-11-12: the archived copy is LinkedIn's own page, with the affiliated pages Exa's profile lacks."""
    _, calls = fetched
    copies, _ = archived
    raw = 'https://web.archive.org/web/20261101000000id_/https://www.linkedin.com/company/acme-data/'
    copies['acme-data'] = linkedin_page.CompanyAbout(
        slug='acme-data', name='Acme Data', website='acmedata.example', affiliated=('Acme Holdings',), page_url=raw)
    about = await lx.read_company_page_exa('acme-data', {'cost': 0.0})
    assert (about.affiliated, about.page_url, about.source) == (('Acme Holdings',), raw, company_domain.SOURCE_LINKEDIN)
    assert calls == []   # Exa was not asked, and nothing was billed


async def test_a_profile_of_another_company_is_not_an_about_page(fetched, archived):
    replies, _ = fetched
    url = 'https://www.linkedin.com/company/acme/'
    replies[url] = reply(COMPANY_CONTENT, title='Acme Data', url=url)
    with pytest.raises(linkedin_page.NotAnAboutPageError, match="profile is of 'acme-data'"):
        await lx.read_company_page_exa('acme', None)


async def test_a_profile_without_a_name_is_not_an_about_page(fetched, archived):
    replies, _ = fetched
    url = 'https://www.linkedin.com/company/acme-data/'
    replies[url] = reply('Page not found', title='', url=url)
    with pytest.raises(linkedin_page.NotAnAboutPageError, match='no "# <name>" heading'):
        await lx.read_company_page_exa('acme-data', None)


@pytest.fixture
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(lx, 'PAGE_FETCH_RETRY_DELAY_SECONDS', 0)


async def test_no_content_from_exa_is_asked_again(monkeypatch, no_retry_delay):
    """2026-11-07: Exa answers 'no content' now and then and serves the page on a later try."""
    answers = [json.dumps({'ok': False, 'error': "the fetch did not complete (status='incomplete', "
                                                 "error='Exa returned no content for this URL.')"}),
               reply(JOB_CONTENT)]
    calls = []

    async def call(url, tool, args, timeout_seconds=None):
        calls.append(args['url'])
        return answers.pop(0)

    monkeypatch.setattr(lx, 'call_mcp_tool', call)
    page = await lx.read_job_page_exa(CANDIDATE)
    assert page.header.company == 'Acme Data' and len(calls) == 2


async def test_no_content_on_every_attempt_fails_the_read(monkeypatch, no_retry_delay):
    calls = []

    async def call(url, tool, args, timeout_seconds=None):
        calls.append(args['url'])
        return json.dumps({'ok': False, 'error': "status='incomplete', error='Exa returned no content'"})

    monkeypatch.setattr(lx, 'call_mcp_tool', call)
    with pytest.raises(lx.ExaPageError, match=f'attempt {lx.PAGE_FETCH_ATTEMPTS} of {lx.PAGE_FETCH_ATTEMPTS}'):
        await lx.read_job_page_exa(CANDIDATE)
    assert len(calls) == lx.PAGE_FETCH_ATTEMPTS


async def test_any_other_failure_is_not_asked_again(monkeypatch, no_retry_delay):
    calls = []

    async def call(url, tool, args, timeout_seconds=None):
        calls.append(args['url'])
        return json.dumps({'ok': False, 'error': 'the fetched page answered HTTP 404'})

    monkeypatch.setattr(lx, 'call_mcp_tool', call)
    with pytest.raises(lx.ExaPageError, match='HTTP 404'):
        await lx.read_job_page_exa(CANDIDATE)
    assert len(calls) == 1
