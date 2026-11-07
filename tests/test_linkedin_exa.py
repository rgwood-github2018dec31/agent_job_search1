'''Read a LinkedIn job page and a company page through OpenRouter's web fetch tool, Exa engine.

The fixtures mirror what Exa returned on 2026-11-07 for a public job page and a company page.
'''
import json

import pytest
from agentic_job_search import linkedin_exa as lx
from agentic_job_search import linkedin_page
from agentic_job_search.config import (
    LINKEDIN_JOB_DESCRIPTION_HEADING,
    LLM_OPENROUTER_MCP_URL,
    MODEL_NAME_PAGE_FETCH,
    PAGE_FETCH_ALLOWED_DOMAINS,
    PAGE_FETCH_ENGINE,
)

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


async def test_a_company_profile_gives_the_name_and_website(fetched):
    replies, _ = fetched
    url = 'https://www.linkedin.com/company/acme-data/'
    replies[url] = reply(COMPANY_CONTENT, title='Acme Data', url=url)
    stats = {'cost': 0.0}
    about = await lx.read_company_page_exa('acme-data', stats)
    assert (about.name, about.website, about.affiliated) == ('Acme Data', 'acmedata.example', ())
    assert stats['cost'] == pytest.approx(0.0012)


async def test_a_profile_of_another_company_is_not_an_about_page(fetched):
    replies, _ = fetched
    url = 'https://www.linkedin.com/company/acme/'
    replies[url] = reply(COMPANY_CONTENT, title='Acme Data', url=url)
    with pytest.raises(linkedin_page.NotAnAboutPageError, match="profile is of 'acme-data'"):
        await lx.read_company_page_exa('acme', None)


async def test_a_profile_without_a_name_is_not_an_about_page(fetched):
    replies, _ = fetched
    url = 'https://www.linkedin.com/company/acme-data/'
    replies[url] = reply('Page not found', title='', url=url)
    with pytest.raises(linkedin_page.NotAnAboutPageError, match='no "# <name>" heading'):
        await lx.read_company_page_exa('acme-data', None)
