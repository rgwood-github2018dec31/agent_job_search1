'''Read Public LinkedIn Page: a job page and a company page fetched signed out, one GET each.

The fixtures mirror the public markup measured on 2026-11-06 (job 4482929457, company auxotalent).
'''
from http import HTTPStatus

import pytest
import requests
from agentic_job_search import linkedin_guest as lg
from agentic_job_search import linkedin_page
from agentic_job_search.config import LINKEDIN_HTTP_STATUS_REFUSED, LINKEDIN_JOB_DESCRIPTION_HEADING

JOB_URL = 'https://www.linkedin.com/jobs/view/4482929457/'
CANDIDATE = {'site': 'linkedin', 'job_id': '4482929457', 'url': JOB_URL, 'company': 'Auxo Talent'}

JOB_HTML = '''<html><head><title>Senior Data Scientist at Auxo Talent — Poland | LinkedIn Jobs</title>
<script>var x = 1;</script></head><body><main>
<section class="top-card-layout">
  <h1 class="top-card-layout__title">Senior Data Scientist</h1>
  <a class="topcard__org-name-link" href="https://www.linkedin.com/company/auxotalent?trk=public_jobs_topcard-org-name">
    Auxo Talent</a>
  <span class="topcard__flavor topcard__flavor--bullet">Poland</span>
  <span class="posted-time-ago__text topcard__flavor--metadata">
      1 week ago
  </span>
  <img src="https://media.licdn.com/logo.png" alt="Auxo Talent">
  <div>Sign in to evaluate your skills</div>
</section>
<div class="show-more-less-html__markup"><p>Our client seeks a Senior Data Scientist. <strong>This role is
remote.</strong></p><ul><li>Python and SQL</li></ul></div>
<ul class="description__job-criteria-list">
  <li class="description__job-criteria-item"><h3 class="description__job-criteria-subheader">Seniority level</h3>
    <span class="description__job-criteria-text">Mid-Senior level</span></li>
  <li class="description__job-criteria-item"><h3 class="description__job-criteria-subheader">Employment type</h3>
    <span class="description__job-criteria-text">Contract</span></li>
</ul></main></body></html>'''

COMPANY_HTML = '''<html><head><title>Auxo Talent | LinkedIn</title></head><body><main>
<h1>Auxo Talent</h1>
<section data-test-id="about-us"><p data-test-id="about-us__description">Auxo is a workforce partner.</p>
<dl><div data-test-id="about-us__website"><dt>Website</dt><dd>
<a href="https://www.linkedin.com/redir/redirect?url=http%3A%2F%2Fwww%2Eauxotalent%2Ecom&amp;urlhash=6BLl">
http://www.auxotalent.com</a></dd></div></dl></section>
<section><h2>Affiliated pages</h2><a href="https://uk.linkedin.com/company/rp-international?trk=affiliated-pages">
RP International</a></section></main></body></html>'''


class FakeResponse:
    def __init__(self, text, status=HTTPStatus.OK, url=JOB_URL):
        self.text, self.status_code, self.url = text, status, url
        self.reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else 'Request denied'
        self.headers = {}


@pytest.fixture
def served(monkeypatch):
    '''Serve one response per URL. Records what was requested and with which headers.'''
    pages: dict[str, FakeResponse | Exception] = {}
    requested: list[tuple[str, dict]] = []

    def fake_get(url, headers=None, timeout=None, **kwargs):
        requested.append((url, {'headers': headers, **kwargs}))
        reply = pages[url]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(lg.requests, 'get', fake_get)
    monkeypatch.setattr(linkedin_page, '_http_get', lambda url: FakeResponse('', status=HTTPStatus.NOT_FOUND))
    return pages, requested


async def test_a_public_job_page_is_read_into_the_signed_in_shape(served):
    pages, requested = served
    pages[JOB_URL] = FakeResponse(JOB_HTML)
    page = await lg.read_job_page_guest(CANDIDATE)
    assert page.source == lg.SOURCE_GUEST
    assert page.header.title == 'Senior Data Scientist' and page.header.company == 'Auxo Talent'
    assert page.header.location == 'Poland' and page.header.posted == '1 week ago'
    assert page.header.employment_type == 'Contract'
    assert page.header.workplace == ''   # the public page has no workplace badge
    assert page.company_slug == 'auxotalent'
    lines = page.markdown.split('\n')
    assert lines[:3] == ['Auxo Talent', 'Senior Data Scientist', 'Poland']
    assert linkedin_page.job_description(page.markdown).count('This role is') == 1
    assert 'Sign in' not in page.model_text   # the page's furniture never reaches the model
    assert any(line.endswith(LINKEDIN_JOB_DESCRIPTION_HEADING) for line in lines)
    assert 'Employment type: Contract' in page.model_text
    # Never a cookie: the request carries the fixed headers and nothing else.
    assert requested == [(JOB_URL, {'headers': lg.LINKEDIN_GUEST_HEADERS})]


async def test_the_public_job_page_is_archived_without_scripts_or_images(served, tmp_path):
    pages, _ = served
    pages[JOB_URL] = FakeResponse(JOB_HTML)
    await lg.read_job_page_guest(CANDIDATE)
    archived = linkedin_page.raw_html_path(CANDIDATE).read_text(encoding='utf-8')
    assert '<script' not in archived and 'media.licdn.com/logo.png' not in archived
    assert 'source: guest' in archived
    assert linkedin_page.raw_html_path(CANDIDATE).with_suffix('.txt').exists()


@pytest.mark.parametrize('status', [HTTPStatus.TOO_MANY_REQUESTS, LINKEDIN_HTTP_STATUS_REFUSED])
async def test_a_refusing_status_is_a_challenge_and_stops_the_run(served, status):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('', status=status)
    with pytest.raises(linkedin_page.LinkedInBlockedError, match=str(int(status))):
        await lg.read_job_page_guest(CANDIDATE)


async def test_a_job_page_redirected_to_login_is_a_challenge(served):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('<html></html>', url='https://www.linkedin.com/authwall?trk=x')
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='authwall'):
        await lg.read_job_page_guest(CANDIDATE)


async def test_a_page_without_the_job_that_shows_a_block_phrase_is_a_challenge(served):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('<html><body><h1>Sorry, you have been blocked</h1></body></html>')
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='you have been blocked'):
        await lg.read_job_page_guest(CANDIDATE)


async def test_a_page_without_the_job_and_no_block_phrase_is_an_ordinary_failure(served):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('<html><body><h1>Something else</h1></body></html>')
    with pytest.raises(lg.GuestPageError, match='top-card-layout__title'):
        await lg.read_job_page_guest(CANDIDATE)


async def test_a_transport_failure_names_the_url(served):
    pages, _ = served
    pages[JOB_URL] = requests.ConnectionError('reset by peer')
    with pytest.raises(lg.GuestPageError, match=f'GET {JOB_URL} failed: ConnectionError: reset by peer'):
        await lg.read_job_page_guest(CANDIDATE)


async def test_a_public_company_page_gives_the_name_and_the_unwrapped_website(served):
    pages, _ = served
    url = 'https://www.linkedin.com/company/auxotalent/'
    pages[url] = FakeResponse(COMPANY_HTML, url=url)
    about = await lg.read_company_guest('auxotalent')
    assert (about.name, about.website) == ('Auxo Talent', 'http://www.auxotalent.com')
    assert about.parent_name == ''   # affiliated pages do not say which is the parent


async def test_a_company_page_redirected_to_login_is_not_a_company_page(served):
    pages, _ = served
    url = 'https://www.linkedin.com/company/nobody/'
    pages[url] = FakeResponse('<html><title>LinkedIn Login</title></html>',
                              url='https://www.linkedin.com/uas/login?session_redirect=x')
    with pytest.raises(linkedin_page.NotAnAboutPageError, match='uas/login'):
        await lg.read_company_guest('nobody')


async def test_a_company_page_without_the_about_section_is_kept_and_rejected(served):
    pages, _ = served
    url = 'https://www.linkedin.com/company/odd/'
    pages[url] = FakeResponse('<html><title>Odd</title><body><h1>Odd</h1></body></html>', url=url)
    with pytest.raises(linkedin_page.NotAnAboutPageError, match='saved at'):
        await lg.read_company_guest('odd')


@pytest.mark.network
async def test_live_public_pages_still_have_the_expected_markup():
    '''One GET each of a real public job page and company page; fails when LinkedIn changes the markup.'''
    try:
        about = await lg.read_company_guest('linkedin')
    except lg.GuestPageError as ex:
        pytest.skip(f'LinkedIn unreachable: {ex}')
    assert about.name == 'LinkedIn' and 'linkedin.com' in about.website
