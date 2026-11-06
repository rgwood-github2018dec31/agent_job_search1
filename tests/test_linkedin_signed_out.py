'''Read LinkedIn Page Signed Out: a job page and a company page fetched signed out, one GET each.

The fixtures mirror the public markup measured on 2026-11-06 (job 4482929457, company auxotalent).
'''
import asyncio
import json
from datetime import datetime, timedelta
from http import HTTPStatus

import pytest
import requests
from agentic_job_search import linkedin_page
from agentic_job_search import linkedin_signed_out as lg
from agentic_job_search.config import (
    COMPANY_PAGE_ARCHIVE_MAX_AGE_DAYS,
    LINKEDIN_HTTP_STATUS_REFUSED,
    LINKEDIN_JOB_DESCRIPTION_HEADING,
    WAYBACK_AVAILABILITY_URL,
    WAYBACK_TIMESTAMP_FORMAT,
)

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
<section><h2>Affiliated pages</h2>
<a href="https://uk.linkedin.com/company/rp-international?trk=affiliated-pages" data-tracking-control-name="affiliated-pages">
<h3>RP International</h3><p>Staffing and Recruiting</p><p>London, London</p></a>
<a href="https://uk.linkedin.com/company/jenrick-e?trk=affiliated-pages" data-tracking-control-name="affiliated-pages">
<h3>Jenrick</h3><p>Staffing and Recruiting</p></a>
<button data-tracking-control-name="affiliated-pages_show_more">Show more affiliated pages</button></section>
<section><h2>Similar pages</h2><a href="https://www.linkedin.com/company/monzo?trk=similar-pages"
data-tracking-control-name="similar-pages"><h3>Monzo</h3></a></section></main></body></html>'''


class FakeResponse:
    def __init__(self, text, status=HTTPStatus.OK, url=JOB_URL):
        self.text, self.status_code, self.url = text, status, url
        self.reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else 'Request denied'
        self.headers = {}

    def json(self):
        return json.loads(self.text)


NO_ARCHIVED_COPY = FakeResponse(json.dumps({'archived_snapshots': {}}), url=WAYBACK_AVAILABILITY_URL)


def archived(slug, captured: datetime):
    '''An availability answer naming a copy of the company page, and the raw copy's URL.'''
    stamp = captured.strftime(WAYBACK_TIMESTAMP_FORMAT)
    page = f'https://www.linkedin.com/company/{slug}/'
    reply = {'archived_snapshots': {'closest': {'available': True, 'status': '200', 'timestamp': stamp,
                                                'url': f'http://web.archive.org/web/{stamp}/{page}'}}}
    return FakeResponse(json.dumps(reply), url=WAYBACK_AVAILABILITY_URL), f'https://web.archive.org/web/{stamp}id_/{page}'


@pytest.fixture
def served(monkeypatch):
    '''Serve one response per URL. Records what was requested and with which headers.'''
    pages: dict[str, FakeResponse | Exception] = {WAYBACK_AVAILABILITY_URL: NO_ARCHIVED_COPY}
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
    page = await lg.read_job_page_signed_out(CANDIDATE)
    assert page.source == lg.SOURCE_SIGNED_OUT
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
    assert requested == [(JOB_URL, {'headers': lg.LINKEDIN_SIGNED_OUT_HEADERS})]


async def test_the_public_job_page_is_archived_without_scripts_or_images(served, tmp_path):
    pages, _ = served
    pages[JOB_URL] = FakeResponse(JOB_HTML)
    await lg.read_job_page_signed_out(CANDIDATE)
    archived = linkedin_page.raw_html_path(CANDIDATE).read_text(encoding='utf-8')
    assert '<script' not in archived and 'media.licdn.com/logo.png' not in archived
    assert f'source: {lg.SOURCE_SIGNED_OUT}' in archived
    assert linkedin_page.raw_html_path(CANDIDATE).with_suffix('.txt').exists()


@pytest.mark.parametrize('status', [HTTPStatus.TOO_MANY_REQUESTS, LINKEDIN_HTTP_STATUS_REFUSED])
async def test_a_refusing_status_is_a_challenge_and_stops_the_run(served, status):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('', status=status)
    with pytest.raises(linkedin_page.LinkedInBlockedError, match=str(int(status))):
        await lg.read_job_page_signed_out(CANDIDATE)


async def test_a_job_page_redirected_to_login_is_a_challenge(served):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('<html></html>', url='https://www.linkedin.com/authwall?trk=x')
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='authwall'):
        await lg.read_job_page_signed_out(CANDIDATE)


async def test_a_page_without_the_job_that_shows_a_block_phrase_is_a_challenge(served):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('<html><body><h1>Sorry, you have been blocked</h1></body></html>')
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='you have been blocked'):
        await lg.read_job_page_signed_out(CANDIDATE)


async def test_a_page_without_the_job_and_no_block_phrase_is_an_ordinary_failure(served):
    pages, _ = served
    pages[JOB_URL] = FakeResponse('<html><body><h1>Something else</h1></body></html>')
    with pytest.raises(lg.SignedOutPageError, match='top-card-layout__title'):
        await lg.read_job_page_signed_out(CANDIDATE)


async def test_a_transport_failure_names_the_url(served):
    pages, _ = served
    pages[JOB_URL] = requests.ConnectionError('reset by peer')
    with pytest.raises(lg.SignedOutPageError, match=f'GET {JOB_URL} failed: ConnectionError: reset by peer'):
        await lg.read_job_page_signed_out(CANDIDATE)


async def test_a_public_company_page_gives_the_name_and_the_unwrapped_website(served):
    pages, _ = served
    url = 'https://www.linkedin.com/company/auxotalent/'
    pages[url] = FakeResponse(COMPANY_HTML, url=url)
    about = await lg.read_company_page_signed_out('auxotalent')
    assert (about.name, about.website) == ('Auxo Talent', 'http://www.auxotalent.com')
    assert about.parent_name == ''   # affiliated pages do not say which is the parent
    assert about.affiliated == ('RP International', 'Jenrick')   # never the similar pages


async def test_a_recent_archived_copy_is_read_and_linkedin_is_not_asked(served):
    pages, requested = served
    availability, raw = archived('auxotalent', datetime.now() - timedelta(days=30))
    pages[WAYBACK_AVAILABILITY_URL] = availability
    pages[raw] = FakeResponse(COMPANY_HTML, url=raw)
    about = await lg.read_company_page_signed_out('auxotalent')
    assert (about.name, about.website, about.page_url) == ('Auxo Talent', 'http://www.auxotalent.com', raw)
    assert not any('linkedin.com/company' in url and 'archive' not in url for url, _ in requested)


@pytest.mark.parametrize('copy', ['too old', 'a login wall', 'archive refuses'])
async def test_without_a_usable_archived_copy_linkedin_is_read(served, copy):
    pages, requested = served
    age = COMPANY_PAGE_ARCHIVE_MAX_AGE_DAYS + 1 if copy == 'too old' else 1
    availability, raw = archived('auxotalent', datetime.now() - timedelta(days=age))
    pages[WAYBACK_AVAILABILITY_URL] = (FakeResponse('', status=HTTPStatus.TOO_MANY_REQUESTS)
                                       if copy == 'archive refuses' else availability)
    pages[raw] = FakeResponse('<html><title>LinkedIn Login</title><body><h1>Welcome back</h1></body></html>', url=raw)
    live = 'https://www.linkedin.com/company/auxotalent/'
    pages[live] = FakeResponse(COMPANY_HTML, url=live)
    about = await lg.read_company_page_signed_out('auxotalent')
    assert about.name == 'Auxo Talent' and about.page_url == ''
    assert requested[-1][0] == live


async def test_after_the_archive_refuses_it_is_not_asked_again_this_run(served):
    pages, requested = served
    pages[WAYBACK_AVAILABILITY_URL] = FakeResponse('flagged as bot traffic', status=HTTPStatus.TOO_MANY_REQUESTS)
    for slug in ('auxotalent', 'other'):
        live = f'https://www.linkedin.com/company/{slug}/'
        pages[live] = FakeResponse(COMPANY_HTML, url=live)
        await lg.read_company_page_signed_out(slug)
    assert [url for url, _ in requested].count(WAYBACK_AVAILABILITY_URL) == 1
    assert requested[0][1]['headers'] == lg.WAYBACK_HEADERS   # the archive is asked as the program it is


async def test_a_company_page_redirected_to_login_is_not_a_company_page(served):
    pages, _ = served
    url = 'https://www.linkedin.com/company/nobody/'
    pages[url] = FakeResponse('<html><title>LinkedIn Login</title></html>',
                              url='https://www.linkedin.com/uas/login?session_redirect=x')
    with pytest.raises(linkedin_page.NotAnAboutPageError, match='uas/login'):
        await lg.read_company_page_signed_out('nobody')


async def test_a_company_page_without_the_about_section_is_kept_and_rejected(served):
    pages, _ = served
    url = 'https://www.linkedin.com/company/odd/'
    pages[url] = FakeResponse('<html><title>Odd</title><body><h1>Odd</h1></body></html>', url=url)
    with pytest.raises(linkedin_page.NotAnAboutPageError, match='saved at'):
        await lg.read_company_page_signed_out('odd')


@pytest.mark.network
async def test_live_public_pages_still_have_the_expected_markup():
    '''One GET each of a real public job page and company page; fails when LinkedIn changes the markup.'''
    try:
        about = await lg.read_company_page_signed_out('linkedin')
    except lg.SignedOutPageError as ex:
        pytest.skip(f'LinkedIn unreachable: {ex}')
    assert about.name == 'LinkedIn' and 'linkedin.com' in about.website


# --- Pace and read-ahead (2026-11-06: a 429 at ~27 requests in 15 minutes) -----------------------

@pytest.mark.parametrize('card, workplace', [
    ('Warsaw (Remote)', 'remote'), ('Porto, Portugal (Hybrid)', 'hybrid'), ('Ghent (On-site)', 'onsite'),
    ('Lisbon Metropolitan Area', ''), ('', ''),
])
def test_the_search_card_gives_the_workplace_the_public_page_lacks(card, workplace):
    assert lg.card_workplace(card) == workplace


async def test_a_signed_out_job_page_takes_its_workplace_from_the_search_card(served):
    pages, _ = served
    pages[JOB_URL] = FakeResponse(JOB_HTML)
    page = await lg.read_job_page_signed_out({**CANDIDATE, 'snippet': 'Poland (Remote)'})
    assert page.header.workplace == 'remote'


async def test_requests_are_spaced_by_the_drawn_interval(monkeypatch):
    clock = {'now': 1000.0}
    slept = []

    async def fake_sleep(seconds):
        slept.append(round(seconds, 3))
        clock['now'] += seconds

    monkeypatch.setattr(lg.time, 'monotonic', lambda: clock['now'])
    monkeypatch.setattr(lg.asyncio, 'sleep', fake_sleep)
    monkeypatch.setattr(lg.random, 'uniform', lambda low, high: low + 2.5)
    pacer = lg.SignedOutPacer(lambda: (61.0, 66.0))
    await pacer.wait('first')     # no wait before the first request
    clock['now'] += 10            # the request and its processing take 10 s
    await pacer.wait('second')
    assert slept == [53.5]        # 61 + 2.5 drawn, less the 10 s already gone


def _candidate(job_id):
    return {'site': 'linkedin', 'job_id': job_id, 'url': f'https://www.linkedin.com/jobs/view/{job_id}/'}


@pytest.fixture
def fake_reads(monkeypatch):
    '''read_job_page_signed_out stubbed: records the order of reads; `fail[job_id]` raises instead.'''
    order: list[str] = []
    fail: dict[str, Exception] = {}

    async def read(candidate):
        job_id = candidate['job_id']
        order.append(job_id)
        await asyncio.sleep(0)
        if job_id in fail:
            raise fail[job_id]
        return linkedin_page.JobPage(text=f'page {job_id}', source=lg.SOURCE_SIGNED_OUT)

    monkeypatch.setattr(lg, 'read_job_page_signed_out', read)
    monkeypatch.setattr(lg, 'SIGNED_OUT_PREFETCH_POLL_SECONDS', 0)
    return order, fail


async def test_pages_are_read_ahead_in_queue_order_as_jobs_are_queued(fake_reads):
    order, _ = fake_reads
    queued = [_candidate('1')]
    prefetcher = lg.JobPagePrefetcher(lambda: queued)
    prefetcher.start()
    assert (await prefetcher.page(queued[0])).text == 'page 1'
    queued.append(_candidate('2'))   # queued later, while Stage 1b searches on
    assert (await prefetcher.page(queued[1])).text == 'page 2'
    await prefetcher.stop()
    assert order == ['1', '2']


async def test_a_failed_read_is_raised_for_its_job_only(fake_reads):
    _order, fail = fake_reads
    fail['1'] = lg.SignedOutPageError('no title')
    queued = [_candidate('1'), _candidate('2')]
    prefetcher = lg.JobPagePrefetcher(lambda: queued)
    prefetcher.start()
    with pytest.raises(lg.SignedOutPageError, match='no title'):
        await prefetcher.page(queued[0])
    assert (await prefetcher.page(queued[1])).text == 'page 2'
    await prefetcher.stop()


async def test_a_challenge_stops_all_reading_and_is_raised_for_every_later_job(fake_reads):
    order, fail = fake_reads
    fail['1'] = linkedin_page.LinkedInBlockedError('HTTP 429')
    queued = [_candidate('1'), _candidate('2'), _candidate('3')]
    prefetcher = lg.JobPagePrefetcher(lambda: queued)
    prefetcher.start()
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='429'):
        await prefetcher.page(queued[0])
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='stopped earlier'):
        await prefetcher.page(queued[2])
    await prefetcher.stop()
    assert order == ['1']   # nothing was requested after the challenge
