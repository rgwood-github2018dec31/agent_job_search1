"""linkedin_page: capture, archive and de-clutter a LinkedIn job page.

The fixture mirrors the structure measured on three real pages (2026-09-24): hashed class names
(never used), stable `JobDetails_<Section>_<jobId>` ids, a top card around the company logo, and
the clutter sections the user listed. It also carries two sections nobody has classified: one
harmless, one that mentions pay.
"""
import json
import logging

import pytest
import yaml
from agentic_job_search import linkedin_page as lp
from agentic_job_search import tools_generic as tools
from agentic_job_search.config import (
    LINKEDIN_PAGE_CAPTURE_JS,
    LINKEDIN_REMOVE_SECTION_PHRASES,
    PAGE_SECTION_SIGNATURE_MAX_WORDS,
)
from agentic_job_search.scrape_openrouter import _guard_call
from bs4 import BeautifulSoup

JOB_ID = '4415518989'
LOGO_URL = 'https://media.licdn.com/dms/image/logo-acme.png'
CSS_URL = 'https://static.licdn.com/aero-v1/sc/h/site.css'

PAGE = f"""<!DOCTYPE html><html><head>
<link rel="stylesheet" href="{CSS_URL}"><link rel="modulepreload" href="https://static.licdn.com/m.js">
<script>window.track()</script></head>
<body><header><a href="/feed">Home</a> My Network</header>
<main id="workspace"><div class="_1274a681">
  <div class="cf8121e6"><div class="d256456b">
    <img src="{LOGO_URL}" alt="Company logo for, Acme Corp.">
    <p>Acme Corp</p><p>Staff AI Engineer</p><p>Toronto, ON · 2 days ago · 40 applicants</p>
    <button onclick="apply()">Apply</button><a href="javascript:void(0)">Save</a>
  </div>
  <div class="f3b32fab"><h2>Use AI to assess how you fit</h2><p>Activate Premium for CA$0</p></div>
  </div>
  <div class="_0fbca0ec"><h2>People you can reach out to</h2><img src="https://media.licdn.com/p/jane.jpg" alt="View Jane Doe’s profile"></div>
  <div id="JobDetails_AboutTheJob_{JOB_ID}"><div><h2>About the job</h2>
    <p>Build agentic systems in Python.</p><p>Salary: $150,000 - $180,000 per year.</p>
    <span data-testid="expandable-text-button">… more</span></div></div>
  <div class="db1ec949"><h2>Set alert for similar jobs</h2><p>Staff AI Engineer, Toronto</p></div>
  <div class="_398a5a3d"><p>New from LinkedIn: see who else viewed this job</p></div>
  <div class="_398a5a3d"><p>Salary insights for this role</p><p>Typical pay: $140K/yr - $190K/yr</p></div>
  <div id="JobDetails_AboutTheCompany_{JOB_ID}"><div><div>
    <h2>About the company</h2>
    <div><div><img src="{LOGO_URL}" alt=""><p>Acme Corp</p><p>12,345 followers</p><button>Follow</button></div></div>
    <div><p>Software Development</p><p>51-200 employees</p></div>
    <div data-testid="expandable-text-box">Acme builds reliable agents for banks.</div>
    <a aria-label="Show more about the company" href="/company/acme">Show more</a>
    <div><h3>Commitments</h3><p>Environmental sustainability</p></div>
    <div><h2>Interested in working with us in the future?</h2><button>I’m interested</button></div>
    <div><p>Trending employee content</p><div data-testid="carousel"><img src="https://media.licdn.com/post.jpg" alt="View image"><p>Our charity walk!</p></div></div>
  </div></div></div>
  <div class="d71e55ac"><h2>More jobs</h2><p>Other Co — Principal Engineer — $300K/yr</p><img src="https://media.licdn.com/other.png" alt=""></div>
</div></main>
<footer><p>Looking for talent?</p><p>Nederlands (Dutch)</p></footer></body></html>"""

HARMLESS_TEXT = 'New from LinkedIn: see who else viewed this job'
PAY_SECTION = 'Salary insights for this role'


def _sig(text: str) -> str:
    return ' '.join(text.split()[:PAGE_SECTION_SIGNATURE_MAX_WORDS])


HARMLESS = _sig(HARMLESS_TEXT)


def _soup() -> BeautifulSoup:
    return BeautifulSoup(PAGE, 'lxml')


def _cleaned() -> BeautifulSoup:
    soup = _soup()
    lp.strip_js(soup)
    lp.remove_known_clutter(soup)
    return soup


def _candidate(**overrides) -> dict:
    candidate = {'site': 'linkedin', 'job_id': JOB_ID, 'url': f'https://www.linkedin.com/jobs/view/{JOB_ID}/',
                 'company': 'Acme Corp', 'title': 'Staff AI Engineer', 'date_posted': '', 'snippet': ''}
    candidate.update(overrides)
    return candidate


# --- read-only capture -----------------------------------------------------------------------------

def test_capture_js_only_reads_the_page():
    """JavaScript may read the page, never drive it (Account safety)."""
    assert _guard_call('browser_evaluate', {'function': LINKEDIN_PAGE_CAPTURE_JS}) is None
    for driving in ('.click(', 'dispatchEvent', '.remove(', 'innerHTML =', 'outerHTML =', 'appendChild', 'setAttribute'):
        assert driving not in LINKEDIN_PAGE_CAPTURE_JS


# --- JS-free archive -----------------------------------------------------------------------------

def test_strip_js_removes_every_script_handler_and_js_url():
    soup = _soup()
    counts = lp.strip_js(soup)
    html = str(soup)
    assert '<script' not in html and 'modulepreload' not in html
    assert 'onclick' not in html and 'javascript:' not in html
    assert counts['scripts'] == 1 and counts['handler_attrs'] == 1 and counts['js_urls'] == 1


def test_only_the_company_logo_is_kept_and_it_is_fetched_once(monkeypatch, tmp_path):
    fetched = []

    class Response:
        status_code = 200

        def __init__(self, url):
            self.content = f'bytes of {url}'.encode()
            self.headers = {'content-type': 'text/css' if url.endswith('.css') else 'image/png'}

    def fake_get(url):
        fetched.append(url)
        return Response(url)

    monkeypatch.setattr(lp, '_http_get', fake_get)
    html_path = tmp_path / 'raw_postings' / '2026-09-24' / f'linkedin-{JOB_ID}.html'

    first = _soup()
    counts = lp.localize_assets(first, 'Acme Corp', html_path)
    assert sorted(fetched) == sorted([LOGO_URL, CSS_URL]), 'no profile photo, post image or other logo is fetched'
    assert counts['logos_kept'] == 2 and counts['images_blanked'] == 3
    logo_files = [p for p in (lp.COMPANY_ASSETS_DIR / 'Acme Corp').iterdir() if p.suffix == '.png']
    assert len(logo_files) == 1, 'both logo slots share one stored file, under the name as written'
    blanked = [img for img in first.find_all('img') if img.get('data-removed') == 'image']
    assert {img.get('alt') for img in blanked} == {'View Jane Doe’s profile', 'View image', ''}
    assert all(img['src'] == '' for img in blanked)
    assert first.find('link', rel='stylesheet')['href'].startswith(lp.STYLESHEET_DIR_NAME + '/')

    lp.localize_assets(_soup(), 'Acme Corp', html_path)
    assert len(fetched) == 2, 'a stored logo and stylesheet are reused, not fetched again'


def test_a_failed_logo_fetch_blanks_the_image_and_warns(monkeypatch, tmp_path, caplog):
    def down(url):
        raise lp.requests.ConnectionError('offline')

    monkeypatch.setattr(lp, '_http_get', down)
    soup = _soup()
    with caplog.at_level(logging.WARNING):
        counts = lp.localize_assets(soup, 'Acme Corp', tmp_path / 'x.html')
    assert counts['logos_kept'] == 0 and counts['fetch_failures'] == 3
    assert all(img['src'] == '' for img in soup.find_all('img'))
    assert any('Company logo' in r.message and LOGO_URL in r.message for r in caplog.records)


def test_company_dir_name_keeps_proper_case():
    assert lp.company_dir_name('Experis Canada') == 'Experis Canada'
    assert lp.company_dir_name('A/B: Labs') == 'A_B_ Labs'


# --- known clutter ---------------------------------------------------------------------------------

def test_known_clutter_is_removed_and_the_facts_survive():
    text = lp.page_text(_cleaned())
    for fact in ('Acme Corp', 'Staff AI Engineer', 'Toronto, ON', 'Build agentic systems in Python.',
                 'Salary: $150,000 - $180,000 per year.', 'About the company', '12,345 followers',
                 'Software Development', 'Acme builds reliable agents for banks.'):
        assert fact in text, fact
    for clutter in ('Home', 'My Network', 'Use AI to assess', 'People you can reach out to', 'Set alert',
                    'Interested in working with us', 'Trending employee content', 'charity walk',
                    'More jobs', 'Other Co', '$300K', 'Show more', 'Looking for talent', 'Nederlands'):
        assert clutter not in text, clutter


def test_every_listed_phrase_is_matched_as_written():
    """A phrase that stops matching fails silently permissive — the section just comes back."""
    soup = _soup()
    lp.strip_js(soup)
    main_html = str(soup.find('main'))  # the footer's phrase goes out with the footer
    removed = {r['what'] for r in lp.remove_known_clutter(soup)}
    on_page = [p for p in LINKEDIN_REMOVE_SECTION_PHRASES if p in main_html]
    assert set(on_page) <= removed


def test_a_removal_rule_never_takes_the_description_or_the_company_card():
    soup = _soup()
    about_job = soup.find(id=f'JobDetails_AboutTheJob_{JOB_ID}')
    about_job.append(BeautifulSoup('<p>More jobs like this one will be posted soon.</p>', 'lxml').p)
    lp.remove_known_clutter(soup)
    assert soup.find(id=f'JobDetails_AboutTheJob_{JOB_ID}') is not None
    assert 'Acme builds reliable agents' in lp.page_text(soup)


# --- unknown blocks --------------------------------------------------------------------------------

def test_only_unclassified_sections_are_unknown():
    blocks = lp.unknown_blocks(_cleaned().find('main'))
    assert [lp.block_signature(b) for b in blocks] == [HARMLESS, PAY_SECTION, 'Commitments']


def test_with_no_known_list_every_clutter_section_is_detected(monkeypatch):
    """What makes automatic removal possible: a section LinkedIn adds shows up here unasked."""
    soup = _soup()
    lp.strip_js(soup)
    for tag in soup.find_all(['header', 'footer', 'nav']):
        tag.decompose()
    signatures = [lp.block_signature(b) for b in lp.unknown_blocks(soup.find('main'))]
    for section in ('Use AI to assess how you fit', 'People you can reach out to', 'Set alert for similar jobs',
                    'Interested in working with us in the future?', 'Trending employee content', 'More jobs'):
        assert any(sig.startswith(section[:20]) for sig in signatures), section


def test_signatures_ignore_numbers_so_one_verdict_covers_every_job():
    soup = BeautifulSoup('<div><p>12 people viewed this job</p></div>', 'lxml')
    assert lp.block_signature(soup.div) == 'N people viewed this job'


def _classifier(monkeypatch, decisions: dict, calls: list):
    async def fake_chat(prompt, model=None, **kwargs):
        calls.append(prompt)
        numbered = [line.split('signature: ', 1)[1] for line in prompt.splitlines() if 'signature: ' in line]
        verdicts = [{'block': i, 'remove': decisions[sig], 'reason': 'test'} for i, sig in enumerate(numbered, 1)]
        return json.dumps({'decisions': verdicts}), 0.001

    monkeypatch.setattr(lp, 'chat_openrouter', fake_chat)


async def test_new_sections_are_judged_once_then_cached(monkeypatch, caplog):
    calls = []
    _classifier(monkeypatch, {HARMLESS: True, PAY_SECTION: True, 'Commitments': False}, calls)
    stats = {'cost': 0.0}
    soup = _cleaned()
    with caplog.at_level(logging.WARNING):
        removed = await lp.decide_unknown_blocks(soup, _candidate(), stats)

    text = lp.page_text(soup)
    assert HARMLESS_TEXT not in text
    assert PAY_SECTION in text, "a section holding pay wording is never removed on the model's say-so"
    assert 'Commitments' in text
    assert [r['what'] for r in removed] == [f'section {HARMLESS!r}']
    assert stats['cost'] == pytest.approx(0.001)
    saved = yaml.safe_load(lp.PAGE_SECTIONS_PATH.read_text())
    assert saved[HARMLESS]['decision'] == 'remove' and saved['Commitments']['decision'] == 'keep'
    assert saved[HARMLESS]['first_seen_job'] == JOB_ID
    assert any('New LinkedIn page section' in r.message for r in caplog.records)
    assert any('holds pay wording' in r.message for r in caplog.records)

    again = _cleaned()
    await lp.decide_unknown_blocks(again, _candidate(job_id='2'), stats)
    assert len(calls) == 1, 'a signature is judged once, ever'
    assert HARMLESS_TEXT not in lp.page_text(again)


async def test_a_hand_edited_verdict_wins(monkeypatch):
    lp.PAGE_SECTIONS_PATH.write_text(yaml.safe_dump({
        HARMLESS: {'decision': 'keep'}, PAY_SECTION: {'decision': 'keep'}, 'Commitments': {'decision': 'remove'}}))
    soup = _cleaned()
    await lp.decide_unknown_blocks(soup, _candidate(), None)  # the autouse stub fails any LLM call
    text = lp.page_text(soup)
    assert HARMLESS_TEXT in text and 'Commitments' not in text


async def test_a_classifier_failure_keeps_everything_and_caches_nothing(monkeypatch, caplog):
    async def down(*args, **kwargs):
        raise RuntimeError('OpenRouter chat failed: 503')

    monkeypatch.setattr(lp, 'chat_openrouter', down)
    soup = _cleaned()
    with caplog.at_level(logging.WARNING):
        removed = await lp.decide_unknown_blocks(soup, _candidate(), None)
    assert removed == []
    assert HARMLESS_TEXT in lp.page_text(soup)
    assert not lp.PAGE_SECTIONS_PATH.exists()
    assert any('Page-section classification failed' in r.message and '503' in r.message for r in caplog.records)


async def test_an_unreadable_decisions_file_is_left_alone(monkeypatch, caplog):
    lp.PAGE_SECTIONS_PATH.write_text('not: [valid')
    soup = _cleaned()
    with caplog.at_level(logging.WARNING):
        removed = await lp.decide_unknown_blocks(soup, _candidate(), None)  # an LLM call would fail the test
    assert removed == []
    assert lp.PAGE_SECTIONS_PATH.read_text() == 'not: [valid'
    assert any('unreadable' in r.message for r in caplog.records)


# --- read_job_page -------------------------------------------------------------------------------

def _browser(page_html: str | None, snapshot: str = 'snapshot text', chars_delta: int = 0):
    calls = []

    async def call(tool, args):
        calls.append(tool)
        if tool == 'browser_evaluate':
            payload = {'url': 'u', 'title': 't', 'html': page_html,
                       'chars': len(page_html.encode('utf-16-le')) // 2 + chars_delta}
            return '### Result\n' + json.dumps(payload) + '\n### Ran Playwright code\n() => { return {}; }'
        if tool == 'browser_snapshot':
            return snapshot
        return 'ok'

    return call, calls


async def test_read_job_page_archives_the_page_then_returns_clean_text(monkeypatch):
    monkeypatch.setattr(lp, '_http_get', lambda url: (_ for _ in ()).throw(lp.requests.ConnectionError('offline')))
    _classifier(monkeypatch, {HARMLESS: True, PAY_SECTION: False, 'Commitments': False}, [])
    call, calls = _browser(PAGE)

    page = await lp.read_job_page(_candidate(), call, {'cost': 0.0})

    assert page.source == 'dom'
    assert calls == ['browser_navigate', 'browser_wait_for', 'browser_evaluate'], 'no click, no snapshot'
    assert 'Build agentic systems in Python.' in page.text and 'More jobs' not in page.text
    html = lp.raw_html_path(_candidate()).read_text()
    assert html.startswith('<!--') and '<script' not in html
    assert 'More jobs' in html, "the archive is the whole page; only the model's copy is cleaned"
    retained = tools.raw_posting_path(_candidate()).read_text()
    assert 'source: dom' in retained and 'Salary: $150,000 - $180,000' in retained, \
        'salary provenance checks the text the model read'


async def test_read_job_page_waits_again_then_falls_back_when_the_description_never_renders(caplog):
    no_description = PAGE.replace(f'id="JobDetails_AboutTheJob_{JOB_ID}"', 'id="somethingElse"')
    call, calls = _browser(no_description, snapshot='- heading "About the job" [level=2]')
    with caplog.at_level(logging.WARNING):
        page = await lp.read_job_page(_candidate(), call)
    assert page.source == 'snapshot' and 'About the job' in page.text
    assert calls.count('browser_evaluate') == 2 and calls[-1] == 'browser_snapshot'
    assert any('reading the accessibility snapshot instead' in r.message for r in caplog.records)


async def test_read_job_page_falls_back_when_the_capture_was_altered_in_transit(caplog):
    call, _calls = _browser(PAGE, chars_delta=7)
    with caplog.at_level(logging.WARNING):
        page = await lp.read_job_page(_candidate(), call)
    assert page.source == 'snapshot'
    assert any('altered in transit' in r.message for r in caplog.records)


PREMIUM_UPSELL = ('Job search faster with Premium Access company insights like strategic priorities, '
                  'headcount trends, and more Activate Premium for CA$0 1-month free trial with 24/7 support.')


@pytest.mark.parametrize(('text', 'expected'), [
    (PREMIUM_UPSELL, False),
    ('Try it for $0 today.', False),
    ('Compensation: CA$170,000 - CA$240,000 / year', True),
    ('The base salary for this role is €90K.', True),
    ('Pay starts at CA$0.50 per click', True),
    ('A signing bonus of $10,000.', True),
])
def test_a_zero_price_is_not_pay_wording(text, expected):
    """The Premium upsell kept tripping the pay guard (11 times, 2026-09-24..28) on 'CA$0'."""
    assert lp.mentions_pay(text) is expected
