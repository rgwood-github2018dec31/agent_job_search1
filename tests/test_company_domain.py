'''Company identity: domain folding, the match rule, the cache, the About page, and the wiring.'''
import json

import agentic_job_search.tools_generic as tools
import pytest
import yaml
from agentic_job_search import company_domain, linkedin_page
from agentic_job_search.company_domain import CompanyIdentity, identities_match
from agentic_job_search.config import LINKEDIN_PAGE_CAPTURE_JS
from bs4 import BeautifulSoup
from fakes import chat_reply

# The shape of a real About page (2026-10-19 captures of two companies), with every name, person
# and number replaced. A subsidiary with no Website field, and an independent company with one.
SUBSIDIARY_ABOUT_HTML = '''<html><head><script>window.x = "captcha security verification";</script></head>
<body><main><h1> Example Data Europe </h1>
<section><dl><dt><h3>Industry</h3></dt><dd>IT Services and IT Consulting</dd></dl></section>
<section><header><h3>Affiliated pages</h3></header><ul>
  <li><a href="https://www.linkedin.com/company/example-foundation/"><div><div>Example Foundation</div>
      <div><span>Non-profit</span></div><div><span>Subsidiary</span></div></div></a></li>
  <li><a href="https://www.linkedin.com/company/example-data-inc/"><div><div><!-- -->Example Data, Inc.<!-- --></div>
      <div><span>IT Services</span></div><div><span><!-- -->Parent<!-- --></span></div></div></a></li>
</ul></section></main></body></html>'''

INDEPENDENT_ABOUT_HTML = '''<html><body><main><h1>Paygate</h1><section><dl>
<dt><h3>Website</h3></dt><dd><a href="https://www.paygate.example.com"><span>https://www.paygate.example.com</span></a></dd>
<dt><h3>Phone</h3></dt><dd><a href="tel:+1 555-0100"><span>+1 555-0100</span></a></dd>
<dt><h3>Industry</h3></dt><dd>Software Development</dd>
</dl></section></main></body></html>'''


def _identity_reply(**fields) -> str:
    return json.dumps({'known': True, 'domain': '', 'parent_name': '', 'parent_domain': '', **fields})


@pytest.fixture
def model_answers(monkeypatch):
    '''Stub the model: name -> reply content. Returns the list of names it was asked about.'''
    answers: dict[str, str] = {}
    asked: list[str] = []

    async def fake_chat(prompt, **kwargs):
        name = prompt.split('Organization name: ', 1)[1].split('\n', 1)[0]
        asked.append(name)
        return chat_reply(answers[name])

    monkeypatch.setattr(company_domain, 'chat_openrouter', fake_chat)
    return answers, asked


# --- Domains and the match rule ----------------------------------------------------------------

@pytest.mark.parametrize('given, expected', [
    ('https://us.nttdata.com/en', 'nttdata.com'),
    ('www.ixopay.com', 'ixopay.com'),
    ('careers.bbc.co.uk', 'bbc.co.uk'),
    ('localhost', ''),
    ('', ''),
])
def test_registrable_domain_folds_subdomains(given, expected):
    assert company_domain.registrable_domain(given) == expected


def test_a_subsidiary_matches_its_parent_by_name():
    '''2026-10-19: NTT DATA Europe & Latam against an application to NTT DATA, Inc.'''
    candidate = CompanyIdentity('NTT DATA Europe & Latam', parent_name='NTT DATA, Inc.', source='linkedin')
    applied = CompanyIdentity('NTT DATA, Inc.', domain='nttdata.com', source='model')
    assert identities_match(candidate, applied)


def test_a_parent_matches_an_applied_subsidiary():
    candidate = CompanyIdentity('Amazon', domain='amazon.com', source='model')
    applied = CompanyIdentity('AWS', domain='aws.amazon.com', parent_domain='amazon.com', source='model')
    assert identities_match(candidate, applied)


def test_companies_sharing_a_domain_match():
    assert identities_match(CompanyIdentity('Acme EU', domain='acme.com', source='model'),
                            CompanyIdentity('Acme Corp', domain='acme.com', source='model'))


def test_siblings_under_one_parent_do_not_match():
    '''Applying to one arm of a group says nothing about another arm that only shares its owner.'''
    candidate = CompanyIdentity('Waymo', domain='waymo.com', parent_name='Alphabet', source='model')
    applied = CompanyIdentity('Google', domain='google.com', parent_name='Alphabet', source='model')
    assert not identities_match(candidate, applied)


def test_same_named_but_different_domains_do_not_match():
    assert not identities_match(CompanyIdentity('Cohere Health', domain='coherehealth.com', source='model'),
                                CompanyIdentity('Cohere', domain='cohere.com', source='model'))


# --- Model lookup and the cache ----------------------------------------------------------------

async def test_a_name_is_asked_about_once_then_served_from_the_cache(model_answers):
    answers, asked = model_answers
    answers['NTT DATA, Inc.'] = _identity_reply(domain='https://us.nttdata.com')
    first = await company_domain.identity_for_name('NTT DATA, Inc.')
    again = await company_domain.identity_for_name('NTT Data Inc')
    assert first == again
    assert first.domain == 'nttdata.com'
    assert asked == ['NTT DATA, Inc.']
    stored = yaml.safe_load(company_domain.COMPANY_IDENTITY_CACHE_PATH.read_text())
    assert list(stored['by_name']) == ['NTT DATA, Inc.'], 'names are stored as written'


async def test_unknown_is_an_answer_and_is_cached(model_answers):
    answers, asked = model_answers
    answers['Zzyzx Labs'] = json.dumps({'known': False, 'domain': '', 'parent_name': '', 'parent_domain': ''})
    identity = await company_domain.identity_for_name('Zzyzx Labs')
    assert identity.source == company_domain.SOURCE_NONE
    await company_domain.identity_for_name('Zzyzx Labs')
    assert asked == ['Zzyzx Labs']


async def test_a_malformed_answer_raises_and_is_not_cached(model_answers):
    answers, _asked = model_answers
    answers['Acme'] = json.dumps({'known': True, 'domain': 'not a domain'})
    with pytest.raises(ValueError, match='registrable domain'):
        await company_domain.identity_for_name('Acme')
    assert company_domain.cached_identity_for_name('Acme') is None


async def test_a_manual_entry_is_never_asked_about_or_overwritten(model_answers):
    _answers, asked = model_answers
    company_domain.COMPANY_IDENTITY_CACHE_PATH.write_text(yaml.safe_dump(
        {'by_name': {'Acme': {'source': 'manual', 'domain': 'acme.example.com'}}, 'by_slug': {}}))
    identity = await company_domain.identity_for_name('Acme')
    assert (identity.domain, identity.source) == ('example.com', 'manual')
    assert asked == []


async def test_linkedin_facts_win_and_the_model_fills_only_the_gaps(model_answers):
    answers, _asked = model_answers
    answers['Example Data Europe'] = _identity_reply(
        domain='exampledata.com', parent_name='Example Data, Inc.', parent_domain='exampledata.com')
    identity = await company_domain.record_linkedin_identity(
        'example-data-europe', 'Example Data Europe', '', 'Example Data, Inc.', 'example-data-inc')
    assert identity == CompanyIdentity('Example Data Europe', domain='exampledata.com',
                                       parent_name='Example Data, Inc.', parent_domain='exampledata.com',
                                       source='linkedin+model')
    assert company_domain.cached_identity_for_slug('example-data-europe') == identity


async def test_a_failed_model_fill_keeps_what_the_page_stated():
    # The autouse stub makes every model call fail.
    identity = await company_domain.record_linkedin_identity(
        'paygate', 'Paygate', 'https://www.paygate.example.com', '', '')
    assert (identity.domain, identity.source) == ('example.com', 'linkedin')


# --- The About page ----------------------------------------------------------------------------

def test_the_about_page_parent_comes_from_affiliated_pages():
    about = linkedin_page.parse_company_about(BeautifulSoup(SUBSIDIARY_ABOUT_HTML, 'lxml'), 'example-data-europe')
    assert about == linkedin_page.CompanyAbout(
        slug='example-data-europe', name='Example Data Europe', website='',
        parent_name='Example Data, Inc.', parent_slug='example-data-inc')


def test_the_about_page_website_comes_from_its_definition_list():
    about = linkedin_page.parse_company_about(BeautifulSoup(INDEPENDENT_ABOUT_HTML, 'lxml'), 'paygate')
    assert (about.name, about.website, about.parent_name) == ('Paygate', 'https://www.paygate.example.com', '')


def test_a_page_with_no_company_name_is_not_an_about_page():
    with pytest.raises(RuntimeError, match='not an About page'):
        linkedin_page.parse_company_about(BeautifulSoup('<html><body>Sign in</body></html>', 'lxml'), 'x')


def test_block_words_inside_scripts_are_not_a_block():
    '''Both healthy 2026-10-19 captures carried "captcha" in their inline scripts.'''
    assert linkedin_page.visible_block_signature(BeautifulSoup(SUBSIDIARY_ABOUT_HTML, 'lxml')) == ''


def test_block_words_in_the_visible_page_are_a_block():
    soup = BeautifulSoup('<html><body><h1>Let us do a quick security check</h1>'
                         '<p>Please complete this captcha</p></body></html>', 'lxml')
    assert linkedin_page.visible_block_signature(soup) == 'captcha'


def _fake_browser(html: str):
    calls = []

    async def call(tool, args):
        calls.append((tool, args))
        if tool == 'browser_evaluate':
            payload = {'html': html, 'chars': len(html.encode('utf-16-le')) // 2}
            return f'### Result\n{json.dumps(payload)}'
        return ''
    return call, calls


async def test_reading_an_about_page_only_navigates_waits_and_captures():
    call, calls = _fake_browser(INDEPENDENT_ABOUT_HTML)
    about = await linkedin_page.read_company_about('paygate', call)
    assert about.website == 'https://www.paygate.example.com'
    assert [tool for tool, _args in calls] == ['browser_navigate', 'browser_wait_for', 'browser_evaluate']
    assert calls[0][1] == {'url': 'https://www.linkedin.com/company/paygate/about/'}
    assert calls[2][1] == {'function': LINKEDIN_PAGE_CAPTURE_JS}, 'the tested read-only capture, nothing else'


async def test_a_challenge_page_raises_the_stop_error():
    call, _calls = _fake_browser('<html><body><h1>Paygate</h1><p>We noticed unusual activity</p></body></html>')
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='unusual activity'):
        await linkedin_page.read_company_about('paygate', call)


def test_the_poster_slug_comes_from_the_about_the_company_card_only():
    soup = BeautifulSoup(
        '<html><body><a href="https://www.linkedin.com/company/someone-else/">Similar job</a>'
        '<div id="JobDetails_AboutTheCompany_123">'
        '<a href="https://www.linkedin.com/company/example-data-europe/life/">Example</a></div></body></html>',
        'lxml')
    assert linkedin_page.posting_company_slug(soup) == 'example-data-europe'


# --- company_matches_applied -------------------------------------------------------------------

@pytest.fixture
def applied_ntt(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {'NTT DATA, Inc.': 'ntt.pdf'})
    monkeypatch.setattr(tools, '_applied_identities', {
        'NTT DATA, Inc.': CompanyIdentity('NTT DATA, Inc.', domain='nttdata.com', source='model')})


@pytest.fixture
def name_match_calls(monkeypatch):
    calls = []

    async def fake_by_name(candidate):
        calls.append(candidate)
        return 'ntt.pdf'
    monkeypatch.setattr(tools, '_company_matches_by_name', fake_by_name)
    return calls


async def test_an_identity_match_needs_no_name_match(applied_ntt, name_match_calls):
    identity = CompanyIdentity('NTT DATA Europe & Latam', parent_name='NTT DATA, Inc.', source='linkedin')
    assert await tools.company_matches_applied('NTT DATA Europe & Latam', identity) == 'ntt.pdf'
    assert name_match_calls == []


async def test_a_resolved_non_match_is_no_without_asking_by_name(applied_ntt, name_match_calls, model_answers):
    answers, _asked = model_answers
    answers['IXOPAY'] = _identity_reply(domain='ixopay.com')
    assert await tools.company_matches_applied('IXOPAY') is None
    assert name_match_calls == []


async def test_an_unidentified_name_falls_back_to_one_name_match_per_run(applied_ntt, name_match_calls):
    '''2026-10-19: the same name got two different answers in one run. Now it gets one.'''
    # The autouse stub makes the identity lookup fail, as an outage would.
    assert await tools.company_matches_applied('NTT DATA Europe & Latam') == 'ntt.pdf'
    assert await tools.company_matches_applied('NTT DATA Europe & Latam') == 'ntt.pdf'
    assert name_match_calls == ['NTT DATA Europe & Latam']
