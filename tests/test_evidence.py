'''Verify Source Reference: a prediction stands only on a citation code finds in the input.'''

import pytest
from agentic_job_search import evidence, job_trace
from agentic_job_search.evidence import Claim

# The page text of linkedin-4477326789 (2026-10-16), trimmed to its header.
OVERSTORY_PAGE = '''Overstory
Staff Machine Learning Engineer - Wildfire
Netherlands · 17 hours ago · Over 100 people clicked apply
Remote
Full-time
About the job
We’re looking for a Staff ML Engineer — someone who owns models end to end.'''


@pytest.fixture(autouse=True)
def _fresh():
    evidence.reset_rejections()
    yield
    evidence.reset_rejections()


@pytest.mark.parametrize('quote', [
    '17 hours ago',
    'NETHERLANDS · 17 HOURS AGO',
    "We're looking for a Staff ML Engineer - someone",  # straight quote and hyphen for curly and em dash
    'Staff Machine Learning   Engineer\n- Wildfire',
    'We’re looking for … owns models end to end.',
    '"Remote"',
])
def test_a_faithful_citation_is_found(quote):
    assert evidence.find_quote(quote, OVERSTORY_PAGE) is not None


@pytest.mark.parametrize('quote', [
    '2025-10-14',                       # the year the extractor invented
    '16-17 hours ago',
    'owns models end to end … We’re looking',   # ellipsis pieces out of order
    '',
    '...',
])
def test_an_invented_or_reordered_citation_is_not_found(quote):
    assert evidence.find_quote(quote, OVERSTORY_PAGE) is None


def test_a_short_citation_proves_nothing_unless_it_is_the_value():
    assert not evidence.quote_is_specific('EU', 'area_wide')
    assert evidence.quote_is_specific('PhD', 'phd')
    assert evidence.quote_is_specific('must be based in Germany', 'country_only')


async def test_regex_hit_needs_no_model(monkeypatch):
    async def no_model(*args, **kwargs):
        raise AssertionError('the relocation model must not be called when regex finds every citation')
    monkeypatch.setattr(evidence, '_relocate', no_model)
    verdicts = await evidence.verify_claims(
        'extract', 'job 1', [Claim('location', 'Netherlands (Remote)', 'Netherlands · 17 hours ago')],
        OVERSTORY_PAGE, 'page.txt')
    assert verdicts['location'].status == evidence.STATUS_REGEX


async def test_an_invented_date_is_rejected_and_counted(monkeypatch):
    async def finds_nothing(call_site, subject, claims, source, stage_stats):
        return {'0': ''}, 'fake reply'
    monkeypatch.setattr(evidence, '_relocate', finds_nothing)
    verdicts = await evidence.verify_claims(
        'extract', 'job 4477326789', [Claim('date_posted', '2025-10-14', '2025-10-14')], OVERSTORY_PAGE, 'page.txt')
    assert verdicts['date_posted'].status == evidence.STATUS_REJECTED
    assert evidence.rejection_counts() == {('extract', 'date_posted'): 1}


async def test_a_relocated_span_still_has_to_be_in_the_source(monkeypatch):
    async def relocates(call_site, subject, claims, source, stage_stats):
        # Claim 0: a real span the citing model paraphrased. Claim 1: the relocator invents one.
        return {'0': 'Over 100 people clicked apply', '1': 'Fluent Dutch required'}, 'fake reply'
    monkeypatch.setattr(evidence, '_relocate', relocates)
    verdicts = await evidence.verify_claims('extract', 'job 1', [
        Claim('applicants', 'over 100', 'more than a hundred applicants'),
        Claim('language_requirement', 'dutch', 'Dutch is a must'),
    ], OVERSTORY_PAGE, 'page.txt')
    assert verdicts['applicants'].status == evidence.STATUS_RELOCATED
    assert verdicts['language_requirement'].status == evidence.STATUS_REJECTED


async def test_a_failed_relocation_call_rejects_what_it_was_asked_about(monkeypatch):
    async def fails(*args, **kwargs):
        raise RuntimeError('timed out after 180.0s')
    monkeypatch.setattr(evidence, '_relocate', fails)
    verdicts = await evidence.verify_claims('extract', 'job 1', [
        Claim('location', 'Netherlands', 'Netherlands'),
        Claim('relocation', 'United Kingdom', 'Based in the United Kingdom'),
    ], OVERSTORY_PAGE, 'page.txt')
    assert verdicts['location'].accepted
    assert not verdicts['relocation'].accepted


async def test_verdicts_are_recorded_into_the_current_job_trace(tmp_path, monkeypatch):
    async def finds_nothing(call_site, subject, claims, source, stage_stats):
        return {}, 'fake reply'
    monkeypatch.setattr(evidence, '_relocate', finds_nothing)
    _trace, token = job_trace.start(tmp_path / 'linkedin-1.jsonl', 'linkedin', '1')
    try:
        await evidence.verify_claims('extract', 'job 1', [Claim('closed', True, 'No longer accepting applications')],
                                     OVERSTORY_PAGE, 'page.txt')
    finally:
        job_trace.finish(token)
    [step] = job_trace.load(tmp_path / 'linkedin-1.jsonl')
    assert step['kind'] == 'evidence'
    assert step['verdicts'][0]['status'] == evidence.STATUS_REJECTED
