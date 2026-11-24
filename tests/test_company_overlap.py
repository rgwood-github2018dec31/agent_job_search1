'''Decide Company Overlap (2026-11-19): a card whose name only overlaps a listed name goes to a model.'''

import pytest
import yaml
from agentic_job_search import agent, applied_organizations, company_overlap
from agentic_job_search import tools_generic as tools
from agentic_job_search.applied_organizations import AppliedOrganization
from fakes import chat_reply

_RECORDS = {
    'KLAM': AppliedOrganization('KLAM', '2026-09-10-klam.pdf', ('KLAM', 'KLAM Romania'), ('klam.com',)),
    'Thinkworks': AppliedOrganization('Thinkworks', '2026-06-01-tw.pdf', ('Thinkworks', 'Tessel Topco')),
}


@pytest.mark.parametrize('card, listed, kind', [
    ('KLAM Systems', 'KLAM', company_overlap.OVERLAP_COMPLETE),
    ('Tessel', 'Tessel Topco', company_overlap.OVERLAP_COMPLETE),
    ('Tessel Consulting', 'Tessel Topco', company_overlap.OVERLAP_PARTIAL),
    ('Trumble Inc.', 'Trumble', company_overlap.OVERLAP_COMPLETE),   # folded alike, not exact: asked
    ('Epamco', 'KLAM', ''),              # a word must be shared whole
    ('The Agile Monkeys', 'Agile Unicorn', company_overlap.OVERLAP_PARTIAL),
])
def test_overlap_kind(card, listed, kind):
    assert company_overlap.overlap_kind(card, listed) == kind


class _Model:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    async def __call__(self, prompt, tool_name, description, parameters, **kwargs):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError('unexpected model call')
        return self.answers.pop(0), chat_reply('', cost_usd=0.0001)


def _verdict(verdict, confidence, matched='', reasoning='because'):
    return {'reasoning': reasoning, 'verdict': verdict, 'confidence': confidence, 'matched_name': matched}


@pytest.fixture
def applied(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {k: r.pdf for k, r in _RECORDS.items()})
    monkeypatch.setattr(tools, '_applied_records', _RECORDS)
    monkeypatch.setattr(tools, '_applied_name_index', applied_organizations.name_index(_RECORDS))
    monkeypatch.setattr(tools, '_applied_overlap_index', company_overlap.overlap_index(_RECORDS))


async def test_a_confident_same_skips_the_card_and_is_cached(monkeypatch, applied):
    model = _Model(_verdict('same', 5, 'KLAM'))
    monkeypatch.setattr(company_overlap, 'chat_openrouter_tool', model)
    assert await tools.company_matches_applied('KLAM Systems', 'Lead AI Engineer') == '2026-09-10-klam.pdf'
    assert 'KLAM Systems' in model.prompts[0] and 'Lead AI Engineer' in model.prompts[0]
    assert await tools.company_matches_applied('KLAM Systems', 'Other title') == '2026-09-10-klam.pdf'   # cached
    saved = yaml.safe_load(company_overlap.COMPANY_OVERLAP_VERDICTS_PATH.read_text(encoding='utf-8'))
    assert saved['klam systems | KLAM']['verdict'] == 'same'


@pytest.mark.parametrize('answer', [
    _verdict('different', 5), _verdict('unclear', 3), _verdict('same', 3, 'KLAM'),   # same, but not confident
])
async def test_anything_but_a_confident_same_lets_the_card_through(monkeypatch, applied, answer):
    monkeypatch.setattr(company_overlap, 'chat_openrouter_tool', _Model(answer))
    assert await tools.company_matches_applied('KLAM Systems') is None
    assert company_overlap.counts['overlap_passed'] == 1


@pytest.mark.parametrize('answer', [
    _verdict('same', 5, 'KLAM Polska'),    # not a listed name
    _verdict('maybe', 4), _verdict('same', 9, 'KLAM'),
])
async def test_an_unusable_answer_fails_open_and_is_counted(monkeypatch, applied, answer):
    monkeypatch.setattr(company_overlap, 'chat_openrouter_tool', _Model(answer))
    assert await tools.company_matches_applied('KLAM Systems') is None
    assert company_overlap.counts['overlap_failed'] == 1
    assert not company_overlap.COMPANY_OVERLAP_VERDICTS_PATH.exists()


async def test_no_model_is_asked_without_an_overlap_or_on_an_exact_name(monkeypatch, applied):
    monkeypatch.setattr(company_overlap, 'chat_openrouter_tool', _Model())
    assert await tools.company_matches_applied('Vorlix') is None
    assert await tools.company_matches_applied('klam romania') == '2026-09-10-klam.pdf'


async def test_a_manual_verdict_in_the_file_decides(monkeypatch, applied):
    company_overlap.COMPANY_OVERLAP_VERDICTS_PATH.write_text(yaml.safe_dump({'turing | Thinkworks': {
        'verdict': 'different', 'confidence': 5, 'source': 'manual'}}), encoding='utf-8')
    monkeypatch.setattr(company_overlap, 'chat_openrouter_tool', _Model())
    assert await tools.company_matches_applied('Tessel') is None


async def test_an_unreadable_verdicts_file_is_never_overwritten(monkeypatch, applied, caplog):
    company_overlap.COMPANY_OVERLAP_VERDICTS_PATH.write_text('{not: [yaml', encoding='utf-8')
    monkeypatch.setattr(company_overlap, 'chat_openrouter_tool', _Model(_verdict('different', 5)))
    assert await tools.company_matches_applied('Tessel') is None
    assert company_overlap.COMPANY_OVERLAP_VERDICTS_PATH.read_text(encoding='utf-8') == '{not: [yaml'
    assert 'move the file aside' in caplog.text


def test_the_summary_line_counts_cards_and_failed_questions():
    assert company_overlap.summary_line({}) == ''
    line = company_overlap.summary_line({'overlap_skipped': 1, 'overlap_passed': 2, 'overlap_failed': 1})
    assert '3 card(s) asked' in line and '1 skipped' in line and '2 passed' in line and '1 question(s) failed' in line
    assert 'failed' not in company_overlap.summary_line({'overlap_passed': 1})


def test_a_failed_overlap_question_is_a_run_health_alert():
    alerts = agent.assess_run_health({'company_overlap': {'overlap_passed': 2, 'overlap_failed': 1}})
    assert any(a.startswith('COMPANY OVERLAP FAILED 1 time(s)') for a in alerts)
    assert not any('COMPANY OVERLAP' in a for a in agent.assess_run_health({'company_overlap': {'overlap_passed': 2}}))
