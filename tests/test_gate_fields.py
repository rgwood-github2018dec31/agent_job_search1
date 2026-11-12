'''Ask Gate Fields: each gate field asked on its own, cited, and read for the gates in code.'''
import json
import re

import pytest
from agentic_job_search import config, evidence, gate_fields, triage
from agentic_job_search.linkedin_page import HeaderCard, JobPage

PAGE = '''Acme
Senior Data Scientist
Vancouver, BC
·
2 days ago
Hybrid
Full-time
## About the job
You are located in Vancouver or willing to relocate.
**Requirements**
- Fluent English
- French is an asset
## About the company
A "work from anywhere" culture across 30 countries.'''
_FIELD_RE = re.compile(r'Field `(\w+)`')


def _page(location: str = 'Vancouver, BC', workplace: str = 'hybrid') -> JobPage:
    return JobPage(text=PAGE, source='dom', markdown=PAGE, header=HeaderCard(location=location, workplace=workplace))


def _reply(value: str, quotes: list[str], basis: str | None = None) -> str:
    reply = {'quotes': quotes, 'reason': 'because the page says so', 'value': value}
    if basis is not None:
        reply['basis'] = basis
    return json.dumps(reply)


def _server(monkeypatch, answer) -> None:
    '''The OpenRouter tool server behind chat_openrouter_tool: `answer(prompt)` gives the submit_answer
    arguments as the model wrote them (a JSON string, possibly broken), or raises.'''
    monkeypatch.setattr(config, 'SHADOW_MODEL', None)

    async def call(url, tool, args, timeout_seconds=None):
        assert args['tool_choice'] == {'type': 'function', 'function': {'name': gate_fields.TOOL_NAME}}
        arguments = answer(args['messages'][-1]['content'])
        return json.dumps({'ok': True, 'content': None, 'cost_usd': 0.001, 'model': 'm', 'provider': 'P',
                           'finish_reason': 'tool_calls', 'usage': {}, 'tool_calls': [
                               {'id': 'c1', 'type': 'function',
                                'function': {'name': gate_fields.TOOL_NAME, 'arguments': arguments}}]})

    monkeypatch.setattr(triage, 'call_mcp_tool', call)


def _fake_chat(monkeypatch, replies: dict[str, list[str]]) -> list[str]:
    '''Each field's replies, served in turn; returns the fields asked, in call order.'''
    asked = []

    def answer(prompt):
        field_name = _FIELD_RE.search(prompt).group(1)
        asked.append(field_name)
        reply = replies[field_name].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    _server(monkeypatch, answer)
    return asked


@pytest.fixture(autouse=True)
def _fresh():
    evidence.reset_rejections()
    yield
    evidence.reset_rejections()


@pytest.fixture
def countries(monkeypatch):
    '''classify_location without the classifier: the countries a relocation names.'''
    known = {'Germany': ['Germany'], 'Canada': ['Canada'], 'Germany; Canada': ['Germany', 'Canada']}

    async def classify(text):
        return {'countries': known.get(text, [])}

    monkeypatch.setattr(gate_fields, 'classify_location', classify)


def _hybrid_replies() -> dict[str, list[str]]:
    return {
        'relocation': [_reply('Vancouver, BC', ['Vancouver, BC', 'Hybrid'], basis='office')],
        'residency_scope': [_reply('city_only', ['You are located in Vancouver or willing to relocate.'])],
        'language_requirement': [_reply('English', ['Fluent English'])],
    }


async def test_a_cited_answer_reaches_the_extract_as_the_gates_read_it(monkeypatch, countries):
    _fake_chat(monkeypatch, _hybrid_replies())
    extract = {'relocation': '', 'residency_scope': '', 'language_requirement': 'english, french'}
    stats = {'cost': 0.0}
    answers = await gate_fields.ask_gate_fields({'job_id': '1', 'company': 'Acme'}, extract, _page(), stats)
    assert extract == {'relocation': 'Vancouver, BC', 'residency_scope': 'country_only', 'language_requirement': 'english'}
    assert answers['residency_scope'].raw_value == 'city_only', 'the office city is kept for the trace'
    assert stats['cost'] == pytest.approx(0.003)


async def test_the_prompt_states_the_header_facts_code_read(monkeypatch, countries):
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        field_name = _FIELD_RE.search(prompt).group(1)
        return _reply(gate_fields.NOT_KNOWN[field_name], [], basis='' if field_name == 'relocation' else None)

    _server(monkeypatch, answer)
    await gate_fields.ask_gate_fields({}, {}, _page(workplace='remote'), {})
    assert all('location "Vancouver, BC"; workplace type "remote"' in p for p in prompts)
    assert all(f'Answer by calling {gate_fields.TOOL_NAME}' in p and 'JSON' not in p for p in prompts)


def test_the_text_question_for_a_model_without_function_calls_asks_for_json():
    '''scripts/local_field_probe.py asks local models, which have no function calls.'''
    text = gate_fields.question(PAGE, 'facts', 'relocation')
    assert 'Reply with only a JSON object' in text and gate_fields.TOOL_NAME not in text


async def test_an_answer_off_the_schema_is_asked_again_and_its_cost_counted(monkeypatch, countries):
    replies = _hybrid_replies()
    replies['residency_scope'].insert(0, _reply('everywhere', ['Hybrid']))   # outside the enum
    _fake_chat(monkeypatch, replies)
    stats = {'cost': 0.0}
    answers = await gate_fields.ask_gate_fields({}, {}, _page(), stats)
    assert answers['residency_scope'].attempt == 2
    assert stats['cost'] == pytest.approx(0.004), 'the refused call was paid for too'


async def test_not_known_is_absent(monkeypatch, countries):
    _fake_chat(monkeypatch, {f: [_reply(gate_fields.NOT_KNOWN[f], [], basis='' if f == 'relocation' else None)]
                             for f in gate_fields.GATE_FIELDS})
    extract = {'relocation': 'Spain', 'residency_scope': 'area_wide', 'language_requirement': 'german'}
    await gate_fields.ask_gate_fields({}, extract, _page(), {})
    assert extract == {'relocation': '', 'residency_scope': '', 'language_requirement': ''}


async def test_a_quote_not_on_the_page_rejects_the_value_and_is_counted(monkeypatch, countries, caplog):
    replies = _hybrid_replies()
    replies['language_requirement'] = [_reply('german', ['Fluent German required'])]
    _fake_chat(monkeypatch, replies)
    extract = {}
    answers = await gate_fields.ask_gate_fields({'job_id': '7'}, extract, _page(), {})
    assert extract['language_requirement'] == ''
    assert answers['language_requirement'].rejected
    assert evidence.rejection_counts() == {(gate_fields.CALL_SITE, 'language_requirement'): 1}
    assert 'Fluent German required' in caplog.text


async def test_a_stated_value_without_any_quote_is_rejected(monkeypatch, countries):
    replies = _hybrid_replies()
    replies['residency_scope'] = [_reply('area_wide', [])]
    _fake_chat(monkeypatch, replies)
    extract = {}
    await gate_fields.ask_gate_fields({}, extract, _page(), {})
    assert extract['residency_scope'] == ''


async def test_a_malformed_reply_is_asked_again(monkeypatch, countries, caplog):
    replies = _hybrid_replies()
    # GLM's shape on 2026-10-30: the reason's opening quote missing.
    replies['residency_scope'].insert(0, '{"quotes": ["Hybrid"], "reason": The job is hybrid, "value": "city_only"}')
    _fake_chat(monkeypatch, replies)
    extract = {}
    answers = await gate_fields.ask_gate_fields({}, extract, _page(), {})
    assert extract['residency_scope'] == 'country_only'
    assert answers['residency_scope'].attempt == 2
    assert 'answered on attempt 2' in caplog.text


async def test_every_attempt_failing_keeps_the_extractor_value(monkeypatch, countries, caplog):
    replies = _hybrid_replies()
    replies['relocation'] = [RuntimeError('timed out after 180.0s'), _reply('Vancouver, BC', ['Vancouver, BC'])[:-2]]
    _fake_chat(monkeypatch, replies)
    extract = {'relocation': 'Canada'}
    answers = await gate_fields.ask_gate_fields({'job_id': '9'}, extract, _page(), {})
    assert extract['relocation'] == 'Canada'
    assert 'timed out after 180.0s' in answers['relocation'].error
    assert "the extractor's 'Canada' stands" in caplog.text


@pytest.mark.parametrize(('relocation', 'basis', 'expected'), [
    ('Germany', 'remote_place_only', ''),            # Q17: remote in an EU country, live anywhere in the EU
    ('Canada', 'remote_place_only', 'Canada'),       # remote in Vancouver: anywhere in Canada, not beyond
    ('Germany; Canada', 'remote_place_only', 'Germany; Canada'),
    ('Germany', 'stated', 'Germany'),                # "must be based in Germany" pins it
    ('Vancouver, BC', 'office', 'Vancouver, BC'),    # Q18: a hybrid holder must live in the office city
    ('anywhere', 'anywhere', ''),
])
async def test_relocation_is_read_for_the_gates_in_code(monkeypatch, countries, relocation, basis, expected):
    replies = _hybrid_replies()
    replies['relocation'] = [_reply(relocation, ['Vancouver, BC'], basis=basis)]
    _fake_chat(monkeypatch, replies)
    extract = {}
    await gate_fields.ask_gate_fields({}, extract, _page(), {})
    assert extract['relocation'] == expected


@pytest.mark.parametrize('reply', [
    _reply('everywhere', ['Hybrid']),                                # residency outside its enum
    json.dumps({'quotes': 'Hybrid', 'reason': 'r', 'value': 'city_only'}),   # quotes not a list
    json.dumps({'quotes': [], 'reason': 'r'}),                       # no value
])
def test_a_reply_off_the_schema_is_refused(reply):
    with pytest.raises(ValueError):
        gate_fields.parse_answer('residency_scope', reply)


def test_relocation_needs_a_known_basis():
    with pytest.raises(ValueError, match='basis'):
        gate_fields.parse_answer('relocation', _reply('Canada', ['Canada'], basis='because'))


async def test_a_quoted_header_fact_from_the_search_card_is_grounded(monkeypatch, countries):
    '''2026-11-06 (Sii, Mapbox): the signed-out page never says "Remote"; the workplace came from the
    "(Remote)" search card, and the model quoted it as the header fact it was told. Not a hallucination.'''
    replies = _hybrid_replies()
    replies['relocation'] = [_reply('Canada', ['Canada', 'remote'], basis='remote_place_only')]
    _fake_chat(monkeypatch, replies)
    extract = {}
    answers = await gate_fields.ask_gate_fields({}, extract, _page(location='Canada', workplace='remote'), {})
    assert not answers['relocation'].rejected
    assert extract['relocation'] == 'Canada'


async def test_the_header_facts_sentence_itself_grounds_nothing(monkeypatch, countries):
    replies = _hybrid_replies()
    replies['language_requirement'] = [_reply('german', ['workplace type'])]
    _fake_chat(monkeypatch, replies)
    answers = await gate_fields.ask_gate_fields({}, {}, _page(), {})
    assert answers['language_requirement'].rejected
