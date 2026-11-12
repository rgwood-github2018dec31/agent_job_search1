'''Job Path: one Job Trace laid over the Stage 2 skeleton (src/agentic_job_search/job_path.py).

The fixtures carry the steps a real trace has on every exit: `saved` (every exit but extract
failed saves the job) and `outcome` (record_job_outcome, from 2026-11-12). The first version of
this overlay was tested without them and labelled every already-applied and triaged-out job
"reason not recorded", which is what a fixture cleaner than reality hides.
'''

import pytest
from agentic_job_search import job_path


def _steps(*kinds_and_data: tuple[str, dict]) -> list[dict]:
    return [{'kind': kind, **data} for kind, data in kinds_and_data]


def _head() -> list[tuple[str, dict]]:
    return [('origin', {'listing': {'company': 'Acme', 'title': 'Engineer'}}), ('page_read', {'source': 'exa'}),
            ('llm_call', {'call_site': 'extract_job_page_openrouter'}), ('extract', {'fields': {}}),
            ('gate_fields', {'answers': {}}), ('classified', {})]


def _rules(fired: str | None = None) -> list[tuple[str, dict]]:
    rules = []
    for rule in job_path.HARD_RULES:
        rules.append(('hard_rule', {'rule': rule, 'fired': rule == fired, 'reason': f'{rule} reason'}))
        if rule == fired:
            break
    return rules


def _left(outcome: str, reason: str, check: str = '', rating: int | None = 1) -> list[tuple[str, dict]]:
    '''How a job that left before the rater ends: saved, then its outcome.'''
    return [('saved', {'rating': rating, 'summary': reason}),
            ('outcome', {'outcome': outcome, 'check': check, 'rating': rating, 'reason': reason})]


def _rated(rating: int, cap_reason: str = '') -> list[tuple[str, dict]]:
    return [('rating', {'raw_rating': rating + bool(cap_reason), 'capped_rating': rating, 'cap_reason': cap_reason}),
            ('outcome', {'outcome': 'rated', 'check': '', 'rating': rating, 'reason': 'fit'}),
            ('saved', {'rating': rating, 'summary': 'fit'})]


def _overlay(*parts) -> dict[str, job_path.Node]:
    return {n.id: n for n in job_path.overlay(_steps(*parts), list(job_path.HARD_RULES))}


def test_a_fired_rule_is_the_exit_and_later_stages_are_not_reached():
    nodes = _overlay(*_head(), *_rules('language'), *_left('hard_ruled', 'language reason'))
    assert nodes['rule:language'].exit_state == 'taken'
    assert nodes['rule:language'].exit_notes == ['language reason']
    assert nodes['rule:stale'].exit_state == 'not_taken'
    assert nodes['rule:location'].state == 'not_reached'
    assert nodes['save'].state == 'not_reached'   # its `saved` step is the exit's, not a stage reached


@pytest.mark.parametrize(('check', 'exit_node'), [('end_client', 'applied_client'),
                                                  ('company_identity', 'applied_domain')])
def test_an_already_applied_job_exits_at_the_check_that_matched(check, exit_node):
    rules = _rules() if check == 'company_identity' else []
    nodes = _overlay(*_head(), *rules, *_left('already_applied', 'Pharma Co already applied to', check))
    assert nodes[exit_node].exit_state == 'taken'
    assert nodes[exit_node].exit_notes == ['Pharma Co already applied to']
    assert nodes['rate'].state == 'not_reached'
    if check == 'end_client':   # the rules after the end-client check were never reached, not skipped
        assert {nodes[f'rule:{r}'].state for r in job_path.HARD_RULES} == {'not_reached'}


def test_a_triaged_out_job_exits_at_triage():
    nodes = _overlay(*_head(), *_rules(), ('llm_call', {'call_site': 'triage_job_fit'}),
                     *_left('triaged_out', 'weak fit', rating=2))
    assert (nodes['triage'].state, nodes['triage'].exit_state) == ('taken', 'taken')
    assert nodes['applied_domain'].state == 'not_reached'


def test_a_failed_extract_exits_at_extract():
    nodes = _overlay(*_head()[:3], ('outcome', {'outcome': 'extract_failed', 'reason': 'page extraction failed'}))
    assert nodes['extract'].exit_state == 'taken'
    assert nodes['gate'].state == 'not_reached'


def test_a_notified_job_takes_no_exit():
    nodes = job_path.overlay(_steps(*_head(), *_rules(), *_rated(4), ('notification', {
        'text': 'hi', 'sent': True, 'message_kind': 'notification'})), list(job_path.HARD_RULES))
    assert not [n.id for n in nodes if n.exit_state == 'taken']
    by_id = {n.id: n for n in nodes}
    assert by_id['notified'].state == 'taken' and by_id['triage'].state == 'skipped'
    assert job_path.outcome(nodes) == 'notified'


def test_a_follow_up_recorded_under_its_old_kind_still_counts_as_sent():
    '''Traces before 2026-11-12 recorded a follow-up's notification under kind `follow-up`.'''
    nodes = _overlay(*_head(), *_rules(), *_rated(4), ('follow-up', {'text': 'again', 'sent': True}))
    assert nodes['notified'].state == 'taken' and nodes['notified'].notes == ['follow-up sent']


def test_a_cap_is_a_detour_not_an_exit():
    nodes = _overlay(*_head(), *_rules(), *_rated(3, cap_reason='french'),
                     ('notification', {'text': None, 'why': 'below the notification rating'}))
    assert nodes['caps'].exit_state == 'detour'
    assert nodes['caps'].exit_label == 'capped 4 → 3'
    assert nodes['notify'].exit_state == 'taken'


def test_an_eval_error_exits_where_the_trace_stops_with_its_reason():
    nodes = _overlay(*_head(), *_rules(), ('llm_call', {'call_site': 'rate_with_openrouter', 'error': 'timed out'}),
                     ('outcome', {'outcome': 'eval_error', 'reason': 'released — timed out'}))
    assert (nodes['rate'].exit_state, nodes['rate'].exit_label) == ('taken', 'eval_error')
    assert nodes['applied_domain'].exit_state == 'not_taken'


def test_a_trace_without_an_outcome_says_so_rather_than_guessing():
    nodes = _overlay(*_head(), *_rules(), ('saved', {'rating': 1, 'summary': 'already applied to end client X'}))
    assert nodes['save'].exit_label == 'outcome not recorded'
    assert nodes['save'].exit_notes[0] == 'saved as: already applied to end client X'
    assert nodes['save'].exit_notes[1].startswith('the trace has no outcome step')


def test_a_trace_without_an_outcome_still_reads_a_fired_rule_and_a_rating():
    '''Before 2026-11-12 no trace has an outcome step; a fired rule or a rating settles the exit on its own.'''
    rejected = _overlay(*_head(), *_rules('degree'), ('saved', {'rating': 1, 'summary': 'auto rejected'}))
    assert rejected['rule:degree'].exit_state == 'taken' and rejected['rule:degree'].exit_notes == ['degree reason']
    notified = job_path.overlay(_steps(*_head(), *_rules(), ('rating', {'raw_rating': 4, 'capped_rating': 4}),
                                       ('saved', {}), ('notification', {'text': 'hi', 'sent': True})),
                                list(job_path.HARD_RULES))
    assert job_path.outcome(notified) == 'notified'


def test_an_unknown_step_is_named_as_unmapped_where_it_lands():
    nodes = _overlay(*_head(), ('llm_call', {'call_site': 'brand_new_check'}), ('mystery', {}),
                     ('outcome', {'outcome': 'eval_error', 'reason': 'boom'}))
    assert any("unmapped LLM call site 'brand_new_check'" in n for n in nodes['classify'].notes)
    assert any("unmapped step kind 'mystery'" in n for n in nodes['classify'].notes)


def test_the_run_view_counts_jobs_per_stage_and_per_exit():
    rules = list(job_path.HARD_RULES)
    jobs = [job_path.overlay(_steps(*_head(), *_rules('language'), *_left('hard_ruled', 'language reason')), rules)
            for _ in range(2)]
    jobs.append(job_path.overlay(_steps(*_head(), *_rules(), *_left('already_applied', 'x', 'company_identity')),
                                 rules))
    nodes = {n.id: n for n in job_path.aggregate(jobs, rules)}
    assert nodes['rule:language'].exit_label == '2 × rejected: language'
    assert nodes['rule:location'].count == 1
    assert nodes['applied_domain'].exit_state == 'taken'


def test_emoji_are_dropped_from_notes():
    assert job_path.cut('⭐ 4/5 — 🧑‍💼 Recruiter') == '4/5 — Recruiter'
