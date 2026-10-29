'''Record Rating Inputs: the rater's system message is saved part by part, each exactly as sent.'''

from pathlib import Path

import yaml
from agentic_job_search import agent, rating_inputs

_REQUIREMENTS = 'Remote only. Python.'
_SUMMARY = 'Senior AI Engineer, agentic systems.'


def _system_message(tmp_path, monkeypatch) -> str:
    requirements = tmp_path / 'JOB_REQUIREMENTS.md'
    requirements.write_text(_REQUIREMENTS, encoding='utf-8')
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', requirements)
    return agent.build_evaluator_prompt(agent._wrap_reference_summary(_SUMMARY, 20))


def test_the_system_message_splits_into_its_parts_with_nothing_lost(tmp_path, monkeypatch):
    message = _system_message(tmp_path, monkeypatch)
    parts = rating_inputs.split_system_message(message)
    assert _REQUIREMENTS in parts['job_requirements'] and parts['job_requirements'].startswith('--- JOB_REQUIREMENTS.md ---')
    assert _SUMMARY in parts['ideal_role_profile'] and 'distilled from 20 jobs' in parts['ideal_role_profile']
    assert parts['evaluator_instructions'] == agent.build_evaluator_instructions().strip()
    assert sum(len(p) for p in parts.values()) <= len(message)


def test_the_full_reference_block_stands_in_for_a_failed_summary():
    parts = rating_inputs.split_system_message(
        '--- REFERENCE JOBS (x) ---\njob one\n--- END REFERENCE JOBS ---\n\nRate the job.')
    assert parts['ideal_role_profile'].endswith('--- END REFERENCE JOBS ---')
    assert parts['evaluator_instructions'] == 'Rate the job.'


def test_the_profile_is_saved_with_how_it_was_made(tmp_path, monkeypatch):
    cache = tmp_path / 'ref_cache.yaml'
    cache.write_text(yaml.safe_dump({
        'hash': 'abc', 'provider': 'openrouter', 'model': 'deepseek/x (Morph)', 'generated_on': '2026-10-29',
        'asked_chars': 2500, 'sources': ['2026-10-26-cat-saved_jd-Acme.pdf'], 'summary': _SUMMARY}), encoding='utf-8')
    saved = rating_inputs.save_run_inputs(_system_message(tmp_path, monkeypatch), tmp_path / 'out', cache)

    profile = (tmp_path / 'out' / saved['ideal_role_profile']['path'].rsplit('/', 1)[1]).read_text(encoding='utf-8')
    header = yaml.safe_load(profile.split('---\n')[1])
    assert header['model'] == 'deepseek/x (Morph)' and header['sources'] == ['2026-10-26-cat-saved_jd-Acme.pdf']
    assert profile.endswith('--- END IDEAL ROLE PROFILE ---')
    assert set(saved) == {'job_requirements', 'ideal_role_profile', 'evaluator_instructions', 'system_message'}


def test_a_profile_the_cache_no_longer_holds_says_so(tmp_path, monkeypatch):
    cache = tmp_path / 'ref_cache.yaml'
    cache.write_text(yaml.safe_dump({'summary': 'a newer profile'}), encoding='utf-8')
    saved = rating_inputs.save_run_inputs(_system_message(tmp_path, monkeypatch), tmp_path / 'out', cache)
    profile = Path(saved['ideal_role_profile']['path']).read_text(encoding='utf-8')
    assert 'no longer holds this profile' in profile


def test_an_unchanged_input_is_written_once(tmp_path, monkeypatch):
    message = _system_message(tmp_path, monkeypatch)
    first = rating_inputs._save(tmp_path / 'out', 'job_requirements', message)
    path = tmp_path / 'out' / first['path'].rsplit('/', 1)[1]
    path.write_text('kept', encoding='utf-8')
    rating_inputs._save(tmp_path / 'out', 'job_requirements', message)
    assert path.read_text(encoding='utf-8') == 'kept'


def test_the_rater_user_message_marks_the_posting():
    assert agent.rating_user_message('Title: X').endswith('--- JOB POSTING ---\nTitle: X\n--- END JOB POSTING ---')

