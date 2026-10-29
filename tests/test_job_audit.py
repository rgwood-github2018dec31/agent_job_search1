'''Write Job Audit: choosing the job, and a render failure costing the PDF, never the run.'''

import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest
from agentic_job_search import job_audit, job_trace
from agentic_job_search import tools_generic as tools


def _candidate(job_id: str) -> dict:
    return {'site': 'linkedin', 'job_id': job_id, 'company': 'Acme', 'title': 'Staff AI Engineer',
            'url': f'https://www.linkedin.com/jobs/view/{job_id}/'}


def _trace(candidate: dict, *kinds: str) -> None:
    _trace_obj, token = job_trace.start(tools.job_trace_path(candidate), candidate['site'], candidate['job_id'])
    try:
        for kind in kinds:
            job_trace.record(kind)
    finally:
        job_trace.finish(token)


def test_only_a_job_that_reached_the_rater_is_chosen():
    rated, gated, untraced = _candidate('1'), _candidate('2'), _candidate('3')
    _trace(rated, 'origin', 'extract', 'rating')
    _trace(gated, 'origin', 'extract', 'hard_rule')
    for seed in range(5):
        assert job_audit.choose_job([gated, rated, untraced], None, seed) is rated


def test_a_requested_job_is_audited_even_if_a_gate_stopped_it():
    rated, gated = _candidate('1'), _candidate('2')
    _trace(rated, 'origin', 'rating')
    _trace(gated, 'origin', 'hard_rule')
    assert job_audit.choose_job([rated, gated], '2', seed=0) is gated


def test_a_requested_job_with_no_trace_falls_back_to_a_rated_one(caplog):
    rated = _candidate('1')
    _trace(rated, 'origin', 'rating')
    with caplog.at_level(logging.WARNING):
        assert job_audit.choose_job([rated], '999', seed=0) is rated
    assert any('job 999 was not evaluated' in r.message for r in caplog.records)


def test_the_audit_file_name_carries_id_company_title_and_run_start():
    path = job_audit.audit_path_for(_candidate('4480376713'), datetime(2026, 10, 19, 16, 45, 2))
    assert path.name == 'linkedin_4480376713-acme-staff_ai_engineer-2026-10-19_164502.pdf'


async def test_a_failed_render_is_logged_with_its_output_and_returns_none(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(job_audit, 'JOB_AUDITS_DIR', tmp_path / 'job_audits')
    # A local command that fails, so the test never reaches uv or the network.
    monkeypatch.setattr(job_audit, 'render_command',
                        lambda candidate, run_started, out: [sys.executable, '-c', 'import sys; sys.exit("no reportlab")'])
    with caplog.at_level(logging.WARNING):
        assert await job_audit.render(_candidate('1'), datetime(2026, 10, 19, 16, 45, 2)) is None
    assert any('Job audit render failed' in r.message for r in caplog.records)


def test_the_report_renders_every_traced_step_and_the_exact_message(tmp_path):
    '''Renders a fixture trace through scripts/job_trace_report.py. Needs reportlab, which the
    project does not depend on: `uv run --with reportlab pytest tests/test_job_audit.py`.'''
    pytest.importorskip('reportlab')
    pypdf = pytest.importorskip('pypdf')
    candidate = _candidate('4476053167')
    trace = tmp_path / 'linkedin-4476053167.trace.jsonl'
    _trace_obj, token = job_trace.start(trace, 'linkedin', '4476053167')
    try:
        job_trace.record('origin', candidate=candidate,
                         listing={'query': 'Staff AI Engineer', 'region': 'European Union'}, position='1 of 1')
        job_trace.record('page_read', url=candidate['url'], source='dom', chars=2137, removed=[{'what': '<footer>', 'chars': 900}],
                         dom_chars=160000, removed_chars=900, posting_company='Acme',
                         company_url='https://www.linkedin.com/company/acme/',
                         raw_html=str(tmp_path / 'linkedin-4476053167.html'), raw_html_bytes=152394,
                         raw_posting=str(tmp_path / 'linkedin-4476053167.txt'), raw_posting_bytes=2359, text='PAGE')
        job_trace.record('llm_call', transport='openrouter', call_site='extract_job_page_openrouter', model='m',
                         system='RULES ' * 50, prompt='Company: Acme\n\n--- PAGE TEXT ---\n' + 'x' * 2137 + '\n--- END PAGE TEXT ---',
                         raw_reply=json.dumps({'tool_calls': [{'function': {'name': 'submit_job_extract', 'arguments': '{"title": "T"}'}}],
                                               'usage': {'prompt_tokens': 2000, 'completion_tokens': 30}}),
                         tools_chars=5000, seconds=1.0)
        job_trace.record('llm_call', transport='openrouter', call_site='rate_with_openrouter', model='m',
                         prompt='FULL PROMPT ' * 400, reply='{"rating": 4}', cost_usd=0.001, seconds=1.0,
                         usage={'prompt_tokens': 1234, 'completion_tokens': 56})
        job_trace.record('evidence', call_site='extract', subject='job', source='page', verdicts=[
            {'field': 'location', 'value': 'Gdańsk, Poland', 'quote': 'Gdańsk', 'status': 'regex',
             'span': [0, 6], 'evidence': 'Gdańsk', 'detail': ''}])
        job_trace.record('extract', fields={'title': 'T', 'location': 'Gdańsk, Poland', 'end_client': 'Globex',
                                            'quotes': {'location': 'Gdańsk'}, 'evidence': {'location': 'regex'}})
        job_trace.record('hard_rule', rule='stale', fired=False, date_posted='3 days ago')
        job_trace.record('rating', triage=None, result={'rating': 4}, raw_rating=4, capped_rating=3,
                         cap_reason='hybrid', seconds=1.0)
        job_trace.record('notification', text='3/5 EXACT TELEGRAM TEXT', sent=True)
    finally:
        job_trace.finish(token)
    script = Path(__file__).parent.parent / 'scripts' / 'job_trace_report.py'
    spec = importlib.util.spec_from_file_location('job_trace_report', script)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    out = report.build('4476053167', '2026-10-16', tmp_path / 'audit.pdf', trace)
    text = ''.join(page.extract_text() for page in pypdf.PdfReader(out).pages)
    for expected in ('Decision path', 'stale', 'hybrid', 'rate_with_openrouter', 'EXACT TELEGRAM TEXT',
                     'The Job Trace', 'Step by step', 'Found by', 'European Union',
                     'Posted by', 'Globex — the poster is recruiting for this client',
                     'query=Staff AI Engineer', 'region=European Union',
                     '1. saved full page=linkedin-4476053167.html (149 KB', 'captured=160,000 chars',
                     '2. removed clutter=1 section(s), 900 chars', '3. saved text=linkedin-4476053167.txt (2,137 chars, 2 KB)',
                     'tokens in=1,234 tokens out=56', 'rating=4', 'system message=300 chars (standing instructions)',
                     'user message=', 'chars before PAGE TEXT', '· Company=Acme',
                     '· PAGE TEXT=2,137 chars = linkedin-4476053167.txt from step 2', 'submit_job_extract tool definition=5,000 chars',
                     'total sent=7,492 chars = 2,000 tokens in (about 3.7 chars per token)'):
        assert expected in ' '.join(text.split()), expected
    first_page_links = [a.get_object()['/A']['/URI'] for a in pypdf.PdfReader(out).pages[0].get('/Annots') or []]
    assert first_page_links == [candidate['url'], 'https://www.linkedin.com/company/acme/']
    # The rating is in the Rating row only, not repeated in the Job row.
    first_page = ' '.join(pypdf.PdfReader(out).pages[0].extract_text().split())
    assert 'rating 4/5' not in first_page
    assert first_page.startswith('Full Trace LinkedIn JD: 4476053167 '), first_page[:80]
    # Long lines are folded, so a word pair may straddle a line break; whitespace is not content.
    # The step-by-step table quotes the prompt's head; the verbatim trace section must hold all of it.
    verbatim_section = ' '.join(text.split('The Job Trace — every step, verbatim', 1)[1].split())
    assert verbatim_section.count('FULL PROMPT') == 400, 'the prompt is printed whole, never cut'
