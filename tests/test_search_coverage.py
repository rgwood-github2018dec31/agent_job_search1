'''Measure Search Coverage: email parsing, the harvest stream, classification, replay choice, report.'''

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest
from agentic_job_search import search_coverage as coverage

VIEW = 'View job: https://www.linkedin.com/comm/jobs/view/{id}/?trackingId=abc%3D&trk=eml-x'

ALERT_BODY = (
    'Jobs that match your profile\n\nBased on your title and location. Update:\n'
    'https://www.linkedin.com/comm/in/me/edit/topcard/?lipi=x\n\n'
    'Senior Data Scientist\nAcme Analytics\nVancouver\n' + VIEW.format(id='4477093554') + '\n\n'
    '---------------------------------------------------------\n\n'
    'Applied AI Scientist\nNorthwind\nToronto, ON\n\n12 school alumni\n' + VIEW.format(id='4490687966') + '\n'
)
APPLIED_BODY = (
    'Your application was sent to Contoso\n\n'
    'Lead Data Scientist\nContoso\nPoland\n' + VIEW.format(id='4493766856') + '\n\n'
    '---------------------------------------------------------\n\n'
    'Applied on November 24, 2026----------------------------------------------------------------\n'
    '        Now, take these next steps for more success\n\nView similar jobs you may be interested in\n\n'
    'Architect, Machine Learning\nFabrikam\nCanada\n\n47 school alumni\n' + VIEW.format(id='4433975263') + '\n'
)


def message(body: str, subject: str, sent: str = '2026-11-23T21:27:12Z') -> dict:
    return {'id': 'm1', 'subject': subject, 'date': sent, 'plaintextBody': body}


def test_parse_alert_message_reads_every_job_with_its_labels():
    jobs = coverage.parse_alert_message(message(ALERT_BODY, 'Senior Data Scientist at Acme'), 't1')
    assert [(job.job_id, job.title, job.company, job.location) for job in jobs] == [
        ('4477093554', 'Senior Data Scientist', 'Acme Analytics', 'Vancouver'),
        ('4490687966', 'Applied AI Scientist', 'Northwind', 'Toronto, ON'),
    ]
    assert {job.email_kind for job in jobs} == {coverage.EMAIL_KIND_RECOMMENDED}
    assert jobs[0].email_date == '2026-11-23'


def test_application_email_marks_only_the_first_job_applied():
    jobs = coverage.parse_alert_message(message(APPLIED_BODY, 'Pat, your application was sent to Contoso'), 't2')
    assert [(job.job_id, job.email_kind) for job in jobs] == [
        ('4493766856', coverage.EMAIL_KIND_APPLIED), ('4433975263', coverage.EMAIL_KIND_RECOMMENDED)]
    assert (jobs[1].title, jobs[1].company, jobs[1].location) == ('Architect, Machine Learning', 'Fabrikam', 'Canada')


def test_job_with_unreadable_labels_keeps_its_id():
    jobs = coverage.parse_alert_message(message('Hello\n' + VIEW.format(id='123') + '\n', 'x'), 't3')
    assert [(job.job_id, job.title) for job in jobs] == [('123', '')]


def test_message_without_body_or_date_raises_with_context():
    with pytest.raises(ValueError, match='no plaintextBody'):
        coverage.parse_alert_message({'id': 'm9', 'date': '2026-11-23T00:00:00Z'}, 't9')
    with pytest.raises(ValueError, match='unreadable date'):
        coverage.parse_alert_message({'id': 'm9', 'date': 'yesterday', 'plaintextBody': ''}, 't9')


def test_merge_prefers_applied_then_earliest():
    early = coverage.AlertJob('1', 't', 'c', 'l', '2026-11-01', coverage.EMAIL_KIND_RECOMMENDED, 'a', 's')
    late = coverage.AlertJob('1', 't', 'c', 'l', '2026-11-05', coverage.EMAIL_KIND_RECOMMENDED, 'b', 's')
    applied = coverage.AlertJob('1', 't', 'c', 'l', '2026-11-09', coverage.EMAIL_KIND_APPLIED, 'c', 's')
    assert coverage.merge_alert_jobs([late, early])['1'] == early
    assert coverage.merge_alert_jobs([early, applied, late])['1'] == applied


def stream(*events: dict) -> list[str]:
    return [json.dumps(event) for event in events]


def tool_use(use_id: str, name: str, tool_input: dict) -> dict:
    return {'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'id': use_id, 'name': name, 'input': tool_input}]}}


def tool_result(use_id: str, content: str, is_error: bool = False) -> dict:
    return {'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': use_id, 'content': content, 'is_error': is_error}]}}


RESULT_OK = {'type': 'result', 'subtype': 'success', 'is_error': False, 'total_cost_usd': 0.005}
INIT_WITH_GMAIL = {'type': 'system', 'subtype': 'init',
                   'tools': [coverage.GMAIL_TOOL_SEARCH_THREADS, coverage.GMAIL_TOOL_GET_THREAD]}
INIT_WITHOUT_TOOLS = {'type': 'system', 'subtype': 'init', 'tools': []}


def test_parse_harvest_stream_reads_threads_from_tool_results():
    thread = {'id': 't1', 'messages': [message(ALERT_BODY, 's')]}
    result = coverage.parse_harvest_stream(stream(
        INIT_WITH_GMAIL,
        tool_use('u1', coverage.GMAIL_TOOL_SEARCH_THREADS, {'query': 'q'}),
        tool_result('u1', json.dumps({'threads': [{'id': 't1'}, {'id': 't2'}]})),
        tool_use('u2', coverage.GMAIL_TOOL_GET_THREAD, {'threadId': 't1'}),
        tool_result('u2', json.dumps(thread)),
        RESULT_OK,
    ))
    assert result.listed_thread_ids == {'t1', 't2'}
    assert result.threads == {'t1': thread}
    assert result.cost_usd == pytest.approx(0.005)
    assert result.problems == []
    assert result.gmail_tools_loaded and result.searched


def test_parse_harvest_stream_reads_a_result_saved_to_a_file(tmp_path: Path):
    thread = {'id': 't7', 'messages': []}
    saved = tmp_path / 'tool-results' / 'big.txt'
    saved.parent.mkdir()
    saved.write_text(json.dumps(thread), encoding='utf-8')
    result = coverage.parse_harvest_stream(stream(
        tool_use('u1', coverage.GMAIL_TOOL_GET_THREAD, {'threadId': 't7'}),
        tool_result('u1', f'Error: result exceeds maximum allowed tokens. Output has been saved to {saved}.'),
        RESULT_OK,
    ))
    assert result.threads == {'t7': thread}


def test_parse_harvest_stream_reports_errors_bad_json_and_unfinished_sessions():
    result = coverage.parse_harvest_stream(stream(
        tool_use('u1', coverage.GMAIL_TOOL_GET_THREAD, {'threadId': 'tx'}),
        tool_result('u1', 'quota exceeded', is_error=True),
        tool_use('u2', coverage.GMAIL_TOOL_GET_THREAD, {'threadId': 'ty'}),
        tool_result('u2', '<html>not json'),
        tool_use('u3', 'mcp__claude_ai_Gmail__send_message', {}),
        tool_result('u3', '{}'),
    ))
    text = '\n'.join(result.problems)
    assert 'quota exceeded' in text and "'threadId': 'tx'" in text
    assert 'not JSON' in text and '<html>not json' in text
    assert 'unexpected tool mcp__claude_ai_Gmail__send_message' in text
    assert 'no result event' in text


def test_harvest_command_allows_only_the_gmail_read_tools():
    command = coverage.harvest_command('claude', 'prompt')
    allowed = command[command.index('--allowedTools') + 1:command.index('--permission-mode')]
    assert allowed == [coverage.GMAIL_TOOL_SEARCH_THREADS, coverage.GMAIL_TOOL_GET_THREAD]
    assert command[command.index('--tools') + 1] == ''
    assert command[command.index('--permission-mode') + 1] == 'dontAsk'
    assert command[command.index('--setting-sources') + 1] == 'user'   # the only source that loads the connectors
    assert json.loads(command[command.index('--settings') + 1]) == {'disableAllHooks': True}


class FakeCompleted:
    def __init__(self, lines: list[str]):
        self.stdout, self.stderr, self.returncode = '\n'.join(lines), '', 0


def test_run_harvest_starts_again_when_the_gmail_tools_are_missing(tmp_path, monkeypatch):
    sessions = iter([
        stream(INIT_WITHOUT_TOOLS, RESULT_OK),
        stream(INIT_WITH_GMAIL, tool_use('u1', coverage.GMAIL_TOOL_SEARCH_THREADS, {}),
               tool_result('u1', json.dumps({'threads': [{'id': 't1'}]})), RESULT_OK),
    ])
    monkeypatch.setattr(coverage, 'SEARCH_COVERAGE_DIR', tmp_path)
    monkeypatch.setattr(coverage.subprocess, 'run', lambda *args, **kwargs: FakeCompleted(next(sessions)))
    monkeypatch.setattr(coverage.time, 'sleep', lambda seconds: None)
    result = coverage.run_harvest('claude', date(2026, 11, 17), set())
    assert result.searched and result.cost_usd == pytest.approx(0.01)
    assert result.problems == ["1 listed thread(s) were never fetched: ['t1']"]


def test_run_harvest_raises_when_no_session_can_search(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage, 'SEARCH_COVERAGE_DIR', tmp_path)
    monkeypatch.setattr(coverage.subprocess, 'run', lambda *args, **kwargs: FakeCompleted(stream(INIT_WITHOUT_TOOLS, RESULT_OK)))
    monkeypatch.setattr(coverage.time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='Gmail connector tools were not in the session'):
        coverage.run_harvest('claude', date(2026, 11, 17), set())


def test_alert_jobs_round_trip(tmp_path: Path):
    jobs = coverage.parse_alert_message(message(APPLIED_BODY, 'your application was sent to Contoso'), 't2')
    coverage.write_alert_jobs(tmp_path / 'linkedin_alerts-1.tsv', jobs)
    assert coverage.read_alert_jobs(tmp_path) == jobs


def write_processed(directory: Path, job_id: str, company: str, title: str) -> None:
    (directory / f'job_posting-linkedin-{job_id}.yaml').write_text(
        f"company: {company}\ndescription: {title}\njob_id: '{job_id}'\nsite: linkedin\n", encoding='utf-8')


@pytest.fixture
def pipeline_record(tmp_path: Path) -> coverage.PipelineRecord:
    processed = tmp_path / 'processed_jobs'
    processed.mkdir()
    write_processed(processed, '100', 'Acme Analytics', 'Senior Data Scientist')
    write_processed(processed, '200', 'Northwind', 'Applied AI Scientist')
    write_processed(processed, '300', 'Globex', 'Staff ML Engineer')
    saved = tmp_path / 'saved_jobs-2026Nov20'
    saved.mkdir()
    (saved / 'job_posting-100-rating_4-acme-x-1.md').write_text('', encoding='utf-8')
    audit = tmp_path / 'audit_logs'
    audit.mkdir()
    (audit / 'audit-2026Nov20-1.md').write_text(
        '## 2. Search queries\n\n- `Staff AI Engineer` — 5 listing(s)\n\n## 3. Jobs\n'
        '| Northwind | Applied AI Scientist | already_applied | x | https://www.linkedin.com/jobs/view/200/ |\n',
        encoding='utf-8')
    return coverage.load_pipeline_record(tmp_path, processed)


def alert(job_id: str, company: str, title: str, location: str = 'Canada',
          kind: str = coverage.EMAIL_KIND_RECOMMENDED, when: str = '2026-11-20') -> coverage.AlertJob:
    return coverage.AlertJob(job_id, title, company, location, when, kind, 't', 's')


def test_load_pipeline_record_and_latest_queries(pipeline_record, tmp_path):
    assert pipeline_record.processed_ids == {'100', '200', '300'}
    assert pipeline_record.ratings == {'100': 4}
    assert pipeline_record.audit_statuses == {'200': 'already_applied'}
    assert coverage.latest_search_queries(tmp_path) == ['Staff AI Engineer']


def test_classify_jobs_buckets(pipeline_record):
    async def classify(text: str) -> dict:
        return {'Berlin, Germany': {'countries': ['Germany']}, 'Austin, TX': {'countries': ['United States']}}.get(
            text, {'countries': ['Canada']})

    jobs = [
        alert('100', 'Acme Analytics', 'Senior Data Scientist'),
        alert('200', 'Northwind', 'Applied AI Scientist'),
        alert('999', 'GLOBEX', 'Staff ML engineer'),
        alert('400', 'Initech', 'AI Architect', 'Austin, TX'),
        alert('500', 'Initech', 'AI Architect', 'Berlin, Germany'),
        alert('600', 'Hooli', 'Principal Engineer', ''),
    ]
    classified = asyncio.run(coverage.classify_jobs(jobs, pipeline_record, ['Canada', 'European Union'], classify))
    assert [(item.job.job_id, item.bucket, item.rating, item.detail) for item in classified] == [
        ('100', coverage.BUCKET_SEEN_RATED, 4, ''),
        ('200', coverage.BUCKET_SEEN_NOT_RATED, None, 'already_applied'),
        ('999', coverage.BUCKET_SIBLING_SEEN, None, 'sibling 300'),
        ('400', coverage.BUCKET_OUT_OF_SEARCH_REGION, None, 'United States'),
        ('500', coverage.BUCKET_NEVER_SEEN, None, ''),
        ('600', coverage.BUCKET_NEVER_SEEN, None, ''),   # no location: never left out
    ]
    assert coverage.coverage_ratio(classified) == (3, 5)


def test_unknown_or_broad_location_counts_as_in_region():
    assert coverage.in_search_regions({'countries': []}, ['Canada'])
    assert coverage.in_search_regions({'countries': ['United States'], 'broad_area': True}, ['Canada'])
    assert not coverage.in_search_regions({'countries': ['United States']}, ['Canada', 'European Union'])
    assert coverage.in_search_regions({'countries': ['Portugal']}, ['European Union'])


def test_choose_replay_jobs_puts_applied_first_then_newest_and_caps():
    never = [
        coverage.ClassifiedJob(alert('1', 'a', 'x', when='2026-11-18'), coverage.BUCKET_NEVER_SEEN),
        coverage.ClassifiedJob(alert('2', 'b', 'x', when='2026-11-21'), coverage.BUCKET_NEVER_SEEN),
        coverage.ClassifiedJob(alert('3', 'c', 'x', kind=coverage.EMAIL_KIND_APPLIED, when='2026-11-17'),
                               coverage.BUCKET_NEVER_SEEN),
        coverage.ClassifiedJob(alert('4', 'd', 'x'), coverage.BUCKET_SEEN_RATED),
        coverage.ClassifiedJob(alert('5', 'e', 'x', when='2026-11-22'), coverage.BUCKET_NEVER_SEEN),
    ]
    chosen, left_out = coverage.choose_replay_jobs(never, already_replayed={'5'}, cap=2)
    assert [job.job_id for job in chosen] == ['3', '2']
    assert left_out == 1


def test_read_replay_summary(tmp_path: Path):
    (tmp_path / 'summary.tsv').write_text(
        'job_id\ttitle\tcompany\toutcome\trating\tsummary\n'
        '11\tA\tB\trated\t4\tgood\n12\tC\tD\textract_failed\t\tpage extraction failed\n', encoding='utf-8')
    assert coverage.read_replay_summary(tmp_path) == {
        '11': {'outcome': 'rated', 'rating': 4, 'summary': 'good'},
        '12': {'outcome': 'extract_failed', 'rating': None, 'summary': 'page extraction failed'},
    }


def test_report_and_summary_name_the_missed_good_job(tmp_path: Path):
    classified = [
        coverage.ClassifiedJob(alert('1', 'Acme', 'Staff AI Engineer'), coverage.BUCKET_NEVER_SEEN),
        coverage.ClassifiedJob(alert('2', 'Acme', 'Staff AI Engineer'), coverage.BUCKET_NEVER_SEEN),
        coverage.ClassifiedJob(alert('3', 'Hooli', 'AI Architect'), coverage.BUCKET_SEEN_RATED, 3),
    ]
    replayed = {'1': {'outcome': 'rated', 'rating': 5}, '2': {'outcome': 'rated', 'rating': 2}}
    report = coverage.build_report(classified, replayed, ['AI Architect'], (date(2026, 11, 16), date(2026, 11, 23)), [])
    assert '**1 of 3**' in report
    assert 'https://www.linkedin.com/jobs/view/1/' in report.split('## Never seen')[0]
    assert '- title "staff ai engineer": 2' in report and '- company "Acme": 2' in report
    summary = coverage.build_telegram_summary(classified, replayed, tmp_path / 'r.md', [])
    assert 'Missed, rated 5: Staff AI Engineer at Acme' in summary
    assert 'rated 2' not in summary


def test_window_for():
    assert coverage.window_for(date(2026, 11, 25), 8, 1) == (date(2026, 11, 17), date(2026, 11, 24))
