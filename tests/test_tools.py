"""Tests for the job search agent."""

import asyncio
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pytest
import requests
import yaml
from agentic_job_search import (
    agent,
    config,
    extract_openrouter,
    linkedin_page,
    location,
    location_review,
    preferences,
    salary,
    scrape_openrouter,
    text_budget,
    triage,
)
from agentic_job_search import tools_generic as tools
from claude_agent_sdk import ResultMessage, SdkMcpTool
from utils_tools_n_agents_common.models import (
    ANTHROPIC_MODEL_NAME_LOW,
    ANTHROPIC_MODEL_NAME_MEDIUM,
    OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC,
    OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE,
    OPENROUTER_MODEL_NAME_SCRAPER,
)

# ---------------------------------------------------------------------------
# underscorify
# ---------------------------------------------------------------------------

def test_underscorify_basic():
    assert tools.underscorify('Hello World') == 'hello_world'


def test_underscorify_special_chars():
    assert tools.underscorify('Shopify Inc.') == 'shopify_inc'


def test_underscorify_consecutive_separators():
    assert tools.underscorify('Senior  Software---Engineer') == 'senior_software_engineer'


def test_underscorify_leading_trailing():
    assert tools.underscorify('  leading and trailing  ') == 'leading_and_trailing'


def test_underscorify_numbers():
    assert tools.underscorify('GPT-4 Engineer') == 'gpt_4_engineer'


# ---------------------------------------------------------------------------
# do_save_job_posting
# ---------------------------------------------------------------------------

async def test_save_job_posting_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)

    result = await tools.do_save_job_posting(
        company='Shopify',
        description='Senior Software Engineer',
        rating=4,
        content='# Senior Software Engineer at Shopify\n\nGreat remote role.',
    )

    assert not result.get('isError')
    saved_dirs = list(tmp_path.glob('saved_jobs-*'))
    assert len(saved_dirs) == 1

    files = list(saved_dirs[0].glob('job_posting-noid-rating_4-shopify-senior_software_engineer-*.md'))
    assert len(files) == 1
    assert files[0].read_text() == '# Senior Software Engineer at Shopify\n\nGreat remote role.'


async def test_save_job_posting_result_contains_path(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)

    result = await tools.do_save_job_posting(
        company='Acme',
        description='Engineer',
        rating=2,
        content='content',
    )

    text = result['content'][0]['text']
    assert 'saved_jobs-' in text
    assert 'job_posting-noid-rating_2-acme-engineer-' in text


async def test_save_job_posting_creates_daily_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)

    for rating in (3, 5):
        await tools.do_save_job_posting(
            company='Corp',
            description='Role',
            rating=rating,
            content=f'content {rating}',
        )

    saved_dirs = list(tmp_path.glob('saved_jobs-*'))
    assert len(saved_dirs) == 1  # same day → same dir
    assert len(list(saved_dirs[0].glob('*.md'))) == 2


async def test_save_job_posting_caps_long_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)

    long_description = 'triaged out ' + 'this is an extremely long triage reason sentence ' * 10
    result = await tools.do_save_job_posting(
        company='A Very Long Company Name That Goes On And On Incorporated',
        description=long_description,
        rating=1,
        content='content',
        job_id='4442638114',
    )

    assert not result.get('isError')
    files = list(tmp_path.glob('saved_jobs-*/job_posting-4442638114-rating_1-*.md'))
    assert len(files) == 1
    assert len(files[0].name.encode()) < 255


# ---------------------------------------------------------------------------
# do_update_job_requirements
# ---------------------------------------------------------------------------

async def test_update_job_requirements_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'JOB_REQUIREMENTS_PATH', tmp_path / 'JOB_REQUIREMENTS.md')

    content = '# Requirements\n\n- Remote only\n- Canada or EU\n'
    result = await tools.do_update_job_requirements(content)

    assert not result.get('isError')
    assert (tmp_path / 'JOB_REQUIREMENTS.md').read_text() == content


async def test_update_job_requirements_returns_content(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'JOB_REQUIREMENTS_PATH', tmp_path / 'JOB_REQUIREMENTS.md')

    content = '# Requirements\n\n- Senior only\n'
    result = await tools.do_update_job_requirements(content)

    # New contents must be in the tool result so the agent has them in context
    assert content in result['content'][0]['text']


async def test_update_job_requirements_overwrites(tmp_path, monkeypatch):
    path = tmp_path / 'JOB_REQUIREMENTS.md'
    monkeypatch.setattr(tools, 'JOB_REQUIREMENTS_PATH', path)

    await tools.do_update_job_requirements('first version')
    await tools.do_update_job_requirements('second version')

    assert path.read_text() == 'second version'


# ---------------------------------------------------------------------------
# log_run_cost
# ---------------------------------------------------------------------------


def test_log_run_cost_writes_jsonl_line(tmp_path):
    log_path = tmp_path / 'cost_log.jsonl'
    record = {'mode': 'non-interactive', 'total_cost': 1.2345}

    tools.log_run_cost(record, log_path=log_path)

    lines = log_path.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_log_run_cost_appends_multiple_runs(tmp_path):
    log_path = tmp_path / 'cost_log.jsonl'

    tools.log_run_cost({'run': 1}, log_path=log_path)
    tools.log_run_cost({'run': 2}, log_path=log_path)

    lines = log_path.read_text(encoding='utf-8').splitlines()
    assert [json.loads(line) for line in lines] == [{'run': 1}, {'run': 2}]


def test_log_run_cost_creates_parent_dirs(tmp_path):
    log_path = tmp_path / 'nested' / 'cost_log.jsonl'

    tools.log_run_cost({'run': 1}, log_path=log_path)

    assert log_path.exists()


def test_new_stage_stats_zeroed():
    stats = agent.new_stage_stats()
    assert stats == {
        'cost': 0.0,
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_read_input_tokens': 0,
        'cache_creation_input_tokens': 0,
        'session_costs': {},
    }


def test_accumulate_stage_stats_sums_usage_and_cost():
    """Two independent requests (separate sessions) sum in both usage and cost."""
    usage = {
        'input_tokens': 100,
        'output_tokens': 50,
        'cache_read_input_tokens': 10,
        'cache_creation_input_tokens': 5,
    }
    stats = agent.new_stage_stats()
    agent.accumulate_stage_stats(stats, _result_msg('req-1', 0.01, **usage))
    agent.accumulate_stage_stats(stats, _result_msg('req-2', 0.01, **usage))

    assert agent.public_stage_stats({'s': stats})['s'] == {
        'cost': pytest.approx(0.02),
        'input_tokens': 200,
        'output_tokens': 100,
        'cache_read_input_tokens': 20,
        'cache_creation_input_tokens': 10,
    }


# ---------------------------------------------------------------------------
# do_check_and_record_job
# ---------------------------------------------------------------------------


def _recent_date() -> str:
    return (date.today() - timedelta(days=5)).isoformat()


def _old_date() -> str:
    return (date.today() - timedelta(days=30)).isoformat()


async def test_check_and_record_job_new(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    result = await tools.do_check_and_record_job(
        'linkedin', '1234567890', 'Shopify', 'Senior Engineer', date_posted=_recent_date()
    )

    assert result['content'][0]['text'] == 'new'
    files = list((tmp_path / 'processed_jobs').glob('job_posting-linkedin-1234567890-*.yaml'))
    assert len(files) == 1


async def test_check_and_record_job_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    await tools.do_check_and_record_job(
        'linkedin', '1234567890', 'Shopify', 'Senior Engineer', date_posted=_recent_date()
    )
    result = await tools.do_check_and_record_job(
        'linkedin', '1234567890', 'Shopify', 'Senior Engineer', date_posted=_recent_date()
    )

    assert result['content'][0]['text'] == 'already_processed'


async def test_check_and_record_job_too_old(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    result = await tools.do_check_and_record_job(
        'linkedin', '9999999999', 'OldCo', 'Stale Role', date_posted=_old_date()
    )

    assert result['content'][0]['text'] == 'too_old'
    assert not list((tmp_path / 'processed_jobs').glob('*.yaml'))


async def test_check_and_record_job_different_sites(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    r1 = await tools.do_check_and_record_job(
        'linkedin', '111', 'Corp', 'Engineer', date_posted=_recent_date()
    )
    r2 = await tools.do_check_and_record_job(
        'indeed', '111', 'Corp', 'Engineer', date_posted=_recent_date()
    )

    assert r1['content'][0]['text'] == 'new'
    assert r2['content'][0]['text'] == 'new'


async def test_check_and_record_job_yaml_content(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    posted = _recent_date()
    await tools.do_check_and_record_job(
        'linkedin', '5555555555', 'Stripe', 'Staff Engineer', date_posted=posted
    )

    files = list((tmp_path / 'processed_jobs').glob('*.yaml'))
    data = yaml.safe_load(files[0].read_text())
    assert data['site'] == 'linkedin'
    assert data['job_id'] == '5555555555'
    assert data['date_posted'] == posted
    assert data['date_recorded'] == date.today().isoformat()
    assert data['company'] == 'Stripe'
    assert data['description'] == 'Staff Engineer'


# ---------------------------------------------------------------------------
# _requires_current_us_auth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('text', [
    'Must be authorized to work in the US without sponsorship',
    'No visa sponsorship available for this role',
    'We are not able to sponsor work visas',
    'We cannot sponsor or transfer visas',
    'Sponsorship is not available for this position',
    'Will not sponsor applicants for work visas',
    'US citizens and permanent residents only',
    'Currently authorized to work in the United States',
    'This position does not provide visa sponsorship',
    'Employment authorization without sponsorship required',
    'must be legally authorized to work in the united states',
    'Must be authorized to work in the U.S.',
    'U.S. citizens and lawful permanent residents only',
])
def test_requires_current_us_auth_matches(text):
    assert tools._requires_current_us_auth(text), f'Expected match for: {text!r}'


@pytest.mark.parametrize('text', [
    'Senior ML Engineer — Remote Canada',
    'Principal Data Scientist — EU remote',
    'We are open to visa sponsorship for exceptional candidates',
    'Staff AI Engineer at Shopify',
    'Sponsorship available for the right candidate',
    'Remote role, open to all locations',
])
def test_requires_current_us_auth_no_false_positives(text):
    assert not tools._requires_current_us_auth(text), f'Expected no match for: {text!r}'


async def test_check_and_record_job_auth_required(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    result = await tools.do_check_and_record_job(
        'linkedin', '7777777777', 'AcmeUS', 'Senior Engineer',
        date_posted=_recent_date(),
        content='Must be authorized to work in the US without sponsorship.',
    )

    assert result['content'][0]['text'] == 'auth_required'
    assert not list((tmp_path / 'processed_jobs').glob('*.yaml'))


async def test_check_and_record_job_auth_not_triggered_for_canada(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    result = await tools.do_check_and_record_job(
        'linkedin', '8888888888', 'Shopify', 'Staff Engineer',
        date_posted=_recent_date(),
        content='Remote role open to candidates in Canada. We welcome all applicants.',
    )

    assert result['content'][0]['text'] == 'new'


# ---------------------------------------------------------------------------
# load_processed_jobs
# ---------------------------------------------------------------------------

async def test_load_processed_jobs_from_md_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    await tools.do_save_job_posting('Shopify', 'Engineer', 4, 'content', job_id='3859234876')
    await tools.do_save_job_posting('Acme', 'Designer', 2, 'content', job_id='1122334455')

    tools.load_processed_jobs()
    assert ('linkedin', '3859234876') in tools._processed_jobs
    assert ('linkedin', '1122334455') in tools._processed_jobs


async def test_load_processed_jobs_from_yaml_files(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    (tmp_path / 'processed_jobs').mkdir()
    (tmp_path / 'processed_jobs' / 'job_posting-indeed-42-2026Apr10-000-co-role.yaml').write_text(
        yaml.dump({'site': 'indeed', 'job_id': '42', 'date_posted': '2026-04-05',
                   'date_recorded': '2026-04-10', 'company': 'co', 'description': 'role'})
    )

    tools.load_processed_jobs()
    assert ('indeed', '42') in tools._processed_jobs


async def test_load_processed_jobs_combines_both(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    await tools.do_save_job_posting('Corp', 'Role', 3, 'content', job_id='111')

    (tmp_path / 'processed_jobs').mkdir()
    (tmp_path / 'processed_jobs' / 'job_posting-indeed-999-2026Apr10-000-co-role.yaml').write_text(
        yaml.dump({'site': 'indeed', 'job_id': '999', 'date_posted': '2026-04-05',
                   'date_recorded': '2026-04-10', 'company': 'co', 'description': 'role'})
    )

    tools.load_processed_jobs()
    assert ('linkedin', '111') in tools._processed_jobs
    assert ('indeed', '999') in tools._processed_jobs


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

def test_build_system_prompt_includes_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', tmp_path / 'JOB_REQUIREMENTS.md')

    (tmp_path / 'R_Garth_Wood-resume-2026Apr08v1.md').write_text('# Garth Wood\n\nExperienced engineer.')

    prompt = agent.build_system_prompt(interactive=True)
    assert 'Garth Wood' in prompt
    assert 'RESUME' in prompt


def test_build_system_prompt_includes_job_requirements(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    req_path = tmp_path / 'JOB_REQUIREMENTS.md'
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', req_path)

    req_path.write_text('# Requirements\n\n- Remote only')

    prompt = agent.build_system_prompt(interactive=True)
    assert 'Remote only' in prompt
    assert 'JOB_REQUIREMENTS.md' in prompt


def test_build_system_prompt_warns_when_no_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', tmp_path / 'JOB_REQUIREMENTS.md')

    agent.build_system_prompt(interactive=True)

    assert 'Warning' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Live agent tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.live_agent_claude
async def test_live_agent_calls_save_job_posting(tmp_path, monkeypatch):
    """Agent uses the save_job_posting tool when instructed to save a job."""
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)

    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        system_prompt=(
            'You are a job search assistant. '
            'When asked to save a job, use the save_job_posting tool exactly once.'
        ),
        mcp_servers={'job_search': tools.make_job_search_server(interactive=False)},
        # bypassPermissions, as every non-interactive site in agent.py uses: acceptEdits
        # auto-approves file edits only, so an in-process MCP tool call is DENIED under it
        # (verified: the call arrives, then shows up in ResultMessage.permission_denials).
        permission_mode='bypassPermissions',
        # Without these the run also loads every global MCP server, burying save_job_posting
        # among dozens of unrelated tools, plus the repo CLAUDE.md. Same leak as the src sites.
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
    )

    async for _ in query(
        prompt=(
            "Save this job posting using the save_job_posting tool: "
            "company='TestCorp', description='Staff Engineer', rating=4, "
            "content='# Staff Engineer at TestCorp\n\nFully remote, Canada OK.'"
        ),
        options=options,
    ):
        pass

    saved_dirs = list(tmp_path.glob('saved_jobs-*'))
    assert len(saved_dirs) == 1, 'Expected a saved_jobs-* directory to be created'
    # Filename order is job_posting-{id}-rating_{n}-{company}-...; the old glob here had company
    # before rating and so could never have matched even once the tool was called.
    files = list(saved_dirs[0].glob('job_posting-*-rating_4-testcorp-*.md'))
    assert len(files) == 1, f'Expected one saved job file, found: {list(saved_dirs[0].iterdir())}'


# ---------------------------------------------------------------------------
# company_matches_applied
# ---------------------------------------------------------------------------

async def test_company_matches_applied_empty_dict_skips_llm(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {})
    sdk_called = []

    async def fake_sdk_query(**kwargs):
        sdk_called.append(True)
        return
        yield  # make it an async generator

    monkeypatch.setattr(tools, 'sdk_query', fake_sdk_query)
    result = await tools.company_matches_applied('Shopify')
    assert result is None
    assert not sdk_called


def _make_sdk_mock(monkeypatch, structured_output: dict):
    """Patch sdk_query so it yields a ResultMessage with structured_output."""
    from claude_agent_sdk import ResultMessage

    async def fake_sdk_query(**kwargs):
        yield ResultMessage(
            subtype='success',
            duration_ms=100,
            duration_api_ms=100,
            is_error=False,
            num_turns=1,
            session_id='fake-session',
            structured_output=structured_output,
        )

    monkeypatch.setattr(tools, 'sdk_query', fake_sdk_query)


async def test_company_matches_applied_returns_filename_on_yes(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {'Shopify': 'shopify_jd.pdf'})
    _make_sdk_mock(monkeypatch, {'matches': True, 'matched_company_name': 'Shopify'})
    result = await tools.company_matches_applied('Shopify Inc.')
    assert result == 'shopify_jd.pdf'


async def test_company_matches_applied_returns_none_on_no(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {'Shopify': 'shopify_jd.pdf'})
    _make_sdk_mock(monkeypatch, {'matches': False, 'matched_company_name': ''})
    result = await tools.company_matches_applied('Acme Corp')
    assert result is None


async def test_check_and_record_job_skips_applied_company(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())

    async def fake_matches(candidate):
        return 'shopify_job.pdf'

    monkeypatch.setattr(tools, 'company_matches_applied', fake_matches)

    result = await tools.do_check_and_record_job(
        'linkedin', '9999', 'Shopify Inc.', 'Staff Engineer', date_posted=_recent_date()
    )
    assert result['content'][0]['text'] == 'already_applied'
    assert not list((tmp_path / 'processed_jobs').glob('*.yaml'))


# ---------------------------------------------------------------------------
# applied-jobs corpus: ingest, load, horizon, agency handling
# ---------------------------------------------------------------------------

def _write_pdf(path: Path, mtime_date: date | None = None) -> Path:
    """Create a placeholder PDF, optionally stamping its mtime to a given date."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'%PDF-1.4 fake')
    if mtime_date is not None:
        ts = datetime.combine(mtime_date, datetime.min.time()).timestamp()
        os.utime(path, (ts, ts))
    return path


def _index_entry(applied_date: date, mtime: float, **overrides) -> dict:
    entry = {
        'applied_date': applied_date,
        'mtime': mtime,
        'text': 'job description text',
        'company': 'CachedCorp',
        'job_title': 'Staff AI Engineer',
        'is_agency': False,
        'end_client': '',
    }
    entry.update(overrides)
    return entry


def _no_legacy_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, 'LEGACY_DOWNLOADS_CACHE_PATH', tmp_path / 'missing_legacy_cache.yaml')


def _reference_texts() -> list[str]:
    """Each in-horizon reference job's pages joined, newest first, for readable assertions."""
    return ['\n\n'.join(pages) for pages in tools._reference_job_pages]


async def test_ingest_date_prefixes_from_mtime_and_preserves_it(tmp_path, monkeypatch):
    _no_legacy_cache(monkeypatch, tmp_path)
    save_dir = tmp_path / 'saved_pdfs'
    applied = tmp_path / 'applied_jobs'
    applied_date = date(2026, 5, 13)
    source = _write_pdf(save_dir / 'cat-saved_jd-acme_job.pdf', applied_date)
    source_mtime = source.stat().st_mtime

    moved = await tools.ingest_save_dir_applied_pdfs(save_dir=save_dir, applied_to_dir=applied)

    assert moved == 1
    assert not source.exists(), 'source PDF should have moved, not been copied'
    destination = applied / '2026-05-13-cat-saved_jd-acme_job.pdf'
    assert destination.exists()
    assert abs(destination.stat().st_mtime - source_mtime) < 1.0, 'mtime should survive the move'


async def test_ingest_rolls_back_partial_move(tmp_path, monkeypatch):
    """A copy that succeeds but whose source unlink fails must not leave a duplicate behind."""
    _no_legacy_cache(monkeypatch, tmp_path)
    save_dir = tmp_path / 'saved_pdfs'
    applied = tmp_path / 'applied_jobs'
    source = _write_pdf(save_dir / 'cat-saved_jd-acme_job.pdf', date(2026, 5, 13))

    def fake_move(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes())  # copy succeeds, source stays
        raise PermissionError('Operation not permitted')

    monkeypatch.setattr(tools.shutil, 'move', fake_move)

    moved = await tools.ingest_save_dir_applied_pdfs(save_dir=save_dir, applied_to_dir=applied)

    assert moved == 0
    assert source.exists(), 'source must remain when the move failed'
    assert not (applied / '2026-05-13-cat-saved_jd-acme_job.pdf').exists(), 'partial copy rolled back'


async def test_ingest_dry_run_moves_nothing(tmp_path, monkeypatch):
    _no_legacy_cache(monkeypatch, tmp_path)
    save_dir = tmp_path / 'saved_pdfs'
    applied = tmp_path / 'applied_jobs'
    source = _write_pdf(save_dir / 'cat-saved_jd-acme_job.pdf', date(2026, 5, 13))

    moved = await tools.ingest_save_dir_applied_pdfs(
        save_dir=save_dir, applied_to_dir=applied, dry_run=True
    )

    assert moved == 1
    assert source.exists()
    assert not applied.exists()


async def test_ingest_is_idempotent_for_already_prefixed_file(tmp_path, monkeypatch):
    _no_legacy_cache(monkeypatch, tmp_path)
    save_dir = tmp_path / 'saved_pdfs'
    applied = tmp_path / 'applied_jobs'
    # An already-ingested file re-saved into the save directory keeps its original date, and a
    # second pass must not re-date it to today.
    _write_pdf(save_dir / '2026-04-13-cat-saved_jd-acme_job.pdf', date(2026, 7, 1))

    await tools.ingest_save_dir_applied_pdfs(save_dir=save_dir, applied_to_dir=applied)
    assert (applied / '2026-04-13-cat-saved_jd-acme_job.pdf').exists()

    moved_again = await tools.ingest_save_dir_applied_pdfs(save_dir=save_dir, applied_to_dir=applied)
    assert moved_again == 0
    assert list(applied.glob('*.pdf')) == [applied / '2026-04-13-cat-saved_jd-acme_job.pdf']


async def test_ingest_prefers_index_date_over_clobbered_mtime(tmp_path, monkeypatch):
    """The whole point of the manifest: a rewritten mtime must not rewrite the applied date."""
    import yaml as yaml_mod
    _no_legacy_cache(monkeypatch, tmp_path)
    save_dir = tmp_path / 'saved_pdfs'
    applied = tmp_path / 'applied_jobs'
    applied.mkdir(parents=True)
    _write_pdf(save_dir / 'cat-saved_jd-acme_job.pdf', date(2026, 7, 25))  # mtime says July
    (applied / 'index.yaml').write_text(yaml_mod.dump({
        'cat-saved_jd-acme_job.pdf': {'applied_date': date(2026, 4, 13)},  # index says April
    }))

    await tools.ingest_save_dir_applied_pdfs(save_dir=save_dir, applied_to_dir=applied)

    assert (applied / '2026-04-13-cat-saved_jd-acme_job.pdf').exists()


async def test_ingest_falls_back_to_legacy_downloads_cache_mtime(tmp_path, monkeypatch):
    import yaml as yaml_mod
    save_dir = tmp_path / 'saved_pdfs'
    applied = tmp_path / 'applied_jobs'
    source = _write_pdf(save_dir / 'cat-saved_jd-acme_job.pdf', date(2026, 7, 25))

    legacy = tmp_path / 'downloads_pdf_cache.yaml'
    legacy_ts = datetime.combine(date(2026, 4, 13), datetime.min.time()).timestamp()
    legacy.write_text(yaml_mod.dump({str(source): {'mtime': legacy_ts, 'company': 'Acme'}}))
    monkeypatch.setattr(tools, 'LEGACY_DOWNLOADS_CACHE_PATH', legacy)

    await tools.ingest_save_dir_applied_pdfs(save_dir=save_dir, applied_to_dir=applied)

    assert (applied / '2026-04-13-cat-saved_jd-acme_job.pdf').exists()


async def test_load_applied_jobs_uses_index_without_llm_call(tmp_path, monkeypatch):
    import yaml as yaml_mod
    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    pdf = _write_pdf(applied / '2026-07-20-cat-saved_jd-acme_job.pdf', date.today())

    index_path = applied / 'index.yaml'
    index_path.write_text(yaml_mod.dump({
        pdf.name: _index_entry(date.today(), pdf.stat().st_mtime),
    }))

    extract_called = []

    async def fake_extract(text, filename):
        extract_called.append(filename)
        return {'company': 'ShouldNotBeCalled', 'job_title': '', 'is_agency': False, 'end_client': ''}

    monkeypatch.setattr(tools, '_extract_applied_job_metadata', fake_extract)

    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    assert extract_called == [], 'LLM should not be called on an index hit'
    assert 'CachedCorp' in tools._applied_companies
    assert _reference_texts() == ['job description text']
    assert len(tools._applied_jobs) == 1


async def test_load_applied_jobs_ignores_the_save_dir(tmp_path, monkeypatch):
    """The corpus and the save directory are separate: an un-ingested PDF is invisible here.

    Pins the split — if load_applied_jobs ever fell back to the save directory, a PDF the user
    merely saved (but never applied to) would silently join the reference corpus and the
    already-applied blocklist.
    """
    _no_legacy_cache(monkeypatch, tmp_path)
    save_dir = tmp_path / 'saved_pdfs'
    applied = tmp_path / 'applied_jobs'
    applied.mkdir()
    _write_pdf(save_dir / 'cat-saved_jd-not_yet_ingested.pdf', date.today())
    monkeypatch.setattr(tools.preferences, 'save_dir', lambda: save_dir)

    async def fake_extract(text, filename):
        raise AssertionError(f'save-dir PDF must never be read by load_applied_jobs: {filename}')

    monkeypatch.setattr(tools, '_extract_applied_job_metadata', fake_extract)

    await tools.load_applied_jobs(applied_to_dir=applied, index_path=applied / 'index.yaml')

    assert tools._applied_companies == {}
    assert _reference_texts() == []
    assert tools._applied_jobs == []


async def test_load_applied_jobs_extracts_and_writes_index(tmp_path, monkeypatch):
    _no_legacy_cache(monkeypatch, tmp_path)
    import yaml as yaml_mod
    applied = tmp_path / 'applied_jobs'
    _write_pdf(applied / '2026-07-20-cat-saved_jd-acme_job.pdf', date.today())

    class FakePage:
        def extract_text(self):
            return 'Acme is hiring a Staff AI Engineer'

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)

    async def fake_extract(text, filename):
        return {'company': 'Acme', 'job_title': 'Staff AI Engineer', 'is_agency': False, 'end_client': ''}

    monkeypatch.setattr(tools, '_extract_applied_job_metadata', fake_extract)

    index_path = applied / 'index.yaml'
    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    assert tools._applied_companies == {'Acme': '2026-07-20-cat-saved_jd-acme_job.pdf'}
    written = yaml_mod.safe_load(index_path.read_text())
    entry = written['2026-07-20-cat-saved_jd-acme_job.pdf']
    assert entry['company'] == 'Acme'
    assert entry['job_title'] == 'Staff AI Engineer'
    assert entry['applied_date'] == date(2026, 7, 20)


async def test_load_applied_jobs_excludes_beyond_horizon(tmp_path, monkeypatch):
    import yaml as yaml_mod
    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    stale_date = date.today() - timedelta(days=tools.APPLIED_JOBS_HORIZON_DAYS + 1)
    stale = _write_pdf(applied / f'{stale_date.isoformat()}-cat-saved_jd-old_job.pdf', stale_date)
    fresh = _write_pdf(applied / f'{date.today().isoformat()}-cat-saved_jd-new_job.pdf', date.today())

    index_path = applied / 'index.yaml'
    index_path.write_text(yaml_mod.dump({
        stale.name: _index_entry(stale_date, stale.stat().st_mtime, company='OldCorp', text='old text'),
        fresh.name: _index_entry(date.today(), fresh.stat().st_mtime, company='NewCorp', text='new text'),
    }))

    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    assert stale.exists(), 'aged-out PDFs are kept on disk, only excluded from use'
    assert tools._applied_companies == {'NewCorp': fresh.name}
    assert _reference_texts() == ['new text']
    assert [j['filename'] for j in tools._applied_jobs] == [fresh.name]


async def test_load_applied_jobs_does_not_blocklist_recruiting_agency(tmp_path, monkeypatch):
    import yaml as yaml_mod
    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    pdf = _write_pdf(applied / f'{date.today().isoformat()}-cat-saved_jd-agencycorp.pdf', date.today())

    index_path = applied / 'index.yaml'
    index_path.write_text(yaml_mod.dump({
        pdf.name: _index_entry(date.today(), pdf.stat().st_mtime, company='AgencyCorp', is_agency=True),
    }))

    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    assert tools._applied_companies == {}, 'an agency name must never block its other postings'
    assert _reference_texts() == ['job description text'], 'still useful as reference signal'


async def test_load_applied_jobs_blocklists_end_client_not_agency(tmp_path, monkeypatch):
    import yaml as yaml_mod
    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    pdf = _write_pdf(applied / f'{date.today().isoformat()}-cat-saved_jd-agencycorp.pdf', date.today())

    index_path = applied / 'index.yaml'
    index_path.write_text(yaml_mod.dump({
        pdf.name: _index_entry(
            date.today(), pdf.stat().st_mtime,
            company='AgencyCorp', is_agency=True, end_client='EndClientCo',
        ),
    }))

    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    assert tools._applied_companies == {'EndClientCo': pdf.name}


def test_applied_jobs_summary_renders_titles_dates_and_agency(monkeypatch):
    monkeypatch.setattr(tools, '_applied_jobs', [
        {'filename': 'a.pdf', 'applied_date': date(2026, 5, 13), 'company': 'BMC Software',
         'job_title': 'Principal Agentic AI Engineer', 'is_agency': False, 'end_client': ''},
        {'filename': 'b.pdf', 'applied_date': date(2026, 7, 20), 'company': 'AgencyCorp',
         'job_title': 'Staff AI Engineer', 'is_agency': True, 'end_client': 'EndClientCo'},
    ])

    summary = tools.applied_jobs_summary()
    lines = summary.splitlines()

    assert lines[0].startswith('- Staff AI Engineer — AgencyCorp (2026-07-20)'), 'newest first'
    assert '[via agency]' in lines[0]
    assert '[hiring company: EndClientCo]' in lines[0]
    assert lines[1] == '- Principal Agentic AI Engineer — BMC Software (2026-05-13)'


def test_applied_jobs_summary_empty_when_no_jobs(monkeypatch):
    monkeypatch.setattr(tools, '_applied_jobs', [])
    assert tools.applied_jobs_summary() == ''


# ---------------------------------------------------------------------------
# build_reference_block
# ---------------------------------------------------------------------------

def test_build_reference_block_empty_when_no_texts(monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [])
    assert agent.build_reference_block() == ''


def test_build_reference_block_includes_text(monkeypatch):
    from agentic_job_search import agent
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [['This is a great remote job at Acme.']])
    block = agent.build_reference_block()
    assert 'Acme' in block
    assert 'REFERENCE JOBS' in block


def test_build_evaluator_prompt_embeds_reference_block():
    block = '--- REFERENCE JOBS ---\n[Reference Job 1]\nGreat job at Acme.\n--- END REFERENCE JOBS ---'
    prompt = agent.build_evaluator_prompt(block)
    assert 'Great job at Acme.' in prompt
    assert prompt.index('REFERENCE JOBS') < prompt.index('You are evaluating a single job posting')


def test_build_evaluator_prompt_strips_null_bytes():
    block = 'Job at Acme\x00 with\x00 nulls from PDF extraction'
    prompt = agent.build_evaluator_prompt(block)
    assert '\x00' not in prompt
    assert 'Job at Acme with nulls from PDF extraction' in prompt


def test_build_evaluator_prompt_without_reference_block():
    prompt = agent.build_evaluator_prompt()
    assert 'REFERENCE JOBS' not in prompt
    assert 'You are evaluating a single job posting' in prompt


def test_build_reference_block_caps_at_max_pdfs(monkeypatch):
    from agentic_job_search import agent
    from agentic_job_search.config import MAX_REFERENCE_JOBS
    n = MAX_REFERENCE_JOBS + 5
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [[f'job text {i}'] for i in range(n)])
    block = agent.build_reference_block()
    assert block.count('[Reference Job') == MAX_REFERENCE_JOBS


# ---------------------------------------------------------------------------
# _categorize_pdf_text and categorize_save_dir_pdfs
# ---------------------------------------------------------------------------

async def test_categorize_pdf_text_saved_jd(monkeypatch):
    _make_sdk_mock(monkeypatch, {'category': 'saved_jd'})
    result = await tools._categorize_pdf_text('job posting text', 'acme_job.pdf')
    assert result == 'saved_jd'


async def test_categorize_pdf_text_returns_none_when_tool_not_called(monkeypatch):
    async def fake_sdk_query(**kwargs):
        return  # tool never called
        yield

    monkeypatch.setattr(tools, 'sdk_query', fake_sdk_query)
    result = await tools._categorize_pdf_text('some random text', 'random.pdf')
    assert result is None


async def test_categorize_save_dir_pdfs_renames_uncategorized(tmp_path, monkeypatch):
    pdf = tmp_path / 'report.pdf'
    pdf.write_bytes(b'%PDF fake')

    async def fake_categorize(text, filename):
        return 'saved_jd'

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    class FakeReader:
        pages: ClassVar[list] = []
        def __init__(self, path): pass

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)
    await tools.categorize_save_dir_pdfs(save_dir=tmp_path)
    assert not pdf.exists()
    assert (tmp_path / 'cat-saved_jd-report.pdf').exists()


async def test_categorize_save_dir_pdfs_skips_rename_on_error(tmp_path, monkeypatch):
    pdf = tmp_path / 'report.pdf'
    pdf.write_bytes(b'%PDF fake')

    async def fake_categorize(text, filename):
        return None  # simulate tool-not-called failure

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    class FakeReader:
        pages: ClassVar[list] = []
        def __init__(self, path): pass

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)
    await tools.categorize_save_dir_pdfs(save_dir=tmp_path)
    assert pdf.exists()  # original file untouched


async def test_categorize_save_dir_pdfs_skips_already_categorized(tmp_path, monkeypatch):
    pdf = tmp_path / 'cat-other-old_report.pdf'
    pdf.write_bytes(b'%PDF fake')
    sdk_called = []

    async def fake_categorize(text, filename):
        sdk_called.append(True)
        return 'other'

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    await tools.categorize_save_dir_pdfs(save_dir=tmp_path)
    assert not sdk_called
    assert pdf.exists()


def test_save_dir_preference_expands_home(monkeypatch):
    monkeypatch.setattr(
        preferences, 'load_preferences',
        lambda force_reload=False: {**preferences.DEFAULT_PREFERENCES, 'save_dir': '~/Elsewhere'},
    )
    assert preferences.save_dir() == Path.home() / 'Elsewhere'


async def test_categorize_save_dir_pdfs_defaults_to_preference(tmp_path, monkeypatch):
    """With no argument the save directory comes from preferences, not a hardcoded ~/Downloads."""
    pdf = tmp_path / 'report.pdf'
    pdf.write_bytes(b'%PDF fake')

    async def fake_categorize(text, filename):
        return 'saved_jd'

    monkeypatch.setattr(tools.preferences, 'save_dir', lambda: tmp_path)
    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    class FakeReader:
        pages: ClassVar[list] = []
        def __init__(self, path): pass

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)
    await tools.categorize_save_dir_pdfs()
    assert (tmp_path / 'cat-saved_jd-report.pdf').exists()


# ---------------------------------------------------------------------------
# do_queue_candidate
# ---------------------------------------------------------------------------

async def test_queue_candidate_appends_to_list(monkeypatch):
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})

    await tools.do_queue_candidate(
        'linkedin', '111', 'https://example.com', 'Staff Engineer', 'Acme', 'Great role'
    )

    assert len(tools._candidates) == 1
    assert tools._candidates[0]['company'] == 'Acme'


async def test_queue_candidate_tracks_query_count(monkeypatch):
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})

    await tools.do_queue_candidate(
        'linkedin', '111', 'https://example.com', 'Staff Engineer', 'Acme', 'snippet', query='Staff ML Engineer'
    )
    await tools.do_queue_candidate(
        'linkedin', '222', 'https://example.com/2', 'Principal Engineer', 'Corp', 'snippet', query='Staff ML Engineer'
    )
    await tools.do_queue_candidate(
        'linkedin', '333', 'https://example.com/3', 'Lead ML Engineer', 'BigCo', 'snippet', query='Lead ML Engineer'
    )

    assert tools._candidates_per_query['Staff ML Engineer'] == 2
    assert tools._candidates_per_query['Lead ML Engineer'] == 1


async def test_queue_candidate_attributes_to_the_running_query_not_the_model_string(monkeypatch):
    """Attribution is the caller's to know, not the model's to remember.

    Region and filter words now live in the search text, so the model passes back the full string
    ("Principal AI Engineer, remote, Canada, senior level") while run_scraper keys on the base
    query. The keys never matched and every per-query candidate count read 0 while queueing
    itself worked fine.
    """
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queue_skipped_counts', {})
    monkeypatch.setattr(tools, '_listing_records', {})
    monkeypatch.setattr(tools, '_current_query', 'Principal AI Engineer')

    await tools.do_queue_candidate(
        'linkedin', '1', 'https://example.com', 'Staff Engineer', 'Acme', 'snip',
        query='Principal AI Engineer, remote, Canada, senior level',
    )

    assert tools._candidates_per_query == {'Principal AI Engineer': 1}


async def test_run_scraper_sets_the_current_query(monkeypatch):
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(tools, '_current_query', None)

    seen = []

    class _RecordingClient(_FakeScraperClient):
        async def query(self, instruction):
            seen.append(tools._current_query)
            await super().query(instruction)

    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    queries = ['Alpha', 'Beta']
    await _scrape(_RecordingClient(dict.fromkeys(queries, healthy)), queries,
                            {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

    assert seen == ['Alpha', 'Beta'], 'each request runs with its own base query set'


def test_min_turns_floor_is_absolute_not_a_budget_fraction():
    """A fraction of the (worst-case) budget fired on every healthy query once harvesting replaced
    click-to-reveal, and a warning that fires on success gets ignored."""
    assert config.SCRAPER_MIN_TURNS_PER_QUERY >= 1
    assert config.SCRAPER_MIN_TURNS_PER_QUERY < config.SCRAPER_MAX_TURNS_PER_QUERY // 3, \
        "must sit below a healthy query's turn count, not scale with the budget"


async def test_queue_candidate_without_query_does_not_track(monkeypatch):
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})

    await tools.do_queue_candidate(
        'linkedin', '444', 'https://example.com', 'Engineer', 'Co', 'snippet'
    )

    assert tools._candidates_per_query == {}
    assert len(tools._candidates) == 1


# ---------------------------------------------------------------------------
# triage.extract_json_object
# ---------------------------------------------------------------------------

def test_extract_json_object_plain():
    assert triage.extract_json_object('{"score": 3, "reason": "ok"}') == {'score': 3, 'reason': 'ok'}


def test_extract_json_object_with_surrounding_prose():
    text = 'Here is my answer:\n{"score": 2, "reason": "wrong stack"}\nHope that helps.'
    assert triage.extract_json_object(text) == {'score': 2, 'reason': 'wrong stack'}


def test_extract_json_object_nested():
    text = 'prefix {"rating": 4, "details": {"stack": "python"}} suffix'
    assert triage.extract_json_object(text) == {'rating': 4, 'details': {'stack': 'python'}}


def test_extract_json_object_skips_broken_prefix():
    text = 'broken { not json } then {"score": 5, "reason": "great"}'
    assert triage.extract_json_object(text) == {'score': 5, 'reason': 'great'}


def test_extract_json_object_raises_without_json():
    with pytest.raises(ValueError):
        triage.extract_json_object('no json here at all')


# ---------------------------------------------------------------------------
# do_submit_job_extract
# ---------------------------------------------------------------------------

async def test_submit_job_extract_captures(monkeypatch):
    monkeypatch.setattr(tools, '_job_extracts', [])

    await tools.do_submit_job_extract(
        'Staff Engineer', 'Acme', 'Build agents in Python.',
        location='Canada (Remote)', date_posted='3 days ago', closed=False, salary='CAD 200k',
    )

    assert len(tools._job_extracts) == 1
    extract = tools._job_extracts[0]
    assert extract['title'] == 'Staff Engineer'
    assert extract['company'] == 'Acme'
    assert extract['location'] == 'Canada (Remote)'
    assert extract['closed'] is False


async def test_submit_job_extract_defaults(monkeypatch):
    monkeypatch.setattr(tools, '_job_extracts', [])

    await tools.do_submit_job_extract('Engineer', 'Corp', 'desc')

    extract = tools._job_extracts[0]
    assert extract['location'] == ''
    assert extract['date_posted'] == ''
    assert extract['closed'] is False
    assert extract['salary'] == ''
    assert extract['sponsorship_note'] == ''
    assert extract['language_requirement'] == ''
    assert extract['relocation'] == ''
    assert extract['education_requirement'] == ''
    assert extract['posting_language'] == ''
    assert 'local_language' not in extract, 'the extractor no longer supplies it — the location does'


async def test_submit_job_extract_normalizes_language_fields(monkeypatch):
    monkeypatch.setattr(tools, '_job_extracts', [])

    await tools.do_submit_job_extract(
        'Scientifique principal', 'Valtech', 'desc', posting_language=' French ',
    )

    assert tools._job_extracts[0]['posting_language'] == 'french'


async def test_submit_job_extract_normalizes_stated_working_language(monkeypatch):
    monkeypatch.setattr(tools, '_job_extracts', [])
    await tools.do_submit_job_extract('t', 'c', 'desc', stated_working_language=' English ')
    assert tools._job_extracts[0]['stated_working_language'] == 'english'


async def test_submit_job_extract_rejects_a_local_language_argument():
    """The implied local language is a fact about a PLACE, resolved by the cached classifier.

    Pinned as a signature check rather than prose: re-adding the parameter would otherwise be
    accepted and silently ignored, which is the shape this repo keeps catching.
    """
    with pytest.raises(TypeError):
        await tools.do_submit_job_extract('t', 'c', 'desc', local_language='french')


# ---------------------------------------------------------------------------
# apply_hard_rules
# ---------------------------------------------------------------------------

# NOTE the default description says "Remote within Canada", which `derive_residency_scope` reads
# as `country_only`. That is load-bearing for the older gate tests -- it is why they still judge
# the anchor -- but it is an accident of the fixture, not a statement of intent. A test about the
# residency rule MUST pass its own description, or it silently tests the wrong branch.
_NEUTRAL_DESCRIPTION = 'Build agentic AI systems in Python.'


def _make_extract(**overrides) -> dict:
    extract = {
        'title': 'Staff AI Engineer', 'company': 'Acme',
        'description': 'Build agentic AI systems in Python. Remote within Canada.',
        'location': 'Canada (Remote)', 'date_posted': '3 days ago',
        'closed': False, 'salary': '', 'sponsorship_note': '',
        'language_requirement': '', 'relocation': '', 'education_requirement': '',
        'posting_language': '', 'likely_working_language': '', 'residency_scope': '',
    }
    extract.update(overrides)
    return extract


def _make_candidate(**overrides) -> dict:
    candidate = {
        'site': 'linkedin', 'job_id': '123', 'url': 'https://example.com/job/123',
        'title': 'Staff AI Engineer', 'company': 'Acme',
        'date_posted': '', 'snippet': '',
    }
    candidate.update(overrides)
    return candidate


async def test_apply_hard_rules_passes_good_job():
    assert await agent.apply_hard_rules(_make_candidate(), _make_extract()) is None


async def test_apply_hard_rules_closed_flag():
    reason = await agent.apply_hard_rules(_make_candidate(), _make_extract(closed=True))
    assert reason is not None and 'closed' in reason


async def test_apply_hard_rules_closed_text():
    extract = _make_extract(description='This job is no longer accepting applications.')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'closed' in reason


async def test_apply_hard_rules_stale_posting():
    old = (date.today() - timedelta(days=45)).isoformat()
    reason = await agent.apply_hard_rules(_make_candidate(), _make_extract(date_posted=old))
    assert reason is not None and 'older than' in reason


async def test_apply_hard_rules_recent_posting_ok():
    recent = (date.today() - timedelta(days=5)).isoformat()
    assert await agent.apply_hard_rules(_make_candidate(), _make_extract(date_posted=recent)) is None


async def test_apply_hard_rules_us_no_sponsorship():
    extract = _make_extract(location='United States (Remote)')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'sponsorship' in reason


async def test_apply_hard_rules_us_with_sponsorship_passes():
    extract = _make_extract(
        location='United States (Remote)',
        sponsorship_note='We are willing to sponsor H-1B visas.',
    )
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


async def test_apply_hard_rules_explicit_no_auth_statement():
    extract = _make_extract(description='Must be authorized to work in the US without sponsorship.')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'sponsorship' in reason


async def test_apply_hard_rules_non_english_language_requirement():
    extract = _make_extract(language_requirement='dutch')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'dutch' in reason and 'language' in reason


async def test_apply_hard_rules_english_plus_other_language_rejects():
    extract = _make_extract(language_requirement='english, ukrainian')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'ukrainian' in reason


async def test_apply_hard_rules_english_only_requirement_passes():
    assert await agent.apply_hard_rules(_make_candidate(), _make_extract(language_requirement='english')) is None
    assert await agent.apply_hard_rules(_make_candidate(), _make_extract(language_requirement='english (b2)')) is None


async def test_apply_hard_rules_relocation_does_not_reject():
    extract = _make_extract(relocation='Portugal')
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


@pytest.fixture
def tax_residence_canada(monkeypatch):
    merged = preferences._deep_merge(preferences.load_preferences(), {'current_tax_residence': 'Canada'})
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)


@pytest.mark.parametrize('relocation', [
    'Canada',
    'Canada (remote within Canada)',
    'Canada (remote within country)',
    'Canada (must be based in Canada)',
    'Remote within  Canada',
    'United States or Canada',
])
def test_relocation_already_met_for_the_residence_country_alone(tax_residence_canada, relocation):
    assert agent.relocation_already_met(relocation)


@pytest.mark.parametrize('relocation', [
    '',
    'Portugal',
    'Germany or UK',
    'Ontario, Canada',
    'Toronto, ON, Canada',
    'Vancouver, BC, Canada',
    'Canada (Ontario, British Columbia, or Alberta)',
    'Canada and United States',
    'canada',          # proper names match case-sensitively, like every place list
    'Canadair',
])
def test_relocation_not_met_when_narrower_or_elsewhere(tax_residence_canada, relocation):
    assert not agent.relocation_already_met(relocation)


def test_relocation_is_never_met_without_a_tax_residence():
    assert preferences.current_tax_residence() == ''
    assert not agent.relocation_already_met('Canada')


def test_build_deterministic_warnings_omits_relocation_to_the_residence_country(tax_residence_canada):
    def relocation_warnings(relocation):
        extract = _make_extract(relocation=relocation, salary='CAD 200,000 - 240,000 per year')
        return [w for w in agent.build_deterministic_warnings(_make_candidate(), extract)
                if w.startswith('Relocation required')]
    assert relocation_warnings('Canada (remote within Canada)') == []
    assert relocation_warnings('Ontario, Canada') == ['Relocation required: Ontario, Canada']
    assert relocation_warnings('Portugal') == ['Relocation required: Portugal']


def test_format_extract_text_tells_the_rater_a_residence_requirement_is_met(tax_residence_canada):
    text = agent.format_extract_text(_make_candidate(), _make_extract(relocation='Canada'))
    assert 'Residence requirement: Canada — already met (current tax residence)' in text
    assert 'Relocation required' not in text
    elsewhere = agent.format_extract_text(_make_candidate(), _make_extract(relocation='Portugal'))
    assert 'Relocation required: Portugal' in elsewhere
    assert 'Residence requirement' not in elsewhere


def test_current_tax_residence_rejects_a_non_string(monkeypatch):
    merged = preferences._deep_merge(
        preferences.load_preferences(), {'current_tax_residence': ['Canada', 'Spain']})
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)
    with pytest.raises(ValueError, match='current_tax_residence'):
        preferences.current_tax_residence()


def test_format_extract_text_includes_language_and_relocation():
    extract = _make_extract(
        language_requirement='english, german', relocation='Berlin, Germany',
        education_requirement='phd',
    )
    text = agent.format_extract_text(_make_candidate(), extract)
    assert 'Language requirement: english, german' in text
    assert 'Relocation required: Berlin, Germany' in text
    assert 'Education requirement: phd' in text


# ---------------------------------------------------------------------------
# derive_education_requirement / advanced-degree hard rule
# ---------------------------------------------------------------------------

async def test_apply_hard_rules_explicit_phd_requirement():
    reason = await agent.apply_hard_rules(_make_candidate(), _make_extract(education_requirement='phd'))
    assert reason is not None and 'degree' in reason and 'phd' in reason


async def test_apply_hard_rules_explicit_masters_requirement():
    reason = await agent.apply_hard_rules(_make_candidate(), _make_extract(education_requirement='master'))
    assert reason is not None and 'degree' in reason and 'master' in reason


async def test_apply_hard_rules_empty_education_requirement_passes():
    assert await agent.apply_hard_rules(_make_candidate(), _make_extract()) is None


async def test_apply_hard_rules_bachelors_requirement_passes():
    extract = _make_extract(education_requirement='bachelor')
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


@pytest.mark.parametrize('description, expected', [
    ('PhD in Machine Learning is required.', 'phd'),
    ('Requirements: MSc in Computer Science.', 'master'),
    ("A Master's degree is a must.", 'master'),
    ('Minimum: Ph.D. in Statistics', 'phd'),
    ('You must hold a doctorate in a quantitative field.', 'phd'),
])
def test_derive_education_requirement_detects_hard_requirements(description, expected):
    assert agent.derive_education_requirement(_make_extract(description=description)) == expected


@pytest.mark.parametrize('description', [
    'MSc preferred.',
    "Master's degree or equivalent practical experience required.",
    "Bachelor's or Master's in Computer Science required.",
    'PhD a plus.',
    'A doctorate is nice to have.',
    "Bachelor's degree required.",
    'You will master the art of distributed systems.',
    'PhD-level colleagues work here; degrees are not required.',
    'No PhD required.',
    'No advanced degree required — we hire on experience.',
    "A Master's degree is not required for this role.",
    'We do not require a PhD.',
    'Senior engineers without a PhD are encouraged to apply; experience is what is required.',
])
def test_derive_education_requirement_ignores_soft_and_irrelevant_mentions(description):
    assert agent.derive_education_requirement(_make_extract(description=description)) == ''


async def test_apply_hard_rules_derives_degree_from_description():
    extract = _make_extract(description='Build agents in Python. A PhD in AI is required.')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'requires advanced degree: phd' in reason


def test_hard_rule_category_buckets_education():
    assert agent._hard_rule_category('requires advanced degree: master') == 'hard_ruled_education'
    assert agent._hard_rule_category('requires unsupported language: dutch') == 'hard_ruled_language'


# ---------------------------------------------------------------------------
# triage_job_fit / triage_rejects
# ---------------------------------------------------------------------------

async def test_triage_job_fit_parses_score(monkeypatch):
    async def fake_generate_local(prompt, system='', model='', max_tokens=0):
        return 'Sure! {"score": 2, "reason": "wrong domain"}'

    monkeypatch.setattr(triage, 'generate_local', fake_generate_local)
    result = await triage.triage_job_fit('job text', 'profile')
    assert result == {'score': 2, 'reason': 'wrong domain'}


async def test_triage_job_fit_fails_open_when_server_down(monkeypatch):
    async def fake_generate_local(prompt, system='', model='', max_tokens=0):
        raise RuntimeError('connection refused')

    monkeypatch.setattr(triage, 'generate_local', fake_generate_local)
    assert await triage.triage_job_fit('job text', 'profile') is None


async def test_triage_job_fit_fails_open_on_garbage(monkeypatch):
    async def fake_generate_local(prompt, system='', model='', max_tokens=0):
        return 'I cannot help with that.'

    monkeypatch.setattr(triage, 'generate_local', fake_generate_local)
    assert await triage.triage_job_fit('job text', 'profile') is None


def test_triage_rejects_below_threshold():
    from agentic_job_search.config import TRIAGE_THRESHOLD
    assert triage.triage_rejects({'score': TRIAGE_THRESHOLD, 'reason': 'x'})
    assert triage.triage_rejects({'score': 1, 'reason': 'x'})


def test_triage_rejects_passes_above_threshold_and_none():
    from agentic_job_search.config import TRIAGE_THRESHOLD
    assert not triage.triage_rejects({'score': TRIAGE_THRESHOLD + 1, 'reason': 'x'})
    assert not triage.triage_rejects(None)


# ---------------------------------------------------------------------------
# build_reference_summary provider chain + cache
# ---------------------------------------------------------------------------

async def test_build_reference_summary_openrouter_first(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [['Great AI job at Acme.']])
    monkeypatch.setattr(agent, 'REFERENCE_SUMMARY_CACHE_PATH', tmp_path / 'ref_cache.yaml')
    calls = []

    async def fake_openrouter(prompt, system='', model='', max_tokens=0):
        calls.append('openrouter')
        return 'Ideal role: senior AI engineering.', 0.001

    monkeypatch.setattr(agent, 'chat_openrouter', fake_openrouter)
    stats = agent.new_stage_stats()
    block = await agent.build_reference_summary(stats)

    assert calls == ['openrouter']
    assert 'Ideal role: senior AI engineering.' in block
    assert 'IDEAL ROLE PROFILE' in block
    assert stats['cost'] == pytest.approx(0.001)


async def test_build_reference_summary_falls_back_to_local(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [['Great AI job at Acme.']])
    monkeypatch.setattr(agent, 'REFERENCE_SUMMARY_CACHE_PATH', tmp_path / 'ref_cache.yaml')
    calls = []

    async def fake_openrouter(prompt, system='', model='', max_tokens=0):
        calls.append('openrouter')
        raise RuntimeError('server down')

    async def fake_local(prompt, system='', model='', max_tokens=0):
        calls.append('ollama')
        return 'Local summary of ideal role.'

    monkeypatch.setattr(agent, 'chat_openrouter', fake_openrouter)
    monkeypatch.setattr(agent, 'generate_local', fake_local)
    block = await agent.build_reference_summary(agent.new_stage_stats())

    assert calls == ['openrouter', 'ollama']
    assert 'Local summary of ideal role.' in block


async def test_build_reference_summary_falls_back_to_anthropic(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [['Great AI job at Acme.']])
    monkeypatch.setattr(agent, 'REFERENCE_SUMMARY_CACHE_PATH', tmp_path / 'ref_cache.yaml')
    calls = []

    async def fake_openrouter(prompt, system='', model='', max_tokens=0):
        calls.append('openrouter')
        raise RuntimeError('server down')

    async def fake_local(prompt, system='', model='', max_tokens=0):
        calls.append('ollama')
        raise RuntimeError('server down too')

    async def fake_anthropic(prompt, stage_stats):
        calls.append('anthropic')
        return 'Anthropic summary.'

    monkeypatch.setattr(agent, 'chat_openrouter', fake_openrouter)
    monkeypatch.setattr(agent, 'generate_local', fake_local)
    monkeypatch.setattr(agent, '_summarize_references_anthropic', fake_anthropic)
    block = await agent.build_reference_summary(agent.new_stage_stats())

    assert calls == ['openrouter', 'ollama', 'anthropic']
    assert 'Anthropic summary.' in block


async def test_build_reference_summary_falls_back_to_full_block_when_all_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [['Great AI job at Acme.']])
    monkeypatch.setattr(agent, 'REFERENCE_SUMMARY_CACHE_PATH', tmp_path / 'ref_cache.yaml')

    async def fail(*args, **kwargs):
        raise RuntimeError('down')

    monkeypatch.setattr(agent, 'chat_openrouter', fail)
    monkeypatch.setattr(agent, 'generate_local', fail)
    monkeypatch.setattr(agent, '_summarize_references_anthropic', fail)
    block = await agent.build_reference_summary(agent.new_stage_stats())

    assert 'REFERENCE JOBS' in block  # old full block as last resort
    assert 'Acme' in block


async def test_build_reference_summary_cache_hit_skips_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [['Great AI job at Acme.']])
    monkeypatch.setattr(agent, 'REFERENCE_SUMMARY_CACHE_PATH', tmp_path / 'ref_cache.yaml')
    calls = []

    async def fake_openrouter(prompt, system='', model='', max_tokens=0):
        calls.append('openrouter')
        return 'Summary v1.', 0.001

    monkeypatch.setattr(agent, 'chat_openrouter', fake_openrouter)
    block1 = await agent.build_reference_summary(agent.new_stage_stats())
    block2 = await agent.build_reference_summary(agent.new_stage_stats())

    assert calls == ['openrouter']  # second call served from cache
    assert block1 == block2


async def test_build_reference_summary_empty_without_texts(monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_reference_job_pages', [])
    assert await agent.build_reference_summary() == ''


# ---------------------------------------------------------------------------
# rate_job provider dispatch
# ---------------------------------------------------------------------------

async def test_rate_job_dispatches_to_openrouter(monkeypatch):
    monkeypatch.setattr(agent, 'MODEL_NAME_RATING', OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE)

    async def fake_rate(system_prompt, user_prompt, model=''):
        return {'rating': 4, 'company': 'Acme', 'title': 'Engineer', 'reasoning': 'good', 'summary': 'acme_engineer'}, 0.002

    monkeypatch.setattr(agent, 'rate_with_openrouter', fake_rate)
    stats = agent.new_stage_stats()
    result = await agent.rate_job('system', 'job text', stats)

    assert result['rating'] == 4
    assert stats['cost'] == pytest.approx(0.002)


async def test_rate_job_dispatches_to_ollama(monkeypatch):
    monkeypatch.setattr(agent, 'MODEL_NAME_RATING', config.OLLAMA_MODEL_NAME_TRIAGE)

    async def fake_rate(system_prompt, user_prompt, model=''):
        return {'rating': 3, 'company': 'Acme', 'title': 'Engineer', 'reasoning': 'ok', 'summary': 'acme_engineer'}

    monkeypatch.setattr(agent, 'rate_with_ollama', fake_rate)
    stats = agent.new_stage_stats()
    result = await agent.rate_job('system', 'job text', stats)

    assert result['rating'] == 3
    assert stats['cost'] == 0.0


async def test_rate_job_dispatches_to_anthropic_by_default(monkeypatch):
    monkeypatch.setattr(agent, 'MODEL_NAME_RATING', ANTHROPIC_MODEL_NAME_MEDIUM)

    async def fake_anthropic(evaluator_prompt, extract_text, stage_stats, model=''):
        return {'rating': 5, 'company': 'Acme', 'title': 'Engineer', 'reasoning': 'great', 'summary': 'acme_engineer'}

    monkeypatch.setattr(agent, '_rate_with_anthropic', fake_anthropic)
    result = await agent.rate_job('system', 'job text', agent.new_stage_stats())
    assert result['rating'] == 5


# ---------------------------------------------------------------------------
# extract_job_page_direct (deterministic fallback)
# ---------------------------------------------------------------------------

def _make_agent_sdk_mock(monkeypatch, structured_output: dict, prompts: list | None = None):
    from claude_agent_sdk import ResultMessage

    async def fake_sdk_query(**kwargs):
        if prompts is not None:
            prompts.append(kwargs['prompt'])
        yield ResultMessage(
            subtype='success', duration_ms=100, duration_api_ms=100, is_error=False,
            num_turns=1, session_id='fake-session', structured_output=structured_output,
        )

    monkeypatch.setattr(agent, 'sdk_query', fake_sdk_query)


def _stub_read_page(monkeypatch, page=None, calls: list | None = None):
    """Replace the code-side page read; records (job_id, url) per read."""
    page = page if page is not None else linkedin_page.JobPage(text='Staff Engineer\nAbout the job\nBuild agents.', source='dom')

    async def fake_read_page(candidate, url, stats):
        if calls is not None:
            calls.append((candidate['job_id'], url))
        return page

    monkeypatch.setattr(agent, 'read_page', fake_read_page)


async def test_extract_job_page_direct_condenses_the_page_it_is_given(monkeypatch):
    prompts = []
    _make_agent_sdk_mock(monkeypatch, {'title': 'Staff Engineer', 'company': 'Acme', 'description': 'Build agents.'}, prompts)

    async def no_read(*args):
        raise AssertionError('a page the caller already read must not be read again')

    monkeypatch.setattr(agent, 'read_page', no_read)
    page = linkedin_page.JobPage(text='PAGE TEXT FROM CODE', source='dom')
    extract = await agent.extract_job_page_direct(_make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), page=page)

    assert extract['title'] == 'Staff Engineer'
    assert extract['closed'] is False  # default filled
    assert 'PAGE TEXT FROM CODE' in prompts[0]
    assert 'de-cluttered by code' in prompts[0]


async def test_extract_job_page_direct_reads_the_page_when_not_given_one(monkeypatch):
    calls = []
    _stub_read_page(monkeypatch, calls=calls)
    _make_agent_sdk_mock(monkeypatch, {'title': 'T', 'company': 'C', 'description': 'D'})

    extract = await agent.extract_job_page_direct(_make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats())
    assert extract is not None
    assert calls == [('123', 'http://localhost:1/mcp')]


async def test_extract_job_page_direct_returns_none_when_the_page_cannot_be_read(monkeypatch):
    async def failed_read(candidate, url, stats):
        return None

    monkeypatch.setattr(agent, 'read_page', failed_read)
    assert await agent.extract_job_page_direct(_make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats()) is None


async def test_extract_job_page_direct_returns_none_without_output(monkeypatch):
    async def fake_sdk_query(**kwargs):
        return
        yield

    monkeypatch.setattr(agent, 'sdk_query', fake_sdk_query)
    page = linkedin_page.JobPage(text='page', source='dom')
    extract = await agent.extract_job_page_direct(_make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), page=page)
    assert extract is None


async def test_read_page_logs_and_returns_none_when_the_browser_fails(monkeypatch, caplog):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def dead_session(url):
        raise ConnectionError('All connection attempts failed')
        yield  # pragma: no cover

    monkeypatch.setattr(agent, 'mcp_session', dead_session)
    with caplog.at_level(logging.WARNING):
        page = await agent.read_page(_make_candidate(job_id='77'), 'http://localhost:1/mcp', agent.new_stage_stats())
    assert page is None
    assert any('Reading the page failed' in r.message and 'job 77' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# extract_job_page_openrouter (page text in, one submit_job_extract out)
# ---------------------------------------------------------------------------

def _chat_response(tool_calls=None, content=None, ok=True, cost=0.001, error=None):
    data = {'ok': ok, 'content': content, 'tool_calls': tool_calls, 'finish_reason': 'tool_calls' if tool_calls else 'stop',
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'cost': cost}, 'cost_usd': cost}
    if error:
        data['error'] = error
    return json.dumps(data)


def _tool_call(call_id, name, args) -> dict:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args) if isinstance(args, dict) else args}}


_SUBMIT_ARGS = {'title': 'Staff Engineer', 'company': 'Acme', 'description': 'Build agents in Python.'}


def _wire_openrouter_loop(monkeypatch, chat_responses: list, chat_requests: list):
    async def fake_call_mcp_tool(url, tool_name, args, timeout_seconds=None):
        # Deep-copy: the loop mutates its messages list in place between calls
        chat_requests.append(json.loads(json.dumps(args)))
        return chat_responses.pop(0)

    monkeypatch.setattr(extract_openrouter, 'call_mcp_tool', fake_call_mcp_tool)


async def _run_loop(page_text='About the job\nBuild agents.'):
    return await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), page_text, agent.new_stage_stats(), 'sys')


def test_extractor_model_has_no_browser_tools():
    """Code reads the page; a model that can click could hit a Dismiss button (2026-09-24)."""
    assert [t['function']['name'] for t in extract_openrouter.OPENROUTER_EXTRACT_TOOLS] == ['submit_job_extract']


async def test_openrouter_loop_happy_path(monkeypatch):
    chat_requests = []
    _wire_openrouter_loop(monkeypatch, [_chat_response(tool_calls=[_tool_call('c1', 'submit_job_extract', _SUBMIT_ARGS)])], chat_requests)

    stats = agent.new_stage_stats()
    extract = await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'PAGE TEXT FROM CODE', stats, 'system prompt')

    assert extract['title'] == 'Staff Engineer'
    assert extract['closed'] is False  # default filled
    assert stats['cost'] == pytest.approx(0.001)
    assert stats['input_tokens'] == 100
    user = chat_requests[0]['messages'][1]
    assert user['role'] == 'user' and 'PAGE TEXT FROM CODE' in user['content']
    assert [t['function']['name'] for t in chat_requests[0]['tools']] == ['submit_job_extract']


async def test_openrouter_loop_rejects_an_unknown_tool(monkeypatch):
    chat_requests = []
    _wire_openrouter_loop(monkeypatch, [
        _chat_response(tool_calls=[_tool_call('c1', 'browser_click', {'target': 'e1'})]),
        _chat_response(tool_calls=[_tool_call('c2', 'submit_job_extract', _SUBMIT_ARGS)]),
    ], chat_requests)

    assert await _run_loop() is not None
    reply = chat_requests[1]['messages'][-1]
    assert reply['role'] == 'tool' and "unknown tool 'browser_click'" in reply['content']


async def test_openrouter_loop_invalid_json_args_retries(monkeypatch):
    chat_requests = []
    _wire_openrouter_loop(monkeypatch, [
        _chat_response(tool_calls=[_tool_call('c1', 'submit_job_extract', '{not json')]),
        _chat_response(tool_calls=[_tool_call('c2', 'submit_job_extract', _SUBMIT_ARGS)]),
    ], chat_requests)

    assert await _run_loop() is not None
    error_msg = chat_requests[1]['messages'][-1]
    assert error_msg['role'] == 'tool'
    assert 'invalid JSON' in error_msg['content']


async def test_openrouter_loop_missing_submit_fields_retries(monkeypatch):
    chat_requests = []
    _wire_openrouter_loop(monkeypatch, [
        _chat_response(tool_calls=[_tool_call('c1', 'submit_job_extract', {'title': 'T'})]),
        _chat_response(tool_calls=[_tool_call('c2', 'submit_job_extract', _SUBMIT_ARGS)]),
    ], chat_requests)

    assert await _run_loop() is not None
    assert 'missing required field' in chat_requests[1]['messages'][-1]['content']


async def test_openrouter_loop_stops_at_iteration_cap(monkeypatch):
    from agentic_job_search.config import EXTRACTOR_OPENROUTER_MAX_ITERATIONS
    chat_requests = []
    _wire_openrouter_loop(monkeypatch, [
        _chat_response(tool_calls=[_tool_call(f'c{i}', 'submit_job_extract', {'title': 'T'})])
        for i in range(EXTRACTOR_OPENROUTER_MAX_ITERATIONS + 5)
    ], chat_requests)

    assert await _run_loop() is None
    assert len(chat_requests) == EXTRACTOR_OPENROUTER_MAX_ITERATIONS


async def test_openrouter_loop_returns_none_when_model_stops_without_submit(monkeypatch):
    _wire_openrouter_loop(monkeypatch, [_chat_response(content='I could not find the job posting.')], [])
    assert await _run_loop() is None


async def test_openrouter_loop_returns_none_on_chat_error(monkeypatch):
    _wire_openrouter_loop(monkeypatch, [_chat_response(ok=False, error='server down')], [])
    assert await _run_loop() is None


async def test_extractor_provider_dispatch_openrouter(monkeypatch):
    monkeypatch.setattr(agent, 'MODEL_NAME_EXTRACTOR', OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC)
    called = {'openrouter': 0, 'anthropic': 0, 'direct': 0}
    reads = []
    page = linkedin_page.JobPage(text='the page', source='dom')
    _stub_read_page(monkeypatch, page=page, calls=reads)

    async def fake_openrouter(candidate, page_text, stats, system_prompt):
        called['openrouter'] += 1
        assert page_text == 'the page'
        assert system_prompt is agent.PAGE_EXTRACTOR_INSTRUCTIONS
        return None

    async def fake_anthropic(candidate, mcp, stats):
        called['anthropic'] += 1
        return None

    async def fake_direct(candidate, url, stats, page=None):
        called['direct'] += 1
        assert page is not None and page.text == 'the page', 'the fallback reuses the page already read'
        return None

    monkeypatch.setattr(agent, 'extract_job_page_openrouter', fake_openrouter)
    monkeypatch.setattr(agent, 'extract_job_page', fake_anthropic)
    monkeypatch.setattr(agent, 'extract_job_page_direct', fake_direct)

    stats = {'extraction': agent.new_stage_stats(), 'rating': agent.new_stage_stats()}
    await agent.evaluate_all_candidates(
        [_make_candidate()], {'type': 'http', 'url': 'http://localhost:1/mcp'}, 'prompt', 'profile', stats
    )

    assert called == {'openrouter': 1, 'anthropic': 0, 'direct': 1}  # fallback still fires
    assert len(reads) == 1, 'the page is read once per job'


# ---------------------------------------------------------------------------
# format_extract_text
# ---------------------------------------------------------------------------

def test_format_extract_text_includes_fields():
    text = agent.format_extract_text(_make_candidate(), _make_extract(salary='CAD 200k'))
    assert 'Title: Staff AI Engineer' in text
    assert 'Company: Acme' in text
    assert 'Salary: CAD 200k' in text
    assert 'https://example.com/job/123' in text
    assert 'Build agentic AI systems' in text


def test_format_extract_text_falls_back_to_candidate_date():
    candidate = _make_candidate(date_posted='2 days ago')
    text = agent.format_extract_text(candidate, _make_extract(date_posted=''))
    assert 'Posted: 2 days ago' in text


# ---------------------------------------------------------------------------
# Local-model preflight and MCP error unwrapping
#
# 2026-08-25: `qwen3.6:latest` stopped being installed and Stage 2c triage failed open on every
# job for ten days. Nothing looked broken - it fails open by design - and the run log only ever
# said "unhandled errors in a TaskGroup (1 sub-exception)", because the RuntimeError naming the
# missing model was nested inside two anyio task groups.
# ---------------------------------------------------------------------------

def test_unwrap_exception_surfaces_root_cause():
    """The message must survive ExceptionGroup nesting, which is how every MCP failure arrives."""
    root = RuntimeError("model 'qwen3.6:latest' not found")
    inner = ExceptionGroup('unhandled errors in a TaskGroup', [root])
    outer = ExceptionGroup('unhandled errors in a TaskGroup', [inner])

    message = triage.unwrap_exception(outer)

    assert "model 'qwen3.6:latest' not found" in message
    assert 'unhandled errors in a TaskGroup' not in message


def test_unwrap_exception_follows_cause_chain():
    root = RuntimeError('ollama said no')
    try:
        try:
            raise root
        except RuntimeError as ex:
            raise ValueError('wrapper') from ex
    except ValueError as ex:
        message = triage.unwrap_exception(ex)

    assert 'ollama said no' in message
    assert 'wrapper' in message


def _patch_mcp(monkeypatch, generate=None, list_models=None):
    """Stub call_mcp_tool, dispatching on tool name. Values may be strings or exceptions."""
    async def fake_call(url, tool_name, args, timeout_seconds=None):
        result = {'generate': generate, 'list_models': list_models}[tool_name]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(triage, 'call_mcp_tool', fake_call)


_INSTALLED = json.dumps({'models': [{'name': 'granite4.1:3b'}, {'name': 'gemma4:26b'}]})


async def test_generate_local_names_installed_models_when_model_missing(monkeypatch):
    """A missing tag must produce the fix, not a 404 the reader has to decode."""
    missing = ExceptionGroup('unhandled errors in a TaskGroup', [
        RuntimeError("MCP tool 'generate' returned an error: 404 Client Error: Not Found for url: "
                     "http://localhost:11434/api/generate model 'gone:latest' not found"),
    ])
    _patch_mcp(monkeypatch, generate=missing, list_models=_INSTALLED)

    with pytest.raises(triage.LocalModelMissingError) as excinfo:
        await triage.generate_local('hi', model='gone:latest')

    message = str(excinfo.value)
    assert 'gone:latest' in message
    assert 'granite4.1:3b' in message and 'gemma4:26b' in message


async def test_generate_local_reraises_other_failures(monkeypatch):
    """A down server must keep reading as a down server, not as a misconfiguration."""
    _patch_mcp(monkeypatch, generate=ConnectionRefusedError('connection refused'), list_models=_INSTALLED)

    with pytest.raises(Exception) as excinfo:
        await triage.generate_local('hi', model='granite4.1:3b')

    assert not isinstance(excinfo.value, triage.LocalModelMissingError)


async def test_preflight_local_model_passes_when_installed(monkeypatch):
    _patch_mcp(monkeypatch, list_models=_INSTALLED)
    assert await triage.preflight_local_model('granite4.1:3b') is None


async def test_preflight_local_model_reports_when_missing(monkeypatch):
    _patch_mcp(monkeypatch, list_models=_INSTALLED)

    problem = await triage.preflight_local_model('qwen3.6:latest')

    assert problem is not None
    assert 'qwen3.6:latest' in problem
    assert 'granite4.1:3b' in problem


async def test_preflight_local_model_is_silent_when_server_is_down(monkeypatch):
    """Fail open quietly: the servers are optional, and crying wolf every run trains it out."""
    _patch_mcp(monkeypatch, list_models=ConnectionRefusedError('connection refused'))
    assert await triage.preflight_local_model('granite4.1:3b') is None


async def test_configured_local_model_is_a_concrete_tag():
    """`:latest` is the tag that vanished on a re-pull; pin a concrete one."""
    assert not config.OLLAMA_MODEL_NAME_TRIAGE.endswith(':latest')


def test_local_model_missing_alert_reaches_run_health():
    """The preflight is worthless if its finding stops at the run log."""
    alerts = agent.assess_run_health({'ui_alerts': [
        {'kind': 'local_model_missing', 'query': '(all)', 'region': '(all)',
         'detail': "Local model 'gone:latest' is not installed"},
    ]})

    assert any('LOCAL TRIAGE DISABLED' in a and 'gone:latest' in a for a in alerts)


# ---------------------------------------------------------------------------
# Live tests for the LLM MCP tool servers (require servers on :8002/:8006)
# ---------------------------------------------------------------------------

def _require_llm_server(port: int) -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex(('127.0.0.1', port)) != 0:
            pytest.skip(f'LLM MCP tool server on :{port} is not running')


@pytest.mark.live
async def test_generate_local_live():
    _require_llm_server(8002)
    # config.OLLAMA_MODEL_NAME_TRIAGE, never a hardcoded copy: this test's whole job is to catch that tag
    # going stale, and a test carrying its own duplicate of the value tracks nothing.
    response = await triage.generate_local('Reply with exactly: OK', model=config.OLLAMA_MODEL_NAME_TRIAGE, max_tokens=500)
    assert response.strip() != ''


@pytest.mark.live
async def test_chat_openrouter_live():
    _require_llm_server(8006)
    content, cost_usd = await triage.chat_openrouter('Reply with exactly: OK', max_tokens=500)
    assert content.strip() != ''
    assert cost_usd >= 0


@pytest.mark.live
async def test_classify_location_live(monkeypatch, tmp_path):
    """The real classifier against the real server. Uses a throwaway cache so a stale entry can
    never make this pass — the point is that the model still answers geography correctly."""
    _require_llm_server(8006)
    from agentic_job_search import location

    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'location_cache.yaml')
    monkeypatch.setattr(location, '_cache', {})
    # Override the autouse network block: this test is exactly the one that should reach out.
    monkeypatch.setattr(location, 'chat_openrouter', triage.chat_openrouter)

    facts = await location.classify_location('Berlin, Germany (Remote across Europe)')
    # Proper-cased, because the prompt asks for it and nothing downstream folds it back. A model
    # that answers 'germany' fails here rather than being quietly normalised into agreement.
    assert facts['countries'] == ['Germany']
    assert facts['regions'] == ['western_europe']
    assert facts['likely_working_language'] == 'german'

    # The distinction the whole gate rests on: a Mediterranean location is a different region,
    # regardless of the language spoken there.
    spain = await location.classify_location('Barcelona, Spain (Remote)')
    assert spain['regions'] == ['southern_europe']

    # Names no country, so no policy can reject it.
    assert (await location.classify_location('European Union'))['countries'] == []


@pytest.mark.live
async def test_triage_job_fit_live():
    _require_llm_server(8002)
    result = await triage.triage_job_fit(
        'Title: Dutch-speaking Junior Accountant\nLocation: Netherlands (on-site)\n'
        'Requires fluent Dutch, 2 years accounting experience, on-site in Amsterdam.',
        'The candidate is a senior AI engineer looking for remote Staff/Principal AI roles in Canada or the EU.',
    )
    assert result is not None, 'local LLM server not reachable'
    assert result['score'] <= 2, f'expected clear non-fit, got {result}'


# ---------------------------------------------------------------------------
# Live tests for save-directory PDF loading
# ---------------------------------------------------------------------------

@pytest.mark.live_agent_claude
async def test_extract_applied_job_metadata_live():
    """Successor to test_extract_company_from_text_live.

    `_extract_company_from_text(text) -> str` was replaced by
    `_extract_applied_job_metadata(text, filename) -> dict` in commit 96bd708, but the old test
    stayed behind calling the deleted name. It never failed, because the live gate skipped it.
    """
    text = """
    Software Engineer — Remote
    Shopify
    We are looking for an experienced software engineer to join our team.
    You will work on our e-commerce platform serving millions of merchants.
    Requirements: 5+ years Python, strong distributed systems knowledge.
    """
    metadata = await tools._extract_applied_job_metadata(text, 'shopify_swe.pdf')

    assert 'shopify' in metadata['company'].lower(), metadata
    assert 'engineer' in metadata['job_title'].lower(), metadata
    # Shopify posts its own roles, so this is the employer, not an agency reposting for a client.
    assert metadata['is_agency'] is False, metadata


@pytest.mark.live_agent_claude
async def test_extract_applied_job_metadata_live_flags_agency():
    """The agency/end-client split is the field that keeps a staffing firm off the blocklist."""
    text = """
    Staff Machine Learning Engineer
    Posted by TalentBridge Recruiting — a specialist technology staffing agency.
    Our client, a global payments company called Northwind Payments, is looking for a
    Staff Machine Learning Engineer to join their fraud detection team.
    Requirements: 8+ years Python, production ML systems.
    """
    metadata = await tools._extract_applied_job_metadata(text, 'talentbridge_mle.pdf')

    assert metadata['is_agency'] is True, metadata
    assert 'northwind' in metadata['end_client'].lower(), metadata


@pytest.mark.live_agent_claude
async def test_categorize_pdf_text_live_saved_jd():
    text = """
    Principal AI Engineer — Remote Canada
    Dayforce
    We are hiring a Principal AI Engineer to lead our machine learning platform.
    Responsibilities: Design agentic AI systems, lead a team of ML engineers,
    build RAG pipelines and LLM evaluation frameworks.
    Requirements: 10+ years experience, Python, deep learning expertise.
    Salary: CAD $220,000 base + equity.
    """
    category = await tools._categorize_pdf_text(text, 'Principal AI Engineer _ Dayforce Jobs.pdf')
    assert category is not None, 'tool was not called'
    assert category == 'saved_jd', f'expected saved_jd, got {category!r}'


@pytest.mark.live_agent_claude
async def test_categorize_pdf_text_live_other():
    text = """
    Your Airbnb booking confirmation
    Check-in: June 12, 2026
    Check-out: June 15, 2026
    Property: Cozy cabin in Whistler
    Total: $450 CAD
    """
    category = await tools._categorize_pdf_text(text, 'Your trip overview – Airbnb.pdf')
    assert category is not None, 'tool was not called'
    assert category != 'saved_jd', f'expected non-jd category, got {category!r}'


@pytest.mark.live_agent_claude
async def test_company_matches_applied_live(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {'BMO Financial Group': 'bmo_job.pdf'})
    result = await tools.company_matches_applied('BMO')
    assert result is not None


# ---------------------------------------------------------------------------
# company normalization / exact-match short circuit
# ---------------------------------------------------------------------------

def test_normalize_company_strips_suffixes_and_punctuation():
    assert tools._normalize_company('Datadog, Inc.') == tools._normalize_company('Datadog')
    assert tools._normalize_company('Acme Technologies Ltd') == 'acme'
    assert tools._normalize_company('The Vista Group') == 'vista'


async def test_company_matches_applied_exact_match_skips_llm(monkeypatch):
    """The per-listing LLM round trip is the Stage 1b bottleneck; exact matches must not pay it."""
    monkeypatch.setattr(tools, '_applied_companies', {'Datadog': 'datadog.pdf'})
    called = []

    async def fail_openrouter(*args, **kwargs):
        called.append(True)
        raise AssertionError('should not be called')

    monkeypatch.setattr(tools, '_company_matches_openrouter', fail_openrouter)

    assert await tools.company_matches_applied('Datadog, Inc.') == 'datadog.pdf'
    assert called == []


async def test_company_matches_applied_falls_back_to_anthropic(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {'Shopify': 'shopify.pdf'})
    monkeypatch.setattr(tools, 'MODEL_NAME_COMPANY_MATCH', OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE)

    async def broken_openrouter(candidate, companies_list):
        raise RuntimeError('OpenRouter MCP server down')

    monkeypatch.setattr(tools, '_company_matches_openrouter', broken_openrouter)
    _make_sdk_mock(monkeypatch, {'matches': True, 'matched_company_name': 'Shopify'})

    assert await tools.company_matches_applied('Shopify Commerce') == 'shopify.pdf'


# ---------------------------------------------------------------------------
# per-run audit log + un-surfaced pools
# ---------------------------------------------------------------------------

def _listing(job_id, company, status='new', queued=True, rating=None, outcome='rated', title='Staff AI Engineer'):
    return {
        'site': 'linkedin', 'job_id': job_id, 'company': company, 'title': title,
        'date_posted': '2 days ago', 'check_status': status,
        'url': tools.job_url('linkedin', job_id),
        'queued': queued, 'outcome': outcome, 'rating': rating, 'summary': f'{company} summary',
    }


def test_job_url_reconstructs_linkedin_url():
    assert tools.job_url('linkedin', '12345') == 'https://www.linkedin.com/jobs/view/12345/'


def test_record_job_outcome_attaches_result(monkeypatch):
    records = {('linkedin', '1'): _listing('1', 'Acme', rating=None, outcome='queued')}
    monkeypatch.setattr(tools, '_listing_records', records)
    tools.record_job_outcome('linkedin', '1', 'rated', rating=5, summary='great fit')
    assert records[('linkedin', '1')]['outcome'] == 'rated'
    assert records[('linkedin', '1')]['rating'] == 5
    assert records[('linkedin', '1')]['summary'] == 'great fit'


def test_unsurfaced_pools_groups_by_drop_reason(monkeypatch):
    monkeypatch.setattr(tools, '_listing_records', {
        ('linkedin', '1'): _listing('1', 'TooOld', status='too_old', queued=False, outcome='too_old'),
        ('linkedin', '2'): _listing('2', 'Applied', status='already_applied', queued=False, outcome='already_applied'),
        ('linkedin', '3'): _listing('3', 'NeverQueued', status='new', queued=False, outcome='new'),
        ('linkedin', '4'): _listing('4', 'MidRated', status='new', queued=True, rating=3),
        ('linkedin', '4b'): _listing('4b', 'MidRated2', status='new', queued=True, rating=2),
        ('linkedin', '5'): _listing('5', 'GoodJob', status='new', queued=True, rating=5),
        ('linkedin', '6'): _listing('6', 'Dupe', status='already_processed', queued=False, outcome='already_processed'),
    })
    pools = tools.unsurfaced_pools()

    assert {r['company'] for r in pools['filtered']} == {'TooOld', 'Applied'}
    assert {r['company'] for r in pools['never_queued']} == {'NeverQueued'}
    assert {r['company'] for r in pools['mid_rated']} == {'MidRated', 'MidRated2'}
    assert 'GoodJob' not in {r['company'] for pool in pools.values() for r in pool}


# ---------------------------------------------------------------------------
# rating thresholds (config constants + predicates)
# ---------------------------------------------------------------------------

def test_should_notify_based_on_rating_boundary():
    assert not config.should_notify_based_on_rating(config.RATING_NOTIFICATION_THRESHOLD - 1)
    assert config.should_notify_based_on_rating(config.RATING_NOTIFICATION_THRESHOLD)
    assert config.should_notify_based_on_rating(config.RATING_MAX)
    assert not config.should_notify_based_on_rating(config.RATING_AUTO_REJECT)


def test_is_auto_reject_rating_boundary():
    assert config.is_auto_reject_rating(config.RATING_AUTO_REJECT)
    assert not config.is_auto_reject_rating(config.RATING_AUTO_REJECT + 1)
    assert not config.is_auto_reject_rating(config.RATING_NOTIFICATION_THRESHOLD)


def test_is_mid_rated_boundary():
    assert config.is_mid_rated(config.RATING_AUTO_REJECT + 1)
    assert config.is_mid_rated(config.RATING_NOTIFICATION_THRESHOLD - 1)
    assert not config.is_mid_rated(config.RATING_AUTO_REJECT)
    assert not config.is_mid_rated(config.RATING_NOTIFICATION_THRESHOLD)
    assert not config.is_mid_rated(0)
    assert not config.is_mid_rated(config.RATING_MAX + 1)


def test_ratings_mid_rated_derived_from_thresholds():
    # The band must be exactly the ratings between the two threshold constants: if either
    # threshold moves and the derivation was replaced by a literal, this fails loudly.
    assert frozenset(
        range(config.RATING_AUTO_REJECT + 1, config.RATING_NOTIFICATION_THRESHOLD)
    ) == config.RATINGS_MID_RATED
    assert frozenset({2, 3}) == config.RATINGS_MID_RATED


def test_write_run_audit_log_covers_all_four_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {'Acme': 'acme.pdf'})
    monkeypatch.setattr(tools, '_candidates_per_query', {'Staff AI Engineer': 2})
    monkeypatch.setattr(tools, '_queries_searched', {'Staff AI Engineer': 7})
    monkeypatch.setattr(tools, '_listing_records', {
        ('linkedin', '1'): _listing('1', 'GoodCorp', rating=5),
        ('linkedin', '2'): _listing('2', 'MehCorp', rating=2),
    })

    path = tools.write_run_audit_log(
        queries=['Staff AI Engineer', 'Principal AI Engineer'],
        applied_jobs_in_horizon=30, applied_jobs_total=43,
        funnel={'listings_seen': 2}, run_dir=tmp_path,
        timestamp=datetime(2026, 7, 28, 17, 5, 0),
    )

    assert path.name == 'audit-2026Jul28-170500.md'
    text = path.read_text()
    assert '30' in text and '43' in text, 'applied-job counts missing'
    assert 'Staff AI Engineer' in text
    # A query that was generated but never searched must be visibly distinct from one that
    # ran and found nothing -- that ambiguity is what hid the truncated-scraper bug.
    assert 'NEVER SEARCHED' in text
    assert 'https://www.linkedin.com/jobs/view/1/' in text
    assert 'GoodCorp' in text and 'MehCorp' in text
    assert 'GoodCorp summary' in text


def test_write_run_audit_log_breaks_down_check_status_per_query(tmp_path, monkeypatch):
    """Per-query dedup columns: 'saturated' and 'barely ran' must not look alike.

    A run-global check_status cannot separate them, which is why a run that inspected 7
    listings instead of 166 read as ordinary dedup saturation.
    """
    monkeypatch.setattr(tools, '_applied_companies', {})
    monkeypatch.setattr(tools, '_candidates_per_query', {'Saturated': 3})
    monkeypatch.setattr(tools, '_queries_searched', {'Saturated': 28, 'Starved': 1, 'Broken': 'error'})
    monkeypatch.setattr(tools, '_check_status_per_query', {
        'Saturated': {'already_processed': 25, 'new': 3},
        'Starved': {'already_processed': 1},
        # 'Broken' errored, so it has no entry at all -- its row must still render.
    })
    monkeypatch.setattr(tools, '_listing_records', {})

    path = tools.write_run_audit_log(
        queries=['Saturated', 'Starved', 'Broken', 'Never'],
        applied_jobs_in_horizon=1, applied_jobs_total=1,
        funnel={}, run_dir=tmp_path, timestamp=datetime(2026, 8, 11, 14, 16, 39),
    )
    rows = {
        line.split('|')[1].strip(): [cell.strip() for cell in line.split('|')[2:-1]]
        for line in path.read_text().splitlines()
        if line.startswith('| ') and '---' not in line
    }

    # columns: Listings inspected | New | Already processed | Already applied | Too old | Queued
    assert rows['Saturated'] == ['28', '3', '25', '0', '0', '3']
    assert rows['Starved'] == ['1', '0', '1', '0', '0', '0']
    assert rows['Broken'] == ['error', '0', '0', '0', '0', '0'], 'an errored query still renders'
    assert rows['Never'] == ['never searched', '0', '0', '0', '0', '0']
    # Section 2 carries the same breakdown inline.
    assert 'already_processed 25, new 3' in path.read_text()


def test_write_run_audit_log_includes_opus_findings(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {})
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {'q': 1})
    monkeypatch.setattr(tools, '_listing_records', {})

    path = tools.write_run_audit_log(
        queries=['q'], applied_jobs_in_horizon=1, applied_jobs_total=1, funnel={},
        audit_findings=[{
            'pool': 'filtered', 'company': 'MissedCorp', 'title': 'Principal AI Engineer',
            'url': 'https://example.com/1', 'opus_rating': 5,
            'verdict': 'FALSE NEGATIVE', 'reasoning': 'strong match on agentic AI',
        }],
        run_dir=tmp_path, timestamp=datetime(2026, 7, 28, 17, 5, 0),
    )

    text = path.read_text()
    assert 'Opus audit' in text
    assert 'FALSE NEGATIVE' in text
    assert 'MissedCorp' in text
    assert 'strong match on agentic AI' in text


# ---------------------------------------------------------------------------
# query generation: provider chain, cap, IC-only instructions
# ---------------------------------------------------------------------------

def test_query_instructions_exclude_managerial_titles():
    text = agent.build_query_generation_instructions()
    assert 'INDIVIDUAL CONTRIBUTOR' in text
    for managerial in ('Manager', 'Head of', 'Director', 'VP'):
        assert managerial in text, f'{managerial} should be named as an exclusion'
    assert str(agent.MAX_SEARCH_QUERIES) in text


async def test_generate_search_queries_uses_openrouter(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'MODEL_NAME_QUERY', OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE)
    monkeypatch.setattr(agent, 'load_resume', lambda: 'resume text')
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', tmp_path / 'missing.md')
    monkeypatch.setattr(tools, 'applied_jobs_summary', lambda: '- Staff AI Engineer — Acme (2026-07-01)')

    async def fake_chat(prompt, **kwargs):
        assert 'Staff AI Engineer — Acme' in prompt, 'applied jobs must reach the prompt'
        return '{"queries": ["Principal AI Engineer", "Staff AI Engineer"]}', 0.01

    monkeypatch.setattr(agent, 'chat_openrouter', fake_chat)
    stage_stats = {'cost': 0.0}

    queries = await agent.generate_search_queries(stage_stats)

    assert queries == ['Principal AI Engineer', 'Staff AI Engineer']
    assert stage_stats['cost'] == 0.01


async def test_generate_search_queries_enforces_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'MODEL_NAME_QUERY', OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE)
    monkeypatch.setattr(agent, 'MAX_SEARCH_QUERIES', 3)
    monkeypatch.setattr(agent, 'load_resume', lambda: 'resume')
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', tmp_path / 'missing.md')
    monkeypatch.setattr(tools, 'applied_jobs_summary', lambda: '')

    async def fake_chat(prompt, **kwargs):
        return '{"queries": ["a", "b", "c", "d", "e"]}', 0.0

    monkeypatch.setattr(agent, 'chat_openrouter', fake_chat)

    queries = await agent.generate_search_queries({'cost': 0.0})

    assert queries == ['a', 'b', 'c'], 'every extra query costs two live LinkedIn searches'


async def test_generate_search_queries_falls_back_to_anthropic(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, 'MODEL_NAME_QUERY', OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE)
    monkeypatch.setattr(agent, 'load_resume', lambda: 'resume')
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', tmp_path / 'missing.md')
    monkeypatch.setattr(tools, 'applied_jobs_summary', lambda: '')

    async def broken_chat(prompt, **kwargs):
        raise RuntimeError('OpenRouter MCP server down')

    monkeypatch.setattr(agent, 'chat_openrouter', broken_chat)

    # The Anthropic fallback path submits queries via an MCP tool call, which this stub does
    # not emulate — so it yields nothing and the function must raise rather than return [].
    called = {}

    async def fake_sdk_query(prompt, options):
        called['prompt'] = prompt
        return
        yield  # make it an async generator

    monkeypatch.setattr(agent, 'sdk_query', fake_sdk_query)

    with pytest.raises(ValueError, match='no queries on any provider'):
        await agent.generate_search_queries({'cost': 0.0})

    assert 'Call the submit_search_queries tool' in called['prompt'], 'Anthropic fallback was attempted'


# ---------------------------------------------------------------------------
# run_scraper: per-query requests on a single shared session
# ---------------------------------------------------------------------------

def _scrape(client, queries, stats=None):
    """Drive run_scraper's Anthropic path.

    run_scraper now takes a provider-specific `run_pass` so the OpenRouter and Anthropic scrapers
    share its per-query loop, pacing, region verification and recovery. These tests exercise that
    shared logic through the Anthropic adapter.
    """
    stats = stats if stats is not None else {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0}
    return agent.run_scraper(agent._anthropic_run_pass(client, stats), queries, stats)


class _FakeScraperClient:
    """Records each query() call so we can assert on request boundaries."""

    def __init__(self, listings_per_query=None):
        self.requests = []
        self.listings_per_query = listings_per_query or {}

    async def query(self, instruction):
        self.requests.append(instruction)
        # Simulate the scraper inspecting listings for whichever query this request names.
        # Both counters move, exactly as do_check_and_record_job moves them: the CALL count and
        # the DISTINCT id set. run_scraper's retry decision reads the distinct one, so a fake that
        # only fed the call count would make every query look like a zero-listing failure.
        for name, count in self.listings_per_query.items():
            if f'"{name}"' in instruction:
                for i in range(count):
                    key = f'{name}-{len(tools._check_status_counts)}-{i}'
                    tools._check_status_counts[key] = 1
                    tools._distinct_listing_ids.add(f'linkedin/{name}-{i}')
                break

    async def receive_response(self):
        return
        yield


def test_scraper_browser_find_and_snapshot_get_their_own_larger_caps():
    """browser_find returns only matches, so cutting it at the shared cap cut signal (2026-09-24);
    a pane-less snapshot must keep every result card (2026-09-30)."""
    assert scrape_openrouter._result_cap('browser_find') == config.SCRAPER_FIND_RESULT_MAX_CHARS
    assert scrape_openrouter._result_cap('browser_snapshot') == config.SCRAPER_SNAPSHOT_RESULT_MAX_CHARS
    assert scrape_openrouter._result_cap('browser_evaluate') == config.SCRAPER_TOOL_RESULT_MAX_CHARS
    assert config.SCRAPER_FIND_RESULT_MAX_CHARS > config.SCRAPER_TOOL_RESULT_MAX_CHARS
    assert config.SCRAPER_SNAPSHOT_RESULT_MAX_CHARS > config.SCRAPER_TOOL_RESULT_MAX_CHARS


async def test_run_scraper_sends_one_request_per_query(monkeypatch):
    """Each query needs its own turn budget; a shared request starves the later ones."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    queries = ['Principal AI Engineer', 'Staff AI Engineer', 'Lead AI Engineer']
    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    client = _FakeScraperClient(dict.fromkeys(queries, healthy))

    await _scrape(client, queries)

    assert len(client.requests) == 3, 'one request per query, no retries at a healthy yield'
    for q in queries:
        assert any(f'"{q}"' in r for r in client.requests), f'{q} was never searched'
        assert tools._queries_searched[q] == healthy


async def test_run_scraper_retries_empty_query_then_continues(monkeypatch):
    """A query that comes back empty gets one retry, and must not abort the remaining queries."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    queries = ['Empty Query', 'Good Query']
    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    client = _FakeScraperClient({'Good Query': healthy})

    await _scrape(client, queries)

    empty_requests = [r for r in client.requests if 'Empty' in r]
    assert len(empty_requests) == 2, 'empty query should be retried once'
    retry = empty_requests[1]
    assert 'NO location filter' in retry, 'retry drops the location filter'
    # A block page must never be retried into -- that is what risks the account.
    assert 'do NOT retry' in retry and 'CAPTCHA' in retry
    assert tools._queries_searched['Empty Query'] == 0
    assert tools._queries_searched['Good Query'] == healthy, 'later queries still run'


async def test_no_scraper_prompt_ever_instructs_clicking_the_results_list(monkeypatch):
    """Covers the per-query and recovery prompts, not just the system prompt.

    The system prompt said "never click the results list" while the recovery prompt still said
    "select them one at a time with browser_click" — exactly the instruction that dismissed three
    of the user's real jobs. Every prompt the scraper can receive has to agree.
    """
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    # 0 listings and 1 listing exercise both recovery branches; a healthy query the normal path.
    client = _FakeScraperClient({'Healthy': agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20, 'Starved': 1})
    await _scrape(client, ['Empty', 'Starved', 'Healthy'])

    prompts = [*client.requests, agent.build_scraper_instructions()]
    assert len(client.requests) == 5, 'two retries plus three initial passes'
    for prompt in prompts:
        assert 'currentJobId' not in prompt, f'click-to-reveal leaked into a prompt: {prompt[:120]}'
        lowered = prompt.lower()
        if 'browser_click' in lowered:
            assert 'never click' in lowered or 'do not click' in lowered, \
                f'a prompt mentions browser_click without forbidding it: {prompt[:160]}'


async def test_run_scraper_retries_low_yield_query_keeping_filters(monkeypatch):
    """A query returning a handful of listings is retried too, not just one returning zero.

    This is the 2026-08-11 signature: LinkedIn stopped honouring filter params in the search
    URL, every search resolved to the same page, and each query inspected exactly ONE listing.
    A `seen == 0` trigger sails straight past that and the run reports itself a success.
    """
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    queries = ['Starved Query']
    client = _FakeScraperClient({'Starved Query': 1})

    await _scrape(client, queries)

    assert len(client.requests) == 2, 'a 1-listing query must be retried'
    retry = client.requests[1]
    assert 'inspected only 1 job listing' in retry
    assert 'identical list of jobs' in retry, 'retry names the repeated-results failure mode'
    # Unlike the zero-listing case, the page loaded — filters must be re-applied, not dropped.
    assert 'clear the Location filter' not in retry


def _reset_scraper_state(monkeypatch):
    for name, value in [
        ('_check_status_counts', {}), ('_distinct_listing_ids', set()), ('_search_ids', {}),
        ('_ui_alerts', []), ('_search_reports', []), ('_candidates', []),
        ('_candidates_per_query', {}), ('_queries_searched', {}), ('_check_status_per_query', {}),
    ]:
        monkeypatch.setattr(tools, name, value)


class _PerPassScraperClient:
    """Yields a different number of NEW distinct listings on each pass of the one query."""

    def __init__(self, query_name, per_pass):
        self.query_name = query_name
        self.per_pass = list(per_pass)
        self.requests = []
        self.next_id = 0

    async def query(self, instruction):
        self.requests.append(instruction)
        count = self.per_pass[len(self.requests) - 1] if len(self.requests) <= len(self.per_pass) else 0
        for _ in range(count):
            tools._check_status_counts[f'k{self.next_id}'] = 1
            tools._distinct_listing_ids.add(f'linkedin/{self.query_name}-{self.next_id}')
            self.next_id += 1

    async def receive_response(self):
        return
        yield


def _low_listing_alerts():
    return [a for a in tools._ui_alerts if a['kind'] == 'low_listings']


async def test_low_listings_alert_not_raised_when_recovery_fixes_the_query(monkeypatch, tmp_path):
    """First pass below the retry floor, recovery rescues it: retry yes, alert no."""
    _reset_scraper_state(monkeypatch)
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    client = _PerPassScraperClient('Niche Query', [2, 20])

    await _scrape(client, ['Niche Query'])

    assert len(client.requests) == 2, 'a below-floor first pass is still retried'
    assert tools._queries_searched['Niche Query'] == 22
    assert _low_listing_alerts() == []


async def test_low_listings_alert_raised_after_failed_recovery_with_expected_count(monkeypatch, tmp_path):
    _reset_scraper_state(monkeypatch)
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    history = [{'funnel': {'listings_distinct_per_query': {'Dead Query': n}}} for n in (40, 48, 50)]
    (tmp_path / 'cost_log.jsonl').write_text('\n'.join(json.dumps(r) for r in history) + '\n')
    client = _PerPassScraperClient('Dead Query', [0, 1])

    await _scrape(client, ['Dead Query'])

    alerts = _low_listing_alerts()
    assert len(alerts) == 1, alerts
    detail = alerts[0]['detail']
    assert 'only 1 distinct listing(s) after recovery (first pass 0' in detail
    assert f'below {config.SCRAPER_ALERT_MIN_LISTINGS_PER_QUERY}' in detail
    assert 'expected ~48 (distinct, median of last 3 runs)' in detail


def _write_cost_log(path, funnels):
    lines = [json.dumps({'stage': 'other', 'cost': 0.1})]  # a non-funnel row must be skipped
    lines += [json.dumps({'timestamp': '2026-09-01', 'funnel': f}) for f in funnels]
    (path / 'cost_log.jsonl').write_text('\n'.join(lines) + '\n')


def test_expected_listings_prefers_distinct_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    _write_cost_log(tmp_path, [
        {'listings_distinct_per_query': {'Q': 10}},
        {'listings_distinct_per_query': {'Q': 30}, 'check_status_per_query': {'Q': {'new': 999}}},
        {'listings_distinct_per_query': {'Q': 20}},
    ])
    assert agent.expected_listings_per_query('Q') == (20.0, 'distinct, median of last 3 runs')


def test_expected_listings_falls_back_to_checked_counts_and_labels_them(monkeypatch, tmp_path):
    """Old runs only have call counts -- a different unit, so the label must say so."""
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    _write_cost_log(tmp_path, [
        {'check_status_per_query': {'Q': {'new': 10, 'already_processed': 40}}},
        {'check_status_per_query': {'Q': {'new': 0, 'already_processed': 0}}},
        {'check_status_per_query': {'Q': {'new': 5, 'already_processed': 45}}},
    ])
    assert agent.expected_listings_per_query('Q') == (50.0, 'listings checked, median of last 3 runs')


def test_expected_listings_uses_only_the_last_limit_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    _write_cost_log(tmp_path, [{'listings_distinct_per_query': {'Q': n}} for n in (100, 100, 1, 2, 3)])
    assert agent.expected_listings_per_query('Q', limit=3) == (2.0, 'distinct, median of last 3 runs')


def test_expected_listings_unknown_query_uses_all_query_median(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    _write_cost_log(tmp_path, [
        {'listings_distinct_per_query': {'A': 10, 'B': 30}},
        {'listings_distinct_per_query': {'C': 20}},
    ])
    assert agent.expected_listings_per_query('New Query') == (20.0, 'distinct, all-query median')


def test_expected_listings_without_history(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, 'RUN_DIR', tmp_path)
    assert agent.expected_listings_per_query('Q') == (None, 'no history yet')
    (tmp_path / 'cost_log.jsonl').write_text('')
    assert agent.expected_listings_per_query('Q') == (None, 'no history yet')


def test_listings_distinct_per_query_leaves_out_failed_queries(monkeypatch):
    monkeypatch.setattr(tools, '_queries_searched', {'Good': 12, 'Broken': 'error'})
    assert agent._listings_distinct_per_query() == {'Good': 12}


async def test_run_scraper_records_per_query_check_status(monkeypatch):
    """Per-query dedup counts: a saturated query must be distinguishable from a starved one."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    class _StatusClient:
        """Records real check_and_record_job statuses rather than opaque unique keys."""

        def __init__(self):
            self.requests = []

        async def query(self, instruction):
            self.requests.append(instruction)
            if '"Saturated"' in instruction:
                tools._check_status_counts['already_processed'] = (
                    tools._check_status_counts.get('already_processed', 0) + 25
                )
                tools._check_status_counts['new'] = tools._check_status_counts.get('new', 0) + 3
                # Distinct ids move too, as do_check_and_record_job moves them. The retry
                # decision reads the distinct set, not these call counts.
                for i in range(28):
                    tools._distinct_listing_ids.add(f'linkedin/saturated-{i}')

        async def receive_response(self):
            return
            yield

    await _scrape(_StatusClient(), ['Saturated'])

    assert tools._check_status_per_query['Saturated'] == {'already_processed': 25, 'new': 3}
    assert tools._queries_searched['Saturated'] == 28


async def test_run_scraper_survives_a_failing_query(monkeypatch):
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    class ExplodingClient(_FakeScraperClient):
        async def query(self, instruction):
            if '"Bad Query"' in instruction:
                raise RuntimeError('session died')
            await super().query(instruction)

    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    client = ExplodingClient({'Good Query': healthy})
    await _scrape(client, ['Bad Query', 'Good Query'])

    assert tools._queries_searched['Bad Query'] == 'error'
    assert tools._queries_searched['Good Query'] == healthy, 'a failed query must not abort the rest'
    assert 'Bad Query' not in tools._check_status_per_query, 'an errored query records no counts'


async def test_stage_1b_falls_back_to_anthropic_when_the_provider_runs_out_of_credit(monkeypatch):
    """THE test for the 2026-09-02 defect: the fallback must actually be reached.

    Everything else about that run was already covered by a test somewhere. What was not, and
    could not be while the dispatch was inline in run_non_interactive, is the only question that
    mattered: when OpenRouter returns 402 for every call, does the Anthropic scraper run? It did
    not, for the whole life of that code path.
    """
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_scrape_per_query', {})
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', OPENROUTER_MODEL_NAME_SCRAPER)

    # The failure must originate INSIDE run_scraper's per-query loop, which is where the real
    # 402 arrived and where it was swallowed. Raising at the session boundary instead would pass
    # against the buggy code too, since that `except` always worked -- it was simply never reached.
    for name in ('_check_status_counts', '_distinct_listing_ids', '_search_ids', '_search_reports',
                 '_candidates', '_candidates_per_query', '_queries_searched', '_query_errors',
                 '_check_status_per_query'):
        monkeypatch.setattr(tools, name, set() if 'ids' in name else {})
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0, 0))

    class _Session:
        async def __aenter__(self):
            async def call(*a, **k):
                return ''
            async def list_tools():
                return []
            call.list_tools = list_tools
            return call

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(agent, 'mcp_session', lambda url: _Session())

    def failing_run_pass(browser_call, tool_defs, stats, per_query):
        async def run_pass(instruction):
            raise triage.ProviderUnavailableError(
                'OpenRouter chat failed: 402 Client Error: Payment Required for url: x', status=402)
        return run_pass

    monkeypatch.setattr(agent, '_openrouter_run_pass', failing_run_pass)

    fallback_ran = []

    async def fake_anthropic(playwright_mcp, queries, stats):
        fallback_ran.append(queries)

    monkeypatch.setattr(agent, '_run_anthropic_scraper', fake_anthropic)

    used_openrouter = await agent.run_stage_1b(
        1234, {'type': 'http'}, ['Q1', 'Q2'], {'cost': 0.0})

    assert used_openrouter is False
    assert fallback_ran == [['Q1', 'Q2']], 'the Anthropic scraper must actually run'
    kinds = [a['kind'] for a in tools._ui_alerts]
    assert 'provider_fallback' in kinds, 'a silent fallback bills Anthropic prices unnoticed'


async def test_stage_1b_reports_the_cause_through_an_exception_group(monkeypatch):
    """The failure crosses the browser MCP session's task group, so it arrives WRAPPED.

    `str(ExceptionGroup)` is "unhandled errors in a TaskGroup (1 sub-exception)" — logging the
    exception directly would record none of the 402, which is how a dead triage model stayed
    invisible for ten days (2026-08-25). The alert must carry the real cause.
    """
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_scrape_per_query', {})
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', OPENROUTER_MODEL_NAME_SCRAPER)

    class _BoomGroup:
        async def __aenter__(self):
            raise ExceptionGroup('unhandled errors in a TaskGroup', [
                triage.ProviderUnavailableError('402 Client Error: Payment Required', status=402)])

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(agent, 'mcp_session', lambda url: _BoomGroup())

    async def fake_anthropic(playwright_mcp, queries, stats):
        pass

    monkeypatch.setattr(agent, '_run_anthropic_scraper', fake_anthropic)
    await agent.run_stage_1b(1234, {'type': 'http'}, ['Q1'], {'cost': 0.0})

    detail = next(a['detail'] for a in tools._ui_alerts if a['kind'] == 'provider_fallback')
    assert '402' in detail, 'the wrapped cause must be unwrapped, not swallowed by the group'


async def test_stage_1b_does_not_fall_back_when_the_provider_is_healthy(monkeypatch):
    """The fallback is ~9x more expensive per search; it must not fire on a healthy run."""
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_scrape_per_query', {})
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', OPENROUTER_MODEL_NAME_SCRAPER)

    class _Ok:
        async def __aenter__(self):
            async def call(*a, **k):
                return ''
            call.list_tools = _empty
            return call

        async def __aexit__(self, *exc):
            return False

    async def _empty():
        return []

    monkeypatch.setattr(agent, 'mcp_session', lambda url: _Ok())

    async def fake_run_scraper(run_pass, queries, stats):
        return None

    fallback_ran = []

    async def fake_anthropic(playwright_mcp, queries, stats):
        fallback_ran.append(queries)

    monkeypatch.setattr(agent, 'run_scraper', fake_run_scraper)
    monkeypatch.setattr(agent, '_run_anthropic_scraper', fake_anthropic)

    assert await agent.run_stage_1b(1234, {'type': 'http'}, ['Q1'], {'cost': 0.0}) is True
    assert fallback_ran == [], 'a healthy OpenRouter run must never pay Anthropic prices'
    assert not tools._ui_alerts


async def test_run_scraper_aborts_the_whole_loop_on_a_provider_failure(monkeypatch):
    """A 402 breaks EVERY query, so the loop must abort and let the caller fall back.

    Regression for 2026-09-02: OpenRouter ran out of credit mid-run, the per-query handler
    swallowed all six 402s, run_scraper returned normally, `scraped = True` was set, and the
    Anthropic fallback that exists precisely for this never fired. The run reported 0 jobs.
    """
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_query_errors', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0, 0))

    attempted = []

    async def run_pass(instruction):
        attempted.append(instruction)
        raise triage.ProviderUnavailableError(
            'OpenRouter chat failed: 402 Client Error: Payment Required for url: x', status=402)

    queries = ['Q1', 'Q2', 'Q3']
    with pytest.raises(triage.ProviderUnavailableError):
        await agent.run_scraper(run_pass, queries, {'cost': 0.0})

    assert len(attempted) == 1, 'the remaining queries must not each burn a doomed request'
    # Every query must read as failed, including the ones the loop never reached: a query that
    # never ran must not be reported as "searched, found nothing" in audit sections 2 and 3.
    assert [tools._queries_searched.get(q) for q in queries] == ['error'] * 3
    assert '402' in tools._query_errors['Q3'], 'the cause must be carried, not just the fact'


async def test_run_scraper_still_survives_a_single_query_failure(monkeypatch):
    """The abort above must not regress per-query resilience: a one-off failure still continues."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_query_errors', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0, 0))

    attempted = []

    async def run_pass(instruction):
        attempted.append(instruction)
        raise RuntimeError('page did not load')

    await agent.run_scraper(run_pass, ['Q1', 'Q2', 'Q3'], {'cost': 0.0})

    assert len(attempted) == 3, 'a one-off failure must not abort the remaining queries'
    assert tools._query_errors['Q2'] == 'page did not load'


@pytest.mark.parametrize('status,expected', [
    (402, triage.ProviderUnavailableError),
    (401, triage.ProviderUnavailableError),
    (403, triage.ProviderUnavailableError),
    (429, triage.ProviderUnavailableError),
    (500, RuntimeError),
    (404, RuntimeError),
])
def test_provider_error_classifies_on_status_not_reason_phrase(status, expected):
    """Match the STATUS: the reason phrase is upstream's text and can change without notice."""
    err = triage.provider_error(
        'OpenRouter chat failed',
        f'{status} Client Error: Some Upstream Wording for url: https://openrouter.ai/api/v1/x')
    assert type(err) is expected
    if expected is triage.ProviderUnavailableError:
        assert err.status == status


def test_provider_error_reads_a_status_out_of_a_json_body():
    err = triage.provider_error('OpenRouter chat failed', '{"error": {"code": 402, "message": "x"}}')
    assert isinstance(err, triage.ProviderUnavailableError)


def test_provider_error_without_any_status_is_not_systemic():
    """No status named means no evidence the PROVIDER is down - do not abort the whole run."""
    err = triage.provider_error('OpenRouter chat failed', 'connection reset by peer')
    assert type(err) is RuntimeError


def test_assess_run_health_alerts_when_every_query_failed():
    """The exact funnel of the 2026-09-02 run, which returned no alerts at all.

    Saturation is guarded by `if distinct:`, so zero listings skipped every existing check and
    the audit log printed "No alerts: page structure sound, regions distinct, yield acceptable"
    while all six queries had died on a 402.
    """
    alerts = agent.assess_run_health({
        'queries_generated': 6,
        'listings_seen': 0,
        'listings_distinct': 0,
        'region_overlap': {},
        'ui_alerts': [],
        'check_status': {},
        'queries_failed': {f'Q{i}': '402 Client Error: Payment Required' for i in range(1, 7)},
    })
    assert alerts, 'a run where every query failed must never report as healthy'
    joined = ' '.join(alerts)
    assert 'ALL 6' in joined
    assert '402' in joined, 'the alert must name the cause, which previously reached only the run log'


def test_assess_run_health_reports_a_partial_query_failure():
    alerts = agent.assess_run_health({
        'queries_generated': 6, 'listings_distinct': 0,
        'queries_failed': {'Q1': 'boom', 'Q2': 'boom'},
    })
    assert any('2 of 6' in a for a in alerts)


def test_partial_query_failure_alert_does_not_claim_no_search_completed():
    """Regression for 2026-09-10: one 502 on one query, five queries fine, 156 distinct listings.

    The alert said "no search completed, so 0 listings is not 'nothing new today'" -- wording
    written for the all-failed case, false on both counts here.
    """
    alerts = agent.assess_run_health({
        'queries_generated': 6, 'listings_distinct': 156,
        'check_status': {'new': 58, 'already_processed': 197, 'already_applied': 20},
        'queries_failed': {'Principal Data Scientist': 'OpenRouter chat failed: 502 Server Error: Bad Gateway'},
    })
    failure = next(a for a in alerts if 'QUERIES FAILED' in a)
    assert '1 of 6' in failure
    assert 'Principal Data Scientist' in failure, 'name the query so the reader need not open the audit log'
    assert 'the other 5 completed' in failure
    assert 'no search completed' not in failure
    assert '0 listings' not in failure
    assert '502' in failure


def test_partial_query_failure_alert_caps_the_names_it_lists():
    alerts = agent.assess_run_health({
        'queries_generated': 6,
        'queries_failed': {f'Q{i}': 'boom' for i in range(1, 6)},
    })
    failure = next(a for a in alerts if 'QUERIES FAILED' in a)
    assert 'Q1, Q2, Q3 +2 more' in failure


async def test_failed_query_keeps_the_counts_it_recorded_before_failing(monkeypatch):
    """Regression for 2026-09-10: a 502 at iteration 71 hit after the Canada search was recorded.

    Those 25 listings (2 queued) were in the run-global counts but not the per-query ones, so
    audit section 3 read "error | 0 listings | ... | 2 queued".
    """
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_query_errors', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0, 0))

    async def run_pass(instruction):
        tools._check_status_counts['already_processed'] = 20
        tools._check_status_counts['new'] = 2
        raise RuntimeError('OpenRouter chat failed: 502 Server Error: Bad Gateway')

    await agent.run_scraper(run_pass, ['Q1'], {'cost': 0.0})

    assert tools._queries_searched['Q1'] == 'error', 'the sentinel audit sections 2/3 switch on must stay'
    assert tools._check_status_per_query['Q1'] == {'already_processed': 20, 'new': 2}


async def test_provider_abort_keeps_the_in_flight_query_counts(monkeypatch):
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_query_errors', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0, 0))

    async def run_pass(instruction):
        tools._check_status_counts['new'] = 3
        raise triage.ProviderUnavailableError('402 Client Error: Payment Required', status=402)

    with pytest.raises(triage.ProviderUnavailableError):
        await agent.run_scraper(run_pass, ['Q1', 'Q2'], {'cost': 0.0})

    assert tools._check_status_per_query['Q1'] == {'new': 3}
    assert 'Q2' not in tools._check_status_per_query, 'a query never reached recorded nothing'


async def test_a_query_the_fallback_re_runs_is_not_still_reported_failed(monkeypatch):
    """2026-09-28: the OpenRouter loop aborted on a 429 and stamped every pending query 'error';
    the Anthropic fallback then searched all of them, yet run health said "4 of 6 QUERIES FAILED"."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_query_errors', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0, 0))

    async def aborting_pass(instruction):
        raise triage.ProviderUnavailableError('429 Client Error: Too Many Requests', status=429)

    with pytest.raises(triage.ProviderUnavailableError):
        await agent.run_scraper(aborting_pass, ['Q1', 'Q2'], {'cost': 0.0})
    assert set(tools._query_errors) == {'Q1', 'Q2'}

    async def fallback_pass(instruction):
        return None

    await agent.run_scraper(fallback_pass, ['Q1', 'Q2'], {'cost': 0.0})
    assert tools._query_errors == {}, 'the re-run succeeded, so no query failed'
    assert 'error' not in tools._queries_searched.values()


def _stage_1b_with_a_provider_failure(monkeypatch, completed: dict) -> list[list[str]]:
    '''Wire run_stage_1b so the OpenRouter path dies with `completed` already searched; returns the
    list the Anthropic fallback is handed (one entry per fallback run).'''
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_scrape_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(agent, 'MODEL_NAME_SCRAPER', OPENROUTER_MODEL_NAME_SCRAPER)

    class _Session:
        async def __aenter__(self):
            async def call(*a, **k):
                return ''

            async def list_tools():
                return []
            call.list_tools = list_tools
            return call

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(agent, 'mcp_session', lambda url: _Session())

    async def dying_scraper(run_pass, queries, stats):
        tools._queries_searched.update(completed)
        raise triage.ProviderUnavailableError('429 Client Error: Too Many Requests', status=429)

    monkeypatch.setattr(agent, 'run_scraper', dying_scraper)
    fallback_ran: list[list[str]] = []

    async def fake_anthropic(playwright_mcp, queries, stats):
        fallback_ran.append(queries)

    monkeypatch.setattr(agent, '_run_anthropic_scraper', fake_anthropic)
    return fallback_ran


async def test_fallback_runs_only_the_queries_openrouter_did_not_finish(monkeypatch):
    '''2026-09-28: the 429 came on query 3 and the fallback re-scraped queries 1 and 2 as well --
    twice the LinkedIn traffic for them, and most of what is now the whole fallback budget.'''
    fallback_ran = _stage_1b_with_a_provider_failure(
        monkeypatch, {'Q1': 50, 'Q2': 14, 'Q3': 'error', 'Q4': 'error'})

    assert await agent.run_stage_1b(1234, {'type': 'http'}, ['Q1', 'Q2', 'Q3', 'Q4'], {'cost': 0.0}) is False

    assert fallback_ran == [['Q3', 'Q4']]
    detail = next(a['detail'] for a in tools._ui_alerts if a['kind'] == 'provider_fallback')
    assert '2 unfinished' in detail and '2 already complete' in detail and '429' in detail


async def test_fallback_runs_every_query_when_none_completed(monkeypatch):
    fallback_ran = _stage_1b_with_a_provider_failure(monkeypatch, {})

    await agent.run_stage_1b(1234, {'type': 'http'}, ['Q1', 'Q2'], {'cost': 0.0})

    assert fallback_ran == [['Q1', 'Q2']]


async def test_fallback_does_not_start_when_every_query_completed(monkeypatch):
    '''A zero-listing query counts as completed (int 0): searched, found nothing.'''
    fallback_ran = _stage_1b_with_a_provider_failure(monkeypatch, {'Q1': 50, 'Q2': 0})

    assert await agent.run_stage_1b(1234, {'type': 'http'}, ['Q1', 'Q2'], {'cost': 0.0}) is False

    assert fallback_ran == []
    assert [a['kind'] for a in tools._ui_alerts] == ['provider_fallback'], 'the failure is still reported'


def _two_pass_scraper(first_turns: int, recovery_listings: int, recovery_verifies: bool):
    '''A run_pass whose first pass does nothing in `first_turns` turns; the recovery pass records
    `recovery_listings` distinct listings and, if asked, verifies every configured region.'''
    passes = []

    async def run_pass(instruction):
        passes.append(instruction)
        if len(passes) == 1:
            return first_turns
        for i in range(recovery_listings):
            tools._check_status_counts[f'rec-{i}'] = 1
            tools._distinct_listing_ids.add(f'linkedin/rec-{i}')
        if recovery_verifies:
            for region in preferences.search_regions():
                tools._search_reports.append({'query': 'Staff AI Engineer', 'region': region['name']})
        return agent.SCRAPER_MIN_TURNS_PER_QUERY
    run_pass.passes = passes
    return run_pass


def _alert_kinds() -> list[str]:
    return [a['kind'] for a in tools._ui_alerts]


async def test_a_region_the_recovery_pass_verified_is_not_reported_unverified(monkeypatch):
    '''2026-09-28: the first pass never started, the alert was raised, the recovery pass verified
    both regions and harvested 50 listings -- and the run still said SEARCH NOT VERIFIED.'''
    _reset_scrape_globals(monkeypatch)
    run_pass = _two_pass_scraper(first_turns=2, recovery_listings=25, recovery_verifies=True)

    await agent.run_scraper(run_pass, ['Staff AI Engineer'], {'cost': 0.0})

    assert len(run_pass.passes) == 2
    assert 'unverified_search' not in _alert_kinds()
    assert 'stopped_early' not in _alert_kinds(), 'a fully recovered query needs no attention'
    assert tools._queries_searched['Staff AI Engineer'] == 25


async def test_a_region_no_pass_verified_is_still_reported(monkeypatch):
    _reset_scrape_globals(monkeypatch)
    run_pass = _two_pass_scraper(first_turns=2, recovery_listings=25, recovery_verifies=False)

    await agent.run_scraper(run_pass, ['Staff AI Engineer'], {'cost': 0.0})

    assert _alert_kinds().count('unverified_search') == 1
    assert 'stopped_early' in _alert_kinds(), 'an unverified query that bailed early is not recovered'


async def test_stopped_early_is_kept_when_recovery_finds_too_little(monkeypatch):
    _reset_scrape_globals(monkeypatch)
    too_few = agent.SCRAPER_MIN_LISTINGS_PER_QUERY - 1
    run_pass = _two_pass_scraper(first_turns=2, recovery_listings=too_few, recovery_verifies=True)

    await agent.run_scraper(run_pass, ['Staff AI Engineer'], {'cost': 0.0})

    assert 'stopped_early' in _alert_kinds()


def test_audit_log_reports_what_a_failed_query_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, '_queries_searched', {'Q1': 'error'})
    monkeypatch.setattr(tools, '_query_errors', {'Q1': '502 Server Error: Bad Gateway'})
    monkeypatch.setattr(tools, '_check_status_per_query', {'Q1': {'already_processed': 20, 'new': 2}})
    monkeypatch.setattr(tools, '_candidates_per_query', {'Q1': 2})
    monkeypatch.setattr(tools, '_listing_records', {})
    monkeypatch.setattr(tools, '_search_reports', [])

    path = tools.write_run_audit_log(
        queries=['Q1'], applied_jobs_in_horizon=0, applied_jobs_total=0,
        funnel={'queries_generated': 1, 'queries_failed': {'Q1': '502 Server Error: Bad Gateway'}},
        run_dir=tmp_path)

    body = Path(path).read_text(encoding='utf-8')
    assert '**FAILED** — 502 Server Error: Bad Gateway (after recording' in body
    assert '| Q1 | error | 2 | 20 | 0 | 0 | 2 |' in body


def test_assess_run_health_stays_quiet_when_no_query_failed():
    assert agent.assess_run_health({'queries_generated': 6, 'queries_failed': {}}) == []


def test_attach_run_health_records_alerts_on_the_funnel():
    """Both exit paths go through this, so the zero-candidate branch cannot skip assessment."""
    funnel = {'queries_generated': 2, 'queries_failed': {'Q1': 'boom', 'Q2': 'boom'}}
    alerts = agent.attach_run_health(funnel)
    assert alerts and funnel['health_alerts'] == alerts


def test_audit_log_does_not_claim_no_alerts_when_queries_failed(tmp_path, monkeypatch):
    """Audit section 1b must name the failure instead of declaring the page structure sound."""
    monkeypatch.setattr(tools, '_queries_searched', {'Q1': 'error'})
    monkeypatch.setattr(tools, '_query_errors', {'Q1': '402 Client Error: Payment Required'})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(tools, '_listing_records', {})
    monkeypatch.setattr(tools, '_search_reports', [])

    funnel = {'queries_generated': 1, 'listings_distinct': 0,
              'queries_failed': {'Q1': '402 Client Error: Payment Required'}}
    agent.attach_run_health(funnel)
    path = tools.write_run_audit_log(
        queries=['Q1'], applied_jobs_in_horizon=0, applied_jobs_total=0, funnel=funnel,
        run_dir=tmp_path)

    body = Path(path).read_text(encoding='utf-8')
    assert 'No alerts' not in body
    assert '402' in body, 'the audit log must carry the cause of the failure'


# ---------------------------------------------------------------------------
# run lock: concurrent-run detection
# ---------------------------------------------------------------------------

def test_read_run_lock_none_when_absent(tmp_path):
    assert tools.read_run_lock(tmp_path / 'nolock.json') is None


def test_acquire_and_release_run_lock_roundtrip(tmp_path):
    lock = tmp_path / 'lock.json'
    assert tools.acquire_run_lock('non-interactive', lock_path=lock) is None

    active = tools.read_run_lock(lock)
    assert active['pid'] == os.getpid()
    assert active['mode'] == 'non-interactive'

    tools.release_run_lock(lock)
    assert not lock.exists()
    assert tools.read_run_lock(lock) is None


def test_acquire_run_lock_reports_conflict(tmp_path, monkeypatch):
    """A second run must be told who holds the lock, not silently proceed."""
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({
        'pid': os.getpid(), 'started_at': '2026-07-28T21:05:00',
        'mode': 'non-interactive', 'argv': 'job-search -n',
    }))

    conflict = tools.acquire_run_lock('non-interactive', lock_path=lock)

    assert conflict is not None
    assert conflict['argv'] == 'job-search -n'


def test_stale_lock_from_dead_pid_is_ignored(tmp_path, monkeypatch):
    """A crashed run must not block every future run with a leftover file."""
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({
        'pid': 999999, 'started_at': '2026-07-28T10:00:00', 'mode': 'non-interactive', 'argv': 'old',
    }))
    monkeypatch.setattr(tools, '_pid_alive', lambda pid: pid != 999999)

    assert tools.read_run_lock(lock) is None, 'dead holder means no run is active'
    assert tools.acquire_run_lock('non-interactive', lock_path=lock) is None, 'stale lock must not block'
    assert tools.read_run_lock(lock)['pid'] == os.getpid()


def test_corrupt_lock_is_treated_as_stale(tmp_path):
    lock = tmp_path / 'lock.json'
    lock.write_text('not json at all')
    assert tools.read_run_lock(lock) is None


def test_release_run_lock_does_not_delete_another_runs_lock(tmp_path, monkeypatch):
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({'pid': os.getpid() + 1, 'started_at': 'x', 'mode': 'n', 'argv': 'other'}))
    monkeypatch.setattr(tools, '_pid_alive', lambda pid: True)

    tools.release_run_lock(lock)

    assert lock.exists(), 'must never release a lock we do not hold'


def test_force_overrides_active_lock(tmp_path, monkeypatch):
    lock = tmp_path / 'lock.json'
    lock.write_text(json.dumps({'pid': os.getpid() + 1, 'started_at': 'x', 'mode': 'n', 'argv': 'other'}))
    monkeypatch.setattr(tools, '_pid_alive', lambda pid: True)

    assert tools.acquire_run_lock('non-interactive', force=True, lock_path=lock) is None
    assert tools.read_run_lock(lock)['pid'] == os.getpid()


def test_describe_run_status_both_states(tmp_path, monkeypatch):
    lock = tmp_path / 'lock.json'
    assert 'No job search run is currently active' in tools.describe_run_status(lock)

    tools.acquire_run_lock('non-interactive', lock_path=lock)
    status = tools.describe_run_status(lock)
    assert 'A run IS active' in status
    assert str(os.getpid()) in status


def test_describe_run_status_falls_back_to_process_scan(tmp_path, monkeypatch):
    """A run started before the lock existed still has to be reported as running."""
    monkeypatch.setattr(tools, 'find_run_processes', lambda: [
        '13779  /path/.venv/bin/python3 /path/agent_job_search1/.venv/bin/job-search -n'
    ])
    status = tools.describe_run_status(tmp_path / 'no_lock.json')
    assert 'ARE running' in status
    assert '13779' in status


def test_find_run_processes_requires_project_marker(monkeypatch):
    """An unrelated main.py on the machine must not be mistaken for a job search run."""
    fake_ps = (
        '  111 /usr/bin/python3 /some/other/project/main.py\n'
        '  222 /path/agent_job_search1/.venv/bin/job-search -n\n'
        '  333 grep job-search\n'
    )

    class FakeCompleted:
        stdout = fake_ps

    monkeypatch.setattr(tools.subprocess, 'run', lambda *a, **k: FakeCompleted())

    found = tools.find_run_processes()

    assert len(found) == 1
    assert found[0].startswith('222')


def test_find_run_processes_survives_ps_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError('ps unavailable')

    monkeypatch.setattr(tools.subprocess, 'run', boom)
    assert tools.find_run_processes() == []


# ---------------------------------------------------------------------------
# stage cost accounting
# ---------------------------------------------------------------------------

def _result_msg(session_id: str, cost: float | None, **usage) -> ResultMessage:
    return ResultMessage(
        subtype='success', duration_ms=1, duration_api_ms=1, is_error=False,
        num_turns=1, session_id=session_id, total_cost_usd=cost,
        usage=usage or None,
    )


class _BillingScraperClient(_FakeScraperClient):
    """One shared SDK session whose cumulative total_cost_usd rises by `cost_per_query` per request.

    Each query yields a healthy listing count, so no low-yield retry adds requests of its own.
    """

    def __init__(self, cost_per_query: float, queries: list[str]):
        super().__init__(dict.fromkeys(queries, agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20))
        self.cost_per_query = cost_per_query

    async def receive_response(self):
        yield _result_msg('fallback-session', self.cost_per_query * len(self.requests))


async def test_anthropic_fallback_stops_at_its_budget(monkeypatch):
    """On 2026-09-28 the fallback billed ~$4.96 of a $5.30 run with nothing to stop it."""
    _reset_scraper_state(monkeypatch)
    monkeypatch.setattr(tools, '_query_errors', {})
    queries = ['Q1', 'Q2', 'Q3', 'Q4']
    budget = 2.0
    client = _BillingScraperClient(1.0, queries)
    stats = {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0,
             'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}

    with pytest.raises(agent.ScraperBudgetExceededError, match='SCRAPER_ANTHROPIC_FALLBACK_MAX_COST_USD'):
        await agent.run_scraper(agent._anthropic_run_pass(client, stats, budget), queries, stats)

    assert len(client.requests) == 2, 'no query may start once the budget is spent'
    assert '"Q2"' in client.requests[1], 'the second request must be Q2, not a retry of Q1'
    assert stats['cost'] == pytest.approx(budget)
    for skipped in ['Q3', 'Q4']:
        assert tools._queries_searched[skipped] == 'error', f'{skipped} must not read as searched-and-empty'


async def test_anthropic_fallback_budget_counts_only_its_own_spend(monkeypatch):
    """OpenRouter spend already in the stage totals must not use up the fallback's budget."""
    _reset_scraper_state(monkeypatch)
    monkeypatch.setattr(tools, '_query_errors', {})
    queries = ['Q1', 'Q2']
    client = _BillingScraperClient(0.1, queries)
    stats = {'cost': 50.0, 'input_tokens': 0, 'output_tokens': 0,
             'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}

    await agent.run_scraper(agent._anthropic_run_pass(client, stats, 1.0), queries, stats)

    assert len(client.requests) == 2


def test_fallback_and_budget_alerts_reach_run_health():
    funnel = {'ui_alerts': [
        {'kind': 'provider_fallback', 'query': '(all)', 'region': '(all)', 'detail': '429'},
        {'kind': 'scrape_budget', 'query': '(all)', 'region': '(all)', 'detail': 'spent $1.50'},
    ], 'queries_generated': 1, 'listings_seen': 25, 'listings_distinct': 25, 'region_overlap': {},
        'check_status': {}}
    alerts = agent.assess_run_health(funnel)
    assert any('FELL BACK TO ANTHROPIC' in a and '429' in a for a in alerts)
    assert any('BUDGET SPENT' in a and '$1.50' in a for a in alerts)


def test_cost_charged_once_for_cumulative_session():
    """Stage 1b sends one request per query on ONE shared session, and total_cost_usd is
    cumulative for that session. Naive summation billed 1+2+3 instead of 3."""
    stats = agent.new_stage_stats()
    for cumulative in (1.0, 2.5, 6.6):
        agent.accumulate_stage_stats(stats, _result_msg('sess-a', cumulative))
    assert stats['cost'] == pytest.approx(6.6)


def test_cost_accumulates_across_distinct_sessions():
    """Extraction opens a fresh session per job; each session's total is its own charge,
    even when a later session reports a smaller cumulative figure than an earlier one."""
    stats = agent.new_stage_stats()
    agent.accumulate_stage_stats(stats, _result_msg('job-1', 0.03))
    agent.accumulate_stage_stats(stats, _result_msg('job-2', 0.01))
    agent.accumulate_stage_stats(stats, _result_msg('job-3', 0.05))
    assert stats['cost'] == pytest.approx(0.09)


def test_cost_interleaved_sessions_billed_independently():
    stats = agent.new_stage_stats()
    agent.accumulate_stage_stats(stats, _result_msg('a', 1.0))
    agent.accumulate_stage_stats(stats, _result_msg('b', 10.0))
    agent.accumulate_stage_stats(stats, _result_msg('a', 3.0))   # +2.0
    agent.accumulate_stage_stats(stats, _result_msg('b', 12.0))  # +2.0
    assert stats['cost'] == pytest.approx(15.0)


def test_accumulate_returns_charged_delta():
    stats = agent.new_stage_stats()
    assert agent.accumulate_stage_stats(stats, _result_msg('s', 2.0)) == pytest.approx(2.0)
    assert agent.accumulate_stage_stats(stats, _result_msg('s', 5.0)) == pytest.approx(3.0)


def test_cost_decrease_within_session_charges_zero(caplog):
    """Defensive: total_cost_usd is expected to be non-decreasing within a session."""
    stats = agent.new_stage_stats()
    agent.accumulate_stage_stats(stats, _result_msg('s', 5.0))
    with caplog.at_level('WARNING'):
        agent.accumulate_stage_stats(stats, _result_msg('s', 2.0))
    assert stats['cost'] == pytest.approx(5.0)
    assert 'lower cumulative cost' in caplog.text
    # the high-water mark is kept, so a later genuine increase is charged from 5.0 not 2.0
    agent.accumulate_stage_stats(stats, _result_msg('s', 6.0))
    assert stats['cost'] == pytest.approx(6.0)


def test_missing_session_id_treated_as_per_request():
    stats = agent.new_stage_stats()
    agent.accumulate_stage_stats(stats, _result_msg('', 1.0))
    agent.accumulate_stage_stats(stats, _result_msg('', 1.0))
    assert stats['cost'] == pytest.approx(2.0)


def test_none_cost_is_a_noop():
    stats = agent.new_stage_stats()
    assert agent.accumulate_stage_stats(stats, _result_msg('s', None)) == 0.0
    assert stats['cost'] == 0.0


def test_usage_counters_still_summed_per_request():
    """Usage is a per-request delta and must keep summing, unlike cost."""
    stats = agent.new_stage_stats()
    for _ in range(3):
        agent.accumulate_stage_stats(
            stats, _result_msg('sess-a', 1.0, input_tokens=10, cache_read_input_tokens=100)
        )
    assert stats['input_tokens'] == 30
    assert stats['cache_read_input_tokens'] == 300


def test_public_stage_stats_strips_bookkeeping_keys():
    """cost_log.jsonl serializes stage_stats directly; its schema must not gain keys."""
    stats = agent.new_stage_stats()
    agent.accumulate_stage_stats(stats, _result_msg('s', 1.0))
    public = agent.public_stage_stats({'scraping': stats})
    assert 'session_costs' not in public['scraping']
    assert public['scraping']['cost'] == pytest.approx(1.0)
    assert set(public['scraping']) == {
        'cost', 'input_tokens', 'output_tokens',
        'cache_read_input_tokens', 'cache_creation_input_tokens',
    }
    json.dumps(public)  # must stay JSON-serializable for log_run_cost


# ---------------------------------------------------------------------------
# scraper browser-tool restriction
# ---------------------------------------------------------------------------

def test_required_browser_tools_are_never_disallowed():
    """Measured on a live LinkedIn results page: browser_evaluate is the only working scroll
    (inner container lazy-loads 7 -> 10 listings) and browser_click drives pagination. Removing
    either silently cuts discovery rather than raising, so guard the list."""
    overlap = set(config.SCRAPER_REQUIRED_BROWSER_TOOLS) & set(config.SCRAPER_DISALLOWED_BROWSER_TOOLS)
    assert not overlap, f'these are load-bearing for discovery: {sorted(overlap)}'


def test_scraper_keeps_obstacle_handling_tools():
    """Scraping is not a deterministic problem: pages change and add bot protection, so the
    tools needed to get past a dialog, consent banner, or login form stay available."""
    obstacle_tools = (
        'fill_form', 'type', 'hover', 'select_option', 'handle_dialog', 'console_messages',
        # take_screenshot is the only way to SEE a block page / CAPTCHA / consent overlay that
        # the accessibility tree renders uninformatively; find and press_key help work a
        # restructured page. A silent zero-listing run costs more than the turns these burn.
        'take_screenshot', 'find', 'press_key',
    )
    for tool in obstacle_tools:
        assert f'mcp__playwright__browser_{tool}' not in config.SCRAPER_DISALLOWED_BROWSER_TOOLS


def test_disallowed_browser_tools_are_well_formed():
    for name in config.SCRAPER_DISALLOWED_BROWSER_TOOLS:
        assert name.startswith('mcp__playwright__browser_'), name
    assert len(set(config.SCRAPER_DISALLOWED_BROWSER_TOOLS)) == len(config.SCRAPER_DISALLOWED_BROWSER_TOOLS)


# ---------------------------------------------------------------------------
# scraper instructions: filters are UI actions, not URL parameters
# ---------------------------------------------------------------------------

def test_scraper_instructions_do_not_rely_on_url_filter_params():
    """LinkedIn strips location/f_WT/f_E/sortBy from job-search URLs (2026-08-11), keeping only
    `keywords`. Emitting them as URL params produced a search that looked right and silently
    returned the wrong jobs, so the prompt must drive the filter chips instead."""
    text = agent.build_scraper_instructions()
    for dead_param in ('f_WT=2', 'f_E=4%2C5%2C6', 'sortBy=DD', '&location='):
        assert dead_param not in text, f'{dead_param} is ignored by LinkedIn; set it via the UI'
    assert 'search-results' in text, 'the /jobs/search/ path redirects'
    assert 'f_SAL=' in text, 'the empty salary param is what clears sticky filter state'


def test_scraper_instructions_cover_every_configured_region_as_a_chip_click():
    """Region is the only axis that varies the result set, and it is applied by CLICKING.

    Typing the region into `keywords` is silently ignored by LinkedIn: for several days every
    "European Union" search returned the account's home metro, and 12 searches collapsed to 101
    unique jobs instead of ~300. So each configured region must appear as a location-chip
    instruction, never as query text.
    """
    text = agent.build_scraper_instructions()
    for region in ('Testland', 'Test Union'):  # pinned in tests/conftest.py
        assert f'**{region}** — search `<QUERY>, remote`' in text
        assert f'set **Location** to `{region}`' in text, 'region is a chip, not query text'
        assert f'{region}, senior level' not in text, 'the region must NOT go into the query text'
    assert 'as 2 searches' in text, 'one search per region'


def test_scraper_instructions_never_craft_filter_urls():
    """Filters go in by clicking, never by hand-assembling the URL.

    geoId/f_TPR demonstrably DO work when navigated to directly — that is exactly why this test
    exists. No human assembles filter parameters by hand, so doing it is a cheap fingerprint for
    anti-automation, and this drives a real logged-in account.
    """
    text = agent.build_scraper_instructions()
    for forbidden in ('&geoId=', '&f_TPR=', '&f_E=', '&f_WT='):
        assert forbidden not in text, 'filter params must never appear in a URL to navigate to'
    assert 'Never put `geoId`' in text, 'the prohibition must be stated explicitly'
    assert 'clicking the chips' in text


def test_scraper_instructions_forbid_renavigating_after_filters_are_clicked():
    """2026-09-10: LinkedIn re-appended its salary param after "Show results", the model
    re-navigated to strip it, and that silently dropped the location it had just clicked — the
    sticky location chip kept showing "Canada", so the page looked filtered when it was not."""
    text = agent.build_scraper_instructions()
    assert 'Do not navigate again to remove it' in text, 'the re-appearing f_SAL must be declared expected'
    assert 'do not navigate again during this search' in text
    assert 'the location chip included' in text, 'a restart must re-click the location too'


def test_scraper_instructions_keep_the_results_list_unclickable_while_chips_are_clicked():
    """Clicking gained a legitimate use (chips); the results-list ban must survive that.

    The only real <button> in a result card is Dismiss, and one stray click permanently removes a
    job from the user's feed — this already destroyed three real jobs in under two minutes.
    """
    text = agent.build_scraper_instructions()
    assert 'NEVER click anything in the job results list' in text
    assert 'is still never clicked' in text, (
        'introducing chip clicks must restate the boundary, not blur it'
    )


def test_scraper_instructions_require_the_ui_contract_before_harvesting():
    """The page is checked structurally, and the verdict is code's, not the model's.

    The contract used to be run BY the model, which then passed the report object to report_search.
    That made the model a courier for data, and on 2026-08-21 it invented an entire page report
    after its evaluate result was diverted to a file. `run_ui_contract` takes no report: code runs
    the JS, reads it and judges it, so the check must still come first but the model never carries
    the finding.
    """
    text = agent.build_scraper_instructions()
    flat = ' '.join(text.split())   # the prompt is hard-wrapped; match on prose, not layout
    assert 'run_ui_contract' in text, 'the contract check must be instructed'
    assert 'that judgement is deliberately not yours to make' in flat
    assert text.index('run_ui_contract') < text.index('harvest_listings'), \
        'the contract must be checked before harvesting'
    assert 'report_search' not in text, \
        'the model must not be asked to hand over the report object — that is the courier bug'


def test_scraper_tools_never_accept_page_or_job_data():
    """The anti-fabrication invariant, asserted structurally.

    A model relaying a payload cannot distinguish copying from producing, so with no data it emits
    a plausible object rather than failing. Every argument it can send must be a DECISION.
    """
    allowed = {'region', 'what_happened'}
    for tool in scrape_openrouter.LOCAL_TOOL_DEFS:
        fn = tool['function']
        props = set((fn['parameters'].get('properties') or {}).keys())
        assert props <= allowed, (
            f"{fn['name']} accepts {sorted(props - allowed)}, which the model would have to copy "
            'from a tool result; code should read that itself')
        assert fn['parameters'].get('additionalProperties') is False, \
            f"{fn['name']} lets the model smuggle extra data in"
        desc = fn['description'].lower()
        tells = ('verbatim', 'exactly as returned', 'do not edit', 'do not summarise', 'unedited')
        assert not [t for t in tells if t in desc], \
            f"{fn['name']} asks for a verbatim relay — a courier argument confessing itself"

def test_scraper_instructions_reject_identical_results_as_a_failure():
    """The converse of 'expect many already_processed': two searches returning the SAME jobs
    means the region text did not apply. Treating that as 'query exhausted' is what let the
    2026-08-11 collapse stop after 10 of 90 turns."""
    text = agent.build_scraper_instructions()
    assert 'IDENTICAL' in text
    assert 'it is not a failure' in text, 'the already_processed guidance must survive'


def test_scraper_instructions_harvest_job_ids_from_componentkey():
    """Job ids come off the DOM, not off the accessibility tree and not from clicking.

    Each card is div[componentkey="job-card-component-ref-<jobId>"], so one read-only evaluate
    yields every listing on the page. This replaced a click-to-reveal design that read ids from
    `currentJobId` after selecting each card.
    """
    # The selectors now live in the harvest JS, which CODE runs -- the model neither writes it nor
    # sees the listings, so it cannot retype it wrongly (it corrupted the contract JS this way on
    # 2026-08-21) nor invent ids.
    assert 'SearchResultsMainContent' in agent.SCRAPER_HARVEST_JS, \
        'the results container selector must be in the harvest JS'
    assert 'job-card-component-ref-' in agent.SCRAPER_HARVEST_JS, \
        'the id-bearing attribute must be in the harvest JS'
    text = agent.build_scraper_instructions()
    assert 'harvest_listings' in text, 'the model must be told to harvest via the tool'
    assert 'componentkey' in text, 'the prompt must still explain where ids come from'


def test_scraper_instructions_forbid_clicking_the_results_list():
    """The hard safety invariant. The only real <button> in a result row is Dismiss, and the
    accessibility tree labels it with the whole card's text -- so clicking what looks like the
    card clicks Dismiss, permanently removing the job from the user's feed. That destroyed three
    real jobs in under two minutes before the harvest approach replaced it."""
    text = agent.build_scraper_instructions()
    assert 'NEVER click anything in the job results list' in text
    assert 'Dismiss' in text, 'the destructive control must be named'
    # The prompt must not tell the model to click its way to an id any more.
    assert 'currentJobId' not in text, 'click-to-reveal is gone; ids come from componentkey'
    assert 'never drive it' in text, 'JS may read the page but never click/submit'


# ---------------------------------------------------------------------------
# human-emulation pacing (protects a real logged-in LinkedIn account)
# ---------------------------------------------------------------------------

def test_pacing_delays_are_ranges_not_constants():
    """A fixed interval is itself a robotic signature, so every delay must be a (min, max)
    range that _human_pause draws from uniformly."""
    for name in (
        'SCRAPER_INTER_SEARCH_DELAY_SECONDS',
        'SCRAPER_INTER_QUERY_DELAY_SECONDS',
    ):
        low, high = getattr(config, name)
        assert 0 < low < high, f'{name} must be a non-degenerate (min, max) range, got {(low, high)}'


async def test_human_pause_draws_from_the_range(monkeypatch):
    slept = []
    real_sleep = asyncio.sleep  # capture before patching -- agent.asyncio IS this module

    async def fake_sleep(seconds):
        slept.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(agent.asyncio, 'sleep', fake_sleep)
    for _ in range(20):
        await agent._human_pause((2.0, 5.0), 'test')
    assert all(2.0 <= s <= 5.0 for s in slept), slept
    assert len(set(slept)) > 1, 'a constant delay is a robotic signature'


async def test_run_scraper_pauses_between_queries(monkeypatch):
    """Inter-query pacing is enforced in CODE, not prompted: a model under turn pressure will
    skip a prompted wait, and a flagged account ends the whole job search."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0.01, 0.02))

    pauses = []
    real_pause = agent._human_pause

    async def spy(delay_range, reason):
        pauses.append(reason)
        await real_pause(delay_range, reason)

    monkeypatch.setattr(agent, '_human_pause', spy)

    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    queries = ['One', 'Two', 'Three']
    client = _FakeScraperClient(dict.fromkeys(queries, healthy))
    await _scrape(client, queries)

    # A pause before every query except the first -- no point waiting before any work is done.
    assert len(pauses) == 2, pauses


def test_scraper_instructions_forbid_pushing_through_blocks():
    """Retrying into a challenge page converts 'this session looks odd' into a confirmed evasion
    pattern, which is what escalates to a restricted account. Backing off keeps it a blip."""
    text = agent.build_scraper_instructions()
    assert 'CAPTCHA' in text and 'Stop immediately' in text
    assert 'Do not apply to anything' in text
    assert str(config.SCRAPER_MAX_LISTINGS_PER_SEARCH) in text, 'the per-search cap must be stated'


def test_min_listings_threshold_keeps_the_recovery_pass_enabled():
    """At 0 the low-yield retry silently never fires, which is the bug it exists to catch."""
    assert config.SCRAPER_MIN_LISTINGS_PER_QUERY >= 1
    assert config.SCRAPER_MIN_LISTINGS_PER_QUERY < config.SCRAPER_MAX_TURNS_PER_QUERY


def test_low_listings_alert_threshold_is_half_the_retry_floor():
    """The alert is checked after recovery, so it must sit below the retry trigger and above 0."""
    assert config.SCRAPER_ALERT_MIN_LISTINGS_PER_QUERY == config.SCRAPER_MIN_LISTINGS_PER_QUERY // 2
    assert 1 <= config.SCRAPER_ALERT_MIN_LISTINGS_PER_QUERY < config.SCRAPER_MIN_LISTINGS_PER_QUERY


# ---------------------------------------------------------------------------
# queue-time title guard ($0 replacement for LinkedIn's dropped f_E filter)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('title, expect_skip', [
    ('Junior AI Engineer', True),
    ('AI Engineering Intern', True),
    ('Machine Learning Engineer (Entry-Level)', True),
    ('Engineering Manager, ML Platform', True),
    ('Head of Data Science', True),
    ('VP of Engineering', True),
    # Must NOT skip: substring collisions and senior titles that merely contain a token.
    ('Staff AI/ML Engineer', False),
    ('Principal Applied Scientist', False),
    ('Internal Tools Engineer', False),        # 'intern' is a substring, not a word
    ('VPN Infrastructure Engineer', False),    # 'VP' is a substring, not a word
    ('Associate Principal Scientist', False),  # 'associate' is senior in many orgs
    ('Graduate Research Scientist', False),    # bare 'graduate' is too ambiguous to reject
])
def test_title_rejection_reason(title, expect_skip):
    reason = tools.title_rejection_reason(title)
    assert (reason is not None) == expect_skip, f'{title!r} -> {reason!r}'


async def test_queue_candidate_skips_below_seniority_titles(monkeypatch):
    """LinkedIn no longer enforces f_E server-side, so junk reaches the queue -- and each queued
    job costs a Stage 2a extract plus a rating call. The skip is $0 and must be auditable."""
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queue_skipped_counts', {})
    monkeypatch.setattr(tools, '_listing_records', {
        ('linkedin', '111'): _listing('111', 'BigCo'),
    })

    await tools.do_queue_candidate(
        'linkedin', '111', 'https://example.com', 'Junior ML Engineer', 'BigCo', 'snip', query='Q'
    )
    await tools.do_queue_candidate(
        'linkedin', '222', 'https://example.com/2', 'Staff ML Engineer', 'Acme', 'snip', query='Q'
    )

    assert [c['title'] for c in tools._candidates] == ['Staff ML Engineer']
    assert tools._candidates_per_query['Q'] == 1, 'skipped listings are not counted as queued'
    assert sum(tools._queue_skipped_counts.values()) == 1
    assert tools._listing_records[('linkedin', '111')]['outcome'] == 'queue_skipped'


# ---------------------------------------------------------------------------
# agent narration reaches the run log
# ---------------------------------------------------------------------------

def test_log_agent_text_mirrors_each_line_to_the_log(caplog):
    """print() reaches the terminal only. On 2026-08-11 the scraper's own account of why it
    found nothing was discarded, leaving only counters to diagnose a 166 -> 7 collapse."""
    with caplog.at_level(logging.INFO, logger='agentic_job_search.agent'):
        agent.log_agent_text('Stage 1b', '  Remote filter did not stick.\n\n  Only 1 result.  \n')

    messages = [r.message for r in caplog.records]
    assert messages == ['Stage 1b: Remote filter did not stick.', 'Stage 1b: Only 1 result.']


# ---------------------------------------------------------------------------
# Workplace type, rating caps, and notification bullets
#
# Regression origin: LinkedIn job 4442818316 (DevologyX Senior AI Engineer, Contract) was
# rated 4/5 and notified on 2026-08-04. It is hybrid, 2-3 days on-site in the Netherlands,
# on a 6-12 month contract. The rater named the hybrid location as a drawback in its own
# reasoning and still rated it 4; the Telegram message showed neither the location nor the
# hybrid arrangement.
# ---------------------------------------------------------------------------

def test_derive_workplace_type_prefers_explicit_field():
    extract = _make_extract(workplace_type='hybrid', location='Canada (Remote)')
    assert agent.derive_workplace_type(extract) == 'hybrid'


def test_derive_workplace_type_normalizes_explicit_variants():
    assert agent.derive_workplace_type(_make_extract(workplace_type='On-Site')) == 'onsite'
    assert agent.derive_workplace_type(_make_extract(workplace_type=' REMOTE ')) == 'remote'


def test_derive_workplace_type_infers_hybrid_from_location():
    """The extractors do not always fill workplace_type, but the wording has always leaked
    into the free-text location string."""
    extract = _make_extract(location='Netherlands (Hybrid - 2-3 days onsite)')
    assert agent.derive_workplace_type(extract) == 'hybrid'


def test_derive_workplace_type_hybrid_beats_remote_mention():
    """LinkedIn's own snippet said 'Netherlands (Remote)' for a job needing 2-3 office days."""
    extract = _make_extract(
        location='Netherlands (Remote)',
        description='Remote-friendly team. You will be on site 3 days per week in our Amsterdam office.',
    )
    assert agent.derive_workplace_type(extract) == 'hybrid'


def test_derive_workplace_type_empty_when_unstated():
    extract = _make_extract(location='Berlin, Germany', description='Build AI systems.')
    assert agent.derive_workplace_type(extract) == ''


def test_hybrid_location_acceptable():
    # Locations come from the pinned test preferences in conftest, not from anyone's real config.
    assert agent.commute_location_is_acceptable('Exampleton, Testland (Hybrid)')
    assert agent.commute_location_is_acceptable('Testville, Testland')
    assert not agent.commute_location_is_acceptable('Netherlands (Hybrid - 2-3 days onsite)')
    assert not agent.commute_location_is_acceptable('')


def test_hybrid_location_unacceptable_when_no_locations_configured(monkeypatch):
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ())
    assert not agent.commute_location_is_acceptable('Exampleton, Testland (Hybrid)')


def test_apply_rating_caps_caps_hybrid_in_unacceptable_location():
    extract = _make_extract(location='Netherlands (Hybrid - 2-3 days onsite)')
    rating, reason = agent.apply_rating_caps(extract, 4)
    assert rating == preferences.hybrid_rating_cap()
    assert 'not an acceptable hybrid location' in reason


def test_apply_rating_caps_allows_hybrid_in_acceptable_location():
    extract = _make_extract(location='Exampleton, Testland (Hybrid)')
    assert agent.apply_rating_caps(extract, 4) == (4, '')


def test_apply_rating_caps_allows_hybrid_in_other_acceptable_location():
    extract = _make_extract(location='Testville, Testland (Hybrid)')
    assert agent.apply_rating_caps(extract, 5) == (5, '')


def test_apply_rating_caps_leaves_remote_alone():
    assert agent.apply_rating_caps(_make_extract(), 5) == (5, '')


def test_apply_rating_caps_does_not_raise_a_low_rating():
    """A cap is a ceiling, never a floor — a 2 stays a 2."""
    extract = _make_extract(location='Netherlands (Hybrid - 2-3 days onsite)')
    assert agent.apply_rating_caps(extract, 2) == (2, '')


def test_build_deterministic_warnings_flags_hybrid_contract_and_missing_salary():
    extract = _make_extract(
        title='Senior AI Engineer (Contract)',
        location='Netherlands (Hybrid - 2-3 days onsite)',
        description='Contract Length: 6-12 Months. Build RAG pipelines in Python.',
        salary='',
    )
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert any('Hybrid' in w and 'not somewhere you would commute' in w for w in warnings)
    assert any('Contract role' in w for w in warnings)
    assert 'No salary listed' in warnings


def test_build_deterministic_warnings_quiet_for_clean_remote_job():
    extract = _make_extract(salary='CAD 200,000 - 240,000 per year')
    assert agent.build_deterministic_warnings(_make_candidate(), extract) == []


def test_build_deterministic_warnings_includes_relocation():
    extract = _make_extract(relocation='Portugal', salary='EUR 100,000')
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert 'Relocation required: Portugal' in warnings


@pytest.mark.parametrize('salary, listed', [
    ('', False),
    ('Not specified', False),
    ('Competitive salary', False),
    ('Not stated ("competitive compensation", performance bonus, stock options)', False),
    ('CAD 220,000', True),
    ('zł392,000 – zł588,000 (annualized base salary)', True),
    ('60-75K', True),
])
def test_salary_figure_listed_requires_an_actual_figure(salary, listed):
    assert agent.salary_figure_listed(_make_extract(salary=salary)) is listed


def test_build_deterministic_warnings_flags_salary_text_with_no_figure():
    """The extractor writes 'Competitive salary' rather than leaving the field empty; a truthiness
    check read 196 saved postings like this as salaried and warned about none of them."""
    extract = _make_extract(salary='Competitive salary')
    assert 'No salary listed' in agent.build_deterministic_warnings(_make_candidate(), extract)


# ---------------------------------------------------------------------------
# Salary classification (salary.py)
#
# Every string below is taken verbatim from the saved-posting corpus. A digit test called all of
# them "salary listed" alike, which is how a one-ended range reached a notification (2026-09-22).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('text, kind, minimum, maximum', [
    # Complete ranges, in the shapes postings actually write them.
    ('CA$208,580–$273,770 annually (typical hiring range; plus bonus and equity eligibility)',
     'range', 208580, 273770),
    ('$190K/yr - $300K/yr', 'range', 190000, 300000),
    ('90K EUR/yr - 110K EUR/yr', 'range', 90000, 110000),
    ('zł392,000 – zł588,000 (annualized base salary)', 'range', 392000, 588000),
    ('60-75K', 'range', 60000, 75000),
    ('€700-€900/day (DOE)', 'range', 700, 900),
    ('513,000 kr—684,000 kr SEK (Remote Sweden)', 'range', 513000, 684000),
    # One-ended: the whole point. Each states a real figure and no second bound.
    ('Up to €50k (dependent on experience)', 'ceiling_only', None, 50000),
    ('Under $40/hr', 'ceiling_only', None, 40),
    ('Daily rate up to 600€', 'ceiling_only', None, 600),
    ('Up to CA$70/hr', 'ceiling_only', None, 70),
    # The ceiling is the amount the marker introduces, not the largest number in the string.
    ('Up to €135,000 Base + Bonus (c.€150,000 OTE)', 'ceiling_only', None, 135000),
    ('Minimum €65,000/year (overpayment possible depending on qualifications)', 'floor_only', 65000, None),
    ('Competitive salary starting from €70K gross annually, depending on skills and experience',
     'floor_only', 70000, None),
    ('$30+ USD per hour, with bonus rates available on some projects', 'floor_only', 30, None),
    ('Min. 32,640 PLN gross per month + performance-based variable pay', 'floor_only', 32640, None),
    # A lone figure is not a range either.
    ('€110,000', 'single', 110000, 110000),
    ('$100/hr', 'single', 100, 100),
    ('$46.50/hour', 'single', 46.5, 46.5),
    ('CAD 220,000', 'single', 220000, 220000),
    # No figure stated. These are the 2026-09-15 lesson and must keep reading as absent.
    ('', 'absent', None, None),
    ('Not specified', 'absent', None, None),
    ('Competitive salary', 'absent', None, None),
    ('Not stated ("competitive compensation", performance bonus, stock options)', 'absent', None, None),
    # Not pay at all, though every one of them carries a digit.
    ('12-month contract / Outside IR35', 'unclassified', None, None),
    ('90th percentile', 'unclassified', None, None),
    ('€1,200/year training budget; flexible compensation package; private medical insurance',
     'unclassified', None, None),
])
def test_classify_salary_reads_the_structure_of_a_pay_string(text, kind, minimum, maximum):
    facts = salary.classify_salary(text)
    assert facts['kind'] == kind
    assert facts['minimum'] == minimum
    assert facts['maximum'] == maximum
    assert facts['text'] == text.strip(), 'the string is carried through as written'
    assert facts['source'] == 'deterministic', 'no rule-answerable string may cost a model call'


@pytest.mark.parametrize('text, currency, period', [
    ('CA$208,580–$273,770 annually', 'CAD', 'year'),
    ('90K EUR/yr - 110K EUR/yr', 'EUR', 'year'),
    ('zł392,000 – zł588,000', 'PLN', ''),
    ('Up to CA$70/hr', 'CAD', 'hour'),
    ('€650/day', 'EUR', 'day'),
    ('EUR 5,014.30 gross/month', 'EUR', 'month'),
    # A bare '$' is genuinely ambiguous — USD, CAD, AUD, SGD. Guessing is not this module's job.
    ('$190K/yr - $300K/yr', '', 'year'),
])
def test_classify_salary_reads_currency_and_period(text, currency, period):
    facts = salary.classify_salary(text)
    assert facts['currency'] == currency
    assert facts['period'] == period


async def test_resolve_salary_does_not_call_the_model_for_a_readable_string(monkeypatch):
    """The rules answer 862 of 911 figured postings in the corpus; none may cost a call."""
    async def _boom(*args, **kwargs):
        raise AssertionError('a readable salary string was escalated to the model')

    monkeypatch.setattr(salary, 'chat_openrouter', _boom)
    facts = await salary.resolve_salary({'salary': 'CA$208,580–$273,770 annually', 'description': ''})
    assert facts['kind'] == 'range'


async def test_resolve_salary_escalates_only_what_the_rules_cannot_read(monkeypatch):
    calls = []

    async def _fake(prompt, model=None):
        calls.append(prompt)
        return ('{"kind": "single", "minimum": 5014.3, "maximum": null, '
                '"currency": "EUR", "period": "month"}', 0.000123)

    monkeypatch.setattr(salary, 'chat_openrouter', _fake)
    extract = {'salary': 'EUR 5,014.30 gross/month (full-time 40h/week, 14 times per year)',
               'description': ''}
    stats = agent.new_stage_stats()

    facts = await salary.resolve_salary(extract, stats)
    assert facts['kind'] == 'single'
    assert facts['source'] == 'llm'
    assert stats['cost'] == pytest.approx(0.000123)
    assert len(calls) == 1

    # The cache is what makes this deterministic: the same string cannot be answered twice, and
    # differing whitespace or case is not compensation.
    again = await salary.resolve_salary(dict(extract, salary=extract['salary'].upper()), stats)
    assert again['source'] == 'cache'
    assert again['kind'] == 'single'
    assert len(calls) == 1, 'a string already parsed must not be parsed again'


async def test_resolve_salary_fails_toward_saying_less_when_the_server_is_down(monkeypatch, caplog):
    """A provider outage must never turn an unreadable string into a complete range."""
    async def _down(*args, **kwargs):
        raise RuntimeError('tools_llm_remote_openrouter is not running')

    monkeypatch.setattr(salary, 'chat_openrouter', _down)
    facts = await salary.resolve_salary({'salary': 'six figures, depending', 'description': ''})
    assert facts['kind'] == 'unclassified'
    assert facts['source'] == 'error'
    assert any('Salary classification failed' in r.message and r.levelname == 'WARNING'
               for r in caplog.records), 'a silent classifier is a silent gate'


async def test_resolve_salary_never_upgrades_a_partial_to_a_range(monkeypatch):
    """A model that labels something `range` without two numbers is downgraded, not trusted."""
    async def _fake(prompt, model=None):
        return ('{"kind": "range", "minimum": 70000, "maximum": null, "currency": "EUR"}', 0.0)

    monkeypatch.setattr(salary, 'chat_openrouter', _fake)
    facts = await salary.resolve_salary({'salary': 'attractive package, six figures', 'description': ''})
    assert facts['kind'] == 'unclassified'


async def test_resolve_salary_shows_the_model_the_compensation_lines_when_the_field_has_no_figure(monkeypatch):
    """The excerpt is selected in code, so the model is handed evidence rather than sent looking."""
    prompts = []

    async def _fake(prompt, model=None):
        prompts.append(prompt)
        return ('{"kind": "range", "minimum": 180000, "maximum": 220000, "currency": "CAD"}', 0.0)

    monkeypatch.setattr(salary, 'chat_openrouter', _fake)
    extract = {
        'salary': 'salary information withheld',
        'description': 'We build things. The base salary range is CAD 180,000 - 220,000. '
                       'We use Python and Kubernetes.',
    }
    facts = await salary.resolve_salary(extract)
    assert facts['kind'] == 'range'
    assert 'CAD 180,000 - 220,000' in prompts[0]
    assert 'Python and Kubernetes' not in prompts[0], 'only the money sentences are sent'


@pytest.mark.parametrize('salary_text, expected', [
    ('Competitive salary', 'No salary listed'),
    ('Up to CA$70/hr', 'Partial salary — only an upper bound: Up to CA$70/hr'),
    ('Minimum €65,000', 'Partial salary — only a lower bound: Minimum €65,000'),
    ('€110,000', 'Partial salary — a single figure, no range: €110,000'),
    ('12-month contract / Outside IR35', 'Salary not interpretable: 12-month contract / Outside IR35'),
])
def test_build_deterministic_warnings_names_what_is_wrong_with_the_salary(salary_text, expected):
    extract = _make_extract(salary=salary_text)
    assert expected in agent.build_deterministic_warnings(_make_candidate(), extract)


def test_build_deterministic_warnings_says_nothing_about_a_complete_range():
    extract = _make_extract(salary='CA$208,580–$273,770 annually')
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert not any('salary' in w.lower() for w in warnings)


def test_merge_salary_warning_puts_the_code_line_before_the_raters_floor_first_bullet():
    """The incident, in one assertion.

    The rater's own bullet led with the floor and nothing contradicted it, because the fold was a
    no-op whenever a figure was listed (job 4470298356, 2026-09-22).
    """
    deterministic = ['Partial salary — only a lower bound: From CA$208,580']
    llm = ['Base range floor ($208.6K) sits just below the $210K target; upper half well above']
    merged = agent.merge_warnings(deterministic, llm)
    assert len(merged) == 1, 'one salary line, not two saying the same thing differently'
    assert merged[0].startswith('Partial salary — only a lower bound:'), \
        "the structure is stated in code first, the rater's judgement second"
    assert '$210K target' in merged[0], "the rater's estimate is kept, not discarded"


def test_merge_salary_warning_leaves_a_complete_range_to_the_rater():
    """Below-target on a full range is the judgement the rater is asked for (requirements.md)."""
    llm = ['Pay is below the CAD $210K target across the whole band']
    assert agent.merge_warnings([], llm) == llm


TEST_BASE_TARGET = 200_000
TEST_TARGET_CURRENCY = 'CAD'
# Shaped like job 4470446597: a range whose top clears the target, which the rater warned about (2026-09-23).
RXNT_SALARY = 'CA$170,000 - CA$240,000 per year'
RXNT_RATER_BULLET = 'Base target sits in upper half of range; lower bound well below target'


@pytest.fixture
def salary_target_set(monkeypatch):
    merged = preferences._deep_merge(
        preferences.load_preferences(),
        {'compensation': {'base_target': TEST_BASE_TARGET, 'currency': TEST_TARGET_CURRENCY}},
    )
    monkeypatch.setattr(preferences, '_cache', merged)
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)


@pytest.mark.parametrize('salary_text, expected', [
    (RXNT_SALARY, True),
    ('CA$120,000 - CA$180,000 per year', False),
    ('US$170,000 - US$240,000 per year', None),   # another currency needs a conversion
    ('$170,000 - $240,000 per year', None),       # a bare '$' is ambiguous
    ('CA$90 - CA$110 per hour', None),            # an hourly rate needs annualising
    ('From CA$208,580 per year', None),           # one end is the partial-salary line's business
    ('CAD 154,700 to CAD 204,700 annually (Ontario); CAD 154,700 to CAD 310,700 (British Columbia)', None),
    ('144,600–322,500 CAD per year (combined base + variable incentive target)', None),
    ('CA$150,000 – CA$175,000/yr (base plus bonus)', None),
])
def test_salary_meets_target_decides_only_same_currency_annual_ranges(
    salary_target_set, salary_text, expected
):
    assert agent.salary_meets_target(_make_extract(salary=salary_text)) is expected


def test_salary_meets_target_is_undecided_without_a_target():
    assert agent.salary_meets_target(_make_extract(salary=RXNT_SALARY)) is None


@pytest.mark.parametrize('compensation', [
    {'base_target': 'lots', 'currency': 'CAD'},
    {'base_target': TEST_BASE_TARGET, 'currency': 'Canadian dollars'},
])
def test_salary_target_rejects_a_malformed_preference(monkeypatch, compensation):
    merged = preferences._deep_merge(preferences.load_preferences(), {'compensation': compensation})
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)
    with pytest.raises(ValueError, match='compensation'):
        preferences.salary_target()


def test_build_deterministic_warnings_flags_only_a_whole_range_below_target(salary_target_set):
    below = agent.build_deterministic_warnings(
        _make_candidate(), _make_extract(salary='CA$120,000 - CA$180,000 per year'))
    assert [w for w in below if w.startswith(agent.BELOW_TARGET_SALARY_WARNING_PREFIX)] == [
        'Salary below target — whole range under CAD 200,000: CA$120,000 - CA$180,000 per year']
    on_target = agent.build_deterministic_warnings(_make_candidate(), _make_extract(salary=RXNT_SALARY))
    assert not any('salary' in w.lower() for w in on_target)


def test_merge_warnings_drops_the_raters_salary_bullets_when_the_range_is_on_target():
    """The incident: the top of the range clears the target, so a lower-half complaint is noise."""
    llm = [
        RXNT_RATER_BULLET,
        'Base midpoint sits below target unless negotiated into upper half',
        'Stack is mostly Java',
    ]
    assert agent.merge_warnings([], llm, salary_on_target=True) == ['Stack is mostly Java']


def test_merge_warnings_folds_the_raters_estimate_after_the_below_target_line():
    deterministic = ['Salary below target — whole range under CAD 200,000: CA$120,000 - CA$180,000 per year']
    llm = ['Pay likely below target even at the top of the band', 'Stack is mostly Java']
    merged = agent.merge_warnings(deterministic, llm)
    assert merged[0].startswith('Salary below target — whole range under CAD 200,000')
    assert merged[0].endswith('Pay likely below target even at the top of the band')
    assert merged[1:] == ['Stack is mostly Java']


def test_format_extract_text_tells_the_rater_the_range_is_on_target(salary_target_set):
    text = agent.format_extract_text(_make_candidate(), _make_extract(salary=RXNT_SALARY))
    assert 'Salary vs target: meets target' in text


def test_format_extract_text_adds_no_target_line_it_cannot_decide(salary_target_set):
    text = agent.format_extract_text(_make_candidate(), _make_extract(salary='$170,000 - $240,000 per year'))
    assert 'Salary vs target:' not in text


def test_format_job_notification_carries_the_extracted_salary():
    """The message used to contain no salary field at all — only the rater's paraphrase of one."""
    extract = _make_extract(salary='CA$208,580–$273,770 annually (typical hiring range)')
    message = agent.format_job_notification(
        _make_candidate(), extract, 4, ['Strong comp: CA$208K–$274K base'], [])
    assert '💰 CA$208,580–$273,770 annually (typical hiring range)' in message


@pytest.mark.parametrize('salary_text, gloss', [
    ('Up to CA$70/hr', '(upper bound only)'),
    ('Minimum €65,000', '(lower bound only)'),
    ('€110,000', '(single figure, no range)'),
])
def test_format_job_notification_says_why_a_salary_line_is_partial(salary_text, gloss):
    message = agent.format_job_notification(
        _make_candidate(), _make_extract(salary=salary_text), 4, [], [])
    assert f'💰 {salary_text} {gloss}' in message


def test_format_job_notification_omits_the_salary_line_when_there_is_no_figure():
    message = agent.format_job_notification(
        _make_candidate(), _make_extract(salary='Competitive salary'), 4, [], ['No salary listed'])
    assert '💰' not in message, 'the warning already says there is no figure'


def test_format_job_notification_shows_the_posting_date():
    extract = _make_extract(date_posted=f'{date.today().year}-09-28')
    message = agent.format_job_notification(_make_candidate(), extract, 4, [], [])
    assert '🗓 Posted Sep 28th' in message


@pytest.mark.parametrize('day, expected', [
    (1, '1st'), (2, '2nd'), (3, '3rd'), (4, '4th'), (11, '11th'), (12, '12th'), (13, '13th'),
    (21, '21st'), (22, '22nd'), (23, '23rd'), (31, '31st'),
])
def test_format_posted_line_ordinal_suffix(day, expected):
    extract = _make_extract(date_posted=f'{date.today().year}-01-{day:02d}')
    assert agent.format_posted_line(_make_candidate(), extract) == f'🗓 Posted Jan {expected}'


def test_format_posted_line_shows_the_year_only_when_not_this_one():
    extract = _make_extract(date_posted=f'{date.today().year - 1}-12-30')
    assert agent.format_posted_line(_make_candidate(), extract) == f'🗓 Posted Dec 30th {date.today().year - 1}'


def test_format_posted_line_keeps_reposted():
    """LinkedIn writes reposts as 'Reposted N days ago'; a repost must not read as a new posting."""
    line = agent.format_posted_line(_make_candidate(), _make_extract(date_posted='Reposted 2 days ago'))
    assert line.startswith('🗓 Reposted ')


def test_format_posted_line_resolves_a_relative_date():
    today = date.today()
    line = agent.format_posted_line(_make_candidate(), _make_extract(date_posted='just now'))
    assert line == f"🗓 Posted {today.strftime('%b')} {today.day}{agent._ordinal_suffix(today.day)}"


def test_format_posted_line_falls_back_to_the_candidate_date():
    candidate = _make_candidate(date_posted=f'{date.today().year}-03-02')
    assert agent.format_posted_line(candidate, _make_extract(date_posted='')) == '🗓 Posted Mar 2nd'


def test_format_posted_line_shows_an_unparseable_date_as_extracted():
    line = agent.format_posted_line(_make_candidate(), _make_extract(date_posted='Recently'))
    assert line == '🗓 Posted: Recently'


def test_format_posted_line_reports_a_missing_date():
    line = agent.format_posted_line(_make_candidate(date_posted=''), _make_extract(date_posted=''))
    assert line == '🗓 Posting date not shown'


def test_format_extract_text_tells_the_rater_the_salary_structure():
    text = agent.format_extract_text(_make_candidate(), _make_extract(salary='Up to €50k'))
    assert 'Salary: Up to €50k' in text, 'the existing label keeps its format'
    assert 'Salary structure: upper bound stated, no lower bound' in text


def test_format_extract_text_adds_no_structure_line_for_a_complete_range():
    text = agent.format_extract_text(
        _make_candidate(), _make_extract(salary='CA$208,580–$273,770 annually'))
    assert 'Salary structure:' not in text, 'a note on every posting is one the rater stops reading'


# ---------------------------------------------------------------------------
# Raw posting retention (run_dir/raw_postings/)
# ---------------------------------------------------------------------------

def test_save_raw_posting_writes_the_page_with_its_provenance_header(tmp_path):
    candidate = dict(_make_candidate(), job_id='4470298356', site='linkedin')
    path = tools.save_raw_posting(candidate, 'CA$208,580 to CA$273,770 per year', 'openrouter_loop', 900)
    assert path is not None and path.is_file()
    content = path.read_text(encoding='utf-8')
    assert 'CA$208,580 to CA$273,770 per year' in content
    assert 'job_id:' in content and 'chars_sent_to_model: 900' in content
    assert path.name == 'linkedin-4470298356.txt'


def test_save_raw_posting_survives_an_unwritable_directory(monkeypatch, caplog):
    """A retention failure is not a run failure: this directory feeds nothing."""
    def _boom(*args, **kwargs):
        raise OSError('read-only file system')

    monkeypatch.setattr(Path, 'mkdir', _boom)
    assert tools.save_raw_posting(_make_candidate(), 'text', 'openrouter_loop', 4) is None
    assert any('Could not retain the raw posting' in r.message for r in caplog.records)


def test_prune_raw_postings_removes_only_directories_past_the_window():
    from agentic_job_search.config import RAW_POSTINGS_RETENTION_DAYS
    today = date(2026, 9, 22)
    fresh = today - timedelta(days=RAW_POSTINGS_RETENTION_DAYS - 1)
    stale = today - timedelta(days=RAW_POSTINGS_RETENTION_DAYS + 1)
    for day in (fresh, stale):
        directory = tools.RAW_POSTINGS_DIR / day.isoformat()
        directory.mkdir(parents=True)
        (directory / 'linkedin-1.txt').write_text('x', encoding='utf-8')

    assert tools.prune_raw_postings(today=today) == 1
    assert (tools.RAW_POSTINGS_DIR / fresh.isoformat()).is_dir()
    assert not (tools.RAW_POSTINGS_DIR / stale.isoformat()).exists()


def test_prune_raw_postings_leaves_anything_that_is_not_a_capture_directory(caplog):
    """The only code in this repo that deletes user data deletes exactly one shape of thing."""
    keep = tools.RAW_POSTINGS_DIR / 'notes-i-put-here-by-hand'
    keep.mkdir(parents=True)
    (keep / 'important.txt').write_text('mine', encoding='utf-8')

    assert tools.prune_raw_postings(today=date(2099, 1, 1)) == 0
    assert (keep / 'important.txt').is_file()
    assert any('Not a capture directory' in r.message for r in caplog.records)


@pytest.mark.parametrize('salary_text, raw_text, verified', [
    ('CA$208,580–$273,770 annually', 'base pay range CA$208,580.00/yr - CA$273,770.00/yr', True),
    # Separators differ between the page and the extract; the number is the same number.
    ('CA$208580 - CA$273770', 'CA$208,580 - CA$273,770', True),
    ('CA$208,600–$273,800 annually', 'base pay range CA$208,580 - CA$273,770', False),
    ('Competitive salary', 'no figures anywhere on this page', True),
])
def test_salary_digits_are_in_the_source(salary_text, raw_text, verified):
    assert salary.salary_digits_are_in_the_source(salary_text, raw_text) is verified


def test_check_salary_provenance_flags_a_figure_the_page_does_not_contain(caplog):
    candidate = dict(_make_candidate(), job_id='4470298356', site='linkedin')
    tools.save_raw_posting(candidate, 'base pay range CA$208,580 - CA$273,770', 'openrouter_loop', 40)
    extract = _make_extract(salary='CA$208,600 - CA$273,800 annually')
    extract['salary_facts'] = salary.classify_salary(extract['salary'])

    assert agent.check_salary_provenance(candidate, extract) is False
    assert extract['salary_facts']['source'] == 'unverified'
    assert any('Salary provenance' in r.message and r.levelname == 'WARNING'
               for r in caplog.records)


def test_check_salary_provenance_passes_when_the_page_backs_the_figure():
    candidate = dict(_make_candidate(), job_id='4470298356', site='linkedin')
    tools.save_raw_posting(candidate, 'base pay range CA$208,580 - CA$273,770', 'openrouter_loop', 40)
    extract = _make_extract(salary='CA$208,580–$273,770 annually')
    extract['salary_facts'] = salary.classify_salary(extract['salary'])
    assert agent.check_salary_provenance(candidate, extract) is True


@pytest.mark.live
async def test_resolve_salary_live(monkeypatch, tmp_path):
    """The real classifier against the real server, on a string the rules deliberately refuse."""
    _require_llm_server(8006)
    monkeypatch.setattr(salary, 'SALARY_CACHE_PATH', tmp_path / 'salary_cache.yaml')
    monkeypatch.setattr(salary, '_cache', {})
    monkeypatch.setattr(salary, 'chat_openrouter', triage.chat_openrouter)
    facts = await salary.resolve_salary(
        {'salary': 'somewhere in the region of eighty thousand euros a year', 'description': ''})
    assert facts['kind'] in salary.SALARY_KINDS
    assert facts['source'] in ('llm', 'error')


def test_the_three_extract_schemas_share_one_salary_description():
    """Two of them carried a 'must stay in sync' comment and no test to enforce it."""
    from agentic_job_search import extract_openrouter as extract_or
    from agentic_job_search import tools_generic
    from agentic_job_search.config import SALARY_FIELD_DESCRIPTION

    openrouter_schema = next(
        t['function']['parameters'] for t in extract_or.OPENROUTER_EXTRACT_TOOLS
        if t['function']['name'] == 'submit_job_extract'
    )
    sdk_schema = tools_generic.submit_job_extract.input_schema['properties']
    for properties in (agent.EXTRACT_OUTPUT_SCHEMA['properties'],
                       openrouter_schema['properties'], sdk_schema):
        assert properties['salary'].get('description') == SALARY_FIELD_DESCRIPTION
    assert 'BOTH ends' in SALARY_FIELD_DESCRIPTION


def test_the_extract_schemas_all_accept_stated_working_language():
    """A field added to one extractor and not the others silently returns empty for that provider."""
    from agentic_job_search import extract_openrouter as extract_or
    from agentic_job_search import tools_generic
    from agentic_job_search.config import STATED_WORKING_LANGUAGE_FIELD_DESCRIPTION

    openrouter_schema = next(
        t['function']['parameters'] for t in extract_or.OPENROUTER_EXTRACT_TOOLS
        if t['function']['name'] == 'submit_job_extract'
    )
    sdk_schema = tools_generic.submit_job_extract.input_schema['properties']
    for properties in (agent.EXTRACT_OUTPUT_SCHEMA['properties'],
                       openrouter_schema['properties'], sdk_schema):
        assert properties['stated_working_language']['description'] == STATED_WORKING_LANGUAGE_FIELD_DESCRIPTION
    parsed = extract_or._extract_from_submit_args(
        {'title': 't', 'company': 'c', 'description': 'd', 'stated_working_language': 'English'})
    assert parsed['stated_working_language'] == 'english'


@pytest.mark.parametrize('description', [
    'Temporary position up to 12 months, based in Toronto.',
    'Work Type: Temporary Full Time.',
    # Employment phrasing the original bare-stem exclusion ('work', 'or permanent') swallowed.
    'This is a temporary work assignment with Acme.',
    'We are hiring on a temporary or permanent basis.',
    'Temporary workforce expansion for the holiday season.',
])
def test_build_deterministic_warnings_flags_temporary_roles_as_contract(description):
    extract = _make_extract(description=description, salary='CAD 200,000 - 240,000 per year')
    assert 'Contract role — full-time preferred' in agent.build_deterministic_warnings(_make_candidate(), extract)


@pytest.mark.parametrize('description', [
    'If you are seeking employment on a temporary work or study permit, review the restrictions.',
    'You must have citizenship, a valid work visa, temporary or permanent residency.',
    'Temporary visa sponsorship is not available for this role.',
    'Applicants must hold a temporary resident permit.',
    'We are unable to hire temporary foreign workers.',
])
def test_build_deterministic_warnings_ignores_temporary_visa_boilerplate(description):
    extract = _make_extract(description=description, salary='CAD 200,000 - 240,000 per year')
    assert agent.build_deterministic_warnings(_make_candidate(), extract) == []


# ---------------------------------------------------------------------------
# posting language and local working language
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,expected', [
    ('french', 'french'),
    ('FRENCH', 'french'),
    ('  german  ', 'german'),
    ('english', ''),
    ('english (uk)', ''),
    ('', ''),
])
def test_foreign_posting_language(value, expected):
    assert agent.foreign_posting_language(_make_extract(posting_language=value)) == expected


@pytest.mark.parametrize('value,expected', [
    ('french', 'french'),
    ('spanish', 'spanish'),
    ('english', ''),
    ('', ''),
])
def test_foreign_likely_working_language(value, expected):
    assert agent.foreign_likely_working_language(
        _make_extract(likely_working_language=value)) == expected


def test_apply_rating_caps_caps_foreign_language_posting():
    extract = _make_extract(posting_language='french')
    rating, reason = agent.apply_rating_caps(extract, 4)
    assert rating == preferences.foreign_language_rating_cap()
    assert 'french' in reason and 'written in' in reason


def test_apply_rating_caps_leaves_english_posting_alone():
    assert agent.apply_rating_caps(_make_extract(posting_language='english'), 5) == (5, '')


def test_apply_rating_caps_foreign_language_does_not_raise_a_low_rating():
    assert agent.apply_rating_caps(_make_extract(posting_language='french'), 2) == (2, '')


def test_apply_rating_caps_likely_working_language_alone_never_caps():
    """A non-English workplace is a warning, not a ceiling — only the JD's own language caps."""
    assert agent.apply_rating_caps(_make_extract(likely_working_language='french'), 5) == (5, '')


def test_apply_rating_caps_applies_lowest_cap_and_reports_every_reason(monkeypatch):
    monkeypatch.setattr(preferences, 'foreign_language_rating_cap', lambda: 2)
    extract = _make_extract(
        location='Netherlands (Hybrid - 2-3 days onsite)', posting_language='dutch',
    )
    rating, reason = agent.apply_rating_caps(extract, 5)
    assert rating == 2
    assert 'not an acceptable hybrid location' in reason
    assert 'written in dutch' in reason


def test_build_deterministic_warnings_flags_foreign_posting_language():
    extract = _make_extract(posting_language='french', salary='CAD 200,000 - 240,000 per year')
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert any('Posting written in French' in w for w in warnings)


def test_build_deterministic_warnings_flags_non_english_likely_working_language():
    """Reads the RESOLVED field; `derive_likely_working_language` is what puts it there."""
    extract = _make_extract(
        likely_working_language='french', location='Montreal, Canada', salary='CAD 200,000 - 240,000 per year',
    )
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert any('Likely working language: French' in w and 'Montreal, Canada' in w for w in warnings)


@pytest.mark.parametrize('overrides', [
    {'posting_language': 'english', 'likely_working_language': 'english'},
    {'posting_language': '', 'likely_working_language': ''},
])
def test_build_deterministic_warnings_quiet_for_english_language_fields(overrides):
    extract = _make_extract(salary='CAD 200,000 - 240,000 per year', **overrides)
    assert agent.build_deterministic_warnings(_make_candidate(), extract) == []


def test_format_extract_text_includes_language_fields():
    text = agent.format_extract_text(
        _make_candidate(), _make_extract(posting_language='french', likely_working_language='french'),
    )
    assert 'Posting written in: french' in text
    assert 'Likely working language: french' in text


def test_format_extract_text_omits_empty_language_fields():
    text = agent.format_extract_text(_make_candidate(), _make_extract())
    assert 'Posting written in:' not in text
    assert 'Likely working language:' not in text


def test_valtech_regression_french_posting_is_capped_and_warned():
    """The 2026-08-11 Valtech posting: a French JD ('Scientifique principal des donnees en IA')
    the extractor handed back as English prose, with location 'Canada (Remote)' and
    language_requirement 'english'. It was rated 4/5 and notified. It must now cap below the
    notification threshold and say why."""
    extract = _make_extract(
        title='Scientifique principal des donnees en IA', company='Valtech',
        location='Canada (Remote)', workplace_type='remote',
        language_requirement='english', posting_language='french',
        description='Senior individual contributor role in data science and applied AI.',
    )
    rating, reason = agent.apply_rating_caps(extract, 4)
    assert rating < 4
    assert 'written in french' in reason

    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert any('Posting written in French' in w for w in warnings)

    # The SECOND bullet this posting used to get is deliberately gone. Its French working language
    # was one sentence in the body ('colleagues outside Quebec'); `Canada (Remote)` implies English,
    # and the implied local language is now a fact about the place alone. The outcome is unchanged
    # — `posting_language` is a fact about the PAGE, still a model judgement, and still caps here.
    assert not any('Likely working language' in w for w in warnings)


def _language_bullets(extract: dict) -> list[str]:
    return [w for w in agent.build_deterministic_warnings(_make_candidate(), extract) if 'language' in w.lower()]


def test_luxembourg_work_from_anywhere_regression_is_marginal():
    """The 2026-09-29 notification: 'Implied local language: Luxembourgish — Luxembourg (Remote —
    "Remote-First, work from anywhere")'. A work-from-anywhere role must not warn as if the holder
    sat in a Luxembourg office; if the location's language is foreign at all, it is marginal."""
    extract = _make_extract(
        location='Luxembourg (Remote — "Remote-First, work from anywhere")', workplace_type='remote',
        description=_NEUTRAL_DESCRIPTION, likely_working_language='french',
        salary='EUR 120,000 - 150,000 per year',
    )
    extract['residency_scope'] = agent.derive_residency_scope(extract)
    assert extract['residency_scope'] == 'area_wide'
    bullets = _language_bullets(extract)
    assert len(bullets) == 1
    assert bullets[0].startswith('Likely working language (marginal — remote across a multilingual area): French')


def test_remote_in_a_broad_area_is_marginal():
    extract = _make_extract(
        location='European Union (Remote)', workplace_type='remote', description=_NEUTRAL_DESCRIPTION,
        likely_working_language='german', location_broad_area=True,
    )
    assert 'marginal' in _language_bullets(extract)[0]


def test_hybrid_in_an_area_wide_posting_is_not_marginal():
    """Marginal needs the holder to be remote: a hybrid role sits in the office's language."""
    extract = _make_extract(
        location='Madrid, Spain (Hybrid)', workplace_type='hybrid', description=_NEUTRAL_DESCRIPTION,
        likely_working_language='spanish', residency_scope='area_wide',
    )
    assert _language_bullets(extract) == ['Likely working language: Spanish — Madrid, Spain (Hybrid)']


def test_a_stated_english_working_language_silences_the_location_guess():
    extract = _make_extract(
        location='Berlin, Germany (Hybrid)', workplace_type='hybrid', description=_NEUTRAL_DESCRIPTION,
        likely_working_language='german', stated_working_language='english',
    )
    assert _language_bullets(extract) == []
    text = agent.format_extract_text(_make_candidate(), extract)
    assert 'Stated working language: english' in text
    assert 'Likely working language:' not in text


def test_a_stated_foreign_working_language_gets_its_own_bullet():
    extract = _make_extract(
        location='Toronto, Canada (Hybrid)', workplace_type='hybrid', description=_NEUTRAL_DESCRIPTION,
        likely_working_language='english', stated_working_language='german',
    )
    assert _language_bullets(extract) == ['Working language: German (stated in posting)']


def test_a_language_requirement_silences_the_location_guess():
    """A required language the user lacks is already a hard rejection, so a surviving requirement
    names only languages the user speaks -- and it is the posting speaking, which wins."""
    extract = _make_extract(
        location='Paris, France (Hybrid)', workplace_type='hybrid', description=_NEUTRAL_DESCRIPTION,
        likely_working_language='french', language_requirement='english',
    )
    assert _language_bullets(extract) == []


def test_format_extract_text_marks_a_marginal_working_language():
    extract = _make_extract(
        location='European Union (Remote)', workplace_type='remote', description=_NEUTRAL_DESCRIPTION,
        likely_working_language='german', location_broad_area=True,
    )
    text = agent.format_extract_text(_make_candidate(), extract)
    assert 'Likely working language: german (marginal: remote across a multilingual area)' in text


# ---------------------------------------------------------------------------
# recruiter / agency postings on the scraped side
# ---------------------------------------------------------------------------

def test_derive_agency_posting_trusts_the_extractor_including_a_false():
    """A False from the extractor is a judgement, not a gap: a model that looked at the page and
    said 'not an agency' must not be overridden by a name regex."""
    assert agent.derive_agency_posting(_make_extract(company='CyberCoders', is_agency=True)) is True
    assert agent.derive_agency_posting(_make_extract(company='Motion Recruitment', is_agency=False)) is False


@pytest.mark.parametrize('extract_kwargs', [
    {'company': 'Motion Recruitment'},
    {'company': 'Insight Global Staffing'},
    {'company': 'Acme Talent Solutions'},
    {'description': 'Our client is a leading fintech scaling its AI platform.'},
    {'description': 'We are recruiting on behalf of a confidential client in healthcare.'},
])
def test_derive_agency_posting_fallback_detects_recruiters(extract_kwargs):
    assert agent.derive_agency_posting(_make_extract(**extract_kwargs)) is True


@pytest.mark.parametrize('extract_kwargs', [
    {},
    # A real product company whose NAME contains 'Agency' -- present in the historical corpus.
    {'company': 'AgencyAnalytics'},
    # 'client' in the work itself is not 'our client is'.
    {'description': 'You will present findings to clients and iterate on their feedback.'},
    {'description': 'Build client-facing dashboards for enterprise customers.'},
])
def test_derive_agency_posting_fallback_does_not_overmatch(extract_kwargs):
    assert agent.derive_agency_posting(_make_extract(**extract_kwargs)) is False


def test_agency_warning_names_the_end_client_when_known():
    extract = _make_extract(company='CyberCoders', is_agency=True,
                            end_client='Automotive Martech Inc', salary='CAD 250,000')
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert 'Posted by a recruiting agency — hiring company: Automotive Martech Inc' in warnings


def test_agency_warning_says_so_when_the_client_is_anonymous():
    """The common case: agencies anonymise. CyberCoders' real posting said 'undisclosed automotive
    martech firm', and 0 of 10 agency-posted applied records named a client."""
    extract = _make_extract(company='CyberCoders', is_agency=True, end_client='', salary='CAD 250,000')
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert 'Posted by a recruiting agency — actual hiring company not named' in warnings


def test_no_agency_warning_for_a_direct_employer():
    extract = _make_extract(salary='CAD 200,000 - 240,000 per year')
    assert not any('agency' in w.lower() for w in agent.build_deterministic_warnings(_make_candidate(), extract))


def test_agency_status_never_changes_the_rating():
    """The user's explicit constraint, and the most likely thing a later change breaks: an agency
    posting is warned about, never penalised. CyberCoders rated 5/5 on the 2026-08-11 run."""
    direct = _make_extract(workplace_type='remote', salary='CAD 250,000')
    agency = _make_extract(workplace_type='remote', salary='CAD 250,000',
                           company='CyberCoders', is_agency=True, end_client='')
    for rating in (1, 3, 4, 5):
        assert agent.apply_rating_caps(agency, rating) == agent.apply_rating_caps(direct, rating), \
            f'agency status changed the rating at {rating}'
    # And specifically: a 5/5 agency posting still reaches the notification threshold.
    assert agent.apply_rating_caps(agency, 5) == (5, '')


async def test_agency_posting_is_not_hard_ruled():
    """Agency status must never reject a job -- only warn. An auto-reject is unappealable."""
    extract = _make_extract(company='Quik Hire Staffing', is_agency=True, end_client='',
                            description='Our client is a leading AI lab. Build agents in Python.')
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


@pytest.mark.parametrize('company, end_client, expected', [
    ('CyberCoders', 'Automotive Martech Inc', 'Automotive Martech Inc'),
    ('CyberCoders', '', ''),
    # Measured on real jobs: glm echoes the poster back as end_client for direct postings.
    ('lululemon', 'lululemon', ''),
    ('Workday', 'Workday, Inc.', ''),
])
def test_derive_end_client_ignores_the_poster_echoed_back(company, end_client, expected):
    assert agent.derive_end_client(_make_extract(company=company, end_client=end_client)) == expected


def test_direct_posting_does_not_get_a_redundant_hiring_company_line():
    extract = _make_extract(company='lululemon', end_client='lululemon', is_agency=False)
    assert 'Hiring company' not in agent.format_extract_text(_make_candidate(), extract)


# ---------------------------------------------------------------------------
# recruiter reposts: one role advertised under many job ids
#
# 2026-09-10: Archer Recruitment advertised ONE pharma programme under 6+ LinkedIn job ids in nine
# days, each a different country and day rate, and every one rated 4-5 and notified. (site, job_id)
# dedup cannot see it and the already-applied blocklist deliberately never holds agency names.
#
# Text similarity cannot catch it either: the extractor REWRITES each description, so three of the
# six overlapped 0.62-0.86 by word shingles and the other three ~0.0, while unrelated repostings by
# direct employers scored 0.7+. So the gate is "same agency, notified recently" in code, and then
# one cheap LLM call that PICKS a prior job_id out of a list it was shown.
# ---------------------------------------------------------------------------

_PHARMA_DESCRIPTION = (
    'Lead the delivery of agentic AI across a global manufacturing operation at one of the most '
    'recognised healthcare and pharmaceutical organisations, as part of a multi-year technology '
    'transformation programme. Python, LangGraph, LangChain, RAG, tool calling, human-in-the-loop.'
)


def _agency_extract(**overrides) -> dict:
    base = {
        'company': 'Archer Recruitment', 'is_agency': True, 'end_client': '',
        'title': 'Lead Agentic AI Engineer', 'salary': '700-750 EUR/day',
        'location': 'European Union (Remote)', 'description': _PHARMA_DESCRIPTION,
        'workplace_type': 'remote',
    }
    base.update(overrides)
    return _make_extract(**base)


def _repost_llm(monkeypatch, same_as_job_id: str, reason: str = 'same pharma programme') -> list:
    """Patch both providers to answer the same way. Returns the list of prompts actually sent.

    A call COUNT rather than a raising stub, deliberately: `recruiter_repost_of` fails open, so a
    stub that raised would be swallowed and a test asserting "no call was made" would pass whether
    or not the call happened.
    """
    calls: list[str] = []
    answer = {'same_as_job_id': same_as_job_id, 'reason': reason}

    async def fake_openrouter(prompt, stage_stats):
        calls.append(prompt)
        return answer

    monkeypatch.setattr(tools, '_repost_match_openrouter', fake_openrouter)
    _make_sdk_mock(monkeypatch, answer)
    return calls


async def test_recruiter_repost_check_is_free_when_that_agency_has_no_recent_posting(monkeypatch):
    """The gate that keeps this off every other job: another agency's posting costs no LLM call."""
    calls = _repost_llm(monkeypatch, '111')
    tools.record_recruiter_notification(
        _make_candidate(job_id='111'), _agency_extract(company='Other Recruiters')
    )
    assert await tools.recruiter_repost_of(_make_candidate(job_id='222'), _agency_extract()) is None
    assert calls == []


async def test_recruiter_repost_matches_a_prior_posting_of_the_same_role(monkeypatch):
    """The Archer case: one role, a new job id, a different country and a different day rate."""
    _repost_llm(monkeypatch, '4464637425')
    tools.record_recruiter_notification(_make_candidate(job_id='4464637425'), _agency_extract())
    match = await tools.recruiter_repost_of(
        _make_candidate(job_id='4465094319'),
        _agency_extract(location='Netherlands (Remote)', salary='800-850 EUR/day'),
    )
    assert match is not None
    assert match['job_id'] == '4464637425'
    assert match['reason'] == 'same pharma programme'


async def test_recruiter_repost_ignores_a_job_id_it_was_never_shown(monkeypatch):
    """Anti-fabrication: the model CHOOSES from a list, so an id outside that list is no match."""
    _repost_llm(monkeypatch, '9999999999')
    tools.record_recruiter_notification(_make_candidate(job_id='4464637425'), _agency_extract())
    assert await tools.recruiter_repost_of(
        _make_candidate(job_id='4465094319'), _agency_extract()
    ) is None


async def test_recruiter_repost_fails_open_when_both_providers_fail(monkeypatch):
    """Deliberately opposite to the blacklist: a duplicate ping costs a glance, a missed
    notification costs a real role."""
    async def boom_openrouter(prompt, stage_stats):
        raise RuntimeError('tool server down')

    async def boom_sdk(**kwargs):
        raise RuntimeError('anthropic down')
        yield  # pragma: no cover — make it an async generator

    monkeypatch.setattr(tools, '_repost_match_openrouter', boom_openrouter)
    monkeypatch.setattr(tools, 'sdk_query', boom_sdk)
    tools.record_recruiter_notification(_make_candidate(job_id='4464637425'), _agency_extract())
    assert await tools.recruiter_repost_of(
        _make_candidate(job_id='4465094319'), _agency_extract()
    ) is None


async def test_recruiter_repost_ignores_a_prior_outside_the_window(monkeypatch):
    calls = _repost_llm(monkeypatch, '111')
    stale = date.today() - timedelta(days=config.RECRUITER_REPOST_WINDOW_DAYS + 1)
    tools.record_recruiter_notification(_make_candidate(job_id='111'), _agency_extract(), today=stale)
    assert tools.load_recruiter_notifications() == []
    assert await tools.recruiter_repost_of(_make_candidate(job_id='222'), _agency_extract()) is None
    assert calls == []


async def test_recruiter_repost_matches_the_agency_across_name_variants(monkeypatch):
    """Folding happens at the comparison; the stored name keeps the case it arrived with."""
    _repost_llm(monkeypatch, '111')
    tools.record_recruiter_notification(
        _make_candidate(job_id='111'), _agency_extract(company='Archer Recruitment Ltd.')
    )
    assert tools.load_recruiter_notifications()[0]['agency'] == 'Archer Recruitment Ltd.'
    match = await tools.recruiter_repost_of(
        _make_candidate(job_id='222'), _agency_extract(company='Archer Recruitment')
    )
    assert match is not None and match['job_id'] == '111'


async def test_recruiter_repost_compares_only_the_most_recent_priors(monkeypatch):
    """An agency that posts daily would otherwise grow the prompt without bound."""
    calls = _repost_llm(monkeypatch, '')
    monkeypatch.setattr(tools, 'RECRUITER_REPOST_MAX_PRIORS', 2)
    for name, days_ago in (('prior-oldest', 5), ('prior-middle', 3), ('prior-newest', 1)):
        tools.record_recruiter_notification(
            _make_candidate(job_id=name), _agency_extract(),
            today=date.today() - timedelta(days=days_ago),
        )

    await tools.recruiter_repost_of(_make_candidate(job_id='new'), _agency_extract())

    assert len(calls) == 1
    prompt = calls[0]
    assert 'prior-newest' in prompt and 'prior-middle' in prompt
    assert 'prior-oldest' not in prompt


def test_repost_prompt_says_what_is_not_evidence_and_which_way_to_err():
    """Measured 2026-09-11: replayed over the real corpus, the first version of this prompt called
    two unrelated roles from ONE aggregator the same role, reasoning from 'adjacent job IDs' and the
    shared agency. A false match hides a real job, so the prompt names the non-evidence explicitly
    and says which way to err."""
    prompt = tools._repost_match_prompt({'job_id': 'new'}, [{'job_id': '1'}])
    assert 'NOT evidence' in prompt
    assert 'adjacent job ids' in prompt
    assert 'If you are not sure, answer with an empty string.' in prompt


def test_corrupt_recruiter_notifications_file_is_treated_as_empty():
    """Fails open, unlike location_recommendations.yaml: this file holds no decision the user made
    by hand, and an unreadable one must never be able to SUPPRESS a notification."""
    tools.RECRUITER_NOTIFICATIONS_PATH.write_text('not: [valid', encoding='utf-8')
    assert tools.load_recruiter_notifications() == []


def test_recruiter_notification_record_fields():
    entry = tools.record_recruiter_notification(
        _make_candidate(job_id='4464637425'), _agency_extract()
    )
    assert entry['agency'] == 'Archer Recruitment'
    assert entry['agency_key'] == 'archer recruitment'
    assert entry['notified'] == date.today().isoformat()
    assert len(entry['description']) <= tools.RECRUITER_DESCRIPTION_MAX_CHARS
    assert tools.load_recruiter_notifications()[0]['job_id'] == '4464637425'


def test_repost_warning_names_the_earlier_posting():
    warning = agent.repost_warning({
        'job_id': '4464637425', 'notified': '2026-09-09',
        'location': 'European Union (Remote)', 'salary': '700-750 EUR/day',
    })
    assert '4464637425' in warning and '2026-09-09' in warning


@pytest.mark.live
async def test_recruiter_repost_judgement_live():
    """The judgement itself, against the real model: same role reworded vs. a genuinely different one.

    Synthetic postings on purpose — the saved corpus is personal, and what is under test is whether
    the model can tell a re-advertised role from a different one by the same agency, which is the
    whole reason this is not a similarity threshold.
    """
    tools.record_recruiter_notification(_make_candidate(job_id='1001'), _agency_extract())

    same = await tools.recruiter_repost_of(
        _make_candidate(job_id='1002'),
        _agency_extract(
            title='Senior Agentic AI Engineer',
            location='Netherlands (Remote — anywhere in the EU)',
            salary='800-850 EUR/day',
            description=(
                'Senior agentic AI engineer for a multi-year technology transformation at a global '
                'healthcare and pharmaceutical manufacturer. Design multi-agent architectures in '
                'Python using LangGraph and LangChain, with RAG, memory, tool calling and '
                'human-in-the-loop, and own them through to production.'
            ),
        ),
    )
    assert same is not None and same['job_id'] == '1001', 'a reworded repost of one role'

    different = await tools.recruiter_repost_of(
        _make_candidate(job_id='1003'),
        _agency_extract(
            title='Embedded Firmware Engineer',
            location='Dublin, Ireland (Hybrid)',
            salary='75,000 EUR/yr',
            description=(
                'Embedded firmware engineer for a medical devices manufacturer. C and C++ on ARM '
                'microcontrollers, RTOS scheduling, board bring-up and IEC 62304 compliance.'
            ),
        ),
    )
    assert different is None, 'a different role from the same agency'


# --- the same thing end to end, through Stage 2 -----------------------------

def _stage2_stats() -> dict:
    return {k: agent.new_stage_stats() for k in ('extraction', 'rating', 'recruiter_repost', 'salary')}


def _stage2_stubs(monkeypatch, extract: dict, rating: int = 5) -> tuple[list, list]:
    """Everything around the repost check stubbed: extraction, triage, rating, saving, Telegram."""
    notifications: list[str] = []
    saved: list[dict] = []

    monkeypatch.setattr(agent, 'MODEL_NAME_EXTRACTOR', OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC)

    async def fake_extract(candidate, page_text, stats, system_prompt):
        return dict(extract)

    async def fake_triage(extract_text, profile_block):
        return None

    async def fake_rate(evaluator_prompt, extract_text, stats):
        return {
            'rating': rating, 'reasoning': 'strong fit', 'summary': 'strong fit',
            'title': extract['title'], 'company': extract['company'], 'pros': [], 'warnings': [],
        }

    async def fake_save(company, description, rating, content, job_id=None):
        saved.append({'company': company, 'rating': rating, 'content': content, 'job_id': job_id})
        return {}

    async def fake_notify(text):
        notifications.append(text)

    _stub_read_page(monkeypatch)
    monkeypatch.setattr(agent, 'extract_job_page_openrouter', fake_extract)
    monkeypatch.setattr(agent, 'triage_job_fit', fake_triage)
    monkeypatch.setattr(agent, 'rate_job', fake_rate)
    monkeypatch.setattr(tools, 'do_save_job_posting', fake_save)
    monkeypatch.setattr(agent, '_send_pipeline_notification', fake_notify)
    return notifications, saved


async def _run_stage2(candidate: dict, funnel: dict) -> None:
    await agent.evaluate_all_candidates(
        [candidate], {'type': 'http', 'url': 'http://localhost:1/mcp'},
        'prompt', 'profile', _stage2_stats(), funnel=funnel,
    )


async def test_stage_2_notifies_and_records_a_first_agency_posting(monkeypatch):
    calls = _repost_llm(monkeypatch, '')
    notifications, _saved = _stage2_stubs(monkeypatch, _agency_extract())
    funnel: dict = {}

    await _run_stage2(_make_candidate(job_id='4464637425', company='Archer Recruitment'), funnel)

    assert len(notifications) == 1
    assert calls == [], 'nothing to compare against yet, so no LLM call'
    assert tools.load_recruiter_notifications()[0]['job_id'] == '4464637425'
    assert 'recruiter_repost_suppressed' not in funnel


async def test_stage_2_suppresses_the_notification_for_a_recruiter_repost(monkeypatch):
    _repost_llm(monkeypatch, '4464637425')
    tools.record_recruiter_notification(_make_candidate(job_id='4464637425'), _agency_extract())
    notifications, saved = _stage2_stubs(
        monkeypatch, _agency_extract(location='Netherlands (Remote)', salary='800-850 EUR/day')
    )
    funnel: dict = {}

    await _run_stage2(_make_candidate(job_id='4465094319', company='Archer Recruitment'), funnel)

    assert notifications == []
    assert funnel.get('recruiter_repost_suppressed') == 1
    # A cap it is not: the rating, the saved file and the audit trail are untouched.
    assert saved and saved[0]['rating'] == 5
    assert 'Likely repost of job 4464637425' in saved[0]['content']
    # The suppressed repost does NOT become the new reference point.
    assert [e['job_id'] for e in tools.load_recruiter_notifications()] == ['4464637425']


async def test_stage_2_does_not_check_a_direct_employer(monkeypatch):
    """Only recruiters repost one role under many ids; a direct employer must never be compared."""
    calls = _repost_llm(monkeypatch, '111')
    tools.record_recruiter_notification(_make_candidate(job_id='111'), _agency_extract(company='Acme'))
    notifications, _ = _stage2_stubs(
        monkeypatch, _make_extract(company='Acme', is_agency=False, salary='CAD 250,000')
    )
    funnel: dict = {}

    await _run_stage2(_make_candidate(job_id='222', company='Acme'), funnel)

    assert calls == []
    assert len(notifications) == 1


async def test_stage_2_does_not_check_a_job_that_would_not_be_notified(monkeypatch):
    """The check exists to stop a duplicate MESSAGE, so a rating below the threshold never pays."""
    calls = _repost_llm(monkeypatch, '4464637425')
    tools.record_recruiter_notification(_make_candidate(job_id='4464637425'), _agency_extract())
    notifications, _ = _stage2_stubs(monkeypatch, _agency_extract(), rating=3)
    funnel: dict = {}

    await _run_stage2(_make_candidate(job_id='4465094319', company='Archer Recruitment'), funnel)

    assert calls == []
    assert notifications == []


async def test_stage_2_compares_against_a_posting_notified_earlier_in_the_same_run(monkeypatch):
    """Both Sep 09 Archer postings arrived in one run, so within-run recording is load-bearing."""
    _repost_llm(monkeypatch, '4464637425')
    notifications, _ = _stage2_stubs(monkeypatch, _agency_extract())
    funnel: dict = {}

    await agent.evaluate_all_candidates(
        [
            _make_candidate(job_id='4464637425', company='Archer Recruitment'),
            _make_candidate(job_id='4465094319', company='Archer Recruitment'),
        ],
        {'type': 'http', 'url': 'http://localhost:1/mcp'},
        'prompt', 'profile', _stage2_stats(), funnel=funnel,
    )

    assert len(notifications) == 1, 'the second posting is the same role as the first'
    assert funnel.get('recruiter_repost_suppressed') == 1


def test_extract_text_surfaces_the_hiring_company():
    extract = _make_extract(company='CyberCoders', is_agency=True, end_client='Automotive Martech Inc')
    text = agent.format_extract_text(_make_candidate(), extract)
    assert 'Hiring company: Automotive Martech Inc' in text


def test_extract_schemas_stay_in_sync_across_both_providers():
    """A field added to one extractor and not the other silently returns empty for that provider,
    which looks identical to 'the posting did not say'."""
    # The SDK @tool decorator keeps the raw schema on the wrapped function.
    anthropic_schema = tools.submit_job_extract.input_schema['properties']
    openrouter_schema = next(
        t['function']['parameters']['properties']
        for t in extract_openrouter.OPENROUTER_EXTRACT_TOOLS
        if t['function']['name'] == 'submit_job_extract'
    )
    assert set(anthropic_schema) == set(openrouter_schema), (
        'extract schemas drifted: '
        f'anthropic-only={sorted(set(anthropic_schema) - set(openrouter_schema))}, '
        f'openrouter-only={sorted(set(openrouter_schema) - set(anthropic_schema))}'
    )
    for field in ('is_agency', 'end_client', 'residency_scope'):
        assert field in anthropic_schema and field in openrouter_schema


# The direct fallback is a THIRD surface, and it drifted unnoticed: `is_agency`/`end_client` are
# declared on both agentic extractors and on neither half of the deterministic path, so a posting
# that falls through to it silently loses agency detection to a regex docs/requirements.md (Flag Agency Posting) measures at 1-in-10
# against the extractor field's 15-in-16. Recorded rather than silently fixed — the point here is
# that residency_scope must not join it.
KNOWN_DIRECT_FALLBACK_GAPS = {'is_agency', 'end_client'}


def test_direct_fallback_schema_matches_the_agentic_extractors():
    anthropic_schema = set(tools.submit_job_extract.input_schema['properties'])
    direct_schema = set(agent.EXTRACT_OUTPUT_SCHEMA['properties'])
    assert anthropic_schema - direct_schema == KNOWN_DIRECT_FALLBACK_GAPS, (
        'a new field reached the agentic extractors but not extract_job_page_direct — a posting '
        'that falls through to the fallback would silently return empty for it'
    )
    assert not direct_schema - anthropic_schema


def test_residency_scope_is_declared_optional_on_every_extractor():
    """A required parameter the model cannot observe is fabrication pressure, not validation."""
    assert 'residency_scope' not in tools.submit_job_extract.input_schema['required']
    openrouter_tool = next(
        t for t in extract_openrouter.OPENROUTER_EXTRACT_TOOLS
        if t['function']['name'] == 'submit_job_extract'
    )
    assert 'residency_scope' not in openrouter_tool['function']['parameters']['required']
    assert 'residency_scope' not in agent.EXTRACT_OUTPUT_SCHEMA['required']


def test_residency_scope_vocabulary_is_closed_and_identical_everywhere():
    expected = ['country_only', 'area_wide', '']
    openrouter_tool = next(
        t for t in extract_openrouter.OPENROUTER_EXTRACT_TOOLS
        if t['function']['name'] == 'submit_job_extract'
    )
    for properties in (
        tools.submit_job_extract.input_schema['properties'],
        openrouter_tool['function']['parameters']['properties'],
        agent.EXTRACT_OUTPUT_SCHEMA['properties'],
    ):
        assert properties['residency_scope']['enum'] == expected
    assert list(agent.RESIDENCY_SCOPES) == ['country_only', 'area_wide']


def test_merge_warnings_dedupes_case_insensitively_and_keeps_order():
    merged = agent.merge_warnings(['Hybrid — Netherlands'], ['hybrid — netherlands', 'Below salary target'])
    assert merged == ['Hybrid — Netherlands', 'Below salary target']


def test_merge_warnings_drops_a_restated_missing_salary():
    """The reported Telegram pair collapses to the one code-built line."""
    merged = agent.merge_warnings(
        ['No salary listed'],
        ['No salary listed — unverified against CAD $210K+ base target', 'AWS-heavy stack'],
    )
    assert merged == ['No salary listed', 'AWS-heavy stack']


def test_merge_warnings_salary_estimate_replaces_the_code_line_in_place():
    merged = agent.merge_warnings(
        ['Hybrid — Netherlands', 'No salary listed', 'Posted by a recruiting agency — actual hiring company not named'],
        ['No salary listed; Poland-market pay likely well below CAD $210K target'],
    )
    assert merged == [
        'Hybrid — Netherlands',
        'No salary listed; Poland-market pay likely well below CAD $210K target',
        'Posted by a recruiting agency — actual hiring company not named',
    ]


def test_merge_warnings_prefixes_an_estimate_that_does_not_state_the_absence():
    merged = agent.merge_warnings(['No salary listed'], ['Poland-market pay likely well below CAD $210K target'])
    assert merged == ['No salary listed — Poland-market pay likely well below CAD $210K target']


def test_merge_warnings_keeps_exactly_one_salary_line():
    merged = agent.merge_warnings(
        ['No salary listed'],
        ['No salary listed — compensation unknown', 'No salary stated; Spain comp likely below CAD $210K base target'],
    )
    assert merged == ['No salary stated; Spain comp likely below CAD $210K base target']


@pytest.mark.parametrize('warning', [
    'Seniority not specified in title (no Principal/Staff label) — leveling and comp should be confirmed',
    'Hybrid in Madrid — acceptable location, but no salary listed so the compensation condition is not met',
])
def test_merge_warnings_leaves_bullets_that_only_mention_pay(warning):
    assert agent.merge_warnings(['No salary listed'], [warning]) == ['No salary listed', warning]


def test_merge_warnings_drops_the_unclear_if_it_meets_target_restatement():
    """The 2026-09-14 BairesDev pair."""
    merged = agent.merge_warnings(['No salary listed'], ['No salary listed — unclear if it meets the CAD $210K+ target'])
    assert merged == ['No salary listed']


def test_merge_warnings_trims_a_salary_absence_clause_after_a_semicolon():
    """The 2026-09-17 Network Solutions bullet: the salary clause trailed an unrelated one."""
    merged = agent.merge_warnings(
        ['No salary listed'],
        ['Senior (not Principal/Staff) title; no pay figure to confirm the CAD $210K+ base target'],
    )
    assert merged == ['No salary listed', 'Senior (not Principal/Staff) title']


def test_merge_warnings_keeps_a_trailing_salary_clause_that_carries_an_estimate():
    warning = 'Early-stage team; no salary figure, equity-heavy offer expected'
    merged = agent.merge_warnings(['No salary listed'], [warning])
    assert merged == ['No salary listed', warning]


def test_merge_warnings_does_not_trim_clauses_when_a_figure_is_listed():
    warning = 'Senior (not Principal/Staff) title; no pay figure to confirm the CAD $210K+ base target'
    assert agent.merge_warnings(['Hybrid — Netherlands'], [warning]) == ['Hybrid — Netherlands', warning]


def test_merge_warnings_leaves_salary_bullets_alone_when_a_figure_is_listed():
    """Below-target is the rater's call when a salary IS listed; nothing is folded."""
    merged = agent.merge_warnings(['Hybrid — Netherlands'], ['Salary band tops out below the CAD $210K target'])
    assert merged == ['Hybrid — Netherlands', 'Salary band tops out below the CAD $210K target']


@pytest.mark.parametrize('clause', [
    'Do not write a warning bullet for any of these: they are detected deterministically and added for you.',
    'Do NOT write a warning about the poster being a recruiting agency',
    'do not write a bullet stating that the role is hybrid or on-site',
    'do not write a bullet that only restates that the role is a contract',
    'Do not write a bullet that only says the salary is missing',
])
def test_evaluator_prompt_tells_the_rater_not_to_restate_deterministic_warnings(clause):
    assert clause in agent.EVALUATOR_INSTRUCTIONS_TEMPLATE


@pytest.mark.parametrize('topic', ['missing salary', 'hybrid/on-site', 'contract vs full-time', 'language expectations'])
def test_evaluator_prompt_does_not_ask_the_rater_for_deterministic_warnings(topic):
    warnings_line = next(line for line in agent.EVALUATOR_INSTRUCTIONS_TEMPLATE.splitlines() if line.startswith('- warnings:'))
    assert topic not in warnings_line


def test_format_job_notification_has_both_bullet_sections():
    extract = _make_extract(location='Netherlands (Hybrid - 2-3 days onsite)')
    message = agent.format_job_notification(
        _make_candidate(), extract, 4, ['Python/FastAPI, RAG'], ['Hybrid — Netherlands'],
    )
    assert '⭐ 4/5 — Acme — Staff AI Engineer' in message
    assert '📍 Netherlands (Hybrid - 2-3 days onsite)' in message
    assert '✅ Good:\n• Python/FastAPI, RAG' in message
    assert '⚠️ Warnings:\n• Hybrid — Netherlands' in message
    assert message.endswith('https://example.com/job/123')


def test_format_job_notification_omits_empty_sections():
    message = agent.format_job_notification(_make_candidate(), _make_extract(), 5, [], [])
    assert '✅ Good:' not in message
    assert '⚠️ Warnings:' not in message


def test_format_job_notification_appends_workplace_when_location_omits_it():
    extract = _make_extract(location='Amsterdam, Netherlands', workplace_type='hybrid')
    message = agent.format_job_notification(_make_candidate(), extract, 4, [], [])
    assert '📍 Amsterdam, Netherlands — hybrid' in message


def test_format_extract_text_surfaces_workplace_line():
    text = agent.format_extract_text(
        _make_candidate(), _make_extract(location='Netherlands (Hybrid - 2-3 days onsite)')
    )
    assert 'Workplace: hybrid' in text


async def test_devologyx_regression_end_to_end():
    """The exact posting that was wrongly surfaced as a 4: capped to 3 (so never notified)
    and carrying hybrid + contract warnings."""
    candidate = _make_candidate(
        job_id='4442818316', company='DevologyX', title='Senior AI Engineer (Contract)',
        url='https://www.linkedin.com/jobs/view/4442818316/', snippet='Netherlands (Remote)',
    )
    extract = _make_extract(
        title='Senior AI Engineer (Contract)', company='DevologyX',
        location='Netherlands (Hybrid - 2-3 days onsite)',
        description=(
            'Senior AI Engineer (Contract) for a large-scale AI transformation programme. '
            'Contract Length: 6-12 Months (Extension Possible). Rate: EUR 700-900/day.'
        ),
        salary='€700-€900/day (DOE)', date_posted='1 week ago', language_requirement='english',
    )
    # As of the 2026-09 location gate this posting is now REJECTED outright rather than capped:
    # the Netherlands is in a rejected region. That is a strictly stronger version of what this
    # regression has always demanded (it must never be notified), so the test asserts the new
    # outcome rather than being weakened to accommodate it.
    assert await agent.apply_hard_rules(candidate, extract) == (
        'located in an excluded region: Netherlands'
    )

    # The original mechanism still has to work on its own, for a posting the location gate does
    # not reach — otherwise this regression would silently stop testing the hybrid cap at all.
    hybrid_only = dict(extract, location='Testville (Hybrid - 2-3 days onsite)')
    assert await agent.apply_hard_rules(candidate, hybrid_only) is None
    unacceptable = dict(extract, location='Nowhereton (Hybrid - 2-3 days onsite)')
    rating, cap_reason = agent.apply_rating_caps(unacceptable, 4)
    assert rating == 3
    assert rating < 4, 'a capped job must fall below the >=4 notification threshold'
    assert 'hybrid' in cap_reason

    warnings = agent.merge_warnings(
        agent.build_deterministic_warnings(candidate, unacceptable), ['Contract, not full-time'],
    )
    assert any('Hybrid' in w for w in warnings)
    assert any('Contract role' in w for w in warnings)


# ---------------------------------------------------------------------------
# preferences: neutral defaults, so an unconfigured checkout inherits nobody's situation
# ---------------------------------------------------------------------------

@pytest.fixture
def neutral_preferences(monkeypatch):
    """Preferences as a fresh clone with no run_dir/preferences.yaml sees them."""
    defaults = preferences._deep_merge(preferences.DEFAULT_PREFERENCES, {})
    monkeypatch.setattr(preferences, '_cache', defaults)
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: defaults)
    return defaults


def test_default_preferences_apply_no_personal_gates(neutral_preferences):
    assert preferences.sponsorship_required_in() == ()
    assert preferences.languages() == ()
    assert preferences.rejected_degrees() == ()
    assert preferences.would_commute_here() == ()
    assert preferences.search_regions() == []
    assert preferences.would_not_live_here() == ()
    assert preferences.would_not_live_here() == ()


async def test_hard_rules_reject_nothing_without_preferences(neutral_preferences, stub_location_classifier):
    """Every personal gate must be inert when unconfigured — not silently inherited."""
    extract = _make_extract(
        location='Berlin, Germany (Remote)',
        description='PhD in Machine Learning is required. Fluent Dutch is required.',
        language_requirement='dutch',
        education_requirement='phd',
        relocation='Germany',
    )
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None
    assert stub_location_classifier == [], 'an unconfigured checkout must not make a paid call'


def test_language_cap_and_warnings_inert_without_preferences(neutral_preferences):
    """With no languages configured there is nothing to be foreign to — no cap, no warning."""
    extract = _make_extract(
        posting_language='french', likely_working_language='french', salary='CAD 200,000 - 240,000 per year',
    )
    assert agent.foreign_posting_language(extract) == ''
    assert agent.foreign_likely_working_language(extract) == ''
    assert agent.apply_rating_caps(extract, 5) == (5, '')
    assert agent.build_deterministic_warnings(_make_candidate(), extract) == []


async def test_hard_rules_still_reject_impersonal_conditions(neutral_preferences):
    """Closed and stale are properties of the posting, not the person — always on."""
    assert 'closed' in await agent.apply_hard_rules(_make_candidate(), _make_extract(closed=True))
    old = (date.today() - timedelta(days=45)).isoformat()
    assert 'older than' in await agent.apply_hard_rules(_make_candidate(), _make_extract(date_posted=old))


def test_sponsorship_prompt_prints_place_names_as_written(monkeypatch):
    """Never `.title()` a place name: it already has its case, and `.title()` makes 'USA' 'Usa'."""
    monkeypatch.setattr(preferences, 'sponsorship_required_in', lambda: ('USA', 'United Kingdom'))
    section = agent._sponsorship_prompt_section()
    assert 'located in USA, United Kingdom and' in section
    assert 'Usa' not in section


def test_prompts_omit_personal_sections_without_preferences(neutral_preferences):
    evaluator = agent.build_evaluator_instructions()
    assert 'visa sponsorship' not in evaluator
    assert 'There is nowhere the candidate would commute to' in evaluator
    assert 'visa sponsorship' not in agent.build_interactive_instructions()
    scraper = agent.build_scraper_instructions()
    assert '&location=' not in scraper, 'no region configured means an unfiltered search'


def test_missing_preferences_file_falls_back_to_defaults(tmp_path, monkeypatch, real_load_preferences):
    monkeypatch.setattr(preferences, '_cache', None)
    monkeypatch.setattr(preferences, 'PREFERENCES_PATH', tmp_path / 'absent.yaml')
    assert real_load_preferences(force_reload=True) == preferences.DEFAULT_PREFERENCES


def test_malformed_preferences_file_raises_with_context(tmp_path, monkeypatch, real_load_preferences):
    path = tmp_path / 'preferences.yaml'
    path.write_text('- not: a mapping\n', encoding='utf-8')
    monkeypatch.setattr(preferences, '_cache', None)
    monkeypatch.setattr(preferences, 'PREFERENCES_PATH', path)
    with pytest.raises(ValueError, match='must contain a YAML mapping'):
        real_load_preferences(force_reload=True)


def test_example_preferences_file_is_neutral():
    """The tracked example must never carry a real person's settings."""
    text = preferences.EXAMPLE_PREFERENCES_PATH.read_text(encoding='utf-8')
    loaded = yaml.safe_load(text)
    assert loaded['sponsorship_required_in'] == []
    assert loaded['reject_required_degrees'] == []
    assert loaded['current_tax_residence'] == ''
    for key in ('would_live_here', 'would_not_live_here', 'not_yet_bucketed', 'would_commute_here'):
        assert loaded['locations'][key] == [], key
    assert 'acceptable_locations' not in loaded['hybrid'], 'the example must use the new key names'
    for region in loaded['search_regions']:
        assert 'Example' in region['name'] or 'Test' in region['name']


# ---------------------------------------------------------------------------
# alias tier: exonyms no normalization can reach
# ---------------------------------------------------------------------------

async def test_exonym_matches_the_listed_english_name(monkeypatch):
    """'München' on the page, 'Munich' on the list. No amount of folding connects those two.

    Germany is in a rejected region, so the alias tier is the ONLY thing that can return '' here —
    the negative control below is what makes that true rather than incidental.
    """
    assert not location.location_token_matches('Munich', 'München, Bavaria, Germany')
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ())
    assert await agent.rejected_location('München, Bavaria, Germany (Remote)') == 'Germany'
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('Munich',))
    assert await agent.rejected_location('München, Bavaria, Germany (Remote)') == ''


async def test_exonym_on_the_deny_list_also_rejects(monkeypatch):
    """Tiers 1 and 2 compare the same kind of list against the same text; they must not disagree
    about what counts as a match, which is how `roma` survived on one list while the other was
    empty."""
    monkeypatch.setattr(preferences, 'would_live_here', lambda: ())
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ())
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ('Seville',))
    # 'Sevilla' on the page, 'Seville' on the deny list: matched through the classifier's
    # place_names, which is the only thing that connects the two spellings.
    assert await agent.rejected_location('Sevilla, Andalusia, Spain (Hybrid)') == 'Seville'


async def test_alias_tier_does_not_fire_when_the_classifier_names_nothing(stub_location_classifier):
    """Fail-open is unchanged: no place names, no alias match, no new way to reject."""
    assert await agent.rejected_location('Remote (EMEA)') == ''


def test_hybrid_cap_uses_the_resolved_place_names(monkeypatch):
    """Without this the async gate and the sync cap would disagree on the same posting."""
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('Seville',))
    extract = _make_extract(location='Sevilla, Spain (Hybrid)', workplace_type='hybrid')
    assert agent.apply_rating_caps(extract, 5) == (3, 'hybrid in Sevilla, Spain (Hybrid) (not an acceptable hybrid location)')
    resolved = dict(extract, place_names=['Sevilla', 'Seville', 'Spain'])
    assert agent.apply_rating_caps(resolved, 5) == (5, '')


def test_place_names_rule_asks_for_omitted_levels_not_proximity():
    """The radius question is answered by administrative containment, not by a distance judgement.

    A posting naming only a town ('Grasse, France') used to come back as ['Grasse', 'France'], so a
    commute list of southern-French cities could not match it and the office capped at 3. The rule
    now asks for the levels the text omits, which makes 'Provence-Alpes-Côte d'Azur' a usable list
    entry. It deliberately does NOT ask "is this near X": `place_names` feeds the GATE as well as
    the cap, and "which region contains this city" is checkable world knowledge while "commutable"
    is a judgement about cars, trains and tolerance.
    """
    prompt = location._PROMPT
    assert 'SUPPLY THE LEVELS IT OMITS' in prompt
    assert 'never proximity' in prompt


def test_classifier_returns_place_names_on_every_path():
    """Every return path must have the same shape, or a consumer reads a missing key as 'no aliases'."""
    assert location._empty('error')['place_names'] == []
    coerced = location._coerce({'countries': ['Spain'], 'regions': ['southern_europe'],
                                'place_names': ['  Sevilla ', 'Seville', 'Sevilla', '']})
    assert coerced['place_names'] == ['Sevilla', 'Seville'], 'trimmed, deduped, order preserved'
    assert location._coerce({})['place_names'] == []


def test_reviewer_flags_an_entry_that_only_matches_when_folded(monkeypatch):
    """The check that would have caught `malaga`, and the distinction it draws.

    An entry that has simply not come up yet looks identical to one that can never match. This
    separates them: it reports only entries whose folded form DID appear this run.
    """
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('malaga', 'Bologna'))
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ())
    location.reset_countries_seen()
    location._record_countries({'countries': ['Spain'], 'regions': ['southern_europe'],
                                'place_names': ['Málaga', 'Andalusia', 'Spain']})
    findings = location_review.dead_entries(location.place_names_seen_this_run())
    assert [f['entry'] for f in findings] == ['malaga'], "'Bologna' merely did not come up"
    assert findings[0]['collides_with'] == 'Málaga'


async def test_a_stale_cache_entry_survives_a_failed_reclassification(monkeypatch, tmp_path):
    """Adding `place_names` invalidates every cached entry. An outage must not turn that into a
    silently disabled gate.

    Treating a `place_names`-less entry as a miss is right — serving it would disable alias
    matching for that location forever. But the entry is stale, not wrong, so if the upgrade call
    cannot be made the old answer is still the best answer available. Without this, one tool-server
    outage fails every previously-cached location open at once, which is a far larger blast radius
    than the per-location fail-open the module is designed around.
    """
    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'location_cache.yaml')
    monkeypatch.setattr(location, '_cache', {
        # Deliberately the PRE-RENAME shape: no `place_names`, and the old `local_language` key.
        'berlin, germany': {'countries': ['germany'], 'regions': ['western_europe'],
                            'broad_area': False, 'local_language': 'german'},
    })
    async def _down(*a, **k):
        raise RuntimeError('unhandled errors in a TaskGroup (1 sub-exception)')
    monkeypatch.setattr(location, 'chat_openrouter', _down)

    facts = await location.classify_location('Berlin, Germany')
    assert facts['source'] == 'stale'
    assert facts['regions'] == ['western_europe'], 'the region gate still works from the old answer'
    assert facts['place_names'] == [], 'but no aliases until it can actually be upgraded'
    assert facts['likely_working_language'] == 'german', (
        'the pre-rename key still reads: this is the one path that serves an un-upgraded entry'
    )

    unseen = await location.classify_location('Somewhere Never Seen')
    assert unseen['source'] == 'error' and unseen['countries'] == []


async def test_an_entry_from_the_national_language_question_is_re_asked(monkeypatch, tmp_path):
    """The old question asked for the first COUNTRY's national language (Luxembourg ->
    luxembourgish, Montreal -> english). An entry carrying only that answer is a miss."""
    calls = []

    async def fake_chat(prompt, **kwargs):
        calls.append(prompt)
        return ('{"countries": ["Luxembourg"], "regions": ["western_europe"], "broad_area": false, '
                '"likely_working_language": "english", "place_names": ["Luxembourg"]}'), 0.0

    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'location_cache.yaml')
    monkeypatch.setattr(location, '_cache', {
        'luxembourg (remote)': {'countries': ['Luxembourg'], 'regions': ['western_europe'],
                                'broad_area': False, 'implied_local_language': 'luxembourgish',
                                'place_names': ['Luxembourg']},
    })
    monkeypatch.setattr(location, 'chat_openrouter', fake_chat)

    facts = await location.classify_location('Luxembourg (Remote)')
    assert len(calls) == 1 and facts['source'] == 'llm'
    assert facts['likely_working_language'] == 'english'


async def test_a_failed_re_ask_serves_the_old_language_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'location_cache.yaml')
    monkeypatch.setattr(location, '_cache', {
        'berlin, germany': {'countries': ['Germany'], 'regions': ['western_europe'], 'broad_area': False,
                            'implied_local_language': 'german', 'place_names': ['Berlin']},
    })
    async def _down(*a, **k):
        raise RuntimeError('tool server down')
    monkeypatch.setattr(location, 'chat_openrouter', _down)

    facts = await location.classify_location('Berlin, Germany')
    assert facts['source'] == 'stale'
    assert facts['likely_working_language'] == 'german'
    assert facts['place_names'] == ['Berlin']


def test_collisions_ignore_deliberate_spelling_pairs(monkeypatch):
    """The lists carry intentional variants; each used to flag the other, in both directions.

    'Valencia'/'València', 'Málaga'/'Malaga' and 'Milan'/'Milano' produced 5 false findings against
    1 real one on the first live run — an 83% false-positive rate, which is the cry-wolf failure
    this alert is supposed to avoid. An entry sitting inside another LISTED place is harmless: both
    are places the user wants, and whole-token matching keeps them apart. Only an entry sitting
    inside a place they never listed is dangerous.
    """
    monkeypatch.setattr(preferences, 'would_commute_here',
                        lambda: ('Valencia', 'València', 'Málaga', 'Malaga', 'Milan', 'Milano', 'Roma'))
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ())
    found = location_review.substring_collisions({'Romania': 'eastern_europe', 'Spain': 'southern_europe'})
    assert [(c['entry'], c['collides_with']) for c in found] == [('Roma', 'Romania')]


def test_countries_seen_takes_the_majority_region_not_the_first():
    """A rare wrong region must not decide what the reviewer reports.

    Austria came back `eastern_europe` from one Vienna entry while six others said
    `western_europe`; `setdefault` meant whichever was seen first won.
    """
    location.reset_countries_seen()
    location._record_countries({'countries': ['Austria'], 'regions': ['eastern_europe']})
    location._record_countries({'countries': ['Austria'], 'regions': ['western_europe']})
    location._record_countries({'countries': ['Austria'], 'regions': ['western_europe']})
    assert location.countries_seen_this_run() == {'Austria': 'western_europe'}


def test_reviewer_never_advises_writing_roma_as_romania(monkeypatch):
    """The folded comparison must stay WHOLE-TOKEN, or the reviewer recommends the original bug.

    'roma' is a folded substring of 'romania'. A dead-entry check written as `folded in _fold(name)`
    reports "'Roma' never matched, but 'Romania' did — write the entry the way the place is
    written", which is advice to reintroduce the exact entry that broke the gate. Folding may
    ignore case and accents; it may never ignore boundaries.
    """
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('Roma',))
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ())
    assert location_review.dead_entries({'Romania', 'Bucharest'}) == []
    # ...but it is still reported as an ambiguous entry by the collision check.
    collisions = location_review.substring_collisions({'Romania': 'eastern_europe'})
    assert [c['entry'] for c in collisions] == ['Roma']


# ---------------------------------------------------------------------------
# country list review — advisory, never enforced
# ---------------------------------------------------------------------------

def _seen(countries: dict[str, str]):
    """Seed the run-scoped accumulator the way classify_location would: country -> region."""
    location.reset_countries_seen()
    location._record_countries({
        'countries': list(countries), 'regions': list(countries.values()),
    })


def test_substring_collisions_flags_roma_against_romania(monkeypatch):
    """The deterministic half. This — not the prompt — is what catches the entry that broke the gate.

    'roma' inside 'romania' is arithmetic, and AGENTS.md's rule that a structural fact must never
    be left to a model's discretion applies to the reviewer as much as to the rater. It also means
    the warning still appears with the tool server down.
    """
    monkeypatch.setattr(preferences, 'would_live_here', lambda: ('Roma', 'Spain'))
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ('Romania',))
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ())
    collisions = location_review.substring_collisions({'Romania': 'eastern_europe', 'Spain': 'southern_europe'})
    assert [c['entry'] for c in collisions] == ['Roma']
    assert collisions[0]['collides_with'] == 'Romania'
    assert collisions[0]['list'] == 'locations.would_live_here'


def test_substring_collisions_ignores_a_whole_word_entry(monkeypatch):
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('Spain',))
    assert location_review.substring_collisions({'Spain': 'southern_europe'}) == []


def test_classifier_keeps_country_names_proper():
    """`_coerce` must not fold a name. Regions and languages ARE folded — they are vocabularies."""
    coerced = location._coerce({
        'countries': ['Spain', '  United  Kingdom '], 'regions': ['Southern_Europe', 'WESTERN_EUROPE'],
        'likely_working_language': 'Spanish',
    })
    assert coerced['countries'] == ['Spain', 'United Kingdom']
    assert coerced['regions'] == ['southern_europe', 'western_europe']
    assert coerced['likely_working_language'] == 'spanish'


async def test_review_skips_rather_than_rebuilds_a_corrupt_file(monkeypatch):
    """Deliberately the OPPOSITE of location_cache.yaml, which rebuilds.

    A cache is recomputable; this is a record of the user's decisions, and rebuilding it would
    silently discard every entry they accepted or rejected.
    """
    location_review.LOCATION_RECOMMENDATIONS_PATH.write_text('[not, a, mapping]')
    _seen({'Serbia': 'eastern_europe'})
    assert await location_review.review_location_lists({}) == ''
    assert location_review.LOCATION_RECOMMENDATIONS_PATH.read_text() == '[not, a, mapping]'


def test_location_recommendations_alert_reaches_assess_run_health():
    """An unrecognised `ui_alerts` kind is silently dropped by assess_run_health's if/elif chain."""
    alerts = agent.assess_run_health({
        'ui_alerts': [{'kind': 'location_recommendations', 'query': '', 'region': '',
                       'detail': '1 country list recommendation(s): Serbia -> exclude'}],
    })
    assert any('COUNTRY LISTS' in alert and 'Serbia' in alert for alert in alerts)


def test_countries_seen_is_recorded_on_a_cache_hit_too():
    """In steady state almost every location is a cache hit, so an llm-only accumulator sees nothing."""
    location.reset_countries_seen()
    location._record_countries({'countries': ['Romania'], 'regions': ['eastern_europe']})
    assert location.countries_seen_this_run() == {'Romania': 'eastern_europe'}
    location.reset_countries_seen()
    assert location.countries_seen_this_run() == {}


# ---------------------------------------------------------------------------
# residency scope — where the posting says the holder must LIVE, which is not the anchor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('location,relocation,expected', [
    # Real strings from run_dir/saved_jobs-2026Sep04..08.
    ('Romania (Remote within country)', 'Romania', 'country_only'),
    ('Poland (Remote within country)', 'Poland', 'country_only'),
    ('Germany (Remote)', 'Germany (must be based in Germany; fully remote)', 'country_only'),
    ('Kelowna, BC, Canada (Remote within country — open only to candidates residing in AB, BC)',
     '', 'country_only'),
    ('Austria (Remote)', 'Germany (work is fully remote from anywhere within Germany)', 'country_only'),
    ('Portugal (Remote — anywhere in the EU/Europe)', 'European Union', 'area_wide'),
    ('France (Remote — Work from Anywhere)', '', 'area_wide'),
    ('Netherlands (Remote; can be based in any EMEA Red Hat country)', '', 'area_wide'),
    ('Germany (Remote within Europe, UTC+0 to UTC+2)',
     'Europe (UTC+0 to UTC+2) — candidate must be based in European time zones', 'area_wide'),
    ('Bulgaria (Remote)', 'European Union region (work from EU and work permit required)', 'area_wide'),
    ('Dublin, County Dublin, Ireland (Remote)', 'Based in Europe or the Americas', 'area_wide'),
    # Silent — the case the whole EU default exists for.
    ('Munich, Bavaria, Germany (Remote)', 'Munich, Germany', ''),
    ('Poland (Remote)', 'Poland', ''),
    ('Gothenburg, Västra Götaland County, Sweden (Remote)', '', ''),
])
def test_derive_residency_scope_over_real_postings(location, relocation, expected):
    """Table-driven over strings taken verbatim from the saved corpus.

    The two that matter most are the last group: a posting naming a country and saying nothing
    about residence is NOT a residency requirement, and must derive as unspecified.
    """
    extract = _make_extract(
        location=location, relocation=relocation, description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == expected


@pytest.mark.parametrize('location,expected', [
    # The pair the whole design turns on: same eight words, different last one.
    ('Germany (Remote) — must be based in Germany', 'country_only'),
    ('Germany (Remote) — must be based in Europe', 'area_wide'),
    ('Germany (Remote, must be located in the EU)', 'area_wide'),
    # A determiner and an adjective may sit before the area name. Both patterns share one
    # definition of that filler; when they drifted apart this string fell through BOTH — the
    # country branch stopped claiming it and no area branch picked it up.
    ('Germany (Remote within our EMEA region)', 'area_wide'),
    # ...but "any" alone is not an area offer. Offices are specific places.
    ('Spain (Remote — based in any of our offices)', ''),
    # Only one filler word, deliberately: country_only is the conservative reading of a mixed
    # statement, and widening this makes "Germany or Europe" read as area-wide.
    ('Germany (Remote) — must be based in Germany or Europe', 'country_only'),
])
def test_derive_residency_scope_distinguishes_a_country_object_from_an_area_object(location, expected):
    """Every country-pinning pattern requires an object that is NOT an area name.

    That is the entire trick, and the two directions are equally load-bearing: a wrong `area_wide`
    silently disables the gate, a wrong `country_only` makes the EU default a no-op.
    """
    extract = _make_extract(location=location, relocation='', description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(extract) == expected


@pytest.mark.parametrize('location', [
    'Germany (Remote within the Schengen area)',
    'Spain (Remote — anywhere in LATAM)',
    'Germany (Remote, any APAC country)',
    'Germany (Remote anywhere in MENA)',
    'Germany (Remote across Europe)',
])
def test_open_ended_areas_are_area_wide(location):
    """An area too large to enumerate is what `area_wide` is for."""
    extract = _make_extract(location=location, relocation='', description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(extract) == 'area_wide'


@pytest.mark.parametrize('location', [
    'Germany (Hybrid) — candidates in the EEA or US welcome',
    'Poland (Remote) — EMEA/EU',
    'Spain (Remote) — open to any E.U. member country',
])
def test_every_europe_branch_shares_one_definition_of_europe(location):
    """`_EUROPE_WORDS` feeds every Europe branch, so each accepts every European area name.

    The branches used to carry their own lowercase subsets (`eu|europe` in one, `eu|eea|emea|european`
    in another), so 'EEA or US' and 'EMEA/EU' fell through where 'EU or US' and 'EU/Europe' did not.
    """
    extract = _make_extract(location=location, relocation='', description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(extract) == 'area_wide'


def test_us_alone_is_not_an_area():
    """US/USA stay out of `_AREA_WORDS`: alone they name one country, not an area to live in."""
    extract = _make_extract(location='United States (Remote) — must be based in the US', relocation='',
                            description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(extract) == 'country_only'


@pytest.mark.parametrize('location', [
    'Germany (Remote within DACH)',
    'Netherlands (Remote across Benelux)',
    'Sweden (Remote anywhere in the Nordics)',
    'Poland (Remote within CEE)',
])
def test_small_closed_blocs_are_judged_on_the_anchor(location):
    """DACH is NOT an area word, and leaving it out is a decision rather than an oversight.

    `area_wide` passes unconditionally, mirroring `broad_area`, whose justification is that a
    posting offering a whole multi-country area "is not limited to the countries it happens to
    name". That reasoning holds for the EU or Schengen and collapses for a three-country acronym:
    "remote within DACH" IS limited to Germany, Austria and Switzerland. Listing it as an area
    would turn a role you must live in one of three specific countries for into an automatic pass
    — strictly worse than judging the anchor, which is what happens now.
    """
    extract = _make_extract(location=location, relocation='', description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(extract) == 'country_only'


@pytest.mark.parametrize('location,expected', [
    # The adjectival form has no preposition for the other branches to hang on. 8 real postings
    # in the corpus say "Europe-based".
    ('Warsaw, Poland (Remote) — must be Europe-based', 'area_wide'),
    ('Poland (Remote) — EU-based candidates only', 'area_wide'),
    ('Spain (Remote) — EMEA-based', 'area_wide'),
    # ...and there is deliberately NO generic "<word>-based means country_only" counterpart.
    # Measured over the corpus, that rule would fire on 'Milestone-based', 'remote based', 'role
    # based' and 'team based' -- more false positives than true ones, in the RESTRICTIVE direction
    # where a mistake costs a job rather than a warning.
    ('Germany (Remote) — Milestone-based contract', ''),
    ('Germany (Remote) — remote based role', ''),
])
def test_adjectival_area_form(location, expected):
    extract = _make_extract(location=location, relocation='', description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(extract) == expected


def test_area_words_are_matched_case_insensitively_unlike_place_names():
    """The two regex families read different things, and the split is deliberate.

    `_AREA_WORDS` reads PROSE, where "remote within europe" is written every way imaginable, so it
    folds case. The place-name lists hold PROPER NAMES, where folding case is what made 'malaga' an
    entry that could never match 'Málaga' — and where 'Nice' the city is not 'nice' the adjective.
    """
    for text in ('Germany (Remote within EUROPE)', 'Germany (remote within europe)',
                 'Germany (Remote Within Europe)'):
        extract = _make_extract(location=text, relocation='', description=_NEUTRAL_DESCRIPTION)
        assert agent.derive_residency_scope(extract) == 'area_wide', text
    # ...while a place name is not folded.
    assert location.location_token_matches('Nice', 'Nice, France')
    assert not location.location_token_matches('Nice', 'NICE, FRANCE')


def test_derive_residency_scope_country_only_beats_area_wide_on_one_posting():
    """'Romania (Remote) — role is remote anywhere in the EU' with an explicit in-country clause.

    Both phrasings appear on real postings at once. country_only is tested first and wins, because
    a wrong `area_wide` silently switches the gate off while a wrong `country_only` surfaces as a
    rejection with a stated reason.
    """
    extract = _make_extract(
        location='Romania (Remote) — role is remote anywhere in the EU',
        relocation='Romania (must be based in Romania)', description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == 'country_only'


def test_residency_scope_extractor_country_only_wins_over_a_silent_fallback():
    extract = _make_extract(
        location='Poland (Remote)', residency_scope='country_only', description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == 'country_only'


def test_residency_scope_fallback_country_only_wins_over_an_extractor_area_wide():
    """The union direction: a restrictive finding from EITHER source sticks.

    The opposite of the `is_agency` rule, deliberately: there the extractor's negative judgement
    wins, because being wrong only loses a warning. Here being wrong disables a gate.
    """
    extract = _make_extract(
        location='Poland (Remote within country)', residency_scope='area_wide',
        description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == 'country_only'


def test_residency_scope_extractor_area_wide_wins_over_a_silent_fallback():
    """Between the two permissive values the extractor wins — it read the body, the regex is a backstop."""
    extract = _make_extract(
        location='Poland (Remote)', residency_scope='area_wide', description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == 'area_wide'


def test_residency_scope_ignores_a_value_outside_the_vocabulary():
    extract = _make_extract(
        location='Poland (Remote)', residency_scope='whatever', description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == ''


async def test_hybrid_in_an_eu_country_does_not_get_the_eu_remote_spare():
    """An office pins you to the office — the Munich/Berlin case the gate was built for."""
    extract = _make_extract(
        location='Warsaw, Poland (Hybrid — 2 days a week in the office)', workplace_type='hybrid',
        description=_NEUTRAL_DESCRIPTION,
    )
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'Poland' in reason


async def test_onsite_in_an_eu_country_does_not_get_the_eu_remote_spare():
    extract = _make_extract(
        location='Warsaw, Poland (On-site)', workplace_type='onsite', description=_NEUTRAL_DESCRIPTION,
    )
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'Poland' in reason


async def test_would_not_live_here_applies_only_when_residence_is_pinned(monkeypatch):
    """`would_not_live_here` says where you will not LIVE — so it bites when the job pins you there.

    This is rules 2 and 3 together. "Must live in Poland" is pinned, so the list rejects it. A
    German (or Polish) employer who accepts any EU location does not pin you, so you would live in
    Spain and the posting is kept — having Poland on the list does not change that. Hybrid and
    on-site are always pinned, by the office.
    """
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ('Poland',))
    monkeypatch.setattr(preferences, 'would_live_here', lambda: ('Spain',))

    pinned = _make_extract(location='Poland (Remote within country)', workplace_type='remote',
                           description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(pinned) == 'country_only'
    assert await agent.location_rejection_reason(pinned) == (
        'Poland — residency pinned to the anchor country'
    )

    office = _make_extract(location='Warsaw, Poland (Hybrid)', workplace_type='hybrid',
                           description=_NEUTRAL_DESCRIPTION)
    assert await agent.location_rejection_reason(office) == 'Poland', 'an office always pins you'

    open_eu = _make_extract(location='Poland (Remote)', workplace_type='remote',
                            description=_NEUTRAL_DESCRIPTION)
    assert agent.derive_residency_scope(open_eu) == ''
    assert await agent.location_rejection_reason(open_eu) == '', 'silent EU anchor: live in Spain'


async def test_relocation_rule_is_unaffected_by_the_eu_remote_spare():
    """The most important test in this change: the location half loosened, the relocation half did not.

    This is what still rejects the two GlobalLogic Romania postings — both carried
    `Relocation required: Romania`, and before the `roma` fix neither rule ever ran.
    """
    extract = _make_extract(
        location='Romania (Remote)', relocation='Romania', workplace_type='remote',
        description=_NEUTRAL_DESCRIPTION,
    )
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason == 'relocation required to an excluded region: Romania'
    assert agent._hard_rule_category(reason) == 'hard_ruled_relocation'


async def test_globallogic_romania_regression_end_to_end():
    """job_posting-4453702157 (rated 4 and notified) and job_posting-4452991539 (rated 3).

    Both were Romanian, both said 'Remote within country', both named Romania as required
    residence — and both sailed through, because 'roma' on the acceptable list exempted them at
    tier 1 before any rule ran.
    """
    extract = _make_extract(
        title='Senior ML/AI Data Scientist', company='GlobalLogic',
        location='Romania (Remote within country)', relocation='Romania',
        workplace_type='remote', posting_language='english',
        description='Greenfield GenAI platform, OpenAI LLMs in production, Azure.',
    )
    assert agent.derive_workplace_type(extract) == 'remote'
    assert agent.derive_residency_scope(extract) == 'country_only'
    assert not agent.commute_location_is_acceptable(extract['location']), "'roma' must not match 'romania'"
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'Romania' in reason


async def test_area_wide_keeps_a_posting_anchored_outside_the_eu():
    """`area_wide` is unconditional, mirroring `broad_area`.

    Pinned by name because it is the one value that can switch the gate off for a non-EU anchor —
    which is exactly why the precedence table never lets the weak source assert it.
    """
    extract = _make_extract(
        location='United Kingdom (Remote — Work from Anywhere)', workplace_type='remote',
        description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == 'area_wide'
    assert await agent.location_rejection_reason(extract) == ''


async def test_location_gate_makes_no_classifier_call_when_unconfigured(
    neutral_preferences, stub_location_classifier
):
    """The EU branch reaches classify_location directly, so it needs its own short-circuit."""
    extract = _make_extract(location='Germany (Remote)', workplace_type='remote',
                            description=_NEUTRAL_DESCRIPTION)
    assert await agent.location_rejection_reason(extract) == ''
    assert stub_location_classifier == []


# ---------------------------------------------------------------------------
# three lists, the commute axis, and the guess
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('section,key,replacement', [
    ('locations', 'exclude', 'would_not_live_here'),
    ('locations', 'reject_regions', 'would_not_live_in_regions is gone'),
    ('hybrid', 'acceptable_locations', 'would_commute_here'),
])
def test_legacy_preference_key_raises_rather_than_being_ignored(
    section, key, replacement, monkeypatch, tmp_path, real_load_preferences
):
    """A renamed key must RAISE. `_deep_merge` drops what it does not recognise, so a stale name
    leaves the list EMPTY and silently switches the rule off — a gate that rejects nothing, with no
    error anywhere. This is the single most dangerous thing about the rename."""
    path = tmp_path / 'preferences.yaml'
    path.write_text(yaml.safe_dump({section: {key: ['Anything']}}))
    monkeypatch.setattr(preferences, 'PREFERENCES_PATH', path)
    monkeypatch.setattr(preferences, '_cache', None)
    monkeypatch.setattr(preferences, 'load_preferences', real_load_preferences)
    with pytest.raises(ValueError) as caught:
        real_load_preferences(force_reload=True)
    assert key in str(caught.value) and replacement.split()[0] in str(caught.value)


def test_would_live_here_does_not_uncap_a_commute_job(monkeypatch):
    """The defect the split exists for: one list answered both questions, so `Spain` on it (a place
    you would LIVE) left an on-site role in any Spanish town uncapped and able to score 5."""
    monkeypatch.setattr(preferences, 'would_live_here', lambda: ('Spain',))
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('Barcelona',))
    ourense = _make_extract(location='Ourense, Galicia, Spain (On-site)', workplace_type='onsite')
    assert agent.apply_rating_caps(ourense, 5)[0] == 3
    barcelona = _make_extract(location='Barcelona, Catalonia, Spain (Hybrid)', workplace_type='hybrid')
    assert agent.apply_rating_caps(barcelona, 5) == (5, '')


async def test_gate_reads_the_union_of_live_and_commute(monkeypatch):
    """Commuting somewhere implies being there, so a commute-only city still exempts the gate."""
    monkeypatch.setattr(preferences, 'would_live_here', lambda: ())
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ('Spain',))
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('Barcelona',))
    assert await agent.rejected_location('Barcelona, Catalonia, Spain (Hybrid)') == ''
    assert await agent.rejected_location('Ourense, Galicia, Spain (On-site)') == 'Spain'


async def test_a_guess_never_rejects_and_is_queued_once(monkeypatch):
    """A guess must never cost a job: the user corrects it, and they cannot correct one they were
    never shown. The posting passes, the country is queued, and a second sighting adds nothing."""
    monkeypatch.setattr(preferences, 'would_live_here', lambda: ('Spain',))
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ('Germany',))
    monkeypatch.setattr(preferences, 'not_yet_bucketed', lambda: ())
    agent.reset_location_guesses()
    assert await agent.rejected_location('Warsaw, Poland (Hybrid)') == ''
    assert await agent.rejected_location('Cracow, Poland (On-site)') == ''
    assert list(agent._location_guesses) == ['Poland']


async def test_not_yet_bucketed_passes_and_is_not_re_guessed(monkeypatch):
    monkeypatch.setattr(preferences, 'would_live_here', lambda: ('Spain',))
    monkeypatch.setattr(preferences, 'would_not_live_here', lambda: ('Germany',))
    monkeypatch.setattr(preferences, 'not_yet_bucketed', lambda: ('Poland',))
    agent.reset_location_guesses()
    assert await agent.rejected_location('Warsaw, Poland (Hybrid)') == ''
    assert agent._location_guesses == {}, 'already bucketed — never asked about again'


def test_not_yet_bucketed_posting_carries_a_warning(monkeypatch):
    monkeypatch.setattr(preferences, 'not_yet_bucketed', lambda: ('Poland',))
    extract = _make_extract(location='Warsaw, Poland (Remote)', workplace_type='remote')
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert any('not yet bucketed' in w and 'Poland' in w for w in warnings)


async def test_guess_resolution_fails_open(monkeypatch):
    """Tool server down: no guess, nothing appended, nothing reported. Every posting already passed."""
    async def _down(*a, **k):
        raise RuntimeError('unhandled errors in a TaskGroup (1 sub-exception)')
    monkeypatch.setattr(agent, 'chat_openrouter', _down)
    agent.reset_location_guesses()
    agent.queue_location_guess('Freedonia')
    assert await agent.resolve_location_guesses({}) == []


async def test_guess_vocabulary_is_closed(monkeypatch):
    """Anti-fabrication: the answer is persisted into the user's config, so it must be one of two
    values or it is dropped."""
    async def _chat(prompt, **kwargs):
        return '{"guess": "maybe", "reason": "unsure"}', 0.0
    monkeypatch.setattr(agent, 'chat_openrouter', _chat)
    agent.reset_location_guesses()
    agent.queue_location_guess('Freedonia')
    assert await agent.resolve_location_guesses({}) == []


@pytest.mark.parametrize('location', [
    'Berlin (On location)', 'Berlin (On-location)', 'Berlin — office-based',
    'Berlin — work from our offices', 'Berlin — presence in the office required',
])
def test_workplace_type_detects_the_office_phrasings(location):
    """All of these returned '' before, which earned them the remote benefit of the doubt AND no
    commute cap — the exact phrasings the user named."""
    assert agent.derive_workplace_type({'location': location, 'description': ''}) in {'onsite', 'hybrid'}


# ---------------------------------------------------------------------------
# location token matching
#
# The tier-1 exempt list and the tier-2 deny list are plain string lists the user writes by hand,
# matched against a posting's location. They used to be matched with `token in haystack`, and
# because tier 1 exempts OUTRIGHT that made one entry able to switch the whole gate off for a
# country it never named.
# ---------------------------------------------------------------------------

def test_place_tokens_are_matched_case_sensitively():
    """The lists hold PROPER names, so 'Nice' the city is not 'nice' the adjective."""
    assert location.location_token_matches('Spain', 'Barcelona, Catalonia, Spain')
    assert not location.location_token_matches('spain', 'Barcelona, Catalonia, Spain')
    assert location.location_token_matches('Nice', 'Nice, France (Remote)')
    assert not location.location_token_matches('Nice', 'a nice remote role')


def test_accented_place_name_matches_itself_and_folding_is_not_a_match():
    """`malaga` was a dead entry: LinkedIn writes `Málaga` and matching is accent-exact.

    The fix is to write the entry correctly, NOT to fold accents at match time — folding would
    also fold the boundaries this file spent a whole diagnosis entry getting right.
    """
    assert location.location_token_matches('Málaga', 'Málaga, Andalusia, Spain (Remote)')
    assert not location.location_token_matches('Málaga', 'malaga, spain')
    assert location.location_token_matches('València', 'València, Spain')


def test_roma_does_not_match_romania_in_any_case_or_accent():
    """'Roma' (Rome) on the exempt list must not exempt Romanian postings, in any case or accent.

    Tier 1 short-circuits before the deny list and the classifier, so a Romanian role was never
    gated at all: job_posting-4453702157 was rated 4 and notified while Bulgaria, Poland, Czechia
    and Lithuania rejected correctly in the same run.

    THIS is the test that fails if the boundary class is left as `[a-z0-9]` while the lowercasing
    is removed. That class only blocks lowercase ASCII neighbours, so it was correct only in
    combination with a caller that lowercased first: matching proper names directly, 'ROMA' reaches
    into 'ROMANIA' and 'Roma' into 'Romaña'. The two changes are coupled and nothing else catches it.
    """
    assert not location.location_token_matches('Roma', 'Romania (Remote within country)')
    assert not location.location_token_matches('ROMA', 'ROMANIA (REMOTE)')
    assert not location.location_token_matches('Roma', 'Romaña, Spain')
    assert not location.location_token_matches('Roma', 'Parma, Emilia-Romagna, Italy')
    # ...while still matching the city it was added for, in either spelling.
    assert location.location_token_matches('Roma', 'Roma, Lazio, Italy')
    assert location.location_token_matches('Rome', 'Rome, Italy (Remote)')


def test_location_token_matcher_handles_punctuation():
    r"""'U.S.' is a real entry, and  cannot match it — the boundary after '.' needs a word char.

    A matcher written as r'' + token + r'' returns False for BOTH strings below, silently
    switching off an entry the user wrote down. That is why the matcher uses letter/digit
    lookarounds instead.
    """
    assert location.location_token_matches('U.S.', 'Remote, U.S. only')
    assert location.location_token_matches('U.S.', 'U.S.')
    assert location.location_token_matches('USA', 'USA (Remote)')


def test_location_token_matcher_rejects_substring_collisions():
    assert not location.location_token_matches('Nice', 'Venice, Veneto, Italy')
    assert location.location_token_matches('Nice', 'Nice, France (Remote)')
    # 'Milan' does not match 'Milano' — the classifier's `place_names` bridges the two.
    assert not location.location_token_matches('Milan', 'Milano, Lombardy, Italy')
    assert location.location_token_matches('Milan', 'Milan, Italy')
    # Multi-word entries and hyphen-adjacent hits still work.
    assert location.location_token_matches('British Columbia', 'Vancouver, British Columbia')
    assert location.location_token_matches('Porto', 'Portugal / Porto-based (Remote)')


def test_location_token_matcher_ignores_empty_tokens():
    assert not location.location_token_matches('', 'anywhere')
    assert not location.location_token_matches('   ', 'anywhere')


def test_eu_member_states_covers_the_countries_the_default_turns_on():
    assert len(location.EU_MEMBER_STATES) == 27
    for member in ('Poland', 'Romania', 'Germany', 'Bulgaria', 'Ireland', 'Sweden'):
        assert location.is_eu_member(member), member
    # The four the "bare remote anchor is assumed EU-wide" default deliberately excludes.
    for outsider in ('United Kingdom', 'Norway', 'Switzerland', 'Serbia'):
        assert not location.is_eu_member(outsider), outsider


def test_eu_membership_tolerates_classifier_spelling_variants():
    """`_coerce` preserves the classifier's case and spelling, so `is_eu_member` folds both.

    The classifier may say 'Czechia' or 'Czech Republic', and a pre-2026-09-09 cache entry served
    on the stale path still says 'germany'. All of them must answer the same.
    """
    assert location.is_eu_member('czech republic')
    assert location.is_eu_member('Czechia')
    assert location.is_eu_member('The Netherlands')
    assert location.is_eu_member('germany')
    assert location.is_eu_member('GERMANY')


def test_eu_constants_are_written_as_proper_names():
    """The constants hold the real names; folding lives in `is_eu_member`, not in the data.

    A folded constant is invisible until someone displays it, and it cannot be unfolded. This
    fails if anyone "simplifies" the sets back to lowercase to match a comparison — the comparison
    is what should move. `_EU_LOOKUP` is derived, so the two can never disagree.
    """
    for name in location.EU_MEMBER_STATES | location._EU_ALIASES:
        assert name != name.casefold(), name
    assert frozenset(
        name.casefold() for name in location.EU_MEMBER_STATES | location._EU_ALIASES
    ) == location._EU_LOOKUP


# ---------------------------------------------------------------------------
# location gate: three tiers (exempt list / deny list / cached classifier)
#
# Purely geographic. These tests exist as much to pin what the rule must NOT look at as what it
# does: it reads no language field, because a "non-English" conjunct would exclude a
# French-language remote role in Canada. See test_location_gate_ignores_likely_working_language.
# ---------------------------------------------------------------------------

async def test_rejected_location_flags_a_country_in_a_rejected_region():
    assert await agent.rejected_location('Germany (Remote)') == 'Germany'


async def test_rejected_location_spares_an_acceptable_region():
    assert await agent.rejected_location('Barcelona, Spain (Remote)') == ''


async def test_rejected_location_exempt_list_beats_the_classifier(stub_location_classifier):
    """Tier 1 short-circuits: an exempt location never reaches a paid call."""
    assert await agent.rejected_location('Testville, Germany') == ''
    assert stub_location_classifier == [], 'exempt locations must not cost a classifier call'


async def test_rejected_location_deny_list_beats_the_classifier(stub_location_classifier):
    assert await agent.rejected_location('Blockedland (Remote)') == 'Blockedland'


async def test_rejected_location_spares_a_location_naming_no_country(stub_location_classifier):
    """'European Union' / 'Remote (EMEA)' name no country, so no policy can reject them."""
    assert await agent.rejected_location('European Union') == ''
    assert await agent.rejected_location('Remote (EMEA)') == ''


async def test_rejected_location_spares_a_multi_country_location_with_one_acceptable():
    """Rejection requires EVERY named country to be rejected — one good option is enough."""
    assert await agent.rejected_location('Remote — anywhere in Spain or the Netherlands') == ''


async def test_rejected_location_rejects_an_english_speaking_country_in_a_rejected_region():
    """The gate is geographic, so being English-speaking does not rescue Ireland or the UK.

    This is the knob the user has to set deliberately: `western_europe` catches them along with
    Germany and the Netherlands. Wanting them back is what the exempt list is for — see the test
    below — rather than a language special case, which is the conflation this whole rule avoids.
    """
    assert await agent.rejected_location('Dublin, Ireland (Remote)') == 'Ireland'


async def test_exempt_list_rescues_a_country_in_a_rejected_region(monkeypatch):
    """Tier 1 is the documented way to keep one country out of a rejected region."""
    monkeypatch.setattr(preferences, 'would_commute_here', lambda: ('Testville', 'Ireland'))
    assert await agent.rejected_location('Dublin, Ireland (Remote)') == ''


async def test_rejected_location_spares_a_broad_area_that_also_names_a_country():
    """'European Union (Remote, UK and EU)' is not a UK-only role because the UK is the one country
    it names. Found by the live backtest, which flipped a real 5/5 posting of exactly this shape."""
    assert await agent.rejected_location('European Union (Remote, UK and EU)') == ''


async def test_rejected_location_still_rejects_a_country_phrased_expansively():
    """The other half of that rule, and the one that keeps it honest: a single anchored country
    described in expansive terms is still that country. Otherwise 'Berlin, Germany (Remote across
    Europe)' — the exact Finom posting this whole gate exists for — would spare itself."""
    assert await agent.rejected_location('Berlin, Germany (Remote across Europe)') == 'Germany'


async def test_rejected_location_rejects_when_every_named_country_is_rejected():
    assert await agent.rejected_location('Germany or the Netherlands') == 'Germany'


async def test_rejected_location_is_inert_without_preferences(neutral_preferences, stub_location_classifier):
    """Unconfigured means no gate AND, deliberately, not one classifier call."""
    assert await agent.rejected_location('Germany (Remote)') == ''
    assert stub_location_classifier == []


async def test_classifier_fails_open_when_the_tool_server_is_down(monkeypatch, tmp_path):
    """Fail-open applies to what the CLASSIFIER would have inferred, not to the user's own rules.

    A place literally named on `would_not_live_here` rejects whether or not a tool server answers —
    matching the text against a list the user wrote needs nobody's help. What an outage costs is
    the inference: no countries resolved, so no guess, no exonym matching, and anything the lists
    do not literally name simply passes.
    """
    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'c.yaml')
    monkeypatch.setattr(location, '_cache', {})
    async def _down(*a, **k):
        raise RuntimeError('unhandled errors in a TaskGroup (1 sub-exception)')
    monkeypatch.setattr(location, 'chat_openrouter', _down)
    monkeypatch.setattr(agent, 'classify_location', location.classify_location)

    assert await agent.rejected_location('Germany (Remote)') == 'Germany', 'your own list still applies'
    assert await agent.rejected_location('Freedonia (Remote)') == '', 'nothing inferred, so nothing rejected'


# ---------------------------------------------------------------------------
# location gate: the hard rules it feeds
# ---------------------------------------------------------------------------

async def test_apply_hard_rules_rejects_a_location_in_a_rejected_region():
    """A remote posting that pins residence to the anchor country is judged on the anchor."""
    extract = _make_extract(location='Germany (Remote within country)', workplace_type='remote')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason == (
        'located in an excluded region: Germany '
        '— residency pinned to the anchor country'
    )
    assert agent._hard_rule_category(reason) == 'hard_ruled_location'


async def test_apply_hard_rules_keeps_a_silent_remote_posting_in_an_eu_country():
    """'Germany (Remote)', saying nothing about residence, is now KEPT.

    This is the deliberate reversal of the 2026-09-04 gate for remote roles: a bare location is an
    anchor, not a residency requirement, and an EU anchor implies work rights the user has, so the
    holder can live in another member state. See
    test_september_2026_regression_is_superseded_for_silent_remote_postings for the full reasoning
    and for everything that still rejects.
    """
    extract = _make_extract(
        location='Germany (Remote)', workplace_type='remote', description=_NEUTRAL_DESCRIPTION,
    )
    assert agent.derive_residency_scope(extract) == '', 'the posting says nothing about residence'
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


async def test_apply_hard_rules_still_rejects_a_silent_remote_posting_outside_the_eu():
    """The counterpart. A UK/Serbian/Norwegian/Swiss anchor implies no EU work rights."""
    for loc in (
        'United Kingdom (Remote)', 'Belgrade, Serbia (Remote)',
        'Oslo, Norway (Remote)', 'Zurich, Switzerland (Remote)',
    ):
        extract = _make_extract(
            location=loc, workplace_type='remote', description=_NEUTRAL_DESCRIPTION,
        )
        assert agent.derive_residency_scope(extract) == '', loc
        reason = await agent.apply_hard_rules(_make_candidate(), extract)
        assert reason is not None and 'excluded region' in reason, loc


async def test_apply_hard_rules_rejects_a_stated_relocation_to_a_rejected_region():
    extract = _make_extract(location='Testville (Remote)', relocation='Germany')
    reason = await agent.apply_hard_rules(_make_candidate(), extract)
    assert reason == 'relocation required to an excluded region: Germany'
    assert agent._hard_rule_category(reason) == 'hard_ruled_relocation'


async def test_apply_hard_rules_ignores_a_non_specific_relocation():
    """Finom's posting said `relocation: European Union` — that names no country and must pass."""
    extract = _make_extract(location='Testville (Remote)', relocation='European Union')
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


async def test_apply_hard_rules_allows_relocation_to_an_acceptable_location():
    extract = _make_extract(location='Spain (Remote)', relocation='Barcelona, Spain')
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


def test_hard_rule_category_order_is_stable():
    """Ordered substring dispatch: 'relocation required...' also contains the word 'location'."""
    assert agent._hard_rule_category('relocation required to an excluded region: Germany') == 'hard_ruled_relocation'
    assert agent._hard_rule_category('located in an excluded region: Germany') == 'hard_ruled_location'
    assert agent._hard_rule_category('requires unsupported language: german') == 'hard_ruled_language'
    assert agent._hard_rule_category('requires advanced degree: phd') == 'hard_ruled_education'
    assert agent._hard_rule_category('blacklisted company: X (y)') == 'hard_ruled_blacklisted'


# ---------------------------------------------------------------------------
# the four language/location facts stay independent
# ---------------------------------------------------------------------------

async def test_location_gate_ignores_likely_working_language():
    """A Spanish-speaking location is acceptable; the gate is geographic, not linguistic.

    This is the pin for the design error that nearly shipped: "non-English AND not on the
    acceptable list" gets Spain and Germany right for the wrong reason, and gets Quebec wrong.
    The classifier really does imply Spanish here — the gate must not consult it.
    """
    extract = _make_extract(location='Spain (Remote)')
    # `agent.classify_location`, which is the stub. Calling `location.classify_location` here made
    # this test pass only on a machine whose real cache happened to hold 'spain (remote)' — on a
    # clean checkout it would have made a live, paid call, or failed open and asserted on that.
    assert (await agent.classify_location('Spain (Remote)'))['likely_working_language'] == 'spanish'
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None


async def test_likely_working_language_alone_never_rejects_or_caps():
    """French-speaking Quebec: warned about, never gated. The Valtech mechanism, restated."""
    extract = _make_extract(location='Montreal, Canada (Remote)')
    extract['likely_working_language'] = await agent.derive_likely_working_language(extract)
    assert extract['likely_working_language'] == 'french'
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None
    assert agent.apply_rating_caps(extract, 5) == (5, '')


async def test_september_2026_regression_is_superseded_for_silent_remote_postings():
    """The seven DE/NL postings rated 4-5 and notified on 2026-09-02..03, revisited.

    The 2026-09-04 gate rejected these on the ANCHOR. That is now deliberately reversed for
    `remote` postings: "Berlin, Germany (Remote across Europe)" says the holder may live anywhere
    in Europe, so it is kept — the user's rule is that residence, not the anchor, is what the gate
    may judge, and that a posting saying nothing is assumed to mean remote-from-the-EU.

    What the 2026-09-04 entry was actually built to stop has NOT been reversed, and this test pins
    the parts that still hold, because the reversal is easy to over-read:
      - the same posting as HYBRID still rejects (an office pins you to the office — the
        Munich/Berlin case that started it);
      - the same posting pinned to Germany still rejects;
      - the language cap and the hybrid cap are untouched.
    """
    extract = _make_extract(
        title='Senior AI Engineer', company='Finom', location='Berlin, Germany (Remote across Europe)',
        description='Agentic AI, RAG, tool calling. Remote across Europe.',
        language_requirement='english', posting_language='english',
    )
    assert agent.foreign_posting_language(extract) == '', 'the posting is in English — cap must not fire'
    assert agent.derive_workplace_type(extract) == 'remote', 'remote — the hybrid cap must not fire'
    assert agent.derive_residency_scope(extract) == 'area_wide'
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None, 'kept: residence is open'

    # ...but the office case, which is what the gate was built for, is unchanged.
    onsite = dict(extract, workplace_type='hybrid', description='Hybrid — 2 days a week in the Berlin office.')
    reason = await agent.apply_hard_rules(_make_candidate(), onsite)
    assert reason is not None and 'Germany' in reason
    assert agent._hard_rule_category(reason) == 'hard_ruled_location'

    # ...and so is a remote posting that pins residence to Germany.
    pinned = dict(extract, location='Berlin, Germany (Remote within country)')
    reason = await agent.apply_hard_rules(_make_candidate(), pinned)
    assert reason is not None and 'Germany' in reason
    assert agent._hard_rule_category(reason) == 'hard_ruled_location'


# ---------------------------------------------------------------------------
# derive_likely_working_language  (cosmetic: no gate reads it)
#
# The location is the ONLY source. The extractor no longer supplies a claim, so there is no
# precedence rule left to test -- which is the point: a fact about a place cannot be decided
# differently for two jobs in the same place.
# ---------------------------------------------------------------------------

async def test_derive_likely_working_language_comes_from_the_location():
    assert await agent.derive_likely_working_language(
        _make_extract(location='Berlin, Germany (Remote)')
    ) == 'german'


async def test_derive_likely_working_language_ignores_a_stale_field_on_the_extract():
    """team.blue's Berlin posting once came back 'english' while Finom's came back 'german'.

    Nothing populates this field before resolution any more, but if anything ever did, the
    classifier still decides -- two Berlin jobs cannot disagree.
    """
    assert await agent.derive_likely_working_language(
        _make_extract(location='Berlin, Germany (Remote)', likely_working_language='english')
    ) == 'german'


async def test_derive_likely_working_language_empty_for_an_unnamed_location():
    assert await agent.derive_likely_working_language(
        _make_extract(location='Remote (Anywhere)')
    ) == ''


# ---------------------------------------------------------------------------
# location cache
# ---------------------------------------------------------------------------

def test_location_cache_key_normalizes_whitespace_and_case():
    from agentic_job_search import location
    assert location.cache_key('  Berlin,   GERMANY \n') == location.cache_key('berlin, germany')


def test_location_cache_survives_a_missing_file(monkeypatch, tmp_path):
    from agentic_job_search import location
    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'nope.yaml')
    monkeypatch.setattr(location, '_cache', None)
    assert location._load_cache() == {}


def test_location_cache_rebuilds_a_corrupt_file(monkeypatch, tmp_path):
    """A corrupt cache costs money to rebuild, never a crashed run."""
    from agentic_job_search import location
    path = tmp_path / 'location_cache.yaml'
    path.write_text('{{{ not yaml at all', encoding='utf-8')
    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', path)
    monkeypatch.setattr(location, '_cache', None)
    assert location._load_cache() == {}


def test_location_classifier_defaults_broad_area_to_false():
    """A model that omits the field must not accidentally wave every job through."""
    from agentic_job_search import location
    assert location._coerce({'countries': ['Germany'], 'regions': ['western_europe']})['broad_area'] is False


def test_location_classifier_coerces_an_unknown_region_rather_than_passing_it_through():
    """An out-of-vocabulary region must read as 'I could not tell', not as 'acceptable'."""
    from agentic_job_search import location
    coerced = location._coerce(
        {'countries': ['Germany'], 'regions': ['middle_earth'], 'likely_working_language': 'german'})
    assert coerced['regions'] == ['unknown']


async def test_location_classification_is_cached(monkeypatch, tmp_path):
    """The cache is what makes an LLM call inside a hard rule deterministic — two jobs in the
    same city cannot get different answers, within a run or across runs."""
    from agentic_job_search import location
    calls = []

    async def fake_chat(prompt, **kwargs):
        calls.append(prompt)
        return ('{"countries": ["germany"], "regions": ["western_europe"], '
                '"likely_working_language": "german"}'), 0.0

    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'location_cache.yaml')
    monkeypatch.setattr(location, '_cache', {})
    monkeypatch.setattr(location, 'chat_openrouter', fake_chat)

    first = await location.classify_location('Berlin, Germany (Remote)')
    second = await location.classify_location('  berlin,   germany (remote)  ')
    assert first['regions'] == second['regions'] == ['western_europe']
    assert len(calls) == 1, 'a repeated location must not be re-asked'
    assert second['source'] == 'cache'


# ---------------------------------------------------------------------------
# output-token ceiling and JSON parse failures
#
# Fixtures here are the REAL bytes from run_dir/logs/run-2026-09-0*.log. A fixture cleaner than
# reality is worse than no fixture: the tidy `{"rating": 5}` these were first written against is
# exactly why a truncation was diagnosed as a ```json fence for a week.
# ---------------------------------------------------------------------------

# Verbatim from run-2026-09-02_102421.log:933 — note `"reasoning":Exceptional`, the opening quote
# of the value simply missing. No fence-stripping or repair fixes this, and none should try.
_ARCHER_MALFORMED = (
    '```json\n{\n  "rating": 4,\n  "company": "Archer Recruitment (pharma client undisclosed)",\n'
    '  "title": "Senior Agentic AI Engineer / Architect",\n  "reasoning":Exceptional domain match'
)

# Verbatim head from run-2026-09-03_115601.log — valid JSON that simply stops (finish_reason=length).
_JOBGETHER_TRUNCATED = (
    '```json\n{\n  "rating": 5,\n  "company": "Jobgether (unnamed partner company)",\n'
    '  "title": "Agentic AI Architect - Anthropic",\n  "reasoning": "Near-perfect match to the'
)


def test_extract_json_object_parses_a_fenced_code_block():
    """Pins the NON-bug. raw_decode scans from the first '{' and ignores everything around it, so
    a ```json fence has always parsed. Recorded so nobody re-diagnoses a truncation as a fence."""
    parsed = triage.extract_json_object('```json\n{"rating": 5, "company": "X"}\n```')
    assert parsed == {'rating': 5, 'company': 'X'}


def test_extract_json_object_error_reports_head_and_tail():
    """The head is where the fence is; the fault is always in the tail."""
    with pytest.raises(ValueError) as excinfo:
        triage.extract_json_object(_JOBGETHER_TRUNCATED)
    message = str(excinfo.value)
    assert 'Head:' in message and 'Tail:' in message
    assert str(len(_JOBGETHER_TRUNCATED)) in message
    assert 'Near-perfect match to the' in message, 'the tail is the diagnostic part'


def test_extract_json_object_error_exposes_a_malformed_value():
    with pytest.raises(ValueError) as excinfo:
        triage.extract_json_object(_ARCHER_MALFORMED)
    assert 'Exceptional domain match' in str(excinfo.value)


async def test_chat_openrouter_sends_no_max_tokens_by_default(monkeypatch):
    """No ceiling is sent at all: the MCP tool declares it optional and applies no clamp, and the
    former 3000 default was inherited by all eight call sites rather than chosen by any."""
    seen = {}

    async def fake_call(url, tool, args, timeout_seconds=None):
        seen.update(args)
        return json.dumps({'ok': True, 'content': '{"ok": 1}', 'finish_reason': 'stop', 'cost_usd': 0.001})

    monkeypatch.setattr(triage, 'call_mcp_tool', fake_call)
    content, _cost = await triage.chat_openrouter('hi')
    assert content == '{"ok": 1}'
    assert 'max_tokens' not in seen


async def test_chat_openrouter_raises_a_named_error_on_a_length_finish(monkeypatch):
    """The real payload shape from run-2026-09-02_102421.log:304 — a company-match question that
    spent its entire budget on reasoning and returned null."""
    async def fake_call(url, tool, args, timeout_seconds=None):
        return json.dumps({
            'ok': True, 'content': None, 'tool_calls': None, 'finish_reason': 'length',
            'model': 'z-ai/glm-5.2',
            'usage': {'prompt_tokens': 233, 'completion_tokens': 3000, 'total_tokens': 3233},
        })

    monkeypatch.setattr(triage, 'call_mcp_tool', fake_call)
    with pytest.raises(triage.TruncatedResponseError) as excinfo:
        await triage.chat_openrouter('hi')
    assert 'finish_reason=length' in str(excinfo.value)
    assert '3000' in str(excinfo.value)


async def test_chat_openrouter_raises_on_truncated_non_empty_content(monkeypatch):
    """The case that used to reach the JSON parser disguised as a parse failure."""
    async def fake_call(url, tool, args, timeout_seconds=None):
        return json.dumps({
            'ok': True, 'content': _JOBGETHER_TRUNCATED, 'finish_reason': 'length',
            'model': 'z-ai/glm-5.3-flash', 'usage': {'completion_tokens': 3000},
        })

    monkeypatch.setattr(triage, 'call_mcp_tool', fake_call)
    with pytest.raises(triage.TruncatedResponseError):
        await triage.chat_openrouter('hi')


async def test_rate_with_openrouter_retries_once_on_truncation(monkeypatch):
    """Losing a rating call is not recoverable: the job is already in processed_jobs/ from Stage
    1b, so an eval_error means it returns already_processed forever. A posting the model had
    scored 5 was lost that way on 2026-09-03."""
    attempts = []

    async def fake_chat(prompt, system='', model='', max_tokens=None):
        attempts.append(max_tokens)
        if len(attempts) == 1:
            raise triage.TruncatedResponseError('truncated', max_tokens=None)
        return '{"rating": 5, "reasoning": "great"}', 0.002

    monkeypatch.setattr(triage, 'chat_openrouter', fake_chat)
    result, _ = await triage.rate_with_openrouter('sys', 'user')
    assert result['rating'] == 5
    assert len(attempts) == 2, 'exactly one retry'
    assert attempts[0] is None and attempts[1] == triage._TRUNCATION_RETRY_MAX_TOKENS


# ---------------------------------------------------------------------------
# company_blacklist_reason  (user-authored company blacklist)
# ---------------------------------------------------------------------------

def _no_llm(monkeypatch):
    """Make any confirmation call an error, so a test can assert none was attempted."""
    async def boom_openrouter(prompt):
        raise AssertionError('blacklist confirmation should not have called OpenRouter')

    async def boom_sdk(**kwargs):
        raise AssertionError('blacklist confirmation should not have called the Anthropic SDK')
        yield  # pragma: no cover — make it an async generator

    monkeypatch.setattr(tools, '_blacklist_confirms_openrouter', boom_openrouter)
    monkeypatch.setattr(tools, 'sdk_query', boom_sdk)


def _confirm_llm(monkeypatch, same_organization: bool):
    """Patch both confirmation providers to answer the same way."""
    async def fake_openrouter(prompt):
        return {'same_organization': same_organization}

    monkeypatch.setattr(tools, '_blacklist_confirms_openrouter', fake_openrouter)
    _make_sdk_mock(monkeypatch, {'same_organization': same_organization})


async def test_blacklist_exact_match_confirmed_returns_reason(monkeypatch):
    _confirm_llm(monkeypatch, True)
    assert await tools.company_blacklist_reason('Blocked Corp') == 'test entry'


async def test_blacklist_name_collision_does_not_reject(monkeypatch):
    """Same name, different organization — the whole point of the confirmation call."""
    _confirm_llm(monkeypatch, False)
    assert await tools.company_blacklist_reason('Blocked Corp') is None


async def test_blacklist_match_is_case_sensitive(monkeypatch):
    _no_llm(monkeypatch)
    assert await tools.company_blacklist_reason('blocked corp') is None
    assert await tools.company_blacklist_reason('BLOCKED CORP') is None


async def test_blacklist_non_match_makes_no_llm_call(monkeypatch):
    _no_llm(monkeypatch)
    assert await tools.company_blacklist_reason('Acme') is None
    assert await tools.company_blacklist_reason('') is None


async def test_blacklist_empty_preference_makes_no_llm_call(monkeypatch):
    _no_llm(monkeypatch)
    merged = preferences._deep_merge(preferences.load_preferences(), {'companies': {'blacklist': []}})
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)
    assert await tools.company_blacklist_reason('Blocked Corp') is None


async def test_blacklist_confirmation_failure_still_rejects(monkeypatch):
    """An outage must not quietly let a blacklisted company back through."""
    async def fail_openrouter(prompt):
        raise RuntimeError('tool server down')

    async def fail_sdk(**kwargs):
        raise RuntimeError('anthropic unavailable')
        yield  # pragma: no cover — make it an async generator

    monkeypatch.setattr(tools, '_blacklist_confirms_openrouter', fail_openrouter)
    monkeypatch.setattr(tools, 'sdk_query', fail_sdk)
    assert await tools.company_blacklist_reason('Blocked Corp') == 'test entry'


async def test_blacklist_no_structured_output_still_rejects(monkeypatch):
    async def fail_openrouter(prompt):
        raise RuntimeError('tool server down')

    async def empty_sdk(**kwargs):
        return
        yield  # pragma: no cover — make it an async generator

    monkeypatch.setattr(tools, '_blacklist_confirms_openrouter', fail_openrouter)
    monkeypatch.setattr(tools, 'sdk_query', empty_sdk)
    assert await tools.company_blacklist_reason('Blocked Corp') == 'test entry'


async def test_blacklist_description_not_budgeted_without_a_name_hit(monkeypatch, caplog):
    """Budgeting it up front logged a truncation for every job (20 on 2026-09-24), all unread."""
    _no_llm(monkeypatch)
    long_description = 'x' * (config.BLACKLIST_CONTEXT_DESCRIPTION_MAX_CHARS * 2)
    with caplog.at_level(logging.WARNING):
        reason = await agent.apply_hard_rules(
            _make_candidate(), _make_extract(company='Acme', description=long_description))
    assert reason is None
    assert 'blacklist-confirmation' not in caplog.text


async def test_blacklist_name_hit_sends_budgeted_description(monkeypatch, caplog):
    prompts = []

    async def capture_openrouter(prompt):
        prompts.append(prompt)
        return {'same_organization': True}

    monkeypatch.setattr(tools, '_blacklist_confirms_openrouter', capture_openrouter)
    head, tail = 'POSTING HEAD', 'POSTING TAIL'
    description = head + 'x' * (config.BLACKLIST_CONTEXT_DESCRIPTION_MAX_CHARS * 2) + tail
    with caplog.at_level(logging.WARNING):
        reason = await tools.company_blacklist_reason(
            'Blocked Corp', context='Company: Blocked Corp', description=description)
    assert reason == 'test entry'
    assert len(prompts) == 1
    assert 'Company: Blocked Corp' in prompts[0]
    assert head in prompts[0]
    assert tail not in prompts[0]
    assert 'blacklist-confirmation' in caplog.text


async def test_blacklist_entry_without_reason_still_rejects(monkeypatch):
    _confirm_llm(monkeypatch, True)
    merged = preferences._deep_merge(
        preferences.load_preferences(), {'companies': {'blacklist': ['Bare Name Corp']}}
    )
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)
    assert await tools.company_blacklist_reason('Bare Name Corp') == 'blacklisted'


# --- expiry -----------------------------------------------------------------

async def test_expired_blacklist_entry_does_not_reject(monkeypatch):
    _no_llm(monkeypatch)
    assert await tools.company_blacklist_reason('Lapsed Corp') is None


def test_expired_blacklist_entry_is_reported():
    names = [name for name, _added in preferences.expired_blacklist_entries()]
    assert 'Lapsed Corp' in names
    assert 'Blocked Corp' not in names


def test_active_blacklist_entry_excluded_from_expired():
    assert 'Blocked Corp' in [name for name, _reason in preferences.blacklisted_companies()]
    assert 'Lapsed Corp' not in [name for name, _reason in preferences.blacklisted_companies()]


def test_blacklist_entry_with_unparseable_date_stays_active(monkeypatch, caplog):
    merged = preferences._deep_merge(
        preferences.load_preferences(),
        {'companies': {'blacklist': [{'name': 'Undated Corp', 'reason': 'x', 'added': 'not-a-date'}]}},
    )
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)
    with caplog.at_level(logging.WARNING):
        active = preferences.blacklisted_companies()
    assert ('Undated Corp', 'x') in active
    assert 'no valid `added` date' in caplog.text


def test_blacklist_expiry_warning_names_the_entry(caplog):
    with caplog.at_level(logging.WARNING):
        preferences.blacklisted_companies()
    assert 'Lapsed Corp' in caplog.text
    assert 'EXPIRED' in caplog.text


async def test_blacklist_yaml_date_object_is_accepted(monkeypatch):
    """PyYAML parses an unquoted YYYY-MM-DD into a date, not a str."""
    _confirm_llm(monkeypatch, True)
    merged = preferences._deep_merge(
        preferences.load_preferences(),
        {'companies': {'blacklist': [
            {'name': 'Dated Corp', 'reason': 'y', 'added': date.today()},
        ]}},
    )
    monkeypatch.setattr(preferences, 'load_preferences', lambda force_reload=False: merged)
    assert await tools.company_blacklist_reason('Dated Corp') == 'y'


# --- integration with apply_hard_rules --------------------------------------

async def test_apply_hard_rules_rejects_blacklisted_poster(monkeypatch):
    _confirm_llm(monkeypatch, True)
    reason = await agent.apply_hard_rules(
        _make_candidate(company='Blocked Corp'), _make_extract(company='Blocked Corp')
    )
    assert 'blacklisted company' in reason
    assert 'Blocked Corp' in reason


async def test_apply_hard_rules_rejects_blacklisted_end_client(monkeypatch):
    """A recruiter reposting a blacklisted company's role must not get around the blacklist."""
    _confirm_llm(monkeypatch, True)
    extract = _make_extract(company='Some Recruiters', is_agency=True, end_client='Blocked Corp')
    reason = await agent.apply_hard_rules(_make_candidate(company='Some Recruiters'), extract)
    assert 'blacklisted company' in reason
    assert 'Blocked Corp' in reason


async def test_apply_hard_rules_passes_non_blacklisted_company(monkeypatch):
    _no_llm(monkeypatch)
    assert await agent.apply_hard_rules(_make_candidate(), _make_extract()) is None


async def test_apply_hard_rules_ignores_expired_blacklist_entry(monkeypatch):
    _no_llm(monkeypatch)
    assert await agent.apply_hard_rules(
        _make_candidate(company='Lapsed Corp'), _make_extract(company='Lapsed Corp')
    ) is None


def test_hard_rule_category_buckets_blacklist():
    assert agent._hard_rule_category('blacklisted company: Blocked Corp (test entry)') == 'hard_ruled_blacklisted'


def test_example_preferences_blacklist_is_empty():
    """The tracked example must not name a real company the user blocked."""
    loaded = yaml.safe_load(preferences.EXAMPLE_PREFERENCES_PATH.read_text(encoding='utf-8'))
    assert loaded['companies']['blacklist'] == []


# ---------------------------------------------------------------------------
# UI contract guard and discovery-health alerts
#
# The whole point of these: on 2026-08-18 the scraper correctly reported "the EU region filter did
# not apply" on every single query, and it reached nothing but the run log. A silent degradation
# looked exactly like "no new jobs today" for four days running.
# ---------------------------------------------------------------------------

def _sound_report(**overrides):
    """A structurally healthy search page, as SCRAPER_UI_CONTRACT_JS would report it."""
    report = {
        'url': 'https://www.linkedin.com/jobs/search-results/?keywords=X&geoId=91000000&f_TPR=r604800',
        'search_box': True,
        'results_container': True,
        'job_cards': 25,
        'chip_row': 6,
        'chips': ['Jobs', 'Past week', 'Senior', 'Gen AI', 'Employment type', 'Company'],
        'location_pin': 'European Union',
        'pagination': True,
        'result_count_text': '99+ results',
        'body_sample': 'staff ai engineer jobs in the european union',
    }
    report.update(overrides)
    return report


def test_sound_page_produces_no_contract_violations():
    assert tools.evaluate_ui_contract(_sound_report()) == []


def test_missing_results_container_is_a_contract_violation():
    """A renamed container is the change that silently returns zero listings."""
    violations = tools.evaluate_ui_contract(_sound_report(results_container=False, job_cards=0))
    assert any('results_container' in v for v in violations)


def test_zero_cards_while_the_page_shows_results_is_a_violation():
    """The exact silent-failure shape: harvest returns nothing, run reports 'no new jobs today'."""
    violations = tools.evaluate_ui_contract(_sound_report(job_cards=0, result_count_text='99+ results'))
    assert any('0 job cards' in v for v in violations)


def test_block_signature_is_detected_before_any_contract_judgement():
    """A challenge and a markup change need OPPOSITE responses, so they must not be conflated."""
    assert tools.detect_block_signature(_sound_report(body_sample='please verify your identity')) == 'verify your identity'
    assert tools.detect_block_signature(_sound_report(body_sample='we noticed unusual activity')) == 'unusual activity'
    assert tools.detect_block_signature(_sound_report()) == ''


def test_filters_confirmed_from_the_url_the_chips_produced():
    """We never navigate to geoId/f_TPR, but reading them back proves the clicks landed."""
    region = {'name': 'European Union', 'linkedin_location': 'European Union', 'geo_id': '91000000'}
    assert tools.check_filters_applied(_sound_report(), region) == []


def test_location_filter_that_did_not_apply_is_caught():
    """The Aug-18 failure: the search ran, looked fine, and returned the wrong region entirely."""
    region = {'name': 'European Union', 'linkedin_location': 'European Union', 'geo_id': '91000000'}
    report = _sound_report(
        url='https://www.linkedin.com/jobs/search-results/?keywords=X&f_TPR=r604800',
        location_pin='Greater Vancouver Metropolitan Area',
    )
    problems = tools.check_filters_applied(report, region)
    assert any('geoId=91000000' in p for p in problems)


def test_missing_date_and_experience_chips_are_caught():
    region = {'name': 'Canada', 'linkedin_location': 'Canada'}
    report = _sound_report(
        url='https://www.linkedin.com/jobs/search-results/?keywords=X',
        location_pin='Canada',
        chips=['Jobs', 'Gen AI'],
    )
    problems = tools.check_filters_applied(report, region)
    assert any('date posted' in p for p in problems)
    assert any('experience level' in p for p in problems)


_GEO_REGIONS = [
    {'name': 'Canada', 'linkedin_location': 'Canada', 'geo_id': '101174742'},
    {'name': 'European Union', 'linkedin_location': 'European Union', 'geo_id': '91000000'},
]
_CANADA_OK_URL = 'https://www.linkedin.com/jobs/search-results/?keywords=X&geoId=101174742&f_TPR=r604800'
_CANADA_NO_GEO_URL = 'https://www.linkedin.com/jobs/search-results/?keywords=X&f_TPR=r604800'


def _report_search(query, region, url, pin):
    report = _sound_report(url=url, location_pin=pin)
    return asyncio.run(tools.report_search.handler({'query': query, 'region': region, 'report': report}))


def _filter_health_alerts():
    funnel = {'ui_alerts': list(tools._ui_alerts), 'queries_generated': 1,
              'listings_seen': 25, 'listings_distinct': 25, 'region_overlap': {}, 'check_status': {}}
    return [a for a in agent.assess_run_health(funnel) if 'FILTERS DID NOT APPLY' in a]


@pytest.fixture
def geo_regions(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)   # check_fingerprint_drift writes here
    monkeypatch.setattr(preferences, 'search_regions', lambda: _GEO_REGIONS)
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])


def test_filter_alert_fixed_before_harvest_does_not_need_attention(geo_regions):
    """The 2026-09-10 run: the model re-navigated to shed a self-injected f_SAL, which dropped geoId
    while the sticky chip still showed "Canada". The check caught it, the model re-clicked the chip,
    the next check passed and only then did it harvest. That is the check working, not an alert."""
    first = _report_search('Principal Data Scientist', 'Canada', _CANADA_NO_GEO_URL, 'Canada')
    assert 'FILTERS NOT APPLIED' in first['content'][0]['text']
    second = _report_search('Principal Data Scientist', 'Canada', _CANADA_OK_URL, 'Canada')
    assert second['content'][0]['text'].startswith('ok')

    assert _filter_health_alerts() == []
    assert [bool(r['filter_problems']) for r in tools._search_reports] == [True, False], \
        'audit §1b must still show the fail -> ok trail'


def test_filter_alert_never_followed_by_a_passing_check_still_alerts(geo_regions):
    _report_search('Principal Data Scientist', 'Canada', _CANADA_NO_GEO_URL, 'Canada')
    assert len(_filter_health_alerts()) == 1


def test_filter_alert_is_not_resolved_by_a_pass_on_another_search(geo_regions):
    """A sound EU page says nothing about whether the Canada search was ever filtered."""
    _report_search('Principal Data Scientist', 'Canada', _CANADA_NO_GEO_URL, 'Canada')
    _report_search('Principal Data Scientist', 'European Union',
                   'https://www.linkedin.com/jobs/search-results/?keywords=X&geoId=91000000&f_TPR=r604800',
                   'European Union')
    _report_search('Staff AI Engineer', 'Canada', _CANADA_OK_URL, 'Canada')
    assert len(_filter_health_alerts()) == 1


def test_region_overlap_detects_a_dead_location_axis(monkeypatch):
    """Two regions returning the same ids means the filter did nothing — not 'query exhausted'."""
    monkeypatch.setattr(tools, '_search_ids', {
        ('Staff AI Engineer', 'Canada'): {'1', '2', '3'},
        ('Staff AI Engineer', 'European Union'): {'1', '2', '3'},
        ('Principal DS', 'Canada'): {'7', '8'},
        ('Principal DS', 'European Union'): {'9', '10'},
    })
    overlaps = tools.region_overlap_report()
    assert overlaps['Staff AI Engineer'] == 1.0, 'identical id sets = the location filter is dead'
    assert overlaps['Principal DS'] == 0.0, 'genuinely distinct regions must not alarm'


def test_fingerprint_drift_reports_changed_chips(tmp_path, monkeypatch):
    """Catches a restructure the run it happens, even when every required element survives."""
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    first = tools.check_fingerprint_drift(_sound_report())
    assert first == '', 'the first observation is a baseline, not a change'
    again = tools.check_fingerprint_drift(_sound_report())
    assert again == '', 'an unchanged page must not alarm'
    # A STRUCTURAL chip disappearing is a real change and must be reported.
    drift = tools.check_fingerprint_drift(_sound_report(chips=['Jobs', 'Past week']))
    assert 'Experience level' in drift and 'gone' in drift


def test_fingerprint_ignores_linkedins_per_query_topical_chips(tmp_path, monkeypatch):
    """Gen AI / LLM / AWS / AI-ML / Analytics change with every query by design.

    Fingerprinting the whole chip row made drift fire on ordinary query-to-query variation on the
    first live run, and a warning that fires on success trains the reader to ignore it.
    """
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    structural = ['Jobs', 'Past week', 'Senior', 'Employment type', 'Company']
    tools.check_fingerprint_drift(_sound_report(chips=[*structural, 'Gen AI', 'AWS']))
    drift = tools.check_fingerprint_drift(_sound_report(chips=[*structural, 'AI/ML', 'Analytics']))
    assert drift == '', 'topical suggestion chips must not count as a UI change'


def test_the_in_my_network_chip_flapping_is_not_drift(tmp_path, monkeypatch):
    """2026-09-28: 'chips gone' on one query, 'new chips' on the next, same run, nothing changed."""
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    structural = ['Jobs', 'Past week', 'Senior', 'Employment type', 'Company']
    tools.check_fingerprint_drift(_sound_report(chips=[*structural, 'In my network']))
    assert tools.check_fingerprint_drift(_sound_report(chips=structural)) == ''
    assert tools.check_fingerprint_drift(_sound_report(chips=[*structural, 'In my network'])) == ''


def test_a_chip_showing_its_chosen_value_is_the_same_chip(tmp_path, monkeypatch):
    '''2026-10-02 an unfiltered page rewrote the fingerprint to "Date posted"/"Experience level";
    2026-10-05 the filtered page read as two chips new and two gone. Filters are not page shape.'''
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    rest = ['Jobs', 'Employment type', 'Company']
    tools.check_fingerprint_drift(_sound_report(chips=[*rest, 'Past week', 'Senior']))
    assert tools.check_fingerprint_drift(_sound_report(chips=[*rest, 'Date posted', 'Experience level'])) == ''
    assert tools.check_fingerprint_drift(_sound_report(chips=[*rest, 'Past 24 hours', 'Director'])) == ''
    stored = yaml.safe_load((tmp_path / config.UI_FINGERPRINT_FILENAME).read_text(encoding='utf-8'))
    assert 'Date posted' in stored['chips'] and 'Past week' not in stored['chips']


def test_a_fingerprint_stored_under_chosen_values_does_not_alarm(tmp_path, monkeypatch):
    '''The file written before this rule holds "Past week" and "Senior"; the next run must not flag it.'''
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    rest = ['Company', 'Employment type', 'Jobs']
    (tmp_path / config.UI_FINGERPRINT_FILENAME).write_text(
        yaml.safe_dump({'chips': [*rest, 'Past week', 'Senior']}), encoding='utf-8')
    assert tools.check_fingerprint_drift(_sound_report(chips=[*rest, 'Past week', 'Senior'])) == ''


def test_a_fingerprint_written_with_a_since_dropped_chip_does_not_alarm(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    structural = ['Company', 'Employment type', 'Jobs', 'Past week', 'Senior']
    (tmp_path / config.UI_FINGERPRINT_FILENAME).write_text(
        yaml.safe_dump({'chips': [*structural, 'In my network']}), encoding='utf-8')
    assert tools.check_fingerprint_drift(_sound_report(chips=structural)) == ''


# --- Discovery-health alerts -------------------------------------------------


def test_low_yield_alert_fires_on_the_run_that_prompted_all_this():
    """2026-08-18: 1 new job from 101 distinct listings, no notification, no signal."""
    alerts = agent.assess_run_health({
        'listings_distinct': 101, 'check_status': {'new': 1},
        'region_overlap': {}, 'ui_alerts': [],
    })
    assert any('LOW YIELD' in a for a in alerts)


def test_low_yield_alert_stays_quiet_on_a_healthy_run():
    """Aug 14: 26 new from 300. An alert that fires on success trains the reader to ignore it."""
    alerts = agent.assess_run_health({
        'listings_distinct': 300, 'check_status': {'new': 26},
        'region_overlap': {}, 'ui_alerts': [],
    })
    assert alerts == []


def test_region_overlap_alert_names_the_query():
    alerts = agent.assess_run_health({
        'listings_distinct': 101, 'check_status': {'new': 50},
        'region_overlap': {'Staff AI Engineer': 1.0}, 'ui_alerts': [],
    })
    assert any('REGION OVERLAP' in a and 'Staff AI Engineer' in a for a in alerts)


def test_block_and_ui_change_alerts_are_distinguishable():
    """They demand different actions from the user, so they must not read the same."""
    alerts = agent.assess_run_health({
        'listings_distinct': 10, 'check_status': {'new': 10}, 'region_overlap': {},
        'ui_alerts': [
            {'kind': 'blocked', 'query': 'Q', 'region': 'Canada', 'detail': 'captcha'},
            {'kind': 'contract', 'query': 'Q2', 'region': 'EU', 'detail': 'results_container missing'},
        ],
    })
    assert any(a.startswith('BLOCKED') and 'did not retry' in a for a in alerts)
    assert any(a.startswith('UI CHANGED') for a in alerts)


def test_contract_js_dedupes_cards_and_chips():
    """Both are rendered TWICE in the DOM. Verified live on 2026-08-18: a raw count reported 50
    cards and 44 chips on a page holding 25 jobs and 13 chips, so any count-based threshold built
    on the raw numbers would be silently wrong."""
    js = agent.SCRAPER_UI_CONTRACT_JS
    assert 'new Set' in js, 'cards and chips must be deduped before counting'
    assert 'cardIds.size' in js, 'the card count must come from the deduped set'


def test_contract_js_finds_the_search_box_by_placeholder():
    """The search box is a plain <input> carrying a PLACEHOLDER and no aria-label — verified live.
    An aria-label-only selector reported search_box=false on a perfectly healthy page, which would
    have failed the contract on every single query."""
    js = agent.SCRAPER_UI_CONTRACT_JS
    assert 'placeholder' in js
    assert 'describe the job' in js.lower()


def test_contract_js_is_read_only():
    """Same rule as the harvest JS: JavaScript may read the page, never drive it."""
    js = agent.SCRAPER_UI_CONTRACT_JS
    for forbidden in ('.click(', '.submit(', 'dispatchEvent', '.value =', 'location.href ='):
        assert forbidden not in js, f'the contract probe must never {forbidden}'


async def test_a_region_that_was_never_verified_is_reported(monkeypatch):
    """A search the model never verified is silently unfiltered — the exact failure this guards.

    On the first live chip run, report_search was called 5 times across 12 expected searches and
    one query searched only one of its two regions, with turn budget to spare (49 of 260). So it
    was a choice, not starvation, and prompting harder is not the fix: code checks the omission.
    """
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    # Only one of the two configured regions (conftest pins Testland + Test Union) was verified.
    monkeypatch.setattr(tools, '_search_reports', [{'query': 'Alpha, remote', 'region': 'Testland'}])

    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    await _scrape(_FakeScraperClient({'Alpha': healthy}), ['Alpha'])

    unverified = [a for a in tools._ui_alerts if a['kind'] == 'unverified_search']
    assert unverified, 'a region with no report_search must raise an alert'
    assert 'Test Union' in unverified[0]['region']
    assert 'Testland' not in unverified[0]['region'], 'the verified region must not be flagged'


async def test_all_regions_verified_raises_no_coverage_alert(monkeypatch):
    """An alert that fires on success trains the reader to ignore it."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})
    monkeypatch.setattr(tools, '_search_reports', [
        {'query': 'Alpha, remote', 'region': 'Testland'},
        {'query': 'Alpha, remote', 'region': 'Test Union'},
    ])

    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    await _scrape(_FakeScraperClient({'Alpha': healthy}), ['Alpha'])

    assert [a for a in tools._ui_alerts if a['kind'] == 'unverified_search'] == []


def test_report_query_matching_tolerates_the_full_search_text():
    """The model reports what it typed ("Principal Data Scientist, remote"), not the base query.

    Exact matching would make the coverage check silently never fire — the same shape of bug as
    the counters it exists to backstop.
    """
    assert agent._same_query('Principal Data Scientist, remote', 'Principal Data Scientist')
    assert agent._same_query('Alpha', 'Alpha')
    assert not agent._same_query('Beta, remote', 'Alpha')
    assert not agent._same_query('', 'Alpha')


# ---------------------------------------------------------------------------
# @playwright/mcp version pin
# ---------------------------------------------------------------------------

def test_playwright_mcp_version_is_concrete():
    """`@latest` is resolved by npm on EVERY run. npx had cached `^0.0.79` under the key
    `@playwright/mcp@latest`; upstream published 0.0.80, the cached tree stopped satisfying
    `latest`, and npx halted the run at an interactive `Ok to proceed? (y)` -- on the startup path
    cron uses. The prompt was the symptom; the real defect is that the tool surface the scraper
    drives a real logged-in LinkedIn account through could change with no commit."""
    assert re.fullmatch(r'\d+\.\d+\.\d+', config.PLAYWRIGHT_MCP_VERSION), (
        f'PLAYWRIGHT_MCP_VERSION must be a concrete version, got '
        f'{config.PLAYWRIGHT_MCP_VERSION!r}'
    )
    assert f'@playwright/mcp@{config.PLAYWRIGHT_MCP_VERSION}' == config.PLAYWRIGHT_MCP_PACKAGE


def test_playwright_mcp_is_pinned_at_every_launch_site():
    """The static guard, in the shape of test_every_claude_agent_options_site_limits_context: the
    fix was applied at two launch sites, and prose cannot stop a third from being added with
    `@latest`. Every literal naming the package must carry a concrete version."""
    source_dir = Path(agent.__file__).parent
    offenders = []
    for path in sorted(p for p in source_dir.rglob('*.py')
                       if '__pycache__' not in p.parts):
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
            if line.lstrip().startswith('#'):
                continue  # prose about the bug, not a launch argument
            for match in re.finditer(r'@playwright/mcp@([\w.{\-]*)', line):
                spec = match.group(1)
                # '{' is an f-string interpolating a version this file pins as concrete above.
                if spec.startswith('{') or re.fullmatch(r'\d+\.\d+\.\d+', spec):
                    continue
                offenders.append(f'{path.relative_to(source_dir)}:{lineno} @playwright/mcp@{spec or "<empty>"}')
    assert not offenders, (
        'unpinned @playwright/mcp reference(s) -- use config.PLAYWRIGHT_MCP_PACKAGE:\n'
        + '\n'.join(offenders)
    )


def _npx_launch_sites(tree):
    """Yield (lineno, argument nodes) for every npx invocation in a parsed module.

    Two shapes exist in this package:
        ['npx', '--yes', PLAYWRIGHT_MCP_PACKAGE, '--port', ...]        -> a list literal
        {"command": "npx", "args": ["--yes", PLAYWRIGHT_MCP_PACKAGE]}  -> a stdio server dict
    """
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.List) and any(
                isinstance(e, ast.Constant) and e.value == 'npx' for e in node.elts):
            yield node.lineno, node.elts
        elif isinstance(node, ast.Dict):
            pairs = {k.value: v for k, v in zip(node.keys, node.values, strict=True)
                     if isinstance(k, ast.Constant)}
            command, args = pairs.get('command'), pairs.get('args')
            if (isinstance(command, ast.Constant) and command.value == 'npx'
                    and isinstance(args, ast.List)):
                yield node.lineno, args.elts


def test_every_npx_launch_passes_yes_and_the_pin():
    """--yes is what stops npx blocking on a cold cache (new machine, cleared cache, or straight
    after a deliberate bump), and it is only safe BECAUSE the version is exact: it can install
    nothing but the pin. So both must hold at EVERY launch site.

    Checked structurally rather than by regex. The previous version searched for
    `"npx",.*?"--yes", PLAYWRIGHT_MCP_PACKAGE` with re.S, and the lazy `.*?` spans any distance --
    measured, a third launch site with no --yes still passed, because the regex paired its bare
    `"npx",` with the EXISTING site's `--yes` further down the file. A guard that assumes exactly
    the sites present when it was written is the failure this repo already has a scar from
    (agent.py fixed 2026-08-21, the four tools_generic.py sites missed for three months).

    Package-wide for the same reason, not just agent.py.
    """
    import ast

    source_dir = Path(agent.__file__).parent
    offenders = []
    sites = 0

    for path in sorted(p for p in source_dir.rglob('*.py')
                       if '__pycache__' not in p.parts):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for lineno, elements in _npx_launch_sites(tree):
            sites += 1
            has_yes = any(isinstance(e, ast.Constant) and e.value == '--yes' for e in elements)
            has_pin = any(isinstance(e, ast.Name) and e.id == 'PLAYWRIGHT_MCP_PACKAGE'
                          for e in elements)
            if not (has_yes and has_pin):
                missing = [n for n, ok in (('--yes', has_yes),
                                           ('PLAYWRIGHT_MCP_PACKAGE', has_pin)) if not ok]
                offenders.append(f'{path.relative_to(source_dir)}:{lineno} missing {missing}')

    # Without this, a walk that matches nothing compares an empty list and passes forever.
    assert sites >= 2, f'expected to find the known npx launch sites, found {sites}'
    assert not offenders, (
        'npx launch site(s) that can block on a cold cache or drift off the pin:\n'
        + '\n'.join(offenders)
    )


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_check_playwright_mcp_version_reports_newer(monkeypatch):
    monkeypatch.setattr(agent, 'PLAYWRIGHT_MCP_VERSION', '0.0.79')
    monkeypatch.setattr(agent.requests, 'get', lambda *a, **kw: _FakeResponse({'version': '0.0.80'}))
    assert agent.check_playwright_mcp_version() == '0.0.80'


@pytest.mark.parametrize('published', ['0.0.79', '0.0.78', '0.0.9'])
def test_check_playwright_mcp_version_silent_when_not_newer(monkeypatch, published):
    """0.0.9 is the ordering case a string comparison gets wrong: '0.0.9' > '0.0.79' lexically."""
    monkeypatch.setattr(agent, 'PLAYWRIGHT_MCP_VERSION', '0.0.79')
    monkeypatch.setattr(agent.requests, 'get', lambda *a, **kw: _FakeResponse({'version': published}))
    assert agent.check_playwright_mcp_version() is None


@pytest.mark.parametrize('failure', [
    ConnectionError('offline'),
    TimeoutError('registry timed out'),
])
def test_check_playwright_mcp_version_fails_open_on_network_error(monkeypatch, failure):
    """This runs before the run lock and before the browser starts. A registry outage must cost a
    log line, never a run."""
    def boom(*args, **kwargs):
        raise failure

    monkeypatch.setattr(agent.requests, 'get', boom)
    assert agent.check_playwright_mcp_version() is None


@pytest.mark.parametrize('payload', [
    {},                                  # no 'version' key
    {'version': 'latest'},               # not comparable
    ValueError('not json'),              # malformed body
])
def test_check_playwright_mcp_version_fails_open_on_bad_payload(monkeypatch, payload):
    monkeypatch.setattr(agent.requests, 'get', lambda *a, **kw: _FakeResponse(payload))
    assert agent.check_playwright_mcp_version() is None


def test_upgrade_prompt_never_blocks_without_a_tty(monkeypatch):
    """The whole point. cron has nobody to answer, so an available upgrade warns and continues --
    it must not read stdin, which is precisely what `@latest` did via npx."""
    monkeypatch.setattr(agent, 'check_playwright_mcp_version', lambda: '0.0.80')
    monkeypatch.setattr(agent.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(agent, '_startup_ui_alerts', [])

    def no_input(*args, **kwargs):
        raise AssertionError('prompted for input with no TTY attached')

    monkeypatch.setattr('builtins.input', no_input)
    agent.prompt_playwright_mcp_upgrade(interactive=True)

    kinds = [alert['kind'] for alert in agent._startup_ui_alerts]
    assert kinds == ['playwright_mcp_outdated']
    assert '0.0.80' in agent._startup_ui_alerts[0]['detail']


def test_upgrade_prompt_never_prompts_in_non_interactive_mode(monkeypatch):
    """`-n` is the autonomous mode -- cron runs it, and so does a person at a terminal, where
    isatty() is True. Gating on the TTY alone meant a hand-run `-n` stopped dead on a question,
    which is the exact failure this mechanism exists to prevent. `-n` means do not ask."""
    monkeypatch.setattr(agent, 'check_playwright_mcp_version', lambda: '0.0.80')
    monkeypatch.setattr(agent.sys.stdin, 'isatty', lambda: True)   # a real terminal
    monkeypatch.setattr(agent, '_startup_ui_alerts', [])

    def no_input(*args, **kwargs):
        raise AssertionError('prompted in non-interactive mode')

    monkeypatch.setattr('builtins.input', no_input)
    agent.prompt_playwright_mcp_upgrade(interactive=False)

    assert [a['kind'] for a in agent._startup_ui_alerts] == ['playwright_mcp_outdated']


@pytest.mark.parametrize('failure', [EOFError, KeyboardInterrupt])
def test_upgrade_prompt_survives_stdin_ending(monkeypatch, failure):
    """Ctrl-D or Ctrl-C at the question must continue on the pin, not traceback out of startup."""
    monkeypatch.setattr(agent, 'check_playwright_mcp_version', lambda: '0.0.80')
    monkeypatch.setattr(agent.sys.stdin, 'isatty', lambda: True)

    def boom(*args, **kwargs):
        raise failure()

    monkeypatch.setattr('builtins.input', boom)
    agent.prompt_playwright_mcp_upgrade(interactive=True)   # returns normally


def test_upgrade_prompt_exits_when_user_chooses_update(monkeypatch):
    monkeypatch.setattr(agent, 'check_playwright_mcp_version', lambda: '0.0.80')
    monkeypatch.setattr(agent.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda *a, **kw: 'u')
    with pytest.raises(SystemExit) as excinfo:
        agent.prompt_playwright_mcp_upgrade(interactive=True)
    assert excinfo.value.code == 0


@pytest.mark.parametrize('answer', ['', 'c', 'continue'])
def test_upgrade_prompt_continues_on_anything_else(monkeypatch, answer):
    monkeypatch.setattr(agent, 'check_playwright_mcp_version', lambda: '0.0.80')
    monkeypatch.setattr(agent.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda *a, **kw: answer)
    agent.prompt_playwright_mcp_upgrade(interactive=True)  # returns normally; runs on the pin


def test_upgrade_prompt_is_silent_when_pin_is_current(monkeypatch):
    """No newer version means no prompt, no alert, and no stdin read even on a TTY."""
    monkeypatch.setattr(agent, 'check_playwright_mcp_version', lambda: None)
    monkeypatch.setattr(agent.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(agent, '_startup_ui_alerts', [])

    def no_input(*args, **kwargs):
        raise AssertionError('prompted when the pin is already current')

    monkeypatch.setattr('builtins.input', no_input)
    agent.prompt_playwright_mcp_upgrade(interactive=True)
    assert agent._startup_ui_alerts == []


def test_playwright_outdated_reaches_run_health():
    """An alert nobody surfaces is this project's most-repeated bug, so pin the path into
    assess_run_health -> run log, audit 1b, and the Telegram NEEDS ATTENTION block."""
    alerts = agent.assess_run_health({'ui_alerts': [{
        'kind': 'playwright_mcp_outdated', 'query': '(all)', 'region': '(all)',
        'detail': '@playwright/mcp 0.0.80 is available; this run uses the pinned 0.0.79'}]})
    assert any('BROWSER TOOLCHAIN OUTDATED' in alert for alert in alerts)
    assert any('0.0.80' in alert for alert in alerts)


@pytest.mark.network
def test_npm_registry_still_reports_a_semver_version():
    """Does the real registry still return what check_playwright_mcp_version() parses?

    Deliberately does the fetch itself instead of calling check_playwright_mcp_version(). That
    function fails open and returns None on ANY error, so an assertion written around it
    (`assert published is None or ...`) is a tautology: green whether the registry answers, is
    unreachable, drops the `version` key, or changes shape entirely. Measured -- it passed with the
    URL pointed at an unroutable host. An assert that can never fail is the same defect as a skip
    that can never pass. The wrapper's own logic is covered by the mocked tests above.

    Marked `network`, not `live`: `live` means the LLM MCP tool servers on :8002/:8006 (see
    pyproject.toml), which this needs no part of. Self-skips on a connection error, following the
    convention _require_llm_server established -- offline is the one outcome that must not fail.
    """
    try:
        response = requests.get(config.PLAYWRIGHT_MCP_REGISTRY_URL, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as ex:
        pytest.skip(f'npm registry unreachable: {type(ex).__name__}: {ex}')

    published = response.json()['version']    # KeyError here is a real failure, not a skip
    assert re.fullmatch(r'\d+\.\d+\.\d+', published), (
        f'the registry returned {published!r}, which is not plain semver -- '
        'check_playwright_mcp_version() cannot compare it'
    )
    assert agent._parse_semver(published) is not None, (
        f'_parse_semver rejects what the registry actually publishes ({published!r})'
    )


@pytest.mark.network
def test_pinned_playwright_mcp_version_exists_in_the_registry():
    """Does the version we pin actually EXIST? Nothing else asks.

    Every other guard checks the pin's SHAPE -- concrete semver, no `@latest`, present at each
    launch site with --yes. A typo'd or yanked version satisfies all of them: measured, a pin of
    '0.0.97' passes the entire suite, then every run dies at `npx` with "No matching version
    found". Worse, the typo silences its own alarm -- check_playwright_mcp_version() sees
    0.0.97 > 0.0.80 and reports no upgrade available, so startup says nothing either.

    Kept separate from test_npm_registry_still_reports_a_semver_version deliberately: "the pin does
    not exist" is fixed in config.py, "the latest response is not semver" means npm changed its
    API. Folding them together would report both as the same red.
    """
    url = config.PLAYWRIGHT_MCP_REGISTRY_URL.rsplit('/', 1)[0] + f'/{config.PLAYWRIGHT_MCP_VERSION}'
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as ex:
        pytest.skip(f'npm registry unreachable: {type(ex).__name__}: {ex}')

    if response.status_code == 404:
        pytest.fail(
            f'PLAYWRIGHT_MCP_VERSION = {config.PLAYWRIGHT_MCP_VERSION!r} does not exist in the '
            f'registry (404 at {url}) -- it is a typo, or the release was yanked. Every run will '
            'fail at npx with "No matching version found". Fix the constant in config.py.'
        )
    # Anything else non-200 is a registry problem, NOT a bad pin -- do not conflate them.
    assert response.status_code == 200, (
        f'unexpected {response.status_code} from {url}; this looks like a registry fault rather '
        f'than a bad pin, so PLAYWRIGHT_MCP_VERSION was not verified this run'
    )


# ---------------------------------------------------------------------------
# mcp version guard
# ---------------------------------------------------------------------------

def test_mcp_pinned_below_2():
    """`mcp_session` in triage.py unpacks three values from `streamable_http_client`; mcp 2.0
    yields two. That helper is the ONLY path to the OpenRouter and Ollama tool servers, so an
    unnoticed upgrade takes out triage, the non-Anthropic rating call, the reference summary and
    the OpenRouter extractor at once -- at runtime, mid-run. Measured 2026-08-21: a bare `uv run`
    re-resolved `mcp>=1.29` to 2.0.0 and did exactly that."""
    pyproject = (Path(__file__).parent.parent / 'pyproject.toml').read_text(encoding='utf-8')
    assert '"mcp>=1.29,<2"' in pyproject, (
        'the <2 pin on mcp was removed from pyproject.toml. Before lifting it, make '
        'triage.mcp_session tolerate both the 2- and 3-value yield and verify against a live server.'
    )


def test_installed_mcp_matches_mcp_session_unpack():
    """The pin above constrains resolution; this asserts the environment actually agrees with the
    3-tuple unpack in triage.mcp_session, so a stale or force-installed venv fails here rather than
    on the first tool-server call of a run."""
    from importlib.metadata import version
    major = int(version('mcp').split('.')[0])
    assert major < 2, (
        f'mcp {version("mcp")} is installed, but triage.mcp_session unpacks three values from '
        'streamable_http_client and mcp >= 2.0 yields two. Run `uv sync --frozen`.'
    )


# ---------------------------------------------------------------------------
# Stage 1b OpenRouter scraper: anti-fabrication and account safety
# ---------------------------------------------------------------------------

def _session(browser_reply):
    """A ScrapeSession whose browser returns a canned response."""
    async def browser(name, args):
        return browser_reply
    return scrape_openrouter.ScrapeSession(browser, 'Staff AI Engineer')


_CHAT_502 = json.dumps({'ok': False, 'error': '502 Server Error: Bad Gateway for url: https://openrouter.ai/api/v1/chat/completions'})
_CHAT_402 = json.dumps({'ok': False, 'error': '402 Client Error: Payment Required for url: https://openrouter.ai/api/v1/chat/completions'})
_CHAT_DONE = json.dumps({'ok': True, 'content': 'done', 'tool_calls': None, 'finish_reason': 'stop', 'usage': {}})


def _run_with_chat_replies(monkeypatch, replies):
    """Drive ScrapeSession.run against canned chat replies; returns how many chat calls were made."""
    calls = []

    async def fake_call_mcp_tool(url, tool_name, args, timeout_seconds=None):
        calls.append(tool_name)
        return replies[len(calls) - 1]

    monkeypatch.setattr(scrape_openrouter, 'call_mcp_tool', fake_call_mcp_tool)
    monkeypatch.setattr(scrape_openrouter, 'TRANSIENT_RETRY_DELAY_SECONDS', 0)
    asyncio.run(_session('').run('sys', 'user', []))
    return len(calls)


_CHAT_INTENT = json.dumps({'ok': True, 'content': "I'll start by navigating.", 'tool_calls': None,
                           'finish_reason': 'stop', 'usage': {}})


def _chat_tool_call(name: str, arguments: dict | None = None) -> str:
    call = {'id': 'c1', 'function': {'name': name, 'arguments': json.dumps(arguments or {})}}
    return json.dumps({'ok': True, 'content': None, 'tool_calls': [call], 'usage': {}})


def _nudged_session(monkeypatch, replies, session=None):
    '''Run a session with the nudge on; returns (session, the messages of each chat call).'''
    sent = []

    async def fake_call_mcp_tool(url, tool_name, args, timeout_seconds=None):
        sent.append(json.loads(json.dumps(args['messages'])))
        return replies[len(sent) - 1]

    monkeypatch.setattr(scrape_openrouter, 'call_mcp_tool', fake_call_mcp_tool)
    monkeypatch.setattr(scrape_openrouter, 'SCRAPER_NO_TOOL_NUDGES', config.SCRAPER_NO_TOOL_NUDGES)
    session = session or _session('')
    asyncio.run(session.run('sys', 'user', []))
    return session, sent


def test_a_text_only_reply_before_any_record_is_nudged_not_taken_as_done(monkeypatch):
    '''2026-09-28: "I'll start by navigating..." and no tool call ended a query after 2 iterations.'''
    session, sent = _nudged_session(monkeypatch, [_CHAT_INTENT, _CHAT_DONE, _CHAT_DONE])

    assert len(sent) == config.SCRAPER_NO_TOOL_NUDGES + 1, 'nudged, then a second silence ends the pass'
    assert sent[1][-2] == {'role': 'assistant', 'content': "I'll start by navigating."}
    assert sent[1][-1] == {'role': 'user', 'content': scrape_openrouter.NO_TOOL_NUDGE}
    assert session.nudges == config.SCRAPER_NO_TOOL_NUDGES


def test_no_nudge_once_listings_were_recorded(monkeypatch):
    '''Text and no tool call after a record is the model finishing, which is the normal ending.'''
    session = _session('')
    session._harvest = [{'job_id': '1'}]

    async def fake_record(jobs, query):
        return 'recorded 1'

    monkeypatch.setattr(tools, 'do_record_listings', fake_record)
    session, sent = _nudged_session(
        monkeypatch, [_chat_tool_call('record_listings'), _CHAT_DONE, _CHAT_DONE], session)

    assert len(sent) == 2
    assert session.nudges == 0


def test_no_nudge_after_the_model_reported_a_problem(monkeypatch):
    '''Stopping after report_problem is what the prompt asks for; a nudge would push it to guess.'''
    monkeypatch.setattr(tools, '_ui_alerts', [])
    session, sent = _nudged_session(monkeypatch, [
        _chat_tool_call('report_problem', {'what_happened': 'no chips'}), _CHAT_DONE, _CHAT_DONE])

    assert len(sent) == 2
    assert session.nudges == 0


def test_no_nudge_after_a_block(monkeypatch):
    '''Account safety: a blocked session is never pushed on.'''
    session = _session('')
    session.blocked = True
    session, sent = _nudged_session(monkeypatch, [_CHAT_INTENT, _CHAT_DONE], session)

    assert len(sent) == 1
    assert session.nudges == 0


def test_scraper_retries_a_transient_502_once(monkeypatch):
    """2026-09-10: one 502 discarded a query 71 iterations in, and its second region never ran."""
    assert _run_with_chat_replies(monkeypatch, [_CHAT_502, _CHAT_DONE]) == 2


def test_scraper_gives_up_after_a_second_502_as_a_one_query_failure(monkeypatch):
    with pytest.raises(RuntimeError) as info:
        _run_with_chat_replies(monkeypatch, [_CHAT_502, _CHAT_502])
    assert not isinstance(info.value, triage.ProviderUnavailableError), \
        'a 502 must not abort every remaining query'


def test_scraper_does_not_retry_a_provider_refusal(monkeypatch):
    """A 402 breaks every query; retrying it only burns pacing before the fallback runs."""
    with pytest.raises(triage.ProviderUnavailableError):
        _run_with_chat_replies(monkeypatch, [_CHAT_402, _CHAT_DONE])


# playwright-mcp echoes the evaluated SCRIPT after the result, braces included. Fixtures that omit
# it are unrealistically easy to parse, and did hide a parser bug that made every real harvest
# return nothing while the tests stayed green.
_CODE_ECHO = ("\n### Ran Playwright code\n```js\n"
              "await page.evaluate('() => { return {count: 0, jobs: []}; }');\n```\n")


def _two_jobs():
    return json.dumps({'count': 2, 'jobs': [
        {'id': '111', 'title': 'Staff AI Engineer', 'company': 'Acme', 'location': 'Canada', 'posted': '2 days ago'},
        {'id': '222', 'title': 'Principal ML Engineer', 'company': 'Globex', 'location': 'Canada', 'posted': ''},
    ]})


def test_evaluate_parser_ignores_the_echoed_script_block():
    """Regression: scanning to the LAST '}' runs past the JSON into the echoed source, so every
    real harvest parsed as None while hand-written fixtures passed."""
    obj = scrape_openrouter.parse_evaluate_result('### Result\n' + _two_jobs() + _CODE_ECHO)
    assert obj is not None and obj['count'] == 2
    assert [j['id'] for j in obj['jobs']] == ['111', '222']


def test_evaluate_parser_handles_braces_inside_strings():
    payload = json.dumps({'count': 1, 'jobs': [{'id': '1', 'title': 'Eng {x} "q"'}]})
    obj = scrape_openrouter.parse_evaluate_result('### Result\n' + payload + _CODE_ECHO)
    assert obj['jobs'][0]['title'] == 'Eng {x} "q"'


def test_click_guard_refuses_the_results_list():
    """The hard safety invariant, enforced in code rather than by prompt wording: to the model a
    card and its Dismiss button are the same node, and one stray click destroys a real job."""
    for args in ({'target': 'job-card-component-ref-4455'},
                 {'element': 'the Dismiss button', 'target': 'e12'},
                 {'target': 'div[componentkey="SearchResultsMainContent"] p'}):
        assert scrape_openrouter._guard_call('browser_click', args) is not None, args
    assert scrape_openrouter._guard_call('browser_click', {'target': 'e1202'}) is None, \
        'filter chips sit above the results and must stay clickable'


def test_click_guard_refuses_javascript_that_drives_the_page():
    assert scrape_openrouter._guard_call(
        'browser_evaluate', {'function': '() => document.querySelector("a").click()'}) is not None
    assert scrape_openrouter._guard_call(
        'browser_evaluate', {'function': '() => ({count: 1})'}) is None


@pytest.mark.asyncio
async def test_harvest_error_forbids_invention_and_records_nothing(monkeypatch):
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_ui_alerts', [])
    session = _session('### Result\n{"error": "results container not found"}' + _CODE_ECHO)
    out = await session.dispatch_local('harvest_listings', {})
    assert 'markup has changed' in out and 'invent' in out.lower()
    assert tools._check_status_counts == {} and tools._candidates == []
    assert [a['kind'] for a in tools._ui_alerts] == ['empty_harvest']


@pytest.mark.asyncio
async def test_result_diverted_to_a_file_is_reported_not_reconstructed(monkeypatch):
    """The 2026-08-21 canary: `filename` sent the evaluate result to disk, the model saw only a
    link, and it invented a page report. Code must say it cannot read the page."""
    monkeypatch.setattr(tools, '_search_reports', [])
    session = _session('### Result\n- [Evaluation result](./step3.json)')
    out = await session.dispatch_local('run_ui_contract', {'region': 'Canada'})
    assert 'could not read' in out.lower() and 'guess' in out.lower()
    assert tools._search_reports == [], 'nothing may be recorded from a page code could not read'


@pytest.mark.parametrize('tool_name, tool_args', [
    ('browser_evaluate', {'function': '() => document.title'}),
    ('browser_snapshot', {}),
    ('browser_find', {'text': 'results'}),
    ('browser_console_messages', {'level': 'error'}),
    ('browser_tool_a_later_release_adds', {}),
])
def test_model_supplied_filename_never_reaches_the_browser(monkeypatch, tool_name, tool_args):
    """`filename` diverts a result to disk and leaves the model a link to reason from. The tool
    schemas come from the live server, so a release that adds the argument to another tool (as
    @playwright/mcp did for browser_find) exposes it to the model with no change here."""
    browser_calls = []

    async def browser(name, args):
        browser_calls.append((name, dict(args)))
        return 'ok'

    tool_call = {'id': 'call1', 'function': {
        'name': tool_name, 'arguments': json.dumps({**tool_args, 'filename': 'matches.md'})}}
    replies = [json.dumps({'ok': True, 'content': None, 'tool_calls': [tool_call], 'usage': {}}), _CHAT_DONE]

    async def fake_call_mcp_tool(url, name, args, timeout_seconds=None):
        return replies.pop(0)

    monkeypatch.setattr(scrape_openrouter, 'call_mcp_tool', fake_call_mcp_tool)
    asyncio.run(scrape_openrouter.ScrapeSession(browser, 'Staff AI Engineer').run('sys', 'user', []))
    assert browser_calls == [(tool_name, tool_args)]


@pytest.mark.asyncio
async def test_record_before_harvest_refuses(monkeypatch):
    monkeypatch.setattr(tools, '_check_status_counts', {})
    out = await _session('').dispatch_local('record_listings', {})
    assert 'nothing harvested' in out.lower()
    assert tools._check_status_counts == {}


@pytest.mark.asyncio
async def test_record_listings_ignores_model_supplied_job_data(monkeypatch, tmp_path):
    """The decisive anti-fabrication test: even when the model sends fabricated listings,
    record_listings uses CODE's harvest. The guarantee is structural, not behavioural."""
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path)
    monkeypatch.setattr(tools, '_processed_jobs', set())
    monkeypatch.setattr(tools, '_listing_records', {})
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queue_skipped_counts', {})
    monkeypatch.setattr(tools, '_applied_companies', {})
    session = _session('### Result\n' + _two_jobs() + _CODE_ECHO)
    await session.dispatch_local('harvest_listings', {})
    await session.dispatch_local('record_listings', {
        'jobs': [{'id': '999', 'title': 'Invented', 'company': 'Fabricated Inc'}]})
    assert {jid for _, jid in tools._listing_records} == {'111', '222'}
    assert not any(r['company'] == 'Fabricated Inc' for r in tools._listing_records.values())


@pytest.mark.asyncio
async def test_record_listings_caps_in_code(monkeypatch, tmp_path):
    """SCRAPER_MAX_LISTINGS_PER_SEARCH used to be a prompt suggestion only."""
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path)
    for name, value in (('_processed_jobs', set()), ('_listing_records', {}),
                        ('_check_status_counts', {}), ('_distinct_listing_ids', set()),
                        ('_search_ids', {}), ('_candidates', []), ('_candidates_per_query', {}),
                        ('_queue_skipped_counts', {}), ('_applied_companies', {})):
        monkeypatch.setattr(tools, name, value)
    cap = config.SCRAPER_MAX_LISTINGS_PER_SEARCH
    jobs = [{'id': str(i), 'title': 'Staff AI Engineer', 'company': f'C{i}'} for i in range(cap + 5)]
    out = await tools.do_record_listings(jobs, query='Staff AI Engineer')
    assert f'recorded {cap} (capped from {cap + 5})' in out
    assert len(tools._distinct_listing_ids) == cap


@pytest.mark.asyncio
async def test_record_listings_returns_counts_not_job_data(monkeypatch, tmp_path):
    """The return value re-enters the model's context on every later iteration, and echoing job
    data back would both cost tokens and hand the model something to later 'recall'."""
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path)
    for name, value in (('_processed_jobs', set()), ('_listing_records', {}),
                        ('_check_status_counts', {}), ('_distinct_listing_ids', set()),
                        ('_search_ids', {}), ('_candidates', []), ('_candidates_per_query', {}),
                        ('_queue_skipped_counts', {}), ('_applied_companies', {})):
        monkeypatch.setattr(tools, name, value)
    out = await tools.do_record_listings(json.loads(_two_jobs())['jobs'], query='Staff AI Engineer')
    assert 'Acme' not in out and 'Globex' not in out and 'Principal ML Engineer' not in out
    assert len(out) < 400, 'the summary must stay short — it is re-read every iteration'


def test_browser_tool_defs_drop_disallowed_and_keep_required():
    def _t(name):
        return {'name': name, 'description': 'd', 'inputSchema': {'type': 'object'}}
    names = ['browser_navigate', 'browser_snapshot', 'browser_click', 'browser_wait_for',
             'browser_evaluate', 'browser_close', 'browser_run_code_unsafe']
    defs = scrape_openrouter.browser_tool_defs([_t(n) for n in names])
    exposed = {d['function']['name'] for d in defs}
    assert 'browser_close' not in exposed and 'browser_run_code_unsafe' not in exposed
    for required in config.SCRAPER_REQUIRED_BROWSER_TOOLS:
        assert required.replace('mcp__playwright__', '') in exposed, required


# ---------------------------------------------------------------------------
# API-level error surfacing (_raise_if_result_error / AgentApiError)
# ---------------------------------------------------------------------------

def _errored_result(api_error_status: int | None = 401, **overrides) -> ResultMessage:
    """A ResultMessage shaped exactly like the one that produced 'error result: success'.

    The CLI reports an HTTP-level failure as is_error=True with subtype='success' and an empty
    errors list, which is why the SDK's own fallback text renders the word 'success'.
    """
    fields = {
        'subtype': 'success',
        'duration_ms': 100,
        'duration_api_ms': 100,
        'is_error': True,
        'num_turns': 1,
        'session_id': 'fake-session',
        'api_error_status': api_error_status,
        'errors': None,
    }
    fields.update(overrides)
    return ResultMessage(**fields)


def test_raise_if_result_error_names_the_status_code():
    """The regression: the status must appear, and the bare word 'success' must not stand alone."""
    with pytest.raises(tools.AgentApiError) as excinfo:
        tools._raise_if_result_error(_errored_result(401), 'PDF categorization failed')

    message = str(excinfo.value)
    assert '401' in message
    assert 'PDF categorization failed' in message
    assert excinfo.value.api_error_status == 401
    # The old message was exactly this and said nothing else.
    assert message != 'Claude Code returned an error result: success'


def test_raise_if_result_error_hints_at_login_for_auth_errors():
    with pytest.raises(tools.AgentApiError) as excinfo:
        tools._raise_if_result_error(_errored_result(403), 'ctx')
    assert 'claude /login' in str(excinfo.value)


def test_raise_if_result_error_hints_at_backoff_for_rate_limits():
    with pytest.raises(tools.AgentApiError) as excinfo:
        tools._raise_if_result_error(_errored_result(429), 'ctx')
    assert 'rate limited' in str(excinfo.value)


def test_raise_if_result_error_survives_missing_status():
    """api_error_status is None on older CLIs; the error must still be raised and readable."""
    with pytest.raises(tools.AgentApiError) as excinfo:
        tools._raise_if_result_error(_errored_result(None), 'ctx')
    assert excinfo.value.api_error_status is None
    assert 'API error' in str(excinfo.value)


def test_raise_if_result_error_is_silent_on_success():
    ok = ResultMessage(
        subtype='success',
        duration_ms=100,
        duration_api_ms=100,
        is_error=False,
        num_turns=1,
        session_id='fake-session',
        structured_output={'category': 'saved_jd'},
    )
    assert tools._raise_if_result_error(ok, 'ctx') is None


async def test_categorize_pdf_text_raises_on_api_error(monkeypatch):
    async def fake_sdk_query(**kwargs):
        yield _errored_result(429)

    monkeypatch.setattr(tools, 'sdk_query', fake_sdk_query)
    with pytest.raises(tools.AgentApiError):
        await tools._categorize_pdf_text('text', 'acme.pdf')


async def test_categorize_save_dir_pdfs_aborts_on_api_error(tmp_path, monkeypatch):
    """One API failure must abort, not spawn a doomed call per PDF."""
    for name in ('a.pdf', 'b.pdf', 'c.pdf'):
        (tmp_path / name).write_bytes(b'%PDF fake')

    calls = []

    async def fake_categorize(text, filename):
        calls.append(filename)
        raise tools.AgentApiError('PDF categorization failed: API error 401', api_error_status=401)

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)

    class FakeReader:
        pages: ClassVar[list] = []
        def __init__(self, path): pass

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)

    with pytest.raises(tools.AgentApiError):
        await tools.categorize_save_dir_pdfs(save_dir=tmp_path)

    assert len(calls) == 1, f'expected abort after the first failure, got {calls}'
    assert sorted(p.name for p in tmp_path.glob('*.pdf')) == ['a.pdf', 'b.pdf', 'c.pdf']


async def test_categorize_save_dir_pdfs_continues_past_per_file_error(tmp_path, monkeypatch):
    """A per-file fault is recoverable and must not kill the run."""
    for name in ('bad.pdf', 'good.pdf'):
        (tmp_path / name).write_bytes(b'%PDF fake')

    async def fake_categorize(text, filename):
        if filename == 'bad.pdf':
            raise ValueError('corrupt content')
        return 'saved_jd'

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)

    class FakeReader:
        pages: ClassVar[list] = []
        def __init__(self, path): pass

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)
    await tools.categorize_save_dir_pdfs(save_dir=tmp_path)

    assert (tmp_path / 'cat-saved_jd-good.pdf').exists()
    assert (tmp_path / 'bad.pdf').exists()


async def test_extract_applied_job_metadata_raises_on_api_error(monkeypatch):
    """This path had no try/except at all, so the raw SDK text reached the user."""
    async def fake_sdk_query(**kwargs):
        yield _errored_result(529)

    monkeypatch.setattr(tools, 'sdk_query', fake_sdk_query)
    with pytest.raises(tools.AgentApiError):
        await tools._extract_applied_job_metadata('text', 'acme.pdf')


async def test_company_matches_applied_raises_on_api_error(monkeypatch):
    monkeypatch.setattr(tools, '_applied_companies', {'Shopify': 'shopify_jd.pdf'})
    # Force the Anthropic path: the OpenRouter branch swallows every exception by design, and
    # a name that normalizes to an exact match would short-circuit before any LLM call at all.
    monkeypatch.setattr(tools, 'MODEL_NAME_COMPANY_MATCH', ANTHROPIC_MODEL_NAME_LOW)

    async def fake_sdk_query(**kwargs):
        yield _errored_result(401)

    monkeypatch.setattr(tools, 'sdk_query', fake_sdk_query)
    with pytest.raises(tools.AgentApiError):
        await tools.company_matches_applied('Acme Data Systems')


# ---------------------------------------------------------------------------
# Context-leak guard: every ClaudeAgentOptions site must limit what it loads
# ---------------------------------------------------------------------------

def test_every_claude_agent_options_site_limits_context():
    """setting_sources=None loads ALL sources, including this repo's CLAUDE.md, and
    strict_mcp_config=False pulls in every unrelated global MCP server — roughly 50K tokens
    per call. agent.py was fixed for this in 2026-08; tools_generic.py was missed entirely,
    which is what this static check exists to stop happening to the next site someone adds.
    """
    import ast

    required = {'setting_sources', 'strict_mcp_config', 'skills'}
    source_dir = Path(tools.__file__).parent
    offenders = []
    sites = 0

    for path in sorted(p for p in source_dir.rglob('*.py')
                       if '__pycache__' not in p.parts):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
            if name != 'ClaudeAgentOptions':
                continue
            sites += 1
            passed = {kw.arg for kw in node.keywords if kw.arg}
            if missing := (required - passed):
                offenders.append(f'{path.relative_to(source_dir)}:{node.lineno} missing {sorted(missing)}')

    assert sites >= 12, f'expected to find the known option sites, found {sites}'
    assert not offenders, 'ClaudeAgentOptions sites that leak context:\n' + '\n'.join(offenders)


@pytest.mark.live_agent_claude
async def test_live_categorization_options_do_not_load_repo_context():
    """Measure the leak fix rather than asserting it structurally.

    docs/requirements.md (Cost efficiency) records the original measurement: 45,523 tokens on a bare options object vs 358
    with setting_sources/strict_mcp_config/skills set. The categorization options set cwd to the
    project root, so without those flags this loads the repo's CLAUDE.md plus every global
    MCP server.
    """
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(
        model=ANTHROPIC_MODEL_NAME_LOW,
        tools=[],
        permission_mode='bypassPermissions',
        setting_sources=[],
        strict_mcp_config=True,
        skills=[],
        cwd=str(tools.PROJECT_DIR),
    )

    async with ClaudeSDKClient(options=options) as client:
        usage = await client.get_context_usage()

    memory_tokens = sum(f.get('tokens', 0) for f in (usage.get('memoryFiles') or []))
    mcp_tokens = sum(t.get('tokens', 0) for t in (usage.get('mcpTools') or []))

    assert memory_tokens == 0, f'CLAUDE.md still being loaded: {memory_tokens} tokens'
    assert mcp_tokens == 0, f'global MCP servers still being loaded: {mcp_tokens} tokens'
    assert usage['totalTokens'] < 10_000, (
        f"context still bloated: {usage['totalTokens']} tokens "
        f'(pre-fix this measured ~45K)'
    )


# ---------------------------------------------------------------------------
# Tool schemas: params the description calls optional must actually be optional
# ---------------------------------------------------------------------------

def test_optional_tool_params_are_not_declared_required():
    """The {"name": str} shorthand marks EVERY key required.

    claude_agent_sdk/__init__.py:422 builds `"required": list(properties.keys())` from the
    shorthand, so three tools spent months demanding fields their own descriptions called
    optional — and `date_posted`, which docs/architecture.md says to "omit if not shown", could not be
    omitted. That is direct fabrication pressure (see the Anti-fabrication requirement): a model
    told a field is mandatory and unable to observe it will either refuse the call or invent a
    value. Measured: the live save_job_posting test failed because the model refused to invent a
    job_id. Optionality requires the full JSON Schema form, which the SDK passes through as-is.
    """
    from claude_agent_sdk import SdkMcpTool

    expected_required = {
        'save_job_posting': {'company', 'description', 'rating', 'content'},
        'check_and_record_job': {'site', 'job_id', 'company', 'description'},
        'queue_candidate': {'site', 'job_id', 'url', 'title', 'company', 'snippet'},
    }

    found = {}
    for name in dir(tools):
        obj = getattr(tools, name)
        if isinstance(obj, SdkMcpTool) and obj.name in expected_required:
            schema = obj.input_schema
            assert isinstance(schema, dict) and 'properties' in schema, (
                f'{obj.name} uses the shorthand schema form, which forces every param required'
            )
            found[obj.name] = set(schema.get('required', []))

    assert found == expected_required, f'{found} != {expected_required}'


def test_optional_tool_params_still_declared_as_properties():
    """Optional does not mean absent: the model must still know the params exist."""
    from claude_agent_sdk import SdkMcpTool

    optional_by_tool = {
        'save_job_posting': {'job_id'},
        'check_and_record_job': {'date_posted', 'url', 'content'},
        'queue_candidate': {'date_posted', 'query'},
    }

    for name in dir(tools):
        obj = getattr(tools, name)
        if isinstance(obj, SdkMcpTool) and obj.name in optional_by_tool:
            props = set(obj.input_schema['properties'])
            missing = optional_by_tool[obj.name] - props
            assert not missing, f'{obj.name} dropped optional params entirely: {sorted(missing)}'


# ---------------------------------------------------------------------------
# Reference corpus ordering: the ideal-role profile must use the NEWEST applied jobs
# ---------------------------------------------------------------------------

def _corpus(applied_dir, records):
    """Build an applied-jobs corpus from (applied_date, text, filename) triples."""
    import yaml as yaml_mod

    index = {}
    for applied_date, text, filename in records:
        pdf = _write_pdf(applied_dir / filename, applied_date)
        index[pdf.name] = _index_entry(applied_date, pdf.stat().st_mtime, text=text)
    index_path = applied_dir / 'index.yaml'
    index_path.write_text(yaml_mod.dump(index))
    return index_path


async def test_reference_texts_keep_the_newest_and_drop_the_oldest(tmp_path, monkeypatch):
    """The regression: the slice kept the OLDEST MAX_REFERENCE_JOBS, not the newest.

    _reference_job_texts was filled in filename order (date-prefixed, so oldest first) and both
    consumers slice [:MAX_REFERENCE_JOBS]. With 60 in-horizon records the rater calibrated against
    the oldest 20 and never saw the user's two most recent months.
    """
    from agentic_job_search.config import MAX_REFERENCE_JOBS

    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    total = MAX_REFERENCE_JOBS + 5
    # day 0 is the oldest, day `total-1` the newest; all comfortably inside the horizon.
    records = [
        (
            date.today() - timedelta(days=total - 1 - i),
            f'job text day{i}',
            f'{(date.today() - timedelta(days=total - 1 - i)).isoformat()}-cat-saved_jd-job{i}.pdf',
        )
        for i in range(total)
    ]
    index_path = _corpus(applied, records)

    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    kept = _reference_texts()[:MAX_REFERENCE_JOBS]
    assert _reference_texts()[0] == f'job text day{total - 1}', 'newest must come first'
    assert f'job text day{total - 1}' in kept, 'the newest record must reach the profile'
    assert 'job text day0' not in kept, 'the oldest record must fall outside the cap'
    # The five oldest are exactly the ones dropped.
    for i in range(5):
        assert f'job text day{i}' not in kept


async def test_reference_texts_order_by_applied_date_not_filename(tmp_path, monkeypatch):
    """Ordering must not rely on filename sort — a legacy file with no date prefix sorts anywhere."""
    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    recent = date.today() - timedelta(days=1)
    old = date.today() - timedelta(days=60)
    index_path = _corpus(applied, [
        # Filename sorts FIRST but the record is OLD — so filename order and date order
        # disagree, and only a date-based sort produces the expected result.
        (old, 'old text', 'aaa-legacy-no-date-prefix.pdf'),
        # Filename sorts LAST but the record is RECENT.
        (recent, 'recent text', 'zzz-legacy-also-no-prefix.pdf'),
    ])

    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    assert _reference_texts() == ['recent text', 'old text']


async def test_a_newly_applied_job_changes_the_reference_set(tmp_path, monkeypatch):
    """Structurally impossible before: new records sorted last and never entered the first N.

    That is why the run logged `Reference summary: cache hit` right after ingesting 8 new PDFs —
    `combined` was byte-identical, so the md5 matched and the profile never regenerated.
    """
    from agentic_job_search.config import MAX_REFERENCE_JOBS

    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    total = MAX_REFERENCE_JOBS + 5
    records = [
        (
            date.today() - timedelta(days=total - i),
            f'job text day{i}',
            f'{(date.today() - timedelta(days=total - i)).isoformat()}-cat-saved_jd-job{i}.pdf',
        )
        for i in range(total)
    ]
    index_path = _corpus(applied, records)
    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)
    before = _reference_texts()[:MAX_REFERENCE_JOBS]

    # Apply to one more job today, on top of an already-full corpus.
    index_path = _corpus(applied, [*records,
        (date.today(), 'brand new job text', f'{date.today().isoformat()}-cat-saved_jd-newest.pdf'),
    ])
    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)
    after = _reference_texts()[:MAX_REFERENCE_JOBS]

    assert 'brand new job text' in after, 'a new application must be able to reach the profile'
    assert before != after, 'the reference set must change when a newer job is applied to'


async def test_build_reference_block_renders_the_newest_jobs(tmp_path, monkeypatch):
    """End-to-end through the consumer that the rater actually sees."""
    from agentic_job_search.config import MAX_REFERENCE_JOBS

    _no_legacy_cache(monkeypatch, tmp_path)
    applied = tmp_path / 'applied_jobs'
    total = MAX_REFERENCE_JOBS + 3
    records = [
        (
            date.today() - timedelta(days=total - 1 - i),
            f'unique-marker-{i}',
            f'{(date.today() - timedelta(days=total - 1 - i)).isoformat()}-cat-saved_jd-job{i}.pdf',
        )
        for i in range(total)
    ]
    index_path = _corpus(applied, records)
    await tools.load_applied_jobs(applied_to_dir=applied, index_path=index_path)

    block = agent.build_reference_block()

    assert block.count('[Reference Job') == MAX_REFERENCE_JOBS
    assert f'unique-marker-{total - 1}' in block, 'newest applied job must appear'
    assert 'unique-marker-0' not in block, 'oldest applied job must be dropped'


# ---------------------------------------------------------------------------
# Recovery pass must not turn the distinct counter into a call counter
# ---------------------------------------------------------------------------

class _DuplicateHarvestClient:
    """Every pass re-records the SAME listing ids: distinct stays flat while calls keep growing.

    This is the shape of a recovery pass re-harvesting the page it already harvested, which is
    what four of six queries did on the 2026-08-24 run.
    """

    def __init__(self, distinct_ids: int, calls_per_pass: int):
        self.requests = []
        self.distinct_ids = distinct_ids
        self.calls_per_pass = calls_per_pass

    async def query(self, instruction):
        self.requests.append(instruction)
        for i in range(self.calls_per_pass):
            tools._check_status_counts[f'call-{len(tools._check_status_counts)}-{i}'] = 1
        for i in range(self.distinct_ids):
            tools._distinct_listing_ids.add(f'linkedin/dup-{i}')

    async def receive_response(self):
        return
        yield


def _reset_scrape_globals(monkeypatch):
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(tools, '_ui_alerts', [])
    monkeypatch.setattr(tools, '_search_reports', [])
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})


async def test_recovery_pass_keeps_seen_a_distinct_count(monkeypatch, caplog):
    """The regression: the recovery branch reassigned `seen` to sum(delta.values()) — a CALL
    count — and never refreshed `calls`, producing "100 distinct listing(s) of 50 checked".

    Distinct can never exceed checked. `seen` also feeds _queries_searched, so a recovered query
    reported a call count where every other query reports distinct listings — inflating coverage
    for exactly the queries that needed rescuing.
    """
    _reset_scrape_globals(monkeypatch)
    distinct = 3  # below SCRAPER_MIN_LISTINGS_PER_QUERY, so a recovery pass is triggered
    assert distinct < agent.SCRAPER_MIN_LISTINGS_PER_QUERY
    client = _DuplicateHarvestClient(distinct_ids=distinct, calls_per_pass=25)

    with caplog.at_level(logging.INFO, logger='agentic_job_search.agent'):
        await _scrape(client, ['Staff AI Engineer'])

    assert len(client.requests) == 2, 'expected a first pass plus one recovery pass'
    # The whole point: after recovery this is still the DISTINCT count, not 50 calls.
    assert tools._queries_searched['Staff AI Engineer'] == distinct

    line = next(
        (r.message for r in caplog.records if 'distinct listing(s) of' in r.message), None
    )
    assert line is not None, 'expected the per-query summary line'
    match = re.search(r'inspected (\d+) distinct listing\(s\) of (\d+) checked', line)
    assert match, f'unexpected summary format: {line}'
    reported_distinct, reported_checked = int(match.group(1)), int(match.group(2))
    assert reported_distinct == distinct
    assert reported_distinct <= reported_checked, (
        f'distinct must never exceed checked, got {reported_distinct} > {reported_checked}: {line}'
    )


async def test_duplicate_listing_line_makes_no_region_claim(monkeypatch, caplog):
    """`calls - seen` cannot tell cross-region duplication from a recovery re-harvest.

    The old wording asserted the duplicates "were surfaced by more than one region", which is an
    inference the number does not support — and it manufactured a false diagnosis (a collapsed
    location filter) on a run where region_overlap correctly reported 0.0.
    """
    _reset_scrape_globals(monkeypatch)
    client = _DuplicateHarvestClient(distinct_ids=3, calls_per_pass=25)

    with caplog.at_level(logging.INFO, logger='agentic_job_search.agent'):
        await _scrape(client, ['Staff AI Engineer'])

    dup_lines = [r.message for r in caplog.records if 'but only' in r.message and 'distinct' in r.message]
    assert dup_lines, 'expected the duplicate-listing line'
    for line in dup_lines:
        assert 'more than one region' not in line, (
            f'the line must not attribute duplicates to regions: {line}'
        )
# ---------------------------------------------------------------------------
# Tracked-source guard: a module on disk but not in git breaks every other machine
# ---------------------------------------------------------------------------

# Directories holding Python this repo must be able to import on a fresh clone. Flat globs, to
# match test_every_claude_agent_options_site_limits_context rather than invent a second convention.
_PY_SOURCE_DIRS = ('src/agentic_job_search', 'tests', 'scripts', '.')


def _git(repo_root, *args):
    import subprocess
    return subprocess.run(['git', *args], cwd=repo_root, capture_output=True, text=True)


def test_every_python_module_is_tracked_by_git():
    """`scrape_openrouter.py` sat on disk untracked for months while agent.py imported it, so a
    clean clone of master died at `ImportError: cannot import name 'scrape_openrouter'` -- the
    agent could not start and this suite could not even collect. Locally everything passed the
    whole time, because locally the file is there. Same shape as the context-leak guard above:
    process did not catch it, so a test has to.

    Compared against the INDEX (`git ls-files`), not HEAD. HEAD is stricter and would also have
    caught this, but it fires on every legitimately new file until it is committed -- i.e. on the
    normal workflow -- and a check that fires on success is one people learn to ignore. `git add`
    is the act being policed here; committing is not.

    An ignored file is caught by the same check, since `git ls-files` omits those too.
    """
    repo_root = Path(agent.__file__).parent.parent.parent
    if _git(repo_root, 'rev-parse', '--is-inside-work-tree').returncode != 0:
        pytest.skip(f'{repo_root} is not a git work tree (installed package or source tarball)')

    listed = _git(repo_root, 'ls-files', '--', '*.py')
    assert listed.returncode == 0, f'git ls-files failed: {listed.stderr.strip()}'
    tracked = {repo_root / line for line in listed.stdout.split()}

    on_disk = set()
    for directory in _PY_SOURCE_DIRS:
        # Recursive for the source trees, so a whole nested subpackage cannot slip through the
        # way scrape_openrouter.py did as a single file. The repo root stays SHALLOW on purpose:
        # rglob from there descends into .venv/, run_dir/ and .uv-cache/, none of them tracked.
        found = ((repo_root / directory).glob('*.py') if directory == '.'
                 else (repo_root / directory).rglob('*.py'))
        on_disk |= {p for p in found if '__pycache__' not in p.parts}

    # Without this, a wrong cwd or a typo'd directory compares two empty sets and passes forever.
    package_modules = [p for p in on_disk if p.parent.name == 'agentic_job_search']
    assert len(package_modules) >= 8, (
        f'expected to find the package modules, found {len(package_modules)} in {repo_root} -- '
        'this guard is not looking where it thinks it is'
    )

    untracked = sorted(str(p.relative_to(repo_root)) for p in on_disk - tracked)
    assert not untracked, (
        'Python file(s) on disk but not tracked by git. A fresh clone will not have them, so any '
        'module importing one fails at import:\n'
        f'  git add {" ".join(untracked)}\n'
        '(a .gitignore rule matching a source file looks identical here -- check that too)'
    )


def test_llm_mcp_call_sites_pass_long_timeout():
    """Every chat/generate call through call_mcp_tool must pass LLM_MCP_CALL_TIMEOUT_SECONDS.

    The shared client's 30s default silently cut off healthy LLM calls on 2026-09-14 (4 of 6
    queries lost). A site that falls back to the default fails here.
    """
    import ast

    assert config.LLM_MCP_CALL_TIMEOUT_SECONDS >= 120
    src_dir = Path(config.__file__).parent
    llm_sites = []
    missing = []
    for path in src_dir.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'call_mcp_tool'):
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                continue
            if node.args[1].value not in ('chat', 'generate'):
                continue
            llm_sites.append(f'{path.name}:{node.lineno}')
            kw = {k.arg: k.value for k in node.keywords}
            if getattr(kw.get('timeout_seconds'), 'id', None) != 'LLM_MCP_CALL_TIMEOUT_SECONDS':
                missing.append(f'{path.name}:{node.lineno}')
    assert len(llm_sites) >= 5, f'expected at least 5 LLM call_mcp_tool sites, found {llm_sites}'
    assert not missing, f'LLM call_mcp_tool sites without timeout_seconds=LLM_MCP_CALL_TIMEOUT_SECONDS: {missing}'



# ---------------------------------------------------------------------------
# Code hygiene (2026-09-21): no silent truncation, no swallowed exceptions
# ---------------------------------------------------------------------------

# Neutral text long enough to push anything after it past every cap the regex gates used to apply.
_GATE_FILLER = 'We build reliable software with a friendly team and ship it every week. ' * 90


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_truncate_reported_leaves_short_text_alone(caplog):
    assert text_budget.truncate_reported('short', 10, 'thing') == 'short'
    assert not _warnings(caplog)


def test_truncate_reported_logs_and_marks_a_cut(caplog):
    out = text_budget.truncate_reported('x' * 50, 10, 'the thing')
    assert out.startswith('x' * 10) and '[truncated: kept 10 of 50 chars]' in out
    assert any('Truncating the thing: 50 chars' in m for m in _warnings(caplog))


def test_snippet_says_how_much_it_hid():
    assert text_budget.snippet('abc', max_chars=5) == 'abc'
    shown = text_budget.snippet('a' * 20, max_chars=5)
    assert shown.startswith('aaaaa') and '+15 of 20 chars not shown' in shown


def test_pages_to_prompt_reports_chars_per_page(caplog):
    caplog.set_level(logging.INFO)
    out = text_budget.pages_to_prompt(['page one', 'page two!'], 'PDF x.pdf', 1000)
    assert out == 'page one\n\npage two!'
    assert any('2 page(s), 17 chars' in r.getMessage() and '[8, 9]' in r.getMessage()
               for r in caplog.records)
    assert not _warnings(caplog)


def test_pages_to_prompt_keeps_content_behind_whitespace_padding(caplog):
    """The categorizer read the first 3000 RAW chars: a PDF opening with padding lost its content."""
    padded = ' ' * 5000 + '\n\n\n' * 500 + 'Staff AI Engineer at Acme'
    out = text_budget.pages_to_prompt([padded], 'PDF padded.pdf', 3000)
    assert 'Staff AI Engineer at Acme' in out
    assert '[truncated' not in out
    assert any('normalized whitespace' in m for m in _warnings(caplog))


def test_pages_to_prompt_truncates_after_normalizing_and_says_so(caplog):
    pages = ['word ' * 400, 'more ' * 400, 'last ' * 400]
    out = text_budget.pages_to_prompt(pages, 'PDF big.pdf', 2500)
    assert '[truncated: kept 2500 of' in out and 'pages 1–1 of 3 complete' in out
    warnings = _warnings(caplog)
    assert any('normalized whitespace' in m for m in warnings)
    assert any('still' in m and 'truncating' in m for m in warnings)


def test_pdf_prompt_cap_is_derived_from_the_model_context():
    assert int(
        config.ANTHROPIC_MODEL_LOW_CONTEXT_TOKENS * config.CHARS_PER_TOKEN_ESTIMATE
        * config.PDF_PROMPT_CONTEXT_SHARE) == config.PDF_PROMPT_MAX_CHARS
    assert config.PDF_PROMPT_MAX_CHARS > config.REFERENCE_JOB_PROMPT_MAX_CHARS


async def test_categorize_sends_text_past_the_old_3000_char_cap(tmp_path, monkeypatch):
    save_dir = tmp_path / 'save'
    save_dir.mkdir()
    (save_dir / 'posting.pdf').write_bytes(b'%PDF-fake')
    monkeypatch.setattr(tools, 'read_pdf_pages', lambda pdf: ['intro ' * 1000, 'the real title'])
    seen = {}

    async def fake_categorize(text, filename):
        seen['text'] = text
        return 'saved_jd'

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    await tools.categorize_save_dir_pdfs(save_dir)
    assert 'the real title' in seen['text']
    assert (save_dir / 'cat-saved_jd-posting.pdf').exists()


def test_residency_gate_reads_past_the_old_cap():
    extract = _make_extract(location='Germany (Remote)',
                            description=_GATE_FILLER + 'Candidates must be based in Germany.')
    assert agent.derive_residency_scope(extract) == 'country_only'


def test_workplace_gate_reads_past_the_old_cap():
    extract = _make_extract(location='Germany', description=_GATE_FILLER + 'This role is hybrid.')
    assert agent.derive_workplace_type(extract) == 'hybrid'


def test_education_gate_reads_past_the_old_cap():
    extract = _make_extract(description=_GATE_FILLER + 'PhD in Machine Learning is required.')
    assert agent.derive_education_requirement(extract) == 'phd'


def test_agency_gate_reads_past_the_old_cap():
    extract = _make_extract(description=_GATE_FILLER + 'Our client is a leading fintech.')
    assert agent.derive_agency_posting(extract) is True


def test_contract_warning_reads_past_the_old_cap():
    extract = _make_extract(description=_GATE_FILLER + 'This is a 6-month contract.', salary='CAD 200,000 - 240,000 per year')
    assert 'Contract role — full-time preferred' in agent.build_deterministic_warnings(_make_candidate(), extract)


def test_gate_filler_is_longer_than_every_old_cap():
    assert len(_GATE_FILLER) > 4000


async def test_sdk_stream_is_closed_when_an_api_error_aborts(monkeypatch):
    """Raising mid-`async for` left the generator open; asyncio.run() then failed closing it."""
    closed = []

    async def fake_sdk_query(**kwargs):
        try:
            yield ResultMessage(subtype='success', duration_ms=1, duration_api_ms=1, is_error=True,
                                num_turns=1, session_id='s')
            yield ResultMessage(subtype='success', duration_ms=1, duration_api_ms=1, is_error=False,
                                num_turns=1, session_id='s', structured_output={'category': 'x'})
        finally:
            closed.append(True)

    monkeypatch.setattr(tools, 'sdk_query', fake_sdk_query)
    with pytest.raises(tools.AgentApiError) as info:
        await tools._categorize_pdf_text('text', 'a.pdf')
    assert closed == [True], 'the SDK stream must be closed before the error propagates'
    assert 'no HTTP status reported' in str(info.value) and 'claude /login' in str(info.value)


async def test_playwright_server_that_never_answers_raises(monkeypatch):
    class FakeProc:
        returncode = None
        terminated = False

        def terminate(self):
            self.terminated = True

    proc = FakeProc()

    async def fake_exec(*cmd):
        return proc

    async def no_sleep(_seconds):
        return None

    def refuse(*args, **kwargs):
        raise requests.ConnectionError('connection refused')

    monkeypatch.setattr(agent.asyncio, 'create_subprocess_exec', fake_exec)
    monkeypatch.setattr(agent.asyncio, 'sleep', no_sleep)
    monkeypatch.setattr(agent.requests, 'get', refuse)
    with pytest.raises(RuntimeError, match='did not answer') as info:
        await agent.start_playwright_server(9999, browser_mode='headless')
    assert 'connection refused' in str(info.value)
    assert proc.terminated, 'a server that never answered must not be left running'


# ---------------------------------------------------------------------------
# Browser death during Stage 2 (2026-09-21: @playwright/mcp died of a heap OOM mid-run)
# ---------------------------------------------------------------------------

class _ExitedProc:
    returncode = -6
    terminated = False

    def terminate(self):
        raise ProcessLookupError('terminate() on an exited process')

    async def wait(self):
        return self.returncode


async def test_browser_server_stop_tolerates_an_exited_process():
    """terminate() on an exited process raised and cost the 2026-09-21 run its summary."""
    browser = agent.BrowserServer('headless')
    browser.proc = _ExitedProc()
    await browser.stop()   # must not raise


class _FakeBrowser:
    """Duck-typed BrowserServer whose liveness the test drives."""

    def __init__(self):
        self.dead = False
        self.restarts = 0
        self.mcp = {'type': 'http', 'url': 'http://localhost:1/mcp'}

    def is_dead(self):
        return self.dead

    async def confirm_dead(self):
        return self.dead

    async def restart(self):
        self.restarts += 1
        self.dead = False
        self.mcp['url'] = 'http://localhost:2/mcp'


def _stage2_with_dying_browser(monkeypatch, browser, dies_on: list[str]):
    """Stub extraction so each job id in ``dies_on`` kills the browser once, then fails normally."""
    monkeypatch.setattr(agent, 'MODEL_NAME_EXTRACTOR', OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC)
    calls = []

    async def fake_read_page(candidate, url, stats):
        # The real read_page logs a dead browser and returns None; the loop then sees is_dead().
        calls.append((candidate['job_id'], url))
        if candidate['job_id'] in dies_on:
            dies_on.remove(candidate['job_id'])
            browser.dead = True
            return None
        return linkedin_page.JobPage(text='page', source='dom')

    async def fake_extract(candidate, page_text, stats, system_prompt):
        return None

    async def fake_direct(candidate, url, stats, page=None):
        return None

    monkeypatch.setattr(agent, 'read_page', fake_read_page)
    monkeypatch.setattr(agent, 'extract_job_page_openrouter', fake_extract)
    monkeypatch.setattr(agent, 'extract_job_page_direct', fake_direct)
    monkeypatch.setattr(agent.tools_module, '_ui_alerts', [])
    return calls


async def test_stage2_restarts_a_dead_browser_and_retries_the_same_job(monkeypatch):
    browser = _FakeBrowser()
    calls = _stage2_with_dying_browser(monkeypatch, browser, dies_on=['1'])
    released = []
    monkeypatch.setattr(agent.tools_module, 'forget_processed_job', lambda s, j: released.append(j))
    funnel = {}
    stats = {'extraction': agent.new_stage_stats(), 'rating': agent.new_stage_stats()}

    await agent.evaluate_all_candidates(
        [_make_candidate(job_id='1'), _make_candidate(job_id='2')], browser.mcp, 'prompt', 'profile',
        stats, funnel=funnel, browser=browser,
    )

    assert [job for job, _ in calls] == ['1', '1', '2'], 'the job the browser died on must be retried'
    assert calls[1][1] == 'http://localhost:2/mcp', 'the retry must use the restarted server'
    assert browser.restarts == 1
    assert 'eval_error' not in funnel, "a dead browser is not the job's fault"
    assert funnel.get('extract_failed') == 2
    assert released == []
    assert [a['kind'] for a in agent.tools_module._ui_alerts] == ['browser_restarted']


async def test_stage2_stops_and_releases_jobs_when_the_browser_stays_dead(monkeypatch, caplog):
    browser = _FakeBrowser()
    calls = _stage2_with_dying_browser(monkeypatch, browser, dies_on=['1', '2'])
    released = []
    monkeypatch.setattr(agent.tools_module, 'forget_processed_job', lambda s, j: released.append(j))
    funnel = {}
    stats = {'extraction': agent.new_stage_stats(), 'rating': agent.new_stage_stats()}
    candidates = [_make_candidate(job_id=str(n)) for n in range(1, 5)]

    with caplog.at_level(logging.ERROR, logger=agent.logger.name):
        await agent.evaluate_all_candidates(
            candidates, browser.mcp, 'prompt', 'profile', stats, funnel=funnel, browser=browser,
        )

    assert [job for job, _ in calls] == ['1', '1', '2'], 'nothing may run after the browser stays dead'
    assert released == ['2', '3', '4'], 'every unevaluated job, including the failing one, goes back'
    assert funnel['browser_dead'] == 3
    assert 'eval_error' not in funnel
    stop_lines = [r for r in caplog.records if 'Stage 2 stopped' in r.message]
    assert len(stop_lines) == 1, 'one error line, not one per remaining job'
    assert 'browser_dead' in [a['kind'] for a in agent.tools_module._ui_alerts]
    assert any('BROWSER DIED' in a for a in agent.assess_run_health(
        {'ui_alerts': agent.tools_module._ui_alerts}))


async def test_a_closed_port_counts_as_dead_before_the_exit_is_reaped(monkeypatch):
    """2026-09-28: the port closed and 3 jobs failed in 43 ms while proc.returncode was still None."""
    browser = agent.BrowserServer('headless')
    browser.proc = type('Running', (), {'returncode': None})()
    browser.port = 1

    async def refused(host, port):
        raise ConnectionRefusedError(61, 'Connect call failed')

    monkeypatch.setattr(agent.asyncio, 'open_connection', refused)
    assert browser.is_dead() is False
    assert await browser.confirm_dead() is True
    assert browser.is_dead() is True, 'browser_usable reads is_dead(), so the verdict must stick'


async def test_a_listening_port_means_the_page_failed_not_the_browser(monkeypatch):
    browser = agent.BrowserServer('headless')
    browser.proc = type('Running', (), {'returncode': None})()
    browser.port = 1

    class _Writer:
        def close(self):
            pass

        async def wait_closed(self):
            pass

    async def accepted(host, port):
        return object(), _Writer()

    monkeypatch.setattr(agent.asyncio, 'open_connection', accepted)
    assert await browser.confirm_dead() is False


def test_forget_processed_job_removes_the_record(tmp_path, monkeypatch):
    processed = tmp_path / 'processed_jobs'
    processed.mkdir()
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', processed)
    monkeypatch.setattr(tools, '_processed_jobs', {('linkedin', '42'), ('linkedin', '420')})
    (processed / 'job_posting-linkedin-42-2026Sep21-1-acme-engineer.yaml').write_text('site: linkedin\n')
    (processed / 'job_posting-linkedin-420-2026Sep21-1-acme-engineer.yaml').write_text('site: linkedin\n')

    assert tools.forget_processed_job('linkedin', '42') == 1

    assert [f.name for f in processed.iterdir()] == ['job_posting-linkedin-420-2026Sep21-1-acme-engineer.yaml'], \
        'a job id that is a prefix of another must not take the other with it'
    assert tools._processed_jobs == {('linkedin', '420')}
    with pytest.raises(FileNotFoundError, match='linkedin/42'):
        tools.forget_processed_job('linkedin', '42')


async def test_main_logs_an_exception_that_ends_the_run(monkeypatch, caplog, tmp_path):
    """On 2026-09-21 the cause of a dead run reached only the terminal, never the run log."""
    requirements = tmp_path / 'JOB_REQUIREMENTS.md'
    requirements.write_text('x')
    monkeypatch.setattr(agent, 'JOB_REQUIREMENTS_PATH', requirements)
    monkeypatch.setattr(agent.sys, 'argv', ['main.py', '-n', '--no-version-check'])
    monkeypatch.setattr(agent, 'setup_logging', lambda *args, **kwargs: None)   # keeps caplog attached
    released = []
    monkeypatch.setattr(agent.tools_module, 'acquire_run_lock', lambda **kwargs: None)
    monkeypatch.setattr(agent.tools_module, 'release_run_lock', lambda: released.append(True))
    monkeypatch.setattr(agent, 'load_processed_jobs', lambda: None)

    async def no_op():
        return None

    for name in ('categorize_save_dir_pdfs', 'ingest_save_dir_applied_pdfs', 'load_applied_jobs'):
        monkeypatch.setattr(agent.tools_module, name, no_op)

    async def crash(**kwargs):
        raise ProcessLookupError('terminate() on an exited process')

    monkeypatch.setattr(agent, 'run_non_interactive', crash)

    with caplog.at_level(logging.ERROR, logger=agent.logger.name), pytest.raises(ProcessLookupError):
        await agent.main()

    assert any('Run aborted by ProcessLookupError' in r.message and r.exc_info for r in caplog.records)
    assert released == [True], 'the run lock must still be released'


def test_scraper_bad_tool_json_is_logged_and_returned_to_the_model(monkeypatch, caplog):
    """It used to become {} silently, running the tool with arguments the model never chose."""
    bad_call = json.dumps({'ok': True, 'content': '', 'finish_reason': 'tool_calls', 'usage': {},
                           'tool_calls': [{'id': 'c1', 'function': {'name': 'browser_snapshot',
                                                                    'arguments': '{not json'}}]})
    sent = []

    async def fake_call_mcp_tool(url, tool_name, args, timeout_seconds=None):
        sent.append(args)
        return [bad_call, _CHAT_DONE][len(sent) - 1]

    browser_calls = []

    async def browser(name, args):
        browser_calls.append(name)
        return ''

    monkeypatch.setattr(scrape_openrouter, 'call_mcp_tool', fake_call_mcp_tool)
    asyncio.run(scrape_openrouter.ScrapeSession(browser, 'Staff AI Engineer').run('sys', 'user', []))
    assert browser_calls == [], 'a call with unparseable arguments must not run'
    tool_reply = sent[1]['messages'][-1]
    assert tool_reply['role'] == 'tool' and 'invalid JSON arguments' in tool_reply['content']
    assert any('invalid JSON arguments' in m and '{not json' in m for m in _warnings(caplog))


def _run_scraper_with_one_browser_call(monkeypatch, tool_name: str, result: str) -> str:
    """Run a ScrapeSession whose model calls tool_name once; return what the model was sent back."""
    call = json.dumps({'ok': True, 'content': '', 'finish_reason': 'tool_calls', 'usage': {},
                       'tool_calls': [{'id': 'c1', 'function': {'name': tool_name, 'arguments': '{}'}}]})
    sent = []

    async def fake_call_mcp_tool(url, name, args, timeout_seconds=None):
        sent.append(args)
        return [call, _CHAT_DONE][len(sent) - 1]

    async def browser(name, args):
        return result

    monkeypatch.setattr(scrape_openrouter, 'call_mcp_tool', fake_call_mcp_tool)
    asyncio.run(scrape_openrouter.ScrapeSession(browser, 'Staff AI Engineer').run('sys', 'user', []))
    return sent[1]['messages'][-1]['content']


def test_scraper_snapshot_drops_the_detail_pane_and_keeps_every_card(monkeypatch, caplog):
    """The pane inflated every snapshot past the cap, and the cut fell inside the cards (2026-09-30)."""
    card = '    - button "Acme Toronto (Remote) Dismiss Staff ML Engineer job" [ref=e{}] [cursor=pointer]\n'
    cards = ''.join(card.format(i) for i in range(config.SCRAPER_TOOL_RESULT_MAX_CHARS // len(card) + 1))
    pane_filler = '      - paragraph: posting text\n' * (config.SCRAPER_SNAPSHOT_RESULT_MAX_CHARS // 10)
    snapshot = ('- main [ref=e1]:\n  - paragraph [ref=e2]: 99+ results\n  - generic [ref=e3]:\n' + cards
                + '  - generic [ref=e4]:\n    - generic [ref=e5]:\n      - heading "About the job" [level=2]\n'
                + pane_filler)
    assert len(cards) > config.SCRAPER_TOOL_RESULT_MAX_CHARS
    assert len(snapshot) > config.SCRAPER_SNAPSHOT_RESULT_MAX_CHARS

    with caplog.at_level(logging.INFO):
        content = _run_scraper_with_one_browser_call(monkeypatch, 'browser_snapshot', snapshot)

    assert cards in content, 'every result card reaches the model'
    assert 'About the job' not in content and '[job detail pane omitted: ' in content
    assert '[truncated' not in content
    assert not any('Truncating' in m for m in _warnings(caplog))


def test_scraper_evaluate_result_over_the_shared_cap_still_warns(monkeypatch, caplog):
    result = 'x' * (config.SCRAPER_TOOL_RESULT_MAX_CHARS + 1)
    content = _run_scraper_with_one_browser_call(monkeypatch, 'browser_evaluate', result)
    assert '[truncated: kept ' in content
    assert any('Truncating browser_evaluate result' in m for m in _warnings(caplog))


def test_parse_posting_date_raises_on_an_unknown_format():
    assert tools.parse_posting_date('') is None
    assert tools.parse_posting_date('2026-09-01') == date(2026, 9, 1)
    assert tools.parse_posting_date('3 days ago') == date.today() - timedelta(days=3)
    with pytest.raises(ValueError, match="'last Tuesday'"):
        tools.parse_posting_date('last Tuesday')


@pytest.mark.parametrize(('raw', 'days_ago'), [
    # Every form below was logged as unparseable on a real run (tallied 2026-09-24).
    ('Reposted 4 days ago', 4),
    ('Reposted 1 day ago', 1),
    ('Reposted 2 days ago (LinkedIn)', 2),
    ('Reposted 3 days ago (Requisition ID: 12345)', 3),
    ('Reposted 3 days ago (no exact date shown)', 3),
    ('Reposted 3 days ago (as of page visit)', 3),
    ('Reposted 2 weeks ago', 14),
    ('Posted 2 months ago', 2 * tools.DAYS_PER_MONTH_APPROX),
])
def test_parse_posting_date_accepts_relative_dates_with_surrounding_text(raw, days_ago):
    assert tools.parse_posting_date(raw) == date.today() - timedelta(days=days_ago)


@pytest.mark.parametrize('raw', [
    '5 minutes ago',
    'Reposted 7 hours ago',
    'Reposted 1 hours ago (2020-01-01)',
    'Reposted 1 hours ago (as of 2020-01-01)',
    'Posted 1 hours ago (viewed 2020-01-01, so ~2020-01-01)',
])
def test_parse_posting_date_relative_beats_a_trailing_viewed_date(raw):
    """A date AFTER the relative phrase is when the page was viewed, not when the job was posted."""
    parsed = tools.parse_posting_date(raw)
    assert date.today() - timedelta(days=1) <= parsed <= date.today()


@pytest.mark.parametrize('raw', [
    '2026-09-20 (4 days ago)',
    '2026-09-20 (posted 4 days ago)',
    '2026-09-20 (reposted 3 hours ago)',
    '2026-09-20 (1 day ago)',
])
def test_parse_posting_date_leading_iso_date_wins(raw):
    assert tools.parse_posting_date(raw) == date(2026, 9, 20)


@pytest.mark.parametrize('raw', ['last Tuesday', 'Reposted recently', '12345 (Requisition ID)'])
def test_parse_posting_date_still_raises_without_a_date(raw):
    with pytest.raises(ValueError):
        tools.parse_posting_date(raw)


async def test_hard_rules_log_an_unparseable_posting_date_once(caplog):
    extract = _make_extract(date_posted='last Tuesday')
    assert await agent.apply_hard_rules(_make_candidate(), extract) is None
    assert sum("'last Tuesday'" in m for m in _warnings(caplog)) == 1


def test_parse_iso_date_raises_naming_the_value():
    with pytest.raises(ValueError, match="'09/01/2026'"):
        tools._parse_iso_date('09/01/2026')


def test_recruiter_notification_with_a_bad_date_is_dropped_and_logged(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'recruiter_notifications.yaml'
    path.write_text(yaml.dump([
        {'job_id': '1', 'notified': date.today().isoformat()},
        {'job_id': '2', 'notified': 'yesterday'},
    ]), encoding='utf-8')
    monkeypatch.setattr(tools, 'RECRUITER_NOTIFICATIONS_PATH', path)
    assert [e['job_id'] for e in tools.load_recruiter_notifications()] == ['1']
    assert any("'2'" in m and "'yesterday'" in m for m in _warnings(caplog))


def test_parse_added_date_raises_and_the_blacklist_warns_with_the_value(monkeypatch, caplog):
    with pytest.raises(ValueError, match="'not-a-date'"):
        preferences._parse_added_date('not-a-date')
    assert preferences._parse_added_date(None) is None
    prefs = preferences.load_preferences()
    prefs['companies']['blacklist'] = [{'name': 'Initech', 'reason': 'r', 'added': 'not-a-date'}]
    monkeypatch.setattr(preferences, 'load_preferences', lambda: prefs)
    assert ('Initech', 'r') in preferences.blacklisted_companies(), 'a bad date keeps the entry active'
    assert any('Initech' in m and "'not-a-date'" in m for m in _warnings(caplog))


def test_extract_json_object_chains_the_last_decode_error():
    with pytest.raises(ValueError) as info:
        triage.extract_json_object('prose {not json} more prose')
    assert isinstance(info.value.__cause__, json.JSONDecodeError)


# ---------------------------------------------------------------------------
# Scraper prompt / tool parity (2026-08-31..09-28: the fallback's prompt named tools it did not have)
# ---------------------------------------------------------------------------

def _every_tool_name_there_is() -> set[str]:
    """Every tool name either scraper has ever served: the SDK tools here plus the local defs."""
    sdk = {value.name for value in vars(tools).values() if isinstance(value, SdkMcpTool)}
    return sdk | {d['function']['name'] for d in scrape_openrouter.LOCAL_TOOL_DEFS}


def _tools_named_in(text: str) -> set[str]:
    known = _every_tool_name_there_is()
    return {word for word in re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text)
            if word in known or word.startswith('browser_')}


async def _scraper_prompts(monkeypatch) -> list[str]:
    """The system prompt plus both recovery instructions run_scraper can send."""
    for name in ('_check_status_counts', '_candidates_per_query', '_queries_searched', '_query_errors',
                 '_check_status_per_query'):
        monkeypatch.setattr(tools, name, {})
    for name in ('_ui_alerts', '_search_reports', '_candidates'):
        monkeypatch.setattr(tools, name, [])
    monkeypatch.setattr(tools, '_distinct_listing_ids', set())
    monkeypatch.setattr(tools, '_search_ids', {})
    monkeypatch.setattr(agent, 'SCRAPER_INTER_QUERY_DELAY_SECONDS', (0, 0))
    monkeypatch.setattr(agent, 'SCRAPER_INTER_SEARCH_DELAY_SECONDS', (0, 0))
    sent = []

    async def run_pass(instruction):
        sent.append(instruction)
        if tools._current_query == 'one listing' and len(sent) == 3:
            tools._distinct_listing_ids.add('linkedin/1')   # a first pass that saw exactly one
        return None

    await agent.run_scraper(run_pass, ['no listings', 'one listing'], {'cost': 0.0})
    assert any('no job listings at all' in s for s in sent) and any('inspected only 1' in s for s in sent)
    return [agent.build_scraper_instructions(), *sent]


def test_the_anthropic_scraper_serves_exactly_the_openrouter_scrapers_own_tools(monkeypatch):
    served = []
    monkeypatch.setattr(agent, 'create_sdk_mcp_server', lambda **kw: served.extend(t.name for t in kw['tools']))
    agent.make_anthropic_scraper_server(browser_call=None)
    assert served == [d['function']['name'] for d in scrape_openrouter.LOCAL_TOOL_DEFS]


async def test_every_tool_a_scraper_prompt_names_is_served_by_both_scrapers(monkeypatch):
    served = []
    monkeypatch.setattr(agent, 'create_sdk_mcp_server', lambda **kw: served.extend(t.name for t in kw['tools']))
    agent.make_anthropic_scraper_server(browser_call=None)
    anthropic = set(served)
    openrouter = {d['function']['name'] for d in scrape_openrouter.LOCAL_TOOL_DEFS}
    disallowed = {name.removeprefix('mcp__playwright__') for name in config.SCRAPER_DISALLOWED_BROWSER_TOOLS}

    named = set().union(*(_tools_named_in(p) for p in await _scraper_prompts(monkeypatch)))
    assert {'run_ui_contract', 'harvest_listings', 'record_listings'} <= named, 'the guard must see the prompt'
    for name in sorted(named):
        if name.startswith('browser_'):
            assert name not in disallowed, f'the prompt names {name}, which neither scraper exposes'
        else:
            assert name in anthropic, f'the prompt names {name}, which the Anthropic fallback does not serve'
            assert name in openrouter, f'the prompt names {name}, which the OpenRouter scraper does not serve'


async def test_an_anthropic_tool_runs_the_same_code_on_the_current_query(monkeypatch):
    captured = {}

    def capture(**kw):
        captured.update({t.name: t for t in kw['tools']})

    monkeypatch.setattr(agent, 'create_sdk_mcp_server', capture)
    seen = []

    async def fake_dispatch(self, name, args):
        seen.append((self.query, name, args))
        return 'ok'

    monkeypatch.setattr(scrape_openrouter.ScrapeSession, 'dispatch_local', fake_dispatch)
    monkeypatch.setattr(tools, '_current_query', 'AI Architect')
    agent.make_anthropic_scraper_server(browser_call=None)
    result = await captured['run_ui_contract'].handler({'region': 'Canada'})
    assert result == {'content': [{'type': 'text', 'text': 'ok'}]}
    assert seen == [('AI Architect', 'run_ui_contract', {'region': 'Canada'})]
