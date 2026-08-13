"""Tests for the job search agent."""

from pathlib import Path
import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta

import pytest
import yaml
from claude_agent_sdk import ResultMessage

from agentic_job_search import agent
from agentic_job_search import config
from agentic_job_search import extract_openrouter
from agentic_job_search import preferences
from agentic_job_search import tools_generic as tools
from agentic_job_search import triage


# ---------------------------------------------------------------------------
# underscorify
# ---------------------------------------------------------------------------

def test_underscorify_basic():
    assert tools.underscorify("Hello World") == "hello_world"


def test_underscorify_special_chars():
    assert tools.underscorify("Shopify Inc.") == "shopify_inc"


def test_underscorify_consecutive_separators():
    assert tools.underscorify("Senior  Software---Engineer") == "senior_software_engineer"


def test_underscorify_leading_trailing():
    assert tools.underscorify("  leading and trailing  ") == "leading_and_trailing"


def test_underscorify_numbers():
    assert tools.underscorify("GPT-4 Engineer") == "gpt_4_engineer"


# ---------------------------------------------------------------------------
# do_save_job_posting
# ---------------------------------------------------------------------------

async def test_save_job_posting_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)

    result = await tools.do_save_job_posting(
        company="Shopify",
        description="Senior Software Engineer",
        rating=4,
        content="# Senior Software Engineer at Shopify\n\nGreat remote role.",
    )

    assert not result.get("isError")
    saved_dirs = list(tmp_path.glob("saved_jobs-*"))
    assert len(saved_dirs) == 1

    files = list(saved_dirs[0].glob("job_posting-noid-rating_4-shopify-senior_software_engineer-*.md"))
    assert len(files) == 1
    assert files[0].read_text() == "# Senior Software Engineer at Shopify\n\nGreat remote role."


async def test_save_job_posting_result_contains_path(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)

    result = await tools.do_save_job_posting(
        company="Acme",
        description="Engineer",
        rating=2,
        content="content",
    )

    text = result["content"][0]["text"]
    assert "saved_jobs-" in text
    assert "job_posting-noid-rating_2-acme-engineer-" in text


async def test_save_job_posting_creates_daily_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)

    for rating in (3, 5):
        await tools.do_save_job_posting(
            company="Corp",
            description="Role",
            rating=rating,
            content=f"content {rating}",
        )

    saved_dirs = list(tmp_path.glob("saved_jobs-*"))
    assert len(saved_dirs) == 1  # same day → same dir
    assert len(list(saved_dirs[0].glob("*.md"))) == 2


async def test_save_job_posting_caps_long_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)

    long_description = "triaged out " + "this is an extremely long triage reason sentence " * 10
    result = await tools.do_save_job_posting(
        company="A Very Long Company Name That Goes On And On Incorporated",
        description=long_description,
        rating=1,
        content="content",
        job_id="4442638114",
    )

    assert not result.get("isError")
    files = list(tmp_path.glob("saved_jobs-*/job_posting-4442638114-rating_1-*.md"))
    assert len(files) == 1
    assert len(files[0].name.encode()) < 255


# ---------------------------------------------------------------------------
# do_update_job_requirements
# ---------------------------------------------------------------------------

async def test_update_job_requirements_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    content = "# Requirements\n\n- Remote only\n- Canada or EU\n"
    result = await tools.do_update_job_requirements(content)

    assert not result.get("isError")
    assert (tmp_path / "JOB_REQUIREMENTS.md").read_text() == content


async def test_update_job_requirements_returns_content(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    content = "# Requirements\n\n- Senior only\n"
    result = await tools.do_update_job_requirements(content)

    # New contents must be in the tool result so the agent has them in context
    assert content in result["content"][0]["text"]


async def test_update_job_requirements_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "JOB_REQUIREMENTS.md"
    monkeypatch.setattr(tools, "JOB_REQUIREMENTS_PATH", path)

    await tools.do_update_job_requirements("first version")
    await tools.do_update_job_requirements("second version")

    assert path.read_text() == "second version"


# ---------------------------------------------------------------------------
# log_run_cost
# ---------------------------------------------------------------------------


def test_log_run_cost_writes_jsonl_line(tmp_path):
    log_path = tmp_path / "cost_log.jsonl"
    record = {"mode": "non-interactive", "total_cost": 1.2345}

    tools.log_run_cost(record, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_log_run_cost_appends_multiple_runs(tmp_path):
    log_path = tmp_path / "cost_log.jsonl"

    tools.log_run_cost({"run": 1}, log_path=log_path)
    tools.log_run_cost({"run": 2}, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"run": 1}, {"run": 2}]


def test_log_run_cost_creates_parent_dirs(tmp_path):
    log_path = tmp_path / "nested" / "cost_log.jsonl"

    tools.log_run_cost({"run": 1}, log_path=log_path)

    assert log_path.exists()


def test_new_stage_stats_zeroed():
    stats = agent.new_stage_stats()
    assert stats == {
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "session_costs": {},
    }


def test_accumulate_stage_stats_sums_usage_and_cost():
    """Two independent requests (separate sessions) sum in both usage and cost."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 5,
    }
    stats = agent.new_stage_stats()
    agent.accumulate_stage_stats(stats, _result_msg("req-1", 0.01, **usage))
    agent.accumulate_stage_stats(stats, _result_msg("req-2", 0.01, **usage))

    assert agent.public_stage_stats({"s": stats})["s"] == {
        "cost": pytest.approx(0.02),
        "input_tokens": 200,
        "output_tokens": 100,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 10,
    }


# ---------------------------------------------------------------------------
# do_check_and_record_job
# ---------------------------------------------------------------------------


def _recent_date() -> str:
    return (date.today() - timedelta(days=5)).isoformat()


def _old_date() -> str:
    return (date.today() - timedelta(days=30)).isoformat()


async def test_check_and_record_job_new(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    result = await tools.do_check_and_record_job(
        "linkedin", "1234567890", "Shopify", "Senior Engineer", date_posted=_recent_date()
    )

    assert result["content"][0]["text"] == "new"
    files = list((tmp_path / "processed_jobs").glob("job_posting-linkedin-1234567890-*.yaml"))
    assert len(files) == 1


async def test_check_and_record_job_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    await tools.do_check_and_record_job(
        "linkedin", "1234567890", "Shopify", "Senior Engineer", date_posted=_recent_date()
    )
    result = await tools.do_check_and_record_job(
        "linkedin", "1234567890", "Shopify", "Senior Engineer", date_posted=_recent_date()
    )

    assert result["content"][0]["text"] == "already_processed"


async def test_check_and_record_job_too_old(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    result = await tools.do_check_and_record_job(
        "linkedin", "9999999999", "OldCo", "Stale Role", date_posted=_old_date()
    )

    assert result["content"][0]["text"] == "too_old"
    assert not list((tmp_path / "processed_jobs").glob("*.yaml"))


async def test_check_and_record_job_different_sites(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    r1 = await tools.do_check_and_record_job(
        "linkedin", "111", "Corp", "Engineer", date_posted=_recent_date()
    )
    r2 = await tools.do_check_and_record_job(
        "indeed", "111", "Corp", "Engineer", date_posted=_recent_date()
    )

    assert r1["content"][0]["text"] == "new"
    assert r2["content"][0]["text"] == "new"


async def test_check_and_record_job_yaml_content(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    posted = _recent_date()
    await tools.do_check_and_record_job(
        "linkedin", "5555555555", "Stripe", "Staff Engineer", date_posted=posted
    )

    files = list((tmp_path / "processed_jobs").glob("*.yaml"))
    data = yaml.safe_load(files[0].read_text())
    assert data["site"] == "linkedin"
    assert data["job_id"] == "5555555555"
    assert data["date_posted"] == posted
    assert data["date_recorded"] == date.today().isoformat()
    assert data["company"] == "Stripe"
    assert data["description"] == "Staff Engineer"


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
    "This position does not provide visa sponsorship",
    'Employment authorization without sponsorship required',
    'must be legally authorized to work in the united states',
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
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    await tools.do_save_job_posting("Shopify", "Engineer", 4, "content", job_id="3859234876")
    await tools.do_save_job_posting("Acme", "Designer", 2, "content", job_id="1122334455")

    tools.load_processed_jobs()
    assert ("linkedin", "3859234876") in tools._processed_jobs
    assert ("linkedin", "1122334455") in tools._processed_jobs


async def test_load_processed_jobs_from_yaml_files(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    (tmp_path / "processed_jobs").mkdir()
    (tmp_path / "processed_jobs" / "job_posting-indeed-42-2026Apr10-000-co-role.yaml").write_text(
        yaml.dump({"site": "indeed", "job_id": "42", "date_posted": "2026-04-05",
                   "date_recorded": "2026-04-10", "company": "co", "description": "role"})
    )

    tools.load_processed_jobs()
    assert ("indeed", "42") in tools._processed_jobs


async def test_load_processed_jobs_combines_both(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)
    monkeypatch.setattr(tools, "PROCESSED_JOBS_DIR", tmp_path / "processed_jobs")
    monkeypatch.setattr(tools, "_processed_jobs", set())

    await tools.do_save_job_posting("Corp", "Role", 3, "content", job_id="111")

    (tmp_path / "processed_jobs").mkdir()
    (tmp_path / "processed_jobs" / "job_posting-indeed-999-2026Apr10-000-co-role.yaml").write_text(
        yaml.dump({"site": "indeed", "job_id": "999", "date_posted": "2026-04-05",
                   "date_recorded": "2026-04-10", "company": "co", "description": "role"})
    )

    tools.load_processed_jobs()
    assert ("linkedin", "111") in tools._processed_jobs
    assert ("indeed", "999") in tools._processed_jobs


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

def test_build_system_prompt_includes_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RUN_DIR", tmp_path)
    monkeypatch.setattr(agent, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    (tmp_path / "R_Garth_Wood-resume-2026Apr08v1.md").write_text("# Garth Wood\n\nExperienced engineer.")

    prompt = agent.build_system_prompt(interactive=True)
    assert "Garth Wood" in prompt
    assert "RESUME" in prompt


def test_build_system_prompt_includes_job_requirements(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RUN_DIR", tmp_path)
    req_path = tmp_path / "JOB_REQUIREMENTS.md"
    monkeypatch.setattr(agent, "JOB_REQUIREMENTS_PATH", req_path)

    req_path.write_text("# Requirements\n\n- Remote only")

    prompt = agent.build_system_prompt(interactive=True)
    assert "Remote only" in prompt
    assert "JOB_REQUIREMENTS.md" in prompt


def test_build_system_prompt_warns_when_no_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent, "RUN_DIR", tmp_path)
    monkeypatch.setattr(agent, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    agent.build_system_prompt(interactive=True)

    assert "Warning" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Live agent tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.live_agent_claude
async def test_live_agent_calls_save_job_posting(tmp_path, monkeypatch):
    """Agent uses the save_job_posting tool when instructed to save a job."""
    monkeypatch.setattr(tools, "RUN_DIR", tmp_path)

    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a job search assistant. "
            "When asked to save a job, use the save_job_posting tool exactly once."
        ),
        mcp_servers={"job_search": tools.make_job_search_server(interactive=False)},
        permission_mode="acceptEdits",
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

    saved_dirs = list(tmp_path.glob("saved_jobs-*"))
    assert len(saved_dirs) == 1, "Expected a saved_jobs-* directory to be created"
    files = list(saved_dirs[0].glob("job_posting-testcorp-*-rating_4-*.md"))
    assert len(files) == 1, f"Expected one saved job file, found: {list(saved_dirs[0].iterdir())}"


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


async def test_ingest_date_prefixes_from_mtime_and_preserves_it(tmp_path, monkeypatch):
    _no_legacy_cache(monkeypatch, tmp_path)
    downloads = tmp_path / 'Downloads'
    applied = tmp_path / 'applied_jobs'
    applied_date = date(2026, 5, 13)
    source = _write_pdf(downloads / 'cat-saved_jd-acme_job.pdf', applied_date)
    source_mtime = source.stat().st_mtime

    moved = await tools.ingest_downloads_applied_pdfs(downloads_dir=downloads, applied_dir=applied)

    assert moved == 1
    assert not source.exists(), 'source PDF should have moved, not been copied'
    destination = applied / '2026-05-13-cat-saved_jd-acme_job.pdf'
    assert destination.exists()
    assert abs(destination.stat().st_mtime - source_mtime) < 1.0, 'mtime should survive the move'


async def test_ingest_rolls_back_partial_move(tmp_path, monkeypatch):
    """A copy that succeeds but whose source unlink fails must not leave a duplicate behind."""
    _no_legacy_cache(monkeypatch, tmp_path)
    downloads = tmp_path / 'Downloads'
    applied = tmp_path / 'applied_jobs'
    source = _write_pdf(downloads / 'cat-saved_jd-acme_job.pdf', date(2026, 5, 13))

    def fake_move(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes())  # copy succeeds, source stays
        raise PermissionError('Operation not permitted')

    monkeypatch.setattr(tools.shutil, 'move', fake_move)

    moved = await tools.ingest_downloads_applied_pdfs(downloads_dir=downloads, applied_dir=applied)

    assert moved == 0
    assert source.exists(), 'source must remain when the move failed'
    assert not (applied / '2026-05-13-cat-saved_jd-acme_job.pdf').exists(), 'partial copy rolled back'


async def test_ingest_dry_run_moves_nothing(tmp_path, monkeypatch):
    _no_legacy_cache(monkeypatch, tmp_path)
    downloads = tmp_path / 'Downloads'
    applied = tmp_path / 'applied_jobs'
    source = _write_pdf(downloads / 'cat-saved_jd-acme_job.pdf', date(2026, 5, 13))

    moved = await tools.ingest_downloads_applied_pdfs(
        downloads_dir=downloads, applied_dir=applied, dry_run=True
    )

    assert moved == 1
    assert source.exists()
    assert not applied.exists()


async def test_ingest_is_idempotent_for_already_prefixed_file(tmp_path, monkeypatch):
    _no_legacy_cache(monkeypatch, tmp_path)
    downloads = tmp_path / 'Downloads'
    applied = tmp_path / 'applied_jobs'
    # An already-ingested file re-downloaded into Downloads keeps its original date, and a
    # second pass must not re-date it to today.
    _write_pdf(downloads / '2026-04-13-cat-saved_jd-acme_job.pdf', date(2026, 7, 1))

    await tools.ingest_downloads_applied_pdfs(downloads_dir=downloads, applied_dir=applied)
    assert (applied / '2026-04-13-cat-saved_jd-acme_job.pdf').exists()

    moved_again = await tools.ingest_downloads_applied_pdfs(downloads_dir=downloads, applied_dir=applied)
    assert moved_again == 0
    assert list(applied.glob('*.pdf')) == [applied / '2026-04-13-cat-saved_jd-acme_job.pdf']


async def test_ingest_prefers_index_date_over_clobbered_mtime(tmp_path, monkeypatch):
    """The whole point of the manifest: a rewritten mtime must not rewrite the applied date."""
    import yaml as yaml_mod
    _no_legacy_cache(monkeypatch, tmp_path)
    downloads = tmp_path / 'Downloads'
    applied = tmp_path / 'applied_jobs'
    applied.mkdir(parents=True)
    _write_pdf(downloads / 'cat-saved_jd-acme_job.pdf', date(2026, 7, 25))  # mtime says July
    (applied / 'index.yaml').write_text(yaml_mod.dump({
        'cat-saved_jd-acme_job.pdf': {'applied_date': date(2026, 4, 13)},  # index says April
    }))

    await tools.ingest_downloads_applied_pdfs(downloads_dir=downloads, applied_dir=applied)

    assert (applied / '2026-04-13-cat-saved_jd-acme_job.pdf').exists()


async def test_ingest_falls_back_to_legacy_downloads_cache_mtime(tmp_path, monkeypatch):
    import yaml as yaml_mod
    downloads = tmp_path / 'Downloads'
    applied = tmp_path / 'applied_jobs'
    source = _write_pdf(downloads / 'cat-saved_jd-acme_job.pdf', date(2026, 7, 25))

    legacy = tmp_path / 'downloads_pdf_cache.yaml'
    legacy_ts = datetime.combine(date(2026, 4, 13), datetime.min.time()).timestamp()
    legacy.write_text(yaml_mod.dump({str(source): {'mtime': legacy_ts, 'company': 'Acme'}}))
    monkeypatch.setattr(tools, 'LEGACY_DOWNLOADS_CACHE_PATH', legacy)

    await tools.ingest_downloads_applied_pdfs(downloads_dir=downloads, applied_dir=applied)

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

    await tools.load_applied_jobs(applied_dir=applied, index_path=index_path)

    assert extract_called == [], 'LLM should not be called on an index hit'
    assert 'CachedCorp' in tools._applied_companies
    assert tools._reference_job_texts == ['job description text']
    assert len(tools._applied_jobs) == 1


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
    await tools.load_applied_jobs(applied_dir=applied, index_path=index_path)

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

    await tools.load_applied_jobs(applied_dir=applied, index_path=index_path)

    assert stale.exists(), 'aged-out PDFs are kept on disk, only excluded from use'
    assert tools._applied_companies == {'NewCorp': fresh.name}
    assert tools._reference_job_texts == ['new text']
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

    await tools.load_applied_jobs(applied_dir=applied, index_path=index_path)

    assert tools._applied_companies == {}, 'an agency name must never block its other postings'
    assert tools._reference_job_texts == ['job description text'], 'still useful as reference signal'


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

    await tools.load_applied_jobs(applied_dir=applied, index_path=index_path)

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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', [])
    assert agent.build_reference_block() == ''


def test_build_reference_block_includes_text(monkeypatch):
    from agentic_job_search import agent
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', ['This is a great remote job at Acme.'])
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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', [f'job text {i}' for i in range(n)])
    block = agent.build_reference_block()
    assert block.count('[Reference Job') == MAX_REFERENCE_JOBS


# ---------------------------------------------------------------------------
# _categorize_pdf_text and categorize_downloads_pdfs
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


async def test_categorize_downloads_pdfs_renames_uncategorized(tmp_path, monkeypatch):
    pdf = tmp_path / 'report.pdf'
    pdf.write_bytes(b'%PDF fake')

    async def fake_categorize(text, filename):
        return 'saved_jd'

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    class FakeReader:
        pages = []
        def __init__(self, path): pass

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)
    await tools.categorize_downloads_pdfs(downloads_dir=tmp_path)
    assert not pdf.exists()
    assert (tmp_path / 'cat-saved_jd-report.pdf').exists()


async def test_categorize_downloads_pdfs_skips_rename_on_error(tmp_path, monkeypatch):
    pdf = tmp_path / 'report.pdf'
    pdf.write_bytes(b'%PDF fake')

    async def fake_categorize(text, filename):
        return None  # simulate tool-not-called failure

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    class FakeReader:
        pages = []
        def __init__(self, path): pass

    monkeypatch.setattr(tools.pypdf, 'PdfReader', FakeReader)
    await tools.categorize_downloads_pdfs(downloads_dir=tmp_path)
    assert pdf.exists()  # original file untouched


async def test_categorize_downloads_pdfs_skips_already_categorized(tmp_path, monkeypatch):
    pdf = tmp_path / 'cat-other-old_report.pdf'
    pdf.write_bytes(b'%PDF fake')
    sdk_called = []

    async def fake_categorize(text, filename):
        sdk_called.append(True)
        return 'other'

    monkeypatch.setattr(tools, '_categorize_pdf_text', fake_categorize)
    await tools.categorize_downloads_pdfs(downloads_dir=tmp_path)
    assert not sdk_called
    assert pdf.exists()


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
    await agent.run_scraper(_RecordingClient({q: healthy for q in queries}), queries,
                            {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

    assert seen == ['Alpha', 'Beta'], 'each request runs with its own base query set'


def test_min_turns_floor_is_absolute_not_a_budget_fraction():
    """A fraction of the (worst-case) budget fired on every healthy query once harvesting replaced
    click-to-reveal, and a warning that fires on success gets ignored."""
    assert config.SCRAPER_MIN_TURNS_PER_QUERY >= 1
    assert config.SCRAPER_MIN_TURNS_PER_QUERY < config.SCRAPER_MAX_TURNS_PER_QUERY // 3, \
        'must sit below a healthy query\'s turn count, not scale with the budget'


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


# ---------------------------------------------------------------------------
# apply_hard_rules
# ---------------------------------------------------------------------------

def _make_extract(**overrides) -> dict:
    extract = {
        'title': 'Staff AI Engineer', 'company': 'Acme',
        'description': 'Build agentic AI systems in Python. Remote within Canada.',
        'location': 'Canada (Remote)', 'date_posted': '3 days ago',
        'closed': False, 'salary': '', 'sponsorship_note': '',
        'language_requirement': '', 'relocation': '', 'education_requirement': '',
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


def test_apply_hard_rules_passes_good_job():
    assert agent.apply_hard_rules(_make_candidate(), _make_extract()) is None


def test_apply_hard_rules_closed_flag():
    reason = agent.apply_hard_rules(_make_candidate(), _make_extract(closed=True))
    assert reason is not None and 'closed' in reason


def test_apply_hard_rules_closed_text():
    extract = _make_extract(description='This job is no longer accepting applications.')
    reason = agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'closed' in reason


def test_apply_hard_rules_stale_posting():
    old = (date.today() - timedelta(days=45)).isoformat()
    reason = agent.apply_hard_rules(_make_candidate(), _make_extract(date_posted=old))
    assert reason is not None and 'older than' in reason


def test_apply_hard_rules_recent_posting_ok():
    recent = (date.today() - timedelta(days=5)).isoformat()
    assert agent.apply_hard_rules(_make_candidate(), _make_extract(date_posted=recent)) is None


def test_apply_hard_rules_us_no_sponsorship():
    extract = _make_extract(location='United States (Remote)')
    reason = agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'sponsorship' in reason


def test_apply_hard_rules_us_with_sponsorship_passes():
    extract = _make_extract(
        location='United States (Remote)',
        sponsorship_note='We are willing to sponsor H-1B visas.',
    )
    assert agent.apply_hard_rules(_make_candidate(), extract) is None


def test_apply_hard_rules_explicit_no_auth_statement():
    extract = _make_extract(description='Must be authorized to work in the US without sponsorship.')
    reason = agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'sponsorship' in reason


def test_apply_hard_rules_non_english_language_requirement():
    extract = _make_extract(language_requirement='dutch')
    reason = agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'dutch' in reason and 'language' in reason


def test_apply_hard_rules_english_plus_other_language_rejects():
    extract = _make_extract(language_requirement='english, ukrainian')
    reason = agent.apply_hard_rules(_make_candidate(), extract)
    assert reason is not None and 'ukrainian' in reason


def test_apply_hard_rules_english_only_requirement_passes():
    assert agent.apply_hard_rules(_make_candidate(), _make_extract(language_requirement='english')) is None
    assert agent.apply_hard_rules(_make_candidate(), _make_extract(language_requirement='english (b2)')) is None


def test_apply_hard_rules_relocation_does_not_reject():
    extract = _make_extract(relocation='Portugal')
    assert agent.apply_hard_rules(_make_candidate(), extract) is None


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

def test_apply_hard_rules_explicit_phd_requirement():
    reason = agent.apply_hard_rules(_make_candidate(), _make_extract(education_requirement='phd'))
    assert reason is not None and 'degree' in reason and 'phd' in reason


def test_apply_hard_rules_explicit_masters_requirement():
    reason = agent.apply_hard_rules(_make_candidate(), _make_extract(education_requirement='master'))
    assert reason is not None and 'degree' in reason and 'master' in reason


def test_apply_hard_rules_empty_education_requirement_passes():
    assert agent.apply_hard_rules(_make_candidate(), _make_extract()) is None


def test_apply_hard_rules_bachelors_requirement_passes():
    extract = _make_extract(education_requirement='bachelor')
    assert agent.apply_hard_rules(_make_candidate(), extract) is None


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


def test_apply_hard_rules_derives_degree_from_description():
    extract = _make_extract(description='Build agents in Python. A PhD in AI is required.')
    reason = agent.apply_hard_rules(_make_candidate(), extract)
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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', ['Great AI job at Acme.'])
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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', ['Great AI job at Acme.'])
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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', ['Great AI job at Acme.'])
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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', ['Great AI job at Acme.'])
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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', ['Great AI job at Acme.'])
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
    monkeypatch.setattr(agent.tools_module, '_reference_job_texts', [])
    assert await agent.build_reference_summary() == ''


# ---------------------------------------------------------------------------
# rate_job provider dispatch
# ---------------------------------------------------------------------------

async def test_rate_job_dispatches_to_openrouter(monkeypatch):
    monkeypatch.setattr(agent, 'RATING_PROVIDER', 'openrouter')

    async def fake_rate(system_prompt, user_prompt, model=''):
        return {'rating': 4, 'company': 'Acme', 'title': 'Engineer', 'reasoning': 'good', 'summary': 'acme_engineer'}, 0.002

    monkeypatch.setattr(agent, 'rate_with_openrouter', fake_rate)
    stats = agent.new_stage_stats()
    result = await agent.rate_job('system', 'job text', stats)

    assert result['rating'] == 4
    assert stats['cost'] == pytest.approx(0.002)


async def test_rate_job_dispatches_to_ollama(monkeypatch):
    monkeypatch.setattr(agent, 'RATING_PROVIDER', 'ollama')

    async def fake_rate(system_prompt, user_prompt, model=''):
        return {'rating': 3, 'company': 'Acme', 'title': 'Engineer', 'reasoning': 'ok', 'summary': 'acme_engineer'}

    monkeypatch.setattr(agent, 'rate_with_ollama', fake_rate)
    stats = agent.new_stage_stats()
    result = await agent.rate_job('system', 'job text', stats)

    assert result['rating'] == 3
    assert stats['cost'] == 0.0


async def test_rate_job_dispatches_to_anthropic_by_default(monkeypatch):
    monkeypatch.setattr(agent, 'RATING_PROVIDER', 'anthropic')

    async def fake_anthropic(evaluator_prompt, extract_text, stage_stats):
        return {'rating': 5, 'company': 'Acme', 'title': 'Engineer', 'reasoning': 'great', 'summary': 'acme_engineer'}

    monkeypatch.setattr(agent, '_rate_with_anthropic', fake_anthropic)
    result = await agent.rate_job('system', 'job text', agent.new_stage_stats())
    assert result['rating'] == 5


# ---------------------------------------------------------------------------
# extract_job_page_direct (deterministic fallback)
# ---------------------------------------------------------------------------

def _make_agent_sdk_mock(monkeypatch, structured_output: dict):
    from claude_agent_sdk import ResultMessage

    async def fake_sdk_query(**kwargs):
        yield ResultMessage(
            subtype='success', duration_ms=100, duration_api_ms=100, is_error=False,
            num_turns=1, session_id='fake-session', structured_output=structured_output,
        )

    monkeypatch.setattr(agent, 'sdk_query', fake_sdk_query)


def _make_mcp_session_mock(monkeypatch, snapshot_text: str, mcp_calls: list):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_mcp_session(url):
        async def call(tool_name, args):
            mcp_calls.append((tool_name, args.get('target')))
            return snapshot_text

        yield call

    monkeypatch.setattr(agent, 'mcp_session', fake_mcp_session)


async def test_extract_job_page_direct_condenses_snapshot(monkeypatch):
    mcp_calls = []
    _make_mcp_session_mock(monkeypatch, 'heading "Staff Engineer" text "Build agents." (no expand button)', mcp_calls)
    _make_agent_sdk_mock(monkeypatch, {'title': 'Staff Engineer', 'company': 'Acme', 'description': 'Build agents.'})

    extract = await agent.extract_job_page_direct(_make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats())

    assert [c[0] for c in mcp_calls] == ['browser_navigate', 'browser_wait_for', 'browser_snapshot']
    assert extract['title'] == 'Staff Engineer'
    assert extract['closed'] is False  # default filled
    assert extract['location'] == ''


async def test_extract_job_page_direct_clicks_more_button(monkeypatch):
    mcp_calls = []
    _make_mcp_session_mock(monkeypatch, 'text "intro" button "… more" [ref=e611] text "rest"', mcp_calls)
    _make_agent_sdk_mock(monkeypatch, {'title': 'T', 'company': 'C', 'description': 'D'})

    await agent.extract_job_page_direct(_make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats())

    assert mcp_calls == [
        ('browser_navigate', None), ('browser_wait_for', None), ('browser_snapshot', None),
        ('browser_click', 'e611'), ('browser_wait_for', None), ('browser_snapshot', None),
    ]


async def test_extract_job_page_direct_returns_none_without_output(monkeypatch):
    mcp_calls = []
    _make_mcp_session_mock(monkeypatch, 'snapshot text', mcp_calls)

    async def fake_sdk_query(**kwargs):
        return
        yield

    monkeypatch.setattr(agent, 'sdk_query', fake_sdk_query)

    extract = await agent.extract_job_page_direct(_make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats())
    assert extract is None


# ---------------------------------------------------------------------------
# extract_job_page_openrouter (function-calling agent loop)
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


def _wire_openrouter_loop(monkeypatch, chat_responses: list, browser_calls: list, chat_requests: list):
    """Wire fake chat responses and a browser-call recorder into the loop."""
    from contextlib import asynccontextmanager

    async def fake_call_mcp_tool(url, tool_name, args):
        # Deep-copy: the loop mutates its messages list in place between calls
        chat_requests.append(json.loads(json.dumps(args)))
        return chat_responses.pop(0)

    @asynccontextmanager
    async def fake_mcp_session(url):
        async def call(tool_name, args):
            browser_calls.append((tool_name, args))
            return f'snapshot of page after {tool_name}'

        yield call

    monkeypatch.setattr(extract_openrouter, 'call_mcp_tool', fake_call_mcp_tool)
    monkeypatch.setattr(extract_openrouter, 'mcp_session', fake_mcp_session)


async def test_openrouter_loop_happy_path(monkeypatch):
    browser_calls, chat_requests = [], []
    responses = [
        _chat_response(tool_calls=[
            _tool_call('c1', 'browser_navigate', {'url': 'https://example.com/job/123'}),
            _tool_call('c2', 'browser_snapshot', {}),
        ]),
        _chat_response(tool_calls=[_tool_call('c3', 'submit_job_extract', _SUBMIT_ARGS)]),
    ]
    _wire_openrouter_loop(monkeypatch, responses, browser_calls, chat_requests)

    stats = agent.new_stage_stats()
    extract = await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'http://localhost:1/mcp', stats, 'system prompt'
    )

    assert extract['title'] == 'Staff Engineer'
    assert extract['closed'] is False  # default filled
    assert [c[0] for c in browser_calls] == ['browser_navigate', 'browser_snapshot']
    assert stats['cost'] == pytest.approx(0.002)
    assert stats['input_tokens'] == 200
    # second request carries the assistant tool_calls turn and both tool results
    roles = [m['role'] for m in chat_requests[1]['messages']]
    assert roles == ['system', 'user', 'assistant', 'tool', 'tool']


async def test_openrouter_loop_invalid_json_args_retries(monkeypatch):
    browser_calls, chat_requests = [], []
    responses = [
        _chat_response(tool_calls=[_tool_call('c1', 'browser_navigate', '{not json')]),
        _chat_response(tool_calls=[_tool_call('c2', 'submit_job_extract', _SUBMIT_ARGS)]),
    ]
    _wire_openrouter_loop(monkeypatch, responses, browser_calls, chat_requests)

    extract = await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), 'sys'
    )

    assert extract is not None
    assert browser_calls == []  # bad call never executed
    error_msg = chat_requests[1]['messages'][-1]
    assert error_msg['role'] == 'tool'
    assert 'invalid JSON' in error_msg['content']


async def test_openrouter_loop_missing_submit_fields_retries(monkeypatch):
    browser_calls, chat_requests = [], []
    responses = [
        _chat_response(tool_calls=[_tool_call('c1', 'submit_job_extract', {'title': 'T'})]),
        _chat_response(tool_calls=[_tool_call('c2', 'submit_job_extract', _SUBMIT_ARGS)]),
    ]
    _wire_openrouter_loop(monkeypatch, responses, browser_calls, chat_requests)

    extract = await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), 'sys'
    )
    assert extract is not None
    assert 'missing required field' in chat_requests[1]['messages'][-1]['content']


async def test_openrouter_loop_stops_at_iteration_cap(monkeypatch):
    from agentic_job_search.config import EXTRACTOR_OPENROUTER_MAX_ITERATIONS
    browser_calls, chat_requests = [], []
    responses = [
        _chat_response(tool_calls=[_tool_call(f'c{i}', 'browser_snapshot', {})])
        for i in range(EXTRACTOR_OPENROUTER_MAX_ITERATIONS + 5)
    ]
    _wire_openrouter_loop(monkeypatch, responses, browser_calls, chat_requests)

    extract = await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), 'sys'
    )
    assert extract is None
    assert len(chat_requests) == EXTRACTOR_OPENROUTER_MAX_ITERATIONS


async def test_openrouter_loop_returns_none_when_model_stops_without_submit(monkeypatch):
    browser_calls, chat_requests = [], []
    responses = [_chat_response(content='I could not find the job posting.')]
    _wire_openrouter_loop(monkeypatch, responses, browser_calls, chat_requests)

    extract = await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), 'sys'
    )
    assert extract is None


async def test_openrouter_loop_returns_none_on_chat_error(monkeypatch):
    browser_calls, chat_requests = [], []
    responses = [_chat_response(ok=False, error='server down')]
    _wire_openrouter_loop(monkeypatch, responses, browser_calls, chat_requests)

    extract = await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), 'sys'
    )
    assert extract is None


async def test_openrouter_loop_truncates_tool_results(monkeypatch):
    from contextlib import asynccontextmanager
    from agentic_job_search.config import EXTRACTOR_TOOL_RESULT_MAX_CHARS
    chat_requests = []
    responses = [
        _chat_response(tool_calls=[_tool_call('c1', 'browser_snapshot', {})]),
        _chat_response(tool_calls=[_tool_call('c2', 'submit_job_extract', _SUBMIT_ARGS)]),
    ]

    async def fake_call_mcp_tool(url, tool_name, args):
        chat_requests.append(json.loads(json.dumps(args)))
        return responses.pop(0)

    @asynccontextmanager
    async def fake_mcp_session(url):
        async def call(tool_name, args):
            return 'x' * (EXTRACTOR_TOOL_RESULT_MAX_CHARS * 3)

        yield call

    monkeypatch.setattr(extract_openrouter, 'call_mcp_tool', fake_call_mcp_tool)
    monkeypatch.setattr(extract_openrouter, 'mcp_session', fake_mcp_session)

    await extract_openrouter.extract_job_page_openrouter(
        _make_candidate(), 'http://localhost:1/mcp', agent.new_stage_stats(), 'sys'
    )
    tool_msg = chat_requests[1]['messages'][-1]
    assert len(tool_msg['content']) == EXTRACTOR_TOOL_RESULT_MAX_CHARS


async def test_extractor_provider_dispatch_openrouter(monkeypatch):
    monkeypatch.setattr(agent, 'EXTRACTOR_PROVIDER', 'openrouter')
    called = {'openrouter': 0, 'anthropic': 0, 'direct': 0}

    async def fake_openrouter(candidate, url, stats, system_prompt):
        called['openrouter'] += 1
        return None

    async def fake_anthropic(candidate, mcp, stats):
        called['anthropic'] += 1
        return None

    async def fake_direct(candidate, url, stats):
        called['direct'] += 1
        return None

    monkeypatch.setattr(agent, 'extract_job_page_openrouter', fake_openrouter)
    monkeypatch.setattr(agent, 'extract_job_page', fake_anthropic)
    monkeypatch.setattr(agent, 'extract_job_page_direct', fake_direct)

    stats = {'extraction': agent.new_stage_stats(), 'rating': agent.new_stage_stats()}
    await agent.evaluate_all_candidates(
        [_make_candidate()], {'type': 'http', 'url': 'http://localhost:1/mcp'}, 'prompt', 'profile', stats
    )

    assert called == {'openrouter': 1, 'anthropic': 0, 'direct': 1}  # fallback still fires


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
    response = await triage.generate_local('Reply with exactly: OK', model='qwen3.6:latest', max_tokens=500)
    assert response.strip() != ''


@pytest.mark.live
async def test_chat_openrouter_live():
    _require_llm_server(8006)
    content, cost_usd = await triage.chat_openrouter('Reply with exactly: OK', max_tokens=500)
    assert content.strip() != ''
    assert cost_usd >= 0


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
# Live tests for Downloads PDF loading
# ---------------------------------------------------------------------------

@pytest.mark.live_agent_claude
async def test_extract_company_from_text_live():
    text = '''
    Software Engineer — Remote
    Shopify
    We are looking for an experienced software engineer to join our team.
    You will work on our e-commerce platform serving millions of merchants.
    Requirements: 5+ years Python, strong distributed systems knowledge.
    '''
    company = await tools._extract_company_from_text(text)
    assert company.strip() != ''
    assert 'shopify' in company.lower()


@pytest.mark.live_agent_claude
async def test_categorize_pdf_text_live_saved_jd():
    text = '''
    Principal AI Engineer — Remote Canada
    Dayforce
    We are hiring a Principal AI Engineer to lead our machine learning platform.
    Responsibilities: Design agentic AI systems, lead a team of ML engineers,
    build RAG pipelines and LLM evaluation frameworks.
    Requirements: 10+ years experience, Python, deep learning expertise.
    Salary: CAD $220,000 base + equity.
    '''
    category = await tools._categorize_pdf_text(text, 'Principal AI Engineer _ Dayforce Jobs.pdf')
    assert category is not None, 'tool was not called'
    assert category == 'saved_jd', f'expected saved_jd, got {category!r}'


@pytest.mark.live_agent_claude
async def test_categorize_pdf_text_live_other():
    text = '''
    Your Airbnb booking confirmation
    Check-in: June 12, 2026
    Check-out: June 15, 2026
    Property: Cozy cabin in Whistler
    Total: $450 CAD
    '''
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
    monkeypatch.setattr(tools, 'COMPANY_MATCH_PROVIDER', 'openrouter')

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
        ('linkedin', '5'): _listing('5', 'GoodJob', status='new', queued=True, rating=5),
        ('linkedin', '6'): _listing('6', 'Dupe', status='already_processed', queued=False, outcome='already_processed'),
    })
    pools = tools.unsurfaced_pools()

    assert {r['company'] for r in pools['filtered']} == {'TooOld', 'Applied'}
    assert {r['company'] for r in pools['never_queued']} == {'NeverQueued'}
    assert {r['company'] for r in pools['mid_rated']} == {'MidRated'}
    assert 'GoodJob' not in {r['company'] for pool in pools.values() for r in pool}


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
    monkeypatch.setattr(agent, 'QUERY_PROVIDER', 'openrouter')
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
    monkeypatch.setattr(agent, 'QUERY_PROVIDER', 'openrouter')
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
    monkeypatch.setattr(agent, 'QUERY_PROVIDER', 'openrouter')
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

class _FakeScraperClient:
    """Records each query() call so we can assert on request boundaries."""

    def __init__(self, listings_per_query=None):
        self.requests = []
        self.listings_per_query = listings_per_query or {}

    async def query(self, instruction):
        self.requests.append(instruction)
        # Simulate the scraper inspecting listings for whichever query this request names.
        for name, count in self.listings_per_query.items():
            if f'"{name}"' in instruction:
                for i in range(count):
                    key = f'{name}-{len(tools._check_status_counts)}-{i}'
                    tools._check_status_counts[key] = 1
                break

    async def receive_response(self):
        return
        yield


async def test_run_scraper_sends_one_request_per_query(monkeypatch):
    """Each query needs its own turn budget; a shared request starves the later ones."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    queries = ['Principal AI Engineer', 'Staff AI Engineer', 'Lead AI Engineer']
    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    client = _FakeScraperClient({q: healthy for q in queries})

    await agent.run_scraper(client, queries, {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

    assert len(client.requests) == 3, 'one request per query, no retries at a healthy yield'
    for q in queries:
        assert any(f'"{q}"' in r for r in client.requests), f'{q} was never searched'
        assert tools._queries_searched[q] == healthy


async def test_run_scraper_retries_empty_query_then_continues(monkeypatch):
    """A query that comes back empty gets one retry, and must not abort the remaining queries."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    queries = ['Empty Query', 'Good Query']
    healthy = agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20
    client = _FakeScraperClient({'Good Query': healthy})

    await agent.run_scraper(client, queries, {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

    empty_requests = [r for r in client.requests if 'Empty' in r]
    assert len(empty_requests) == 2, 'empty query should be retried once'
    retry = empty_requests[1]
    assert 'NO region words' in retry, 'retry drops the region text'
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
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    # 0 listings and 1 listing exercise both recovery branches; a healthy query the normal path.
    client = _FakeScraperClient({'Healthy': agent.SCRAPER_MIN_LISTINGS_PER_QUERY + 20, 'Starved': 1})
    await agent.run_scraper(
        client, ['Empty', 'Starved', 'Healthy'],
        {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0},
    )

    prompts = client.requests + [agent.build_scraper_instructions()]
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
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_queries_searched', {})
    monkeypatch.setattr(tools, '_check_status_per_query', {})

    queries = ['Starved Query']
    client = _FakeScraperClient({'Starved Query': 1})

    await agent.run_scraper(client, queries, {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

    assert len(client.requests) == 2, 'a 1-listing query must be retried'
    retry = client.requests[1]
    assert 'inspected only 1 job listing' in retry
    assert 'identical list of jobs' in retry, 'retry names the repeated-results failure mode'
    # Unlike the zero-listing case, the page loaded — filters must be re-applied, not dropped.
    assert 'clear the Location filter' not in retry


async def test_run_scraper_records_per_query_check_status(monkeypatch):
    """Per-query dedup counts: a saturated query must be distinguishable from a starved one."""
    monkeypatch.setattr(tools, '_check_status_counts', {})
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

        async def receive_response(self):
            return
            yield

    await agent.run_scraper(_StatusClient(), ['Saturated'], {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

    assert tools._check_status_per_query['Saturated'] == {'already_processed': 25, 'new': 3}
    assert tools._queries_searched['Saturated'] == 28


async def test_run_scraper_survives_a_failing_query(monkeypatch):
    monkeypatch.setattr(tools, '_check_status_counts', {})
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
    await agent.run_scraper(client, ['Bad Query', 'Good Query'], {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

    assert tools._queries_searched['Bad Query'] == 'error'
    assert tools._queries_searched['Good Query'] == healthy, 'a failed query must not abort the rest'
    assert 'Bad Query' not in tools._check_status_per_query, 'an errored query records no counts'


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

    assert lock.exists(), "must never release a lock we do not hold"


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


def test_scraper_instructions_cover_every_configured_region_as_query_text():
    """Region is now the only axis that varies the result set: the AI-powered UI has no sort
    control, so the former 'each region in both sort orders' would run the same search twice."""
    text = agent.build_scraper_instructions()
    for region in ('Testland', 'Test Union'):  # pinned in tests/conftest.py
        assert f'**{region}** — search text:' in text
        assert f'{region}, senior level' in text, 'region goes in the query text, not a filter'
    assert 'as 2 searches' in text, 'one search per region'


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
    text = agent.build_scraper_instructions()
    assert 'SearchResultsMainContent' in text, 'the results container selector must be given'
    assert 'job-card-component-ref-' in text, 'the id-bearing attribute must be given'
    assert 'browser_evaluate' in text


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
    client = _FakeScraperClient({q: healthy for q in queries})
    await agent.run_scraper(client, queries, {'cost': 0.0, 'input_tokens': 0, 'output_tokens': 0})

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
    assert agent.hybrid_location_is_acceptable('Exampleton, Testland (Hybrid)')
    assert agent.hybrid_location_is_acceptable('Testville, Testland')
    assert not agent.hybrid_location_is_acceptable('Netherlands (Hybrid - 2-3 days onsite)')
    assert not agent.hybrid_location_is_acceptable('')


def test_hybrid_location_unacceptable_when_no_locations_configured(monkeypatch):
    monkeypatch.setattr(preferences, 'hybrid_acceptable_locations', lambda: ())
    assert not agent.hybrid_location_is_acceptable('Exampleton, Testland (Hybrid)')


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
    assert any('Hybrid' in w and 'not an acceptable hybrid location' in w for w in warnings)
    assert any('Contract role' in w for w in warnings)
    assert 'No salary listed' in warnings


def test_build_deterministic_warnings_quiet_for_clean_remote_job():
    extract = _make_extract(salary='CAD 220,000')
    assert agent.build_deterministic_warnings(_make_candidate(), extract) == []


def test_build_deterministic_warnings_includes_relocation():
    extract = _make_extract(relocation='Portugal', salary='EUR 100,000')
    warnings = agent.build_deterministic_warnings(_make_candidate(), extract)
    assert 'Relocation required: Portugal' in warnings


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
    extract = _make_extract(salary='CAD 220,000')
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


def test_agency_posting_is_not_hard_ruled():
    """Agency status must never reject a job -- only warn. An auto-reject is unappealable."""
    extract = _make_extract(company='Quik Hire Staffing', is_agency=True, end_client='',
                            description='Our client is a leading AI lab. Build agents in Python.')
    assert agent.apply_hard_rules(_make_candidate(), extract) is None


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
    for field in ('is_agency', 'end_client'):
        assert field in anthropic_schema and field in openrouter_schema


def test_merge_warnings_dedupes_case_insensitively_and_keeps_order():
    merged = agent.merge_warnings(['Hybrid — Netherlands'], ['hybrid — netherlands', 'Below salary target'])
    assert merged == ['Hybrid — Netherlands', 'Below salary target']


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


def test_devologyx_regression_end_to_end():
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
    assert agent.apply_hard_rules(candidate, extract) is None  # not a hard reject — it is a cap

    rating, cap_reason = agent.apply_rating_caps(extract, 4)
    assert rating == 3
    assert rating < 4, 'a capped job must fall below the >=4 notification threshold'
    assert 'hybrid' in cap_reason

    warnings = agent.merge_warnings(
        agent.build_deterministic_warnings(candidate, extract), ['Contract, not full-time'],
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
    assert preferences.hybrid_acceptable_locations() == ()
    assert preferences.search_regions() == []


def test_hard_rules_reject_nothing_without_preferences(neutral_preferences):
    """The three personal gates must be inert when unconfigured — not silently inherited."""
    extract = _make_extract(
        location='United States (Remote)',
        description='PhD in Machine Learning is required. Fluent Dutch is required.',
        language_requirement='dutch',
        education_requirement='phd',
    )
    assert agent.apply_hard_rules(_make_candidate(), extract) is None


def test_hard_rules_still_reject_impersonal_conditions(neutral_preferences):
    """Closed and stale are properties of the posting, not the person — always on."""
    assert 'closed' in agent.apply_hard_rules(_make_candidate(), _make_extract(closed=True))
    old = (date.today() - timedelta(days=45)).isoformat()
    assert 'older than' in agent.apply_hard_rules(_make_candidate(), _make_extract(date_posted=old))


def test_prompts_omit_personal_sections_without_preferences(neutral_preferences):
    evaluator = agent.build_evaluator_instructions()
    assert 'visa sponsorship' not in evaluator
    assert 'No hybrid/on-site location is acceptable' in evaluator
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
    assert loaded['hybrid']['acceptable_locations'] == []
    for region in loaded['search_regions']:
        assert 'Example' in region['name'] or 'Test' in region['name']
