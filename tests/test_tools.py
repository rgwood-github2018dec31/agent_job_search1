"""Tests for the job search agent."""

import os
import pytest
from datetime import datetime
from pathlib import Path

from agentic_job_search import agent
from agentic_job_search import tools_generic as tools


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
# do_notify_user
# ---------------------------------------------------------------------------

async def test_notify_user_calls_send_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(tools, "send_telegram", lambda msg: sent.append(msg))

    result = await tools.do_notify_user("Found a match: Senior Engineer @ Shopify")

    assert not result.get("isError")
    assert sent == ["Found a match: Senior Engineer @ Shopify"]


async def test_notify_user_returns_error_on_failure(monkeypatch):
    def boom(msg):
        raise RuntimeError("network error")

    monkeypatch.setattr(tools, "send_telegram", boom)

    result = await tools.do_notify_user("hello")

    assert result.get("isError")
    assert "network error" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# do_check_and_record_job
# ---------------------------------------------------------------------------

from datetime import date, timedelta


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
# Live Telegram test (skipped — run manually with: pytest -p no:skip -k test_send_telegram_live)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="live test — run manually to verify Telegram integration")
async def test_send_telegram_live():
    """Sends a real Telegram message. Run once to verify credentials work."""
    agent.load_env()
    tools.send_telegram("test message from agentic_job_search test suite")


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
# load_downloads_applied_pdfs
# ---------------------------------------------------------------------------

async def test_load_downloads_applied_pdfs_ignores_uncategorized(tmp_path, monkeypatch):
    fake_home = tmp_path / 'home'
    (fake_home / 'Downloads').mkdir(parents=True)
    (fake_home / 'Downloads' / 'uncategorized.pdf').write_bytes(b'%PDF fake')

    monkeypatch.setattr(tools, '_applied_companies', {})
    monkeypatch.setattr(tools, '_reference_job_texts', [])
    monkeypatch.setattr(Path, 'home', staticmethod(lambda: fake_home))

    await tools.load_downloads_applied_pdfs(cache_path=tmp_path / 'cache.yaml')

    assert tools._applied_companies == {}


async def test_load_downloads_applied_pdfs_uses_cache(tmp_path, monkeypatch):
    import yaml as yaml_mod
    fake_home = tmp_path / 'home'
    (fake_home / 'Downloads').mkdir(parents=True)
    pdf_path = fake_home / 'Downloads' / 'cat-saved_jd-acme_job.pdf'
    pdf_path.write_bytes(b'%PDF-1.4 fake')
    mtime = pdf_path.stat().st_mtime

    cache_path = tmp_path / 'cache.yaml'
    cache_path.write_text(yaml_mod.dump({
        str(pdf_path): {'mtime': mtime, 'company': 'CachedCorp', 'text': 'job description text'}
    }))

    extract_called = []

    async def fake_extract(text):
        extract_called.append(text)
        return 'ShouldNotBeCalled'

    monkeypatch.setattr(tools, '_extract_company_from_text', fake_extract)
    monkeypatch.setattr(Path, 'home', staticmethod(lambda: fake_home))

    await tools.load_downloads_applied_pdfs(cache_path=cache_path)

    assert extract_called == [], 'LLM should not be called on cache hit'
    assert 'CachedCorp' in tools._applied_companies


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
