"""Tests for the job search agent."""

import pytest

import main


# ---------------------------------------------------------------------------
# underscorify
# ---------------------------------------------------------------------------

def test_underscorify_basic():
    assert main.underscorify("Hello World") == "hello_world"


def test_underscorify_special_chars():
    assert main.underscorify("Shopify Inc.") == "shopify_inc"


def test_underscorify_consecutive_separators():
    assert main.underscorify("Senior  Software---Engineer") == "senior_software_engineer"


def test_underscorify_leading_trailing():
    assert main.underscorify("  leading and trailing  ") == "leading_and_trailing"


def test_underscorify_numbers():
    assert main.underscorify("GPT-4 Engineer") == "gpt_4_engineer"


# ---------------------------------------------------------------------------
# do_save_job_posting
# ---------------------------------------------------------------------------

async def test_save_job_posting_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)

    result = await main.do_save_job_posting(
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
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)

    result = await main.do_save_job_posting(
        company="Acme",
        description="Engineer",
        rating=2,
        content="content",
    )

    text = result["content"][0]["text"]
    assert "saved_jobs-" in text
    assert "job_posting-noid-rating_2-acme-engineer-" in text


async def test_save_job_posting_creates_daily_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)

    for rating in (3, 5):
        await main.do_save_job_posting(
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
    monkeypatch.setattr(main, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    content = "# Requirements\n\n- Remote only\n- Canada or EU\n"
    result = await main.do_update_job_requirements(content)

    assert not result.get("isError")
    assert (tmp_path / "JOB_REQUIREMENTS.md").read_text() == content


async def test_update_job_requirements_returns_content(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    content = "# Requirements\n\n- Senior only\n"
    result = await main.do_update_job_requirements(content)

    # New contents must be in the tool result so the agent has them in context
    assert content in result["content"][0]["text"]


async def test_update_job_requirements_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "JOB_REQUIREMENTS.md"
    monkeypatch.setattr(main, "JOB_REQUIREMENTS_PATH", path)

    await main.do_update_job_requirements("first version")
    await main.do_update_job_requirements("second version")

    assert path.read_text() == "second version"


# ---------------------------------------------------------------------------
# do_notify_user
# ---------------------------------------------------------------------------

async def test_notify_user_calls_send_telegram(monkeypatch):
    sent = []
    monkeypatch.setattr(main, "send_telegram", lambda msg: sent.append(msg))

    result = await main.do_notify_user("Found a match: Senior Engineer @ Shopify")

    assert not result.get("isError")
    assert sent == ["Found a match: Senior Engineer @ Shopify"]


async def test_notify_user_returns_error_on_failure(monkeypatch):
    def boom(msg):
        raise RuntimeError("network error")

    monkeypatch.setattr(main, "send_telegram", boom)

    result = await main.do_notify_user("hello")

    assert result.get("isError")
    assert "network error" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# load_reviewed_job_ids
# ---------------------------------------------------------------------------

def test_load_reviewed_job_ids_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    assert main.load_reviewed_job_ids() == set()


async def test_load_reviewed_job_ids_from_saved_files(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)

    await main.do_save_job_posting("Shopify", "Engineer", 4, "content", job_id="3859234876")
    await main.do_save_job_posting("Acme", "Designer", 2, "content", job_id="1122334455")

    ids = main.load_reviewed_job_ids()
    assert ids == {"3859234876", "1122334455"}


async def test_load_reviewed_job_ids_excludes_noid(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)

    await main.do_save_job_posting("Corp", "Role", 3, "content", job_id=None)

    assert main.load_reviewed_job_ids() == set()


async def test_save_job_posting_filename_contains_job_id_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)

    await main.do_save_job_posting("Stripe", "Staff Engineer", 5, "content", job_id="9876543210")

    files = list(tmp_path.glob("saved_jobs-*/job_posting-9876543210-rating_5-stripe-staff_engineer-*.md"))
    assert len(files) == 1
    # Timestamp suffix should be a numeric string
    stem = files[0].stem  # e.g. job_posting-9876543210-rating_5-stripe-staff_engineer-1744123456
    timestamp_part = stem.split("-")[-1]
    assert timestamp_part.isdigit()


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

def test_build_system_prompt_includes_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(main, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    (tmp_path / "R_Garth_Wood-resume-2026Apr08v1.md").write_text("# Garth Wood\n\nExperienced engineer.")

    prompt = main.build_system_prompt()
    assert "Garth Wood" in prompt
    assert "RESUME" in prompt


def test_build_system_prompt_includes_job_requirements(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    req_path = tmp_path / "JOB_REQUIREMENTS.md"
    monkeypatch.setattr(main, "JOB_REQUIREMENTS_PATH", req_path)

    req_path.write_text("# Requirements\n\n- Remote only")

    prompt = main.build_system_prompt()
    assert "Remote only" in prompt
    assert "JOB_REQUIREMENTS.md" in prompt


def test_build_system_prompt_warns_when_no_resume(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(main, "JOB_REQUIREMENTS_PATH", tmp_path / "JOB_REQUIREMENTS.md")

    main.build_system_prompt()

    assert "Warning" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Live Telegram test (skipped — run manually with: pytest -p no:skip -k test_send_telegram_live)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="live test — run manually to verify Telegram integration")
async def test_send_telegram_live():
    """Sends a real Telegram message. Run once to verify credentials work."""
    main.load_env()
    main.send_telegram("test message from agent_job_search1 test suite")


# ---------------------------------------------------------------------------
# Live agent tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.live_agent_claude
async def test_live_agent_calls_save_job_posting(tmp_path, monkeypatch):
    """Agent uses the save_job_posting tool when instructed to save a job."""
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)

    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a job search assistant. "
            "When asked to save a job, use the save_job_posting tool exactly once."
        ),
        mcp_servers={"job_search": main.job_search_server},
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
