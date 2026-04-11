# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a personal autonomous job search agent built on the Claude Agent SDK. It:
1. Reads the user's resume
2. Searches LinkedIn for relevant job postings
3. In interactive mode: presents jobs to the user for feedback and refines job requirements over time
4. In non-interactive mode: runs autonomously, notifying the user of any jobs rated 4 or 5
5. [In a future version] Autonomously applies to jobs via email (with cover letter + resume), web forms, or LinkedIn

## Modes

### Interactive (default)
The user is at the console providing feedback. The agent presents jobs one at a time, collects yes/no feedback and reasons, and updates `JOB_REQUIREMENTS.md` as preferences are learned.

```bash
python main.py
```

### Non-interactive
Designed for periodic/scheduled runs (e.g. cron). The agent searches LinkedIn autonomously, rates all jobs, and sends Telegram notifications for any rated 4 or 5. `JOB_REQUIREMENTS.md` is never modified. The `update_job_requirements` tool is not available in this mode.

```bash
python main.py --non-interactive   # or: python main.py -n
```

## Commands

```bash
uv sync                  # Install dependencies
uv sync --group test     # Install test dependencies
uv lock --upgrade && uv sync  # Update all dependencies to latest versions
python main.py           # Run in interactive mode
python main.py -n        # Run in non-interactive mode

# Tests
uv run pytest                          # Run all tests (skips live tests if no API key)
uv run pytest -m "not live_agent_claude"  # Unit tests only
uv run pytest -m live_agent_claude     # Live agent tests (requires ANTHROPIC_API_KEY)
```

## Testing

Run `uv run --group test pytest` after most code changes to catch regressions. The test suite is fast and covers all core tool logic.

## Architecture

The project uses the **[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)** (`claude-agent-sdk>=0.1.56`) as its foundation:

- **MCP (Model Context Protocol)** — enables tool use for web browsing, LinkedIn interaction, and email sending
- **anyio** — async runtime for periodic background tasks (e.g., scheduled LinkedIn searches)

Core workflow:
1. Resume ingestion — agent reads and understands the user's resume
2. Periodic LinkedIn search — agent scrapes/searches for matching job listings
3. User preference loop — agent surfaces jobs to the user and records approval/rejection to refine its model of desired roles
4. Application — once a job is approved, agent drafts cover letter and applies via email, web form, or LinkedIn

The entry point is `main.py`. All agent logic will live here or in modules imported by it.

## Job deduplication and age filtering

On startup, `load_reviewed_job_ids()` scans all `saved_jobs-*/job_posting-*.md` files and extracts LinkedIn job IDs from their filenames. These IDs are injected into the system prompt as an "ALREADY REVIEWED" list.

The agent is instructed to:
- **Skip any job ID in the ALREADY REVIEWED list** — prevents re-processing jobs from previous runs
- **Skip any job posted more than 3 weeks ago** — today's date is injected into the system prompt so the agent can compute the cutoff accurately

Both rules are enforced at the prompt level since the agent reads job metadata (IDs, posting dates) while browsing LinkedIn via Playwright.
