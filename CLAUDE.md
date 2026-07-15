# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a personal autonomous job search agent built on the Claude Agent SDK. It:
1. Reads the user's resume
2. Searches LinkedIn for relevant job postings
3. In interactive mode: presents jobs to the user for feedback and refines job requirements over time
4. In non-interactive mode: runs autonomously, notifying the user of any jobs rated 4 or 5

## Modes

### Interactive (default)
The user is at the console providing feedback. The agent presents jobs one at a time, collects yes/no feedback and reasons, and updates `JOB_REQUIREMENTS.md` as preferences are learned.

```bash
python main.py
```

### Non-interactive
Designed for periodic/scheduled runs (e.g. cron). The agent searches LinkedIn autonomously, rates all jobs, and sends Telegram notifications for any rated 4 or 5. `JOB_REQUIREMENTS.md` is read but never modified.

Requires the `tools_telegram` MCP server to be running on port 8004 for job-match notifications. Pipeline summary/stats are sent directly via the Telegram Bot API regardless of whether the server is up.

```bash
bash /Users/rgwood/repos2022feb22v1/tools_telegram/scripts/start-tool-server.sh &
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

Run `uv run pytest` after most code changes to catch regressions. The test suite is fast and covers all core tool logic.

## File structure

```
main.py                           # Entry point
src/agentic_job_search/
  agent.py                        # Orchestration, prompts, pipeline stages
  tools_generic.py                # Tool implementations and MCP server factories
  config.py                       # Model name constants
tests/
  test_tools.py                   # Unit tests for all tools
run_dir/
  JOB_REQUIREMENTS.md             # Agent-managed job preferences (interactive mode)
  saved_jobs-{date}/              # Evaluated job postings (Markdown)
  processed_jobs/                 # Per-job YAML records for deduplication
```

## Architecture

The project uses the **[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)** (`claude-agent-sdk>=0.1.56`) as its foundation with **MCP (Model Context Protocol)** for tool use.

### Model tiers (`config.py`)

- `MODEL_NAME_MEDIUM` (`claude-sonnet-5`) — query generation and job evaluation
- `MODEL_NAME_LOW` (`claude-haiku-4-5`) — LinkedIn scraping stage

### Non-interactive pipeline (`run_non_interactive` in `agent.py`)

1. **Stage 1a — Query generation** (Sonnet): `generate_search_queries()` derives 2–6 short LinkedIn search queries from resume + `JOB_REQUIREMENTS.md`
2. **Stage 1b — Scraping** (Haiku): `run_scraper()` runs queries on LinkedIn, calls `check_and_record_job` + `queue_candidate` for each result listing without navigating to individual job pages
3. **Stage 2 — Evaluation** (Sonnet): `evaluate_candidate()` opens a fresh agent session per candidate, navigates to the job URL, rates 1–5, saves via `save_job_posting`, and calls `notify_user` for ratings ≥ 4

The evaluator prompt is built once (`build_evaluator_prompt()`) and reused across all jobs to maximise prompt caching.

### MCP server factories (`tools_generic.py`)

| Factory | Used in | Tools provided |
|---|---|---|
| `make_job_search_server(interactive)` | Interactive mode | `check_and_record_job`, `save_job_posting`, `notify_user`, `update_job_requirements` (if interactive) |
| `make_scraper_server()` | Stage 1b | `check_and_record_job`, `queue_candidate` |
| `make_evaluator_server()` | Stage 2 | `save_job_posting`, `notify_user` |

### Tools (`tools_generic.py`)

| Tool | Purpose |
|---|---|
| `check_and_record_job` | Returns `already_processed`, `too_old`, or `new`; records to `processed_jobs/*.yaml` |
| `save_job_posting` | Saves evaluated job to `saved_jobs-{date}/job_posting-*.md` |
| `queue_candidate` | Adds a job to the in-memory evaluation queue (Stage 1b only) |
| `update_job_requirements` | Rewrites `JOB_REQUIREMENTS.md` (interactive mode only) |
| `notify_user` | Sends a Telegram notification |

## Job deduplication and age filtering

On startup, `load_processed_jobs()` populates `_processed_jobs` from two sources:
- **Old format**: extracts LinkedIn job IDs from `saved_jobs-*/job_posting-*.md` filenames
- **New format**: loads `(site, job_id)` pairs from `processed_jobs/*.yaml` files

At runtime, `check_and_record_job` enforces:
- **Skip** if `(site, job_id)` is in `_processed_jobs` → returns `already_processed`
- **Skip** if posting date is parseable and > 21 days old → returns `too_old`
- **Proceed** otherwise → writes a YAML record, adds to `_processed_jobs`, returns `new`

`date_posted` accepts absolute (`YYYY-MM-DD`) or relative (`"4 days ago"`) formats; omit if not shown.

## Requirements

### Actors

- **User** — the job seeker; runs the agent and provides feedback on job postings
- **Scheduler** — a scheduled system event (e.g. cron) that triggers autonomous runs; a kind of User with no interactive input
- **LinkedIn** — external job board; serves search results and job pages (secondary actor)
- **Telegram** — external messaging service; delivers notifications to the User (secondary actor)

### Business Object Model

- **Resume** — user's CV stored as Markdown in `run_dir/`
- **JOB_REQUIREMENTS.md** — agent-managed preference file; read-only in non-interactive mode
- **Search Query** — short LinkedIn search string derived from Resume and JOB_REQUIREMENTS.md (2–6 per run)
- **Job Posting** — a LinkedIn listing with company, title, description, URL, job_id, date_posted, and a 1–5 rating
- **Processed Job Record** — `processed_jobs/*.yaml` keyed by `(site, job_id)`; drives deduplication across runs
- **Saved Job** — evaluated posting stored as `saved_jobs-{date}/job_posting-{id}-rating_{n}-*.md`

### Use Cases

**User**
- **Run interactive review**: present Job Postings one at a time and collect feedback
  - includes: Evaluate Job Fit
  - includes: Refine Job Requirements
- **Refine Job Requirements**: rewrite JOB_REQUIREMENTS.md based on User feedback on a Job Posting

**Scheduler**
- **Run autonomous search**: discover and rate Job Postings without user interaction
  - includes: Generate Search Queries
  - includes: Scrape Job Postings
  - includes: Evaluate Job Fit

**System** (invoked via includes)
- **Generate Search Queries**: derive Search Queries from Resume and JOB_REQUIREMENTS.md
- **Scrape Job Postings**: execute Search Queries on LinkedIn and collect candidate Job Postings
  - includes: Deduplicate Job Posting
  - includes: Filter Stale Job Posting
- **Deduplicate Job Posting**: skip a Job Posting already present in Processed Job Records
- **Filter Stale Job Posting**: skip a Job Posting whose scraped date is > 21 days old
- **Evaluate Job Fit**: navigate to Job Posting URL, score fit 1–5, save as Saved Job
  - includes: Notify User of Match
  - extends: Rate Closed or Expired Job Posting
- **Rate Closed or Expired Job Posting**: force rating to 1 if Job Posting shows "No longer accepting applications" or posting date > 30 days old; extends Evaluate Job Fit
- **Rate Unsupported US Job**: force rating to 1 if Job Posting is US-located without explicit visa sponsorship; extends Evaluate Job Fit
- **Notify User of Match**: send Telegram notification when a Saved Job has rating ≥ 4

### Non-functional Requirements

- **Cost efficiency** — Haiku for the high-volume scraping stage; Sonnet for evaluation; evaluator prompt built once and reused to maximise prompt-cache hits
- **Idempotency** — processed-job records persist across runs so jobs are never evaluated twice
- **Notification latency** — Telegram alerts sent immediately when a job is rated 4 or 5 during evaluation
- **Rating hard rules (applied before fit scoring)**: closed postings → 1; postings > 30 days old → 1; US jobs without explicit sponsorship → 1

## Git conventions

- Use `git mv` when moving or renaming tracked files
- Use `git rm` when deleting tracked files
