# Agent Job Search

A personal autonomous job search agent built on the [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview). It searches LinkedIn for relevant job postings, rates them against your resume and preferences, and notifies you of strong matches via Telegram.

## Setup

```bash
uv sync              # Install dependencies
uv sync --group test # Install test dependencies
```

Create a `.env` file in the project root:
```
TELEGRAM_BOT_TOKEN=<your bot token>
TELEGRAM_CHAT_ID=<your chat ID>
```

The agent uses a persistent browser profile at `~/.linkedin-agent-profile/` so your LinkedIn session is remembered between runs. Log in on first launch when the browser opens.

Place your resume in `run_dir/` as Markdown, matching the pattern `*-resume-*.md`. The most recently modified match is used. A PDF resume is not read — convert it to Markdown.

## Usage

```bash
python main.py        # Interactive mode (default)
python main.py -n     # Non-interactive / autonomous mode
```

### Interactive mode

The agent reads your resume and `run_dir/JOB_REQUIREMENTS.md`, then searches LinkedIn and presents jobs one at a time. Give feedback (yes/no + reason) and the agent updates `JOB_REQUIREMENTS.md` to refine future searches.

### Non-interactive mode

Designed for scheduled/cron runs. The agent searches LinkedIn autonomously, rates all jobs, and sends Telegram notifications for jobs rated 4 or 5. `JOB_REQUIREMENTS.md` is read but never modified.

## How it works

**Non-interactive pipeline:**

1. **Query generation** (Sonnet) — derives 2–6 LinkedIn search queries from your resume and job requirements
2. **Scraping** (Haiku) — runs the queries, records candidates from search result listings without visiting individual job pages
3. **Evaluation** (Sonnet) — one fresh agent session per candidate: navigates to the job page, reads the full description, rates 1–5, saves to `run_dir/saved_jobs-{date}/`, and sends a Telegram notification for ratings ≥ 4

**Job deduplication:** `check_and_record_job` is called before any job is processed. It returns `already_processed`, `too_old` (posted > 21 days ago), or `new`, and records each new job to `run_dir/processed_jobs/*.yaml`.

## Tests

```bash
uv run pytest                            # All tests (skips live tests without API key)
uv run pytest -m "not live_agent_claude" # Unit tests only
uv run pytest -m live_agent_claude       # Live agent tests (requires ANTHROPIC_API_KEY)
```
