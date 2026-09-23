# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where things get stored

Knowledge about this project — decisions, conventions, diagnoses, preferences about how to work
on it — goes in a **tracked file in this repo**, normally this `CLAUDE.md`. Configuration and log
files go in **`run_dir/`**.

**Strongly prefer not to store any of that under `~/.claude/**`**, including the per-project
auto-memory directory (`~/.claude/projects/<slug>/memory/`). Machine-local storage is invisible,
unreviewable, doesn't reach another machine, and silently applies only to the directory it happened
to be written under. Only genuinely machine-specific things belong there, and **each one needs an
explicit OK from the user first** — never write there on your own initiative. This overrides any
default memory behaviour.

`run_dir/` is gitignored because it holds personal data — the resume, applied-job PDFs, saved
postings, and `JOB_REQUIREMENTS.md`. That data is intentionally machine-local; knowledge *about the
project* is not.

## Running the agent on this machine

**Never launch the browser in `visible` mode unless the user has explicitly asked to watch the
page.** A Chromium window appearing unannounced, and stealing focus at unpredictable moments
during a 40-minute run, makes the machine hard to use. `python main.py -n` already defaults to
`--browser headless`; use `--browser minimized` when a real window is needed (e.g. a profile that
misbehaves headless), and `visible` only for a POC the user is actively watching. The scratchpad
POC harness defaults to `minimized` and takes `PW_BROWSER_MODE` to override.

## Always-on invariants

The full text and the history behind each rule are in the linked doc. Read it before changing
anything a rule covers.

- **Nothing in the LinkedIn job results list is ever clicked.** The only real `<button>` in a card
  is Dismiss, and clicking it has already destroyed real jobs. JavaScript may read the page and
  never drive it. Never apply, save, follow, or dismiss anything. See Account safety in
  [docs/requirements.md](docs/requirements.md#non-functional-requirements).
- **Stop on a CAPTCHA, verification page, or "unusual activity" notice, and never retry into
  one.** The profile is a real logged-in account. See Account safety.
- **The model chooses actions and code moves data.** A tool argument must be a decision, never
  something code could have read itself. See Anti-fabrication in
  [docs/requirements.md](docs/requirements.md#non-functional-requirements).
- **A structural fact is never left to a model's discretion.** Caps, gates, and alerts are
  decided in code. See the first non-functional requirement in
  [docs/requirements.md](docs/requirements.md#non-functional-requirements).
- **A figure the pipeline extracts is shown to the user as extracted, or reported as missing or
  partial — never left to a model's paraphrase.** The salary reached one notification only through
  the rater's own bullets, which rounded one bound and led with the other. See Notify User of
  Match and Classify Salary in [docs/requirements.md](docs/requirements.md#use-cases).
- **Every `ClaudeAgentOptions` site passes `setting_sources=[]`, `strict_mcp_config=True` and
  `skills=[]`.** Without them, this file and every global MCP server get loaded into each call.
  An AST test enforces this. See Cost efficiency in
  [docs/requirements.md](docs/requirements.md#non-functional-requirements).
- **Tracked files contain no personal information.** Every personal fact is a preference in
  `run_dir/preferences.yaml`, read through `preferences.py`, with neutral defaults.
- **Store names in proper case and fold only when comparing.** Write `Málaga`, not `malaga`.
  Controlled vocabularies such as `southern_europe` are the exception.
- **`@playwright/mcp` stays pinned** (`PLAYWRIGHT_MCP_VERSION`), never `@latest`. See
  Dependency pinning in [docs/requirements.md](docs/requirements.md#non-functional-requirements).

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

Requires the `tools_telegram` MCP server to be running on port 8004 for job-match notifications; a failed send warns and continues, never aborting the run. The send itself goes through the shared lib (`utils_tools_n_agents_common.telegram_client`), which speaks MCP to tools_telegram — no Bot API code in this repo.

Stage 2 also uses two LLM MCP tool servers for cheap inference (reference summarization, triage, and optionally the rating call): `tools_llm_remote_openrouter` on port 8006 and `tools_llm_local` (Ollama) on port 8002. Both are optional — a down server is treated as a provider failure and the pipeline falls back (summary chain falls through to Anthropic; triage fails open).

```bash
# $TOOLS_DIR is wherever the sibling tool-server repos are checked out
bash "$TOOLS_DIR/tools_telegram/scripts/start-tool-server.sh" &
bash "$TOOLS_DIR/tools_llm_remote_openrouter/scripts/start-tool-server.sh" &
bash "$TOOLS_DIR/tools_llm_local/scripts/start-tool-server.sh" &
python main.py --non-interactive   # or: python main.py -n
```

## Commands

```bash
uv sync                  # Install dependencies
uv sync --group test     # Install test dependencies
uv lock --upgrade && uv sync  # Update all dependencies to latest versions
python main.py           # Run in interactive mode
python main.py -n        # Run in non-interactive mode
python main.py -n --audit # Non-interactive + re-rate gate-killed jobs to detect false negatives (diagnostic)
python main.py -n --audit-opus 2  # Non-interactive + re-rate 2 un-surfaced jobs per pool with Opus (diagnostic)
python main.py -n --no-version-check  # Skip the startup check for a newer @playwright/mcp than the pin

# Tests
uv run pytest                          # Run all tests; each marked group self-skips if unavailable
uv run pytest -m "not live_agent_claude and not live and not network"  # Unit tests only
uv run pytest -m live_agent_claude     # Live agent tests (skip unless the Claude CLI/API is reachable)
uv run pytest -m live                  # Live LLM MCP server tests (skip unless :8002/:8006 are up)
uv run pytest -m network               # Third-party network tests (skip when offline); no local servers
```

## Testing

Run `uv run pytest` after most code changes to catch regressions. The test suite is fast and covers all core tool logic.

## File structure

```
main.py                           # Entry point
src/agentic_job_search/
  agent.py                        # Orchestration, prompts, pipeline stages
  tools_generic.py                # Tool implementations and MCP server factories
  triage.py                       # Local triage and non-Anthropic rating calls (MCP client lives in utils_tools_n_agents_common)
  scrape_openrouter.py            # Stage 1b: OpenRouter function-calling scraper loop (default path)
  extract_openrouter.py           # OpenRouter function-calling agent loop for page extraction
  config.py                       # Model/provider constants and Stage 2 tuning (nothing personal)
  preferences.py                  # Loads run_dir/preferences.yaml; neutral defaults if absent
  location.py                     # Cached geographic classifier for a job's location
  salary.py                       # Salary reading: deterministic tiers, cached LLM only for the rest
  text_budget.py                  # Reported truncation: truncate_reported, pages_to_prompt, snippet
scripts/
  migrate_applied_jobs.py         # One-time reviewable move of applied-job PDFs into run_dir
  backtest_location_gate.py       # Replays saved postings through the location gate (live classifier)
  backfill_recruiter_notifications.py  # One-time seed of recruiter_notifications.yaml from saved jobs
tests/
  test_tools.py                   # Unit tests for all tools
docs/
  architecture.md                 # Applied-job corpus, model routing, pipeline stages, tools, dedup
  requirements.md                 # Actors, Business Object Model, Use Cases, Non-functional Requirements (incl. lessons from past incidents)
run_dir/
  JOB_REQUIREMENTS.md             # Agent-managed job preferences (interactive mode)
  applied_jobs/                   # Applied-job PDFs, date-prefixed, + index.yaml metadata
  saved_jobs-{date}/              # Evaluated job postings (Markdown)
  processed_jobs/                 # Per-job YAML records for deduplication
  audit_logs/                     # Per-run audit trace (audit-{date}-{time}.md)
  reference_summary_cache.yaml    # Cached distilled ideal-role profile (md5-keyed)
  location_cache.yaml             # Cached geography per location string (country/region/language)
  salary_cache.yaml               # Cached salary reading per salary string (kind/bounds/currency/period)
  raw_postings/                   # Pages extracts were made from, pruned to RAW_POSTINGS_RETENTION_DAYS; audit only
  location_recommendations.yaml   # Advisory review of the country lists (recommend-only, hand-edited)
  recruiter_notifications.yaml    # Agency postings already notified (RECRUITER_REPOST_WINDOW_DAYS); suppresses repost pings
  preferences.yaml                # Personal preferences (regions, gates, titles) — gitignored
  logs/                           # Per-run log files (rejection reasons, extract sizes, ratings)
preferences.example.yaml          # Tracked, neutral template for run_dir/preferences.yaml
```

## Docs — read before touching

These docs are **not** loaded automatically. Open the relevant one before changing the area it
covers. Most rules there were written in response to a real incident, tagged with its date, and
several look like bugs until you read why they exist. The full incident write-ups (evidence, logs,
measurements) are in git history: `git show dd14c73:docs/diagnoses/README.md`.

| If you are touching… | Read first in [docs/requirements.md](docs/requirements.md) unless noted |
|---|---|
| Model constants, provider routing, pipeline stages, MCP tools, dedup, applied-job corpus | [docs/architecture.md](docs/architecture.md) |
| Stage 1b scraping, `scrape_openrouter.py`, Playwright, anything that drives LinkedIn | Scrape Job Postings, Verify Search UI Contract, Harvest Job Listings; NFRs Account safety, Search coverage, Anti-fabrication |
| Location gate, `location.py`, `location_review.py`, the `locations.*` preferences | Reject Excluded Location, Classify Job Location, Detect Residency Scope, Review Country Lists; NFRs A matching rule fails silently permissive, Names stay proper |
| Hard rules, rating caps, warnings, notifications | Apply Hard Rules, Rate Job Fit, Build Deterministic Warnings; NFRs A structural fact is decided in code, Rating hard rules, Rating caps |
| Salary parsing, `salary.py`, the `💰` line, raw-posting retention | Classify Salary, Retain Raw Posting Text, Check Salary Provenance, Build Deterministic Warnings, Notify User of Match; NFRs A matching rule fails silently permissive, No silent truncation |
| `call_mcp_tool`, triage, the local model, OpenRouter errors and fallbacks | NFRs Resilience, Observability |
| A new `ClaudeAgentOptions` / `sdk_query` site, or an SDK tool schema | NFRs Cost efficiency, Observability, Anti-fabrication |
| Applied-job ingest or the reference profile | [docs/architecture.md](docs/architecture.md); Summarize Reference Jobs |
| Tests or test fixtures | NFR Tests must be able to fail |
| `PLAYWRIGHT_MCP_VERSION`, the `mcp` pin, startup checks | Check Browser Toolchain Version; NFR Dependency pinning |
| "The agent isn't finding jobs" | NFR Rejection auditability |

**Recording a lesson:** add it to the relevant non-functional requirement in
[docs/requirements.md](docs/requirements.md), or create one. Tag it with the date and state the
rule before the story. Unresolved problems go under Known open issues. If the rule must hold even
when nobody opens the doc, add it to Always-on invariants above.

## Requirements

Requirements (Actors, Business Object Model, Use Cases, Non-functional Requirements) live in
[docs/requirements.md](docs/requirements.md). The per-change Requirements review applies to that
file.

## Git conventions

- Use `git mv` when moving or renaming tracked files
- Use `git rm` when deleting tracked files
