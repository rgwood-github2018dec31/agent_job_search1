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

Stage 2 also uses two LLM MCP tool servers for cheap inference (reference summarization, triage, and optionally the rating call): `tools_llm_remote_openrouter` on port 8006 and `tools_llm_local` (Ollama) on port 8002. Both are optional — a down server is treated as a provider failure and the pipeline falls back (summary chain falls through to Anthropic; triage fails open).

```bash
bash /Users/rgwood/repos2022feb22v1/tools_telegram/scripts/start-tool-server.sh &
bash /Users/rgwood/repos2022feb22v1/tools_llm_remote_openrouter/scripts/start-tool-server.sh &
bash /Users/rgwood/repos2022feb22v1/tools_llm_local/scripts/start-tool-server.sh &
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

# Tests
uv run pytest                          # Run all tests (skips live tests if no API key)
uv run pytest -m "not live_agent_claude and not live"  # Unit tests only
uv run pytest -m live_agent_claude     # Live agent tests (requires ANTHROPIC_API_KEY)
uv run pytest -m live                  # Live LLM MCP server tests (require servers on :8002/:8006)
```

## Testing

Run `uv run pytest` after most code changes to catch regressions. The test suite is fast and covers all core tool logic.

## File structure

```
main.py                           # Entry point
src/agentic_job_search/
  agent.py                        # Orchestration, prompts, pipeline stages
  tools_generic.py                # Tool implementations and MCP server factories
  triage.py                       # LLM MCP-server client, local triage, non-Anthropic rating calls
  extract_openrouter.py           # OpenRouter function-calling agent loop for page extraction
  config.py                       # Model/provider constants and Stage 2 tuning
scripts/
  migrate_applied_jobs.py         # One-time reviewable move of applied-job PDFs into run_dir
tests/
  test_tools.py                   # Unit tests for all tools
run_dir/
  JOB_REQUIREMENTS.md             # Agent-managed job preferences (interactive mode)
  applied_jobs/                   # Applied-job PDFs, date-prefixed, + index.yaml metadata
  saved_jobs-{date}/              # Evaluated job postings (Markdown)
  processed_jobs/                 # Per-job YAML records for deduplication
  audit_logs/                     # Per-run audit trace (audit-{date}-{time}.md)
  reference_summary_cache.yaml    # Cached distilled ideal-role profile (md5-keyed)
  logs/                           # Per-run log files (rejection reasons, extract sizes, ratings)
```

## Applied-job corpus (`run_dir/applied_jobs/`)

Jobs the user has applied to are the strongest available signal of what to search for. PDFs are
saved to `~/Downloads`, categorized by `categorize_downloads_pdfs()` into `cat-saved_jd-*.pdf`,
then moved into `run_dir/applied_jobs/` by `ingest_downloads_applied_pdfs()` on every startup,
renamed to `{YYYY-MM-DD}-{original}.pdf`.

The applied date is carried three ways, most durable first: the filename prefix, the
`applied_date` in `index.yaml`, and the file mtime (restored via `os.utime` after the move).
mtime alone is untrustworthy — any copy, backup restore, or rsync rewrites it.

`load_applied_jobs()` reads the corpus into three globals, keyed on `index.yaml` (which caches
extracted text and metadata per filename by mtime, so the Haiku metadata call only runs for new
or changed PDFs):

- `_applied_jobs` — feeds `applied_jobs_summary()` into the Stage 1a query prompt
- `_reference_job_texts` — feeds the ideal-role profile used by the rater
- `_applied_companies` — the already-applied blocklist used by `check_and_record_job`

Only records within `APPLIED_JOBS_HORIZON_DAYS` (90) populate these; older PDFs stay on disk but
are never read.

**Recruiters:** `_extract_applied_job_metadata()` returns `is_agency` and `end_client` alongside
company and title. An agency's own name never enters the blocklist — applying once through an
aggregator (Jobgether, CNA Search, SearchLabs, Syndesus) would otherwise suppress every other
company it posts for. The end client is blocklisted instead when the posting names one; agency
postings still contribute reference and query signal.

## Architecture

The project uses the **[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)** (`claude-agent-sdk>=0.1.56`) as its foundation with **MCP (Model Context Protocol)** for tool use.

### Model tiers and providers (`config.py`)

- `MODEL_NAME_HIGH` (`claude-opus-5`) — audit reference standard only (`--audit-opus`); never used in the normal pipeline
- `MODEL_NAME_MEDIUM` (`claude-sonnet-5`) — Anthropic fallback for query generation and rating
- `MODEL_NAME_LOW` (`claude-haiku-4-5`) — scraping, page extraction, and applied-job metadata
- `OPENROUTER_MODEL` (`z-ai/glm-5.2`) — **default** for query generation (`QUERY_PROVIDER`), company matching (`COMPANY_MATCH_PROVIDER`), and rating (`RATING_PROVIDER`). Each falls back to Anthropic if the OpenRouter MCP server is down
- `RATING_PROVIDER` — `'openrouter'` (default, `z-ai/glm-5.2`) | `'anthropic'` (`MODEL_NAME_MEDIUM`) | `'ollama'` (`LOCAL_MODEL`) — selects who makes the final rating call
- `EXTRACTOR_PROVIDER` — `'anthropic'` (default, Haiku agentic session) | `'openrouter'` (function-calling agent loop in `extract_openrouter.py`: glm-5.2 drives the browser tools via the OpenRouter MCP server's `chat` tool with OpenAI-style `tools`; capped at `EXTRACTOR_OPENROUTER_MAX_ITERATIONS`, tool results truncated to `EXTRACTOR_TOOL_RESULT_MAX_CHARS`). The deterministic Haiku fallback covers failures of either provider
- Non-Anthropic models are reached via the LLM MCP tool servers (`LLM_OPENROUTER_MCP_URL` :8006, `LLM_LOCAL_MCP_URL` :8002) using `call_mcp_tool()` in `triage.py`

### Non-interactive pipeline (`run_non_interactive` in `agent.py`)

1. **Stage 1a — Query generation** (glm, Anthropic fallback): `generate_search_queries()` derives at most `MAX_SEARCH_QUERIES` (6) LinkedIn search queries from resume + `JOB_REQUIREMENTS.md` + the in-horizon applied-job titles (`applied_jobs_summary()`). Individual-contributor titles only — people-management titles are explicitly excluded
2. **Stage 1b — Scraping** (Haiku): `run_scraper()` issues **one request per query on a single shared session** (`SCRAPER_MAX_TURNS_PER_QUERY` turns each), calling `check_and_record_job` + `queue_candidate` for each result listing without navigating to individual job pages. Both halves are load-bearing: batching all queries into one request lets the first few exhaust the turn budget so the rest are never searched (while still reporting `0 jobs`); opening a fresh client per query makes later ones fail with "browser is in use" against the shared Playwright context, which looks identical to an auth wall in the logs
3. **Stage 2 — Evaluation** (`evaluate_all_candidates()`), per candidate:
   - **2a Extract** (Haiku, agentic): `extract_job_page()` navigates to the job URL, expands the description, and submits a condensed extract via `submit_job_extract` (nav chrome and boilerplate stripped). If the session ends without submitting (Haiku can exhaust its turn budget hunting through truncated snapshots of large pages), `extract_job_page_direct()` falls back to a deterministic path: navigate + wait + snapshot (+ one "… more" expand click) driven directly over the Playwright MCP session, then one non-agentic Haiku call condenses the full snapshot — same browser and logged-in profile, so bot-detection exposure is identical
   - **2b Hard rules** ($0, deterministic): `apply_hard_rules()` auto-rates 1 for closed postings, postings > `JOB_STALE_AGE_DAYS` (30) days old, US jobs without explicit sponsorship, and jobs with an explicit non-English language requirement (from the extract's `language_requirement` field). A required relocation (`relocation` field) does NOT reject — it is flagged as "relocation required: <location>" in the saved job and the Telegram notification, and the evaluator is instructed not to penalize EU relocation. Every rejection is logged with its reason (console + `run_dir/logs/run-*.log`)
   - **2c Triage** ($0, local LLM): `triage_job_fit()` scores fit 1–5 with Ollama; scores ≤ `TRIAGE_THRESHOLD` (1) are saved with the triage score and skip the rating call; fails open if the server is down
   - **2d Rating** (configurable): `rate_job()` makes one non-agentic structured-output call, saves via `do_save_job_posting()`, and sends a Telegram notification for ratings ≥ 4

Before Stage 2, `build_reference_summary()` distills the reference-job PDFs into a ~2.5K-char "ideal role profile" (provider chain: OpenRouter → local Ollama → Anthropic Haiku; falls back to the full reference block if all fail). The result is cached in `run_dir/reference_summary_cache.yaml` keyed by an md5 of the reference texts, so it is only regenerated when the PDFs change. The evaluator prompt embeds this summary instead of the former 20×3,000-char reference block.

Key per-job observability is logged via the `logging` module: extract input tokens vs. condensed size, hard-rule short-circuits, triage score vs. final rating, and the rating provider used. Per-stage costs land in `cost_log.jsonl` under `reference_summary` / `extraction` / `rating` (triage and hard rules are free), charged per session — see the Observability requirement.

### MCP server factories (`tools_generic.py`)

| Factory | Used in | Tools provided |
|---|---|---|
| `make_job_search_server(interactive)` | Interactive mode | `check_and_record_job`, `save_job_posting`, `update_job_requirements` (if interactive) |
| `make_scraper_server()` | Stage 1b | `check_and_record_job`, `queue_candidate` |
| `make_evaluator_server()` | Stage 2a | `submit_job_extract` |

In Stage 2, saving and Telegram notification are plain Python calls (`do_save_job_posting()`, `_send_pipeline_notification()`), not MCP tools.

### Tools (`tools_generic.py`)

| Tool | Purpose |
|---|---|
| `check_and_record_job` | Returns `already_processed`, `too_old`, or `new`; records to `processed_jobs/*.yaml` |
| `save_job_posting` | Saves evaluated job to `saved_jobs-{date}/job_posting-*.md` |
| `queue_candidate` | Adds a job to the in-memory evaluation queue (Stage 1b only) |
| `submit_job_extract` | Captures the condensed page extract from the Stage 2a extractor |
| `update_job_requirements` | Rewrites `JOB_REQUIREMENTS.md` (interactive mode only) |

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
- **OpenRouter** — external paid LLM gateway (via MCP tool server :8006); summarizes reference jobs and optionally rates fit (secondary actor)
- **Local LLM** — local Ollama models (via MCP tool server :8002); triages job fit for free and serves as summarization fallback (secondary actor)

### Business Object Model

- **Resume** — user's CV stored as Markdown in `run_dir/`
- **JOB_REQUIREMENTS.md** — agent-managed preference file; read-only in non-interactive mode
- **Search Query** — short LinkedIn search string derived from Resume, JOB_REQUIREMENTS.md, and Applied Job Records
- **Applied Job Record** — a job the User applied to: a date-prefixed PDF in `run_dir/applied_jobs/` plus its `index.yaml` metadata (applied date, company, job title, recruiting-agency flag, end client). Active for 3 months; older records are retained but unused
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
  - includes: Ingest Applied Jobs
  - includes: Generate Search Queries
  - includes: Scrape Job Postings
  - includes: Evaluate Job Fit

**System** (invoked via includes)
- **Ingest Applied Jobs**: move categorized applied-job PDFs from Downloads into `run_dir/applied_jobs/`, stamping each filename with its applied date and preserving mtime; already-dated files are a no-op, so it is safe on every run
  - includes: Flag Recruiting Agency
- **Flag Recruiting Agency**: record whether an Applied Job Record's poster is a staffing firm or aggregator rather than the hiring employer, along with the end client if named; agency names are never added to the already-applied blocklist
- **Generate Search Queries**: derive at most `MAX_SEARCH_QUERIES` (6) Search Queries from Resume, JOB_REQUIREMENTS.md, and the Applied Job Records within the 3-month horizon (glm, Anthropic fallback), covering both the titles already applied to and adjacent titles. Individual-contributor titles only — people-management titles (Manager / Head of / Director / VP) are excluded
- **Write Run Audit Log**: at the end of every run, write `run_dir/audit_logs/audit-{date}-{time}.md` tracing applied-job counts, every Search Query and whether it was actually searched, listings and candidates per query, and the outcome of every individual Job Posting with its URL and summary
- **Audit Un-surfaced Jobs**: (`--audit-opus N`) sample N Job Postings from each of three un-surfaced pools — filtered at Stage 1, seen but never queued, and rated 2–3 — re-rate each with Opus, and report any the strong model scores ≥4 as a false negative. Diagnostic only; nothing is saved or notified
- **Scrape Job Postings**: execute Search Queries on LinkedIn and collect candidate Job Postings. Every Search Query runs as **four searches** — Canada and European Union, each in **both sort orders** — and all four run every time:
  - `sortBy=DD` (newest first) catches fresh postings early, but LinkedIn's date sort ignores relevance and returns a lot of loosely-matching noise
  - **default sort** (`sortBy` omitted) is LinkedIn's own relevance ranking, which surfaces the best-matching roles; these are usually older and so mostly already seen

  The relevance-sorted searches are **expected to return a high proportion of `already_processed`** results — that is the cost of the few genuinely best-matching jobs they surface, and is not a failure signal. Only page 1 of each of the four searches is scanned; four searches shallow beats one search deep. Search URLs always carry `f_WT=2` (remote) and `f_E=4%2C5%2C6` (experience level Mid-Senior/Director/Executive — keeps entry-level/associate/junior noise out of the funnel). A search returning zero results is retried once before being counted as empty. If the whole first scraper pass inspects **zero** listings (auth-wall / block / empty-shell signature, distinct from "seen but deduped"), `run_scraper` automatically runs one code-enforced recovery pass with the date-sort and location filters dropped before giving up

  **Scraping is not a deterministic problem and must stay LLM-driven.** It is tempting to
  replace the agentic scraper with a parser: at any single moment the mechanics *are*
  reproducible in code (navigate → scroll the inner results container → snapshot → regex →
  click Next). That reproducibility is a snapshot of one day's markup. Job boards restructure
  their DOM without notice and will almost certainly keep adding bot detection, consent
  interstitials, login walls, and challenge pages. A hard-coded parser fails *silently* against
  all of those — it returns zero listings, which is indistinguishable from "no new jobs today".
  An LLM navigating the page can recognise an obstacle and work around it. Accordingly the
  scraper keeps the browser tools needed to handle the unexpected (`fill_form`, `type`,
  `hover`, `select_option`, `handle_dialog`, `console_messages`) even though none are used on
  the happy path, and `SCRAPER_DISALLOWED_BROWSER_TOOLS` removes only tools that are redundant
  with an untruncated snapshot or have no role in reading a results list. Cost work on Stage 1b
  should reduce what enters the model's context, never remove the model's ability to navigate
  - includes: Deduplicate Job Posting
  - includes: Filter Stale Job Posting
- **Deduplicate Job Posting**: skip a Job Posting already present in Processed Job Records, or whose company is on the already-applied blocklist derived from in-horizon Applied Job Records
- **Filter Stale Job Posting**: skip a Job Posting whose scraped date is > 21 days old
- **Summarize Reference Jobs**: distill Reference Job PDFs into a compact ideal-role profile via provider chain (OpenRouter → Local LLM → Anthropic), cached until the PDFs change
- **Evaluate Job Fit**: extract, filter, triage, and rate a candidate Job Posting, saving it as a Saved Job
  - includes: Extract Job Posting
  - includes: Apply Hard Rules
  - includes: Triage Job Posting
  - includes: Rate Job Fit
  - includes: Notify User of Match
- **Extract Job Posting**: navigate to Job Posting URL and capture a condensed, information-dense extract of the page; provider configurable — cheap Anthropic model (agentic session, default) or OpenRouter model via a function-calling loop
- **Apply Hard Rules**: deterministically ($0) force rating to 1 if the Job Posting is closed ("No longer accepting applications"), > 30 days old, US-located without explicit visa sponsorship, or explicitly requires a non-English language; every rejection is logged with its reason
- **Flag Required Relocation**: annotate (never reject) a Job Posting requiring relocation/residence in a specific location with "relocation required: <location>" in the Saved Job and Notification; extends Evaluate Job Fit
- **Triage Job Posting**: score fit 1–5 with the Local LLM; clear low fits (score ≤ 1) are saved with the triage score and skip Rate Job Fit; fails open if the Local LLM is unavailable
- **Extract Job Posting (fallback)**: when the agentic extractor fails to submit, deterministically fetch the page snapshot over the shared Playwright session and condense it with one non-agentic cheap-model call; extends Extract Job Posting
- **Rate Job Fit**: one non-agentic structured-output call scoring fit 1–5 against JOB_REQUIREMENTS.md and the ideal-role profile; provider configurable (Anthropic Sonnet default, OpenRouter `z-ai/glm-5.2`, or Local LLM)
- **Notify User of Match**: send Telegram notification when a Saved Job has rating ≥ 4

### Non-functional Requirements

- **Scraper tool surface** — Stage 1b restricts the Playwright toolset via `disallowed_tools` (`SCRAPER_DISALLOWED_BROWSER_TOOLS`), which is the only option that removes a tool from the model's context. `allowed_tools` does **not** do this: it only auto-grants permission, which `permission_mode='bypassPermissions'` already grants, making it inert — measured, the model saw all 24 Playwright tools with `allowed_tools` set to 4. `browser_evaluate` and `browser_click` are load-bearing for discovery (inner-container scroll lazy-loads 7 → 10 listings; Next yields a fresh page) and a unit test asserts they are never disallowed
- **Cost efficiency** — Haiku for scraping, page extraction, and applied-job metadata; the agentic browser work never runs on Sonnet. Query generation is the one deliberate exception: a single Sonnet call per run (~$0.33), because top-of-funnel query relevance is the pipeline bottleneck and everything downstream is gated by it. Applied-job metadata is cached in `index.yaml` by mtime, so the extraction cost is paid once per PDF. Hard rules and local triage reject clear non-fits for $0 before any paid rating call. The rating call is a single non-agentic structured-output call (provider configurable; Sonnet pinned explicitly by default, never the CLI default model). Reference jobs are distilled once into a ~2.5K-char cached profile instead of a ~60K-char block; the evaluator prompt is built once per run and reused across all jobs to maximise prompt-cache hits; the extractor is capped at 8 turns and restricted to the browser tools it needs
- **Observability** — the generated Search Queries are logged; every `check_and_record_job` outcome is logged (`new`/`already_processed`/`already_applied`/`too_old`/`auth_required`) so seen-but-deduped is distinguishable from never-seen; per-job logs of the extract signal (`date_posted`/`location`/`closed`/`language_requirement`/`relocation`), extract compression, hard-rule short-circuits, triage score vs. final rating, and rating provider. Each run emits a single `Run funnel: {...}` line (queries → listings seen → check-status counts → candidates → extract ok/failed → hard-ruled by reason → triaged-out → rated 1–5), also written to `cost_log.jsonl` under `funnel`. Per-stage costs (`reference_summary`, `extraction`, `rating`) in `cost_log.jsonl`. Cost is charged **per session**: `total_cost_usd` is cumulative within a `ClaudeSDKClient` session, so only the increase over that session's last reported value is billed (`cost_delta_for` in `agent.py`, keyed on `session_id`). Stage 1b shares one session across all queries and was previously over-counted ~3.5x; stages that open a fresh session per call were always correct. Token counters in `msg.usage` are per-request and keep summing. Every `ResultMessage` is logged with `session_id`, `num_turns`, usage, and `cost_reported` vs `cost_charged` — previously these went to the `rich` console only and never reached the run log, which is why the over-count went unnoticed for nine runs. Third-party HTTP transport logging (`httpx` / `httpcore` / `mcp.client.streamable_http`, one line per MCP tool call) is suppressed to WARNING so the run log stays the run's own audit trail; set `HTTP_LOG_LEVEL=INFO` to restore it when debugging a tool server
- **Search coverage** — every generated Search Query must actually be searched, in **both sort orders** (date and relevance) across both regions: four searches per query. Stage 1b sends one request per query (own turn budget) on one shared session (one browser attachment); the run audit log distinguishes "never searched" from "searched, found nothing", which a single `0 jobs` count cannot. Turn starvation is silent — the model stops mid-query and the run still reports a listing count — so `SCRAPER_MAX_TURNS_PER_QUERY` (90) is set with headroom for four searches rather than tuned tight, and `num_turns` is logged per request to make exhaustion visible
- **Rejection auditability** — `python main.py -n --audit` runs normally but additionally re-rates every gate-killed Job Posting (hard-ruled or triaged-out) with the strong rater, logging any **false negative** (gate dropped it but the strong rater scores ≥3) plus a run-end count. Diagnostic only — saving/notification behaviour is unchanged
- **Resilience** — the LLM MCP tool servers are optional: summarization falls through its provider chain (last resort: full reference block), triage fails open to the rating call
- **Idempotency** — processed-job records persist across runs so jobs are never evaluated twice; applied-job ingest is a no-op for already-dated files
- **Applied-date durability** — an Applied Job Record's date is carried by the filename prefix, `index.yaml`, and mtime independently, so it survives a move, copy, or backup restore that drops filesystem metadata
- **Applied-job horizon** — only Applied Job Records from the last `APPLIED_JOBS_HORIZON_DAYS` (90) feed query generation, the ideal-role profile, and the already-applied blocklist; older PDFs are retained on disk, never deleted
- **Notification latency** — Telegram alerts sent immediately when a job is rated 4 or 5 during evaluation
- **Rating hard rules (applied deterministically in code before any LLM scoring)**: closed postings → 1; postings > 30 days old → 1; US jobs without explicit sponsorship → 1; explicit non-English language requirement → 1. Required relocation is flagged, never auto-rejected

## Git conventions

- Use `git mv` when moving or renaming tracked files
- Use `git rm` when deleting tracked files
