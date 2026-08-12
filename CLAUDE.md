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
  config.py                       # Model/provider constants and Stage 2 tuning (nothing personal)
  preferences.py                  # Loads run_dir/preferences.yaml; neutral defaults if absent
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
  preferences.yaml                # Personal preferences (regions, gates, titles) — gitignored
  logs/                           # Per-run log files (rejection reasons, extract sizes, ratings)
preferences.example.yaml          # Tracked, neutral template for run_dir/preferences.yaml
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
company and title. An agency's own name never enters the blocklist — applying once through a
staffing firm or job aggregator would otherwise suppress every other company it posts for. The end client is blocklisted instead when the posting names one; agency
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
- Non-Anthropic models are reached via the LLM MCP tool servers (`LLM_OPENROUTER_MCP_URL` :8006, `LLM_LOCAL_MCP_URL` :8002) using `call_mcp_tool()` in `triage.py`. **Always route OpenRouter/Ollama inference through these servers, never the raw HTTP APIs** — that is where keys, config, and per-call `cost_usd` tracking live. Note the local Ollama `generate` tool has **no default model configured**, so always pass `model` explicitly

### Non-interactive pipeline (`run_non_interactive` in `agent.py`)

1. **Stage 1a — Query generation** (glm, Anthropic fallback): `generate_search_queries()` derives at most `MAX_SEARCH_QUERIES` (6) LinkedIn search queries from resume + `JOB_REQUIREMENTS.md` + the in-horizon applied-job titles (`applied_jobs_summary()`). Individual-contributor titles only — people-management titles are explicitly excluded
2. **Stage 1b — Scraping** (Haiku): `run_scraper()` issues **one request per query on a single shared session** (`SCRAPER_MAX_TURNS_PER_QUERY` turns each), calling `check_and_record_job` + `queue_candidate` for each result listing without navigating to individual job pages. Both halves are load-bearing: batching all queries into one request lets the first few exhaust the turn budget so the rest are never searched (while still reporting `0 jobs`); opening a fresh client per query makes later ones fail with "browser is in use" against the shared Playwright context, which looks identical to an auth wall in the logs
3. **Stage 2 — Evaluation** (`evaluate_all_candidates()`), per candidate:
   - **2a Extract** (Haiku, agentic): `extract_job_page()` navigates to the job URL, expands the description, and submits a condensed extract via `submit_job_extract` (nav chrome and boilerplate stripped). If the session ends without submitting (Haiku can exhaust its turn budget hunting through truncated snapshots of large pages), `extract_job_page_direct()` falls back to a deterministic path: navigate + wait + snapshot (+ one "… more" expand click) driven directly over the Playwright MCP session, then one non-agentic Haiku call condenses the full snapshot — same browser and logged-in profile, so bot-detection exposure is identical
   - **2b Hard rules** ($0, deterministic): `apply_hard_rules()` auto-rates 1 for closed postings, postings > `JOB_STALE_AGE_DAYS` (30) days old, jobs in a `sponsorship_required_in` location without explicit sponsorship, jobs requiring a language outside `languages` (from the extract's `language_requirement` field), and jobs that hard-require a degree listed in `reject_required_degrees` (from the extract's `education_requirement` field, backed by `derive_education_requirement()`). The three location/language/education gates are **preference-driven** — each is off when its preference list is empty. A required relocation (`relocation` field) does NOT reject — it is flagged as "relocation required: <location>" in the saved job and the Telegram notification, and the evaluator is given `relocation_note` as guidance. Every rejection is logged with its reason (console + `run_dir/logs/run-*.log`)
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
| `queue_candidate` | Adds a job to the in-memory evaluation queue (Stage 1b only); skips below-seniority / people-management titles for $0 |
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

## Known diagnoses

### 2026-08-11 — LinkedIn migrated the account to AI-powered job search

A run surfaced 1 job: 7 listings inspected, against 162–166 on the two previous days. It was
**not** dedup saturation (the funnel was empty before dedup ran), **not** a bot block (no
CAPTCHA, no login wall, "99+ results" render fine), and **not** a code change — the working
tree, `preferences.yaml`, `JOB_REQUIREMENTS.md` and `uv.lock` were all last modified *before*
the healthy run, so byte-identical inputs produced 166 listings and then 7.

The account has been moved to LinkedIn's **AI-powered job search**. The page says so outright:
*"You're now using AI-powered job search. Some filters may no longer be available, but you can
type them into search to refine your results."* Two things follow, both verified by driving the
agent's own Playwright profile (`~/.linkedin-agent-profile`), not just a desktop browser.

**1. Filters are gone from both the URL and the chip row.** `/jobs/search/` redirects to
`/jobs/search-results/` and strips every filter param, keeping only `keywords`:

| Param | Honored? | Evidence |
|---|---|---|
| `keywords` | **yes** | only survivor |
| `location=<region>` | no | stayed on the profile's own metro area; an EU search returned Canadian jobs |
| `f_WT=2` (remote) | no | hybrid and on-site jobs in results |
| `f_E=4%2C5%2C6` | no | experience level not applied |
| `sortBy=DD` | no | stripped |
| `f_SAL=` (empty) | **yes** | an empty value *clears* sticky state |

A salary filter never requested (`f_SA_id_225001:272001`) was injected from account-side state,
silently narrowing every search — the *same value* in both the desktop browser and the agent
profile, so this state is account-level, not per-browser. The search box is now a
natural-language field ("Describe the job you want"), and LinkedIn had rewritten the query to
`Staff AI Engineer remote` — folding the old `f_WT=2` into the text.

**2. The accessibility tree is a trap on this page — and following it destroys real jobs.** The
result cards have no `<a href>` and no id in the a11y tree; only the *selected* card's id shows,
in the URL as `currentJobId`. That made it look like ids had to be revealed by clicking each card.
**That approach is wrong and dangerous, and must not be reintroduced.**

The only real `<button>` in a result row is its **Dismiss** control — a 32×32 target whose
`aria-label` is "Dismiss `<job title>` job". Accessible names concatenate row text, so Playwright
renders each card as a single button labelled like
`button "Staff ML Engineer … Samsara … Dismiss … job"`. A model selecting "the card" by ref is
selecting the Dismiss button, and `browser_click` hits its centre. A run built on this dismissed
**three real jobs from the user's feed in under two minutes** (SandboxAQ and Samsara Staff MLE
roles, plus an Autodesk listing) before it was killed. Prompt wording cannot prevent this: to the
model, the card and the Dismiss are the same node.

**The ids were in the DOM the whole time.** The results container is
`div[componentkey="SearchResultsMainContent"]`, and every card is
`div[componentkey="job-card-component-ref-<jobId>"]` — the id is the attribute suffix. One
read-only `browser_evaluate` yields all 25 listings with id, title, company, location and posting
date (verified live: 25 jobs, 0 malformed). So Stage 1b **never clicks the results list at all**.

Two gaps let the bad run pass as a success, both now closed: the recovery pass triggered only at
**exactly zero** listings, so `seen == 1` sailed through; and the scraper's own narration was
`print()`ed to the terminal and never logged, so its account of what it saw was discarded every
run — which is why this needed a browser POC to diagnose rather than a log read.

**Do not** reintroduce `f_WT` / `f_E` / `location` / `sortBy` as query parameters, **do not** read
job ids from the accessibility tree, and **do not** click anything in the results list. Unit tests
guard all three.

### 2026-07-23 — "the agent isn't finding any jobs" is a top-of-funnel problem

A `--audit` run over 25 candidates proved the bottleneck is **candidate relevance and volume at
Stage 1**, not the evaluation pipeline. Every competing hypothesis was disproven: dedup saturation
(all 25 listings came back `new`), an auth wall / broken LinkedIn session, extraction failures
(25/25 succeeded), and over-aggressive gates — the audit re-rated every gate-killed job with the
strong rater and found **0 false negatives**.

The real problem is that LinkedIn queries surface a pool that is mostly off-target — wrong
seniority, wrong discipline, wrong stack relative to the configured preferences. Only ~12% (3/25)
scored 4/5. "Not finding jobs" is a low base rate plus high variance — some runs land 0–1 fours —
and notifications only fire at ≥4.

**Do not loosen the gates in response to a quiet run.** The audit confirmed they are accurate. The
leverage is all in Stage 1: better LinkedIn URL filters and query yield. Fixes already applied on
2026-07-23: the `f_E=4%2C5%2C6` seniority filter on scraper URLs, the code-enforced recovery pass on
a zero-listing scrape, full funnel instrumentation, and `--audit` mode. Offered but deprioritized:
the year-off date-clamp bug, tightening the sponsorship rule, and more queries / more result pages.

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
- **Preferences** — `run_dir/preferences.yaml` (gitignored), loaded by `preferences.py`: search regions, sponsorship-required locations, languages, rejected degree levels, hybrid cap and acceptable locations, target/excluded titles, relocation note. This is the **only** home for facts about the person running the agent; tracked source must stay neutral. `preferences.example.yaml` in the project root documents the format with placeholder values
- **JOB_REQUIREMENTS.md** — agent-managed preference file; read-only in non-interactive mode
- **Search Query** — short LinkedIn search string derived from Resume, JOB_REQUIREMENTS.md, and Applied Job Records
- **Applied Job Record** — a job the User applied to: a date-prefixed PDF in `run_dir/applied_jobs/` plus its `index.yaml` metadata (applied date, company, job title, recruiting-agency flag, end client). Active for 3 months; older records are retained but unused
- **Job Posting** — a LinkedIn listing with company, title, description, URL, job_id, date_posted, `workplace_type` (`remote` / `hybrid` / `onsite`), `education_requirement` (`master` / `phd` / empty — hard requirements only), and a 1–5 rating
- **Rating** — the evaluator's verdict on a Job Posting: 1–5 score, reasoning, filename label, plus **pros** and **warnings** bullet lists that drive the Notification and the Saved Job
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
- **Write Run Audit Log**: at the end of every run, write `run_dir/audit_logs/audit-{date}-{time}.md` tracing applied-job counts, every Search Query and whether it was actually searched, listings and candidates per query **broken down by check status (new / already processed / already applied / too old)**, and the outcome of every individual Job Posting with its URL and summary. The per-query breakdown is what separates a dedup-saturated query from one that barely ran
- **Screen Candidate Title**: at queue time, deterministically ($0) skip a listing whose title is below target seniority (intern / junior / entry-level / new grad / apprentice / trainee / co-op) or matches a Preferences `titles.exclude` word, before it reaches the paid extract-and-rate path. Conservative by design — word-boundary matches only, `associate` and bare `graduate` never reject, and work arrangement is never judged from a search card. Every skip is logged and counted in the run funnel under `queue_skipped`; extends Scrape Job Postings
- **Audit Un-surfaced Jobs**: (`--audit-opus N`) sample N Job Postings from each of three un-surfaced pools — filtered at Stage 1, seen but never queued, and rated 2–3 — re-rate each with Opus, and report any the strong model scores ≥4 as a false negative. Diagnostic only; nothing is saved or notified
- **Scrape Job Postings**: execute Search Queries on LinkedIn and collect candidate Job Postings. Every Search Query runs as **one search per configured region** (see Preferences `search_regions`), and all of them run every time.

  **Filters and region go into the natural-language query text, never into URL parameters or filter chips.** As of 2026-08-11 the account is on LinkedIn's AI-powered job search: filter params are stripped from the URL, the chip row is gone, and LinkedIn's own guidance is to type filters into the search box (see Known diagnoses). Each search navigates to `/jobs/search-results/?keywords=<QUERY TEXT>&f_SAL=` — `keywords` is the only parameter still honoured, and the empty `f_SAL=` clears a leftover account-level salary filter — with the region and the words `remote` and `senior level` in the query text. Sort order is **no longer a coverage axis**: the AI-powered UI exposes no sort control, so the former "each region in both sort orders" fan-out would simply run the same search twice. Region is the one axis that still varies the result set.

  Searches are **expected to return a high proportion of `already_processed`** results — that is the cost of the few genuinely best-matching jobs they surface, and is not a failure signal. Only page 1 is scanned. The converse also holds: **two searches returning the identical list of jobs means the region text did not apply**, and is a failure to report rather than a sign the query is exhausted.

  The scraper **reads back the result count and the query text the search box actually contains** before collecting anything; that read-back is the record of whether the search was real.
  - includes: Harvest Job Listings

  A search returning zero results is retried once before being counted as empty. A query inspecting fewer than `SCRAPER_MIN_LISTINGS_PER_QUERY` (5) listings is retried once by `run_scraper`: at **zero** (auth-wall / block / empty-shell signature, distinct from "seen but deduped") the retry drops the region text; **above zero** — the signature of a results list that was never walked card by card — the retry re-states the selection procedure. **A challenge or verification page is never retried into**: the recovery prompt tells the model to stop and report instead, because pushing through a block is the one failure that can cost the account.

- **Harvest Job Listings**: read every listing off the results page in **one read-only `browser_evaluate`** (`SCRAPER_HARVEST_JS` in `agent.py`), returning id, title, company, location and posting date per card. Cards are `div[componentkey="job-card-component-ref-<jobId>"]` inside `div[componentkey="SearchResultsMainContent"]` — **the LinkedIn job id is the attribute suffix**, so no clicking is needed to identify a listing. Cards appear twice in the DOM (dedupe by id) and each label is rendered twice (a visually-hidden copy carrying "(Verified job)" plus the visible one), so the extractor collapses both. **Nothing in the results list is ever clicked** — see Account safety. If the harvest returns an error or zero jobs while the page visibly shows results, the markup has changed and the scraper must say so rather than fall back to clicking. Capped at `SCRAPER_MAX_LISTINGS_PER_SEARCH` (25); invoked by Scrape Job Postings

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
- **Apply Hard Rules**: deterministically ($0) force rating to 1 if the Job Posting is closed ("No longer accepting applications"), > 30 days old, located where the user needs sponsorship but none is offered, explicitly requires a language the user does not have, or hard-requires a degree the user does not hold. The last three read their thresholds from Preferences and are inactive when unconfigured; every rejection is logged with its reason
  - includes: Detect Advanced Degree Requirement
- **Detect Advanced Degree Requirement**: resolve whether the Job Posting hard-requires a Master's or PhD, via the extract's structured `education_requirement` field or, when the extractor leaves it empty, `derive_education_requirement()` — a sentence-level scan requiring a degree token **and** a requirement word **and** no softener (`preferred`, `or equivalent`, a Bachelor's alternative) and no negation (`No PhD required`). Only hard requirements resolve to `master`/`phd`; preferred or experience-substitutable degrees resolve to empty and never reject. Invoked by Apply Hard Rules
- **Flag Workplace Type**: capture the work arrangement as a structured `workplace_type` field (`remote` / `hybrid` / `onsite`) rather than as free text inside the location; when the extractor omits it, `derive_workplace_type()` infers it from the location and description. Any required office days are `hybrid`, even when the board badges the listing "Remote" — LinkedIn's `f_WT=2` filter is not reliable. Extends Extract Job Posting
- **Flag Required Relocation**: annotate (never reject) a Job Posting requiring relocation/residence in a specific location with "relocation required: <location>" in the Saved Job and Notification; extends Evaluate Job Fit
- **Triage Job Posting**: score fit 1–5 with the Local LLM; clear low fits (score ≤ 1) are saved with the triage score and skip Rate Job Fit; fails open if the Local LLM is unavailable
- **Extract Job Posting (fallback)**: when the agentic extractor fails to submit, deterministically fetch the page snapshot over the shared Playwright session and condense it with one non-agentic cheap-model call; extends Extract Job Posting
- **Rate Job Fit**: one non-agentic structured-output call scoring fit 1–5 against JOB_REQUIREMENTS.md and the ideal-role profile, also returning **pros** and **warnings** bullet lists; provider configurable (Anthropic Sonnet default, OpenRouter `z-ai/glm-5.2`, or Local LLM). `apply_rating_caps()` then applies the deterministic hybrid ceiling below
- **Cap Hybrid Rating**: after rating, deterministically cap a `hybrid`/`onsite` Job Posting at the Preferences `hybrid.rating_cap` unless its location matches `hybrid.acceptable_locations`. A cap is not a rejection — the job is still saved and still appears in the audit log, it just falls below the ≥ 4 notification threshold; extends Rate Job Fit
- **Notify User of Match**: send Telegram notification when a Saved Job has rating ≥ 4. The message carries the rating, company, title, a `📍` location line including workplace type, a bulleted **✅ Good** list (from the rater's `pros`) and a bulleted **⚠️ Warnings** list (deterministic warnings first, then the rater's, de-duplicated), and the URL. Sent as plain text — no `parse_mode` — so it uses bullets and emoji, never Markdown. The Saved Job reuses the same bullet block so file and message agree
- **Build Deterministic Warnings**: derive warnings in code independent of the rater — hybrid/on-site (with "not an acceptable hybrid location" where it applies), required relocation, contract-vs-full-time, and missing salary. The rater cannot be trusted to self-report these: it once saw "Hybrid - 2-3 days onsite" and rated the job 4/5 without warning. Salary-below-target is deliberately left to the LLM, since parsing multi-currency day rates into a CAD annual figure is too brittle for a deterministic rule

### Non-functional Requirements

- **Scraper tool surface** — Stage 1b restricts the Playwright toolset via `disallowed_tools` (`SCRAPER_DISALLOWED_BROWSER_TOOLS`), which is the only option that removes a tool from the model's context. `allowed_tools` does **not** do this: it only auto-grants permission, which `permission_mode='bypassPermissions'` already grants, making it inert — measured, the model saw all 24 Playwright tools with `allowed_tools` set to 4. `browser_evaluate` and `browser_click` are load-bearing for discovery (inner-container scroll lazy-loads 7 → 10 listings; Next yields a fresh page) and a unit test asserts they are never disallowed
- **Cost efficiency** — Haiku for scraping, page extraction, and applied-job metadata; the agentic browser work never runs on Sonnet. Query generation is the one deliberate exception: a single Sonnet call per run (~$0.33), because top-of-funnel query relevance is the pipeline bottleneck and everything downstream is gated by it. Applied-job metadata is cached in `index.yaml` by mtime, so the extraction cost is paid once per PDF. Hard rules and local triage reject clear non-fits for $0 before any paid rating call. Since LinkedIn no longer enforces `f_E` server-side, a $0 title check at queue time (`title_rejection_reason()`) keeps below-seniority and people-management listings out of the paid Stage 2 path, where each would cost an extract plus a rating call; it is deliberately conservative (word-boundary matches only, and neither `associate` nor bare `graduate` rejects) because an auto-skip is unappealable, and it never filters on work arrangement, which is unreliable on a search card. The rating call is a single non-agentic structured-output call (provider configurable; Sonnet pinned explicitly by default, never the CLI default model). Reference jobs are distilled once into a ~2.5K-char cached profile instead of a ~60K-char block; the evaluator prompt is built once per run and reused across all jobs to maximise prompt-cache hits; the extractor is capped at 8 turns and restricted to the browser tools it needs. **Stage 1b is the one place where cost is explicitly not the priority** — Account safety is. Its turn budget was raised to 180 and its prompt is deliberately verbose, because selecting listings one at a time with randomised pauses costs turns and wall-clock by design. Do not optimise Stage 1b for speed or turn count; optimise what enters the model's context instead
- **Observability** — the generated Search Queries are logged; every `check_and_record_job` outcome is logged (`new`/`already_processed`/`already_applied`/`too_old`/`auth_required`) so seen-but-deduped is distinguishable from never-seen; per-job logs of the extract signal (`date_posted`/`location`/`closed`/`language_requirement`/`relocation`/`education_requirement`), extract compression, hard-rule short-circuits, triage score vs. final rating, and rating provider. Each run emits a single `Run funnel: {...}` line (queries → listings seen → check-status counts → candidates → extract ok/failed → hard-ruled by reason → triaged-out → rated 1–5), also written to `cost_log.jsonl` under `funnel`. Per-stage costs (`reference_summary`, `extraction`, `rating`) in `cost_log.jsonl`. Cost is charged **per session**: `total_cost_usd` is cumulative within a `ClaudeSDKClient` session, so only the increase over that session's last reported value is billed (`cost_delta_for` in `agent.py`, keyed on `session_id`). Stage 1b shares one session across all queries and was previously over-counted ~3.5x; stages that open a fresh session per call were always correct. Token counters in `msg.usage` are per-request and keep summing. Every `ResultMessage` is logged with `session_id`, `num_turns`, usage, and `cost_reported` vs `cost_charged` — previously these went to the `rich` console only and never reached the run log, which is why the over-count went unnoticed for nine runs. Third-party HTTP transport logging (`httpx` / `httpcore` / `mcp.client.streamable_http`, one line per MCP tool call) is suppressed to WARNING so the run log stays the run's own audit trail; set `HTTP_LOG_LEVEL=INFO` to restore it when debugging a tool server. **Agent `TextBlock` narration is mirrored to the run log** via `log_agent_text()`, not just `print()`ed to the console — that is where the scraper says "these two searches returned identical results" or "the Remote filter did not stick", and discarding it left the 2026-08-11 collapse diagnosable only from counters. `check_status` is reported **per query** as well as run-global (`_check_status_per_query`, in the Stage 1b log line, audit §2/§3 and the run funnel), so a dedup-saturated query is distinguishable from one that barely ran — a distinction the run-global count cannot make
- **Account safety** — Stage 1b drives a **real logged-in LinkedIn account** through the shared Playwright profile (`~/.linkedin-agent-profile`), and a flagged or banned account ends the job search permanently. This outranks coverage, cost, and run time: a slow incomplete run always beats an aggressive one. All scraping goes through the Playwright profile; never scrape from the user's own browser. Specifically:
  - **Nothing in the job results list is ever clicked.** Not a card, not a title, not a logo. The only real `<button>` in a row is **Dismiss**, which the accessibility tree disguises as the card itself, and one stray click permanently removes a job from the User's feed — this already happened, to three real jobs. Clicking is also unnecessary: Harvest Job Listings reads everything from the DOM. This is an invariant, not a preference, and a unit test asserts the prompt states it
  - **JavaScript may read the page, never drive it** — `browser_evaluate` for extraction and scrolling only; never to click, submit, or dispatch events
  - **Never apply, save, follow, or dismiss** anything
  - **Randomised pacing** between searches (`SCRAPER_INTER_SEARCH_DELAY_SECONDS`) and between queries (`SCRAPER_INTER_QUERY_DELAY_SECONDS`). Every delay is a **(min, max) range drawn uniformly** — a fixed interval is itself a robotic signature, and a unit test rejects degenerate ranges. Inter-query pacing is enforced **in code** (`_human_pause`), not merely prompted, because a model under turn pressure will skip a prompted wait. There is no per-listing delay any more: harvesting is one read-only call, so there is no interaction burst to disguise
  - **Stop on any CAPTCHA, verification challenge, or "unusual activity" notice** rather than work around it, and the recovery pass explicitly refuses to retry into a block. A challenge means LinkedIn already suspects automation; solving or immediately retrying converts a soft signal into a confirmed evasion pattern, which is what escalates to a restriction. Backing off keeps it a blip — the 2026-08-06 run went 0 listings at 11:43 and 74 at 12:08 after a pause. This applies only to *challenge pages*; zero results or a slow load still get a normal retry
- **Search coverage** — every generated Search Query must actually be searched, once per configured region. Stage 1b sends one request per query (own turn budget) on one shared session (one browser attachment); the run audit log distinguishes "never searched" from "searched, found nothing", which a single `0 jobs` count cannot. Turn starvation is silent — the model stops mid-query and the run still reports a listing count — so `SCRAPER_MAX_TURNS_PER_QUERY` (180) is set with headroom: harvesting is one call per search, but each listing still costs a `check_and_record_job` (and possibly a `queue_candidate`) turn, so the budget scales with `SCRAPER_MAX_LISTINGS_PER_SEARCH` × regions, and `num_turns` is logged per request to make exhaustion visible. **Under-coverage is detected by yield and by turn usage, not only by a zero count**: a query below `SCRAPER_MIN_LISTINGS_PER_QUERY` (5) is retried, and one that stops well short of its turn budget is warned about. Stopping *early* is as much a failure mode as running out — on 2026-08-11 every query used 10 of 90 turns and returned exactly one listing, and a `seen == 0` trigger missed all of it
- **Rejection auditability** — `python main.py -n --audit` runs normally but additionally re-rates every gate-killed Job Posting (hard-ruled or triaged-out) with the strong rater, logging any **false negative** (gate dropped it but the strong rater scores ≥3) plus a run-end count. Diagnostic only — saving/notification behaviour is unchanged
- **Resilience** — the LLM MCP tool servers are optional: summarization falls through its provider chain (last resort: full reference block), triage fails open to the rating call
- **Idempotency** — processed-job records persist across runs so jobs are never evaluated twice; applied-job ingest is a no-op for already-dated files
- **Applied-date durability** — an Applied Job Record's date is carried by the filename prefix, `index.yaml`, and mtime independently, so it survives a move, copy, or backup restore that drops filesystem metadata
- **Applied-job horizon** — only Applied Job Records from the last `APPLIED_JOBS_HORIZON_DAYS` (90) feed query generation, the ideal-role profile, and the already-applied blocklist; older PDFs are retained on disk, never deleted
- **Notification latency** — Telegram alerts sent immediately when a job is rated 4 or 5 during evaluation
- **Rating hard rules (applied deterministically in code before any LLM scoring)**: closed postings → 1; postings > 30 days old → 1; jobs in a `sponsorship_required_in` location with no sponsorship offered → 1; a required language outside `languages` → 1; a hard requirement for a degree in `reject_required_degrees` → 1. The last three are preference-driven and disabled when their lists are empty. Required relocation is flagged, never auto-rejected. The degree rule fires **only on hard requirements** — "MSc preferred", "Master's or equivalent experience", and "Bachelor's or Master's" all survive, because an auto-reject is unappealable and a posting that would accept experience instead must never be killed
- **Rating cap (applied deterministically in code AFTER LLM scoring)**: a `hybrid` or `onsite` Job Posting whose location is not in the Preferences `hybrid.acceptable_locations` is capped at `hybrid.rating_cap`. This is a **ceiling, not a rejection or a floor** — the job is saved, recorded, and auditable, and a rating already at or below the cap is untouched; it simply cannot reach the ≥ 4 notification threshold. The cap exists because the prompt alone is not sufficient: the evaluator rated a hybrid role in an unacceptable location 4/5 while naming the hybrid location as a drawback in its own reasoning. Every cap is logged (`Rating capped: … 4 → 3 (reason)`) and counted in the run funnel as `rating_capped`
- **Knowledge locality** — project decisions, diagnoses, and conventions live in tracked repo files. Machine-local paths (including `~/.claude` and its per-project memory directory) are never used for project knowledge, because they do not reach another machine. `run_dir/` is the one deliberate exception: gitignored because it holds personal data (resume, applied-job PDFs, saved postings, `JOB_REQUIREMENTS.md`, `preferences.yaml`)
- **No personal information in tracked files** — nothing in git may identify or describe whoever is running the agent: no home location, work-authorization or immigration status, languages, education, employers or recruiters dealt with, compensation targets, or absolute paths containing a username. Every such fact is a **preference**, and preferences live only in `run_dir/preferences.yaml`. Tracked code reads them through `preferences.py` and must behave sanely when they are absent — the defaults are neutral, so an unconfigured checkout applies no work-authorization, language, or education gate rather than inheriting someone else's situation. Tests pin their own fixed preferences in `tests/conftest.py` and must never read the real file. When adding a rule that encodes a personal fact, add a preference key; do not hardcode the fact

## Git conventions

- Use `git mv` when moving or renaming tracked files
- Use `git rm` when deleting tracked files
