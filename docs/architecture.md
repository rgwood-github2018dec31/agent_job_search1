# Architecture and data model

## Applied-job corpus (`run_dir/applied_jobs/`)

Jobs the user has applied to are the strongest available signal of what to search for. PDFs are
saved to the configured save directory (the `save_dir` preference, default `~/Downloads`),
categorized by `categorize_save_dir_pdfs()` into `cat-saved_jd-*.pdf`, then moved into
`run_dir/applied_jobs/` by `ingest_save_dir_applied_pdfs()` on every startup, renamed to
`{YYYY-MM-DD}-{original}.pdf`.

**Two distinct directories, never conflated.** `save_dir` is the **source**: external,
user-controlled, configurable, and only ever globbed and moved *out of*. `APPLIED_JOBS_DIR`
(`run_dir/applied_jobs/`) is the **destination**: agent-owned, deliberately **not** configurable,
and the only directory `load_applied_jobs()` reads. A PDF merely sitting in the save directory is
invisible to the corpus — it must be ingested first — which is what keeps a saved-but-not-applied
posting out of the reference profile and the already-applied blocklist. Functions touching both
take them as separate `save_dir` / `applied_to_dir` parameters; neither ever defaults from the
other. A unit test pins that `load_applied_jobs()` ignores the save directory.

The applied date is carried three ways, most durable first: the filename prefix, the
`applied_date` in `index.yaml`, and the file mtime (restored via `os.utime` after the move).
mtime alone is untrustworthy — any copy, backup restore, or rsync rewrites it.

`load_applied_jobs()` reads the corpus into three globals, keyed on `index.yaml` (which caches
extracted text and metadata per filename by mtime, so the Haiku metadata call only runs for new
or changed PDFs):

- `_applied_jobs` — feeds `applied_jobs_summary()` into the Stage 1a query prompt
- `_reference_job_pages` — per-page text of each job, feeding the ideal-role profile used by the rater. Pages are joined only when a prompt is built, by `text_budget.pages_to_prompt()`, which logs chars per page and normalizes, then truncates, only past the cap, with a WARNING each time. Ordered **newest applied first**, sorted by `applied_date` rather than filename, because both consumers cap it at `MAX_REFERENCE_JOBS` and taking the oldest `MAX_REFERENCE_JOBS` meant a newly applied job could never influence the profile (2026-08-25; see Summarize Reference Jobs in [requirements.md](requirements.md))
- `_applied_companies` — the already-applied blocklist used by `check_and_record_job`

Only records within `APPLIED_JOBS_HORIZON_DAYS` populate these; older PDFs stay on disk but
are never read.

**Recruiters:** `_extract_applied_job_metadata()` returns `is_agency` and `end_client` alongside
company and title. An agency's own name never enters the blocklist — applying once through a
staffing firm or job aggregator would otherwise suppress every other company it posts for. The end client is blocklisted instead when the posting names one; agency
postings still contribute reference and query signal.

## Architecture

The project uses the **[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)** (`claude-agent-sdk>=0.2.139`) as its foundation with **MCP (Model Context Protocol)** for tool use.

### Model selection and routing (`config.py`)

All model constants are defined once in the shared `utils_tools_n_agents_common.models` library (`ANTHROPIC_MODEL_NAME_*` for the Claude tiers, `OPENROUTER_MODEL_NAME_*` for the OpenRouter ids); call sites import them from the library directly — a model swap happens in the library, not here. **This file references const names only and never restates their values** — the library is the single source of truth for what they resolve to, so a doc value cannot drift from the code. Version pins (`PLAYWRIGHT_MCP_VERSION`, `mcp<2`) follow the same single-source rule: the value is written once, at the const or the dependency pin, and comments/docs refer to it by name.

**One const per stage; the family prefix IS the route.** There are no `*_PROVIDER` selector strings — two encodings of one decision drift apart, so each stage has a single `MODEL_NAME_<role>` const whose value points at a shared family const, and `models.route_for()` dispatches on the id-shape contract (`vendor/model` → OpenRouter MCP :8006, `name:tag` → Ollama MCP :8002, bare name → Anthropic SDK). Repoint a const at a different family to change the route, at a different const within a family to change only the model:

- `MODEL_NAME_QUERY`, `MODEL_NAME_COMPANY_MATCH`, `MODEL_NAME_RATING` — query generation, company matching, and the final rating call; all default to `OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE`. Each falls back to Anthropic (`ANTHROPIC_MODEL_NAME_MEDIUM` for rating, `ANTHROPIC_MODEL_NAME_LOW` for company match) if the OpenRouter MCP server is down. Point `MODEL_NAME_RATING` at `OLLAMA_MODEL_NAME_TRIAGE` to rate via the local Ollama server, or at `ANTHROPIC_MODEL_NAME_MEDIUM` to rate via the SDK
- `MODEL_NAME_EXTRACTOR` — page extraction; defaults to `OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC` (the function-calling agent loop in `extract_openrouter.py` drives the browser tools via the OpenRouter MCP server's `chat` tool with OpenAI-style `tools`; capped at `EXTRACTOR_OPENROUTER_MAX_ITERATIONS`, tool results truncated to `EXTRACTOR_TOOL_RESULT_MAX_CHARS`). Pointing it at an `ANTHROPIC_*` const selects the agentic SDK session instead. The deterministic Haiku fallback covers loop failures either way
- `MODEL_NAME_SCRAPER` — Stage 1b scraping; defaults to `OPENROUTER_MODEL_NAME_SCRAPER` via the function-calling loop in `scrape_openrouter.py`. Pointing it at an `ANTHROPIC_MODEL_NAME_*` const routes Stage 1b through the original `ClaudeSDKClient` scraper, kept intact as the rollback path and used automatically if the OpenRouter loop fails. Measured 2026-08-21 on one live search: **$0.0287 vs $0.4330** for identical traffic on Haiku (**9.3x**), because the DeepSeek pin caches implicitly (~88% hit rate) and has **no cache-write fee** — cache writes were 42% of the Haiku bill. **Do not point this at `OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE`**: glm-5.2 measured $0.1932/M cache-read — ~2x Haiku's rate, costing *more* than what it replaces — and the same trap applied to `deepseek-v4-pro` and `qwen3.8-max`. The win is specific to the flash tier
- `ANTHROPIC_MODEL_NAME_HIGH` — audit reference standard only (`--audit-opus`); never used in the normal pipeline
- `OLLAMA_MODEL_NAME_TRIAGE` — machine-local Ollama tag for Stage 2c triage (the one model constant defined in this repo, not the library — it is machine-local and can vanish on a re-pull, which `preflight_local_model()` catches)
- Non-Anthropic models are reached via the LLM MCP tool servers (`LLM_OPENROUTER_MCP_URL` :8006, `LLM_LOCAL_MCP_URL` :8002) using `call_mcp_tool()` in `triage.py`. **Always route OpenRouter/Ollama inference through these servers, never the raw HTTP APIs** — that is where keys, config, and per-call `cost_usd` tracking live. Note the local Ollama `generate` tool has **no default model configured**, so always pass `model` explicitly

### Non-interactive pipeline (`run_non_interactive` in `agent.py`)

1. **Stage 1a — Query generation** (glm, Anthropic fallback): `generate_search_queries()` derives at most `MAX_SEARCH_QUERIES` LinkedIn search queries from resume + `JOB_REQUIREMENTS.md` + the in-horizon applied-job titles (`applied_jobs_summary()`). Individual-contributor titles only — people-management titles are explicitly excluded
2. **Stage 1b — Scraping** (`OPENROUTER_MODEL_NAME_SCRAPER` via `scrape_openrouter.py`; Haiku on the Anthropic fallback): `run_scraper()` issues **one request per query**, calling `run_ui_contract` → `harvest_listings` → `record_listings` per search without navigating to individual job pages. `run_scraper` is provider-agnostic — it takes a `run_pass(instruction)` callable, so its per-query loop, randomised pacing, region-verification check and low-yield recovery pass are shared rather than duplicated. On the **OpenRouter** path each request is a fresh conversation while the browser session persists: on the Anthropic path all six queries shared one transcript, so context per turn grew 41K → 162K across a run and every later query paid for every earlier snapshot. On the **Anthropic** path the one-request-per-query split is still load-bearing — batching all queries into one request lets the first few exhaust `SCRAPER_MAX_TURNS_PER_QUERY` so the rest are never searched (while still reporting `0 jobs`), and opening a fresh client per query makes later ones fail with "browser is in use" against the shared Playwright context, which looks identical to an auth wall in the logs
3. **Stage 2 — Evaluation** (`evaluate_all_candidates()`), per candidate:
   - **2a Extract** (Haiku, agentic): `extract_job_page()` navigates to the job URL, expands the description, and submits a condensed extract via `submit_job_extract` (nav chrome and boilerplate stripped). If the session ends without submitting (Haiku can exhaust its turn budget hunting through truncated snapshots of large pages), `extract_job_page_direct()` falls back to a deterministic path: navigate + wait + snapshot (+ one "… more" expand click) driven directly over the Playwright MCP session, then one non-agentic Haiku call condenses the full snapshot — same browser and logged-in profile, so bot-detection exposure is identical
   - **2b Hard rules** ($0 on the happy path): `apply_hard_rules()` auto-rates 1 for a **blacklisted company** (poster or `end_client`; checked first, free unless the exact case-sensitive name matches, in which case one cheap confirmation call runs — see Reject Blacklisted Company), closed postings, postings > `JOB_STALE_AGE_DAYS` days old, jobs in a `sponsorship_required_in` location without explicit sponsorship, jobs requiring a language outside `languages` (from the extract's `language_requirement` field), and jobs that hard-require a degree listed in `reject_required_degrees` (from the extract's `education_requirement` field, backed by `derive_education_requirement()`). The blacklist and the three location/language/education gates are **preference-driven** — each is off when its preference list is empty. A required relocation (`relocation` field) does NOT reject — it is flagged as "relocation required: <location>" in the saved job and the Telegram notification, and the evaluator is given `relocation_note` as guidance. Every rejection is logged with its reason (console + `run_dir/logs/run-*.log`)
   - **2c Triage** ($0, local LLM): `triage_job_fit()` scores fit 1–5 with Ollama; scores ≤ `TRIAGE_THRESHOLD` are saved with the triage score and skip the rating call; fails open if the server is down
   - **2d Rating** (configurable): `rate_job()` makes one non-agentic structured-output call, then `apply_rating_caps()` applies the deterministic ceilings (hybrid location, foreign `posting_language`; lowest cap wins, all reasons reported), saves via `do_save_job_posting()`, and sends a Telegram notification for ratings ≥ 4

Before Stage 2, `build_reference_summary()` distills the reference-job PDFs into an "ideal role profile" of at most `REFERENCE_SUMMARY_MAX_CHARS` (provider chain: OpenRouter → local Ollama → Anthropic Haiku; falls back to the full reference block if all fail). The result is cached in `run_dir/reference_summary_cache.yaml` keyed by an md5 of the reference texts, so it is only regenerated when the PDFs change. The `MAX_REFERENCE_JOBS` it distills are the **most recently applied**, which is the whole point of the cap — a profile built from the oldest ones cannot track a change in direction, and reports `cache hit` while doing it. The evaluator prompt embeds this summary instead of the full reference block (`MAX_REFERENCE_JOBS` jobs of up to `REFERENCE_JOB_PROMPT_MAX_CHARS` each).

Key per-job observability is logged via the `logging` module: extract input tokens vs. condensed size, hard-rule short-circuits, triage score vs. final rating, and the rating model used (the log line names the `MODEL_NAME_RATING` value). Per-stage costs land in `cost_log.jsonl` under `reference_summary` / `extraction` / `rating` (triage and hard rules are free), charged per session — see the Observability requirement.

### MCP server factories (`tools_generic.py`)

| Factory | Used in | Tools provided |
|---|---|---|
| `make_job_search_server(interactive)` | Interactive mode | `check_and_record_job`, `save_job_posting`, `update_job_requirements` (if interactive) |
| `make_scraper_server()` | Stage 1b (**Anthropic fallback only**) | `check_and_record_job`, `queue_candidate`, `report_search` |
| `make_evaluator_server()` | Stage 2a | `submit_job_extract` |

In Stage 2, saving and Telegram notification are plain Python calls (`do_save_job_posting()`, `_send_pipeline_notification()`), not MCP tools.

### Tools (`tools_generic.py`)

| Tool | Purpose |
|---|---|
| `check_and_record_job` | Returns `already_processed`, `too_old`, or `new`; records to `processed_jobs/*.yaml` |
| `save_job_posting` | Saves evaluated job to `saved_jobs-{date}/job_posting-*.md` |
| `queue_candidate` | Adds a job to the in-memory evaluation queue (Stage 1b only); skips below-seniority / people-management titles for $0 |
| `do_record_listings` | Records a whole search's harvest in one call **from code's copy**; enforces `SCRAPER_MAX_LISTINGS_PER_SEARCH`, returns counts only |
| `run_ui_contract` / `harvest_listings` / `record_listings` / `report_problem` | The Stage 1b OpenRouter toolset. **None accepts page or job data** — see Anti-fabrication |
| `submit_job_extract` | Captures the condensed page extract from the Stage 2a extractor |
| `update_job_requirements` | Rewrites `JOB_REQUIREMENTS.md` (interactive mode only) |

## Job deduplication and age filtering

On startup, `load_processed_jobs()` populates `_processed_jobs` from two sources:
- **Old format**: extracts LinkedIn job IDs from `saved_jobs-*/job_posting-*.md` filenames
- **New format**: loads `(site, job_id)` pairs from `processed_jobs/*.yaml` files

At runtime, `check_and_record_job` enforces:
- **Skip** if `(site, job_id)` is in `_processed_jobs` → returns `already_processed`
- **Skip** if posting date is parseable and older than `JOB_MAX_AGE_DAYS` → returns `too_old` (an unparseable date logs one WARNING and counts as unknown age)
- **Proceed** otherwise → writes a YAML record, adds to `_processed_jobs`, returns `new`

`date_posted` accepts absolute (`YYYY-MM-DD`) or relative (`"4 days ago"`) formats; omit if not shown.
