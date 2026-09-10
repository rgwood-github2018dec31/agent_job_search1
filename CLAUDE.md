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
  triage.py                       # LLM MCP-server client, local triage, non-Anthropic rating calls
  scrape_openrouter.py            # Stage 1b: OpenRouter function-calling scraper loop (default path)
  extract_openrouter.py           # OpenRouter function-calling agent loop for page extraction
  config.py                       # Model/provider constants and Stage 2 tuning (nothing personal)
  preferences.py                  # Loads run_dir/preferences.yaml; neutral defaults if absent
  location.py                     # Cached geographic classifier for a job's location
scripts/
  migrate_applied_jobs.py         # One-time reviewable move of applied-job PDFs into run_dir
  backtest_location_gate.py       # Replays saved postings through the location gate (live classifier)
tests/
  test_tools.py                   # Unit tests for all tools
run_dir/
  JOB_REQUIREMENTS.md             # Agent-managed job preferences (interactive mode)
  applied_jobs/                   # Applied-job PDFs, date-prefixed, + index.yaml metadata
  saved_jobs-{date}/              # Evaluated job postings (Markdown)
  processed_jobs/                 # Per-job YAML records for deduplication
  audit_logs/                     # Per-run audit trace (audit-{date}-{time}.md)
  reference_summary_cache.yaml    # Cached distilled ideal-role profile (md5-keyed)
  location_cache.yaml             # Cached geography per location string (country/region/language)
  location_recommendations.yaml   # Advisory review of the country lists (recommend-only, hand-edited)
  preferences.yaml                # Personal preferences (regions, gates, titles) — gitignored
  logs/                           # Per-run log files (rejection reasons, extract sizes, ratings)
preferences.example.yaml          # Tracked, neutral template for run_dir/preferences.yaml
```

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
- `_reference_job_texts` — feeds the ideal-role profile used by the rater. Ordered **newest applied first**, sorted by `applied_date` rather than filename, because both consumers cap it at `MAX_REFERENCE_JOBS` (20) and taking the oldest 20 meant a newly applied job could never influence the profile (see the 2026-08-25 entry)
- `_applied_companies` — the already-applied blocklist used by `check_and_record_job`

Only records within `APPLIED_JOBS_HORIZON_DAYS` (90) populate these; older PDFs stay on disk but
are never read.

**Recruiters:** `_extract_applied_job_metadata()` returns `is_agency` and `end_client` alongside
company and title. An agency's own name never enters the blocklist — applying once through a
staffing firm or job aggregator would otherwise suppress every other company it posts for. The end client is blocklisted instead when the posting names one; agency
postings still contribute reference and query signal.

## Architecture

The project uses the **[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)** (`claude-agent-sdk>=0.2.139`) as its foundation with **MCP (Model Context Protocol)** for tool use.

### Model selection and routing (`config.py`)

All model constants are defined once in the shared `utils_tools_n_agents_common.models` library (`ANTHROPIC_MODEL_NAME_*` for the Claude tiers, `OPENROUTER_MODEL_NAME_*` for the OpenRouter ids); call sites import them from the library directly — a model swap happens in the library, not here. **This file references const names only and never restates their values** — the library is the single source of truth for what they resolve to, so a doc value cannot drift from the code.

**One const per stage; the family prefix IS the route.** There are no `*_PROVIDER` selector strings — two encodings of one decision drift apart, so each stage has a single `MODEL_NAME_<role>` const whose value points at a shared family const, and `models.route_for()` dispatches on the id-shape contract (`vendor/model` → OpenRouter MCP :8006, `name:tag` → Ollama MCP :8002, bare name → Anthropic SDK). Repoint a const at a different family to change the route, at a different const within a family to change only the model:

- `MODEL_NAME_QUERY`, `MODEL_NAME_COMPANY_MATCH`, `MODEL_NAME_RATING` — query generation, company matching, and the final rating call; all default to `OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE`. Each falls back to Anthropic (`ANTHROPIC_MODEL_NAME_MEDIUM` for rating, `ANTHROPIC_MODEL_NAME_LOW` for company match) if the OpenRouter MCP server is down. Point `MODEL_NAME_RATING` at `OLLAMA_MODEL_NAME_TRIAGE` to rate via the local Ollama server, or at `ANTHROPIC_MODEL_NAME_MEDIUM` to rate via the SDK
- `MODEL_NAME_EXTRACTOR` — page extraction; defaults to `OPENROUTER_MODEL_NAME_DEFAULT_AGENTIC` (the function-calling agent loop in `extract_openrouter.py` drives the browser tools via the OpenRouter MCP server's `chat` tool with OpenAI-style `tools`; capped at `EXTRACTOR_OPENROUTER_MAX_ITERATIONS`, tool results truncated to `EXTRACTOR_TOOL_RESULT_MAX_CHARS`). Pointing it at an `ANTHROPIC_*` const selects the agentic SDK session instead. The deterministic Haiku fallback covers loop failures either way
- `MODEL_NAME_SCRAPER` — Stage 1b scraping; defaults to `OPENROUTER_MODEL_NAME_SCRAPER` via the function-calling loop in `scrape_openrouter.py`. Pointing it at an `ANTHROPIC_MODEL_NAME_*` const routes Stage 1b through the original `ClaudeSDKClient` scraper, kept intact as the rollback path and used automatically if the OpenRouter loop fails. Measured 2026-08-21 on one live search: **$0.0287 vs $0.4330** for identical traffic on Haiku (**9.3x**), because the DeepSeek pin caches implicitly (~88% hit rate) and has **no cache-write fee** — cache writes were 42% of the Haiku bill. **Do not point this at `OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE`**: glm-5.2 measured $0.1932/M cache-read — ~2x Haiku's rate, costing *more* than what it replaces — and the same trap applied to `deepseek-v4-pro` and `qwen3.8-max`. The win is specific to the flash tier
- `ANTHROPIC_MODEL_NAME_HIGH` — audit reference standard only (`--audit-opus`); never used in the normal pipeline
- `OLLAMA_MODEL_NAME_TRIAGE` — machine-local Ollama tag for Stage 2c triage (the one model constant defined in this repo, not the library — it is machine-local and can vanish on a re-pull, which `preflight_local_model()` catches)
- Non-Anthropic models are reached via the LLM MCP tool servers (`LLM_OPENROUTER_MCP_URL` :8006, `LLM_LOCAL_MCP_URL` :8002) using `call_mcp_tool()` in `triage.py`. **Always route OpenRouter/Ollama inference through these servers, never the raw HTTP APIs** — that is where keys, config, and per-call `cost_usd` tracking live. Note the local Ollama `generate` tool has **no default model configured**, so always pass `model` explicitly

### Non-interactive pipeline (`run_non_interactive` in `agent.py`)

1. **Stage 1a — Query generation** (glm, Anthropic fallback): `generate_search_queries()` derives at most `MAX_SEARCH_QUERIES` (6) LinkedIn search queries from resume + `JOB_REQUIREMENTS.md` + the in-horizon applied-job titles (`applied_jobs_summary()`). Individual-contributor titles only — people-management titles are explicitly excluded
2. **Stage 1b — Scraping** (`OPENROUTER_MODEL_NAME_SCRAPER` via `scrape_openrouter.py`; Haiku on the Anthropic fallback): `run_scraper()` issues **one request per query**, calling `run_ui_contract` → `harvest_listings` → `record_listings` per search without navigating to individual job pages. `run_scraper` is provider-agnostic — it takes a `run_pass(instruction)` callable, so its per-query loop, randomised pacing, region-verification check and low-yield recovery pass are shared rather than duplicated. On the **OpenRouter** path each request is a fresh conversation while the browser session persists: on the Anthropic path all six queries shared one transcript, so context per turn grew 41K → 162K across a run and every later query paid for every earlier snapshot. On the **Anthropic** path the one-request-per-query split is still load-bearing — batching all queries into one request lets the first few exhaust `SCRAPER_MAX_TURNS_PER_QUERY` so the rest are never searched (while still reporting `0 jobs`), and opening a fresh client per query makes later ones fail with "browser is in use" against the shared Playwright context, which looks identical to an auth wall in the logs
3. **Stage 2 — Evaluation** (`evaluate_all_candidates()`), per candidate:
   - **2a Extract** (Haiku, agentic): `extract_job_page()` navigates to the job URL, expands the description, and submits a condensed extract via `submit_job_extract` (nav chrome and boilerplate stripped). If the session ends without submitting (Haiku can exhaust its turn budget hunting through truncated snapshots of large pages), `extract_job_page_direct()` falls back to a deterministic path: navigate + wait + snapshot (+ one "… more" expand click) driven directly over the Playwright MCP session, then one non-agentic Haiku call condenses the full snapshot — same browser and logged-in profile, so bot-detection exposure is identical
   - **2b Hard rules** ($0 on the happy path): `apply_hard_rules()` auto-rates 1 for a **blacklisted company** (poster or `end_client`; checked first, free unless the exact case-sensitive name matches, in which case one cheap confirmation call runs — see Reject Blacklisted Company), closed postings, postings > `JOB_STALE_AGE_DAYS` (30) days old, jobs in a `sponsorship_required_in` location without explicit sponsorship, jobs requiring a language outside `languages` (from the extract's `language_requirement` field), and jobs that hard-require a degree listed in `reject_required_degrees` (from the extract's `education_requirement` field, backed by `derive_education_requirement()`). The blacklist and the three location/language/education gates are **preference-driven** — each is off when its preference list is empty. A required relocation (`relocation` field) does NOT reject — it is flagged as "relocation required: <location>" in the saved job and the Telegram notification, and the evaluator is given `relocation_note` as guidance. Every rejection is logged with its reason (console + `run_dir/logs/run-*.log`)
   - **2c Triage** ($0, local LLM): `triage_job_fit()` scores fit 1–5 with Ollama; scores ≤ `TRIAGE_THRESHOLD` (1) are saved with the triage score and skip the rating call; fails open if the server is down
   - **2d Rating** (configurable): `rate_job()` makes one non-agentic structured-output call, then `apply_rating_caps()` applies the deterministic ceilings (hybrid location, foreign `posting_language`; lowest cap wins, all reasons reported), saves via `do_save_job_posting()`, and sends a Telegram notification for ratings ≥ 4

Before Stage 2, `build_reference_summary()` distills the reference-job PDFs into a ~2.5K-char "ideal role profile" (provider chain: OpenRouter → local Ollama → Anthropic Haiku; falls back to the full reference block if all fail). The result is cached in `run_dir/reference_summary_cache.yaml` keyed by an md5 of the reference texts, so it is only regenerated when the PDFs change. The 20 it distills are the **most recently applied**, which is the whole point of the cap — a profile built from the oldest 20 cannot track a change in direction, and reports `cache hit` while doing it. The evaluator prompt embeds this summary instead of the former 20×3,000-char reference block.

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
- **Skip** if posting date is parseable and > 21 days old → returns `too_old`
- **Proceed** otherwise → writes a YAML record, adds to `_processed_jobs`, returns `new`

`date_posted` accepts absolute (`YYYY-MM-DD`) or relative (`"4 days ago"`) formats; omit if not shown.

## Known diagnoses

### 2026-09-10 — a filter alert for a search that was already fixed

The run summary said `FILTERS DID NOT APPLY on Principal Data Scientist / Canada: location:
geoId=101174742 absent from the URL`. **The check was right, and no bad data was harvested.** What
happened: after "Show results" on the Experience level chip, LinkedIn put the account's salary
param back (`f_SAL=f_SA_id_227001:277001`, the known self-injection). The prompt described the
empty `f_SAL=` as clearing a filter that "silently narrows every search", so the model navigated
to the bare URL again to get rid of it. That dropped the `geoId` it had just clicked in, but the
location chip is sticky and kept **showing** "Canada". The model trusted the chip, and
`check_filters_applied` caught it. The model re-clicked the location chip, the second check passed,
and it harvested only after that. On query 3 of the same run the model saw the same `f_SAL` and
left it alone. Same facts, two different decisions: model discretion again.

Two fixes. The prompt now says the reappearing `f_SAL` is expected and that you never navigate
again once a filter has been clicked (a restart must re-click every chip, location included). And
a passing `report_search` now marks earlier `filters` alerts for the **same query and region** as
`resolved`, which `assess_run_health` skips. Audit §1b still lists both reports, so the fail→ok
trail is kept. A pass on a different region or query never resolves one.

**The more expensive failure in the same query was a 502.** After the Canada harvest, one
`502 Bad Gateway` from OpenRouter at iteration 71 raised out of `ScrapeSession.run`. The query was
marked FAILED and its EU search never ran. A 502 correctly isn't a `ProviderUnavailableError`, but
nothing retried it either. The loop now retries the same chat call once on 500/502/503/504.
401/402/403/429 still abort right away. It's still unknown whether the self-injected `f_SAL` really
narrows results: the model saw lower-band salaries in them and no salary chip. Nobody has tested it.

### 2026-09-09 — `roma` matched `Romania`, and the gate never ran

Two Romanian postings were rated **4** (notified) and **3** on Sep 07-08 —
`saved_jobs-2026Sep07/job_posting-4453702157-*` and `saved_jobs-2026Sep08/job_posting-4452991539-*`.
Both said `Romania (Remote within country)`. Both carried `Relocation required: Romania`. The
relocation hard rule that exists for exactly this was correct, and **never got to run.**

`hybrid_location_is_acceptable()` matched its tokens with a bare `token in haystack`, and
`hybrid.acceptable_locations` contains **`roma`** — Rome. `'roma' in 'romania (remote within
country)'` is `True`. That list is **tier 1 and exempts outright**, so for every Romanian posting
the deny list, the classifier and `reject_regions: [eastern_europe]` were never consulted. It
swallowed `parma, emilia-romagna, italy` too.

**Nothing looked wrong, and that is the shape worth remembering.** In the *same* run Bulgaria,
Poland, Czechia and Lithuania were auto-rejected correctly. A gate that fails for one country while
its four neighbours pass through it correctly produces no anomaly anywhere: the log line for an
exempted job reads exactly like the log line for a genuinely acceptable one, because "no rejection"
is what both look like. **The failure mode of a matching rule is silently *more permissive*.**

**The evidence was also unreachable from the cache.** Tier 1 short-circuits *before*
`classify_location`, so `location_cache.yaml` holds no record of the jobs the bad entry exempted.
`romania` is in it at all only because one posting phrased its location as `"romania (remote) — role
is remote anywhere in the eu"` and reached the classifier by a different route (`derive_local_language`,
since renamed `derive_implied_local_language`).
This is why the new reviewer's country universe comes from a **live run-scoped accumulator** rather
than from the cache — and why that accumulator records on the **cache-hit path too**, since in
steady state almost every location is a hit.

**Three more live collisions existed on the same list**, none of which had bitten yet: `turin` ⊂
`turing` (a location or relocation string naming Turing Labs would have exempted the job), `usa` ⊂
`jerusalem` / `sausalito`, and `milan` ⊂ `milano` (masked only because `milano` is separately
listed).

**The obvious fix is itself a trap.** `re.escape('u.s.')` is `u\.s\.`, and `\bu\.s\.\b` requires a
word character after the final `.` — so it matches **neither** `'u.s.'` nor `'remote, u.s. only'`.
Writing the boundary as `\b…\b` would have silently deleted a real entry while fixing the reported
one. The matcher uses letter/digit lookarounds instead, and
`test_location_token_matcher_handles_punctuation` fails if anyone "simplifies" it back.

**`tests/conftest.py` had been doing it right the whole time.** Its `_GEO_RE` is word-boundary
anchored, with a comment explaining that `nice` is inside `Venice` — *"The production classifier is
an LLM and has no such problem; the stub must not invent one."* The test double was more careful
than the production code it doubles, and nothing compared them.

**A closing note on why `roma` was ever on the list.** It is one of a set of hand-maintained
spelling pairs — `rome`/`roma`, `lisbon`/`lisboa`, `milan`/`milano` — added because the lists were
lowercase ASCII substrings and could not otherwise reach a place written in its own language. **The
workaround caused the bug it was working around.** The same representation also made `malaga` an
entry that could never match anything, because LinkedIn writes `Málaga` — and a dead entry is
indistinguishable from one that has not come up yet. The lists now hold proper names, matched
case-sensitively and accent-exactly, and an exonym no spelling can reach (`Torino` → `Turin`) is
resolved through the classifier's `place_names`, so the pairs no longer need maintaining by hand.

**Removing the folding is what forced the boundary class to change**, and the two cannot be
separated: `(?<![a-z0-9])` blocks only lowercase ASCII neighbours, which was correct only in
combination with a caller that lowercased first. Matching proper names with it, `ROMA` reaches into
`ROMANIA` and `Roma` into `Romaña` — this same bug, in caps and via an accent. One test covers all
three forms for that reason.

**And the fix nearly recommended the bug back.** The reviewer's new dead-entry check asks "would
this entry have matched if it were written the way the place is written", which means folding case
and accents. Written as a substring test, folded `roma` is inside folded `romania`, so it reports
*"'Roma' never matched, but 'Romania' did — write the entry the way the place is written."* Folding
is allowed to ignore case and accents; it is never allowed to ignore boundaries. Pinned by
`test_reviewer_never_advises_writing_roma_as_romania`.

**Separately, and deliberately: the anchor rule is now conditioned on residence.** The 2026-09-04
entry below rejected on the location the role is advertised in. This file has said three times that
*"a bare location is not a residency requirement"* — and the code did not act on it. It does now,
for `remote` postings only, via a new extractor field `residency_scope` (see Detect Residency
Scope). Measured over the saved corpus, that keeps **168** previously-rejected postings, including
5/5 roles in Poland, France and Berlin. That is a large, intentional loosening of a gate written
four days earlier, and it is what `relocation_note` has said all along. **What the 2026-09-04 entry
was actually built to stop is untouched**: hybrid and on-site roles are not spared, a stated
`relocation` is not spared, and `locations.exclude` is not spared — the spare sits *below* the deny
list, which after this change is the only way to drop one specific EU country for a remote role. An
early draft put it above, silently making that list inert, and only a test named for the escape
hatch caught it.

### 2026-09-04 — geography had no field, so the rater invented one per job

Seven remote roles in Germany and the Netherlands were rated 4-5 and notified across Sep 02-03
(Finom/Berlin 5, Flip/Stuttgart 5, team.blue/Berlin 5, NVIDIA/Germany 4, Archer/Germany 4,
LangChain/Amsterdam 4, Jobgether/NL 4). On the Sep 03 run **13 of 24 jobs rated 4-5 were in
non-English-speaking countries.** Nothing malfunctioned, and that is the finding: every gate
declined *correctly*.

| Gate | Field | Value | Outcome |
|---|---|---|---|
| hard language rule | `language_requirement` | `english` | no reject — correct |
| foreign-language cap | `posting_language` | `english` | no cap — correct |
| hybrid cap | `workplace_type` | `remote` | never consulted a location list |
| — | `implied_local_language` | `german` / `dutch` | warning bullet only, **by design** |

**There was no rule for "the location is fine to visit but not to live in."** `hybrid.acceptable_locations`
is the only place the north/south distinction was written down, and a remote job never reaches it.

**The rater filled the vacuum, inconsistently.** It capped BENCH/Munich to 3 in prose — *"relocation
required: Munich, Germany contradicts the remote label … capping the rating at 3"* — and left
Finom/Berlin and Flip/Stuttgart at 5 on identical facts. That is the hybrid-cap and Valtech lesson
for the third time: **a structural fact left to the model's discretion is decided differently every
time it comes up.**

**Three design traps, each of which was walked into before being caught.**

1. **"Non-English AND not on the acceptable list" is the wrong rule, and it passes casual
   inspection.** It gets Germany and Spain right — for the wrong reason — and gets **Quebec wrong**:
   `acceptable_locations` lists `vancouver` and `british columbia`, not `canada`, so a
   French-language remote role in Montreal would have been rejected by a rule whose own regression
   test (`test_valtech_regression_french_posting_is_capped_and_warned`) demands it only warn. The
   fix is that the gate is **purely geographic and reads no language field at all**. Spain is
   acceptable because Spain is a place you would be, not because of anything about Spanish.
2. **A bare location is not a residency requirement.** "Germany (Remote)" means the role is anchored
   in Germany while the holder could live anywhere in the EU. So `location` and an explicit
   `relocation` statement are judged separately: the first is a preference about where roles may be
   anchored, the second is the JD asserting a categorical requirement. `relocation: European Union`
   names no country and must not reject — it did in an early draft.
3. **Hand-enumerating "northern Europe" is guesswork that rots.** Is Slovenia Adriatic? Romania
   Mediterranean? Bare "France" — Paris or Nice? So the residue goes to a **cached** classifier, and
   the live backtest immediately justified it: it split `Aix-en-Provence → southern_europe` (keep)
   from `Hauts-de-France → western_europe` (reject) with no city list at all, which no list I wrote
   by hand had done.

**Why an LLM call inside a hard rule is not the thing this file forbids** — stated here because the
next reader will see it and reasonably assume a bug. The model answers a question about a *place*,
never about the job; the **cache** makes it deterministic (two Berlin jobs cannot get different
answers, and a second backtest run was byte-identical and free); and the **policy stays in code**.
The precedent is `company_blacklist_reason()`. It fails **open**, opposite to the blacklist, because
a classifier call happens only when nothing the user wrote down matched.

**Separately: the parse failures were misdiagnosed from their own error message.** Five jobs hit
`eval_error` on Sep 03 and the message began `'```json\n{...`, so the obvious reading was that a
fenced code block broke the parser. **It never did** — `extract_json_object` scans to the first `{`
and calls `raw_decode`, which ignores everything around it; a test now pins that. The real causes:

- **Truncation.** `finish_reason: "length"` with `completion_tokens: 3000` — exactly
  `LLM_JSON_MAX_TOKENS` — appears **13 times** across Sep 01-03, and survived the glm-5.2 →
  glm-5.3-flash change. `chat_openrouter` inspected `finish_reason` only when `content` was null, so
  a truncated *non-empty* body went to the parser as if complete.
- **One genuinely malformed body**: Archer's `"reasoning":Exceptional domain match` — the opening
  quote of the value simply missing. No pre-processing fixes that, and none should try; repairing
  truncated JSON would fabricate a rating from a partial one.

The error printed `text[:300]`. **The head is where the fence is; the fault is always in the tail.**
Same shape as 2026-08-24's `api_error_status`: the field naming the cause was in the payload and
discarded.

**And the cap nobody chose cost double.** No call site ever passed `max_tokens` — all eight inherited
the 3000 default — against models allowing **131,072**, with the MCP tool declaring it optional and
applying no clamp, and the two agentic call sites already sending nothing. A truncated company-match
(a true/false question) paid $0.0068 for 3000 reasoning tokens, returned `null`, then paid again for
the Anthropic fallback. It is now not sent at all.

**An `eval_error` is terminal and silent.** The job was written to `processed_jobs/` back in Stage 1b,
so the next run returns `already_processed` and it is never rated again. A Netherlands posting the
model had scored **5** was lost permanently, behind one WARNING. `rate_with_openrouter` now retries
once on truncation.

### 2026-09-02 — the fallback that existed, was correct, and could not fire

A run found 0 jobs. The proximate cause was external and trivial: **the OpenRouter account ran
out of credit mid-run.** Stage 1a generated its six queries, Stage 1b's first request completed
one normal iteration ($0.0006, prompt=6,904), and the next call returned `402 Payment Required`.
The balance hit zero between those two calls.

**The interesting part is everything the codebase then did with that fact.** Three mechanisms
that exist precisely for this each failed, and all three are shapes already documented above.

**1. The Anthropic fallback was unreachable.** `run_non_interactive` wraps the OpenRouter
scraper in a `try/except` whose comment states the intent exactly — *"A provider outage must not
end the run: fall through to the Anthropic scraper"*. But `run_scraper` catches **every**
per-query exception first (*"One failed query must not abort the remaining ones"*) and
`continue`s. So `run_scraper` returned **normally**, `scraped = True` was set, and
`_run_anthropic_scraper` was never called. The outer handler could only ever catch failures
raised *outside* the per-query loop — `mcp_session`, `list_tools`. Two correct-sounding comments,
one directly above the other in the call chain, describing incompatible control flow; the inner
one wins silently. **A fallback guarded by an `except` that a lower frame already swallowed is
not a fallback, and nothing tests that it can be reached.**

**2. A systemic failure was retried per query, six times, with full pacing.** Queries 2-6 each
failed on their *first* iteration (`prompt=0 out=0`) after paying the complete randomised
human-emulation delay — 16.6s, 17.1s, 9.2s, 10.2s, 16.1s. ~70 seconds of anti-detection pacing
spent between calls that could not succeed. This is the 2026-08-24 `categorize_save_dir_pdfs()`
finding exactly (*"One API-level failure spawned 9 doomed CLI subprocesses"*), and it takes the
same fix: **`ProviderUnavailableError` (401/402/403/429) aborts the loop; everything else still
warns and continues.** Classification is on the **status**, never the reason phrase — the phrase
is upstream's text and can change without notice, the code is the contract.

**3. The run reported itself healthy.** The `if not candidates:` branch builds its own funnel
dict, calls `write_run_audit_log` directly, and **returns before the line where
`assess_run_health` runs** — so audit §1b printed *"No alerts: page structure sound, regions
distinct, yield acceptable"* while every query had errored, and Telegram said only
`0 candidates found` with no `⚠️ NEEDS ATTENTION` block and no mention of the 402. Even had it
been called, `assess_run_health` had **no concept of a failed query**: its saturation check is
guarded by `if distinct:`, so **zero listings skips every check it owns** — the one case where
something is most obviously wrong was the one case it could not alert on. Both exit paths now go
through `attach_run_health()` / `health_alert_block()` rather than through two copies that drift.

**The cause reached the run log and stopped there.** `_queries_searched[query] = 'error'` records
*that* a query failed, never *why*; §2 rendered six bald `**FAILED**` lines. The 402 — the single
string that made this diagnosable in about two minutes — appeared nowhere a person would look.
`_query_errors` now carries the reason into the audit log and the alert, deliberately beside the
`'error'` sentinel rather than replacing it, since the sentinel is what §2/§3 switch on.

**A note on the fallback's cost, because it is now reachable.** Anthropic Stage 1b is measured at
$0.4330 vs $0.0287 per search (9.3x, see 2026-08-21), and Stage 1b once reached 86.9% of total
run cost. Falling back is still right — a $0.43 run beats a zero-job run — but it *must* stay
visible in the run summary, which is the third fix above. An OpenRouter outage nobody is told
about would otherwise quietly bill Anthropic prices for days, which is the same class of silent
drift as the $0.50 → $10 climb that entry documents.

### 2026-09-01 — `npx @playwright/mcp@latest` stopped a run at an interactive prompt

`uv run job-search -n` halted during browser startup:

```
Need to install the following packages:
@playwright/mcp@0.0.80
Ok to proceed? (y)
```

Both launch sites used `npx @playwright/mcp@latest`, which npm re-resolves **on every run**. npx
had cached the tree as `"@playwright/mcp": "^0.0.79"` under the key `@playwright/mcp@latest`
(`~/.npm/_npx/9833c18b2d85bc59/package.json`), and **`^0.0.79` on a `0.0.x` version means exactly
`0.0.79`** — caret does not widen the patch range below `0.1.0`. Upstream published `0.0.80`, the
cached tree stopped satisfying `latest`, and npx blocked on stdin asking to install. That is the
startup path a scheduled run uses, where nothing can answer.

**The prompt is the symptom; the silent upgrade is the defect.** This package is the tool surface
Stage 1b drives a **real logged-in LinkedIn account** through, and the `--snapshot-mode` analysis
in the 2026-08-21 entry below is verified against **one specific bundle** — which 18 action tools
call `setIncludeSnapshot()`. Under `@latest` that could stop being true with no commit, no diff,
and no test failure. It is the same shape as the `mcp>=1.29` → 2.0.0 re-resolution in that same
entry: a dependency that moves underneath a documented measurement.

**The fix is a pin plus a prompt, not a pin alone.** `PLAYWRIGHT_MCP_VERSION` in `config.py` is a
concrete version and `--yes` is passed to npx — safe *because* the version is exact, since npx can
then install nothing but the pin, so a cold cache installs instead of blocking. Pinning alone,
though, just trades a loud stop for a silent freeze at a version nobody revisits. So
`check_playwright_mcp_version()` queries the registry at startup and
`prompt_playwright_mcp_upgrade()` acts on the answer.

**Three things about where and how it asks, each of which is the point:**

1. **It runs before `acquire_run_lock()`** — before the processed-job load, before PDF ingest, and
   long before `start_playwright_server()` (`agent.py`) or the interactive stdio server the SDK
   spawns. Choosing to upgrade therefore unwinds nothing: no lock held, no browser started. A
   check that ran after the lock would have to release it, and one that ran after npx would be too
   late to prevent the very prompt it exists to replace.
2. **It asks only in interactive mode AND on a TTY — both conditions.** The first version gated on
   `sys.stdin.isatty()` alone, reasoning that `-n` is run by hand as well as by cron so keying on
   the mode would deny a person the choice. **That was wrong, and it reproduced the original bug.**
   `-n` from a terminal has `isatty() == True`, so a hand-run `-n` stopped dead on a question —
   a mechanism built to stop the run blocking on a prompt, blocking the run on a prompt. `-n`
   means *do not ask me things*, whoever started it. Anywhere it does not ask it logs one WARNING,
   raises a `playwright_mcp_outdated` health alert, and continues on the pin. `EOFError` and
   `KeyboardInterrupt` at the question are caught and treated as "continue", so Ctrl-D or Ctrl-C
   cannot traceback out of startup either.
3. **The registry call fails open on everything** — offline, timeout, malformed body, a version
   string that will not parse. It is queried over plain HTTPS rather than via `npm view`, which
   shells out through the npm cache and fails for unrelated reasons (on this machine, currently,
   `cache folder contains root-owned files`).

`--no-version-check` skips the call outright for a scheduled run. Measured **~0.25s** against a healthy registry (the `/latest` endpoint returns 3KB), not the timeout value — and note the ceiling is not 3s either: `requests` applies a scalar `timeout` to the connect **and** the read, so the worst case is ~6s, with DNS resolution bounded by neither.

**The alert nearly went nowhere.** `run_non_interactive()` resets `tools_module._ui_alerts = []` on
entry, and the check appends *before* that runs — so the first version of this raised an alert that
was then discarded, the most-repeated bug in this file. Startup alerts now accumulate in
`_startup_ui_alerts` and the reset seeds from it.

**Version comparison is numeric, deliberately.** `'0.0.9' > '0.0.79'` is true as strings and false
as versions; a parametrised test pins that case.

### 2026-08-25 — the local model tag disappeared, and triage died without saying so

`test_generate_local_live` stopped passing. Run with the servers up and outside the sandbox it
**fails**, not skips:

```
RuntimeError: MCP tool 'generate' at http://127.0.0.1:8002/mcp returned an error:
  404 Client Error: Not Found for url: http://localhost:11434/api/generate
```

`LOCAL_MODEL` was `qwen3.6:latest`, and that tag stopped existing when the model was re-pulled as
`qwen3.6:27b-mlx` / `35b-mlx` around Aug 15. **Ollama's `latest` is not a moving alias — it is a
tag like any other, and a re-pull can simply not create it.**

**Nothing was broken, which is the problem.** Triage fails open by design, so every job just
skipped the free gate and went to the paid rating call. Ten days, invisible:

| run | `Triage:` scored | `Triage unavailable` |
|---|---|---|
| Aug 14 | 20 | 3 |
| Aug 18 | 0 | 53 |
| Aug 20 | 0 | 62 |
| Aug 21 | 0 | 63 |
| Aug 23 | 0 | 49 |
| Aug 24 | 0 | 28 |

**Three independent mechanisms hid it, and only the first is about this bug.**

1. **The cause could not reach the log, structurally.** `streamable_http_client` and
   `ClientSession` each open an anyio task group, so the `RuntimeError` naming the missing model
   surfaces to the caller nested two `ExceptionGroup`s deep — and `str()` of an `ExceptionGroup`
   is `unhandled errors in a TaskGroup (1 sub-exception)`. `triage_job_fit` logged `{ex}`, so 250+
   warnings carried **zero** bytes of the one string that identified the fault. This is the same
   shape as the 2026-08-24 `api_error_status` finding: the diagnostic field existed and was
   discarded. `unwrap_exception()` now flattens `.exceptions` / `__cause__` / `__context__`
   before logging. **Every MCP failure in this codebase arrives wrapped**, so anything catching
   around `call_mcp_tool` must unwrap before logging or it is logging nothing.
2. **A run-level misconfiguration was rediscovered per job.** 63 identical warnings where one
   would do — again the shape of 2026-08-24's `categorize_save_dir_pdfs()`. `preflight_local_model()`
   now runs once before Stage 2, and on a miss logs one ERROR **naming every installed model**,
   raises a `local_model_missing` alert into `assess_run_health` (so it reaches audit §1b and the
   Telegram `⚠️ NEEDS ATTENTION` block), and disables triage for the run. A *server* that is down
   still fails open silently — that is the supported case, and alerting on it would cry wolf.
3. **The test that existed to catch this hardcoded its own copy of the value.**
   `test_generate_local_live` passed `model='qwen3.6:latest'` as a literal rather than
   `config.LOCAL_MODEL`, so it tracked nothing and could not have detected the drift even if it
   had run. It now reads the constant, and `test_configured_local_model_is_a_concrete_tag` rejects
   a `:latest` pin outright.

**A note on the sandbox, because it is why nobody noticed.** `_require_llm_server` probes
`127.0.0.1:8002` with `connect_ex`, which the Bash sandbox always refuses — so in-sandbox this
test *skips* unconditionally, and a skip that can never become a pass is not a passing test
(the 2026-08-24 `conftest.py` gate, again). **Live LLM tests must be run with the sandbox
disabled or they are decorative.**

**The replacement is deliberately small.** `granite4.1:3b` — 65.9 tok/s, 3.9s cold start, 2.1GB,
against `qwen3.6:35b-mlx` at 52.5 tok/s / 21s / 21.9GB. Triage is one cheap 1-5 JSON score; it
does not need a 30B model, and Granite advertises structured JSON output as a first-class
capability. **The risk of sizing down is a false 1** — triage rejects at `TRIAGE_THRESHOLD = 1`
and a rejected job is saved as "triaged out" and never rated, so a mis-score costs a real job.
Measured against 17 real saved postings spanning ratings 1-5: **zero false 1s**, every 4/5 posting
scored 4 or 5, and the only 1s it assigned were on postings the strong rater also rated 1. Where
it disagreed it erred **high** (3→4, 2→3, 1→2), which is the direction `TRIAGE_INSTRUCTIONS` asks
for. Re-run that comparison before swapping this model again; `test_triage_job_fit_live` alone
only proves the model can reject an obvious non-fit, not that it spares a good one.

### 2026-08-25 — the rater calibrated on the jobs the user had already moved on from

Two defects found by watching a live run rather than by a failure. Neither broke anything loudly;
each silently degraded a documented mechanism, which is why both survived.

**1. The ideal-role profile was built from the OLDEST applied jobs.** `build_reference_summary()`
and `build_reference_block()` both slice `texts[:MAX_REFERENCE_JOBS]` (20), and
`_reference_job_texts` was filled by `for pdf in sorted(applied_to_dir.glob('*.pdf'))` — filenames
are `YYYY-MM-DD-`prefixed, so ascending sort is **oldest first**. With 60 in-horizon records the
rater calibrated against 2026-05-26 -> 2026-08-09 and never saw the 40 most recent.

The tell was a log line that reads like good news. Immediately after ingesting 8 new PDFs the run
logged `Reference summary: cache hit` — the new records sort last, never enter the first 20, so
`combined` was byte-identical and the md5 matched. **A newly applied job could not influence the
profile at all**; it only entered once older records aged out of the 90-day horizon. The mechanism
whose stated premise is "the strongest available signal of what to search for" was structurally
incapable of tracking a change in direction.

The two halves disagreed, which is what kept it invisible: Stage 1a reads `applied_jobs_summary()`
and *did* pick up the new titles (the query set shifted to `Lead AI Engineer` /
`Senior Applied AI Engineer` that same run), so query generation tracked the current direction
while the rater scoring those results lagged it by months.

Fixed in `load_applied_jobs()` by sorting on `applied_date` descending — deliberately not on
filename, because a legacy file with no date prefix sorts arbitrarily. Both call sites are
unchanged. A count-only test (`test_build_reference_block_caps_at_max_pdfs`) had been passing all
along while the code kept exactly the wrong 20; the new tests assert **which** records survive.

**2. The recovery pass turned the distinct-listing counter into a call counter.** The first pass
computes `seen` as a genuine distinct count; the recovery branch then did
`seen = sum(delta.values())` and never refreshed `calls`, emitting the impossible

```
query "Principal AI Engineer" inspected 100 distinct listing(s) of 50 checked
```

on 4 of 6 queries. `seen` also feeds `_queries_searched`, so a recovered query reported a call count
where every other query reports distinct listings — and it **inflates**, so a duplicate-saturated
recovery reads as excellent coverage. That is the 2026-08-18 "saturated run looks healthy" shape,
reintroduced in the one code path that only runs when a query is already in trouble. Counting
distinct rather than calls is load-bearing precisely here.

**A derived metric that names a cause it cannot observe is worse than no metric.** The same block
logged `{calls - seen} were surfaced by more than one region` — but `calls - seen` cannot
distinguish cross-region duplication from a recovery pass re-harvesting the same page. On this run
it produced a confident and entirely wrong diagnosis (that the location filter had collapsed on four
queries) while `region_overlap_report()` — a real Jaccard over per-region id sets — correctly
reported 0.0 and the filter was working. The line now states what it measured and defers to
`region_overlap` for cause; a test asserts the region claim cannot come back.

### 2026-08-24 — "error result: success", and the leak that never left `tools_generic.py`

A run failed to categorize all 9 uncategorized PDFs in the save directory, nine times over:

```
Warning: categorization failed for <file>, leaving uncategorized: Claude Code returned an error result: success
```

**The string is not ours.** It comes from the installed SDK
(`claude_agent_sdk/_internal/query.py:383`): when the CLI emits a result frame with
`is_error: true`, the SDK builds its text from `"; ".join(errors)` and — with `errors` empty —
falls back to `str(subtype)`. `subtype` was `"success"`, so the word is a *fallback label leaking
through*, not a status. The SDK's own type definition says what the combination means:

```python
# HTTP status code (e.g. 429, 500, 529) of the failing API call when
# ``is_error`` is True and ``subtype`` is "success"; None otherwise.
api_error_status: int | None = None
```

So `is_error=True` + `subtype=="success"` is **an HTTP-level API failure**, and the one field that
names the cause — `api_error_status` — was discarded by us on every path: `grep -rn
"is_error\|subtype" src/` returned **nothing**. Four `sdk_query` loops each tested
`isinstance(msg, ResultMessage) and msg.structured_output`, so an errored result simply looked
like a result with no output, and the real exception arrived later from the SDK stripped of the
status code. 401, 429 and 529 were indistinguishable — which is why the first hypothesis
("we weren't logged in") could be neither confirmed nor refuted from nine identical messages.

**Three failure modes, only the first of which is about the error itself.**

1. **Undiagnosable.** Nine identical failures carried zero actionable information. Fixed by
   `_raise_if_result_error()`, which inspects each `ResultMessage` *before* the SDK's trailing
   uninformative `ProcessError` and raises `AgentApiError` naming the status, subtype,
   `terminal_reason` and a hint (401/403 → run `claude /login`; 429/529 → back off).
2. **Invisible, and silently degrading.** Nothing about categorization reached
   `run_dir/logs/run-*.log` — it was `console.print()`-only. Uncategorized PDFs never get the
   `cat-saved_jd-` prefix, so `ingest_save_dir_applied_pdfs()` never moves them into the corpus,
   so the applied-job corpus stops growing and query generation quietly degrades. **This is the
   same shape as the 2026-08-18 saturated run: a degraded run that looks like a normal one.**
   Now logged, and `print_result_stats()` reports `is_error`/`subtype`/`api_error_status` on every
   result so an errored one is no longer indistinguishable from a good one.
3. **A systemic error retried per file.** One API-level failure spawned 9 doomed CLI subprocesses.
   `categorize_save_dir_pdfs()` now separates the two classes: `AgentApiError` logs at ERROR and
   **aborts the run** (the next PDF would fail identically, and a corpus that stopped growing must
   not pass as a normal run), while a per-file fault — a corrupt PDF — still warns and continues.

**The adjacent finding was the expensive one.** All four `ClaudeAgentOptions` sites in
`tools_generic.py` pass `cwd=str(PROJECT_DIR)` but left `setting_sources`, `strict_mcp_config` and
`skills` at their defaults — the exact bug the 2026-08-21 entry above documents as fixed. It was
fixed *in `agent.py`*, and that scoping is precisely why nobody looked here. **Measured with
`get_context_usage()` on the real categorization options: 22,945 tokens → 294, a 78x reduction**,
all of it `memoryFiles` (this file, loaded in full into a Haiku call that classifies a PDF into one
word). Nine PDFs paid it per run, as did every `company_matches_applied` and
`company_blacklist_reason` call.

**The blacklist path deliberately still swallows this error.** `company_blacklist_reason` catches
`AgentApiError` and rejects on the exact name match anyway — required behaviour, since a tool
outage must never quietly readmit a blacklisted company (`test_blacklist_confirmation_failure_still_rejects`).

**A gate that asks the wrong question skips instead of failing.** `conftest.py` gated
`live_agent_claude` on `ANTHROPIC_API_KEY`, which the Agent SDK does not need — it authenticates
through the logged-in CLI. On a developer machine every live test therefore *silently skipped*, and
the new context-usage measurement would have skipped with them. The gate now probes the CLI the SDK
would actually spawn. **A skip that never turns into a pass is not a passing test** — turning it on
exposed two live tests that had been failing unseen, and each turned out to be a real `src/` bug
rather than test rot:

1. **`_extract_company_from_text` was deleted from `src/` in commit 96bd708**, replaced by
   `_extract_applied_job_metadata`, but its test kept calling the old name for three months.
   Replaced by tests of the successor, including one covering the agency / `end_client` split.
2. **`save_job_posting` was uncallable, and the shorthand tool schema is why.** The
   `{"name": str}` form builds `"required": list(properties.keys())`
   (`claude_agent_sdk/__init__.py:422`), so **every** declared param is mandatory. Three tools
   therefore demanded fields their own descriptions call optional: `save_job_posting.job_id`,
   `check_and_record_job.{date_posted,url,content}` and `queue_candidate.{date_posted,query}` —
   including the `date_posted` this very file tells the model to "omit if not shown", which it
   could not do. **This is fabrication pressure of exactly the kind the Anti-fabrication
   requirement describes**: a model told a field is mandatory and unable to observe it must either
   refuse or invent. Observed live, verbatim: *"I don't want to invent an ID since it's the key
   this posting gets stored under."* It refused — this time. Optionality needs the full JSON Schema
   form, which the SDK passes through untouched; `test_optional_tool_params_are_not_declared_required`
   pins it.

**Interactive mode looks broken by the same probe, and is NOT yet fixed.** With `job_id` optional
the model called the tool correctly and the call was still refused —
`permission_mode="acceptEdits"` auto-approves file edits only, **not in-process MCP tools**, and
the denial surfaces in `ResultMessage.permission_denials`. `main()` configures interactive mode
(`agent.py:2780`) with exactly that mode, no `can_use_tool` callback and no `allowed_tools`, so
`save_job_posting`, `check_and_record_job` and `update_job_requirements` should all be denied
there; every other site in `agent.py` uses `bypassPermissions`. The live test now uses
`bypassPermissions` to match. **The production change was deliberately not made**: interactive mode
also holds the Playwright server against the real logged-in LinkedIn account, so broadening its
permissions is an Account-safety decision, not a test fix.

### 2026-08-13 — the extractor translates, so the JD's own language was invisible

`saved_jobs-2026Aug11/job_posting-4450073662-rating_4-valtech-…md` is a French-language posting —
*"Scientifique principal des données en IA"* — that was **rated 4/5 and notified**. Nothing in the
pipeline was wrong on its own terms; the language simply had no field to land in.

Three things that file shows, each of which shaped the fix:

1. **The extract was English.** Only `title` survived in French; the description came back as
   fluent English prose. The extractor translates as it condenses, so its own output carries no
   evidence of the source language — and any code-side heuristic run over the *extract* would
   conclude "english". That is why `posting_language` must be judged from the source page by the
   extractor, and why there is deliberately no `derive_posting_language()`.
2. **The location said nothing.** `location` was `Canada (Remote)` and `language_requirement` was
   `english` — both accurate. The Quebec signal was one sentence in the body: *"English proficiency
   required for regular communications with clients and colleagues outside Quebec."* A
   location→language lookup table in code would have missed it, which is why `local_language` was
   also the extractor's judgement and was explicitly told to read the whole body.
   **This half is superseded (2026-09-09).** The field is now `implied_local_language`, resolved
   from the location alone by the cached classifier, and the extractor no longer supplies it — a
   language a *place* speaks is world knowledge, and leaving it to a per-job judgement meant two
   jobs in one city could disagree. The cost is exactly this posting's second bullet: a working
   language named only in the body is no longer detected. **The outcome here is unchanged** —
   `posting_language` is a fact about the *page*, is still the extractor's judgement for the reason
   in point 1, and is what caps this posting.
3. **The rater saw the French title and still said nothing.** Its four warnings covered salary,
   stack, and the consulting-agency angle. As with the hybrid cap, a structural fact cannot be left
   to the rater's discretion.

The fix is a cap, not a rejection: `foreign_language_rating_cap` (default 3) keeps such a posting
saved and auditable while putting it below the ≥ 4 notification threshold. A non-English
`implied_local_language` is only a warning — many roles in non-English locations genuinely operate
in English, and the user asked to be told, not to have them dropped.

### 2026-08-21 — Stage 1b was 87% of the bill, and this file was part of the reason

Stage 1b reached **86.9% of total run cost** — $48.24 of $55.52 over ten runs — climbing $2.68
(Aug 17) → $7.44 (Aug 18) → **$10.06** (Aug 21) to inspect 300 listings of which only 147 were
distinct. Nothing was broken; nothing reported a problem. Cost was only ever visible as a stage
total, which is why a 68% → 99% drift went unremarked for weeks.

Cost is `Σ_turns(context_size)`, and **~90% of it was cache traffic** (47% read, 42% write, 10%
output; a least-squares fit over 47 requests recovers Haiku list price at the 1h TTL to 1.5%).
Turns were never the constraint — the 260 budget peaked at 163.

Three findings, in increasing order of embarrassment:

1. **The scraper's 40K-token floor was almost entirely not about scraping.** `scraper_options`
   left `setting_sources`, `strict_mcp_config` and `skills` at their defaults, and the SDK
   docstring is explicit: *"When `None`, all sources are loaded… Must include `project` to load
   CLAUDE.md files."* **Measured with `get_context_usage()` on a bare options object declaring no
   MCP servers at all: 45,523 tokens → 358 with the three set.** The 45,165-token difference is
   `Memory files` **31,384** (this file) and `MCP tools` **13,781** (nine unrelated global servers
   — telegram, rag-local, yt-dl, image-gen…), re-read on all 776 turns. The cost history tracks
   this file's own growth: 8.7KB in July at $0.50/run → 67KB now at $10/run. **This file was
   materially responsible for the bill it documents.** Fixed at every `ClaudeAgentOptions` site
   in `agent.py`; moving Stage 1b off the Agent SDK removes it there as well. **That "in
   `agent.py`" was load-bearing in the worst way** — it was literally true and read as complete,
   and the four sites in `tools_generic.py` stayed unfixed for three more months (see the
   2026-08-24 entry). A prose claim scoped to one file cannot police a second one; the AST guard
   `test_every_claude_agent_options_site_limits_context` now checks every site in the package, so
   the next one someone adds fails a test rather than quietly costing money.
2. **Every randomised safety pause was buying a full page snapshot.** Verified in the installed
   `@playwright/mcp` bundle (`PLAYWRIGHT_MCP_VERSION`, then 0.0.79): 18 action tools call
   `setIncludeSnapshot()`, resolving
   `config.snapshot?.mode ?? "full"` — and **`browser_wait_for` is one of them**. The pacing added
   *for* Account safety was the thing being billed. ~20 auto-snapshots × ~8K tokens × 12 searches ≈
   1.92M, against the measured 1.96M cache-write. The arithmetic closes. `--snapshot-mode none`
   fixes it and leaves explicit `browser_snapshot` untouched (it sets `_includeSnapshot =
   "explicit"`, which ignores the config), so the obstacle path is unaffected. **Not yet applied.**
3. **~370 of 776 turns re-emitted data the model already had** — the per-listing
   `check_and_record_job` loop after a harvest that already returned all 25 cards.

**The fix was the model, not the mechanics.** `deepseek-v4-flash` prices identical traffic 9.3x
cheaper ($0.0287 vs $0.4330 on one measured search) because it caches implicitly at ~88% with **no
cache-write fee**. Beware the obvious wrong choices: this project's own `OPENROUTER_MODEL`
(`z-ai/glm-5.2`) costs **more** than Haiku at $0.1932/M cache-read, as do `deepseek-v4-pro` and
`qwen3.8-max` — five of eleven candidates evaluated lose money. There is **no qwen3.8 flash
variant**; the 3.8 line is a premium tier.

**Three capability gaps, all harness-fixable, none about page comprehension.** The flash model
reasoned about the page correctly throughout, but: it could not infer playwright's `target`
convention (burned 14 iterations discovering the bare ref `e1202`, then regressed twice — fixed by
~10 lines of prompt); it **retyped the canonical JS from memory and corrupted it**; and it
**fabricated a page report** when `filename` diverted its evaluate result to disk. The last one
produced the Anti-fabrication requirement, and the fix for the middle one — dedicated tools that
take no page code — is strictly better than what it replaced.

**Two process lessons worth more than the savings.** First, `uv run` silently re-resolved
`mcp>=1.29` to 2.0.0, whose `streamable_http_client` yields two values instead of three, breaking
`mcp_session` — i.e. every OpenRouter and Ollama call — at runtime, mid-run. Now pinned `<2` with
two tests. Second, the harness's own test fixtures omitted playwright's `### Ran Playwright code`
echo, so a parser that scanned to the last `}` ran into the echoed script and returned None on
**every real harvest** while the tests stayed green. **Fixtures that are cleaner than reality are
worse than no fixtures.**

### 2026-08-18 — the filter chips were there all along, and the region axis was dead

A run queued **1 candidate from 300 listings inspected** — the end of a four-day slide (26 → 4 → 4
→ 2 → 1 new jobs) during which **not one notification was sent** and nothing reported a problem.
Stage 1b was mechanically healthy throughout: 12/12 searches ran, no block, no auth wall.

**The 2026-08-11 entry below was wrong, and this is the correction.** Its central claim — that the
filter chip row is gone and filters must be typed into the query text — does not hold. Verified
live against `~/.linkedin-agent-profile`:

| Claim (2026-08-11) | Reality (2026-08-18) |
|---|---|
| chip row is gone | **present**: Date posted, Experience level, Employment type, Company, + contextual chips |
| `location` cannot be set | **location chip works**; autocomplete resolves a region to a `geoId` |
| filters stripped from the URL | applying a chip **puts `geoId` / `f_TPR` in the URL**, and they survive direct navigation |
| no sort/date control | **Date posted** = Past month / Past week / Past 24 hours (`f_TPR=r604800`) |
| only page 1 reachable | results footer exposes **1 · 2 · 3 · Next** |

The trick is simply that you must search on the **bare query** and filter on the results page.
Typing the region into `keywords` is silently ignored — every "European Union" search returned
Greater Vancouver instead. Measured, same query:

| | searches | listings | unique | never-seen |
|---|---|---|---|---|
| region in query text (the broken design) | 12 | 300 | 101 | **1** |
| location chip, two regions | 2 | 50 | 50 | **30** |

Zero id overlap between the two chip searches; only 3 of those 30 new jobs had been seen by the
production run at all. The old design did 6× the searching for a thirtieth of the yield.

**Filters are applied by CLICKING, never by crafting the URL.** `geoId=91000000&f_TPR=r604800`
demonstrably works when navigated to directly — that is exactly why it is forbidden. No human
assembles filter parameters by hand, and this drives a real logged-in account. The URL parameters
are used for one thing only: **reading back** after the clicks to confirm they landed
(`check_filters_applied`). Known ids: Canada `101174742`, European Union `91000000`.

**Three failures were invisible, and all three are now instrumented.**

1. **The scraper diagnosed the dead region itself, on every single query** — the run log carries
   *"🔴 Issue detected: The EU region filter did not apply"* six times over — and it reached
   nothing but the log, because `log_agent_text` output is read by neither the audit log nor the
   funnel. Judgement now happens in code: the model reports the page structure via
   `report_search`, and `evaluate_ui_contract` / `check_filters_applied` decide. Same lesson as
   the hybrid and language caps: **a structural fact must never be left to a model's discretion.**
2. **The duplication hid itself.** `seen` counted `check_and_record_job` *calls*, so two regions
   returning the same 25 cards read as 50 listings — indistinguishable from real coverage, and it
   pushed the query *further* from the `SCRAPER_MIN_LISTINGS_PER_QUERY` retry. Distinct ids are
   now tracked separately (`listings_distinct`, `region_overlap`); Aug 18 would have scored 1.0.
3. **A saturated run looked exactly like a healthy quiet one.** `assess_run_health` now alerts on
   low yield, region overlap, contract breaches, blocks, low listing counts and early stops, in
   the run log, the audit log (§1b) and the Telegram summary.

**Verifying the contract probe against the live page was essential and caught two real bugs**, both
of which would have failed every query: the search box is a plain `<input>` with a **placeholder**
and no `aria-label`, and cards *and chips* are each rendered **twice** in the DOM (a raw count read
50 cards / 44 chips on a 25-job, 13-chip page). Do not write a selector for this page without
running it against the real thing.

**Still true from 2026-08-11, and safety-critical:** job ids come from the `componentkey` attribute
(`div[componentkey="job-card-component-ref-<jobId>"]` inside
`div[componentkey="SearchResultsMainContent"]`), and **nothing in the results list is ever
clicked** — the only real `<button>` in a card is Dismiss, which the accessibility tree disguises
as the card itself, and which already destroyed three real jobs. Clicking filter chips is fine;
they sit above the results. That boundary is asserted by unit tests.

Also still true: the account-level **salary filter self-injects** (`f_SA_id_227001:277001`
reappeared unbidden), so the empty `f_SAL=` stays on the search URL; and the **location is sticky**
across sessions, so it must be set explicitly every search rather than assumed.

### 2026-08-11 — LinkedIn migrated the account to AI-powered job search

> **SUPERSEDED on 2026-08-18 — see the entry above.** The account really was migrated to
> AI-powered job search, but the conclusions drawn about filters were wrong: the chip row
> exists, `geoId`/`f_TPR` work, and typing filters into the query text does nothing. The
> parts about `componentkey` job ids and never clicking the results list remain correct
> and safety-critical. Kept for the reasoning trail.

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
- **Preferences** — `run_dir/preferences.yaml` (gitignored), loaded by `preferences.py`: the `save_dir` where saved job PDFs land (default `~/Downloads`), search regions (each with a `linkedin_location` typed into the location chip and an optional `geo_id` used only to verify the click landed), sponsorship-required locations, languages, `foreign_language_rating_cap`, rejected degree levels, hybrid cap and acceptable locations, the geographic gate (`locations.exclude` and `locations.reject_regions`, the first holding **proper place names** matched case-sensitively and accent-exactly), target/excluded titles, relocation note, and the **company blacklist** (`companies.blacklist`: per entry a `name`, a `reason`, and an `added` date; entries expire after `COMPANY_BLACKLIST_EXPIRY_DAYS` (180)). This is the **only** home for facts about the person running the agent; tracked source must stay neutral. `preferences.example.yaml` in the project root documents the format with placeholder values
- **JOB_REQUIREMENTS.md** — agent-managed preference file; read-only in non-interactive mode
- **Search Query** — short LinkedIn search string derived from Resume, JOB_REQUIREMENTS.md, and Applied Job Records
- **Applied Job Record** — a job the User applied to: a date-prefixed PDF in `run_dir/applied_jobs/` plus its `index.yaml` metadata (applied date, company, job title, recruiting-agency flag, end client). Active for 3 months; older records are retained but unused
- **Job Posting** — a LinkedIn listing with company, title, description, URL, job_id, date_posted, `workplace_type` (`remote` / `hybrid` / `onsite`), `education_requirement` (`master` / `phd` / empty — hard requirements only), `posting_language` (the language the **source page** is written in), `implied_local_language` (the working language the job's location implies, from the classifier — never from the posting), `place_names` (every other name the location's places go by, English and local, from the classifier), `residency_scope` (`country_only` / `area_wide` / empty — whether the JD pins residence to the anchor country, offers a whole multi-country area, or says nothing; **not** the same as the anchor), `is_agency` (poster is a staffing firm / aggregator, not the employer) and `end_client` (who would actually hire, when the posting names them; empty when the poster echoes itself back), and a 1–5 rating
- **Location Recommendation** — an entry in `run_dir/location_recommendations.yaml`: a country with `first_seen`, `region`, `recommendation` (`exclude` / `hybrid_acceptable` / `keep_as_is`), `reason` and a user-owned `status` (`pending` / `accepted` / `rejected`); or an `entry_warnings` record naming a list entry that is dead or ambiguous. **Written by the agent, never read as policy** — it is a suggestion the user promotes by hand
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
- **Ingest Applied Jobs**: move categorized applied-job PDFs from the configured save directory (`save_dir` preference) into `run_dir/applied_jobs/`, stamping each filename with its applied date and preserving mtime; already-dated files are a no-op, so it is safe on every run
  - includes: Flag Recruiting Agency
- **Flag Recruiting Agency**: record whether an **Applied Job Record's** poster is a staffing firm or aggregator rather than the hiring employer, along with the end client if named; agency names are never added to the already-applied blocklist
- **Generate Search Queries**: derive at most `MAX_SEARCH_QUERIES` (6) Search Queries from Resume, JOB_REQUIREMENTS.md, and the Applied Job Records within the 3-month horizon (glm, Anthropic fallback), covering both the titles already applied to and adjacent titles. Individual-contributor titles only — people-management titles (Manager / Head of / Director / VP) are excluded
- **Write Run Audit Log**: at the end of every run, write `run_dir/audit_logs/audit-{date}-{time}.md` tracing applied-job counts, every Search Query and whether it was actually searched, listings and candidates per query **broken down by check status (new / already processed / already applied / too old)**, and the outcome of every individual Job Posting with its URL and summary. The per-query breakdown is what separates a dedup-saturated query from one that barely ran. §1 also reports the active blacklist count and names every **expired** blacklist entry with its date — an entry that quietly stopped rejecting is exactly the kind of change that otherwise goes unnoticed for months
- **Screen Candidate Title**: at queue time, deterministically ($0) skip a listing whose title is below target seniority (intern / junior / entry-level / new grad / apprentice / trainee / co-op) or matches a Preferences `titles.exclude` word, before it reaches the paid extract-and-rate path. Conservative by design — word-boundary matches only, `associate` and bare `graduate` never reject, and work arrangement is never judged from a search card. Every skip is logged and counted in the run funnel under `queue_skipped`; extends Scrape Job Postings
- **Audit Un-surfaced Jobs**: (`--audit-opus N`) sample N Job Postings from each of three un-surfaced pools — filtered at Stage 1, seen but never queued, and rated 2–3 — re-rate each with Opus, and report any the strong model scores ≥4 as a false negative. Diagnostic only; nothing is saved or notified
- **Scrape Job Postings**: execute Search Queries on LinkedIn and collect candidate Job Postings. Every Search Query runs as **one search per configured region** (see Preferences `search_regions`), and all of them run every time.

  **Filters are set by CLICKING the filter chips, never by putting them in the URL and never in the query text.** Each search navigates to `/jobs/search-results/?keywords=<QUERY>, remote&f_SAL=` — a bare keyword URL, which is what a bookmarked or shared LinkedIn search looks like — and then clicks: the **location chip** (clear, type the region, pick the autocomplete suggestion), **Date posted** → `SCRAPER_DATE_POSTED_LABEL` (Past week), and **Experience level** → `SCRAPER_EXPERIENCE_LABEL` (Senior). The empty `f_SAL=` stays, because the account-level salary filter self-injects.

  Two things this replaces, both wrong: putting the region in the **query text** (silently ignored — every EU search returned the home metro for days), and navigating **directly to `geoId=`/`f_TPR=`** (works, but no human assembles filter parameters by hand, so it is a cheap fingerprint for anti-automation). The `geo_id` preference exists **only** to read back afterwards and confirm the click landed. Sort order is not a coverage axis — the UI exposes no sort control. Region is.

  Searches are **expected to return a high proportion of `already_processed`** results — that is the cost of the few genuinely best-matching jobs they surface, and is not a failure signal. Only page 1 is scanned. The converse also holds: **two searches returning the identical list of jobs means the location filter did not apply**, and is a failure to report rather than a sign the query is exhausted — now measured in code as `region_overlap` rather than left to the model to volunteer.
  - includes: Verify Search UI Contract
  - includes: Harvest Job Listings

  A search returning zero results is retried once before being counted as empty. A query inspecting fewer than `SCRAPER_MIN_LISTINGS_PER_QUERY` (5) **distinct** listings is retried once by `run_scraper`: at **zero** (auth-wall / block / empty-shell signature, distinct from "seen but deduped") the retry drops the location filter; **above zero** — the signature of a results list that was never walked — the retry re-states the harvest procedure. Counting **distinct** listings rather than `check_and_record_job` calls is load-bearing: two regions returning the same 25 cards used to read as 50 and pushed a collapsed query *further* from this threshold. **A challenge or verification page is never retried into**: the recovery prompt tells the model to stop and report instead, because pushing through a block is the one failure that can cost the account.

- **Verify Search UI Contract**: on every search, run one read-only `browser_evaluate` (`SCRAPER_UI_CONTRACT_JS` in `agent.py`) reporting the page's structure — search box, results container, distinct job-card count, chip row and its labels, location pin, pagination, result-count text. On the OpenRouter path the model calls **`run_ui_contract(region)`** and code runs the JS, reads the result and judges it: **code reads, code judges, the model only triggers.** (The Anthropic fallback still hands the object to `report_search` — the older "model reports; code judges" form, which is exactly the courier design that produced a fabricated report; see Anti-fabrication.) `report_search` checks, in this order:

  1. **Block signature** (`UI_BLOCK_SIGNATURES`: CAPTCHA, "unusual activity", verification, forced re-login) — checked *first* because a challenge and a markup change demand opposite responses. The query stops and is **never retried into**.
  2. **Contract violations** (`evaluate_ui_contract`) — a required `UI_CONTRACT_ELEMENTS` entry missing, or zero cards while the page itself reports results, which is the precise shape of a silent failure that otherwise reads as "no new jobs today". The query stops, the user is alerted, **remaining queries still run**.
  3. **Filter confirmation** (`check_filters_applied`) — the `geo_id` and `f_TPR=r<seconds>` the clicks should have produced are present in the URL, and the chips have relabelled to their chosen values. This is the only use of those parameters.
  4. **Fingerprint drift** (`check_fingerprint_drift`) — the chip labels are compared against `run_dir/ui_fingerprint.yaml` and any change is reported *even when every required element still passes*, so a restructure is noticed the run it happens rather than months later.

  Selectors for this page **must be verified against the live page before being trusted**: writing them from the accessibility tree alone produced two failures that would have broken every query (the search box is a plain `<input>` with a placeholder and no `aria-label`; cards *and* chips are each rendered twice in the DOM). Invoked by Scrape Job Postings

- **Harvest Job Listings**: read every listing off the results page in **one read-only `browser_evaluate`** (`SCRAPER_HARVEST_JS` in `agent.py`), returning id, title, company, location and posting date per card. On the OpenRouter path the model calls **`harvest_listings()`** (no arguments): code runs the canonical JS, keeps the listings, and returns only a count — so the model can neither retype the JS wrongly (it corrupted the contract JS this way on 2026-08-21) nor relay job data. **`record_listings()`** then records from code's copy via `do_record_listings`, which enforces `SCRAPER_MAX_LISTINGS_PER_SEARCH` **in code** (previously a prompt suggestion) and returns counts only. Cards are `div[componentkey="job-card-component-ref-<jobId>"]` inside `div[componentkey="SearchResultsMainContent"]` — **the LinkedIn job id is the attribute suffix**, so no clicking is needed to identify a listing. Cards appear twice in the DOM (dedupe by id) and each label is rendered twice (a visually-hidden copy carrying "(Verified job)" plus the visible one), so the extractor collapses both. **Nothing in the results list is ever clicked** — see Account safety. If the harvest returns an error or zero jobs while the page visibly shows results, the markup has changed and the scraper must say so rather than fall back to clicking. Capped at `SCRAPER_MAX_LISTINGS_PER_SEARCH` (25); invoked by Scrape Job Postings

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
- **Summarize Reference Jobs**: distill the **most recently applied** `MAX_REFERENCE_JOBS` (20) Reference Job PDFs into a compact ideal-role profile via provider chain (OpenRouter → Local LLM → Anthropic), cached until those PDFs change. Newest-first ordering is load-bearing, not cosmetic: the cap is applied with a plain slice, so the sort order decides which 20 the rater ever sees
- **Evaluate Job Fit**: extract, filter, triage, and rate a candidate Job Posting, saving it as a Saved Job
  - includes: Extract Job Posting
  - includes: Flag Agency Posting
  - includes: Deduplicate by End Client
  - includes: Apply Hard Rules
  - includes: Reject Blacklisted Company
  - includes: Triage Job Posting
  - includes: Rate Job Fit
  - includes: Notify User of Match
- **Extract Job Posting**: navigate to Job Posting URL and capture a condensed, information-dense extract of the page; route and model configurable via `MODEL_NAME_EXTRACTOR` (OpenRouter function-calling loop by default; pointing it at an `ANTHROPIC_*` const selects the agentic SDK session)
  - includes: Detect Posting Language
- **Detect Posting Language**: capture `posting_language` — the language the **source page** is written in. It must be judged from the original page, **never** from the extractor's own output: the extractor translates as it condenses, so a French posting comes back as English prose with only the title left in French (see the Valtech entry in Known diagnoses). There is deliberately **no** code-side heuristic over the posting's *prose* — guessing a language from condensed text is exactly the mistake this field exists to avoid, so `posting_language` has no `derive_` fallback and is the one language fact the extractor still supplies. It drives Cap Foreign-Language Rating and a deterministic warning; it never rejects. Invoked by Extract Job Posting
- **Detect Implied Local Language**: resolve `implied_local_language` — the working language the job's **location** implies — from Classify Job Location alone, via `derive_implied_local_language()`. **The extractor is not asked and cannot override it.** A place implies a language (Italy implies Italian); that is reusable world knowledge about a location, so the cache is what makes it deterministic — two Berlin jobs cannot be answered differently, within a run or across runs. This deliberately reverses the earlier design in which the extractor supplied the field and its *foreign* finding beat the classifier's. That rule bought one thing — a language named only in the posting body, like Valtech's "colleagues outside Quebec" on a `Canada (Remote)` posting — at the price of a fact about a place being re-decided per job; such a body-only signal is no longer detected, and the Valtech posting's outcome is unchanged because `posting_language` still caps it. **It gates nothing and caps nothing** — it warns, and the geographic rule deliberately does not read it. Invoked by Evaluate Job Fit
- **Apply Hard Rules**: force rating to 1 if the Job Posting is closed ("No longer accepting applications"), > 30 days old, located where the user needs sponsorship but none is offered, explicitly requires a language the user does not have, hard-requires a degree the user does not hold, is **anchored in an excluded region**, or **states a required relocation to one**. Every rule but the blacklist and the geographic pair is deterministic and $0; all but closed/stale read their thresholds from Preferences and are inactive when unconfigured. Every rejection is logged with its reason
  - includes: Reject Excluded Location
  - includes: Detect Advanced Degree Requirement
  - includes: Reject Blacklisted Company
- **Reject Blacklisted Company**: force rating to 1 when the Job Posting's poster **or** its `end_client` is on the user's Preferences blacklist. Runs **first** among the hard rules — it is the most decisive and costs nothing when the name does not match. The trigger is an **exact, case-sensitive** name match (`company_blacklist_reason()`), deliberately *not* the normalized fuzzy `company_matches_applied()` path: a standing user-authored blacklist must never over-block. A name hit then costs **one** cheap LLM confirmation call, because an exact name string can collide across organizations ("Cohere" the AI lab is not Cohere Health); a confirmed collision does not reject. **Any failure of that call rejects anyway** — the name matched, and a tool-server outage must not quietly readmit a blacklisted company. The end client is checked for the same reason Deduplicate by End Client exists: a recruiter reposting the role would otherwise get around it. Preference-driven and inactive when the list is empty. Entries **expire 180 days after their `added` date** — an expired entry stops rejecting and is reported at WARNING and in the audit log rather than silently outliving its reason; an entry with a missing or unparseable date stays **active** and is warned about, because a typo must not switch an unappealable rule off. Counted in the funnel as `hard_ruled_blacklisted`
- **Detect Advanced Degree Requirement**: resolve whether the Job Posting hard-requires a Master's or PhD, via the extract's structured `education_requirement` field or, when the extractor leaves it empty, `derive_education_requirement()` — a sentence-level scan requiring a degree token **and** a requirement word **and** no softener (`preferred`, `or equivalent`, a Bachelor's alternative) and no negation (`No PhD required`). Only hard requirements resolve to `master`/`phd`; preferred or experience-substitutable degrees resolve to empty and never reject. Invoked by Apply Hard Rules
- **Classify Job Location**: resolve a location string into geographic facts — the countries named, one region per country (`southern_europe` / `northern_europe` / `western_europe` / `eastern_europe` / `north_america` / `other` / `unknown`), a `broad_area` flag, the local working language, and **`place_names`** — every other name those places go by, English and local, properly cased and accented — via one cheap LLM call **cached in `run_dir/location_cache.yaml`** by normalized location string. Only reached for a location neither Preferences list names, so in steady state it is a cache hit or never called. **This does not violate "a structural fact must never be left to a model's discretion"**, and the distinction matters enough to state: the model answers a question about a *place*, never about the job — reusable world knowledge, checkable by a human, identical for every job in that location — while the **cache** is what makes it deterministic (two Berlin jobs cannot get different answers, within a run or across runs) and the **policy stays in code** (region → reject is a deterministic read of a preference). Same shape as `company_blacklist_reason()`, which matches a user-authored list deterministically and spends one cheap confirmation call only on the ambiguous case. **`place_names` deliberately does not ask "does this match the user's list"** — that question depends on the list, so its answer would go stale on every list edit, whereas a question about the place alone is cached by the place alone and stays true forever. A cached entry predating the field is treated as a miss and reclassified: serving it would silently disable alias matching for that location, which is the same shape as the dead `malaga` entry the field exists to fix. **`countries` come back as proper names too**, like `place_names` and unlike `regions` and `implied_local_language` — those are enum vocabularies with no correct written form, a country is a place. So the constant naming EU membership (`location.EU_MEMBER_STATES`) is written `Austria`, `Czechia`, and `is_eu_member()` folds at the comparison against a **derived** index rather than storing a second, folded copy of the list: a constant holds the real name of the thing it names, code folds where it must, and a name folded into the data can never be unfolded. Pinned by `test_eu_constants_are_written_as_proper_names`. **Fails open** — see Resilience.

  **Measured limitation, and the reason country-level overrides belong on the exempt list.** The cache guarantees a given location *string* is always answered the same way; it does **not** guarantee two strings naming the same country agree. Measured 2026-09-04: `Dublin, Ireland` → `western_europe` but bare `Ireland (Remote)` → `northern_europe`, and the same split for `London, United Kingdom` vs `United Kingdom`. Today that changes no outcome, because both regions are rejected — but it is a live trap for anyone who tries to *keep* a country by removing its region from `reject_regions`, since the other phrasing would still reject. **To keep or drop one specific country, put it in `hybrid.acceptable_locations` (tier 1) or `locations.exclude` (tier 2)**, both plain string matches that no classifier variance can reach. Region policy is for the coarse north/south decision only.

  Verified against real postings by `scripts/backtest_location_gate.py`; re-run it whenever the classifier model or the region policy changes. The first live run justified the classifier over the hand-written country table it replaced: it split `Aix-en-Provence → southern_europe` (keep) from `Hauts-de-France → western_europe` (reject) with no city list at all
- **Reject Excluded Location**: force rating to 1 when the Job Posting's `location` is anchored in an excluded region, or its `relocation` field states a required move to one. **Purely geographic — it reads no language field, and must never be changed to.** An earlier draft rejected on "non-English AND not on the acceptable list", which produces the right answers for Spain and Germany for the wrong reason and the *wrong* answer for a French-language remote role in Canada (`hybrid.acceptable_locations` lists Vancouver and British Columbia, not Canada). What language a place speaks is `implied_local_language`, it warns only, and folding it back in here reintroduces that bug. Resolution is three-tier, cheapest first: `hybrid.acceptable_locations` **exempts** outright (so `france` can be excluded while Nice and Toulouse pass), `locations.exclude` rejects outright, and anything neither names goes to Classify Job Location with `locations.reject_regions` deciding. A job is rejected only when **every** country named is in a rejected region, so "Spain or the Netherlands" passes, and so does anything naming no country. **`broad_area` is the third way to pass, and it is not redundant:** a posting offering a whole multi-country area is not limited to the countries it happens to mention, so `European Union (Remote, UK and EU)` must pass even though the UK is its only named country — a real 5/5 posting of exactly that shape was flipped by the first live backtest. Its counterpart is what keeps the flag honest: a single anchored country phrased expansively is **not** a broad area, so `Berlin, Germany (Remote across Europe)` is not `broad_area`. (That posting is nonetheless **kept** now — not by `broad_area` but by `residency_scope: area_wide`, since "remote across Europe" is the JD saying where you may live. The 2026-09-04 policy of rejecting it on the anchor is deliberately superseded for remote roles; see below.) Both directions are pinned by tests, because a `broad_area` that crept toward true would silently disable the gate. **A bare location is not a residency requirement** — "Germany (Remote)" means the role is anchored in Germany while the holder could live anywhere in the EU — which is why `location` and `relocation` are judged separately against the same lists. **For a `remote` posting the gate now ACTS on that premise rather than only asserting it** (this file had asserted it three times without the code doing anything about it). A fourth way to pass sits beside `broad_area`, in tier 3 only, keyed on the extract's `residency_scope`: `country_only` (the JD pins residence — "Remote within country", "must be based in X") is judged on the anchor exactly as before; `area_wide` ("anywhere in the EU", "Work from Anywhere") passes, like `broad_area`; and **unspecified passes iff every country the classifier names is an EU member state** (`location.EU_MEMBER_STATES`), because an EU anchor implies work rights the user has and a UK, Serbian, Norwegian or Swiss one does not. So a silent `Poland (Remote)` is kept and a silent `United Kingdom (Remote)` still rejects. **Three things are deliberately NOT spared**: `hybrid`/`onsite` (an office pins you to the office — the Munich/Berlin case this gate was built for), a stated `relocation` (the JD asserting a requirement outright), and `locations.exclude`, because the spare sits **below** the deny list. That last one is load-bearing: after this change `exclude` is the only way to drop one specific EU country for remote roles, so it must stay absolute — an early draft put the spare above it and silently made the list inert, caught by `test_excluded_locations_beat_the_eu_remote_spare`. **Every list match is whole-word, case-sensitive and accent-exact, against place names written the way places are written.** Lowercasing the lists is what made `malaga` an entry that could never match `Málaga` while looking exactly like one that had merely not come up yet. Case-sensitivity and the boundary class had to change **together**: `(?<![a-z0-9])` blocks only lowercase ASCII neighbours, so matching proper names with it lets `ROMA` reach into `ROMANIA` and `Roma` into `Romaña` — the original bug in caps and via an accent. The class is `(?<![^\W_])` (unicode letters and digits) and there is deliberately no `re.IGNORECASE`, because `Nice` the city is not `nice` the adjective. **A fourth tier catches what exact matching cannot**: once the classifier returns, both lists are re-checked against its `place_names`, which is how a posting saying `Torino` matches a list saying `Turin`, and `München` matches `Munich`. Exonyms are unreachable by any normalization, and hand-maintaining spelling pairs is what introduced `roma` in the first place. Resolved once per candidate into `extract['place_names']` and written back, so the three sync consumers (`hybrid_location_is_acceptable`, `apply_rating_caps`, `build_deterministic_warnings`) agree with the async gate instead of a hybrid role in `Sevilla` being gated correctly and still capped. **`sponsorship_required_in` is the one list still on lowercase substring matching** — a separate rule whose match direction is inverted, left alone rather than widening this change. `hybrid.acceptable_locations` contained `roma` (Rome) and `'roma' in 'romania (remote within country)'` is `True`, so tier 1 exempted **every** Romanian posting before the deny list, the classifier and `reject_regions` were ever consulted. The boundary must be a lookaround on the token's own edge characters, **never** `\b…\b`: `re.escape('u.s.')` is `u\.s\.` and `\bu\.s\.\b` matches nothing at all, so the obvious fix silently deletes a real entry. Preference-driven and completely inert (not one classifier call) when both lists are empty. Counted as `hard_ruled_location` / `hard_ruled_relocation`; invoked by Apply Hard Rules
- **Detect Residency Scope**: capture `residency_scope` — whether the JD explicitly pins residence to the country the role is anchored in (`country_only`), explicitly offers a whole multi-country area (`area_wide`), or says nothing (empty) — on the **existing Stage 2a extract call**, with `derive_residency_scope()` as a deterministic fallback. **The precedence is asymmetric and load-bearing**: a `country_only` finding from *either* source wins, and is tested first — the union direction, because only the *permissive* value can silently switch a gate off. A wrong `area_wide` **silently disables the geographic gate** for that job — the same failure `broad_area` is pinned against in both directions — while a wrong `country_only` surfaces as a rejection with a stated reason in the audit log and the funnel. Between the two *permissive* values the extractor wins (the `is_agency` shape), since it read the body and the regex is a backstop. The fallback patterns are anchored on a **concrete non-area object**, which is the whole trick: "must be based in Germany" pins you and "must be based in Europe" does not, and they are the same words up to the last one. Verified against all 95 postings saved 2026-09-04..08. Logged **raw and derived** on the extract-signal line, because a silent correction is a mis-extraction you can no longer count. Extends Extract Job Posting
- **Review Country Lists**: once per non-interactive run, compare the countries actually seen this run (accumulated in `location.py` on **both** the cache-hit and LLM paths — in steady state almost every location is a hit, so an LLM-only accumulator would report a country once and never again) against `locations.exclude` and `hybrid.acceptable_locations`, and record findings in `run_dir/location_recommendations.yaml`. **Recommend-only: it never edits a preference**, in both directions (add to `exclude`, add to `acceptable_locations`). Two halves, split by who can be trusted with which question. It runs **three** deterministic checks. **Dead entries**: an entry that matched nothing this run but whose case- and accent-folded form did — exactly how `malaga` is caught, and a distinction a plain "never matched" count cannot make, since an entry that has simply not come up yet looks identical. That folded comparison is **whole-token, never a substring**: folded `roma` sits inside folded `romania`, so a substring form would advise rewriting the entry as `Romania` — advice to reintroduce the very bug. Folding may ignore case and accents; it may never ignore boundaries. **Region disagreements** are the third: a country belongs to exactly one region — that is what makes it world knowledge rather than a judgement — so a second answer in the cache means one is wrong AND the cache has frozen it (measured: 5 of 40 countries, Ireland 17:1, United Kingdom 10:1, Austria 6:1, plus Lithuania and Serbia genuinely split 1:1). `countries_seen_this_run()` therefore reports a country's **majority** region rather than the first one seen — taking the first made the region a coin flip for exactly those countries, and it is how Austria came to be reported as `eastern_europe` off one Vienna entry. **Substring collisions are the second check, also code, also free, every run** — `'roma' ⊂ 'romania'` is arithmetic, "a structural fact must never be left to a model's discretion" applies to the reviewer as much as to the rater, and it means the check that would have caught the entry that broke the gate still works with the tool server down. Only "should this country be excluded?" is an LLM call, and only when a country is undecided, so steady state is **zero calls**; a country already recorded is never re-queried and an entry whose `status` the user edited is never rewritten. Editing the lists re-opens the still-`pending` recommendations, since they were made against a policy that no longer holds. **A corrupt recommendations file is skipped, not rebuilt** — deliberately the opposite of `location_cache.yaml`, because a cache is recomputable and a record of the user's decisions is not. New findings raise a `location_recommendations` health alert (audit §1c + the Telegram `⚠️ NEEDS ATTENTION` block); an unchanged pending set stays silent rather than crying wolf every run. Fails open on everything
- **Flag Workplace Type**: capture the work arrangement as a structured `workplace_type` field (`remote` / `hybrid` / `onsite`) rather than as free text inside the location; when the extractor omits it, `derive_workplace_type()` infers it from the location and description. Any required office days are `hybrid`, even when the board badges the listing "Remote" — LinkedIn's `f_WT=2` filter is not reliable. Extends Extract Job Posting
- **Flag Required Relocation**: annotate a Job Posting requiring relocation/residence in a specific location with "relocation required: <location>" in the Saved Job and Notification. Annotation is the default and remains so for an acceptable location — but a stated relocation to an **excluded region** now hard-rejects (see Reject Excluded Location), because that is the JD asserting a categorical requirement rather than code inferring one. A non-specific value (`European Union`, `EMEA`) names no country and never rejects; extends Evaluate Job Fit
- **Triage Job Posting**: score fit 1–5 with the Local LLM; clear low fits (score ≤ 1) are saved with the triage score and skip Rate Job Fit; fails open if the Local LLM is unavailable
- **Preflight Local Model**: once per run, before Stage 2, verify `OLLAMA_MODEL_NAME_TRIAGE` is actually installed on the local Ollama server. When it is not, report it **once** at ERROR naming every model that *is* installed, raise a `local_model_missing` health alert (audit §1b + the Telegram `⚠️ NEEDS ATTENTION` block), and disable Triage Job Posting for the run. The run still completes and jobs still reach Rate Job Fit — this is a louder fail-open, not a new abort. Silent when the *server* is unreachable, which is the already-supported outage case. Exists because a machine-local Ollama tag can vanish on a re-pull, and when `qwen3.6:latest` did, triage was dead for ten days behind 250+ warnings that named nothing
- **Check Browser Toolchain Version**: once at startup — **before the run lock, the processed-job load, and any `npx` spawn** — compare `PLAYWRIGHT_MCP_VERSION` against the npm registry over plain HTTPS. It asks only in **interactive mode on a TTY**: it prints the newer version with the three upgrade steps and offers `[u]` exit-so-you-can-upgrade or `[c]` continue on the pin; exiting unwinds nothing because no lock is held and no browser has started. In **`-n`, or with no TTY, it never reads stdin** — it logs one WARNING, raises a `playwright_mcp_outdated` health alert, and continues on the pin. Both halves of that gate are load-bearing: gating on the TTY alone let a hand-run `-n` block on the question, since `isatty()` is true in a terminal regardless of mode. **Fails open on every error** (offline, timeout, unparseable version, `EOFError`/`KeyboardInterrupt` at the prompt): a registry outage or a Ctrl-D costs a log line, never a run. **Running the `npx` step alone does not complete an upgrade** — the pin in `config.py` selects the version, so until the constant is bumped the old version keeps being used and the notice keeps appearing. `--no-version-check` skips the check. Upgrading is never automatic: it is an edit to the constant, so the bundle the scraper drives a real account through cannot change without a commit
- **Extract Job Posting (fallback)**: when the agentic extractor fails to submit, deterministically fetch the page snapshot over the shared Playwright session and condense it with one non-agentic cheap-model call; extends Extract Job Posting
- **Rate Job Fit**: one non-agentic structured-output call scoring fit 1–5 against JOB_REQUIREMENTS.md and the ideal-role profile, also returning **pros** and **warnings** bullet lists; route and model configurable via `MODEL_NAME_RATING` (OpenRouter `OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE` by default; `ANTHROPIC_MODEL_NAME_MEDIUM` selects the SDK, `OLLAMA_MODEL_NAME_TRIAGE` the local server). `apply_rating_caps()` then applies the deterministic hybrid ceiling below
- **Cap Hybrid Rating**: after rating, deterministically cap a `hybrid`/`onsite` Job Posting at the Preferences `hybrid.rating_cap` unless its location matches `hybrid.acceptable_locations`. A cap is not a rejection — the job is still saved and still appears in the audit log, it just falls below the ≥ 4 notification threshold; extends Rate Job Fit
- **Cap Foreign-Language Rating**: after rating, deterministically cap a Job Posting whose `posting_language` is outside Preferences `languages` at `foreign_language_rating_cap` (default 3). A ceiling, not a rejection, for the same reason as the hybrid cap. A non-English `implied_local_language` never caps — it only warns, because plenty of roles in non-English locations operate in English. Preference-driven: inert while `languages` is empty; extends Rate Job Fit
- **Notify User of Match**: send Telegram notification when a Saved Job has rating ≥ 4. The message carries the rating, company, title, a `📍` location line including workplace type, a bulleted **✅ Good** list (from the rater's `pros`) and a bulleted **⚠️ Warnings** list (deterministic warnings first, then the rater's, de-duplicated), and the URL. Sent as plain text — no `parse_mode` — so it uses bullets and emoji, never Markdown. The Saved Job reuses the same bullet block so file and message agree
- **Build Deterministic Warnings**: derive warnings in code independent of the rater — hybrid/on-site (with "not an acceptable hybrid location" where it applies), **posting written in a non-English language**, **non-English local working language** (naming the location), required relocation, contract-vs-full-time, missing salary, and **agency-posted** (naming the hiring company when known). The two language warnings are independent: a French JD gets the first (plus the cap), an English JD in Montreal or Madrid gets only the second. The rater cannot be trusted to self-report any of these: it once saw "Hybrid - 2-3 days onsite" and rated the job 4/5 without warning, it rated the French-language Valtech posting 4/5 and notified it with no language warning at all, and on the 2026-08-11 run it volunteered the agency fact for CyberCoders and Jobgether but not for Hire Feed or Genius Innovation Lab — 2 of the 4 agency postings that triggered a notification said nothing. Salary-below-target is deliberately left to the LLM, since parsing multi-currency day rates into a CAD annual figure is too brittle for a deterministic rule
- **Flag Agency Posting**: capture whether a scraped Job Posting's poster is a staffing firm, recruiting agency, or job aggregator rather than the hiring employer, plus the `end_client` when the posting names one. Captured as structured `is_agency` / `end_client` fields on the **existing Stage 2a extract call** (no extra LLM call), with `derive_agency_posting()` as a deterministic fallback used **only** when the extractor leaves `is_agency` unset — the extractor's `False` is a judgement and wins. The fallback is intentionally weak (phrase-level "our client is…", agency markers in the poster's name): measured against the 2026-08-11 corpus it caught 1 of 10 agency postings on its own, while the extractor field classified 15 of 16 correctly, so the LLM field is the real mechanism and the regex is a backstop. `derive_end_client()` discards an end client equal to the poster — extractors routinely echo the company back (measured: `end_client='lululemon'` on a lululemon posting). **Agency status warns only**: it never rejects and never changes the rating; extends Evaluate Job Fit
- **Deduplicate by End Client**: when a scraped Job Posting names an `end_client`, match **that** against the already-applied blocklist (`company_matches_applied`), because Stage 1b matched the agency's name and so misses a recruiter reposting a role at a company already applied to directly. Counted in the funnel as `end_client_already_applied`. Deliberately **not** extended to the agency's own name (see the Recruiters note above) and deliberately **not** used to dedupe the same role across different posters

### Non-functional Requirements

- **Anti-fabrication (the model chooses ACTIONS; code moves DATA)** — a tool argument must be a **decision**, never a payload code could have read itself. An LLM has no mechanism separating "I am copying this" from "I am producing this": both are token generation conditioned on context. So when the data is not in context — a call errored, a result was truncated, diverted to a file, or scrolled out — it does not fail, it emits the most plausible continuation, which is a well-formed, schema-valid object. **There is no failure mode where a courier notices it lacks the data**; absence produces fiction, and the fiction passes shape validation because shape was never the problem. Measured 2026-08-21: the model passed `filename` to `browser_evaluate`, the result went to disk, and it then invented an entire page report for `report_search`. Code caught it (`UI CONTRACT FAILED`) only because judgement already lived in code. **Test any signature by asking: if the model had never seen the data, could it still fill this argument plausibly?** If yes it is a *courier argument* — delete it and let code fetch the data. Diagnostic: **"verbatim" / "exactly as returned" / "do not summarise" in a tool description is a courier argument confessing itself**, and the instruction is unenforceable. This cannot be fixed by prompting — "never invent a report" competes with the drive to produce a plausible completion and loses precisely when data is missing; removing the argument removes the *opportunity*, which is not probabilistic. Stage 1b's toolset (`run_ui_contract`, `harvest_listings`, `record_listings`, `report_problem`) therefore accepts only `region` and `what_happened`; the harvest is held in `ScrapeSession` and handed to `do_record_listings` directly, `filename` is stripped from `browser_evaluate`/`browser_snapshot`, and **"I don't know" is a first-class, rewarded outcome** (`report_problem`) because the model fabricated partly for lack of any way to say so. Unit tests pin all of it, including one that calls `record_listings` *with* fabricated job data and asserts the fabrication is ignored. **A required parameter the model cannot observe is fabrication pressure, not validation.** The SDK's `{"name": str}` shorthand marks every param required (`claude_agent_sdk/__init__.py:422`), which silently made `date_posted`, `url`, `content`, `query` and `job_id` mandatory on tools whose descriptions call them optional. A model that cannot see a posting's date must then either refuse the call or invent one — and refusing is the behaviour that is *not* guaranteed. Declare optional params with the full JSON Schema form and a correct `required` list; `test_optional_tool_params_are_not_declared_required` pins the three tools involved. **Stage 2a's `submit_job_extract` is generative and cannot lose its courier role** — it needs provenance validation instead (`job_id` present in the navigated URL, title/company as normalized substrings of a re-fetched snapshot, enums restricted to the allowed set, falling back to `extract_job_page_direct()` on mismatch); designed, **not yet implemented**
- **Scraper tool surface** — Stage 1b restricts the Playwright toolset via `disallowed_tools` (`SCRAPER_DISALLOWED_BROWSER_TOOLS`), which is the only option that removes a tool from the model's context. `allowed_tools` does **not** do this: it only auto-grants permission, which `permission_mode='bypassPermissions'` already grants, making it inert — measured, the model saw all 24 Playwright tools with `allowed_tools` set to 4. `browser_evaluate` and `browser_click` are load-bearing for discovery (inner-container scroll lazy-loads 7 → 10 listings; Next yields a fresh page) and a unit test asserts they are never disallowed
- **Cost efficiency** — Haiku for scraping, page extraction, and applied-job metadata; the agentic browser work never runs on Sonnet. Query generation is the one deliberate exception: a single Sonnet call per run (~$0.33), because top-of-funnel query relevance is the pipeline bottleneck and everything downstream is gated by it. Applied-job metadata is cached in `index.yaml` by mtime, so the extraction cost is paid once per PDF. Hard rules and local triage reject clear non-fits for $0 before any paid rating call. **A triage gate that has silently stopped running still looks free — it just stops saving anything**, which is what Preflight Local Model exists to make visible. The triage model is sized to the job (one 1-5 JSON score): `OLLAMA_MODEL_NAME_TRIAGE`, not the 30B tier, measured to track the strong rater with zero false 1s across ratings 1-5. The company blacklist is the one hard rule that can cost anything, and only on an exact name hit — a non-matching name short-circuits before any call, so the gate is free on essentially every posting; do **not** "optimise" this by dropping the confirmation call, which is what stops a same-named organization being rejected by mistake. Since LinkedIn no longer enforces `f_E` server-side, a $0 title check at queue time (`title_rejection_reason()`) keeps below-seniority and people-management listings out of the paid Stage 2 path, where each would cost an extract plus a rating call; it is deliberately conservative (word-boundary matches only, and neither `associate` nor bare `graduate` rejects) because an auto-skip is unappealable, and it never filters on work arrangement, which is unreliable on a search card. The rating call is a single non-agentic structured-output call (route and model pinned explicitly by `MODEL_NAME_RATING`, never the CLI default model). **No output-token ceiling is sent anywhere**, deliberately: `LLM_JSON_MAX_TOKENS = 3000` was inherited as a default argument by all eight call sites rather than chosen by any, against models allowing 131,072 completion tokens with no server-side clamp, and every one of the 13 truncations across 2026-09-01..03 was `completion_tokens: 3000` exactly and survived a model change. Removing it *lowers* cost: a truncated company-match paid for 3000 reasoning tokens, returned `null`, and then paid again for the Anthropic fallback — finishing the answer is cheaper than paying for both halves of a failure. If a ceiling is ever genuinely needed, pass `max_tokens` at the call site that needs it. Review Country Lists adds at most one cheap call per run and normally **zero** — a country is asked about once, ever — and its substring-collision half is deterministic code that costs nothing and keeps working when the tool server does not. The geographic gate is free on essentially every posting — two user-authored list checks, then a **cached** classification per distinct location string — and it runs before the rating call, so a rejected job costs no rating. Reference jobs are distilled once into a ~2.5K-char cached profile instead of a ~60K-char block; the evaluator prompt is built once per run and reused across all jobs to maximise prompt-cache hits; the extractor is capped at 8 turns and restricted to the browser tools it needs. **Account safety still outranks cost in Stage 1b**, and its prompt stays deliberately verbose — but cost there is no longer unmanaged: it reached **86.9% of total run cost** ($48.24 of $55.52 over ten runs, peaking at $10.06), so it moved to `MODEL_NAME_SCRAPER` (`OPENROUTER_MODEL_NAME_SCRAPER`), measured 9.3x cheaper on identical traffic. Pacing was **not** traded away: the randomised delays are unchanged. Do not optimise Stage 1b for speed or turn count; optimise **what enters the model's context**, and **which model reads it**. Two mechanisms did most of the work, both measured rather than assumed: a fresh conversation per query (the shared Anthropic transcript grew context per turn 41K → 162K across a run), and batching a search's recording into one `do_record_listings` call instead of ~25 per-listing turns — which also removed 69% of output tokens, since the model no longer re-emits job objects. **Every `ClaudeAgentOptions` site in the package must pass `setting_sources=[]`, `strict_mcp_config=True` and `skills=[]`** — unset, they load this file in full plus every global MCP server into even a one-word classification call. This was fixed in `agent.py` on 2026-08-21 and the four sites in `tools_generic.py` were missed for three months because the note said "in `agent.py`" (measured 2026-08-24: **22,945 → 294 tokens, 78x**, on the PDF categorization options). Prose cannot enforce this, so `test_every_claude_agent_options_site_limits_context` walks the AST of every module in the package and fails on any site missing them
- **Observability** — the generated Search Queries are logged; every `check_and_record_job` outcome is logged (`new`/`already_processed`/`already_applied`/`too_old`/`auth_required`) so seen-but-deduped is distinguishable from never-seen; per-job logs of the extract signal (`date_posted`/`location`/`closed`/`language_requirement`/`posting_language`/`implied_local_language`/`relocation`/**`residency_scope`, raw and derived**/`education_requirement`/`is_agency`/`end_client`), extract compression, hard-rule short-circuits, triage score vs. final rating, and the rating model used. Blacklist activity is logged in three distinct forms so each is separable: a confirmed rejection, a **name collision** the confirmation call cleared (a blacklist entry matching a different organization by name), and a confirmation **failure that rejected anyway**; expired and undated entries warn once per run. Each run emits a single `Run funnel: {...}` line (queries → listings seen → check-status counts → candidates → extract ok/failed → hard-ruled by reason, including `hard_ruled_blacklisted` → triaged-out → rated 1–5), also written to `cost_log.jsonl` under `funnel`. Per-stage costs (`reference_summary`, `extraction`, `rating`) in `cost_log.jsonl`. **Stage 1b cost is also reported per query** (`_scrape_per_query` → funnel `scrape_per_query`, plus a `Stage 1b cost: "<query>" $X over N iteration(s)` log line carrying prompt/cached/output tokens), because a scraper regression was previously visible only as a stage total — which is how a 68% → 99% drift went unnoticed for weeks. On the OpenRouter path `cached_tokens` from `usage.prompt_tokens_details` feeds the existing `cache_read_input_tokens` field so the two providers stay comparable. New `ui_alerts` kinds: `empty_harvest` (harvest errored or returned nothing), `blocked_click` (a results-list click refused in code), `model_reported` (the model called `report_problem`), and `provider_fallback` (the OpenRouter scraper failed and the Anthropic one ran). Cost is charged **per session**: `total_cost_usd` is cumulative within a `ClaudeSDKClient` session, so only the increase over that session's last reported value is billed (`cost_delta_for` in `agent.py`, keyed on `session_id`). Stage 1b shares one session across all queries and was previously over-counted ~3.5x; stages that open a fresh session per call were always correct. Token counters in `msg.usage` are per-request and keep summing. Every `ResultMessage` is logged with `session_id`, `num_turns`, usage, and `cost_reported` vs `cost_charged` — previously these went to the `rich` console only and never reached the run log, which is why the over-count went unnoticed for nine runs. Every `ResultMessage` **also** logs `is_error`, `subtype` and `api_error_status`, and an errored one emits a dedicated ERROR line: an HTTP-level API failure is reported by the CLI as `is_error=True` with `subtype="success"`, so without those fields an errored result is indistinguishable from a good one — which is exactly how nine identical `error result: success` warnings stayed undiagnosable (2026-08-24). **`api_error_status` is the only field that separates 401 from 429 from 529**, so any new `sdk_query` loop must run `_raise_if_result_error()` rather than testing `structured_output` alone, which silently reads an error as an empty result. The MCP side has the exact same failure: **an exception from `call_mcp_tool` must be passed through `unwrap_exception()` before it is logged**, because `streamable_http_client` and `ClientSession` each open an anyio task group and `str(ExceptionGroup)` is `unhandled errors in a TaskGroup (1 sub-exception)` — which is how 250+ triage warnings carried none of the `model 'qwen3.6:latest' not found` that caused them (2026-08-25). New `ui_alerts` kind `local_model_missing` (OLLAMA_MODEL_NAME_TRIAGE not installed; triage disabled for the run). The extract-signal line also logs `place_names`. New `ui_alerts` kind `location_recommendations` (the country-list reviewer wrote something new), which fires **only on findings written this run** — a permanently-pending set warning on every run trains the user to ignore the block, the same cry-wolf reasoning applied to a merely-unreachable tool server; and a new `cost_log.jsonl` stage `location_review`, `$0.0000` in steady state, which is itself the evidence the guard works. Funnel counters `hard_ruled_location` and `hard_ruled_relocation`; `_hard_rule_category` is an **ordered substring dispatcher**, so the relocation branch must sit before the location one — the relocation reason contains the word "location" too, and the bucketing must not depend on that accident. The extract-signal line logs `implied_local_language` as a single value: it has one source (the classifier), so there is no extractor claim left to correct and nothing for a raw→derived pair to count. `residency_scope` keeps its pair, because it still has two. Location classification logs the countries, regions and cost per **new** location; a cache hit is silent by design, since one line per job for an unchanged fact is noise. **A truncated LLM response is now a named error, not a parse failure**: `chat_openrouter` raises `TruncatedResponseError` on `finish_reason == 'length'` **whether or not `content` is empty** — it previously checked only when content was null, so a truncated *non-empty* body reached `extract_json_object` disguised as malformed JSON. `extract_json_object`'s error reports the length, the head **and the tail**: the head is where a ```` ```json ```` fence sits and the fault is always in the tail, which is exactly how a truncation was misdiagnosed as a fence problem for a week when `finish_reason` had been in the payload the whole time. New `ui_alerts` kind `playwright_mcp_outdated` (a newer `@playwright/mcp` than the pin is published and no TTY was available to ask), raised into `_startup_ui_alerts` rather than `tools_module._ui_alerts` because `run_non_interactive()` resets the latter on entry and would discard anything appended during startup. PDF categorization is logged to the run log rather than `console.print()`-only, because a categorization that never happens means the applied-job corpus stops growing — a degraded run that otherwise looks entirely normal. Third-party HTTP transport logging (`httpx` / `httpcore` / `mcp.client.streamable_http`, one line per MCP tool call) is suppressed to WARNING so the run log stays the run's own audit trail; set `HTTP_LOG_LEVEL=INFO` to restore it when debugging a tool server. **Agent `TextBlock` narration is mirrored to the run log** via `log_agent_text()`, not just `print()`ed to the console — that is where the scraper says "these two searches returned identical results" or "the Remote filter did not stick", and discarding it left the 2026-08-11 collapse diagnosable only from counters. The funnel also carries `listings_distinct` (distinct jobs, beside the call-count `listings_seen` — the gap between them IS the duplicate-coverage signal), `region_overlap` (Jaccard of harvested ids across regions per query; 1.0 means the location filter did nothing), `ui_alerts` (blocks, contract breaches, filters that did not apply, fingerprint drift, low listing counts, early stops) and `health_alerts` from `assess_run_health()`. Those alerts are written to the run log, to **audit log §1b Discovery health**, and into the Telegram run summary under `⚠️ NEEDS ATTENTION`, with the last few runs' yield read back from `cost_log.jsonl` for context — because a saturated run and a healthy quiet run were previously indistinguishable to the user, and four consecutive zero-notification days passed unremarked. The funnel also carries **`queries_failed`** (query -> reason, from `_query_errors`), and `assess_run_health` alerts on it **before** the saturation check — saturation is guarded by `if distinct:`, so a run with zero listings skipped every check that existed and reported itself healthy while all six queries were dying on a 402 (2026-09-02). Recording *that* a query failed is not enough: `_queries_searched` stores only an `'error'` sentinel, so the cause reached the run log and neither the audit log nor Telegram — `_query_errors` carries it into both, beside the sentinel rather than replacing it. The alert distinguishes **all** queries failing (no search completed; 0 listings means nothing) from **some** failing (it names them and says the rest completed) — the all-failed wording on a run with 156 distinct listings made a healthy run read as a dead one (2026-09-10). A failed query also keeps the `check_status` counts it recorded before failing in `_check_status_per_query`, since those listings are real and already queued; otherwise audit §3 reads "error | 0 listings | 2 queued". **Both run exit paths must assess health**: the zero-candidate branch returns before the main path's assessment, so it calls `attach_run_health()` itself and renders the same `health_alert_block()`; two copies of that logic is exactly how one of them came to be missing. `check_status` is reported **per query** as well as run-global (`_check_status_per_query`, in the Stage 1b log line, audit §2/§3 and the run funnel), so a dedup-saturated query is distinguishable from one that barely ran — a distinction the run-global count cannot make. A `filters` ui_alert that a later passing `report_search` for the **same query and region** resolved (the model re-clicked the chip before harvesting) is marked `resolved` and kept out of `⚠️ NEEDS ATTENTION`. The check did its job there, and alerting on it every run teaches the user to ignore the block. Audit §1b still shows both reports
- **Account safety** — Stage 1b drives a **real logged-in LinkedIn account** through the shared Playwright profile (`~/.linkedin-agent-profile`), and a flagged or banned account ends the job search permanently. This outranks coverage, cost, and run time: a slow incomplete run always beats an aggressive one. All scraping goes through the Playwright profile; never scrape from the user's own browser. Specifically:
  - **The browser toolchain version is pinned** (`PLAYWRIGHT_MCP_VERSION`), never `@latest`. `@playwright/mcp` is the entire tool surface the scraper drives that account through, and the snapshot-mode and tool-count facts recorded in Known diagnoses hold for one specific bundle. An unpinned spec lets that change with no commit and no test failure — and it already stopped a run dead at an npx install prompt (2026-09-01). `npx --yes` is safe only *because* the version is exact: it can install nothing but the pin. Upgrades are proposed at startup by Check Browser Toolchain Version and applied by editing the constant, so a new bundle is always watched through one live search before it runs unattended
  - **Nothing in the job results list is ever clicked.** Not a card, not a title, not a logo. The only real `<button>` in a row is **Dismiss**, which the accessibility tree disguises as the card itself, and one stray click permanently removes a job from the User's feed — this already happened, to three real jobs. Clicking is also unnecessary: Harvest Job Listings reads everything from the DOM. This is an invariant, not a preference. On the OpenRouter path it is **enforced in code**, not merely prompted: `_guard_call()` in `scrape_openrouter.py` refuses any `browser_click`/`browser_hover` naming `job-card-component-ref`, `SearchResultsMainContent` or `Dismiss`, and any `browser_evaluate` that clicks or dispatches events — each refusal logged and counted as a `blocked_click` alert. This is not theoretical: on 2026-08-21 the model tried to drive the page with JavaScript `.click()` and was refused. Unit tests assert both the guard and that the prompt states the rule
  - **JavaScript may read the page, never drive it** — `browser_evaluate` for extraction and scrolling only; never to click, submit, or dispatch events
  - **Never apply, save, follow, or dismiss** anything
  - **Randomised pacing** between searches (`SCRAPER_INTER_SEARCH_DELAY_SECONDS`) and between queries (`SCRAPER_INTER_QUERY_DELAY_SECONDS`). Every delay is a **(min, max) range drawn uniformly** — a fixed interval is itself a robotic signature, and a unit test rejects degenerate ranges. Inter-query pacing is enforced **in code** (`_human_pause`), not merely prompted, because a model under turn pressure will skip a prompted wait. There is no per-listing delay any more: harvesting is one read-only call, so there is no interaction burst to disguise
  - **Stop on any CAPTCHA, verification challenge, or "unusual activity" notice** rather than work around it, and the recovery pass explicitly refuses to retry into a block. A challenge means LinkedIn already suspects automation; solving or immediately retrying converts a soft signal into a confirmed evasion pattern, which is what escalates to a restriction. Backing off keeps it a blip — the 2026-08-06 run went 0 listings at 11:43 and 74 at 12:08 after a pause. This applies only to *challenge pages*; zero results or a slow load still get a normal retry
- **Search coverage** — every generated Search Query must actually be searched, once per configured region. Stage 1b sends one request per query (own turn budget) on one shared session (one browser attachment); the run audit log distinguishes "never searched" from "searched, found nothing", which a single `0 jobs` count cannot. Turn starvation is silent — the model stops mid-query and the run still reports a listing count — so `SCRAPER_MAX_TURNS_PER_QUERY` (180) is set with headroom: harvesting is one call per search, but each listing still costs a `check_and_record_job` (and possibly a `queue_candidate`) turn, so the budget scales with `SCRAPER_MAX_LISTINGS_PER_SEARCH` × regions, and `num_turns` is logged per request to make exhaustion visible. **Under-coverage is detected by yield and by turn usage, not only by a zero count**: a query below `SCRAPER_MIN_LISTINGS_PER_QUERY` (5) is retried, and one that stops well short of its turn budget is warned about. Stopping *early* is as much a failure mode as running out — on 2026-08-11 every query used 10 of 90 turns and returned exactly one listing, and a `seen == 0` trigger missed all of it
- **Rejection auditability** — `python main.py -n --audit` runs normally but additionally re-rates every gate-killed Job Posting (hard-ruled or triaged-out) with the strong rater, logging any **false negative** (gate dropped it but the strong rater scores ≥3) plus a run-end count. Diagnostic only — saving/notification behaviour is unchanged
- **Resilience** — the LLM MCP tool servers are optional: summarization falls through its provider chain (last resort: full reference block), triage fails open to the rating call. **A server that is down and a model that is misconfigured are different failures and must not be reported the same way.** An unreachable server is the supported case: quiet, fail open, no alert. An `OLLAMA_MODEL_NAME_TRIAGE` that is not installed is a misconfiguration that silently disables the free gate on every job, so `preflight_local_model()` catches it once before Stage 2, logs one ERROR naming the models that *are* installed, raises a `local_model_missing` health alert, and disables triage for the run rather than letting each job rediscover it (2026-08-25: ten days of dead triage, 63 identical warnings in the last run alone). `generate_local()` raises `LocalModelMissingError` rather than a generic failure for the same reason. **The remote provider needs the same distinction, and for a third reason: control flow.** A page that would not load breaks one query and the next should still run; a 402, 401, 403 or 429 breaks *every* query, so `provider_error()` raises `ProviderUnavailableError` and `run_scraper` re-raises it — aborting the query loop so the caller falls back to the Anthropic scraper, instead of burning the remaining requests and their pacing delays on calls that cannot succeed. Queries the loop never reached are marked `'error'` too, so a query that never ran is never reported as "searched, found nothing". Classify on the **HTTP status**, never the reason phrase. **A 500/502/503/504 is a third class: a gateway hiccup, not a refusal.** It gets one retry of the same chat call after `TRANSIENT_RETRY_DELAY_SECONDS` (`TRANSIENT_HTTP_STATUSES` in `scrape_openrouter.py`, deliberately disjoint from the abort set), and a second failure is a one-query failure rather than an abort. Without it, one 502 on 2026-09-10 discarded a query's conversation 71 iterations in and lost its second region. A **transient** 500/502/503/504 is neither: `ScrapeSession.run` retries that one chat call once, after `TRANSIENT_RETRY_DELAY_SECONDS`, and only then fails the query. Without the retry, one gateway hiccup threw away a query's whole conversation 71 iterations in, along with its second region (2026-09-10). Because the fallback is now genuinely reachable and ~9x more expensive per search, a provider outage must always surface as a health alert — an unnoticed one bills Anthropic prices indefinitely. **The location classifier fails OPEN, deliberately the opposite of the blacklist's fail-closed rule**, and the asymmetry is the point: a blacklist hit means a name the user wrote down already matched, so an outage must not readmit it; a classifier call happens only because *nothing* the user wrote down matched, so an outage must not begin excluding jobs it would otherwise have kept. An unparseable region, an unknown place, or a down tool server all yield "no country named", which no policy can reject. Review Country Lists fails open on everything and never aborts a run or changes a rating; a corrupt `location_recommendations.yaml` is **skipped, not rebuilt**, because unlike a cache it holds decisions the user made by hand
- **Idempotency** — processed-job records persist across runs so jobs are never evaluated twice; applied-job ingest is a no-op for already-dated files
- **Applied-date durability** — an Applied Job Record's date is carried by the filename prefix, `index.yaml`, and mtime independently, so it survives a move, copy, or backup restore that drops filesystem metadata
- **Applied-job horizon** — only Applied Job Records from the last `APPLIED_JOBS_HORIZON_DAYS` (90) feed query generation, the ideal-role profile, and the already-applied blocklist; older PDFs are retained on disk, never deleted
- **Notification latency** — Telegram alerts sent immediately when a job is rated 4 or 5 during evaluation
- **Rating hard rules (applied in code before any LLM scoring)**: a company on the user's Preferences **blacklist** (poster or end client) → 1; closed postings → 1; postings > 30 days old → 1; jobs in a `sponsorship_required_in` location with no sponsorship offered → 1; a required language outside `languages` → 1; a hard requirement for a degree in `reject_required_degrees` → 1. a `location` anchored in a `locations.reject_regions` region (or named in `locations.exclude`) → 1; a stated `relocation` to such a region → 1. **For a `remote` posting the anchor rule is conditioned on `residency_scope`**: it fires when the JD pins residence to the anchor country, and otherwise only when a country the posting names is outside the EU — a silent `Poland (Remote)` is kept, a silent `United Kingdom (Remote)` is not. `hybrid`/`onsite`, the `relocation` rule, and `locations.exclude` are all unconditioned. All but the closed/stale rules are preference-driven and disabled when their lists are empty. The blacklist is the one rule here that is not purely deterministic — an exact case-sensitive name hit costs one cheap LLM confirmation call, and that call **failing rejects anyway**; it is also the only rule the user authors by hand, is unrelated to the automatically derived already-applied blocklist, and an **expired** entry (180 days after its `added` date) does not reject. Required relocation is **flagged rather than rejected for an acceptable location**, and rejected only when it names an excluded region — the narrow exception, because there the JD is asserting a categorical requirement rather than code inferring one from a location string. The geographic rules are the second pair here that are not purely deterministic: an unlisted location costs one cheap **cached** classification, and unlike the blacklist that call **failing fails open** (nothing the user wrote down had matched). The degree rule fires **only on hard requirements** — "MSc preferred", "Master's or equivalent experience", and "Bachelor's or Master's" all survive, because an auto-reject is unappealable and a posting that would accept experience instead must never be killed. **Being posted by a recruiting agency is not a hard rule and not a rating cap** — it warns only. Applying once through a staffing firm must never suppress the other companies it posts for, and an agency posting can be an excellent role (CyberCoders rated 5/5 on 2026-08-11). Two unit tests pin this: the rating is identical with and without `is_agency`, and an agency posting is never hard-ruled
- **Rating caps (applied deterministically in code AFTER LLM scoring)**: two ceilings, both in `apply_rating_caps()` — a `hybrid`/`onsite` Job Posting whose location is not in the Preferences `hybrid.acceptable_locations` is capped at `hybrid.rating_cap`, and a Job Posting whose `posting_language` is outside Preferences `languages` is capped at `foreign_language_rating_cap`. When several apply the **lowest wins and every applicable reason is reported**, so the log line and the saved job name everything that held the rating down. A cap is a **ceiling, not a rejection or a floor** — the job is saved, recorded, and auditable, and a rating already at or below the cap is untouched; it simply cannot reach the ≥ 4 notification threshold. The caps exist because the prompt alone is not sufficient: the evaluator rated a hybrid role in an unacceptable location 4/5 while naming the hybrid location as a drawback in its own reasoning, and rated a French-language posting 4/5 after translating it into English while condensing. Every cap is logged (`Rating capped: … 4 → 3 (reason)`) and counted in the run funnel as `rating_capped`
- **An advisory reviewer must not be able to argue with the config it reviews** — the country-list
  reviewer's prompt used to present `relocation_note` as *"their stated policy on where they will
  work, in their words"* while presenting the lists as merely *"as they stand today"*. Handed an
  apparent contradiction between an authoritative-sounding sentence and a provisional-sounding
  config, the model resolved it against the config: on its first live run **10 of 11
  recommendations told the user to put Germany, France, Poland and Romania on
  `acceptable_locations`** — i.e. to switch off the geographic gate — each reasoned as *"EU member
  state your policy deems acceptable, but the western_europe reject rule would wrongly auto-reject
  it."* `relocation_note` is guidance to the **rater** about tolerating relocation; `reject_regions`
  is where the person will **live**. The prompt now states that the lists are decisions and are
  correct, makes `keep_as_is` the explicit default, and is not given the note at all — after which
  the same run returned 31 `keep_as_is` of 32. Two related rules fall out: an entry flagged as
  "ambiguous" only counts against a place the user did **not** list (checking entries against their
  own siblings made deliberate spelling pairs — `Valencia`/`València`, `Málaga`/`Malaga`,
  `Milan`/`Milano` — flag each other in both directions, 5 false findings against 1 real, which is
  the cry-wolf failure the alert exists to avoid); and a paid review that keeps **none** of the
  reply logs a WARNING, because writing zero recommendations is otherwise indistinguishable from
  having nothing to recommend.
- **Names stay proper; fold at the comparison, never at the store** — `EU` not `eu`, `Málaga` not
  `malaga`, `Spain` not `spain`. Case-folding is a lossy transform: you can always `casefold()` a
  name at the point you compare it, and you can never recover the case afterwards. So a name keeps
  the case it arrived with all the way through — the classifier's `countries` and `place_names`,
  the preference lists (`locations.exclude`, `hybrid.acceptable_locations`,
  `sponsorship_required_in`), `end_client`, blacklist entries — and each comparison folds locally
  if it needs to (`is_eu_member`, the sponsorship substring test, `undecided_countries`).
  **Controlled vocabularies are the deliberate exception and are stored folded**, because they are
  identifiers rather than names: `regions` (`southern_europe`), `languages`, `workplace_type`,
  `education_requirement`, `residency_scope`, `reject_required_degrees`. Cache keys fold too
  (`cache_key`), because a key is not displayed. This is not a style rule — it was violated twice
  in one change and cost real behaviour both times: lowercasing the location lists made `malaga` an
  entry that could never match `Málaga`, and lowercasing the classifier's `countries` while the
  lists held proper names made `undecided_countries` report **every** already-decided country as
  undecided, sending each to the LLM on every run while the "zero calls in steady state" claim
  still read as true. Regex literals follow the same rule (`_AREA_WORDS` says `EU`, `EMEA`,
  `Schengen`) even where `re.IGNORECASE` means the case does no work — reading `emea` in source
  gives no hint whether it is an acronym, a country or a typo.
- **Knowledge locality** — project decisions, diagnoses, and conventions live in tracked repo files. Machine-local paths (including `~/.claude` and its per-project memory directory) are never used for project knowledge, because they do not reach another machine. `run_dir/` is the one deliberate exception: gitignored because it holds personal data (resume, applied-job PDFs, saved postings, `JOB_REQUIREMENTS.md`, `preferences.yaml`)
- **No personal information in tracked files** — nothing in git may identify or describe whoever is running the agent: no home location, work-authorization or immigration status, languages, education, employers or recruiters dealt with, compensation targets, or absolute paths containing a username. Every such fact is a **preference**, and preferences live only in `run_dir/preferences.yaml`. Tracked code reads them through `preferences.py` and must behave sanely when they are absent — the defaults are neutral, so an unconfigured checkout applies no work-authorization, language, or education gate rather than inheriting someone else's situation. Tests pin their own fixed preferences in `tests/conftest.py` and must never read the real file. When adding a rule that encodes a personal fact, add a preference key; do not hardcode the fact. The region vocabulary in `location.py` is the boundary case worth naming: *which* regions exist is world knowledge and belongs in tracked source, while *which* of them the user will not work in is personal and lives only in `preferences.yaml`

## Git conventions

- Use `git mv` when moving or renaming tracked files
- Use `git rm` when deleting tracked files
