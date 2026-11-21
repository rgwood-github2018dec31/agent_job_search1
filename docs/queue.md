# Queue

Planned work that has been approved to queue but not yet carried out. Newest item last, one section per
item. When an item is done, its requirement changes go into [requirements.md](requirements.md) and the
section is removed here.

## 1. Estimate a role's languages from the whole JD, not the place alone (queued 2026-11-12)

### Context

Job 4487061547 (Moxie, Staff AI Engineer) is listed at "Any-Martin-Rieux, Hauts-de-France, France (Remote)".
The notification said `Likely spoken languages: french 100%`, and the rater repeated it as "a French-speaking
environment". The rest of the JD points to English: it is written in English, the salary is in USD
(`$97K/yr - $225K/yr`), and it says "global, remote-first team" and "700+ practices nationwide".

The cause is that `derive_likely_spoken_languages()` (`agent.py:3412`) asks `classify_location()` about the
place only, and caches the answer by place. The only things that can override it are a stated working
language or a required one, and this JD states neither. The `marginal` tag needs `area_wide` or `broad_area`,
which this posting doesn't have either.

A second case has the same cause. Job 4484622049 (whizdom.ai, "Poland (Remote)") warned
`Likely spoken languages: polish 80%, english 20%`. That JD is written in English, requires no language, and
says "Flexible remote environment with strong overlap with European time zones preferred". The only place
constraint it names is a time zone, so the holder can live anywhere, even outside the EU, and English-only
should be fine. `residency_scope` came back `residency_scope_not_known`. Because of that, the warning wasn't
marked marginal, and the rater wrote "Remote in EU".

You answered Q1 with: "given all the information in the JD we should make a guess as to what languages are
expected for the role". So we add a model estimate, made per job from the whole JD. This reverses the
documented rule "The extractor is not asked and cannot override it" for this one case.

### Design

- **P1 — When it runs.** The estimate is only asked for when the place-only split would warn
  (`location_languages_warn(extract)`) and the posting states no working language
  (`not posting_states_working_language(extract)`). Everywhere else nothing changes and nothing is spent.
- **P2 — The call.** One single-turn call to a new `MODEL_NAME_ROLE_LANGUAGES_ESTIMATE = MODEL_NAME_SINGLE_TURN`
  (in `config.py`), with retries like `gate_fields.ask_field`. The input is the page text
  (`page.model_text`) plus code-read facts, each on its own line so it can be quoted: header location,
  workplace badge, `Posting written in: <posting_language>`,
  `Languages the posting requires: <language_requirement or "none named">`, the salary as stated, and the
  location's split. The prompt says that a posting written in one language and requiring no other is
  evidence for that language, and that a place listed for a remote role is where the holder may live, not
  where the team sits. The reply is `{"quotes": [...], "reason": "...", "languages": {name: fraction}}`.
- **P3 — Citation (Verify Source Reference).** Each quote must be found by `evidence.find_quote` in the page
  text joined with the fact lines. This works the same way `gate_fields.ask_gate_fields` grounds header facts.
  If there are no quotes, or any quote is not found, the estimate is rejected and counted with
  `evidence.count_rejection`, and the place-only split stands as it does today. Code checks the shape:
  lowercase language names, fractions ≥ 0, normalised to 1, and a "cannot estimate" answer allowed. It reuses
  `location._language_shares` if that fits.
- **P4 — How it's used.** This is decided in code. If the estimate is verified,
  `location_languages_warn` / `working_language_warnings` check `WORKING_LANGUAGE_MIN_KNOWN_MASS` against the
  estimate instead of the place split:
  - Warning bullet: `Expected languages (model estimate from posting): english 90%, french 10%`, shown only
    when the estimate itself is under the threshold.
  - The place split is still shown, labelled `Local language: french 100% — <location>` (daily life where
    you would live), the same way the Lucentive case is labelled.
  - Rater input (`agent.py:3141` block): `Expected languages: … (model estimate from the posting; cites: …)`
    next to `Likely spoken languages: … (local)`.
  - The estimate gates nothing and caps nothing, like `likely_spoken_languages` today.
- **P5 — Cache.** Results are cached per job id in `run_dir/role_languages_estimate_cache.yaml` through
  `cache_io`, following `salary_estimate.py` (with a schema version and the model name). A job that is seen
  again gets the same answer. If the call fails, it fails open to today's behaviour and logs a warning with
  the request and the reply (`reply.describe()`).
- **P7 — A remote role whose only constraint is a time zone is `area_wide`.** Change the `residency_scope`
  definition in `gate_fields.DEFINITIONS`: a remote job whose only stated constraint on where the holder
  lives is a time-zone overlap, required or preferred, names no country. It is therefore `area_wide`, meaning
  any country in those time zones. Reading rule (d), "a time-zone requirement is not a place", stays as it is
  for `relocation`, and `relocation` still names no place. Results:
  - The gates become more permissive, never stricter (`country_only` is the only scope that rejects).
  - The location's language warning gets the `marginal` tag through `working_language_is_marginal`.
  - The rater no longer reads "Poland" as the place the holder must live.
  - The Ask Gate Fields entry in `docs/requirements.md` gets a dated note for whizdom.ai 4484622049.
- **P6a — Fits the latest commits (checked 2026-11-12, HEAD `fffb319`).** Nothing since `c7736a9` touches
  `gate_fields.py` or the language code. Three recent changes shape the details:
  - `a6c9219`: a step can no longer replace its kind, so `role_languages_estimate` is a new, unique step kind.
    It is not an exit and records no outcome step.
  - `fffb319`: the Job Path / Job Audit draws a job's path from its outcome steps, so add the new step to the
    Job Audit report rendering (`job_audit.py`, `scripts/job_trace_report.py`) next to `gate_fields`.
  - `a7514cd` and `79a5260`: the page text may come from Exa or a re-read. Use whatever `page.model_text`
    is after Stage 2's re-read, the same text Ask Gate Fields reads.
- **P6b — Forced tool call, not a JSON reply (2026-11-12).** Uncommitted work in progress moves Ask Gate
  Fields to a forced `submit_answer` call through `triage.chat_openrouter_tool`, raising
  `UnusableToolCallError`. Once that lands, the estimate call uses the same pattern (a `submit_role_languages`
  tool with `quotes`, `reason`, `languages`) instead of `chat_openrouter` + `extract_json_object`.
- **P6 — Job Trace.** The run records a `role_languages_estimate` step through `job_trace.record`, with the
  input facts, quotes, verdicts and the result.

### Files

- New `src/agentic_job_search/role_languages_estimate.py`: question, parse, ground, cache and
  `estimate_role_languages(candidate, extract, page, stage_stats)`. It reuses `triage.chat_openrouter`,
  `evidence`, `cache_io`, `text_budget` and `extract_json_object`, the same way `gate_fields.py` does.
- `src/agentic_job_search/agent.py`:
  - Call the new function in Stage 2 after `ask_gate_fields`, because it needs `stated_working_language`,
    and before the rating.
  - Store the result as `extract['role_languages_estimate']` and add it to the extract field list
    (`agent.py:4659`).
  - Update `location_languages_warn`, `working_language_warnings`, the rater-input block and
    `deterministic_pros`. A verified estimate with mostly user languages may add
    `Team likely works in English (model estimate)`, but not the "stated in posting" pro.
  - Add a `stage_stats` entry for the cost.
- `src/agentic_job_search/config.py`: the model constant and the attempts and temperature constants.
- `docs/requirements.md`:
  - New use case **Estimate Role Languages**, with <<extend>> Detect Likely Spoken Languages.
  - Amend Detect Likely Spoken Languages: drop "the extractor cannot override it" for this case and add a dated
    note for the Moxie 4487061547 incident.
  - Add the field to the Job Posting business object and note that the call cites.
- `docs/architecture.md`: the model routing line. `AGENTS.md`: the file-structure line for the new module and
  its cache.

### Verification

- Unit tests in a new `tests/test_role_languages_estimate.py`, with a mocked `chat_openrouter`, fixture
  taken from the Moxie page text:
  - A verified English estimate replaces the French warning with the Local-language label.
  - An ungrounded quote falls back to the French warning.
  - No call is made when a working language is stated or the place doesn't warn.
  - A cached result is reused.
  - A server error fails open.
  - Each test can fail (NFR "Tests must be able to fail").
- A test that the `residency_scope` definition text includes the time-zone rule, and a mapping test showing
  `area_wide` plus remote gives a marginal warning. `tests/` already covers `working_language_is_marginal`.
- `uv run pytest -m "not live_agent_claude and not live and not network"` and `uvx ruff check`.
- Live check (costs cents): `python main.py --job-ids <file listing 4487061547 and 4484622049>`, then check
  the saved postings in `run_dir/replays/`:
  - Moxie should show English-heavy `Expected languages` and `Local language: french 100%`.
  - whizdom.ai should show English-heavy `Expected languages`, `residency_scope` `area_wide`, and a
    marginal Polish local-language line.
  - The rater reasoning of both should match.

## 2. Find out why GLM times out when rating by function call (queued 2026-11-12)

### Context

On 2026-11-12 the rater moved to a forced `submit_rating` function call, with `reasoning` ahead of `rating`
(`triage.rate_with_openrouter`; Ask Structured Answer in requirements.md). GLM was left on JSON in text
(`config.MODEL_NAME_RATING_TEXT_REPLY`) because of a GLM-only problem found by
`scripts/replay_forced_tool_ab.py` (`run_dir/replays/forced-tool-ab-20261112-124054`):

- As a function call, `glm-5.3-flash` (the fallback rater, `MODEL_NAME_RATING_FALLBACKS`) timed out at
  `LLM_MCP_SINGLE_TURN_TIMEOUT_SECONDS` on 3 of 10 saved rater inputs. The other 7 took a median of about 100 s.
- In text, the same 10 had no timeout and took a median of about 89 s.
- DeepSeek answered all of its asks in either form in about 9 s.

That is only 10 inputs, each asked once, and that run used the rating-first schema, not today's reasoning-first one.
The cause is not known. It may be longer thinking when GLM is made to call a function, or a slow endpoint for function calls.

### Steps

- Read the 3 timed-out calls: which endpoint was pinned (`routing_for`), and whether the server logged a partial reply.
- Ask GLM again on its saved inputs in both forms, with repeats and today's reasoning-first schema:
  `uv run python scripts/replay_forced_tool_ab.py --rater 206 --gate 0 --model z-ai/glm-5.3-flash --forms text=2,tool=2`
  (costs cents; needs a go-ahead). Compare time, completion tokens and timeouts.
- If the function call is no slower, drop GLM from `MODEL_NAME_RATING_TEXT_REPLY`. Otherwise record the cause
  in requirements.md (Known open issues) and keep it in text.
