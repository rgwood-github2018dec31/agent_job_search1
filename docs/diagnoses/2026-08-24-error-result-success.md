# 2026-08-24 — "error result: success", and the leak that never left `tools_generic.py`

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
`skills` at their defaults — the exact bug the [2026-08-21 entry](2026-08-21-stage-1b-cost.md) documents as fixed. It was
fixed *in `agent.py`*, and that scoping is precisely why nobody looked here. **Measured with
`get_context_usage()` on the real categorization options: 22,945 tokens → 294, a 78x reduction**,
all of it `memoryFiles` (CLAUDE.md, loaded in full into a Haiku call that classifies a PDF into one
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
