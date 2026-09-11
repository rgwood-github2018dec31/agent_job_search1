# 2026-08-25 — the local model tag disappeared, and triage died without saying so

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
