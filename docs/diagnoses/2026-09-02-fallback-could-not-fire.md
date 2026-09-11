# 2026-09-02 — the fallback that existed, was correct, and could not fire

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
