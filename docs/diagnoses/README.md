# Known diagnoses

One file per incident, newest first. Each one records what broke, why nothing reported it, and
which rule or test now prevents it. Add a new entry at the top.

| Date | Diagnosis | Takeaway |
|---|---|---|
| 2026-09-10 | [A filter alert for a search that was already fixed](2026-09-10-filter-alert-already-fixed.md) | A later passing `report_search` resolves an earlier `filters` alert; a 502 gets one retry |
| 2026-09-09 | [`roma` matched `Romania`, and the gate never ran](2026-09-09-roma-matched-romania.md) | List entries are whole-word, case- and accent-exact proper names; a matching rule fails silently *more permissive* |
| 2026-09-04 | [Geography had no field, so the rater invented one per job](2026-09-04-geography-had-no-field.md) | A purely geographic gate backed by a cached classifier; no `max_tokens` ceiling |
| 2026-09-02 | [The fallback that existed, was correct, and could not fire](2026-09-02-fallback-could-not-fire.md) | `ProviderUnavailableError` aborts the query loop; both run exit paths assess health |
| 2026-09-01 | [`npx @playwright/mcp@latest` stopped a run at an interactive prompt](2026-09-01-npx-playwright-latest-prompt.md) | Pin `PLAYWRIGHT_MCP_VERSION`; ask about upgrades only in interactive mode on a TTY |
| 2026-08-25 | [The local model tag disappeared, and triage died without saying so](2026-08-25-local-model-tag-disappeared.md) | Unwrap `ExceptionGroup`s before logging; preflight the local model once per run |
| 2026-08-25 | [The rater calibrated on the jobs the user had already moved on from](2026-08-25-rater-calibrated-on-oldest-jobs.md) | Reference jobs are sorted newest applied first; count distinct listings, not calls |
| 2026-08-24 | ["error result: success", and the leak that never left `tools_generic.py`](2026-08-24-error-result-success.md) | `api_error_status` names the cause; every `ClaudeAgentOptions` site limits context |
| 2026-08-21 | [Stage 1b was 87% of the bill, and CLAUDE.md was part of the reason](2026-08-21-stage-1b-cost.md) | Scraper moved to the OpenRouter flash tier (9.3x cheaper); anti-fabrication toolset |
| 2026-08-18 | [The filter chips were there all along, and the region axis was dead](2026-08-18-filter-chips-were-there.md) | Filter by clicking chips on the results page; judgement happens in code |
| 2026-08-13 | [The extractor translates, so the JD's own language was invisible](2026-08-13-extractor-translates.md) | `posting_language` is judged from the source page; foreign-language rating cap |
| 2026-08-11 | [LinkedIn migrated the account to AI-powered job search](2026-08-11-linkedin-ai-job-search.md) | *Partly superseded by 08-18.* Job ids come from `componentkey`; never click the results list |
| 2026-07-23 | ["The agent isn't finding any jobs" is a top-of-funnel problem](2026-07-23-top-of-funnel.md) | Don't loosen gates after a quiet run; the leverage is in Stage 1 |
