# 2026-07-23 — "the agent isn't finding any jobs" is a top-of-funnel problem

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
