# 2026-08-25 — the rater calibrated on the jobs the user had already moved on from

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
