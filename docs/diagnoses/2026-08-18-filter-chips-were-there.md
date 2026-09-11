# 2026-08-18 — the filter chips were there all along, and the region axis was dead

A run queued **1 candidate from 300 listings inspected** — the end of a four-day slide (26 → 4 → 4
→ 2 → 1 new jobs) during which **not one notification was sent** and nothing reported a problem.
Stage 1b was mechanically healthy throughout: 12/12 searches ran, no block, no auth wall.

**The [2026-08-11 entry](2026-08-11-linkedin-ai-job-search.md) was wrong, and this is the correction.** Its central claim — that the
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
