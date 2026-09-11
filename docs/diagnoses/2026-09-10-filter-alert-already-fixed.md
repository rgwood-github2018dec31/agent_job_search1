# 2026-09-10 — a filter alert for a search that was already fixed

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
