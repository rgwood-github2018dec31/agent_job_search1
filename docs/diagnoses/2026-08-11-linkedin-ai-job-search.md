# 2026-08-11 — LinkedIn migrated the account to AI-powered job search

> **SUPERSEDED on 2026-08-18 — see [2026-08-18](2026-08-18-filter-chips-were-there.md).** The account really was migrated to
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
