# 2026-08-13 — the extractor translates, so the JD's own language was invisible

`saved_jobs-2026Aug11/job_posting-4450073662-rating_4-valtech-…md` is a French-language posting —
*"Scientifique principal des données en IA"* — that was **rated 4/5 and notified**. Nothing in the
pipeline was wrong on its own terms; the language simply had no field to land in.

Three things that file shows, each of which shaped the fix:

1. **The extract was English.** Only `title` survived in French; the description came back as
   fluent English prose. The extractor translates as it condenses, so its own output carries no
   evidence of the source language — and any code-side heuristic run over the *extract* would
   conclude "english". That is why `posting_language` must be judged from the source page by the
   extractor, and why there is deliberately no `derive_posting_language()`.
2. **The location said nothing.** `location` was `Canada (Remote)` and `language_requirement` was
   `english` — both accurate. The Quebec signal was one sentence in the body: *"English proficiency
   required for regular communications with clients and colleagues outside Quebec."* A
   location→language lookup table in code would have missed it, which is why `local_language` was
   also the extractor's judgement and was explicitly told to read the whole body.
   **This half is superseded (2026-09-09).** The field is now `implied_local_language`, resolved
   from the location alone by the cached classifier, and the extractor no longer supplies it — a
   language a *place* speaks is world knowledge, and leaving it to a per-job judgement meant two
   jobs in one city could disagree. The cost is exactly this posting's second bullet: a working
   language named only in the body is no longer detected. **The outcome here is unchanged** —
   `posting_language` is a fact about the *page*, is still the extractor's judgement for the reason
   in point 1, and is what caps this posting.
3. **The rater saw the French title and still said nothing.** Its four warnings covered salary,
   stack, and the consulting-agency angle. As with the hybrid cap, a structural fact cannot be left
   to the rater's discretion.

The fix is a cap, not a rejection: `foreign_language_rating_cap` (default 3) keeps such a posting
saved and auditable while putting it below the ≥ 4 notification threshold. A non-English
`implied_local_language` is only a warning — many roles in non-English locations genuinely operate
in English, and the user asked to be told, not to have them dropped.
