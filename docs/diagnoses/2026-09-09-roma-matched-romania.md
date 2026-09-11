# 2026-09-09 — `roma` matched `Romania`, and the gate never ran

Two Romanian postings were rated **4** (notified) and **3** on Sep 07-08 —
`saved_jobs-2026Sep07/job_posting-4453702157-*` and `saved_jobs-2026Sep08/job_posting-4452991539-*`.
Both said `Romania (Remote within country)`. Both carried `Relocation required: Romania`. The
relocation hard rule that exists for exactly this was correct, and **never got to run.**

`hybrid_location_is_acceptable()` matched its tokens with a bare `token in haystack`, and
`hybrid.acceptable_locations` contains **`roma`** — Rome. `'roma' in 'romania (remote within
country)'` is `True`. That list is **tier 1 and exempts outright**, so for every Romanian posting
the deny list, the classifier and `reject_regions: [eastern_europe]` were never consulted. It
swallowed `parma, emilia-romagna, italy` too.

**Nothing looked wrong, and that is the shape worth remembering.** In the *same* run Bulgaria,
Poland, Czechia and Lithuania were auto-rejected correctly. A gate that fails for one country while
its four neighbours pass through it correctly produces no anomaly anywhere: the log line for an
exempted job reads exactly like the log line for a genuinely acceptable one, because "no rejection"
is what both look like. **The failure mode of a matching rule is silently *more permissive*.**

**The evidence was also unreachable from the cache.** Tier 1 short-circuits *before*
`classify_location`, so `location_cache.yaml` holds no record of the jobs the bad entry exempted.
`romania` is in it at all only because one posting phrased its location as `"romania (remote) — role
is remote anywhere in the eu"` and reached the classifier by a different route (`derive_local_language`,
since renamed `derive_implied_local_language`).
This is why the new reviewer's country universe comes from a **live run-scoped accumulator** rather
than from the cache — and why that accumulator records on the **cache-hit path too**, since in
steady state almost every location is a hit.

**Three more live collisions existed on the same list**, none of which had bitten yet: `turin` ⊂
`turing` (a location or relocation string naming Turing Labs would have exempted the job), `usa` ⊂
`jerusalem` / `sausalito`, and `milan` ⊂ `milano` (masked only because `milano` is separately
listed).

**The obvious fix is itself a trap.** `re.escape('u.s.')` is `u\.s\.`, and `\bu\.s\.\b` requires a
word character after the final `.` — so it matches **neither** `'u.s.'` nor `'remote, u.s. only'`.
Writing the boundary as `\b…\b` would have silently deleted a real entry while fixing the reported
one. The matcher uses letter/digit lookarounds instead, and
`test_location_token_matcher_handles_punctuation` fails if anyone "simplifies" it back.

**`tests/conftest.py` had been doing it right the whole time.** Its `_GEO_RE` is word-boundary
anchored, with a comment explaining that `nice` is inside `Venice` — *"The production classifier is
an LLM and has no such problem; the stub must not invent one."* The test double was more careful
than the production code it doubles, and nothing compared them.

**A closing note on why `roma` was ever on the list.** It is one of a set of hand-maintained
spelling pairs — `rome`/`roma`, `lisbon`/`lisboa`, `milan`/`milano` — added because the lists were
lowercase ASCII substrings and could not otherwise reach a place written in its own language. **The
workaround caused the bug it was working around.** The same representation also made `malaga` an
entry that could never match anything, because LinkedIn writes `Málaga` — and a dead entry is
indistinguishable from one that has not come up yet. The lists now hold proper names, matched
case-sensitively and accent-exactly, and an exonym no spelling can reach (`Torino` → `Turin`) is
resolved through the classifier's `place_names`, so the pairs no longer need maintaining by hand.

**Removing the folding is what forced the boundary class to change**, and the two cannot be
separated: `(?<![a-z0-9])` blocks only lowercase ASCII neighbours, which was correct only in
combination with a caller that lowercased first. Matching proper names with it, `ROMA` reaches into
`ROMANIA` and `Roma` into `Romaña` — this same bug, in caps and via an accent. One test covers all
three forms for that reason.

**And the fix nearly recommended the bug back.** The reviewer's new dead-entry check asks "would
this entry have matched if it were written the way the place is written", which means folding case
and accents. Written as a substring test, folded `roma` is inside folded `romania`, so it reports
*"'Roma' never matched, but 'Romania' did — write the entry the way the place is written."* Folding
is allowed to ignore case and accents; it is never allowed to ignore boundaries. Pinned by
`test_reviewer_never_advises_writing_roma_as_romania`.

**Separately, and deliberately: the anchor rule is now conditioned on residence.** The 2026-09-04
entry ([2026-09-04](2026-09-04-geography-had-no-field.md)) rejected on the location the role is advertised in. CLAUDE.md had said three times that
*"a bare location is not a residency requirement"* — and the code did not act on it. It does now,
for `remote` postings only, via a new extractor field `residency_scope` (see Detect Residency
Scope). Measured over the saved corpus, that keeps **168** previously-rejected postings, including
5/5 roles in Poland, France and Berlin. That is a large, intentional loosening of a gate written
four days earlier, and it is what `relocation_note` has said all along. **What the 2026-09-04 entry
was actually built to stop is untouched**: hybrid and on-site roles are not spared, a stated
`relocation` is not spared, and `locations.exclude` is not spared — the spare sits *below* the deny
list, which after this change is the only way to drop one specific EU country for a remote role. An
early draft put it above, silently making that list inert, and only a test named for the escape
hatch caught it.
