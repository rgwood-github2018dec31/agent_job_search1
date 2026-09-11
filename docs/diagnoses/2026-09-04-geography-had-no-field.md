# 2026-09-04 — geography had no field, so the rater invented one per job

Seven remote roles in Germany and the Netherlands were rated 4-5 and notified across Sep 02-03
(Finom/Berlin 5, Flip/Stuttgart 5, team.blue/Berlin 5, NVIDIA/Germany 4, Archer/Germany 4,
LangChain/Amsterdam 4, Jobgether/NL 4). On the Sep 03 run **13 of 24 jobs rated 4-5 were in
non-English-speaking countries.** Nothing malfunctioned, and that is the finding: every gate
declined *correctly*.

| Gate | Field | Value | Outcome |
|---|---|---|---|
| hard language rule | `language_requirement` | `english` | no reject — correct |
| foreign-language cap | `posting_language` | `english` | no cap — correct |
| hybrid cap | `workplace_type` | `remote` | never consulted a location list |
| — | `implied_local_language` | `german` / `dutch` | warning bullet only, **by design** |

**There was no rule for "the location is fine to visit but not to live in."** `hybrid.acceptable_locations`
is the only place the north/south distinction was written down, and a remote job never reaches it.

**The rater filled the vacuum, inconsistently.** It capped BENCH/Munich to 3 in prose — *"relocation
required: Munich, Germany contradicts the remote label … capping the rating at 3"* — and left
Finom/Berlin and Flip/Stuttgart at 5 on identical facts. That is the hybrid-cap and Valtech lesson
for the third time: **a structural fact left to the model's discretion is decided differently every
time it comes up.**

**Three design traps, each of which was walked into before being caught.**

1. **"Non-English AND not on the acceptable list" is the wrong rule, and it passes casual
   inspection.** It gets Germany and Spain right — for the wrong reason — and gets **Quebec wrong**:
   `acceptable_locations` lists `vancouver` and `british columbia`, not `canada`, so a
   French-language remote role in Montreal would have been rejected by a rule whose own regression
   test (`test_valtech_regression_french_posting_is_capped_and_warned`) demands it only warn. The
   fix is that the gate is **purely geographic and reads no language field at all**. Spain is
   acceptable because Spain is a place you would be, not because of anything about Spanish.
2. **A bare location is not a residency requirement.** "Germany (Remote)" means the role is anchored
   in Germany while the holder could live anywhere in the EU. So `location` and an explicit
   `relocation` statement are judged separately: the first is a preference about where roles may be
   anchored, the second is the JD asserting a categorical requirement. `relocation: European Union`
   names no country and must not reject — it did in an early draft.
3. **Hand-enumerating "northern Europe" is guesswork that rots.** Is Slovenia Adriatic? Romania
   Mediterranean? Bare "France" — Paris or Nice? So the residue goes to a **cached** classifier, and
   the live backtest immediately justified it: it split `Aix-en-Provence → southern_europe` (keep)
   from `Hauts-de-France → western_europe` (reject) with no city list at all, which no list I wrote
   by hand had done.

**Why an LLM call inside a hard rule is not the thing this project forbids** — stated here because the
next reader will see it and reasonably assume a bug. The model answers a question about a *place*,
never about the job; the **cache** makes it deterministic (two Berlin jobs cannot get different
answers, and a second backtest run was byte-identical and free); and the **policy stays in code**.
The precedent is `company_blacklist_reason()`. It fails **open**, opposite to the blacklist, because
a classifier call happens only when nothing the user wrote down matched.

**Separately: the parse failures were misdiagnosed from their own error message.** Five jobs hit
`eval_error` on Sep 03 and the message began `'```json\n{...`, so the obvious reading was that a
fenced code block broke the parser. **It never did** — `extract_json_object` scans to the first `{`
and calls `raw_decode`, which ignores everything around it; a test now pins that. The real causes:

- **Truncation.** `finish_reason: "length"` with `completion_tokens: 3000` — exactly
  `LLM_JSON_MAX_TOKENS` — appears **13 times** across Sep 01-03, and survived the glm-5.2 →
  glm-5.3-flash change. `chat_openrouter` inspected `finish_reason` only when `content` was null, so
  a truncated *non-empty* body went to the parser as if complete.
- **One genuinely malformed body**: Archer's `"reasoning":Exceptional domain match` — the opening
  quote of the value simply missing. No pre-processing fixes that, and none should try; repairing
  truncated JSON would fabricate a rating from a partial one.

The error printed `text[:300]`. **The head is where the fence is; the fault is always in the tail.**
Same shape as 2026-08-24's `api_error_status`: the field naming the cause was in the payload and
discarded.

**And the cap nobody chose cost double.** No call site ever passed `max_tokens` — all eight inherited
the 3000 default — against models allowing **131,072**, with the MCP tool declaring it optional and
applying no clamp, and the two agentic call sites already sending nothing. A truncated company-match
(a true/false question) paid $0.0068 for 3000 reasoning tokens, returned `null`, then paid again for
the Anthropic fallback. It is now not sent at all.

**An `eval_error` is terminal and silent.** The job was written to `processed_jobs/` back in Stage 1b,
so the next run returns `already_processed` and it is never rated again. A Netherlands posting the
model had scored **5** was lost permanently, behind one WARNING. `rate_with_openrouter` now retries
once on truncation.
