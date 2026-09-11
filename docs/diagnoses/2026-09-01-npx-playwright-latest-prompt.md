# 2026-09-01 — `npx @playwright/mcp@latest` stopped a run at an interactive prompt

`uv run job-search -n` halted during browser startup:

```
Need to install the following packages:
@playwright/mcp@0.0.80
Ok to proceed? (y)
```

Both launch sites used `npx @playwright/mcp@latest`, which npm re-resolves **on every run**. npx
had cached the tree as `"@playwright/mcp": "^0.0.79"` under the key `@playwright/mcp@latest`
(`~/.npm/_npx/9833c18b2d85bc59/package.json`), and **`^0.0.79` on a `0.0.x` version means exactly
`0.0.79`** — caret does not widen the patch range below `0.1.0`. Upstream published `0.0.80`, the
cached tree stopped satisfying `latest`, and npx blocked on stdin asking to install. That is the
startup path a scheduled run uses, where nothing can answer.

**The prompt is the symptom; the silent upgrade is the defect.** This package is the tool surface
Stage 1b drives a **real logged-in LinkedIn account** through, and the `--snapshot-mode` analysis
in the [2026-08-21 entry](2026-08-21-stage-1b-cost.md) is verified against **one specific bundle** — which 18 action tools
call `setIncludeSnapshot()`. Under `@latest` that could stop being true with no commit, no diff,
and no test failure. It is the same shape as the `mcp>=1.29` → 2.0.0 re-resolution in that same
entry: a dependency that moves underneath a documented measurement.

**The fix is a pin plus a prompt, not a pin alone.** `PLAYWRIGHT_MCP_VERSION` in `config.py` is a
concrete version and `--yes` is passed to npx — safe *because* the version is exact, since npx can
then install nothing but the pin, so a cold cache installs instead of blocking. Pinning alone,
though, just trades a loud stop for a silent freeze at a version nobody revisits. So
`check_playwright_mcp_version()` queries the registry at startup and
`prompt_playwright_mcp_upgrade()` acts on the answer.

**Three things about where and how it asks, each of which is the point:**

1. **It runs before `acquire_run_lock()`** — before the processed-job load, before PDF ingest, and
   long before `start_playwright_server()` (`agent.py`) or the interactive stdio server the SDK
   spawns. Choosing to upgrade therefore unwinds nothing: no lock held, no browser started. A
   check that ran after the lock would have to release it, and one that ran after npx would be too
   late to prevent the very prompt it exists to replace.
2. **It asks only in interactive mode AND on a TTY — both conditions.** The first version gated on
   `sys.stdin.isatty()` alone, reasoning that `-n` is run by hand as well as by cron so keying on
   the mode would deny a person the choice. **That was wrong, and it reproduced the original bug.**
   `-n` from a terminal has `isatty() == True`, so a hand-run `-n` stopped dead on a question —
   a mechanism built to stop the run blocking on a prompt, blocking the run on a prompt. `-n`
   means *do not ask me things*, whoever started it. Anywhere it does not ask it logs one WARNING,
   raises a `playwright_mcp_outdated` health alert, and continues on the pin. `EOFError` and
   `KeyboardInterrupt` at the question are caught and treated as "continue", so Ctrl-D or Ctrl-C
   cannot traceback out of startup either.
3. **The registry call fails open on everything** — offline, timeout, malformed body, a version
   string that will not parse. It is queried over plain HTTPS rather than via `npm view`, which
   shells out through the npm cache and fails for unrelated reasons (on this machine, currently,
   `cache folder contains root-owned files`).

`--no-version-check` skips the call outright for a scheduled run. Measured **~0.25s** against a healthy registry (the `/latest` endpoint returns 3KB), not the timeout value — and note the ceiling is not 3s either: `requests` applies a scalar `timeout` to the connect **and** the read, so the worst case is ~6s, with DNS resolution bounded by neither.

**The alert nearly went nowhere.** `run_non_interactive()` resets `tools_module._ui_alerts = []` on
entry, and the check appends *before* that runs — so the first version of this raised an alert that
was then discarded, the most-repeated bug in these diagnoses. Startup alerts now accumulate in
`_startup_ui_alerts` and the reset seeds from it.

**Version comparison is numeric, deliberately.** `'0.0.9' > '0.0.79'` is true as strings and false
as versions; a parametrised test pins that case.
