# 2026-08-21 — Stage 1b was 87% of the bill, and CLAUDE.md was part of the reason

Stage 1b reached **86.9% of total run cost** — $48.24 of $55.52 over ten runs — climbing $2.68
(Aug 17) → $7.44 (Aug 18) → **$10.06** (Aug 21) to inspect 300 listings of which only 147 were
distinct. Nothing was broken; nothing reported a problem. Cost was only ever visible as a stage
total, which is why a 68% → 99% drift went unremarked for weeks.

Cost is `Σ_turns(context_size)`, and **~90% of it was cache traffic** (47% read, 42% write, 10%
output; a least-squares fit over 47 requests recovers Haiku list price at the 1h TTL to 1.5%).
Turns were never the constraint — the 260 budget peaked at 163.

Three findings, in increasing order of embarrassment:

1. **The scraper's 40K-token floor was almost entirely not about scraping.** `scraper_options`
   left `setting_sources`, `strict_mcp_config` and `skills` at their defaults, and the SDK
   docstring is explicit: *"When `None`, all sources are loaded… Must include `project` to load
   CLAUDE.md files."* **Measured with `get_context_usage()` on a bare options object declaring no
   MCP servers at all: 45,523 tokens → 358 with the three set.** The 45,165-token difference is
   `Memory files` **31,384** (CLAUDE.md) and `MCP tools` **13,781** (nine unrelated global servers
   — telegram, rag-local, yt-dl, image-gen…), re-read on all 776 turns. The cost history tracks
   CLAUDE.md's own growth: 8.7KB in July at $0.50/run → 67KB now at $10/run. **CLAUDE.md was
   materially responsible for the bill it documents.** Fixed at every `ClaudeAgentOptions` site
   in `agent.py`; moving Stage 1b off the Agent SDK removes it there as well. **That "in
   `agent.py`" was load-bearing in the worst way** — it was literally true and read as complete,
   and the four sites in `tools_generic.py` stayed unfixed for three more months (see the
   [2026-08-24 entry](2026-08-24-error-result-success.md)). A prose claim scoped to one file cannot police a second one; the AST guard
   `test_every_claude_agent_options_site_limits_context` now checks every site in the package, so
   the next one someone adds fails a test rather than quietly costing money.
2. **Every randomised safety pause was buying a full page snapshot.** Verified in the installed
   `@playwright/mcp` bundle (the pin at the time — `PLAYWRIGHT_MCP_VERSION` and its git history
   say exactly which): 18 action tools call
   `setIncludeSnapshot()`, resolving
   `config.snapshot?.mode ?? "full"` — and **`browser_wait_for` is one of them**. The pacing added
   *for* Account safety was the thing being billed. ~20 auto-snapshots × ~8K tokens × 12 searches ≈
   1.92M, against the measured 1.96M cache-write. The arithmetic closes. `--snapshot-mode none`
   fixes it and leaves explicit `browser_snapshot` untouched (it sets `_includeSnapshot =
   "explicit"`, which ignores the config), so the obstacle path is unaffected. **Not yet applied.**
3. **~370 of 776 turns re-emitted data the model already had** — the per-listing
   `check_and_record_job` loop after a harvest that already returned all 25 cards.

**The fix was the model, not the mechanics.** `deepseek-v4-flash` prices identical traffic 9.3x
cheaper ($0.0287 vs $0.4330 on one measured search) because it caches implicitly at ~88% with **no
cache-write fee**. Beware the obvious wrong choices: this project's own `OPENROUTER_MODEL`
(`z-ai/glm-5.2`) costs **more** than Haiku at $0.1932/M cache-read, as do `deepseek-v4-pro` and
`qwen3.8-max` — five of eleven candidates evaluated lose money. There is **no qwen3.8 flash
variant**; the 3.8 line is a premium tier.

**Three capability gaps, all harness-fixable, none about page comprehension.** The flash model
reasoned about the page correctly throughout, but: it could not infer playwright's `target`
convention (burned 14 iterations discovering the bare ref `e1202`, then regressed twice — fixed by
~10 lines of prompt); it **retyped the canonical JS from memory and corrupted it**; and it
**fabricated a page report** when `filename` diverted its evaluate result to disk. The last one
produced the Anti-fabrication requirement, and the fix for the middle one — dedicated tools that
take no page code — is strictly better than what it replaced.

**Two process lessons worth more than the savings.** First, `uv run` silently re-resolved
`mcp>=1.29` to 2.0.0, whose `streamable_http_client` yields two values instead of three, breaking
`mcp_session` — i.e. every OpenRouter and Ollama call — at runtime, mid-run. Now pinned `<2` with
two tests. Second, the harness's own test fixtures omitted playwright's `### Ran Playwright code`
echo, so a parser that scanned to the last `}` ran into the echoed script and returned None on
**every real harvest** while the tests stayed green. **Fixtures that are cleaner than reality are
worse than no fixtures.**
