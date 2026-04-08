# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a personal autonomous job search agent built on the Claude Agent SDK. It:
1. Reads the user's resume
2. Periodically searches LinkedIn for relevant job postings
3. Asks the user whether specific jobs are suitable, building up a preference profile over time
4. Autonomously applies to jobs via email (with cover letter + resume), web forms, or LinkedIn

## Commands

```bash
uv sync                  # Install dependencies
uv sync --group test     # Install test dependencies
python main.py           # Run the agent

# Tests
uv run pytest                          # Run all tests (skips live tests if no API key)
uv run pytest -m "not live_agent_claude"  # Unit tests only
uv run pytest -m live_agent_claude     # Live agent tests (requires ANTHROPIC_API_KEY)
```

## Architecture

The project uses the **[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)** (`claude-agent-sdk>=0.1.56`) as its foundation:

- **MCP (Model Context Protocol)** — enables tool use for web browsing, LinkedIn interaction, and email sending
- **anyio** — async runtime for periodic background tasks (e.g., scheduled LinkedIn searches)

Core workflow:
1. Resume ingestion — agent reads and understands the user's resume
2. Periodic LinkedIn search — agent scrapes/searches for matching job listings
3. User preference loop — agent surfaces jobs to the user and records approval/rejection to refine its model of desired roles
4. Application — once a job is approved, agent drafts cover letter and applies via email, web form, or LinkedIn

The entry point is `main.py`. All agent logic will live here or in modules imported by it.
