import os

import pytest

from agentic_job_search.agent import load_env
load_env()


def _claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("live_agent_claude") and not _claude_available():
        pytest.skip("Skipping: no ANTHROPIC_API_KEY — Claude not available")
