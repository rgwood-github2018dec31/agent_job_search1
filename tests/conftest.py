import os

import pytest

import agentic_job_search.preferences as preferences
from agentic_job_search.agent import load_env
load_env()

# Fixed preferences for the whole suite. Tests must never read the real run_dir/preferences.yaml —
# that file is personal and machine-local, so depending on it would make results differ per
# developer and silently change when someone edits their own job-search settings.
TEST_PREFERENCES = {
    'search_regions': [
        {'name': 'Testland', 'linkedin_location': 'Testland'},
        {'name': 'Test Union', 'linkedin_location': 'Test Union'},
    ],
    'sponsorship_required_in': ['united states'],
    'languages': ['english'],
    'reject_required_degrees': ['master', 'phd'],
    'hybrid': {
        'rating_cap': 3,
        'acceptable_locations': ['testville', 'exampleton'],
    },
    'titles': {
        'prefer': 'Prefer INDIVIDUAL CONTRIBUTOR titles (Principal / Staff / Lead / Senior).',
        'exclude': ['Manager', 'Head of', 'Director', 'VP'],
    },
    'relocation_note': 'Relocating within the EU is acceptable for a remote role.',
}


_REAL_LOAD_PREFERENCES = preferences.load_preferences


@pytest.fixture
def real_load_preferences():
    """The unpatched loader, for tests that exercise file loading itself."""
    return _REAL_LOAD_PREFERENCES


@pytest.fixture(autouse=True)
def _fixed_preferences(monkeypatch):
    """Pin preferences for every test, and block reads of the real preferences file."""
    merged = preferences._deep_merge(preferences.DEFAULT_PREFERENCES, TEST_PREFERENCES)
    monkeypatch.setattr(preferences, '_cache', merged)
    monkeypatch.setattr(
        preferences, 'load_preferences',
        lambda force_reload=False: merged,
    )
    yield


def _claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("live_agent_claude") and not _claude_available():
        pytest.skip("Skipping: no ANTHROPIC_API_KEY — Claude not available")
