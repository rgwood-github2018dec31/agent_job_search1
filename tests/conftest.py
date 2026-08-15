import os
from datetime import date, timedelta

import pytest

import agentic_job_search.preferences as preferences
import agentic_job_search.tools_generic as tools
from agentic_job_search import agent
from agentic_job_search.config import COMPANY_BLACKLIST_EXPIRY_DAYS
from agentic_job_search.agent import load_env
load_env()

# Fixed preferences for the whole suite. Tests must never read the real run_dir/preferences.yaml —
# that file is personal and machine-local, so depending on it would make results differ per
# developer and silently change when someone edits their own job-search settings.
TEST_PREFERENCES = {
    # Deliberately nonexistent: every test passes an explicit directory, and this makes sure a
    # test that forgets to can never glob or rename PDFs in the developer's own ~/Downloads.
    'save_dir': '/nonexistent/test-save-dir',
    'search_regions': [
        {'name': 'Testland', 'linkedin_location': 'Testland'},
        {'name': 'Test Union', 'linkedin_location': 'Test Union'},
    ],
    'sponsorship_required_in': ['united states'],
    'languages': ['english'],
    'foreign_language_rating_cap': 3,
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
    'companies': {
        # Fake companies only. The `added` date is recomputed per run so the active entry never
        # ages past COMPANY_BLACKLIST_EXPIRY_DAYS and starts silently passing tests.
        'blacklist': [
            {'name': 'Blocked Corp', 'reason': 'test entry', 'added': date.today().isoformat()},
            {
                'name': 'Lapsed Corp', 'reason': 'expired test entry',
                'added': (date.today() - timedelta(days=COMPANY_BLACKLIST_EXPIRY_DAYS + 1)).isoformat(),
            },
        ],
    },
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
    # Fresh per test: the warn-once set would otherwise make a warning assertion depend on
    # whether an earlier test happened to touch the same blacklist entry.
    monkeypatch.setattr(preferences, '_warned_blacklist_entries', set())
    yield


@pytest.fixture(autouse=True)
def _no_human_pacing(monkeypatch):
    """Zero the scraper's human-emulation delays for every test.

    The real values are tens of seconds per query — deliberately, since they protect a live
    LinkedIn account — which would make the suite take minutes and look like a hang. Zeroing
    them here rather than in each test means a newly added pacing pause can never silently
    stall the suite.
    """
    for name in (
        'SCRAPER_INTER_SEARCH_DELAY_SECONDS',
        'SCRAPER_INTER_QUERY_DELAY_SECONDS',
    ):
        monkeypatch.setattr(agent, name, (0.0, 0.0))
    yield


@pytest.fixture(autouse=True)
def _reset_scraper_run_state(monkeypatch):
    """Clear Stage 1b module state that run_scraper writes, so tests cannot leak into each other.

    run_scraper sets tools._current_query as it goes; a test that ran it left the last query set,
    and a later queue_candidate test then attributed its candidate to that stale value.
    """
    monkeypatch.setattr(tools, '_current_query', None)
    yield


def _claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("live_agent_claude") and not _claude_available():
        pytest.skip("Skipping: no ANTHROPIC_API_KEY — Claude not available")
