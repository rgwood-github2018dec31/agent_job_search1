import functools
import re
import os
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

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
    # Fictional places, so no test can depend on the real classifier or on real geography.
    # 'blockedland' is on the deny list; 'testville'/'exampleton' are exempt; anything else
    # reaches the classifier, which tests stub via the `stub_location_classifier` fixture.
    'locations': {
        'exclude': ['blockedland'],
        'reject_regions': ['northern_europe', 'western_europe', 'eastern_europe'],
    },
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


# Fictional-but-shaped geography for the location classifier. Unit tests must never reach the
# real MCP tool server: it costs money, needs a network, and would make the suite's verdict depend
# on a third party. Note the classifier fails OPEN, so an unstubbed test would have *passed* while
# silently making a live call on every job -- which is exactly the "a skip that never becomes a
# pass" shape CLAUDE.md warns about, one layer down.
# Fictional-but-shaped geography for the location classifier. Unit tests must never reach the
# real MCP tool server: it costs money, needs a network, and would make the suite's verdict depend
# on a third party. Note the classifier fails OPEN, so an unstubbed test would have *passed* while
# silently making a live call on every job -- the "a skip that never becomes a pass" shape one
# layer down.
#
# Each entry maps a name to (country, region, local language). Cities resolve to their country,
# because the real classifier returns COUNTRIES -- a stub that returned 'berlin' as a country
# would let a test pass against behaviour the production path never produces.
FAKE_GEOGRAPHY = {
    'canada': ('canada', 'north_america', 'english'),
    'quebec': ('canada', 'north_america', 'french'),
    'montreal': ('canada', 'north_america', 'french'),
    'united states': ('united states', 'north_america', 'english'),
    'germany': ('germany', 'western_europe', 'german'),
    'berlin': ('germany', 'western_europe', 'german'),
    'stuttgart': ('germany', 'western_europe', 'german'),
    'munich': ('germany', 'western_europe', 'german'),
    'netherlands': ('netherlands', 'western_europe', 'dutch'),
    'amsterdam': ('netherlands', 'western_europe', 'dutch'),
    'ireland': ('ireland', 'western_europe', 'english'),
    'dublin': ('ireland', 'western_europe', 'english'),
    'united kingdom': ('united kingdom', 'western_europe', 'english'),
    'uk': ('united kingdom', 'western_europe', 'english'),
    'france': ('france', 'western_europe', 'french'),
    'paris': ('france', 'western_europe', 'french'),
    'nice': ('france', 'southern_europe', 'french'),
    'toulouse': ('france', 'southern_europe', 'french'),
    'spain': ('spain', 'southern_europe', 'spanish'),
    'barcelona': ('spain', 'southern_europe', 'spanish'),
    'portugal': ('portugal', 'southern_europe', 'portuguese'),
    'greece': ('greece', 'southern_europe', 'greek'),
    'poland': ('poland', 'eastern_europe', 'polish'),
    'czechia': ('czechia', 'eastern_europe', 'czech'),
    'prague': ('czechia', 'eastern_europe', 'czech'),
    'sweden': ('sweden', 'northern_europe', 'swedish'),
}

# Word boundaries, because 'nice' is inside 'Venice' and 'uk' is inside almost everything. The
# production classifier is an LLM and has no such problem; the stub must not invent one.
_GEO_RE = re.compile(
    r'\b(' + '|'.join(sorted((re.escape(n) for n in FAKE_GEOGRAPHY), key=len, reverse=True)) + r')\b'
)


def fake_classify(text):
    """Every COUNTRY named in `text`, first mention first, deduped. Unmatched text names none."""
    haystack = ' '.join(str(text or '').split()).lower()
    countries, regions, language = [], [], ''
    for match in _GEO_RE.finditer(haystack):
        country, region, local_language = FAKE_GEOGRAPHY[match.group(1)]
        if not language:
            language = local_language
        if country in countries:
            continue
        countries.append(country)
        regions.append(region)
    return {'countries': countries, 'regions': regions, 'local_language': language, 'source': 'stub'}


@pytest.fixture(autouse=True)
def stub_location_classifier(monkeypatch):
    """Autouse: no unit test may reach the real classifier. Returns the call log for assertions.

    Patches the binding the production path actually uses (agent imported the name at module
    load), and separately blocks the network underneath the real classifier so a test that calls
    `location.classify_location` directly still cannot escape. Tests exercising the real
    classifier monkeypatch `chat_openrouter` themselves, which wins over this.
    """
    calls = []

    async def _fake(text):
        calls.append(text)
        return fake_classify(text)

    async def _no_network(*args, **kwargs):
        raise AssertionError('a unit test tried to reach the OpenRouter MCP server')

    monkeypatch.setattr(agent, 'classify_location', _fake)
    monkeypatch.setattr('agentic_job_search.location.chat_openrouter', _no_network)
    return calls


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


@functools.lru_cache(maxsize=1)
def _claude_available() -> bool:
    """Whether a live Claude call can actually be made.

    ANTHROPIC_API_KEY alone is the wrong question: the Agent SDK authenticates through the
    logged-in Claude Code CLI, so a developer machine with no API key at all can still make live
    calls — and gating on the env var silently skipped every live test there. Probe the CLI the
    SDK would actually spawn instead, and fall back to the env var for API-key setups (CI).
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        import claude_agent_sdk

        bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        cli = str(bundled) if bundled.exists() else shutil.which("claude")
    except Exception:
        return False
    if not cli:
        return False
    # `--version` does not authenticate, so confirm the CLI can reach the API with a trivial
    # prompt. Cached, so this costs one short call per pytest session at most.
    try:
        completed = subprocess.run(
            [str(cli), "-p", "hi", "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return False
    return completed.returncode == 0


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("live_agent_claude") and not _claude_available():
        pytest.skip("Skipping: Claude not available (no API key and CLI cannot reach the API)")
