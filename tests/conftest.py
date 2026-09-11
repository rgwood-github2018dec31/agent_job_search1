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
from agentic_job_search import location
from agentic_job_search import location_review
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
    'sponsorship_required_in': ['United States'],
    'languages': ['english'],
    'foreign_language_rating_cap': 3,
    'reject_required_degrees': ['master', 'phd'],
    # Fictional places, so no test can depend on the real classifier or on real geography.
    # 'Blockedland' is on the deny list; 'Testville'/'Exampleton' are exempt; anything else
    # reaches the classifier, which tests stub via the `stub_location_classifier` fixture.
    # PROPER NAMES, like the real config: these lists are matched case-sensitively, so a lowercase
    # fixture would test a matching rule production does not have.
    # Three lists plus the commute axis. Fictional places wherever the test is about the
    # MECHANISM; real countries only where a test is about geography the gate must actually know.
    'locations': {
        'would_live_here': ['Testville', 'Exampleton', 'Spain', 'Portugal', 'Italy', 'Greece', 'Canada'],
        'would_not_live_here': [
            'Blockedland', 'Germany', 'Ireland', 'Poland', 'Sweden', 'Romania', 'France',
            'Netherlands', 'United Kingdom', 'Serbia', 'Norway', 'Switzerland', 'Bulgaria',
            'Czechia', 'Lithuania', 'Austria', 'Estonia', 'Finland',
        ],
        'not_yet_bucketed': [],
        'would_commute_here': ['Testville', 'Exampleton', 'Barcelona'],
    },
    'hybrid': {
        'rating_cap': 3,
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
# pass" shape docs/requirements.md (Tests must be able to fail) warns about, one layer down.
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
    'Canada': ('Canada', 'north_america', 'english'),
    'Quebec': ('Canada', 'north_america', 'french'),
    'Montreal': ('Canada', 'north_america', 'french'),
    'United States': ('United States', 'north_america', 'english'),
    'Germany': ('Germany', 'western_europe', 'german'),
    'Berlin': ('Germany', 'western_europe', 'german'),
    'Stuttgart': ('Germany', 'western_europe', 'german'),
    'Munich': ('Germany', 'western_europe', 'german'),
    'Netherlands': ('Netherlands', 'western_europe', 'dutch'),
    'Amsterdam': ('Netherlands', 'western_europe', 'dutch'),
    'Ireland': ('Ireland', 'western_europe', 'english'),
    'Dublin': ('Ireland', 'western_europe', 'english'),
    'United Kingdom': ('United Kingdom', 'western_europe', 'english'),
    'UK': ('United Kingdom', 'western_europe', 'english'),
    'France': ('France', 'western_europe', 'french'),
    'Paris': ('France', 'western_europe', 'french'),
    'Nice': ('France', 'southern_europe', 'french'),
    'Toulouse': ('France', 'southern_europe', 'french'),
    'Spain': ('Spain', 'southern_europe', 'spanish'),
    'Barcelona': ('Spain', 'southern_europe', 'spanish'),
    'Portugal': ('Portugal', 'southern_europe', 'portuguese'),
    'Greece': ('Greece', 'southern_europe', 'greek'),
    'Poland': ('Poland', 'eastern_europe', 'polish'),
    'Czechia': ('Czechia', 'eastern_europe', 'czech'),
    'Prague': ('Czechia', 'eastern_europe', 'czech'),
    'Sweden': ('Sweden', 'northern_europe', 'swedish'),
    # An exonym pair: the posting says one, the list may say the other.
    'Sevilla': ('Spain', 'southern_europe', 'spanish'),
    'Seville': ('Spain', 'southern_europe', 'spanish'),
    'Torino': ('Italy', 'southern_europe', 'italian'),
    # A rejected-region place whose exonym is the only thing that can rescue it.
    'München': ('Germany', 'western_europe', 'german'),
    'Bulgaria': ('Bulgaria', 'eastern_europe', 'bulgarian'),
    'Lithuania': ('Lithuania', 'eastern_europe', 'lithuanian'),
    'Romania': ('Romania', 'eastern_europe', 'romanian'),
    'Bucharest': ('Romania', 'eastern_europe', 'romanian'),
    'Italy': ('Italy', 'southern_europe', 'italian'),
    'Rome': ('Italy', 'southern_europe', 'italian'),
    # Non-EU countries in rejected regions. These are the four the "a silent remote posting in an
    # EU country is assumed remote-from-the-EU" default deliberately does NOT cover, so the stub
    # has to be able to name them or the counterpart test proves nothing.
    'Serbia': ('Serbia', 'eastern_europe', 'serbian'),
    'Belgrade': ('Serbia', 'eastern_europe', 'serbian'),
    'Norway': ('Norway', 'northern_europe', 'norwegian'),
    'Oslo': ('Norway', 'northern_europe', 'norwegian'),
    'Switzerland': ('Switzerland', 'western_europe', 'german'),
    'Zurich': ('Switzerland', 'western_europe', 'german'),
}

# Word boundaries, because 'nice' is inside 'Venice' and 'uk' is inside almost everything. The
# production classifier is an LLM and has no such problem; the stub must not invent one.
# Case-insensitive so the keys can stay proper names: folding happens at the comparison, and
# `_GEO_KEYS` maps a folded match back to the name as written.
_GEO_RE = re.compile(
    r'\b(' + '|'.join(sorted((re.escape(n) for n in FAKE_GEOGRAPHY), key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)
_GEO_KEYS = {name.casefold(): name for name in FAKE_GEOGRAPHY}


# Other names a place goes by, for the alias tier. The real classifier returns these for every
# location; the stub only needs them where a test turns on an exonym.
FAKE_EXONYMS = {
    'Sevilla': ['Seville', 'Sevilla'],
    'Seville': ['Seville', 'Sevilla'],
    'Torino': ['Turin', 'Torino'],
    'Munich': ['Munich', 'München'],
    'München': ['Munich', 'München'],
}


# Multi-country areas: an offer of one of these is an unrejected option in its own right.
FAKE_BROAD_AREAS = ('European Union', ' EU ', 'EMEA', 'anywhere', 'worldwide', 'across Europe')


def fake_classify(text):
    """Every COUNTRY named in `text`, first mention first, deduped. Unmatched text names none."""
    haystack = ' '.join(str(text or '').split())
    names = [_GEO_KEYS[match.group(1).casefold()] for match in _GEO_RE.finditer(haystack)]
    countries, regions, language = [], [], ''
    for name in names:
        country, region, implied_local_language = FAKE_GEOGRAPHY[name]
        if not language:
            language = implied_local_language
        if country in countries:
            continue
        countries.append(country)
        regions.append(region)
    # 'across europe' qualifies a single anchored country ("Berlin, Germany (Remote across
    # Europe)") rather than offering the whole area, so it only counts when no country is named.
    folded = f' {haystack.casefold()} '
    broad = any(area.casefold() in folded for area in FAKE_BROAD_AREAS[:4]) or (
        not countries and any(area.casefold() in folded for area in FAKE_BROAD_AREAS)
    )
    # `place_names` must be present, or every test exercises a contract production does not have.
    # The stub returns the matched names as written plus their country, which is enough shape for
    # the alias tier: a test wanting a real exonym pair adds it to FAKE_EXONYMS above.
    place_names = []
    for name in names:
        place_names += [name, FAKE_GEOGRAPHY[name][0]]
        place_names += FAKE_EXONYMS.get(name, [])
    return {
        'countries': countries, 'regions': regions, 'broad_area': broad,
        'implied_local_language': language, 'place_names': list(dict.fromkeys(place_names)),
        'source': 'stub',
    }


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
def _reset_countries_seen():
    """The run-scoped country accumulator must not leak between tests.

    Same shape as _reset_scraper_run_state below, and as the _startup_ui_alerts bug docs/requirements.md
    (Tests must be able to fail) calls the most-repeated one in this project: a module global that a run resets, but a test
    does not, so the second test sees the first one's data.
    """
    location.reset_countries_seen()
    yield
    location.reset_countries_seen()


@pytest.fixture(autouse=True)
def _isolated_location_cache(monkeypatch, tmp_path):
    """No test may read the developer's real run_dir/location_cache.yaml.

    Until `region_disagreements()` existed nothing read the cache in aggregate, so the real file
    leaking in was invisible; the moment something did, a unit test started asserting on whatever
    locations happened to be on this machine.
    """
    monkeypatch.setattr(location, 'LOCATION_CACHE_PATH', tmp_path / 'location_cache.yaml')
    monkeypatch.setattr(location, '_cache', {})


@pytest.fixture(autouse=True)
def _isolated_location_recommendations(monkeypatch, tmp_path):
    """No test may read or write the developer's real run_dir/location_recommendations.yaml.

    Autouse for the same reason `save_dir` points at /nonexistent above: a test that forgets would
    silently rewrite a file holding the user's own accept/reject decisions.
    """
    monkeypatch.setattr(
        location_review, 'LOCATION_RECOMMENDATIONS_PATH', tmp_path / 'location_recommendations.yaml'
    )


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
