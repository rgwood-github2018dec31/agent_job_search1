'''Personal job-search preferences, loaded from a gitignored file in ``run_dir/``.

Everything in this module is about *the person running the agent* — where they live, which
regions they can work in, what languages they speak, which titles they want. None of that
belongs in tracked source (see the Knowledge locality requirement in CLAUDE.md), so the values
live in ``run_dir/preferences.yaml`` alongside the resume and JOB_REQUIREMENTS.md.

``preferences.example.yaml`` in the project root documents every field. The defaults below are
deliberately neutral — an unconfigured checkout searches without a location filter and applies no
work-authorization, language, or education gate, rather than silently inheriting someone else's
situation.
'''

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from agentic_job_search.config import COMPANY_BLACKLIST_EXPIRY_DAYS

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent
RUN_DIR = PROJECT_DIR / 'run_dir'
PREFERENCES_PATH = RUN_DIR / 'preferences.yaml'
EXAMPLE_PREFERENCES_PATH = PROJECT_DIR / 'preferences.example.yaml'

# Neutral defaults: no personal situation encoded. Every gate that could reject a job is off.
DEFAULT_PREFERENCES: dict[str, Any] = {
    # Where saved job-posting PDFs land before categorization and ingest. Not a gate and not a
    # personal fact, so this defaults to a real path rather than a neutral empty value.
    'save_dir': '~/Downloads',
    'search_regions': [],           # [{'name':…, 'linkedin_location':…, 'geo_id':…}]; empty = unfiltered search
    'sponsorship_required_in': [],  # lowercase location substrings where the user needs visa sponsorship
    'languages': [],                # languages the user works in; empty = no language gate
    # Ceiling for a posting WRITTEN IN a language outside `languages`. Inert while `languages`
    # is empty, like every other language rule.
    'foreign_language_rating_cap': 3,
    'reject_required_degrees': [],  # e.g. ['master', 'phd']; empty = no education gate
    'hybrid': {
        'rating_cap': 3,
        'acceptable_locations': [],  # empty = every hybrid/on-site job is capped
    },
    'titles': {
        'prefer': '',    # free text appended to the query-generation prompt
        'exclude': [],   # title words that must never appear in a generated query
    },
    'relocation_note': '',  # free text for the evaluator on acceptable relocation
    'companies': {
        # [{'name': ..., 'reason': ..., 'added': 'YYYY-MM-DD'}]; empty = no company is blocked
        'blacklist': [],
    },
}

_cache: dict[str, Any] | None = None

# Names already warned about this process, so an expired or undated entry is reported once per
# run rather than once per candidate job.
_warned_blacklist_entries: set[str] = set()


def load_preferences(force_reload: bool = False) -> dict[str, Any]:
    '''Read run_dir/preferences.yaml over the neutral defaults, cached for the process.

    A missing file is not an error — the agent still runs, just without any personal gate. It is
    logged at WARNING because an unconfigured run silently rates very differently.
    '''
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    prefs = _deep_merge(DEFAULT_PREFERENCES, {})
    if PREFERENCES_PATH.exists():
        try:
            loaded = yaml.safe_load(PREFERENCES_PATH.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError as ex:
            raise ValueError(
                f'Could not parse preferences file {PREFERENCES_PATH}: {ex}. '
                f'Fix the YAML or delete the file to fall back to neutral defaults.'
            ) from ex
        if not isinstance(loaded, dict):
            raise ValueError(
                f'Preferences file {PREFERENCES_PATH} must contain a YAML mapping at the top level, '
                f'got {type(loaded).__name__}. See {EXAMPLE_PREFERENCES_PATH.name} for the format.'
            )
        prefs = _deep_merge(prefs, loaded)
        logger.info(f'Loaded preferences from {PREFERENCES_PATH}')
    else:
        logger.warning(
            f'No preferences file at {PREFERENCES_PATH} — running with neutral defaults '
            f'(no location filter, no work-authorization/language/education gates). '
            f'Copy {EXAMPLE_PREFERENCES_PATH.name} there and fill it in.'
        )

    _cache = prefs
    return prefs


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    '''Recursive dict merge; override wins. Nested dicts merge, scalars and lists replace.'''
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_dir() -> Path:
    '''Directory where saved job-posting PDFs land, before categorization and ingest.

    This is the SOURCE only — an external, user-controlled folder the agent moves files out of.
    It is never the applied-jobs corpus, which lives at tools_generic.APPLIED_JOBS_DIR
    (run_dir/applied_jobs/) and is deliberately not configurable.
    '''
    return Path(str(load_preferences()['save_dir'])).expanduser()


def search_regions() -> list[dict[str, str]]:
    '''Regions to search. Empty means one unfiltered search.

    Each entry is {'name', 'linkedin_location', 'geo_id'}:
      - `linkedin_location` is the text TYPED INTO the location chip's autocomplete, which is how
        the region is actually applied. This is deliberate: filters are set by clicking, never by
        crafting `geoId=` URLs, because no human assembles filter parameters by hand.
      - `geo_id` is optional and is used ONLY to verify afterwards that the click landed — the id
        LinkedIn puts in the URL once the suggestion is picked (e.g. European Union = 91000000).
        Before 2026-08-18 the region was typed into `keywords`, which LinkedIn silently ignored,
        so every EU search returned the account's home metro for days without anything noticing.
    '''
    return list(load_preferences()['search_regions'])


def sponsorship_required_in() -> tuple[str, ...]:
    '''Lowercase location substrings where the user would need visa sponsorship.'''
    return tuple(str(loc).lower() for loc in load_preferences()['sponsorship_required_in'])


def languages() -> tuple[str, ...]:
    '''Lowercase languages the user works in; a posting requiring any other language is rejected.'''
    return tuple(str(lang).lower() for lang in load_preferences()['languages'])


def foreign_language_rating_cap() -> int:
    '''Highest rating a posting written in a language outside `languages` can receive.'''
    return int(load_preferences()['foreign_language_rating_cap'])


def rejected_degrees() -> tuple[str, ...]:
    '''Lowercase degree levels ('master', 'phd') that hard-reject when a posting requires them.'''
    return tuple(str(degree).lower() for degree in load_preferences()['reject_required_degrees'])


def hybrid_rating_cap() -> int:
    return int(load_preferences()['hybrid']['rating_cap'])


def hybrid_acceptable_locations() -> tuple[str, ...]:
    '''Lowercase location substrings where a hybrid/on-site role is acceptable.'''
    return tuple(str(loc).lower() for loc in load_preferences()['hybrid']['acceptable_locations'])


def preferred_titles_note() -> str:
    return str(load_preferences()['titles']['prefer'] or '')


def excluded_title_words() -> tuple[str, ...]:
    return tuple(str(word) for word in load_preferences()['titles']['exclude'])


def relocation_note() -> str:
    return str(load_preferences()['relocation_note'] or '')


def _parse_added_date(raw: Any) -> date | None:
    '''Parse a blacklist entry's `added` field. Returns None if absent or unparseable.'''
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip(), '%Y-%m-%d').date()
    except ValueError:
        return None


def _blacklist_entries() -> list[tuple[str, str, date | None]]:
    '''Normalize the raw blacklist into (name, reason, added_date) triples.

    Accepts either a mapping per entry or a bare string (a name with no reason and no date), so a
    hand-edited list of plain names still works. Names are returned **as written** — the match is
    case-sensitive, unlike every other preference accessor here.
    '''
    entries: list[tuple[str, str, date | None]] = []
    for raw in load_preferences()['companies']['blacklist']:
        if isinstance(raw, dict):
            name = str(raw.get('name') or '').strip()
            reason = str(raw.get('reason') or '').strip()
            added = _parse_added_date(raw.get('added'))
        else:
            name, reason, added = str(raw or '').strip(), '', None
        if name:
            entries.append((name, reason, added))
    return entries


def blacklisted_companies() -> tuple[tuple[str, str], ...]:
    '''(name, reason) pairs for unexpired blacklist entries. Empty means no blacklist gate.

    An entry with a missing or unparseable `added` date stays **active** and is warned about. An
    auto-reject is unappealable, so a typo in a date must not silently switch the rule off; the
    warning is how the typo gets noticed.
    '''
    active: list[tuple[str, str]] = []
    for name, reason, added in _blacklist_entries():
        if added is None:
            if name not in _warned_blacklist_entries:
                _warned_blacklist_entries.add(name)
                logger.warning(
                    f'Blacklist entry {name!r} has no valid `added` date (expected YYYY-MM-DD) — '
                    f'treating it as active. Add a date in {PREFERENCES_PATH} so it can expire.'
                )
            active.append((name, reason))
            continue
        age_days = (date.today() - added).days
        if age_days > COMPANY_BLACKLIST_EXPIRY_DAYS:
            if name not in _warned_blacklist_entries:
                _warned_blacklist_entries.add(name)
                logger.warning(
                    f'Blacklist entry {name!r} EXPIRED (added {added.isoformat()}, '
                    f'{age_days} days ago > {COMPANY_BLACKLIST_EXPIRY_DAYS}) — it no longer rejects '
                    f'anything. Bump its `added` date in {PREFERENCES_PATH} to renew it, or delete it.'
                )
            continue
        active.append((name, reason))
    return tuple(active)


def expired_blacklist_entries() -> tuple[tuple[str, str], ...]:
    '''(name, added-date) pairs past COMPANY_BLACKLIST_EXPIRY_DAYS. Reported, never enforced.'''
    return tuple(
        (name, added.isoformat())
        for name, _reason, added in _blacklist_entries()
        if added is not None and (date.today() - added).days > COMPANY_BLACKLIST_EXPIRY_DAYS
    )
