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
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent
RUN_DIR = PROJECT_DIR / 'run_dir'
PREFERENCES_PATH = RUN_DIR / 'preferences.yaml'
EXAMPLE_PREFERENCES_PATH = PROJECT_DIR / 'preferences.example.yaml'

# Neutral defaults: no personal situation encoded. Every gate that could reject a job is off.
DEFAULT_PREFERENCES: dict[str, Any] = {
    'search_regions': [],           # [{'name': ..., 'linkedin_location': ...}]; empty = unfiltered search
    'sponsorship_required_in': [],  # lowercase location substrings where the user needs visa sponsorship
    'languages': [],                # languages the user works in; empty = no language gate
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
}

_cache: dict[str, Any] | None = None


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


def search_regions() -> list[dict[str, str]]:
    '''Regions to search, each {'name', 'linkedin_location'}. Empty means one unfiltered search.'''
    return list(load_preferences()['search_regions'])


def sponsorship_required_in() -> tuple[str, ...]:
    '''Lowercase location substrings where the user would need visa sponsorship.'''
    return tuple(str(loc).lower() for loc in load_preferences()['sponsorship_required_in'])


def languages() -> tuple[str, ...]:
    '''Lowercase languages the user works in; a posting requiring any other language is rejected.'''
    return tuple(str(lang).lower() for lang in load_preferences()['languages'])


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
