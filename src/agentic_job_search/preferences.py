"""Personal job-search preferences, loaded from a gitignored file in ``run_dir/``.

Everything in this module is about *the person running the agent* — where they live, which
regions they can work in, what languages they speak, which titles they want. None of that
belongs in tracked source (see the Knowledge locality requirement in AGENTS.md), so the values
live in ``run_dir/preferences.yaml`` alongside the resume and JOB_REQUIREMENTS.md.

``preferences.example.yaml`` in the project root documents every field. The defaults below are
deliberately neutral — an unconfigured checkout searches without a location filter and applies no
work-authorization, language, or education gate, rather than silently inheriting someone else's
situation.
"""

import logging
import re
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
_ISO_CURRENCY_RE = re.compile(r'[A-Z]{3}')

# Neutral defaults: no personal situation encoded. Every gate that could reject a job is off.
DEFAULT_PREFERENCES: dict[str, Any] = {
    # Where saved job-posting PDFs land before categorization and ingest. Not a gate and not a
    # personal fact, so this defaults to a real path rather than a neutral empty value.
    'save_dir': '~/Downloads',
    'search_regions': [],           # [{'name':…, 'linkedin_location':…, 'geo_id':…}]; empty = unfiltered search
    'sponsorship_required_in': [],  # place names ('United States') where the user needs visa sponsorship; substring-matched, folded at the call site
    'languages': [],                # languages the user works in; empty = no language gate
    # Ceiling for a posting WRITTEN IN a language outside `languages`. Inert while `languages`
    # is empty, like every other language rule.
    'foreign_language_rating_cap': 3,
    'reject_required_degrees': [],  # e.g. ['master', 'phd']; empty = no education gate
    # WHERE A ROLE MAY BE ANCHORED, as three lists plus one for commuting. Purely geographic:
    # this asks "is this a place I would be", never anything about the language spoken there
    # (that is `languages`, a separate fact).
    #
    #   would_live_here      your rule  -> exempts
    #   would_not_live_here  your rule  -> rejects
    #   not_yet_bucketed     the agent's guess -> PASSES, flagged, awaiting your decision
    #   would_commute_here   a different question entirely -> the hybrid/on-site cap only
    #
    # A place named by none of the first three is guessed once, appended to `not_yet_bucketed`,
    # and reported. A guess NEVER rejects. All empty means no geographic gate at all.
    'locations': {
        'would_live_here': [],      # place names (case-sensitive, accent-exact)
        'would_not_live_here': [],
        'not_yet_bucketed': [],     # agent-appended; gates nothing
        'would_commute_here': [],   # empty = every hybrid/on-site job is capped
    },
    'hybrid': {
        'rating_cap': 3,
    },
    'titles': {
        'prefer': '',    # free text appended to the query-generation prompt
        'exclude': [],   # title words that must never appear in a generated query
    },
    'relocation_note': '',  # free text for the evaluator on acceptable relocation
    # Annual base pay a complete salary range must REACH (by its upper bound) to count as on
    # target. Both unset = no code-side comparison; the rater still judges pay on its own.
    'compensation': {
        'base_target': None,  # annual amount, e.g. 150000
        'currency': '',       # ISO 4217 code the target is quoted in, e.g. 'EUR'
    },
    'companies': {
        # [{'name': ..., 'reason': ..., 'added': 'YYYY-MM-DD'}]; empty = no company is blocked
        'blacklist': [],
    },
}

# Old key -> new key. A renamed key must RAISE, never be ignored: `_deep_merge` drops anything it
# does not recognise, so a stale name leaves the list empty and silently switches the rule off --
# a gate that rejects nothing, or a cap that caps everything, with no error anywhere.
LEGACY_KEYS = {
    ('locations', 'exclude'): 'locations.would_not_live_here',
    ('locations', 'reject_regions'): 'locations.would_not_live_in_regions is gone; bucket the '
                                     'countries into locations.would_live_here / would_not_live_here',
    ('hybrid', 'acceptable_locations'): 'locations.would_commute_here',
}


def _reject_legacy_keys(loaded: dict[str, Any]) -> None:
    for (section, key), replacement in LEGACY_KEYS.items():
        if key in (loaded.get(section) or {}):
            raise ValueError(
                f'{PREFERENCES_PATH} uses the old key `{section}.{key}`. Rename it to '
                f'`{replacement}` — see {EXAMPLE_PREFERENCES_PATH.name}. Legacy keys are rejected '
                f'rather than ignored, because an ignored key leaves the list empty and silently '
                f'switches the rule off.'
            )


_cache: dict[str, Any] | None = None

# Names already warned about this process, so an expired or undated entry is reported once per
# run rather than once per candidate job.
_warned_blacklist_entries: set[str] = set()


def load_preferences(force_reload: bool = False) -> dict[str, Any]:
    """Read run_dir/preferences.yaml over the neutral defaults, cached for the process.

    A missing file is not an error — the agent still runs, just without any personal gate. It is
    logged at WARNING because an unconfigured run silently rates very differently.
    """
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
        _reject_legacy_keys(loaded)
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
    """Recursive dict merge; override wins. Nested dicts merge, scalars and lists replace."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_dir() -> Path:
    """Directory where saved job-posting PDFs land, before categorization and ingest.

    This is the SOURCE only — an external, user-controlled folder the agent moves files out of.
    It is never the applied-jobs corpus, which lives at tools_generic.APPLIED_JOBS_DIR
    (run_dir/applied_jobs/) and is deliberately not configurable.
    """
    return Path(str(load_preferences()['save_dir'])).expanduser()


def search_regions() -> list[dict[str, str]]:
    """Regions to search. Empty means one unfiltered search.

    Each entry is {'name', 'linkedin_location', 'geo_id'}:
      - `linkedin_location` is the text TYPED INTO the location chip's autocomplete, which is how
        the region is actually applied. This is deliberate: filters are set by clicking, never by
        crafting `geoId=` URLs, because no human assembles filter parameters by hand.
      - `geo_id` is optional and is used ONLY to verify afterwards that the click landed — the id
        LinkedIn puts in the URL once the suggestion is picked (e.g. European Union = 91000000).
        Before 2026-08-18 the region was typed into `keywords`, which LinkedIn silently ignored,
        so every EU search returned the account's home metro for days without anything noticing.
    """
    return list(load_preferences()['search_regions'])


def _places(key: str) -> tuple[str, ...]:
    """One of the `locations` place lists, as written, blank entries dropped."""
    return tuple(
        str(place).strip() for place in load_preferences()['locations'][key] if str(place).strip()
    )


def sponsorship_required_in() -> tuple[str, ...]:
    """Place names where the user would need visa sponsorship, as written.

    Returned unfolded like the other place-name lists. This rule still compares by SUBSTRING and
    case-insensitively at its call site, unlike the two geographic lists -- but the folding happens
    there, where it is a comparison, rather than here, where it would be a lossy store.
    """
    return tuple(
        str(loc).strip() for loc in load_preferences()['sponsorship_required_in'] if str(loc).strip()
    )


def would_live_here() -> tuple[str, ...]:
    """Places the user has said they would live. Exempts a posting from the geographic gate.

    Returned **as written**, like the company blacklist and unlike the vocabulary lists: these are
    proper names, matched case-sensitively and accent-exactly. Lowercasing them is what made
    `malaga` an entry that could never match `Málaga`.
    """
    return _places('would_live_here')


def would_not_live_here() -> tuple[str, ...]:
    """Places the user has said they would NOT live. Rejects outright, ahead of any guess."""
    return _places('would_not_live_here')


def not_yet_bucketed() -> tuple[str, ...]:
    """Places the agent has guessed about, awaiting the user's decision.

    **Gates nothing.** A posting here is rated and shown normally, carrying a warning that says the
    location was guessed. That is the whole reason the agent is allowed to write this one list and
    no other: it cannot reject a job by appending to it.
    """
    return _places('not_yet_bucketed')


def would_commute_here() -> tuple[str, ...]:
    """Places the user would travel to an OFFICE. A different question from would_live_here.

    Drives the hybrid/on-site rating cap only. Living somewhere does not imply commuting across it
    -- `Spain` on the live list once meant an on-site role in any Spanish town was uncapped.
    """
    return _places('would_commute_here')


def languages() -> tuple[str, ...]:
    """Lowercase languages the user works in; a posting requiring any other language is rejected."""
    return tuple(str(lang).lower() for lang in load_preferences()['languages'])


def foreign_language_rating_cap() -> int:
    """Highest rating a posting written in a language outside `languages` can receive."""
    return int(load_preferences()['foreign_language_rating_cap'])


def rejected_degrees() -> tuple[str, ...]:
    """Lowercase degree levels ('master', 'phd') that hard-reject when a posting requires them."""
    return tuple(str(degree).lower() for degree in load_preferences()['reject_required_degrees'])


def hybrid_rating_cap() -> int:
    return int(load_preferences()['hybrid']['rating_cap'])


def preferred_titles_note() -> str:
    return str(load_preferences()['titles']['prefer'] or '')


def excluded_title_words() -> tuple[str, ...]:
    return tuple(str(word) for word in load_preferences()['titles']['exclude'])


def relocation_note() -> str:
    return str(load_preferences()['relocation_note'] or '')


def salary_target() -> tuple[float, str] | None:
    """(annual base target, ISO currency) — or None when either is unset.

    A malformed value raises rather than reading as unset: an ignored target silently switches the
    below-target comparison off and hands it back to the rater.
    """
    compensation = load_preferences()['compensation']
    raw_target, raw_currency = compensation.get('base_target'), compensation.get('currency')
    if raw_target in (None, '') or not raw_currency:
        return None
    try:
        target = float(raw_target)
    except (TypeError, ValueError) as ex:
        raise ValueError(
            f'{PREFERENCES_PATH}: `compensation.base_target` must be an annual amount such as '
            f'150000, got {raw_target!r} ({type(ex).__name__}: {ex})'
        ) from ex
    currency = str(raw_currency).strip().upper()
    if not _ISO_CURRENCY_RE.fullmatch(currency):
        raise ValueError(
            f'{PREFERENCES_PATH}: `compensation.currency` must be an ISO 4217 code such as EUR, '
            f'got {raw_currency!r}'
        )
    return target, currency


def _parse_added_date(raw: Any) -> date | None:
    """Parse a blacklist entry's `added` field. None if absent; raises ValueError if unparseable."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).strip(), '%Y-%m-%d').date()
    except ValueError as ex:
        raise ValueError(f'`added` value {raw!r} is not a YYYY-MM-DD date: {ex}') from ex


def _blacklist_entries() -> list[tuple[str, str, date | None, str]]:
    """Normalize the raw blacklist into (name, reason, added_date, date_problem) tuples.

    `date_problem` is empty, or says why `added` could not be read; the caller that owns the
    once-per-entry warning reports it.

    Accepts either a mapping per entry or a bare string (a name with no reason and no date), so a
    hand-edited list of plain names still works. Names are returned **as written** — the match is
    case-sensitive, unlike every other preference accessor here.
    """
    entries: list[tuple[str, str, date | None, str]] = []
    for raw in load_preferences()['companies']['blacklist']:
        date_problem = ''
        if isinstance(raw, dict):
            name = str(raw.get('name') or '').strip()
            reason = str(raw.get('reason') or '').strip()
            try:
                added = _parse_added_date(raw.get('added'))
            except ValueError as ex:
                added, date_problem = None, str(ex)
        else:
            name, reason, added = str(raw or '').strip(), '', None
        if name:
            entries.append((name, reason, added, date_problem))
    return entries


def blacklisted_companies() -> tuple[tuple[str, str], ...]:
    """(name, reason) pairs for unexpired blacklist entries. Empty means no blacklist gate.

    An entry with a missing or unparseable `added` date stays **active** and is warned about. An
    auto-reject is unappealable, so a typo in a date must not silently switch the rule off; the
    warning is how the typo gets noticed.
    """
    active: list[tuple[str, str]] = []
    for name, reason, added, date_problem in _blacklist_entries():
        if added is None:
            if name not in _warned_blacklist_entries:
                _warned_blacklist_entries.add(name)
                logger.warning(
                    f'Blacklist entry {name!r} has no valid `added` date '
                    f'({date_problem or "none given"}; expected YYYY-MM-DD) — treating it as '
                    f'active. Add a date in {PREFERENCES_PATH} so it can expire.'
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
    """(name, added-date) pairs past COMPANY_BLACKLIST_EXPIRY_DAYS. Reported, never enforced."""
    return tuple(
        (name, added.isoformat())
        for name, _reason, added, _date_problem in _blacklist_entries()
        if added is not None and (date.today() - added).days > COMPANY_BLACKLIST_EXPIRY_DAYS
    )


# --- the one thing the agent writes into preferences.yaml ---

_BUCKET_KEY = 'not_yet_bucketed'
_BUCKET_HEADER = (
    '  # Agent-appended guesses about places you have not bucketed yet. These gate NOTHING — a\n'
    '  # posting here is still rated and shown, just flagged. Move each line up into\n'
    '  # would_live_here or would_not_live_here to make it a rule, then delete it from here.\n'
)


def append_not_yet_bucketed(entries: list[tuple[str, str, str]]) -> bool:
    """Append `(place, guess, reason)` rows under `locations.not_yet_bucketed`. True if written.

    A TEXTUAL append, not a re-serialization: the file is hand-written and commented, and this
    project has pyyaml only, whose `safe_dump` would delete every comment in it.

    Safe by construction — the result is parsed before it replaces the original, and anything
    unexpected leaves the file untouched and logs. This is the only list the agent may write, and
    it is allowed *because* `not_yet_bucketed` cannot reject a job; the agent never appends to a
    list that gates anything.
    """
    if not entries:
        return False
    try:
        original = PREFERENCES_PATH.read_text(encoding='utf-8')
    except OSError as ex:
        logger.error(f'Could not read {PREFERENCES_PATH} to record location guesses: {ex}')
        return False

    lines = original.splitlines(keepends=True)
    today = date.today().isoformat()
    new_rows = [
        f'    - {place}   # guessed {today}: {guess}'
        f'{" — " + reason if reason else ""}\n'
        for place, guess, reason in entries
    ]

    insert_at = _bucket_insertion_point(lines)
    if insert_at is None:
        logger.error(
            f'Could not find `locations:` in {PREFERENCES_PATH}, so {len(entries)} location '
            f'guess(es) were not recorded. They are still reported; add the key by hand:\n'
            f'{_BUCKET_HEADER}  {_BUCKET_KEY}:\n' + ''.join(new_rows)
        )
        return False
    lines[insert_at:insert_at] = new_rows

    candidate = ''.join(lines)
    try:
        reparsed = yaml.safe_load(candidate)
        assert isinstance(reparsed, dict) and _BUCKET_KEY in (reparsed.get('locations') or {})
    except (yaml.YAMLError, AssertionError) as ex:
        logger.error(
            f'Appending location guesses would have made {PREFERENCES_PATH} unparseable ({ex}); '
            f'the file was left untouched.'
        )
        return False

    try:
        PREFERENCES_PATH.write_text(candidate, encoding='utf-8')
    except OSError as ex:
        logger.error(f'Could not write location guesses to {PREFERENCES_PATH}: {ex}')
        return False
    logger.info(f'Recorded {len(entries)} location guess(es) in {PREFERENCES_PATH}')
    return True


def _bucket_insertion_point(lines: list[str]) -> int | None:
    """Index to insert at: after the last `not_yet_bucketed` item, or the key is created.

    Returns None when there is no `locations:` block to attach to.
    """
    locations_at = next((i for i, ln in enumerate(lines) if ln.rstrip('\n') == 'locations:'), None)
    if locations_at is None:
        return None

    end_of_block = len(lines)
    for i in range(locations_at + 1, len(lines)):
        stripped = lines[i].rstrip('\n')
        if stripped and not stripped[0].isspace():      # next top-level key
            end_of_block = i
            break

    key_at = next(
        (i for i in range(locations_at + 1, end_of_block)
         if lines[i].strip().rstrip(':') == _BUCKET_KEY and lines[i].strip().endswith(':')),
        None,
    )
    if key_at is None:                                   # create the key at the end of the block
        new_lines = [_BUCKET_HEADER, f'  {_BUCKET_KEY}:\n']
        lines[end_of_block:end_of_block] = new_lines
        return end_of_block + len(new_lines)

    after = key_at + 1
    while after < end_of_block and lines[after].strip().startswith('- '):
        after += 1
    return after
