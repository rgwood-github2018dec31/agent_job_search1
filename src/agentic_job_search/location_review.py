"""Advisory hygiene checks on the user's place lists. Recommends; never enforces.

The geographic gate decides where a role may be anchored from hand-written lists in
``run_dir/preferences.yaml`` -- ``locations.would_live_here``, ``locations.would_not_live_here``
and ``locations.would_commute_here`` -- matched whole-word, case-sensitively and accent-exactly.
Those rules make two kinds of bad entry possible, and neither announces itself. Both checks here
are code, free, and run every time:

1. **Substring collisions.** ``'Roma' in 'Romania'`` is arithmetic, and the rule that a structural
   fact must never be left to a model's discretion applies to a reviewer exactly as it does to the
   rater. It also means the warning still appears when the tool server is down -- which matters,
   because this is the check that would have caught the entry that silently disabled the gate for
   a whole country.
2. **Dead entries** -- one that never matches, but would with its case or accents fixed
   (``malaga`` against postings that say ``Málaga``).

The judgement question -- "would they live in this country?" -- is not asked here. It lives in
``agent.resolve_location_guesses``, which appends its answer to ``locations.not_yet_bucketed``.

Nothing here writes ``preferences.yaml``. Findings wait at ``status: pending`` in
``location_recommendations.yaml`` until the user fixes the entry by hand.
"""

import logging
import unicodedata
from datetime import date
from typing import Any

import yaml

from agentic_job_search.location import (
    countries_seen_this_run, location_token_matches, place_names_seen_this_run,
)
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search import preferences
from agentic_job_search.triage import unwrap_exception

logger = logging.getLogger(__name__)

LOCATION_RECOMMENDATIONS_PATH = RUN_DIR / 'location_recommendations.yaml'


def _load() -> dict[str, Any]:
    """Read the recommendations file, or raise if it is unreadable.

    Deliberately NOT the rebuild-on-corruption behaviour of location.py's cache. A cache is
    recomputable; this is a record of the user's decisions, and silently rebuilding it would throw
    away every entry they marked accepted or rejected.
    """
    if not LOCATION_RECOMMENDATIONS_PATH.exists():
        return {'countries': {}, 'entry_warnings': [], 'reviewed': {}}
    loaded = yaml.safe_load(LOCATION_RECOMMENDATIONS_PATH.read_text(encoding='utf-8')) or {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f'{LOCATION_RECOMMENDATIONS_PATH} must contain a YAML mapping at the top level, got '
            f'{type(loaded).__name__}. Fix or delete it; it is advisory and safe to delete.'
        )
    loaded.setdefault('countries', {})
    loaded.setdefault('entry_warnings', [])
    loaded.setdefault('reviewed', {})
    return loaded


_HEADER = (
    '# Advisory only. NOTHING here changes what the agent does.\n'
    '#\n'
    '# To act on a finding, fix the entry in run_dir/preferences.yaml yourself and set it to\n'
    '# `status: accepted`. To dismiss one, set `status: rejected`. Each finding is reported once,\n'
    '# when it is first written, so an unread backlog does not warn on every run.\n'
    '#\n'
    '# Rewritten by the agent, so comments you add below are NOT preserved.\n'
)


def _write(doc: dict[str, Any]) -> None:
    try:
        LOCATION_RECOMMENDATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCATION_RECOMMENDATIONS_PATH.write_text(
            _HEADER + yaml.safe_dump(doc, sort_keys=True, allow_unicode=True), encoding='utf-8'
        )
    except OSError as ex:
        logger.warning(f'Could not write {LOCATION_RECOMMENDATIONS_PATH}: {ex}')


def substring_collisions(countries: dict[str, str]) -> list[dict[str, str]]:
    """List entries that sit inside a place they do not mean. Deterministic and free.

    This is the check that catches `Roma` inside `Romania` -- the entry that exempted every
    Romanian posting before the deny list, the classifier and the region policy were consulted. It
    is code and not a prompt because it is a checkable fact, and because it must keep working when
    the tool server does not.

    An entry is skipped against places on its OWN list: `Valencia`/`València` and `Málaga`/`Malaga`
    are deliberate spelling variants, and each flagging the other produced 5 false findings against
    1 real one. Across lists it is the dangerous case and is always reported -- allow beats deny, so
    `Roma` on `would_live_here` sitting inside `Romania` on `would_not_live_here` is exactly how a
    whole country gets silently exempted.
    """
    listed = _all_listed_places()
    own_list: dict[str, set[str]] = {}
    for list_name, token in listed:
        own_list.setdefault(list_name, set()).add(_fold(token))

    places = set(countries) | {token for _list, token in listed}
    collisions: list[dict[str, str]] = []
    for list_name, token in listed:
        for other in sorted(places):
            if other == token or _fold(other) in own_list[list_name]:
                continue
            if _fold(token) not in _fold(other) or location_token_matches(token, other):
                continue
            collisions.append({
                'entry': token,
                'list': list_name,
                'collides_with': other,
                'detail': (
                    f'{token!r} on {list_name} is a substring of {other!r}. Whole-token matching '
                    f'keeps them apart today, but the entry reads as ambiguous — prefer an '
                    f'unambiguous spelling.'
                ),
                'kind': 'collision',
                'status': 'pending',
            })
    return collisions



def _all_listed_places() -> list[tuple[str, str]]:
    """(list name, entry) for every user-authored place list, for the hygiene checks."""
    return [
        (f'locations.{name}', token)
        for name, getter in (
            ('would_live_here', preferences.would_live_here),
            ('would_not_live_here', preferences.would_not_live_here),
            ('would_commute_here', preferences.would_commute_here),
        )
        for token in getter()
    ]


def _fold(text: str) -> str:
    """Case- and accent-folded form, for asking 'would this entry have matched if it were written
    the way the place actually is?' -- never for matching."""
    stripped = unicodedata.normalize('NFKD', str(text or ''))
    return ''.join(c for c in stripped if not unicodedata.combining(c)).casefold()


def dead_entries(place_names: set[str]) -> list[dict[str, str]]:
    """List entries that match nothing seen, but WOULD if case or accents were fixed.

    Deterministic and free, like the collision check. This is how `malaga` gets caught: LinkedIn
    writes `Málaga`, matching is accent-exact, and the entry had therefore never matched anything
    in the corpus -- while looking exactly like an entry that simply had not come up yet. That is
    the distinction this check makes and a plain "never matched" count cannot.
    """
    listed = _all_listed_places()
    findings: list[dict[str, str]] = []
    for list_name, token in listed:
        if any(location_token_matches(token, name) for name in place_names):
            continue
        # WHOLE-TOKEN on the folded forms, never a substring. Folding plus `in` would report
        # 'Roma' as "should have been written 'Romania'" -- advice that walks straight back into
        # the bug this whole entry exists for. Folding is allowed to ignore case and accents; it
        # is never allowed to ignore boundaries.
        folded = _fold(token)
        would = [name for name in sorted(place_names) if location_token_matches(folded, _fold(name))]
        if would:
            findings.append({
                'entry': token,
                'list': list_name,
                'collides_with': would[0],
                'detail': (
                    f'{token!r} never matched anything this run, but {would[0]!r} did — matching is '
                    f'case-sensitive and accent-exact, so write the entry the way the place is '
                    f'written.'
                ),
                'kind': 'dead_entry',
                'status': 'pending',
            })
    return findings


async def review_location_lists(stage_stats: dict[str, Any] | None = None) -> str:
    """Review the country lists against what this run saw. Returns an alert detail, or ''.

    Fails open on everything: this is advisory, and it must never abort a run or change a rating.
    """
    try:
        return await _review(stage_stats)
    except Exception as ex:
        logger.warning(f'Location list review skipped: {unwrap_exception(ex)}')
        return ''


async def _review(stage_stats: dict[str, Any] | None) -> str:
    """Deterministic hygiene on the user's place lists. No LLM, no cost, every run.

    The judgement half -- "would they live in this country?" -- moved to
    `agent.resolve_location_guesses`, which answers it inline, appends the answer to
    `locations.not_yet_bucketed` where the user acts on it, and sends one Telegram. Asking the same
    question twice in two places would have been two sources of truth for one decision.

    What is left has no equivalent and cannot be asked of a model anyway: `'Roma' in 'Romania'` is
    arithmetic, and an entry that never matches anything is a fact about the corpus.
    """
    if not (preferences.would_live_here() or preferences.would_not_live_here()):
        return ''

    doc = _load()
    seen = countries_seen_this_run()
    findings = substring_collisions(seen) + dead_entries(place_names_seen_this_run())
    new_findings = [
        finding for finding in findings
        if not any(
            existing.get('entry') == finding['entry']
            and existing.get('list') == finding['list']
            and existing.get('collides_with') == finding['collides_with']
            for existing in doc['entry_warnings']
        )
    ]
    if not new_findings:
        return ''

    doc['entry_warnings'].extend(new_findings)
    doc['reviewed'] = {'on': date.today().isoformat()}
    _write(doc)

    by_kind: dict[str, list] = {'collision': [], 'dead_entry': []}
    for finding in new_findings:
        by_kind.setdefault(finding.get('kind', 'collision'), []).append(finding)
    parts = []
    if by_kind['collision']:
        parts.append(
            f"{len(by_kind['collision'])} ambiguous list entr(y/ies): "
            + ', '.join(f"{c['entry']!r} sits inside {c['collides_with']!r}" for c in by_kind['collision'])
        )
    if by_kind['dead_entry']:
        parts.append(
            f"{len(by_kind['dead_entry'])} list entr(y/ies) that never match: "
            + ', '.join(f"{c['entry']!r} (postings say {c['collides_with']!r})" for c in by_kind['dead_entry'])
        )
    return f"{'; '.join(parts)} — see {LOCATION_RECOMMENDATIONS_PATH.name} (advisory; nothing changed)"
