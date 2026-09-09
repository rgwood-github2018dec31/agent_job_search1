"""Advisory review of the user's country lists. Recommends; never enforces.

The geographic gate decides where a role may be anchored from two hand-written lists in
``run_dir/preferences.yaml`` -- ``locations.exclude`` and ``hybrid.acceptable_locations`` -- and,
for anything neither names, from a cached classifier plus ``locations.reject_regions``. Region
policy is coarse and phrasing-dependent at the margins (measured: ``Dublin, Ireland`` ->
western_europe, bare ``Ireland`` -> northern_europe), so the lists are how a specific country is
actually decided. Nothing previously told the user which countries had turned up and been left to
the region tier -- ``locations.exclude`` was empty and all 26 countries seen rode on it.

Two halves, deliberately split by who can be trusted with which question:

1. **Substring collisions are found in code, free, every run.** ``'roma' in 'romania'`` is
   arithmetic, and CLAUDE.md's rule that a structural fact must never be left to a model's
   discretion applies to a reviewer exactly as it does to the rater. It also means the warning
   still appears when the tool server is down -- which matters, because this is the check that
   would have caught the entry that silently disabled the gate for a whole country.
2. **"Should this country be excluded?" is a judgement**, and gets one cheap LLM call -- only when
   there is an undecided country, or the lists changed since the last review.

Nothing here writes ``preferences.yaml``. An auto-reject is unappealable, so a recommendation
waits at ``status: pending`` until the user promotes it by hand.
"""

import hashlib
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
from agentic_job_search.triage import chat_openrouter, extract_json_object, unwrap_exception
from utils_tools_n_agents_common.models import OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE

logger = logging.getLogger(__name__)

LOCATION_RECOMMENDATIONS_PATH = RUN_DIR / 'location_recommendations.yaml'

# Recommendation vocabulary. Closed in code, because this is a generative call whose output is
# persisted -- an open string invites a fourth value nothing handles (the location._coerce shape).
RECOMMENDATIONS = ('exclude', 'hybrid_acceptable', 'keep_as_is')

# A country nobody has decided on is asked about once. More than this in one run means something
# unusual happened; a payload that can grow without bound is a truncation waiting to happen.
MAX_COUNTRIES_PER_REVIEW = 40

_PROMPT = """You are helping someone tune the country lists of an automated job-search filter.

Their own stated policy on where they will work, in their words:
\"\"\"{relocation_note}\"\"\"

Lists as they stand today:
- exclude (a job anchored here is auto-rejected): {exclude}
- acceptable_locations (places they would actually be, exempt from every geographic rule): {acceptable}
- reject_regions (regions rejected when neither list names the country): {reject_regions}

Countries that showed up in job postings this run and that NEITHER list names, with the region the
classifier assigned each:
{countries}

For EACH of those countries, recommend one of:
- "exclude"           -- they would not live there; put it on the exclude list
- "hybrid_acceptable" -- they would live there; put it on acceptable_locations
- "keep_as_is"        -- the region policy above already handles it correctly; no list entry needed

Return ONLY a JSON object, no prose and no code fence:
{{"recommendations": [{{"country": "<one of the countries listed above>",
                       "recommendation": "<exclude|hybrid_acceptable|keep_as_is>",
                       "reason": "<one short sentence>"}}, ...]}}
"""


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
    '# To act on a recommendation, edit run_dir/preferences.yaml yourself and set this entry to\n'
    '# `status: accepted`. To dismiss one, set `status: rejected`. Either way the agent stops\n'
    "# asking about that country. Entries you have not touched stay `pending`, and the agent\n"
    '# re-reports only ones it wrote this run.\n'
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


def _list_fingerprint() -> str:
    """md5 of the three lists, so EDITING them re-opens the pending recommendations.

    Keyed on the country universe alone, a recommendation made under the old lists would stand
    forever: the countries did not change, so nothing would ask again. Editing the lists is
    exactly when a not-yet-acted-on recommendation is worth revisiting.
    """
    material = '|'.join(
        ','.join(sorted(tokens)) for tokens in (
            preferences.excluded_locations(),
            preferences.hybrid_acceptable_locations(),
            preferences.rejected_regions(),
        )
    )
    return hashlib.md5(material.encode('utf-8')).hexdigest()


def substring_collisions(countries: dict[str, str]) -> list[dict[str, str]]:
    """List entries that are a proper substring of a country name, or of another entry.

    Deterministic and free. This is the check that catches `roma` inside `romania` -- the entry
    that exempted every Romanian posting at tier 1, before the deny list, the classifier and the
    region policy were consulted. It is code and not a prompt because it is a checkable fact, and
    because it must keep working when the tool server does not.
    """
    listed = [
        ('locations.exclude', token) for token in preferences.excluded_locations()
    ] + [
        ('hybrid.acceptable_locations', token) for token in preferences.hybrid_acceptable_locations()
    ]
    haystacks = set(countries) | {token for _list, token in listed}
    collisions: list[dict[str, str]] = []
    for list_name, token in listed:
        for other in sorted(haystacks):
            # Fold for DETECTION -- entries are proper names while the classifier's countries come
            # back lowercase, so an exact comparison would see no collision at all. The exact
            # matcher still decides whether it is a genuine whole-token match.
            if other == token or _fold(token) not in _fold(other) or location_token_matches(token, other):
                continue
            collisions.append({
                'entry': token,
                'list': list_name,
                'collides_with': other,
                'detail': (
                    f'{token!r} is a substring of {other!r}. Whole-word matching keeps them apart '
                    f'today, but the entry reads as ambiguous — prefer an unambiguous spelling.'
                ),
                'status': 'pending',
            })
    return collisions


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
    listed = [
        ('locations.exclude', token) for token in preferences.excluded_locations()
    ] + [
        ('hybrid.acceptable_locations', token) for token in preferences.hybrid_acceptable_locations()
    ]
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
                'status': 'pending',
            })
    return findings


def undecided_countries(seen: dict[str, str], doc: dict[str, Any]) -> dict[str, str]:
    """Countries neither list names and that no recommendation already covers."""
    named = tuple(preferences.excluded_locations()) + tuple(preferences.hybrid_acceptable_locations())
    recorded = doc.get('countries') or {}
    recorded_folded = {str(key).casefold() for key in recorded}
    return {
        country: region
        for country, region in sorted(seen.items())
        if country.casefold() not in recorded_folded
        # Folded at the COMPARISON, not at either source. Both sides are proper names today, but a
        # cache entry written before that was true still holds 'spain', and a country that reads as
        # undecided because of its casing is one the reviewer asks the LLM about on every run --
        # silently turning "zero calls in steady state" into a call per country per run.
        and not any(location_token_matches(token.casefold(), country.casefold()) for token in named)
    }


def _coerce_recommendations(raw: Any, allowed: dict[str, str]) -> list[dict[str, str]]:
    """Keep only well-formed recommendations about countries we actually asked about.

    Anti-fabrication: the model is being asked a judgement question and its answer is persisted, so
    the country must be one we named and the verdict must be in the closed vocabulary.
    """
    kept: list[dict[str, str]] = []
    for item in (raw.get('recommendations') if isinstance(raw, dict) else None) or []:
        if not isinstance(item, dict):
            continue
        # The country keeps its proper name; only the ENUM is folded, because that is a
        # vocabulary rather than a name. Matching back to what we asked about is done
        # case-insensitively here, at the comparison.
        country = ' '.join(str(item.get('country') or '').split())
        recommendation = str(item.get('recommendation') or '').strip().casefold()
        canonical = next((a for a in allowed if a.casefold() == country.casefold()), '')
        if not canonical or recommendation not in RECOMMENDATIONS:
            continue
        kept.append({
            'country': canonical,
            'recommendation': recommendation,
            'reason': ' '.join(str(item.get('reason') or '').split())[:300],
        })
    return kept


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
    if not (preferences.excluded_locations() or preferences.rejected_regions()):
        return ''  # no geographic gate configured, so there are no lists to tune

    doc = _load()
    seen = countries_seen_this_run()

    findings = substring_collisions(seen) + dead_entries(place_names_seen_this_run())
    new_collisions = [
        collision for collision in findings
        if not any(
            existing.get('entry') == collision['entry']
            and existing.get('list') == collision['list']
            and existing.get('collides_with') == collision['collides_with']
            for existing in doc['entry_warnings']
        )
    ]

    fingerprint = _list_fingerprint()
    lists_changed = doc['reviewed'].get('lists_fingerprint') != fingerprint
    if lists_changed:
        # The lists moved, so a recommendation the user has not acted on was made against a policy
        # that no longer holds -- drop it and ask again. Anything they DID act on (accepted or
        # rejected) is their decision and is never rewritten. Without this the fingerprint would
        # record a change it never responded to.
        doc['countries'] = {
            country: entry for country, entry in doc['countries'].items()
            if (entry or {}).get('status') != 'pending'
        }

    undecided = undecided_countries(seen, doc)
    recommendations: list[dict[str, str]] = []

    if undecided:
        asked = dict(sorted(undecided.items())[:MAX_COUNTRIES_PER_REVIEW])
        content, cost = await chat_openrouter(_PROMPT.format(
            relocation_note=preferences.relocation_note() or '(not stated)',
            exclude=list(preferences.excluded_locations()) or '(empty)',
            acceptable=list(preferences.hybrid_acceptable_locations()) or '(empty)',
            reject_regions=list(preferences.rejected_regions()) or '(empty)',
            countries='\n'.join(f'- {country} ({region})' for country, region in asked.items()),
        ))
        if stage_stats is not None:
            stage_stats['cost'] = stage_stats.get('cost', 0.0) + cost
        recommendations = _coerce_recommendations(extract_json_object(content), asked)
        today = date.today().isoformat()
        for item in recommendations:
            doc['countries'][item['country']] = {
                'first_seen': today,
                'region': asked.get(item['country'], 'unknown'),
                'recommendation': item['recommendation'],
                'reason': item['reason'],
                'status': 'pending',
                'model': OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE,
            }
        logger.info(
            f'Location list review: {len(recommendations)} recommendation(s) for '
            f'{sorted(asked)} (${cost:.6f})'
        )

    if not (new_collisions or recommendations):
        if lists_changed:
            doc['reviewed'] = {'lists_fingerprint': fingerprint, 'on': date.today().isoformat()}
            _write(doc)
        return ''

    doc['entry_warnings'].extend(new_collisions)
    doc['reviewed'] = {'lists_fingerprint': fingerprint, 'on': date.today().isoformat()}
    _write(doc)

    # Only NEW findings raise an alert. A permanently-pending set that puts a warning in every
    # Telegram summary trains the user to ignore the block -- the same cry-wolf reasoning the
    # Resilience requirement applies to a tool server that is merely down.
    parts = []
    if recommendations:
        parts.append(
            f"{len(recommendations)} country list recommendation(s): "
            + ', '.join(f"{item['country']} -> {item['recommendation']}" for item in recommendations)
        )
    if new_collisions:
        parts.append(
            f"{len(new_collisions)} ambiguous list entr(y/ies): "
            + ', '.join(f"{c['entry']!r} inside {c['collides_with']!r}" for c in new_collisions)
        )
    return f"{'; '.join(parts)} — see {LOCATION_RECOMMENDATIONS_PATH.name} (advisory; nothing changed)"
