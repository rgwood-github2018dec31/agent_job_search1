"""Geographic classification of a job's location, with a persistent cache.

This answers ONE question, and it is deliberately narrow: **where in the world is this place?**
It says nothing about the job, the fit, or whether to reject — code applies the user's policy to
the facts returned here (see `agent.rejected_location`).

Why an LLM call sits behind a hard rule at all
----------------------------------------------
CLAUDE.md's rule that "a structural fact must never be left to a model's discretion" exists
because the *rater* scored Berlin 5/5 one day and Munich 3/5 the next on identical facts. This is
categorically different, on three counts:

1. The model answers a question about a **place**, never about the job. "Which country is Berlin
   in, which region is that, what is spoken there" is reusable world knowledge, checkable by a
   human, and identical for every job in that location.
2. **The cache is what makes it deterministic.** Two Berlin jobs cannot get different answers,
   within a run or across runs — and that determinism is precisely what the anti-discretion rule
   protects. A hand-maintained country table would be deterministic too, but it is guesswork at
   the edges (is Slovenia Adriatic? is Romania Mediterranean?) and it silently rots.
3. The **policy stays in code**: region -> accept/reject is a deterministic read of a user
   preference, not a judgement the model is asked to make.

The same shape already exists in `tools_generic.company_blacklist_reason`, which matches a
user-authored list deterministically and spends one cheap confirmation call only on the
ambiguous case.

Failure is fail-OPEN, deliberately the opposite of the blacklist's fail-closed rule. Nothing the
user wrote down matched this location, so a tool-server outage must not start rejecting jobs it
would otherwise have kept.
"""

import functools
import logging
import re
from datetime import date
from typing import Any

import yaml

from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.triage import chat_openrouter, extract_json_object, unwrap_exception
from utils_tools_n_agents_common.models import OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE

logger = logging.getLogger(__name__)

LOCATION_CACHE_PATH = RUN_DIR / 'location_cache.yaml'

# The region vocabulary the classifier may use. Kept coarse on purpose: these are the distinctions
# a person actually makes when deciding where they would live, and a finer taxonomy would just be
# more surface for the model to be inconsistent across.
LOCATION_REGIONS = (
    'southern_europe',   # Mediterranean + Iberia: Spain, Portugal, Italy, Greece, Malta, Cyprus,
                         # Croatia, southern France
    'northern_europe',   # Nordics and the Baltics
    'western_europe',    # Germany, Netherlands, Belgium, Austria, Switzerland, France (northern),
                         # Ireland, the UK, Luxembourg
    'eastern_europe',    # Poland, Czechia, Slovakia, Hungary, Romania, Bulgaria, Ukraine, ...
    'north_america',
    'other',             # a real place, outside the regions above
    'unknown',           # not resolvable to a country — 'European Union', 'Remote (EMEA)', ''
)

# EU membership, as of 2026-09-09. World knowledge, so it belongs in tracked source — which
# regions the user will not work in is personal and lives in run_dir/preferences.yaml. Lowercase,
# spelled the way the classifier returns countries; the aliases below cover the variants it also
# uses, because `_coerce` deliberately does not normalize country names.
EU_MEMBER_STATES = frozenset({
    'austria', 'belgium', 'bulgaria', 'croatia', 'cyprus', 'czechia', 'denmark', 'estonia',
    'finland', 'france', 'germany', 'greece', 'hungary', 'ireland', 'italy', 'latvia',
    'lithuania', 'luxembourg', 'malta', 'netherlands', 'poland', 'portugal', 'romania',
    'slovakia', 'slovenia', 'spain', 'sweden',
})

# Spellings the classifier uses interchangeably with the canonical names above.
_EU_ALIASES = frozenset({
    'czech republic', 'the netherlands', 'holland', 'republic of ireland', 'hellas',
})


def is_eu_member(country: str) -> bool:
    """True if `country` names an EU member state, tolerating the classifier's spelling variants."""
    normalized = ' '.join(str(country or '').split()).lower()
    return normalized in EU_MEMBER_STATES or normalized in _EU_ALIASES


@functools.lru_cache(maxsize=512)
def _token_pattern(token: str) -> re.Pattern[str]:
    r"""Compiled whole-token matcher for one place name. Case-SENSITIVE and accent-exact.

    Two things about the boundary class, both load-bearing:

    1. It is a lookaround, never ``\b``. ``re.escape('U.S.')`` is ``U\.S\.``, and ``\b`` after a
       '.' needs a word character to follow — so ``\bU\.S\.\b`` matches NEITHER 'U.S.' nor
       'Remote, U.S. only'. The obvious fix silently deletes an entry the user wrote down.
    2. It is ``[^\W_]`` (unicode letters and digits, minus underscore), not ``[a-z0-9]``. The
       lowercase-ASCII class was only ever safe because the caller lowercased first. Matching
       proper names directly, it lets 'ROMA' reach into 'ROMANIA' and 'Roma' into 'Romaña' —
       the original bug back again, in caps and via an accent. Case-sensitivity and this class
       changed together and cannot be separated.

    No ``re.IGNORECASE``: place names are proper names, and 'Nice' the city is not 'nice' the
    adjective.
    """
    return re.compile(r'(?<![^\W_])' + re.escape(token) + r'(?![^\W_])')


def location_token_matches(token: str, text: str) -> bool:
    """True if `token` appears in `text` as a whole place name rather than as any old substring.

    Bare ``token in text`` is what let 'roma' (Rome, on the acceptable-locations list) match
    'romania (remote within country)' and 'parma, emilia-romagna, italy'. Because that list is
    tier 1 of the geographic gate and exempts OUTRIGHT, every Romanian posting skipped the deny
    list, the classifier and the region policy — one of them was rated 4 and notified. The test
    double in tests/conftest.py had used word boundaries all along, with a comment about 'nice'
    inside 'Venice'; production never did.

    Case and accents are preserved on both sides: the lists hold place names as places are
    actually written ('Málaga', 'Sevilla'), and folding them was how 'malaga' came to be an entry
    that could never match anything. Only whitespace is normalized, so a stray space in the YAML
    (' Malaga ') and a tab in a scraped location are both harmless. A place the exact match still
    misses is resolved through `place_names` from the classifier, not by loosening this.
    """
    token = ' '.join(str(token or '').split())
    if not token:
        return False
    return _token_pattern(token).search(' '.join(str(text or '').split())) is not None


_PROMPT = """You are a geography reference. Answer ONLY about the PLACE named below.

Location text: "{text}"

Return ONLY a JSON object, no prose and no code fence:
{{"countries": ["<country in English>", ...],
  "regions": ["<one region per country, same order>", ...],
  "broad_area": <true|false>,
  "local_language": "<dominant working language of the FIRST country, lowercase English name>",
  "place_names": ["<every place this text refers to, English and local spellings>", ...]}}

Rules:
- `regions` must use exactly these values: {regions}.
- One entry in `regions` for each entry in `countries`, in the same order.
- `broad_area` is true when the text offers a whole MULTI-COUNTRY area to work from — "European
  Union", "EU", "EMEA", "Europe", "anywhere", "worldwide", "LATAM". It is true even when specific
  countries are ALSO named, e.g. "European Union (Remote, UK and EU)" is true, because the offer is
  not limited to the countries it happens to name. A single country, however phrased, is false:
  "Germany (Remote across Europe)" is false, since the role is anchored in Germany.
- If the text names no specific country, return an empty `countries` and `regions`.
- If several countries are named ("the UK or the Netherlands"), list every one of them.
- southern_europe means the Mediterranean and Iberia, including SOUTHERN France (Nice, Marseille,
  Montpellier, Toulouse). Northern France, including Paris, is western_europe.
- A city implies its country: "Berlin" -> Germany, "Barcelona" -> Spain.
- `place_names` lists every place the text refers to at EVERY level -- city, region/state, country
  -- in BOTH the common English form and the local form, properly capitalised and accented, e.g.
  "Sevilla, Andalusia, Spain" -> ["Sevilla", "Seville", "Andalucia", "Andalucía", "Andalusia",
  "Spain", "España"]. Include the form as written in the text. Omit workplace words ("Remote",
  "Hybrid"), and return an empty list when the text names no place.
"""

_cache: dict[str, Any] | None = None

# Countries this run actually encountered, country -> region. Run-scoped, NOT the disk cache: the
# cache accumulates forever and says nothing about what today's postings named. Reset explicitly
# at the top of a run (an un-reset module global is the most-repeated bug in CLAUDE.md).
_countries_seen_this_run: dict[str, str] = {}
_place_names_seen_this_run: set[str] = set()


def _record_countries(result: dict[str, Any]) -> None:
    """Note the countries in a classification. Called on BOTH return paths of classify_location.

    The cache-hit path matters as much as the LLM path: in steady state almost every location is a
    hit, so recording only new classifications would report a country once, ever, and then never
    again.
    """
    regions = result.get('regions') or []
    for index, country in enumerate(result.get('countries') or []):
        _countries_seen_this_run.setdefault(country, regions[index] if index < len(regions) else 'unknown')
    _place_names_seen_this_run.update(result.get('place_names') or [])


def countries_seen_this_run() -> dict[str, str]:
    """Countries encountered this run, country -> region."""
    return dict(_countries_seen_this_run)


def place_names_seen_this_run() -> set[str]:
    """Every place name this run encountered, as written by the classifier.

    Countries are lowercase (that is how `countries` comes back); `place_names` are proper names.
    The reviewer needs the proper-name set to tell a list entry that has not come up yet from one
    that can never match — 'malaga' against a corpus that says 'Málaga'.
    """
    return set(_place_names_seen_this_run)


def reset_countries_seen() -> None:
    """Clear the run-scoped accumulators. Called once at the start of a run."""
    _countries_seen_this_run.clear()
    _place_names_seen_this_run.clear()


def _load_cache() -> dict[str, Any]:
    """Read the on-disk cache. A missing or corrupt file is rebuilt, never fatal."""
    global _cache
    if _cache is not None:
        return _cache
    _cache = {}
    if LOCATION_CACHE_PATH.exists():
        try:
            loaded = yaml.safe_load(LOCATION_CACHE_PATH.read_text(encoding='utf-8')) or {}
            if isinstance(loaded, dict):
                _cache = loaded
            else:
                logger.warning(
                    f'Location cache {LOCATION_CACHE_PATH} is not a mapping '
                    f'(got {type(loaded).__name__}) — rebuilding it.'
                )
        except (yaml.YAMLError, OSError) as ex:
            logger.warning(f'Could not read location cache {LOCATION_CACHE_PATH}: {ex} — rebuilding it.')
    return _cache


def _write_cache() -> None:
    try:
        LOCATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCATION_CACHE_PATH.write_text(
            yaml.safe_dump(_load_cache(), sort_keys=True, allow_unicode=True), encoding='utf-8'
        )
    except OSError as ex:
        # A cache that cannot be persisted costs money, not correctness — the run continues.
        logger.warning(f'Could not write location cache {LOCATION_CACHE_PATH}: {ex}')


def cache_key(text: str) -> str:
    """Normalized cache key. Whitespace and case are not geography."""
    return ' '.join(str(text or '').split()).lower()


def _empty(reason: str) -> dict[str, Any]:
    """The fail-open answer: no country named, so no policy can reject."""
    return {
        'countries': [], 'regions': [], 'broad_area': False, 'local_language': '',
        'place_names': [], 'source': reason,
    }


def _coerce(raw: dict) -> dict[str, Any]:
    """Validate the model's JSON into the contract, dropping anything unrecognised.

    An out-of-vocabulary region becomes 'unknown' rather than being passed through: the policy
    check compares against a configured list, and a typo'd region that matched nothing would
    silently read as "acceptable" instead of "I could not tell".
    """
    countries = [str(c).strip().lower() for c in (raw.get('countries') or []) if str(c).strip()]
    regions = [str(r).strip().lower() for r in (raw.get('regions') or [])]
    regions = [r if r in LOCATION_REGIONS else 'unknown' for r in regions]
    # Pad or trim so the two lists always correspond positionally.
    regions = (regions + ['unknown'] * len(countries))[:len(countries)]
    return {
        'countries': countries,
        'regions': regions,
        'broad_area': bool(raw.get('broad_area')),
        'local_language': str(raw.get('local_language') or '').strip().lower(),
        # Proper names, deliberately NOT lowercased: they are matched case-sensitively against the
        # user's lists. Deduped preserving order so the cache file stays stable.
        'place_names': list(dict.fromkeys(
            name for name in (
                ' '.join(str(n or '').split()) for n in (raw.get('place_names') or [])
            ) if name
        )),
    }


async def classify_location(text: str) -> dict[str, Any]:
    """Geographic facts about `text`: {countries, regions, local_language}.

    Cached on disk by normalized location string, so a repeated location costs nothing and — more
    importantly — cannot be answered two different ways. Every failure returns the empty answer,
    which no policy can reject.
    """
    key = cache_key(text)
    if not key:
        return _empty('empty')

    cache = _load_cache()
    # A cached entry predating `place_names` is treated as a MISS and reclassified. Serving it
    # would silently disable alias matching for that location forever, which is the same shape as
    # the dead `malaga` entry this field exists to fix -- a mechanism that reports success while
    # doing nothing. One-off cost: ~81 entries at the flash-tier rate.
    if key in cache and 'place_names' in cache[key]:
        entry = dict(cache[key])
        entry['source'] = 'cache'
        _record_countries(entry)
        return entry

    try:
        content, cost = await chat_openrouter(_PROMPT.format(text=text, regions=', '.join(LOCATION_REGIONS)))
        result = _coerce(extract_json_object(content))
    except Exception as ex:
        # Fail open: nothing the user wrote down matched this location, so an outage must not
        # start rejecting jobs. Logged at WARNING because a silent classifier is a silent gate.
        logger.warning(f'Location classification failed for {text!r}, treating as unknown: {unwrap_exception(ex)}')
        if stale := cache.get(key):
            # ...but a `place_names`-less entry is stale, not wrong. Reclassifying it is an upgrade,
            # and an upgrade that cannot happen must not cost us the answer we already had: without
            # this, one outage silently disables the region gate for every location ever cached.
            entry = dict(stale)
            entry.setdefault('place_names', [])
            entry['source'] = 'stale'
            _record_countries(entry)
            return entry
        return _empty('error')

    result['model'] = OPENROUTER_MODEL_NAME_DEFAULT_INTELLIGENCE
    result['classified_on'] = date.today().isoformat()
    cache[key] = result
    _write_cache()
    logger.info(
        f'Location classified: {text!r} -> countries={result["countries"]} '
        f'regions={result["regions"]} local_language={result["local_language"]!r} (${cost:.6f})'
    )
    entry = dict(result)
    entry['source'] = 'llm'
    _record_countries(entry)
    return entry
