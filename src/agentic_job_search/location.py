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

import logging
from datetime import date
from typing import Any

import yaml

from agentic_job_search.config import OPENROUTER_MODEL
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.triage import chat_openrouter, extract_json_object, unwrap_exception

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

_PROMPT = """You are a geography reference. Answer ONLY about the PLACE named below.

Location text: "{text}"

Return ONLY a JSON object, no prose and no code fence:
{{"countries": ["<country in English>", ...],
  "regions": ["<one region per country, same order>", ...],
  "broad_area": <true|false>,
  "local_language": "<dominant working language of the FIRST country, lowercase English name>"}}

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
"""

_cache: dict[str, Any] | None = None


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
    return {'countries': [], 'regions': [], 'broad_area': False, 'local_language': '', 'source': reason}


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
    if key in cache:
        entry = dict(cache[key])
        entry['source'] = 'cache'
        return entry

    try:
        content, cost = await chat_openrouter(_PROMPT.format(text=text, regions=', '.join(LOCATION_REGIONS)))
        result = _coerce(extract_json_object(content))
    except Exception as ex:
        # Fail open: nothing the user wrote down matched this location, so an outage must not
        # start rejecting jobs. Logged at WARNING because a silent classifier is a silent gate.
        logger.warning(f'Location classification failed for {text!r}, treating as unknown: {unwrap_exception(ex)}')
        return _empty('error')

    result['model'] = OPENROUTER_MODEL
    result['classified_on'] = date.today().isoformat()
    cache[key] = result
    _write_cache()
    logger.info(
        f'Location classified: {text!r} -> countries={result["countries"]} '
        f'regions={result["regions"]} local_language={result["local_language"]!r} (${cost:.6f})'
    )
    entry = dict(result)
    entry['source'] = 'llm'
    return entry
