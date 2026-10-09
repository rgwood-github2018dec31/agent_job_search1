"""A salary's equivalent in the user's own currency: display only, computed in code, fails open.

The job-match message shows a salary exactly as the posting wrote it. For a currency the user does
not think in, that leaves the conversion to be done by hand — or, worse, by the rater in a bullet.
This module appends a figure code computed from a published daily rate. It never replaces the
posting's text, and nothing here feeds a rating, a cap, a gate or a warning: `salary_meets_target`
still abstains on another currency.

Which currency to convert into, and which to leave alone, are preferences
(`compensation.convert_to`, `compensation.keep_currencies`); unset means nothing is converted.

Failure fails **toward showing less**: no rate, an ambiguous currency, or a string stating several
ranges yields no conversion at all, never a guessed one.
"""

import logging
import math
from datetime import date
from http import HTTPStatus
from typing import Any

import requests
import yaml

import agentic_job_search.preferences as preferences
from agentic_job_search.config import (
    FX_RATES_FETCH_TIMEOUT_SECONDS,
    FX_RATES_STALE_MAX_DAYS,
    FX_RATES_URL,
    SALARY_CONVERSION_SIGNIFICANT_FIGURES,
)
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.salary import count_ranges, currencies_named
from agentic_job_search.text_budget import snippet

logger = logging.getLogger(__name__)

FX_RATES_CACHE_PATH = RUN_DIR / 'fx_rates_cache.yaml'

# Bumped when the cache file's shape changes; a table under another schema is refetched.
CACHE_SCHEMA_VERSION = 1

_table: dict[str, Any] | None = None
# One fetch attempt per run: a rate service that is down stays down for the next job too, and a
# run must not pay the timeout once per notified job.
_fetch_attempted = False
# Currencies already reported as having no rate, so the WARNING is logged once per run.
_unrated_reported: set[str] = set()


def reset_run_state() -> None:
    """Forget the loaded table and this run's fetch attempt. Called per run and per test."""
    global _table, _fetch_attempted
    _table = None
    _fetch_attempted = False
    _unrated_reported.clear()


def _http_get(url: str, params: dict[str, str]) -> requests.Response:
    """The one outbound call, kept thin so tests replace it."""
    return requests.get(url, params=params, timeout=FX_RATES_FETCH_TIMEOUT_SECONDS)


def parse_rates(payload: object, base: str) -> dict[str, dict[str, Any]]:
    """The service's answer as {quote currency: {'rate': quote units per one base, 'date': ...}}.

    Raises ValueError on any other shape. A table that parsed loosely would turn a changed API
    into wrong figures in a notification rather than into a missing one.
    """
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f'exchange-rate response for base {base} must be a non-empty list of rate rows, got '
            f'{type(payload).__name__}: {snippet(payload)}'
        )
    rates: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f'exchange-rate row for base {base} is not a mapping: {snippet(row)}')
        quote, rate = row.get('quote'), row.get('rate')
        valid_rate = isinstance(rate, (int, float)) and not isinstance(rate, bool) and math.isfinite(rate) and rate > 0
        if row.get('base') != base or not isinstance(quote, str) or not quote or not valid_rate:
            raise ValueError(
                f'exchange-rate row for base {base} needs a matching `base`, a `quote` code and a '
                f'positive `rate`, got {snippet(row)}'
            )
        rates[quote.upper()] = {'rate': float(rate), 'date': str(row.get('date') or '')}
    return rates


def fetch_rates(base: str) -> dict[str, dict[str, Any]]:
    """Today's rates against `base` from the service. Raises on any failure; the caller decides."""
    response = _http_get(FX_RATES_URL, {'base': base})
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f'exchange-rate service {FX_RATES_URL} answered HTTP {response.status_code} for base '
            f'{base}: {snippet(response.text)}'
        )
    return parse_rates(response.json(), base)


def _load_table() -> dict[str, Any]:
    """Read the on-disk rate table. A missing or corrupt file is refetched, never fatal."""
    global _table
    if _table is not None:
        return _table
    _table = {}
    if FX_RATES_CACHE_PATH.exists():
        try:
            loaded = yaml.safe_load(FX_RATES_CACHE_PATH.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                _table = loaded
            elif loaded is not None:
                logger.warning(
                    f'Exchange-rate cache {FX_RATES_CACHE_PATH} is not a mapping '
                    f'({type(loaded).__name__}) — refetching.'
                )
        except Exception as ex:
            logger.warning(
                f'Could not read exchange-rate cache {FX_RATES_CACHE_PATH}: {type(ex).__name__}: {ex} — refetching.'
            )
    return _table


def _write_table() -> None:
    """Persist the table. A table that cannot be written costs a fetch next run, not correctness."""
    try:
        FX_RATES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FX_RATES_CACHE_PATH.write_text(
            yaml.safe_dump(_load_table(), sort_keys=True, allow_unicode=True), encoding='utf-8'
        )
    except Exception as ex:
        logger.warning(
            f'Could not write exchange-rate cache {FX_RATES_CACHE_PATH}: {type(ex).__name__}: {ex} — continuing.'
        )


def _table_age_days(table: dict[str, Any], base: str, today: date) -> int | None:
    """Days since the table was fetched, or None when it is not a usable table for `base`."""
    if table.get('schema') != CACHE_SCHEMA_VERSION or table.get('base') != base:
        return None
    if not isinstance(table.get('rates'), dict):
        return None
    try:
        fetched_on = date.fromisoformat(str(table.get('fetched_on')))
    except ValueError:
        return None
    return (today - fetched_on).days


def rates_for(base: str, today: date | None = None) -> dict[str, dict[str, Any]] | None:
    """Rates against `base`: today's, fetched at most once a day and once per run.

    When the fetch fails, a table no older than `FX_RATES_STALE_MAX_DAYS` is served instead; past
    that, None — and the caller shows no conversion.
    """
    global _table, _fetch_attempted
    today = today or date.today()
    table = _load_table()
    age_days = _table_age_days(table, base, today)
    if age_days == 0:
        return table['rates']

    if not _fetch_attempted:
        _fetch_attempted = True
        try:
            rates = fetch_rates(base)
        except Exception as ex:
            fallback = (
                f"serving the table fetched on {table.get('fetched_on')}"
                if age_days is not None and age_days <= FX_RATES_STALE_MAX_DAYS
                else 'no usable cached table, so salaries are shown without a conversion this run'
            )
            logger.warning(
                f'Exchange-rate fetch for base {base} from {FX_RATES_URL} failed: '
                f'{type(ex).__name__}: {ex} — {fallback}.'
            )
        else:
            _table = {
                'schema': CACHE_SCHEMA_VERSION, 'base': base, 'fetched_on': today.isoformat(),
                'source': FX_RATES_URL, 'rates': rates,
            }
            _write_table()
            logger.info(f'Exchange rates fetched for base {base}: {len(rates)} currencies')
            return rates

    if age_days is not None and age_days <= FX_RATES_STALE_MAX_DAYS:
        return table['rates']
    return None


def round_significant(value: float, figures: int = SALARY_CONVERSION_SIGNIFICANT_FIGURES) -> float:
    """148312.4 -> 148000.0 and 112.9 -> 113.0 at three figures: readable at any pay period."""
    if value == 0:
        return 0.0
    return round(value, figures - 1 - math.floor(math.log10(abs(value))))


def _usable_bound(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def convert_bounds(
    minimum: float | None, maximum: float | None, currency: str, text: str = ''
) -> dict[str, Any] | None:
    """The bounds in the user's conversion currency, or None when no conversion should be shown.

    None — never a guess — when: no conversion preference is set; the currency is unresolved (a
    bare '$') or one the user keeps as written; a bound is missing or implausible; `text` states
    several ranges or names several currencies, so the bounds do not speak for the whole string;
    or there is no rate. The pay period is left alone: an hourly rate converts to an hourly rate.
    """
    conversion = preferences.salary_conversion()
    if conversion is None:
        return None
    convert_to, keep = conversion
    if not currency or currency in keep:
        return None
    bounds = [b for b in (minimum, maximum) if b is not None]
    if not bounds or not all(_usable_bound(b) for b in bounds):
        return None
    if minimum is not None and maximum is not None and minimum > maximum:
        return None
    if count_ranges(text) > 1 or len(currencies_named(text)) > 1:
        return None

    rates = rates_for(convert_to)
    entry = (rates or {}).get(currency)
    if not isinstance(entry, dict) or not _usable_bound(entry.get('rate')):
        if currency not in _unrated_reported:
            _unrated_reported.add(currency)
            logger.warning(
                f'No exchange rate from {currency} to {convert_to} — a salary in {currency} is '
                f'shown without a conversion.'
            )
        return None
    rate = float(entry['rate'])
    return {
        'currency': convert_to,
        'from_currency': currency,
        'minimum': round_significant(minimum / rate) if minimum is not None else None,
        'maximum': round_significant(maximum / rate) if maximum is not None else None,
        # Quote units per one unit of the conversion currency, as the service publishes it.
        'rate': rate,
        'rate_date': str(entry.get('date') or ''),
    }
