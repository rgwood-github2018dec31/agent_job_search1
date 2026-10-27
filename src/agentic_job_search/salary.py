"""What amounts does a compensation phrase state? Deterministic first, cached, fails toward less.

This answers ONE question, and it is deliberately narrow: **is this string a complete pay range,
one end of one, or not a pay figure at all?** It says nothing about whether the pay is good, or
whether the job is worth surfacing — judging pay against a target stays with the rater (see the
Build Deterministic Warnings requirement), and no policy here rejects or caps anything.

Why this module exists (2026-09-22)
-----------------------------------
`salary_figure_listed()` asked `re.search(r'\\d', salary)` and called anything with a digit a
listed salary. That is true of `CA$208,580-$273,770 annually`, of `Up to €50k`, of `Minimum
€65,000`, of `€110,000` and of `12-month contract / Outside IR35` alike — 93 of 911 figured
postings in the saved corpus state only one end, and a few state no pay at all. A partial range
was therefore indistinguishable from a complete one, the warning stayed silent, and the only
salary text reaching the user was whatever the rater chose to paraphrase: for job 4470298356 it
rounded the top from 273,770 up to 274K in one bullet and led with the floor alone in another.
**A presence check is not a completeness check.**

Why an LLM call sits behind a rule decided in code
--------------------------------------------------
The same three-count argument `location.py` makes, and for the same reason:

1. The model answers a question about a **string** — "what amounts does this phrase state?" — never
   about the job or the fit. It is reusable, checkable by a human, and identical everywhere that
   phrase appears.
2. **The cache is what makes it deterministic.** Two postings whose salary reads `Up to $200/hr
   equivalent` cannot be answered differently, within a run or across runs.
3. The **policy stays in code**: which kinds warn, and with what wording, is a table in
   `agent.build_deterministic_warnings`, not a judgement the model is asked to make.

Nearly everything is answered by the regex tiers below, for free. The model is reached only for a
string they could not read at all.

Failure fails **toward saying less**: a down server yields `unclassified`, which warns that the
string could not be read and gates nothing. It never yields `range`, because the one thing this
module must never do is quietly upgrade a partial range to a complete one.
"""

import hashlib
import logging
import re
from datetime import date
from typing import Any

from utils_tools_n_agents_common.mcp_client import unwrap_exception

from agentic_job_search import cache_io, evidence
from agentic_job_search.config import (
    MODEL_NAME_SALARY,
    SALARY_CONTEXT_MAX_CHARS,
    SALARY_MILLION_MULTIPLIER,
    SALARY_THOUSAND_MULTIPLIER,
    THOUSANDS_GROUP_DIGITS,
)
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.text_budget import truncate_reported
from agentic_job_search.triage import chat_openrouter

logger = logging.getLogger(__name__)

SALARY_CACHE_PATH = RUN_DIR / 'salary_cache.yaml'

# Bumped when a field is added to the contract below. An entry written under an older schema is a
# MISS and is reclassified: serving it would silently disable whatever the new field feeds, which
# is the same shape of bug as a mechanism reporting success while doing nothing (location.py:342).
# 2 (2026-10-19): the model cites the words stating the pay (`quote`); a version-1 answer carries
# no citation, so it is reclassified rather than served unverified.
CACHE_SCHEMA_VERSION = 2

# The vocabulary of answers. `unclassified` is not a failure mode to be minimised — it is the
# honest answer for a string no rule could read, and it is what tier 8 exists to produce rather
# than guessing.
SALARY_KIND_ABSENT = 'absent'
SALARY_KINDS = (SALARY_KIND_ABSENT, 'floor_only', 'ceiling_only', 'single', 'range', 'unclassified')
SALARY_PERIODS = ('year', 'month', 'week', 'day', 'hour')

# Currency symbols and codes, longest-first so 'CA$' wins over '$' and 'CAD' over 'CA'. A BARE '$'
# is deliberately absent: it is genuinely ambiguous (USD, CAD, AUD, SGD...), and guessing a
# currency is not this module's job. An unresolved currency does not make a range partial. A bare
# 'kr' is absent for the same reason (SEK, NOK, DKK and ISK all write it); it still marks an
# amount as money in `_MONETARY_MARKER_RE`.
_CURRENCY_BY_TOKEN = {
    'CA$': 'CAD', 'C$': 'CAD', 'CAD': 'CAD', 'US$': 'USD', 'USD': 'USD', 'A$': 'AUD', 'AUD': 'AUD',
    '€': 'EUR', 'EUR': 'EUR', '£': 'GBP', 'GBP': 'GBP', 'zł': 'PLN', 'PLN': 'PLN', 'CHF': 'CHF',
    'SEK': 'SEK', 'NOK': 'NOK', 'DKK': 'DKK', 'CZK': 'CZK', 'HUF': 'HUF', 'RON': 'RON',
    'INR': 'INR', '₹': 'INR', '¥': 'JPY', 'JPY': 'JPY', 'SGD': 'SGD', 'BRL': 'BRL', 'MXN': 'MXN',
}
_CURRENCY_BY_FOLDED_TOKEN = {token.casefold(): code for token, code in _CURRENCY_BY_TOKEN.items()}
# Every ISO code a reading can carry. The closed vocabulary a model's answer is checked against.
SALARY_CURRENCIES = frozenset(_CURRENCY_BY_TOKEN.values())


def _currency_token_pattern(token: str) -> str:
    """One token's regex. A letter at either edge must not continue into a word.

    Unanchored, the codes matched INSIDE words: 'Toronto: $120,000' read as RON and 'Europe:
    80,000 GBP' as EUR. That only made the target comparison abstain until a reading's currency
    began to drive a conversion shown to the user (2026-10-08). A digit may still touch the token
    ('CAD154,700'), and a symbol edge ('CA$', '€') needs no guard.
    """
    lead = r'(?<![^\W\d_])' if token[0].isalpha() else ''
    trail = r'(?![^\W\d_])' if token[-1].isalpha() else ''
    return f'{lead}{re.escape(token)}{trail}'


_CURRENCY_RE = re.compile(
    '|'.join(_currency_token_pattern(token) for token in sorted(_CURRENCY_BY_TOKEN, key=len, reverse=True)),
    re.IGNORECASE,
)

# Period words, mapped onto the controlled vocabulary above.
_PERIOD_PATTERNS = (
    ('year', re.compile(r'/\s*yr\b|\bper\s+year\b|\bannual(?:ly|ised|ized)?\b|\bp\.?a\.?\b|\byearly\b', re.IGNORECASE)),
    ('month', re.compile(r'/\s*mo\b|\bper\s+month\b|\bmonthly\b|\bgross/month\b', re.IGNORECASE)),
    ('week', re.compile(r'/\s*wk\b|\bper\s+week\b|\bweekly\b', re.IGNORECASE)),
    ('day', re.compile(r'/\s*day\b|\bper\s+day\b|\bdaily\s+rate\b|\bday\s+rate\b|\bdiem\b', re.IGNORECASE)),
    ('hour', re.compile(r'/\s*h(?:r|our)?\b|\bper\s+hour\b|\bhourly\b', re.IGNORECASE)),
)

# An amount: digits with optional grouping separators, an optional decimal tail, an optional
# k/m suffix. Deliberately NOT anchored to a currency — plenty of postings write '60-75K'.
_AMOUNT_RE = re.compile(r'\d[\d.,   ]*\d|\d')
_AMOUNT_WITH_SUFFIX_RE = re.compile(
    r'(?P<digits>\d[\d.,   ]*\d|\d)\s*(?P<suffix>[kKmM])?(?![\w])'
)
_THOUSANDS_SEPARATOR_RE = re.compile(rf'[.,   ](?=\d{{{THOUSANDS_GROUP_DIGITS}}}(?!\d))')

# Tier 2: the extractor writes these instead of leaving the field empty. Keeping this list is what
# preserves the 2026-09-15 lesson — 'Competitive salary' is an absence, not a figure.
_NO_FIGURE_VOCABULARY_RE = re.compile(
    r'\bnot\s+(?:specified|stated|disclosed|listed|provided|given|mentioned)\b'
    r'|\bno\s+(?:salary|compensation|pay)\b'
    r'|\bcompetitive\b|\bnegotiable\b|\bundisclosed\b|\bD\.?O\.?E\.?\b'
    r'|\bdepend\w*\s+on\s+experience\b|\bmarket\s+rate\b|\bTBD\b|\bN/?A\b',
    re.IGNORECASE,
)

# Tier 4: two amounts joined by a range token. The gap between them may carry a currency and
# spacing ('$190K/yr - $300K/yr', '90K EUR/yr - 110K EUR/yr', 'zł392,000 – zł588,000') but not
# another sentence, so it is bounded rather than open.
_RANGE_SEPARATOR = r'(?:-{1,2}|‐|‑|‒|–|—|―|~|\bto\b|\bup\s+to\b|\.{2,})'
_RANGE_GAP_MAX_CHARS = 14
_RANGE_RE = re.compile(
    r'(?P<low>\d[\d.,   ]*\d|\d)\s*(?P<low_suffix>[kKmM])?(?![A-Za-z])'
    rf'(?P<low_tail>[^\d\n]{{0,{_RANGE_GAP_MAX_CHARS}}}?)'
    + _RANGE_SEPARATOR +
    rf'(?P<gap>[^\d\n]{{0,{_RANGE_GAP_MAX_CHARS}}}?)'
    r'(?P<high>\d[\d.,   ]*\d|\d)\s*(?P<high_suffix>[kKmM])?(?![A-Za-z])',
    re.IGNORECASE,
)

# Amounts a posting states that are explicitly NOT pay. Without this, '€1,200/year training
# budget; flexible compensation package' reads as a single-figure salary — it carries a currency,
# a period and one amount, and every rule above says yes.
_NON_PAY_AMOUNT_RE = re.compile(
    r'\b(?:training|learning|education(?:al)?|development|equipment|wellness|home\s*office|'
    r'travel|relocation)\s+(?:budget|allowance|stipend|fund)\b',
    re.IGNORECASE,
)

# Tiers 5 and 6. Both are matched anywhere in the string rather than anchored at the start:
# 'Competitive salary starting from €70K gross annually' and 'Daily rate up to 600€' both put the
# marker in the middle. `_CEILING_RE` is tested FIRST, because 'up to' also matches nothing in
# `_FLOOR_RE` while 'from X up to Y' is already a range by tier 4.
_CEILING_RE = re.compile(
    r'\bup\s+to\b|\bmax(?:imum)?(?:\s+of)?\b|\bno\s+more\s+than\b|\bunder\b|\bbelow\b|\bat\s+most\b',
    re.IGNORECASE,
)
_FLOOR_RE = re.compile(
    r'\bfrom\b|\bstarting\s+(?:at|from)\b|\bmin(?:imum|\.)?\b|\bat\s+least\b|\bupwards\s+of\b'
    r'|\bstarts?\s+at\b|\bor\s+more\b|\d\s*\+',
    re.IGNORECASE,
)

# An amount labelled as the base, ahead of a bonus or on-target figure: '€135,000 base + €15,000
# bonus (OTE €150,000) per year' states one base salary. Three amounts and no direction marker
# left it `unclassified` twice, and the model agreed (2026-10-19, 2026-10-20).
_BASE_LABEL_RE = re.compile(r'\s*(?:base|basic)(?:\s+(?:salary|pay))?\b', re.IGNORECASE)

# Tier 7: a lone amount counts as a salary only when something marks it as money — a currency, a
# period, or a k/m magnitude. Without one, '12-month contract / Outside IR35' would read as pay.
_MONETARY_MARKER_RE = re.compile(
    r'[€£$₹¥]|\bzł\b|\bkr\b|\b(?:CAD|USD|EUR|GBP|PLN|CHF|SEK|NOK|DKK|CZK|HUF|RON|INR|JPY|SGD|BRL|MXN)\b'
    r'|\d\s*[kKmM](?![\w])|/\s*(?:yr|hr|hour|day|mo|month|week|wk)\b'
    r'|\bper\s+(?:year|month|week|day|hour|annum)\b|\bannual(?:ly)?\b|\bgross\b|\bsalary\b|\bbase\s+pay\b',
    re.IGNORECASE,
)

# Sentences of the description worth showing the classifier when the salary field itself has no
# figure. Selected in code, never by the model, and scanned over the whole description.
_COMPENSATION_SENTENCE_RE = re.compile(
    r'[€£$₹¥]|\bzł\b'
    r'|\b(?:CAD|USD|EUR|GBP|PLN|CHF|SEK|NOK|DKK|CZK|HUF|RON|INR|JPY|SGD|BRL|MXN)\b'
    r'|\bsalary\b|\bcompensation\b|\bbase\s+pay\b|\bremuneration\b|\bpay\s+range\b',
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
# A currency written against a bare zero is a price tag, not pay: LinkedIn's Premium upsell reads
# 'Activate Premium for CA$0 1-month free trial', which kept a page section marked for removal
# 11 times across 2026-09-24..28.
_ZERO_PRICE_RE = re.compile(rf'(?:{_CURRENCY_RE.pattern}|\$)\s?0(?![\d.,]*\d)', re.IGNORECASE)

_PROMPT = '''You are a compensation-string parser. Read ONLY the text below and report what
amounts it states. Do not estimate, do not infer a market rate, do not judge whether the pay is
good.

Salary field: "{text}"
{context}
Return ONLY a JSON object, no prose and no code fence:
{{"kind": "<absent|floor_only|ceiling_only|single|range|unclassified>",
  "minimum": <number or null>,
  "maximum": <number or null>,
  "currency": "<ISO code such as CAD, EUR, GBP, PLN, or empty>",
  "period": "<year|month|week|day|hour, or empty>",
  "quote": "<the words of the text above that state the amounts, copied as written>"}}

Rules:
- `range` ONLY when the text states BOTH a lower and an upper amount. Set both numbers.
- `floor_only` when it states a lower amount only ("from €70,000", "minimum 65000", "$30+/hour").
- `ceiling_only` when it states an upper amount only ("up to €50k", "no more than $200/hr").
- `single` when it states exactly one amount with no direction ("€110,000", "$100/hr").
- `absent` when the text says there is no figure ("competitive", "not specified", "negotiable").
- `unclassified` when the text states no compensation at all, or you cannot tell. Use it freely:
  a wrong `range` is far worse than an honest `unclassified`.
- Numbers are plain, with no separators and no currency: 208580, not "CA$208,580".
- Write amounts at full magnitude: "75K" is 75000, "€1.2M" is 1200000.
- A bare "$" is ambiguous. Leave `currency` empty unless the text names one.
- `period` is the pay period the amounts are quoted per, empty when the text does not say.
- When the text separates base pay from a bonus, commission or OTE (on-target earnings), report
  only the base amount(s); never the bonus or the OTE.
- `quote` is copied from the text above, unchanged. Code checks the amounts against it and
  discards an answer whose quote it cannot find in the text, so give no amount you cannot quote.
'''

_cache: dict[str, Any] | None = None


def _load_cache() -> dict[str, Any]:
    '''Read the on-disk cache. A missing or corrupt file is rebuilt, never fatal.'''
    global _cache
    if _cache is None:
        _cache = cache_io.load(SALARY_CACHE_PATH, 'salary cache')
    return _cache


def _write_cache() -> None:
    '''Persist the cache. A cache that cannot be written costs money, not correctness.'''
    cache_io.save(SALARY_CACHE_PATH, _load_cache(), 'salary cache')


def cache_key(text: str, context: str = '') -> str:
    '''Normalized cache key. Whitespace and case are not compensation.

    With a context excerpt the key is a hash of both, since the same empty-looking salary field can
    sit beside different descriptions — the `reference_summary_cache.yaml` shape.
    '''
    normalized = ' '.join(str(text or '').split()).lower()
    if not context:
        return normalized
    digest = hashlib.md5(f'{normalized}\n{context}'.encode()).hexdigest()
    return f'{normalized}#{digest}'


def _facts(kind: str, text: str, source: str, minimum: float | None = None,
           maximum: float | None = None, currency: str = '', period: str = '') -> dict[str, Any]:
    """One salary reading. The only place the contract's shape is written down."""
    return {
        'kind': kind if kind in SALARY_KINDS else 'unclassified',
        'minimum': minimum, 'maximum': maximum,
        'currency': currency, 'period': period,
        # The string as written. A value keeps its case — it is shown to the user verbatim.
        'text': str(text or '').strip(),
        'source': source,
    }


def parse_amount(raw: str, suffix: str = '') -> float | None:
    """'208,580' -> 208580.0, '75' + 'K' -> 75000.0, '46.50' -> 46.5.

    Raises nothing: an unparseable amount returns None and the caller decides, which is what keeps
    a malformed figure from being reported as a bound.
    """
    digits = _THOUSANDS_SEPARATOR_RE.sub('', str(raw or '').strip())
    digits = digits.replace(' ', '').replace(' ', '').replace(' ', '').replace(',', '.')
    try:
        value = float(digits)
    except ValueError:
        return None
    if suffix and suffix.lower() == 'k':
        value *= SALARY_THOUSAND_MULTIPLIER
    elif suffix and suffix.lower() == 'm':
        value *= SALARY_MILLION_MULTIPLIER
    return value


def currencies_named(text: str) -> tuple[str, ...]:
    """Every distinct currency the text names, as ISO codes in order of first appearance.

    More than one means the amounts are not all in one currency ('£60,000 or €70,000'), so a caller
    converting the bounds cannot assume the first currency belongs to them.
    """
    codes = (_CURRENCY_BY_FOLDED_TOKEN[m.group(0).casefold()] for m in _CURRENCY_RE.finditer(str(text or '')))
    return tuple(dict.fromkeys(codes))


def detect_currency(text: str) -> str:
    """The currency the text names, or '' when it names none or only a bare '$' or 'kr'."""
    named = currencies_named(text)
    return named[0] if named else ''


def detect_period(text: str) -> str:
    """The pay period the text quotes amounts per, or '' when it does not say."""
    for period, pattern in _PERIOD_PATTERNS:
        if pattern.search(str(text or '')):
            return period
    return ''


def classify_salary(text: str) -> dict[str, Any]:
    '''Read a compensation string with rules alone: sync, free, and never truncated.

    Deterministic checks scan the whole string (see the text-budget module docstring). Returning a
    plain dict rather than a dataclass keeps it writable straight into the YAML cache and into the
    extract, and keeps every consumer — warnings, notification, extract text — sync.
    '''
    raw = str(text or '')
    stripped = raw.strip()
    if not stripped:
        return _facts('absent', raw, 'deterministic')

    currency = detect_currency(stripped)
    period = detect_period(stripped)

    if not _AMOUNT_RE.search(stripped):
        # No digit at all. 'Competitive salary' is an absence; 'six figures' is a string no rule
        # here can read, and saying so is more useful than calling it an absence.
        kind = 'absent' if _NO_FIGURE_VOCABULARY_RE.search(stripped) else 'unclassified'
        return _facts(kind, raw, 'deterministic', currency=currency, period=period)

    if match := _RANGE_RE.search(stripped):
        low = parse_amount(match.group('low'), match.group('low_suffix') or '')
        high = parse_amount(match.group('high'), match.group('high_suffix') or '')
        # A bare 'K' on the low end of '60-75K' applies to both: nobody writes 60 to 75,000.
        if low is not None and high is not None and not match.group('low_suffix') and match.group('high_suffix'):
            low = parse_amount(match.group('low'), match.group('high_suffix'))
        if low is not None and high is not None and _MONETARY_MARKER_RE.search(stripped):
            return _facts('range', raw, 'deterministic', minimum=low, maximum=high,
                          currency=currency, period=period)

    amounts = [
        parse_amount(m.group('digits'), m.group('suffix') or '')
        for m in _AMOUNT_WITH_SUFFIX_RE.finditer(stripped)
    ]
    amounts = [a for a in amounts if a is not None]
    monetary = bool(_MONETARY_MARKER_RE.search(stripped))

    if monetary and amounts:
        # The FIRST amount, not the largest or smallest: a floor/ceiling marker qualifies the
        # amount it introduces, and a later figure is something else. 'Up to €135,000 Base + Bonus
        # (c.€150,000 OTE)' has a ceiling of 135,000 — the 150,000 is on-target earnings.
        if _CEILING_RE.search(stripped):
            return _facts('ceiling_only', raw, 'deterministic', maximum=amounts[0],
                          currency=currency, period=period)
        if _FLOOR_RE.search(stripped):
            return _facts('floor_only', raw, 'deterministic', minimum=amounts[0],
                          currency=currency, period=period)
        first = next(m for m in _AMOUNT_WITH_SUFFIX_RE.finditer(stripped)
                     if parse_amount(m.group('digits'), m.group('suffix') or '') is not None)
        if _BASE_LABEL_RE.match(stripped, first.end()) and not _NON_PAY_AMOUNT_RE.search(stripped):
            return _facts('single', raw, 'deterministic', minimum=amounts[0], maximum=amounts[0],
                          currency=currency, period=period)
        if len(amounts) == 1 and not _NON_PAY_AMOUNT_RE.search(stripped):
            return _facts('single', raw, 'deterministic', minimum=amounts[0], maximum=amounts[0],
                          currency=currency, period=period)

    # Tier 8, deliberately broad. '12-month contract / Outside IR35' and '€1,200/year training
    # budget; flexible compensation package' both land here, and both used to read as a salary.
    return _facts('unclassified', raw, 'deterministic', currency=currency, period=period)


def count_ranges(text: str) -> int:
    """How many low–high ranges the string states.

    A reading keeps the FIRST range only, so a string quoting one per region — 'CAD 154,700 to CAD
    204,700 (Ontario); CAD 154,700 to CAD 310,700 (British Columbia)' — has a `maximum` that is not
    the top of what the posting offers. A caller comparing bounds needs to know there is more.
    """
    return sum(1 for _ in _RANGE_RE.finditer(str(text or '')))


def compensation_context(description: str) -> str:
    '''Sentences of the description that mention money, for the classifier to read.

    Selected in code, so the model is handed evidence rather than asked to go looking. Only used
    when the salary field itself carries no figure — the case where the extractor may simply have
    missed a pay line that is sitting in the body.
    '''
    sentences = [
        ' '.join(s.split()) for s in _SENTENCE_SPLIT_RE.split(str(description or ''))
        if _COMPENSATION_SENTENCE_RE.search(s)
    ]
    if not sentences:
        return ''
    return truncate_reported(
        ' '.join(sentences), SALARY_CONTEXT_MAX_CHARS, 'salary context excerpt'
    )


def mentions_pay(text: str) -> bool:
    """True when the text carries pay wording once zero-amount prices ('CA$0') are set aside.

    A deterministic check over the WHOLE text: nothing is truncated or excerpted first.
    """
    return bool(_COMPENSATION_SENTENCE_RE.search(_ZERO_PRICE_RE.sub(' ', str(text or ''))))


def _coerce(raw: dict, text: str) -> dict[str, Any]:
    """Validate the model's JSON into the contract; raise ValueError on anything that does not fit.

    An out-of-vocabulary kind, period or currency, a non-number bound, or a `range` without two
    numbers raises rather than being coerced: the coerced answer used to be cached as if the model
    had given it, and nobody saw the reply. The caller falls back to `unclassified` — a warning
    that the text could not be read, never a silent upgrade — and does not cache it, so the log
    carries the reply and the string is asked again next time.
    """
    kind = str(raw.get('kind') or '').strip().lower()
    if kind not in SALARY_KINDS:
        raise ValueError(f'kind {raw.get("kind")!r} is not one of {sorted(SALARY_KINDS)}')
    period = str(raw.get('period') or '').strip().lower()
    if period and period not in SALARY_PERIODS:
        raise ValueError(f'period {raw.get("period")!r} is not one of {sorted(SALARY_PERIODS)}')
    currency = str(raw.get('currency') or '').strip().upper()
    if currency and currency not in SALARY_CURRENCIES:
        raise ValueError(f'currency {raw.get("currency")!r} is not a known currency code')

    def number(key: str) -> float | None:
        value = raw.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{key} {value!r} is not a number')
        return float(value)

    minimum, maximum = number('minimum'), number('maximum')
    # A `range` the model did not give two numbers for is not a range: this module must never
    # upgrade a partial to a full one.
    if kind == 'range' and (minimum is None or maximum is None):
        raise ValueError(f'kind is range but minimum={minimum} maximum={maximum}')
    return _facts(kind, text, 'llm', minimum=minimum, maximum=maximum,
                  currency=currency, period=period)


async def resolve_salary(extract: dict, stage_stats: dict | None = None) -> dict[str, Any]:
    '''The reading for one posting: rules first, the model only for what they could not read.

    Cached on disk, so a compensation phrase is parsed once ever and — more importantly — cannot be
    answered two different ways. Every failure falls back to the deterministic reading, which for
    an escalated string is `unclassified`: a warning that the text could not be read, and no gate.
    '''
    text = str(extract.get('salary') or '')
    deterministic = classify_salary(text)
    if deterministic['kind'] != 'unclassified':
        return deterministic

    # Only when no figure was found is the description worth reading: that is the case where the
    # extractor may have missed a pay line rather than the posting omitting one.
    context = '' if _AMOUNT_RE.search(text) else compensation_context(extract.get('description') or '')
    key = cache_key(text, context)
    cache = _load_cache()
    cached = cache.get(key)
    if cached and cached.get('schema') == CACHE_SCHEMA_VERSION:
        entry = _facts(
            cached.get('kind', 'unclassified'), text, 'cache',
            minimum=cached.get('minimum'), maximum=cached.get('maximum'),
            currency=cached.get('currency', ''), period=cached.get('period', ''),
        )
        return entry

    prompt = _PROMPT.format(
        text=text, context=f'\nCompensation lines from the posting body:\n"{context}"\n' if context else ''
    )
    try:
        reply = await chat_openrouter(prompt, model=MODEL_NAME_SALARY)
        result, quote = reply.parse(lambda raw: (_coerce(raw, text), str(raw.get('quote') or '')),
                                    f'salary classification of {text!r}')
    except Exception as ex:
        # Fail toward saying less. The deterministic reading is `unclassified`, which warns that
        # the string could not be read — never a silent upgrade to a complete range.
        logger.warning(
            f'Salary classification failed for {text!r}, leaving it unclassified: {unwrap_exception(ex)}'
        )
        return _facts('unclassified', text, 'error')

    if stage_stats is not None:
        stage_stats['cost'] = stage_stats.get('cost', 0.0) + reply.cost_usd
    if not await _cited_bounds_are_stated(result, quote, f'{text}\n{context}', stage_stats):
        # Not cached: a rejected answer is not an answer, and the next posting asks again.
        return _facts('unclassified', text, 'rejected')
    cache[key] = {
        'schema': CACHE_SCHEMA_VERSION, 'kind': result['kind'], 'quote': quote,
        'minimum': result['minimum'], 'maximum': result['maximum'],
        'currency': result['currency'], 'period': result['period'],
        'classified_on': date.today().isoformat(), 'model': reply.model, 'provider': reply.provider,
    }
    _write_cache()
    # DEBUG: the job's `Extract signal:` line already carries `->kind=…` and `salary_source=llm`.
    logger.debug(
        f'Salary classified: {text!r} -> kind={result["kind"]} minimum={result["minimum"]} '
        f'maximum={result["maximum"]} currency={result["currency"]!r} period={result["period"]!r} '
        f'({reply.describe()})'
    )
    return result


def salary_facts_of(extract: dict) -> dict[str, Any]:
    """The reading for an extract, classifying on the spot when the pipeline has not resolved one.

    Every consumer goes through here rather than reading `extract['salary']` directly, so a
    formatter or a warning can never re-derive the structure of a salary string its own way.
    """
    facts = extract.get('salary_facts')
    if isinstance(facts, dict) and facts.get('kind') in SALARY_KINDS:
        return facts
    return classify_salary(extract.get('salary') or '')


def salary_digits_are_in_the_source(salary_text: str, raw_text: str) -> bool:
    """True when every digit group of the salary string also appears in the retained page text.

    Provenance, not validation: `submit_job_extract` is generative, so a figure it hands back is
    only as good as the page it came from, and until raw postings were retained there was nothing
    to check one against.

    Amounts are compared as NUMBERS, not as strings: a page writing 'CA$208,580.00/yr' and an
    extract writing 'CA$208,580' state the same number, and a digit-string comparison would call
    that a mismatch and cry wolf on nearly every posting.
    """
    wanted = _amounts(salary_text)
    if not wanted:
        return True
    present = amounts_as_stated(raw_text)
    return all(value in present for value in wanted)


def _amounts(text: str) -> list[float]:
    parsed = (parse_amount(match) for match in _AMOUNT_RE.findall(str(text or '')))
    return [value for value in parsed if value is not None]


def amounts_as_stated(text: str) -> set[float]:
    """Every amount a text states, each both bare and with its k/m magnitude applied.

    A page writing '80-100K euros' and an extract writing '€80,000 - €100,000' state the
    same range; the 2026-10-14 run warned on exactly that twice. A range's shared suffix is
    carried to its low end, so '80-100K' yields 80000 as well as 80.
    """
    text = str(text or '')
    found = set(_amounts(text))
    for match in _AMOUNT_WITH_SUFFIX_RE.finditer(text):
        if (value := parse_amount(match['digits'], match['suffix'] or '')) is not None:
            found.add(value)
    for match in _RANGE_RE.finditer(text):
        suffix = match['low_suffix'] or match['high_suffix'] or ''
        if (value := parse_amount(match['low'], suffix)) is not None:
            found.add(value)
    return found


async def _cited_bounds_are_stated(result: dict[str, Any], quote: str, source: str,
                                   stage_stats: dict | None) -> bool:
    """Verify Source Reference for the classifier: the cited words must be in the text it read,
    and every bound it gave must be an amount those words state.

    A citation written in words ('eighty thousand euros') has no digits to compare, so it stands
    on being found; one with digits must contain each bound, as numbers.
    """
    bounds = [b for b in (result['minimum'], result['maximum']) if b is not None]
    if not bounds:
        return True
    claim = evidence.Claim('bounds', '-'.join(f'{b:g}' for b in bounds), quote)
    verdict = (await evidence.verify_claims(
        'salary_classify', repr(result['text']), [claim], source, 'salary string + compensation lines',
        stage_stats))['bounds']
    if not verdict.accepted:
        return False
    stated = amounts_as_stated(verdict.evidence)
    if not _amounts(verdict.evidence) or all(b in stated for b in bounds):
        return True
    evidence.count_rejection('salary_classify', 'bounds_not_in_quote')
    logger.warning(f'Hallucination rejected (salary_classify, {result["text"]!r}): bounds {bounds} are not '
                   f'amounts of the cited {verdict.evidence!r} (states {sorted(stated)}) — left unclassified')
    return False
