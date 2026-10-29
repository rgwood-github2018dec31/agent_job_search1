'''Verify Source Reference: a model's prediction stands only on text found in its input.

Every LLM call that predicts a FACT about its input (the extractor's fields, a salary's bounds, a
place's country, a rater's warning) also cites the input text that supports it. Code then looks
for that citation in the input:

1. Regex search, after normalisation (case, whitespace, dashes, quotes, an ellipsis splitting the
   quote into ordered pieces). Found: accepted.
2. Only for what regex did not find: one batched call to MODEL_NAME_EVIDENCE, which is shown the
   source and the claims and must copy, for each claim, the span of the source that states it.
   That span is then searched for by regex, exactly like a citation. The model only POINTS; a
   span it names that is not in the source proves nothing. This is what rescues a citation a
   model tidied, translated or paraphrased.
3. Still not found: rejected as a hallucination. The caller treats the field as absent/unknown,
   never as a negative answer, so a rejected prediction can make a rule pass, never fire.

A failure of the step-2 call rejects every claim it was asked about: a prediction nobody could
ground does not get to gate a job.
'''

import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from agentic_job_search import job_trace
from agentic_job_search.config import EVIDENCE_MIN_QUOTE_CHARS, MODEL_NAME_EVIDENCE
from agentic_job_search.triage import chat_openrouter

logger = logging.getLogger(__name__)

# The sibling field a schema adds for each predicted field: `location` -> `location_quote`.
QUOTE_SUFFIX = '_quote'

STATUS_REGEX = 'regex'
STATUS_RELOCATED = 'relocated_by_llm'
STATUS_REJECTED = 'rejected'

_DASHES = dict.fromkeys(map(ord, '‐‑‒–—―−'), '-')
_QUOTES = {**dict.fromkeys(map(ord, '‘’‚‛′'), "'"),
           **dict.fromkeys(map(ord, '“”„‟″'), '"')}
_WHITESPACE_RE = re.compile(r'\s+')
_ELLIPSIS_RE = re.compile(r'\s*(?:…|\.{3,})\s*')
_EDGE_CHARS = ' \t\n.,;:!?"\'()[]{}*-'


def normalize_for_match(text: object) -> str:
    '''Fold what a faithful citation may legitimately differ in: case, width, whitespace, dash and
    quote styles. Letters, digits and word order are left alone.'''
    folded = unicodedata.normalize('NFKC', str(text or '')).translate(_DASHES).translate(_QUOTES)
    return _WHITESPACE_RE.sub(' ', folded.casefold()).strip()


def find_quote(quote: object, source: object) -> tuple[int, int] | None:
    '''Where `quote` occurs in `source` (both normalised), or None.

    An ellipsis in the quote splits it into pieces that must occur in order, so a model may elide
    the middle of a long sentence. Each piece must start and end on a word boundary, so 'UK' is
    not found inside 'Ukraine'. Positions index the normalised source.
    '''
    pieces = [piece.strip(_EDGE_CHARS) for piece in _ELLIPSIS_RE.split(normalize_for_match(quote))]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return None
    text = normalize_for_match(source)
    start = end = None
    for piece in pieces:
        pattern = (r'(?<!\w)' if piece[0].isalnum() else '') + re.escape(piece) + (r'(?!\w)' if piece[-1].isalnum() else '')
        found = re.search(pattern, text[end or 0:])
        if found is None:
            return None
        piece_start = (end or 0) + found.start()
        start = piece_start if start is None else start
        end = piece_start + len(piece)
    return start, end


def quote_is_specific(quote: object, value: object, min_chars: int = EVIDENCE_MIN_QUOTE_CHARS) -> bool:
    '''False for a citation too short to prove anything ('a', 'EU') unless it is the value itself.

    A caller whose source is itself a few words (a location string) passes a lower `min_chars`:
    there 'UK' is the whole of the evidence, and word boundaries keep it from matching 'Ukraine'.
    '''
    normalised = normalize_for_match(quote).strip(_EDGE_CHARS)
    return bool(normalised) and (
        len(normalised) >= min_chars or normalised == normalize_for_match(value).strip(_EDGE_CHARS))


@dataclass(frozen=True)
class Claim:
    '''One prediction to ground: `field` predicted as `value`, citing `quote` (may be empty).'''

    field: str
    value: Any
    quote: str = ''


@dataclass(frozen=True)
class Verdict:
    '''What verification decided for one Claim, with the evidence it decided on.'''

    claim: Claim
    status: str
    span: tuple[int, int] | None = None
    evidence: str = ''
    detail: str = ''

    @property
    def accepted(self) -> bool:
        return self.status != STATUS_REJECTED


# Rejections this run, for the run audit: (call_site, field) -> count. Reset at the top of a run.
_rejections: Counter = Counter()


def reset_rejections() -> None:
    _rejections.clear()


def count_rejection(call_site: str, field: str) -> None:
    '''Count a rejection decided by a caller's own check (e.g. salary amounts compared as numbers).'''
    _rejections[(call_site, field)] += 1


def rejection_counts() -> dict[tuple[str, str], int]:
    return dict(_rejections)


def _grounded(claim: Claim, source: str, min_chars: int) -> tuple[int, int] | None:
    if not quote_is_specific(claim.quote, claim.value, min_chars):
        return None
    return find_quote(claim.quote, source)


_RELOCATE_PROMPT = '''Below is a SOURCE text and a list of CLAIMS another model made about it. Each
claim has a field, the value predicted, and the words it cited (which may be wrong, paraphrased,
translated or empty).

For EACH claim, find the passage of the SOURCE that states it, and copy that passage EXACTLY as it
is written in the SOURCE: same language, same words, no paraphrase, at most two sentences. If the
SOURCE does not state the claim, give an empty string. Never invent a passage.

Return ONLY a JSON object, no prose and no code fence:
{{"spans": {{"<claim id>": "<passage copied from the SOURCE, or empty>", ...}}}}

CLAIMS:
{claims}

--- SOURCE ---
{source}
--- END SOURCE ---
'''


async def _relocate(call_site: str, subject: str, claims: list[Claim], source: str,
                    stage_stats: dict | None) -> tuple[dict[str, str], str]:
    '''Ask MODEL_NAME_EVIDENCE for the source span behind each claim. Returns (id -> span, reply note).'''
    listed = json.dumps(
        [{'id': str(i), 'field': c.field, 'value': c.value, 'cited': c.quote} for i, c in enumerate(claims)],
        ensure_ascii=False, indent=1, default=str,
    )
    reply = await chat_openrouter(_RELOCATE_PROMPT.format(claims=listed, source=source), model=MODEL_NAME_EVIDENCE)
    if stage_stats is not None:
        stage_stats['cost'] = stage_stats.get('cost', 0.0) + reply.cost_usd

    def coerce(obj: dict) -> dict[str, str]:
        spans = obj['spans']
        if not isinstance(spans, dict):
            raise TypeError(f'"spans" is {type(spans).__name__}, not an object')
        return {str(k): str(v or '') for k, v in spans.items()}

    spans = reply.parse(coerce, what=f'evidence relocation for {call_site} ({subject})')
    return spans, reply.describe()


async def verify_claims(
    call_site: str, subject: str, claims: list[Claim], source: str, source_label: str,
    stage_stats: dict | None = None, use_llm: bool = True, min_quote_chars: int = EVIDENCE_MIN_QUOTE_CHARS,
) -> dict[str, Verdict]:
    '''Ground each claim in `source`; returns field -> Verdict.

    `call_site` names the LLM call whose predictions these are (for the log and the run audit),
    `subject` the job or input they are about, `source_label` where `source` came from (a path
    when there is one), so a rejection can be checked from the log line alone.
    '''
    verdicts: dict[str, Verdict] = {}
    missed: list[Claim] = []
    for claim in claims:
        span = _grounded(claim, source, min_quote_chars)
        if span is not None:
            verdicts[claim.field] = Verdict(claim, STATUS_REGEX, span, claim.quote)
        else:
            missed.append(claim)

    if missed and use_llm:
        try:
            spans, reply_note = await _relocate(call_site, subject, missed, source, stage_stats)
        except Exception as ex:
            spans, reply_note = {}, f'relocation call failed: {type(ex).__name__}: {ex}'
            logger.warning(f'Evidence ({call_site}, {subject}): {reply_note}; '
                           f'rejecting the {len(missed)} claim(s) it was asked to ground')
        for i, claim in enumerate(missed):
            relocated = spans.get(str(i), '')
            span = find_quote(relocated, source) if quote_is_specific(relocated, claim.value, min_quote_chars) else None
            if span is not None:
                verdicts[claim.field] = Verdict(claim, STATUS_RELOCATED, span, relocated, reply_note)
            else:
                verdicts[claim.field] = Verdict(claim, STATUS_REJECTED, None, relocated, reply_note)
    else:
        for claim in missed:
            verdicts[claim.field] = Verdict(claim, STATUS_REJECTED, None, '', 'LLM relocation not used')

    for verdict in verdicts.values():
        if verdict.accepted:
            continue
        claim = verdict.claim
        _rejections[(call_site, claim.field)] += 1
        logger.warning(
            f'Hallucination rejected ({call_site}, {subject}): {claim.field}={claim.value!r} '
            f'cited {claim.quote!r}, which is not in {source_label} ({len(source)} chars); '
            f'relocation offered {verdict.evidence!r} [{verdict.detail}] — treated as absent'
        )
    _record_trace(call_site, subject, source_label, verdicts)
    return verdicts


CITATION_INSTRUCTION = (
    'For EVERY field you fill other than description, also fill the matching <field>_quote with '
    'the words of the source text that state it, copied as written there (same language, no '
    'paraphrase; "…" may join two pieces of one passage). Code looks each quote up in the source and '
    'DISCARDS a field whose quote it cannot find, so a field you cannot quote is better left empty.'
)


def quote_property(name: str) -> dict:
    '''The `<field>_quote` schema property for field `name`.'''
    return {'type': 'string',
            'description': f'Words of the source text that state {name}, copied as written; required whenever {name} is filled'}


def with_quotes(properties: dict[str, dict], fields: tuple[str, ...]) -> dict[str, dict]:
    '''`properties` with each cited field's `<field>_quote` placed right after the field.

    A model fills a schema in order. With every quote after every field (until 2026-10-28) it
    reached the quotes last and skipped them: Hyra's extract cited none of its six fields, so each
    went to an LLM relocation call (4,759 output tokens) and two were rejected.
    '''
    missing = [name for name in fields if name not in properties]
    if missing:
        raise ValueError(f'with_quotes: cited fields {missing} are not in the schema properties {list(properties)}')
    out: dict[str, dict] = {}
    for name, spec in properties.items():
        out[name] = spec
        if name in fields:
            out[f'{name}{QUOTE_SUFFIX}'] = quote_property(name)
    return out


def is_filled(value: object) -> bool:
    '''Whether a field value is a claim that needs a citation: True, or a non-empty string.'''
    return value is True or (isinstance(value, str) and bool(value.strip()))


def unquoted_fields(args: dict, fields: tuple[str, ...]) -> list[str]:
    '''The cited fields a model filled in `args` without their `<field>_quote`.'''
    quotes = quotes_from(args, fields)
    return [name for name in fields if is_filled(args.get(name)) and not quotes[name]]


def quotes_from(args: dict, fields: tuple[str, ...]) -> dict[str, str]:
    '''The citations a model returned for `fields`, '' where it gave none.'''
    return {name: str(args.get(f'{name}{QUOTE_SUFFIX}') or '').strip() for name in fields}


def _record_trace(call_site: str, subject: str, source_label: str, verdicts: dict[str, Verdict]) -> None:
    '''Hand the verdicts to the current job's trace, when one is being recorded.'''
    job_trace.record('evidence', call_site=call_site, subject=subject, source=source_label, verdicts=[
        {'field': v.claim.field, 'value': v.claim.value, 'quote': v.claim.quote, 'status': v.status,
         'span': list(v.span) if v.span else None, 'evidence': v.evidence, 'detail': v.detail}
        for v in verdicts.values()
    ])
