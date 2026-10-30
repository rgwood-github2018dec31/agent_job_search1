'''Ask Gate Fields: the extract fields that feed the gates, asked one at a time.

`relocation`, `residency_scope` and `language_requirement` decide rejections. Asked together with every
other field in the one extract call, the extractor gave a different value on a second run of the same
page on 7-25 of 243 pages (2026-10-30). Asked one at a time, with what code read off the page header
stated as facts, a few reading rules and an explicit "not known" answer, the same model changed on 0-5%
and agreed with a second model on all but 2 of 117 fields (scripts/local_field_probe.py, which tunes
these definitions and imports them from here).

Each answer is {quotes, reason, value}, in that order, so the model copies the page's words and says why
before it commits. A stated value stands only on quotes code finds in the page the model was shown
(Verify Source Reference); otherwise it is absent. What a value MEANS for the gates is decided here in
code, not left to the model: a remote job naming only an EU place leaves relocation empty, so the
location gate's EU rule applies (live anywhere in the EU); an office city is kept, since a hybrid or
on-site holder must live there.

The extractor still fills these fields, so a field whose calls all fail keeps the extractor's
verified value.
'''
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field

from agentic_job_search import evidence, job_trace
from agentic_job_search.config import GATE_FIELDS_ATTEMPTS, GATE_FIELDS_TEMPERATURE, MODEL_NAME_GATE_FIELDS
from agentic_job_search.linkedin_page import JobPage
from agentic_job_search.location import classify_location, is_eu_member
from agentic_job_search.triage import chat_openrouter, extract_json_object

logger = logging.getLogger(__name__)

GATE_FIELDS = ('relocation', 'residency_scope', 'language_requirement')
CALL_SITE = 'gate_fields'
SYSTEM = ('You read one job posting page and answer one question about it. Answer only from the page and '
          'the header facts given with it.')

# What a field is when the page does not say. Never '': an empty answer reads the same as no answer.
NOT_KNOWN = {
    'relocation': 'relocation_area_required_not_known',
    'residency_scope': 'residency_scope_not_known',
    'language_requirement': 'language_requirement_not_known',
}
RESIDENCY_SCOPES = ['city_only', 'country_only', 'area_wide']
# Why relocation has its value; code reads it (map_relocation). The numbers are the definition's rules.
RELOCATION_BASES = ['stated', 'remote_area', 'remote_place_only', 'office', 'anywhere']
# Rules every field reads the page by. Each came from a page the models misread (2026-10-30, 40 pages).
READING_RULES = (
    "How to read the page: (a) the posting's own words about where this job may be done beat the header "
    'location, which is only the one place LinkedIn lists: header "Canada" with "we hire across the United States '
    'and Canada" in the text means both; (b) when the header shows no workplace type and the page never says the '
    'job is remote or hybrid, the job is on-site; (c) company boilerplate describes the company, not this job: a '
    '"work from anywhere" culture, offices or countries the company is in, perks, and pay ranges or benefits '
    'listed per country never beat the header location or an explicit "based in"; only the posting saying this '
    'job may be done from anywhere does; (d) a time-zone requirement is not a place.')
PLACES_FORMAT = ('Write places as the page names them, countries in full English ("United States", not "US"), '
                 'the office city as the header names it ("Vancouver, BC", "Greater Toronto Area, Canada"), several '
                 'separated by "; ", with no qualifiers such as time zones.')
DEFINITIONS = {
    'relocation': (
        'Where the holder must live to do this job: a country, region or city, or several. Decide by these rules, '
        'in order, and give the rule as "basis": (1) "stated": the posting says where ("must be based in '
        'Portugal", "relocation to Madrid") -> that place; (2) "remote_area": the job is remote and names countries '
        'or an area it is open to ("remote in the EU", "Canada, United Kingdom") -> those; (3) "remote_place_only": '
        'the job is remote and the page names only a city, region or country, such as the header location '
        '"Vancouver, BC" -> that place\'s country ("Canada"); (4) "office": the job is on-site or hybrid -> the '
        "office's city, since the holder must live near enough to go there; (5) \"anywhere\": the posting says "
        'the holder may work from anywhere in the world -> "anywhere". When none of these settles it -> '
        f'"{NOT_KNOWN["relocation"]}" with basis "". {PLACES_FORMAT}'),
    'residency_scope': (
        'How wide the area the holder may live in is. "city_only" if the job is on-site or hybrid (the holder '
        "must live near the office's city), or the posting names one city to live in. \"country_only\" only if "
        'the posting says the holder must live in one country ("Remote within country", "must be based in '
        'Germany", "Canada-based position"); a remote job whose page merely names one place does not say this. '
        '"area_wide" if any country of a multi-country area or list will do ("remote anywhere in the EU", "any '
        f'EMEA country", "Canada, United Kingdom"), or anywhere in the world. "{NOT_KNOWN["residency_scope"]}" '
        'when the page does not settle it.'),
    'language_requirement': (
        'Languages the posting requires, comma-separated lowercase, e.g. "english, german". A language marked as '
        'an asset, a plus or nice to have is not required; a requirement may be implied ("Bilingualism (English '
        'and French). French is an asset but not required." -> "english"). '
        f'"{NOT_KNOWN["language_requirement"]}" when the posting names no language it requires.'),
}


def output_schema(field_name: str) -> dict:
    '''The reply's shape: the evidence and the reason come before the value (decoded in this order).'''
    value: dict = {'type': 'string'}
    if field_name == 'residency_scope':
        value['enum'] = [*RESIDENCY_SCOPES, NOT_KNOWN[field_name]]
    properties = {'quotes': {'type': 'array', 'items': {'type': 'string'}}, 'reason': {'type': 'string'},
                  'value': value}
    if field_name == 'relocation':
        properties['basis'] = {'type': 'string', 'enum': [*RELOCATION_BASES, '']}
    return {'type': 'object', 'properties': properties, 'required': list(properties)}


def header_facts(location: str, workplace: str) -> str:
    '''What code read off the header card, so the model is told rather than left to guess it.'''
    return (f'Header facts, read by code from the page header: location "{location or "not shown"}"; '
            f'workplace type "{workplace or "not shown"}".')


def question(page_text: str, facts: str, field_name: str) -> str:
    keys = '"quotes", "reason", "value"' + (', "basis"' if field_name == 'relocation' else '')
    return (f'--- PAGE ---\n{page_text}\n--- END PAGE ---\n\n{facts}\n\n{READING_RULES}\n\n'
            f'Field `{field_name}`: {DEFINITIONS[field_name]}\n'
            f'Reply with only a JSON object with these keys, in this order: {keys}. "quotes" is a list of the '
            'passages of the page that give the value, each copied exactly from one place on the page (for a header '
            'fact, the page\'s own words, such as "Vancouver, BC" or "On-site"), [] when nothing on the page states '
            'it; "reason" is one sentence on why those passages and the header facts give the value; "value" is the '
            'field\'s value, never "".')


@dataclass
class Answer:
    '''One field as the model answered it and as code read it.'''
    field: str
    raw_value: str = ''
    basis: str = ''
    quotes: list[str] = field(default_factory=list)
    quotes_not_found: list[str] = field(default_factory=list)
    reason: str = ''
    rejected: bool = False
    value: str = ''          # what the gates read
    attempt: int = 0
    error: str = ''          # set when every attempt failed; the extractor's value stands


def parse_answer(field_name: str, content: str) -> Answer:
    '''The reply as an Answer. Raises ValueError naming what is wrong; the caller adds the reply metadata.'''
    reply = extract_json_object(content)
    missing = [key for key in output_schema(field_name)['required'] if key not in reply]
    if missing:
        raise ValueError(f'reply lacks {missing}')
    quotes = reply['quotes']
    if not isinstance(quotes, list):
        raise ValueError(f'quotes is a {type(quotes).__name__}, not a list')
    raw_value = str(reply['value']).strip()
    allowed = output_schema(field_name)['properties']['value'].get('enum')
    if not raw_value or (allowed and raw_value not in allowed):
        raise ValueError(f'value {raw_value!r} is not allowed (empty, or not one of {allowed})')
    basis = str(reply.get('basis') or '').strip()
    if field_name == 'relocation' and basis not in [*RELOCATION_BASES, '']:
        raise ValueError(f'basis {basis!r} is not one of {RELOCATION_BASES}')
    return Answer(field_name, raw_value=raw_value, basis=basis, reason=str(reply['reason']).strip(),
                  quotes=[str(q) for q in quotes if str(q).strip()])


async def ask_field(field_name: str, page_text: str, facts: str, stage_stats: dict) -> Answer:
    '''One field from MODEL_NAME_GATE_FIELDS, asked again on a failed or malformed reply. Raises when every
    attempt failed, with each failure's reason.'''
    failures = []
    for attempt in range(1, GATE_FIELDS_ATTEMPTS + 1):
        try:
            reply = await chat_openrouter(question(page_text, facts, field_name), SYSTEM, MODEL_NAME_GATE_FIELDS,
                                          temperature=GATE_FIELDS_TEMPERATURE)
        except Exception as ex:
            failures.append(f'attempt {attempt}: {type(ex).__name__}: {ex}')
            continue
        stage_stats['cost'] = stage_stats.get('cost', 0.0) + reply.cost_usd
        try:
            answer = parse_answer(field_name, reply.content)
        except ValueError as ex:
            failures.append(f'attempt {attempt}: unusable reply ({ex}; {reply.describe()}): {reply.content!r}')
            continue
        answer.attempt = attempt
        if attempt > 1:
            logger.warning(f'Ask Gate Fields: {field_name} answered on attempt {attempt} after {failures}')
        return answer
    raise RuntimeError(f'{MODEL_NAME_GATE_FIELDS} {field_name}: every attempt failed: {failures}')


def ground(answer: Answer, page_text: str) -> None:
    '''Verify Source Reference for one answer: a stated value needs quotes, each found in the page.'''
    if answer.raw_value == NOT_KNOWN[answer.field]:
        return
    answer.quotes_not_found = [q for q in answer.quotes if evidence.find_quote(q, page_text) is None]
    answer.rejected = not answer.quotes or bool(answer.quotes_not_found)


async def map_relocation(answer: Answer) -> str:
    '''What the gates read for relocation. A remote job naming only EU places is no residence requirement:
    the location gate's EU rule lets the holder live anywhere in the EU (residency_spare 'eu_only').'''
    if answer.basis == 'anywhere' or answer.raw_value.casefold() == 'anywhere':
        return ''
    if answer.basis == 'remote_place_only':
        countries = (await classify_location(answer.raw_value)).get('countries') or []
        if countries and all(is_eu_member(country) for country in countries):
            return ''
    return answer.raw_value


async def map_value(answer: Answer) -> str:
    '''What the gates read, decided in code from the model's answer.'''
    if answer.rejected or answer.raw_value == NOT_KNOWN[answer.field]:
        return ''
    if answer.field == 'relocation':
        return await map_relocation(answer)
    if answer.field == 'residency_scope':
        # An office pins residence to its city; the gates know the country level, which it implies.
        return 'country_only' if answer.raw_value == 'city_only' else answer.raw_value
    return ', '.join(part.strip().lower() for part in answer.raw_value.split(',') if part.strip())


async def ask_gate_fields(candidate: dict, extract: dict, page: JobPage, stage_stats: dict) -> dict[str, Answer]:
    '''Ask each GATE_FIELDS field on its own and write what the gates read into `extract`, in place.

    A field whose every attempt failed keeps the extractor's verified value, logged with the failures.
    Returns field -> Answer, also recorded as the `gate_fields` Job Trace step.
    '''
    page_text = page.model_text
    facts = header_facts(page.header.location or page.listed_location, page.header.workplace)
    subject = f"job {candidate.get('job_id')} {candidate.get('company')}"
    results = await asyncio.gather(*(ask_field(f, page_text, facts, stage_stats) for f in GATE_FIELDS),
                                   return_exceptions=True)
    answers = {}
    for field_name, result in zip(GATE_FIELDS, results, strict=True):
        if isinstance(result, BaseException):
            answer = Answer(field_name, value=str(extract.get(field_name) or ''), error=f'{type(result).__name__}: {result}')
            logger.warning(f"Ask Gate Fields ({subject}): {field_name} not asked, the extractor's "
                           f'{answer.value!r} stands: {answer.error}')
            answers[field_name] = answer
            continue
        answer = result
        ground(answer, page_text)
        if answer.rejected:
            evidence.count_rejection(CALL_SITE, field_name)
            logger.warning(
                f'Hallucination rejected ({CALL_SITE}, {subject}): {field_name}={answer.raw_value!r} cited '
                f'{answer.quotes!r}; not found in the page ({len(page_text)} chars): {answer.quotes_not_found!r} '
                '— treated as absent')
        answer.value = await map_value(answer)
        extract[field_name] = answer.value
        answers[field_name] = answer
    job_trace.record('gate_fields', facts=facts, answers={f: asdict(a) for f, a in answers.items()})
    logger.debug(f'Ask Gate Fields ({subject}): '
                 + json.dumps({f: [a.raw_value, a.basis, a.value] for f, a in answers.items()}, ensure_ascii=False))
    return answers
