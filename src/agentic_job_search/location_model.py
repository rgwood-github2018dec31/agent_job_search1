'''Job Location: where a job is done and where its holder may be employed, then whether that suits the user.

Replaces `relocation` + `residency_scope` + the anchor location as the geographic gate's input
(docs/requirements.md: Ask Location Model, Decide Location Eligibility, Adjudicate Location). Until it is
wired in, only scripts/replay_location_model.py calls it.

A job has one or more OPTIONS ("office in Leuven OR remote from any European country"). Each option has
a workplace type, the places the holder must attend (working locations) and the places the company can
employ the holder in (employment locations). The model reads these off the page with a quote per entry;
code verifies each quote and decides. A defaulted value is code's, from the header, never the model's:

- A remote job whose page states no employment place is employable across the header's country; an EU
  country widens to the whole EU (free movement). (User, 2026-11-06.)
- Remote employment in a country outside the user's work rights is fine unless the country needs a visa
  for remote work (`sponsorship_required_in`) or the page restricts the job to it.

What cannot be decided from verified facts -- no place resolves, the page contradicts itself, a stated
place lost its quotes -- is UNDETERMINED, never quietly eligible: the 2026-11-02 run lost a correct
relocation to one mistyped quote, and an empty field there reads as "no requirement".
'''
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import asdict, dataclass, field

from agentic_job_search import evidence, preferences
from agentic_job_search.config import GATE_FIELDS_ATTEMPTS, GATE_FIELDS_TEMPERATURE, MODEL_NAME_GATE_FIELDS
from agentic_job_search.gate_fields import header_facts
from agentic_job_search.location import EU_MEMBER_STATES, classify_location, location_token_matches
from agentic_job_search.triage import chat_openrouter, extract_json_object

logger = logging.getLogger(__name__)

CALL_SITE = 'location_model'
ADJUDICATE_CALL_SITE = 'location_adjudicate'
MODEL_NAME_LOCATION = MODEL_NAME_GATE_FIELDS

WORKPLACE_TYPES = ('remote', 'hybrid', 'onsite', 'unknown')
ATTENDANCE = ('regular', 'occasional')
REQUIREMENT = ('required', 'preferred')
SPONSORSHIP = ('offered', 'refused', 'case_by_case', 'unknown')
ENGAGEMENT = ('employee', 'contractor', 'eor', 'unknown')

ELIGIBLE, ELIGIBLE_CAPPED, INELIGIBLE, UNDETERMINED = 'eligible', 'eligible_capped', 'ineligible', 'undetermined'
VERDICT_RANK = {ELIGIBLE: 0, ELIGIBLE_CAPPED: 1, UNDETERMINED: 2, INELIGIBLE: 3}

# Areas a posting names, as country sets. World knowledge, so tracked; folded only to look up.
EEA_NON_EU = frozenset({'Iceland', 'Liechtenstein', 'Norway'})
EU_FREE_MOVEMENT = EU_MEMBER_STATES | EEA_NON_EU | {'Switzerland'}
EUROPE = EU_FREE_MOVEMENT | {
    'United Kingdom', 'Albania', 'Andorra', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Monaco',
    'Montenegro', 'North Macedonia', 'San Marino', 'Serbia', 'Ukraine', 'Vatican City',
}
ANYWHERE = 'anywhere'
# Matched inside the place as the page wrote it ("any European country (incl. the UK)"), most specific
# first: "European Union" before "European". Checked before the classifier, which has no notion of an area.
AREA_PATTERNS: tuple[tuple[re.Pattern[str], frozenset[str] | str], ...] = (
    (re.compile(r'\b(?:anywhere|worldwide|world-wide|global(?:ly)?)\b', re.IGNORECASE), ANYWHERE),
    # A country left to the holder ("from anywhere within your country of residence", Weflow 4483924663).
    (re.compile(r'\b(?:your|their|own|home)\s+country\b|\bcountry\s+of\s+residence\b', re.IGNORECASE), ANYWHERE),
    (re.compile(r'\b(?:EEA|European Economic Area)\b'), EU_MEMBER_STATES | EEA_NON_EU),
    (re.compile(r'\b(?:EU|European Union)\b(?!\s*time)'), EU_MEMBER_STATES),
    (re.compile(r'\b(?:Europe|European|EMEA)\b', re.IGNORECASE), EUROPE),
)
# How a right to work in an area reads: an EU citizen may work across the free-movement area.
WORK_RIGHT_AREAS: dict[str, frozenset[str]] = {
    'european union': EU_FREE_MOVEMENT, 'eu': EU_FREE_MOVEMENT, 'eea': EU_FREE_MOVEMENT,
}

SYSTEM = ('You read one job posting page and describe where the job is done and where its holder may be '
          'employed. Answer only from the page and the header facts given with it.')
READING_RULES = (
    "How to read the page: (a) the posting's own words about where this job may be done beat the header "
    'location, which is only the one place LinkedIn lists; (b) when the header shows no workplace type and the '
    'page never says the job is remote or hybrid, the job is on-site; (c) company boilerplate describes the '
    'company, not this job: a "work from anywhere" culture, offices or countries the company is in, perks, and '
    'pay ranges or benefits listed per country are not employment locations; (d) a time-zone requirement is not '
    'a place -- put it in "timezone_constraint"; (e) a recruiter or aggregator rewriting another company\'s job '
    "may carry that company's boilerplate: read what it says about THIS job.")
DEFINITION = '''Describe the job as a JSON object with these keys, in this order:

"options": a list of the ways the job may be done. Usually one. Several only when the page offers
  alternatives ("office in Leuven or fully remote from any European country", "remote in Portugal; hybrid in
  Poland", "London-based hybrid, elsewhere in the UK remote"). Each option is an object:
  "workplace_type": one of "remote", "hybrid", "onsite", "unknown".
  "workplace_quotes": passages giving the workplace type (the header badge counts: quote it as written).
  "working_locations": places the holder must physically attend, for hybrid or on-site work and for
    required visits. Travel to customers or events is not a working location. Each {"place": ...,
    "attendance": "regular" (hybrid or on-site days) or "occasional"
    (visits a few times a quarter or year), "quotes": [...]}. [] for a remote job with no visits.
  "employment_locations": places the company will employ the holder in, i.e. where the holder must live or be
    authorised to work ("must be based in Portugal", "remote within the EU", "right to work in the UK", "we hire
    in Canada and the United States", "Remote - Ontario only"). Each {"place": ..., "quotes": [...]}. Leave it
    [] when the page says nothing about where the holder must be: code then reads the header location. The
    header location, the job title or a team's location ("join our team in Tampere") repeated is not a
    statement of where the holder must be.
    "anywhere" when the page says the job may be done from anywhere in the world.
  "requirement": "required", or "preferred" when the page only prefers this option ("preference for
    Vancouver-based candidates").
"sponsorship": {"value": "offered" | "refused" | "case_by_case" | "unknown", "quotes": [...]} -- visa or
  work-permit sponsorship.
"engagement": {"value": "employee" | "contractor" | "eor" | "unknown", "quotes": [...]} -- "eor" when the page
  says it employs through an Employer of Record.
"timezone_constraint": {"value": the constraint as written, or "", "quotes": [...]}.
"contradiction": {"value": one sentence when the page's OWN WORDS state two incompatible answers about where the
  job is done ("Remote - Ontario only" and "fully remote across Canada and the United States"), else "",
  "quotes": [...]}. The header badge or location disagreeing with the page's words is not a contradiction: the
  page's words win (rule a). Nor is a job title naming a country, or a benefit describing the company's general
  hybrid policy.

Places: as the page names them, countries in full English ("United States", not "US"); areas as named
("European Union", "EMEA", "Europe"). Every quote is copied exactly from one place on the page or the header
facts, never joined from fragments. Reply with only the JSON object.'''


# Words a quote must hold beyond the place names and workplace badges to state a restriction:
# "based in Poland" and "Residency in Poland" do (2026-11-06 replay); "Poland" and "Remote (Germany)" do not.
STATEMENT_MIN_EXTRA_WORDS = 2
_BADGE_WORDS_RE = re.compile(r'\b(?:remote|hybrid|on-?site|full-?time|part-?time|contract)\b', re.IGNORECASE)
_WORD_RE = re.compile(r'[^\W\d_]+')


def _residual_words(quote: str, *names: str) -> int:
    text = _BADGE_WORDS_RE.sub(' ', quote)
    for name in names:
        for part in re.split(r'[,|;()]', name or ''):
            if part.strip():
                text = re.sub(re.escape(part.strip()), ' ', text, flags=re.IGNORECASE)
    return len(_WORD_RE.findall(text))


@dataclass
class Entry:
    '''One place the model read off the page, with what verification found of its quotes.'''
    place: str
    quotes: list[str] = field(default_factory=list)
    attendance: str = ''
    quotes_not_found: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return bool(self.quotes) and len(self.quotes_not_found) < len(self.quotes)

    def says_more_than(self, *names: str) -> bool:
        '''Whether a found quote states something about the place, beyond naming it.

        "Poland" or "Berlin, Germany | Remote | Full-time" quoted as an employment place is the header
        repeated, not a restriction; read as one, it switched off the silent-page widening and was the
        main cause of a verdict changing between two runs of the same page (2026-11-06 replay).
        '''
        return any(_residual_words(quote, self.place, *names) >= STATEMENT_MIN_EXTRA_WORDS
                   for quote in self.quotes if quote not in self.quotes_not_found)


@dataclass
class Option:
    workplace_type: str
    workplace_quotes: list[str] = field(default_factory=list)
    working_locations: list[Entry] = field(default_factory=list)
    employment_locations: list[Entry] = field(default_factory=list)
    requirement: str = 'required'


@dataclass
class JobLocation:
    options: list[Option]
    sponsorship: str = 'unknown'
    engagement: str = 'unknown'
    timezone_constraint: str = ''
    contradiction: str = ''
    header_location: str = ''
    header_workplace: str = ''
    rejected_quotes: list[str] = field(default_factory=list)


@dataclass
class Decision:
    verdict: str
    reason: str
    option_verdicts: list[tuple[str, str]] = field(default_factory=list)
    adjudicated: dict | None = None


def _quotes(value: object) -> list[str]:
    return [str(q) for q in value if str(q).strip()] if isinstance(value, list) else []


def _choice(value: object, allowed: tuple[str, ...], default: str, what: str) -> str:
    text = str(value or '').strip() or default
    if text not in allowed:
        raise ValueError(f'{what} {text!r} is not one of {allowed}')
    return text


def parse_reply(content: str, header_location: str, header_workplace: str) -> JobLocation:
    '''The model's reply as a JobLocation. Raises ValueError naming what is wrong.'''
    reply = extract_json_object(content)
    raw_options = reply.get('options')
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError(f'"options" is {raw_options!r}, not a non-empty list')
    options = []
    for index, raw in enumerate(raw_options):
        if not isinstance(raw, dict):
            raise ValueError(f'option {index} is a {type(raw).__name__}, not an object')

        def entries(key: str, raw: dict = raw, index: int = index) -> list[Entry]:
            items = raw.get(key) or []
            if not isinstance(items, list):
                raise ValueError(f'option {index} {key} is a {type(items).__name__}, not a list')
            return [Entry(str(item.get('place') or '').strip(), _quotes(item.get('quotes')),
                          str(item.get('attendance') or '').strip())
                    for item in items if isinstance(item, dict) and str(item.get('place') or '').strip()]

        options.append(Option(
            workplace_type=_choice(raw.get('workplace_type'), WORKPLACE_TYPES, 'unknown', f'option {index} workplace_type'),
            workplace_quotes=_quotes(raw.get('workplace_quotes')),
            working_locations=entries('working_locations'),
            employment_locations=entries('employment_locations'),
            requirement=_choice(raw.get('requirement') or 'required', REQUIREMENT, 'required', f'option {index} requirement'),
        ))

    def sub(key: str) -> dict:
        value = reply.get(key) or {}
        return value if isinstance(value, dict) else {}

    return JobLocation(
        options=options,
        sponsorship=_choice(sub('sponsorship').get('value') or 'unknown', SPONSORSHIP, 'unknown', 'sponsorship'),
        engagement=_choice(sub('engagement').get('value') or 'unknown', ENGAGEMENT, 'unknown', 'engagement'),
        timezone_constraint=str(sub('timezone_constraint').get('value') or '').strip(),
        contradiction=str(sub('contradiction').get('value') or '').strip(),
        header_location=header_location, header_workplace=header_workplace,
    )


def question(page_text: str, facts: str) -> str:
    return f'--- PAGE ---\n{page_text}\n--- END PAGE ---\n\n{facts}\n\n{READING_RULES}\n\n{DEFINITION}'


async def ground(job: JobLocation, source: str, subject: str, stage_stats: dict | None) -> None:
    '''Verify every entry's quotes against the page (regex first, then the evidence model's relocation).

    A quote not found is dropped from its entry, never the whole job: one mistyped quote of three
    dropped a correct relocation on 2026-11-02. An entry with no quote left is unverified, and
    `decide` treats a stated place it cannot verify as undetermined, not as unstated.
    '''
    claims, owners = [], []
    for o_index, option in enumerate(job.options):
        for kind in ('working_locations', 'employment_locations'):
            for e_index, entry in enumerate(getattr(option, kind)):
                for q_index, quote in enumerate(entry.quotes):
                    claims.append(evidence.Claim(f'{o_index}.{kind}.{e_index}.{q_index}', entry.place, quote))
                    owners.append((entry, quote))
    if not claims:
        return
    verdicts = await evidence.verify_claims(CALL_SITE, subject, claims, source, 'job page',
                                            stage_stats=stage_stats)
    for claim, (entry, quote) in zip(claims, owners, strict=True):
        if not verdicts[claim.field].accepted:
            entry.quotes_not_found.append(quote)
            job.rejected_quotes.append(f'{entry.place}: {quote}')


async def ask(page_text: str, header_location: str, header_workplace: str, subject: str,
              stage_stats: dict | None = None) -> JobLocation:
    '''One Job Location from MODEL_NAME_LOCATION, asked again on a failed or malformed reply.'''
    facts = header_facts(header_location, header_workplace)
    failures = []
    for attempt in range(1, GATE_FIELDS_ATTEMPTS + 1):
        try:
            reply = await chat_openrouter(question(page_text, facts), SYSTEM, MODEL_NAME_LOCATION,
                                          temperature=GATE_FIELDS_TEMPERATURE)
        except Exception as ex:
            failures.append(f'attempt {attempt}: {type(ex).__name__}: {ex}')
            continue
        if stage_stats is not None:
            stage_stats['cost'] = stage_stats.get('cost', 0.0) + reply.cost_usd
        try:
            job = parse_reply(reply.content, header_location, header_workplace)
        except ValueError as ex:
            failures.append(f'attempt {attempt}: unusable reply ({ex}; {reply.describe()}): {reply.content!r}')
            continue
        await ground(job, f'{page_text}\n\n{facts}', subject, stage_stats)
        return job
    raise RuntimeError(f'Ask Location Model ({subject}, {MODEL_NAME_LOCATION}): every attempt failed: {failures}')


# --- Decide Location Eligibility -------------------------------------------------------------------

@dataclass
class Places:
    countries: frozenset[str] = frozenset()
    anywhere: bool = False
    unresolved: tuple[str, ...] = ()


async def resolve(place: str) -> Places:
    '''The countries a place name covers: a known area (AREA_PATTERNS), else the location classifier.'''
    areas = [countries for pattern, countries in AREA_PATTERNS if pattern.search(place)]
    if ANYWHERE in areas:
        return Places(anywhere=True)
    if areas:
        # "UK and EU": the area, plus any country also named in the same place.
        named = frozenset((await classify_location(place)).get('countries') or [])
        return Places(countries=areas[0] | named)
    countries = frozenset((await classify_location(place)).get('countries') or [])
    return Places(countries=countries) if countries else Places(unresolved=(place,))


async def resolve_all(places: list[str]) -> Places:
    resolved = await asyncio.gather(*(resolve(p) for p in places))
    return Places(countries=frozenset().union(*(r.countries for r in resolved)),
                  anywhere=any(r.anywhere for r in resolved),
                  unresolved=tuple(u for r in resolved for u in r.unresolved))


@dataclass
class Profile:
    '''The user's side, as country sets: may work in, would live in, would not live in, needs a visa in.'''
    work_rights: frozenset[str]
    live: frozenset[str]
    not_live: frozenset[str]
    visa: frozenset[str]
    commute_places: tuple[str, ...]
    live_places: tuple[str, ...]


async def load_profile() -> Profile:
    rights = []
    for place in preferences.right_to_live_and_work_in():
        area = WORK_RIGHT_AREAS.get(place.casefold())
        rights.append(area if area is not None else (await resolve(place)).countries)
    live = await resolve_all(list(preferences.would_live_here()))
    not_live = await resolve_all(list(preferences.would_not_live_here()))
    visa = await resolve_all(list(preferences.sponsorship_required_in()))
    return Profile(work_rights=frozenset().union(*rights) if rights else frozenset(),
                   live=live.countries, not_live=not_live.countries, visa=visa.countries,
                   commute_places=preferences.would_commute_here(), live_places=preferences.would_live_here())


def widen_header_country(countries: frozenset[str]) -> frozenset[str]:
    '''A remote job anchored in a country is employable across it; in an EU country, across the EU.'''
    return countries | (EU_MEMBER_STATES if countries & EU_MEMBER_STATES else frozenset())


async def _commutable(place: str, profile: Profile) -> bool:
    '''On `would_commute_here`, by name or by any name the classifier knows the place by ("Toronto, ON").'''
    names = [place, *((await classify_location(place)).get('place_names') or [])]
    return any(location_token_matches(token, name) for token in profile.commute_places for name in names)


async def decide_option(option: Option, job: JobLocation, profile: Profile) -> tuple[str, str]:
    '''(verdict, reason) for one option.'''
    unverified = [e.place for e in option.employment_locations if not e.verified]
    if unverified:
        return UNDETERMINED, f'employment place(s) {unverified} stated but not found on the page'
    stated = [e for e in option.employment_locations if e.says_more_than(job.header_location)]
    header = await resolve(job.header_location) if job.header_location else Places()
    if stated:
        employment = await resolve_all([e.place for e in stated])
        basis = 'stated'
    elif option.workplace_type in ('hybrid', 'onsite') and option.working_locations:
        employment = await resolve_all([e.place for e in option.working_locations])
        basis = 'office'
    else:
        employment = Places(countries=widen_header_country(header.countries), unresolved=header.unresolved)
        basis = 'header'
    if employment.anywhere:
        return ELIGIBLE, 'employable anywhere'
    if not employment.countries:
        return UNDETERMINED, f'no employment country resolved from {list(employment.unresolved) or job.header_location!r}'

    livable = employment.countries & profile.work_rights & profile.live
    foreign = employment.countries - profile.work_rights
    foreign_ok = sorted(c for c in foreign if c not in profile.visa and c not in profile.not_live) if basis == 'header' else []
    visa_ok = sorted(c for c in foreign & profile.visa if job.sponsorship == 'offered')
    if not (livable or foreign_ok or visa_ok):
        if foreign & profile.visa and job.sponsorship != 'offered':
            return INELIGIBLE, f'employment in {sorted(foreign & profile.visa)} needs a visa and sponsorship is {job.sponsorship} ({basis})'
        return INELIGIBLE, f'employment only in {sorted(employment.countries)} ({basis}); none is a country you would live in with the right to work'
    where = sorted(livable) or foreign_ok or visa_ok

    regular = [e.place for e in option.working_locations if e.attendance != 'occasional']
    if option.workplace_type == 'remote':
        # A remote job's "customer sites" is travel, not an office (Docker 4484255958 was capped on
        # one repeat and not the other, 2026-11-06): only a place that resolves can make it attended.
        regular = [place for place in regular if (await resolve(place)).countries]
    if option.workplace_type in ('hybrid', 'onsite') or regular:
        if not regular:
            return UNDETERMINED, f'{option.workplace_type} with no office named'
        if any([await _commutable(place, profile) for place in regular]):
            return ELIGIBLE, f'{option.workplace_type} at {regular}, a place you would commute to'
        offices = await resolve_all(regular)
        if offices.countries and offices.countries <= profile.not_live:
            return INELIGIBLE, f'{option.workplace_type} at {regular}, in a country you would not live in'
        return ELIGIBLE_CAPPED, f'{option.workplace_type} at {regular}, not a place you would commute to'
    return ELIGIBLE, f'remote; employable in {where[:5]}{"…" if len(where) > 5 else ""} ({basis})'


async def decide(job: JobLocation, profile: Profile) -> Decision:
    '''The job's verdict: the best of its required options (all options when none is required).'''
    if job.contradiction:
        return Decision(UNDETERMINED, f'the page contradicts itself: {job.contradiction}')
    required = [o for o in job.options if o.requirement == 'required'] or job.options
    option_verdicts = [await decide_option(o, job, profile) for o in required]
    best = min(option_verdicts, key=lambda v: VERDICT_RANK[v[0]])
    return Decision(best[0], best[1], option_verdicts)


# --- Adjudicate Location ---------------------------------------------------------------------------

ADJUDICATE_DEFINITION = '''Code could not decide from the facts above whether this job suits the user, for the reason
given. Decide it from the page. Reply with only a JSON object: {"quotes": [passages of the page your decision
stands on, copied exactly], "reason": one sentence, "verdict": "eligible" | "ineligible" | "uncertain"}.
"eligible" only if the page lets the holder be employed in a country the user may work in and would live in (or
from anywhere); "ineligible" only if the page rules every such country out; otherwise "uncertain".'''


async def adjudicate(page_text: str, job: JobLocation, decision: Decision, profile: Profile, subject: str,
                     stage_stats: dict | None = None) -> dict:
    '''An undetermined job decided by the model, cited. Returns {verdict, reason, quotes, rejected}.'''
    user = (f'The user may work in: {sorted(profile.work_rights)}. Would live in: {list(profile.live_places)}. '
            f'Would not live in: {sorted(profile.not_live)}. Needs a visa to work remotely for: {sorted(profile.visa)}.')
    prompt = (f'--- PAGE ---\n{page_text}\n--- END PAGE ---\n\n{header_facts(job.header_location, job.header_workplace)}'
              f'\n\nWhat was read: {asdict(job)}\n\n{user}\n\nWhy code could not decide: {decision.reason}\n\n'
              f'{ADJUDICATE_DEFINITION}')
    reply = await chat_openrouter(prompt, SYSTEM, MODEL_NAME_LOCATION, temperature=GATE_FIELDS_TEMPERATURE)
    if stage_stats is not None:
        stage_stats['cost'] = stage_stats.get('cost', 0.0) + reply.cost_usd
    try:
        answer = extract_json_object(reply.content)
        verdict = _choice(answer.get('verdict'), ('eligible', 'ineligible', 'uncertain'), 'uncertain', 'verdict')
    except ValueError as ex:
        raise ValueError(f'Adjudicate Location ({subject}): unusable reply ({ex}; {reply.describe()}): '
                         f'{reply.content!r}') from ex
    quotes = _quotes(answer.get('quotes'))
    claims = [evidence.Claim(f'q{i}', verdict, q) for i, q in enumerate(quotes)]
    verdicts = await evidence.verify_claims(ADJUDICATE_CALL_SITE, subject, claims, page_text, 'job page',
                                            stage_stats=stage_stats) if claims else {}
    grounded = bool(claims) and all(v.accepted for v in verdicts.values())
    # A verdict that stands on nothing found in the page is no verdict: kept, with a warning.
    return {'verdict': verdict if grounded else 'uncertain', 'model_verdict': verdict,
            'reason': str(answer.get('reason') or '').strip(), 'quotes': quotes, 'grounded': grounded}
