'''Decide Location Eligibility: the user's location rules over a Job Location, decided in code.

Each case is a posting the 2026-11-06 survey of retained pages found the old gate getting wrong or
treating inconsistently; job ids in the test names.
'''
import json

import pytest
from agentic_job_search import location_model as lm
from agentic_job_search.location import EU_MEMBER_STATES

COUNTRY_OF = {
    'Finland': 'Finland', 'Poland': 'Poland', 'Germany': 'Germany', 'Austria': 'Austria',
    'France': 'France', 'United Kingdom': 'United Kingdom', 'Canada': 'Canada', 'Spain': 'Spain',
    'United States': 'United States', 'Australia': 'Australia', 'Paris': 'France', 'Barcelona': 'Spain',
    'Lisbon': 'Portugal', 'Gdańsk, Poland': 'Poland', 'Ljubljana, Slovenia': 'Slovenia', 'Ontario': 'Canada', 'Leuven': 'Belgium',
}


@pytest.fixture(autouse=True)
def fake_classifier(monkeypatch):
    async def classify(text):
        country = COUNTRY_OF.get(text)
        return {'countries': [country] if country else []}

    monkeypatch.setattr(lm, 'classify_location', classify)


PROFILE = lm.Profile(
    work_rights=lm.EU_FREE_MOVEMENT | {'Canada', 'United Kingdom'},
    live=frozenset({'Spain', 'Portugal', 'Canada'}),
    not_live=frozenset({'Finland', 'Germany', 'Poland', 'France', 'United Kingdom', 'Austria', 'Belgium'}),
    visa=frozenset({'United States'}),
    commute_places=('Barcelona',), live_places=('Spain', 'Portugal', 'Canada'),
)


def entry(place, quote=None, attendance=''):
    return lm.Entry(place, [quote or f'based in {place}'], attendance)


def job(*options, header='', sponsorship='unknown', contradiction=''):
    return lm.JobLocation(list(options), sponsorship=sponsorship, contradiction=contradiction, header_location=header)


def remote(*employment):
    return lm.Option('remote', employment_locations=list(employment))


async def verdict(j):
    return (await lm.decide(j, PROFILE)).verdict


async def test_a_silent_remote_job_in_an_eu_country_is_employable_across_the_eu():
    '''Sii Poland 4484344125 kept and Auxo 4482929457 rejected on the same silence; now both kept.'''
    assert await verdict(job(remote(), header='Poland')) == lm.ELIGIBLE


async def test_a_silent_remote_job_in_a_country_you_would_not_live_in_outside_the_eu_is_ineligible():
    assert await verdict(job(remote(), header='United Kingdom')) == lm.INELIGIBLE


async def test_a_stated_single_country_overrides_the_eu_widening():
    '''Solita 4478266275: "requires you to be located in Finland" was never extracted; rated 3/5.'''
    assert await verdict(job(remote(entry('Finland')), header='Finland')) == lm.INELIGIBLE


async def test_the_stated_country_is_judged_not_the_header():
    '''Moneyhub 4482711232: header Slovenia, page "right to work in the UK".'''
    decision = await lm.decide(job(remote(entry('United Kingdom')), header='Ljubljana, Slovenia'), PROFILE)
    assert decision.verdict == lm.INELIGIBLE and 'United Kingdom' in decision.reason


async def test_a_multi_country_list_is_judged_whole():
    '''Owkin 4476851147: "remotely in France, UK, or Germany" was collapsed to the header's Germany.'''
    decision = await lm.decide(job(remote(entry('France'), entry('United Kingdom'), entry('Germany'))), PROFILE)
    assert decision.verdict == lm.INELIGIBLE
    assert all(country in decision.reason for country in ('France', 'Germany', 'United Kingdom'))


async def test_any_one_eligible_option_makes_the_job_eligible():
    '''Axelera: an office in Leuven, or remote from any European country.'''
    office = lm.Option('onsite', working_locations=[entry('Leuven', attendance='regular')])
    europe = remote(entry('any European country (incl. the UK)'))
    assert await verdict(job(office, europe, header='Slovakia')) == lm.ELIGIBLE


async def test_a_preferred_option_never_decides_when_a_required_one_exists():
    preferred = lm.Option('remote', employment_locations=[entry('Spain')], requirement='preferred')
    assert await verdict(job(preferred, remote(entry('Germany')))) == lm.INELIGIBLE


async def test_a_stated_place_whose_quotes_were_not_found_is_undetermined_never_silent():
    '''The 2026-11-02 lesson: a failed citation must not read as "the page states nothing".'''
    lost = lm.Entry('Finland', ['located in Finlnad'], quotes_not_found=['located in Finlnad'])
    decision = await lm.decide(job(remote(lost), header='Finland'), PROFILE)
    assert decision.verdict == lm.UNDETERMINED


async def test_one_found_quote_keeps_an_entry_whose_other_quote_was_mistyped():
    '''Workleap 4484276438 lost relocation=Canada when one of three quotes had a typo.'''
    canada = lm.Entry('Canada', ['Canada', 'pancadienne'], quotes_not_found=['pancadienne'])
    assert canada.verified
    assert await verdict(job(remote(canada), header='Canada')) == lm.ELIGIBLE


@pytest.mark.parametrize('sponsorship, expected', [('unknown', lm.INELIGIBLE), ('refused', lm.INELIGIBLE),
                                                   ('offered', lm.ELIGIBLE)])
async def test_a_visa_country_needs_sponsorship_even_remote(sponsorship, expected):
    assert await verdict(job(remote(), header='United States', sponsorship=sponsorship)) == expected


async def test_a_silent_remote_job_in_a_foreign_non_visa_country_is_eligible():
    assert await verdict(job(remote(), header='Australia')) == lm.ELIGIBLE


async def test_a_foreign_country_the_page_restricts_the_job_to_is_ineligible():
    assert await verdict(job(remote(entry('Australia')), header='Australia')) == lm.INELIGIBLE


async def test_an_office_you_would_commute_to_is_eligible_and_one_in_a_country_you_would_live_in_is_capped():
    barcelona = lm.Option('hybrid', working_locations=[entry('Barcelona', attendance='regular')])
    lisbon = lm.Option('hybrid', working_locations=[entry('Lisbon', attendance='regular')])
    paris = lm.Option('hybrid', working_locations=[entry('Paris', attendance='regular')])
    assert await verdict(job(barcelona)) == lm.ELIGIBLE
    assert await verdict(job(lisbon)) == lm.ELIGIBLE_CAPPED
    assert await verdict(job(paris)) == lm.INELIGIBLE


async def test_occasional_visits_do_not_make_a_remote_job_hybrid():
    visits = lm.Option('remote', working_locations=[entry('Paris', attendance='occasional')])
    assert await verdict(job(visits, header='Spain')) == lm.ELIGIBLE


async def test_a_contradicting_page_is_undetermined():
    assert await verdict(job(remote(), header='Canada', contradiction='Ontario only vs across Canada and the US')) \
        == lm.UNDETERMINED


async def test_no_resolvable_place_is_undetermined_not_eligible():
    assert await verdict(job(remote(), header='Somewhere unknown')) == lm.UNDETERMINED


@pytest.mark.parametrize('place, expected', [
    ('any European country (incl. the UK)', lm.EUROPE),
    ('European Union', EU_MEMBER_STATES),
    ('the EEA', EU_MEMBER_STATES | lm.EEA_NON_EU),
    ('EMEA', lm.EUROPE),
])
async def test_areas_resolve_inside_the_phrase_the_page_wrote(place, expected):
    assert (await lm.resolve(place)).countries >= expected


async def test_anywhere_is_anywhere():
    assert (await lm.resolve('anywhere in the world')).anywhere


def test_parse_reply_rejects_a_reply_with_no_options():
    with pytest.raises(ValueError, match='options'):
        lm.parse_reply(json.dumps({'options': []}), 'Canada', 'Remote')


def test_parse_reply_reads_entries_and_defaults():
    reply = {'options': [{'workplace_type': 'hybrid', 'workplace_quotes': ['Hybrid'],
                          'working_locations': [{'place': 'Paris', 'attendance': 'regular', 'quotes': ['Paris office']}],
                          'employment_locations': [], 'requirement': ''}],
             'sponsorship': {'value': 'refused', 'quotes': ['no sponsorship']}}
    parsed = lm.parse_reply(json.dumps(reply), 'Paris, France', 'Hybrid')
    assert parsed.options[0].requirement == 'required'
    assert parsed.options[0].working_locations[0].place == 'Paris'
    assert parsed.sponsorship == 'refused' and parsed.engagement == 'unknown'


@pytest.mark.parametrize('quote', ['Poland', 'Remote (Poland)', 'Gdańsk, Poland | Remote | Full-time'])
async def test_an_employment_place_quoting_only_the_header_is_read_as_silence(quote):
    '''SoftServe 4474954498: "Poland" quoted as a restriction flipped the verdict between two runs.'''
    bare = lm.Entry('Poland', [quote])
    assert await verdict(job(remote(bare), header='Gdańsk, Poland')) == lm.ELIGIBLE


async def test_a_quote_that_states_the_restriction_still_restricts():
    stated = lm.Entry('Poland', ['This is a remote position open to candidates residing in Poland.'])
    assert await verdict(job(remote(stated), header='Poland')) == lm.INELIGIBLE


async def test_an_office_matches_the_commute_list_by_any_name_the_classifier_knows(monkeypatch):
    async def classify(text):
        return {'countries': ['Spain'], 'place_names': ['Barcelona', 'Catalonia', 'Spain']}

    monkeypatch.setattr(lm, 'classify_location', classify)
    office = lm.Option('hybrid', working_locations=[entry('BCN, Catalunya', attendance='regular')])
    assert await verdict(job(office)) == lm.ELIGIBLE


async def test_a_country_of_residence_left_to_the_holder_is_employable_anywhere():
    '''Weflow 4483924663: "from anywhere within your country of residence".'''
    own = entry('your country of residence', 'work from anywhere within your country of residence')
    assert await verdict(job(remote(own), header='Germany')) == lm.ELIGIBLE


async def test_customer_sites_do_not_make_a_remote_job_attended():
    travel = lm.Option('remote', working_locations=[entry('Customer sites', attendance='regular')])
    assert await verdict(job(travel, header='Spain')) == lm.ELIGIBLE
