"""Applied Organization records (Stage 1's skip list) and Search Organization (Stage 2's warning), 2026-11-19."""

import json
import logging

import jsonschema
import pytest
import yaml
from agentic_job_search import applied_organizations, company_domain, organization_search, web_search
from agentic_job_search.applied_organizations import AppliedOrganization
from agentic_job_search.company_domain import CompanyIdentity
from fakes import chat_reply
from pypdf import PdfWriter
from pypdf.annotations import Link


def _record(key: str, pdf: str = '2026-09-24-x.pdf', names: tuple = (), domains: tuple = (), slugs: tuple = ()):
    return AppliedOrganization(key=key, pdf=pdf, names=(key, *names), domains=domains, slugs=slugs)


class _FakeModel:
    '''Stands in for chat_openrouter_tool: hands out the queued arguments, one per call.'''

    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, prompt, tool_name, description, parameters, **kwargs):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f'unexpected model call ({tool_name})')
        return self.answers.pop(0), chat_reply('', cost_usd=0.001)


@pytest.fixture
def fake_search(monkeypatch):
    '''Stub the tool server's web_search: set `results`; `queries` lists what was searched.'''
    state = {'results': [], 'queries': []}

    async def fake_call(url, tool, args, timeout_seconds):
        assert tool == 'web_search'
        state['queries'].append(args['query'])
        return json.dumps({'ok': True, 'query': args['query'], 'query_run': args['query'],
                           'results': state['results'], 'cost_usd': 0.007})

    monkeypatch.setattr(web_search, 'call_mcp_tool', fake_call)
    return state


# --- without_legal_suffix -----------------------------------------------------------------------

@pytest.mark.parametrize('name, expected', [
    ('Orbiq America, Inc.', 'Orbiq America'),
    ('Orbiq Canada ULC', 'Orbiq Canada'),
    ('Orbiq Deutschland B.V. & Co. KG', 'Orbiq Deutschland'),
    ('Orbiq Software (Schweiz) GmbH', 'Orbiq Software (Schweiz)'),
    ('Acme S.A.', 'Acme'),
    ('Example Pty Ltd', 'Example'),
    ('Insight Software', 'Insight Software'),   # a descriptive word is kept
    ('The Quality Group', 'The Quality Group'),
    ('Inc', 'Inc'),                             # never stripped to nothing
])
def test_without_legal_suffix_strips_only_trailing_legal_forms(name, expected):
    assert applied_organizations.without_legal_suffix(name) == expected


# --- saved postings -> slug ---------------------------------------------------------------------

def _saved_posting(root, folder, name, company, title, slug):
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'# {title} at {company}\n\nTitle: {title}\nCompany: {company}\n'
                    f'Company LinkedIn page: https://www.linkedin.com/company/{slug}/\n', encoding='utf-8')


def test_saved_posting_slugs_link_company_and_title_to_the_page(tmp_path):
    _saved_posting(tmp_path, 'saved_jobs-2026Nov04', 'a.md', 'Fable', 'Senior AI Engineer', 'makeitfable')
    _saved_posting(tmp_path, 'saved_jobs-2026Nov05', 'b.md', 'Float', 'Staff ML', 'floatfinancial')
    _saved_posting(tmp_path, 'saved_jobs-2026Nov06', 'c.md', 'Float', 'Staff ML', 'float-scheduling')
    slugs = applied_organizations.saved_posting_slugs(tmp_path)
    assert slugs[('fable', 'senior ai engineer')] == 'makeitfable'
    assert ('float', 'staff ml') not in slugs, 'two pages for one pair: neither is used'


# --- build_records ------------------------------------------------------------------------------

@pytest.fixture
def cached_pages(monkeypatch):
    '''Cached LinkedIn pages, as Stage 2 stores them in company_identities.yaml.'''
    company_domain._load_cache()['by_slug'].update({
        'orbiq-spain': {'schema': 1, 'source': 'linkedin', 'name': 'Orbiq Spain', 'website': 'https://www.orbiq.com/es',
                         'domain': 'orbiq.com', 'parent_name': 'Orbiq', 'parent_slug': 'orbiq'},
        'orbiq': {'schema': 1, 'source': 'linkedin', 'name': 'Orbiq', 'website': 'https://www.orbiq.com',
                   'domain': 'orbiq.com'},
    })
    company_domain._load_cache()['by_name']['Orbiq Spain'] = {'schema': 1, 'source': 'none'}


async def test_a_record_holds_the_linkedin_page_and_the_relatives_a_result_names_but_never_its_parent(
        monkeypatch, tmp_path, cached_pages, fake_search):
    # The applied job's saved posting names its LinkedIn page (SAVED_POSTINGS_ROOT is tmp_path in tests).
    _saved_posting(tmp_path, 'saved_jobs-2026Sep24', 'o.md', 'Orbiq Spain', 'AI Architect', 'orbiq-spain')
    fake_search['results'] = [
        {'title': 'EX-21.01', 'url': 'https://www.sec.gov/x', 'content': 'ORACLE CORPORATION Subsidiaries of the '
         'Registrant | Orbiq America, Inc. | Delaware | | Orbiq Canada ULC | Canada |'},
        {'title': 'Float', 'url': 'https://example.com/float', 'content': 'Acme Widgets is a customer of Float.'},
    ]
    model = _FakeModel({'related': [
        {'name': 'Orbiq America, Inc.', 'relation': 'subsidiary', 'result': 1, 'quote': 'Orbiq America, Inc. | Delaware'},
        {'name': 'Orbiq Canada ULC', 'relation': 'subsidiary', 'result': 1, 'quote': 'Orbiq Canada ULC'},
        {'name': 'Orbiq Japan', 'relation': 'subsidiary', 'result': 1, 'quote': 'Orbiq Japan KK'},   # not in it
        {'name': 'Acme Widgets', 'relation': 'other', 'result': 2, 'quote': 'Acme Widgets is a customer'},
    ]})
    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', model)
    stats: dict = {}

    records = await applied_organizations.build_records(
        {'Orbiq Spain': '2026-09-24-orbiq.pdf'}, {'Orbiq Spain': ['AI Architect']}, stats)

    record = records['Orbiq Spain']
    assert record.slugs == ('orbiq-spain',), 'not the parent page'
    assert record.domains == ('orbiq.com',)
    for name in ('Orbiq Spain', 'Orbiq America, Inc.', 'Orbiq America', 'Orbiq Canada ULC', 'Orbiq Canada'):
        assert name in record.names, name
    assert 'Orbiq' not in record.names, 'the parent never counts (2026-11-19)'
    assert 'Orbiq Japan' not in record.names, 'its quote is not in the results'
    assert 'Acme Widgets' not in record.names, 'its result does not name the applied organization'
    assert record.sources['Orbiq America'] == applied_organizations.SOURCE_LEGAL_SUFFIX
    assert fake_search['queries'] == ['Orbiq Spain orbiq.com parent company subsidiaries regional offices']
    assert stats['cost'] > 0

    # Built once: the next load reads the file and asks nothing.
    again = await applied_organizations.build_records(
        {'Orbiq Spain': '2026-09-24-orbiq.pdf'}, {'Orbiq Spain': ['AI Architect']}, {})
    assert again['Orbiq Spain'].names == record.names and len(fake_search['queries']) == 1


async def test_a_failed_web_search_is_retried_next_run_and_manual_entries_survive(monkeypatch, fake_search):
    path = applied_organizations.APPLIED_ORGANIZATIONS_PATH
    path.write_text(yaml.safe_dump({'Acme': {'schema': 1, 'pdf': 'a.pdf', 'web_search': {'error': 'down'},
                                             'names': [{'name': 'Acme Labs', 'source': 'manual'}]}}), encoding='utf-8')
    company_domain._load_cache()['by_name']['Acme'] = {'schema': 1, 'source': 'none'}

    records = await applied_organizations.build_records({'Acme': 'a.pdf'}, {}, {})

    assert fake_search['queries'], 'the failed search was tried again'
    assert 'Acme Labs' in records['Acme'].names
    assert applied_organizations.name_index(records)['acme labs'] == ('Acme', 'manual')


async def test_an_unreadable_records_file_is_never_overwritten(caplog):
    path = applied_organizations.APPLIED_ORGANIZATIONS_PATH
    path.write_text('- not: a mapping of records\n', encoding='utf-8')
    records = await applied_organizations.build_records({'Acme': 'a.pdf'}, {}, {})
    assert records['Acme'].names == ('Acme',)
    assert path.read_text(encoding='utf-8') == '- not: a mapping of records\n'
    assert 'fix or move the file aside' in caplog.text


# --- Search Organization ------------------------------------------------------------------------

_RECORDS = {
    'NTT DATA, Inc.': _record('NTT DATA, Inc.', domains=('nttdata.com',)),
    'Veltra': _record('Veltra', domains=('veltra.com',)),
    'Tiny Agency': _record('Tiny Agency'),   # nobody knows its domain
}


async def test_step_1_never_matches_on_the_parent_a_linkedin_page_names(monkeypatch):
    """2026-11-19, the user's rule: a parent never counts. NTT DATA Europe & Latam's page names NTT DATA,
    Inc. as its parent, which alone matches nothing; the model is asked, and says no match."""
    identity = CompanyIdentity('NTT DATA Europe & Latam', parent_name='NTT DATA, Inc.', parent_domain='nttdata.com',
                               source='linkedin')
    model = _FakeModel({'options': [_option('no match', 5)]})
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)
    match = await organization_search.search_organization(
        'NTT DATA Europe & Latam', 'ntt-data-europe-latam', identity, 'job text', _RECORDS, {})
    assert match.record is None and match.step == organization_search.STEP_DATABASE
    assert 'Parent' not in model.prompts[0]


async def test_step_1_matches_a_subsidiary_its_record_lists():
    records = {**_RECORDS, 'NTT DATA, Inc.': _record('NTT DATA, Inc.', names=('NTT DATA Europe & Latam',),
                                                     domains=('nttdata.com',))}
    identity = CompanyIdentity('NTT DATA Europe & Latam', parent_name='NTT DATA, Inc.', source='linkedin')
    match = await organization_search.search_organization(
        'NTT DATA Europe & Latam', 'ntt-data-europe-latam', identity, 'job text', records, {})
    assert (match.record.key, match.step) == ('NTT DATA, Inc.', organization_search.STEP_DOMAIN)
    assert match.warning().startswith('Possibly applied already: NTT DATA, Inc. (2026-09-24)')


async def test_step_1_matches_a_domain_the_job_text_writes_for_the_organization():
    match = await organization_search.search_organization(
        'Veltra Systems', '', None, 'Apply at https://careers.veltra.com/jobs/1', _RECORDS, {})
    assert match.record.key == 'Veltra' and 'veltra.com' in match.reason


async def test_step_1_ignores_a_domain_of_another_organization_in_the_job_text(monkeypatch):
    '''A posting that mentions aws.amazon.com is not Amazon's.'''
    model = _FakeModel({'options': [{'organization': 'no match', 'confidence': 5, 'explanation': 'x', 'quote': ''}]})
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)
    match = await organization_search.search_organization(
        'Acme', '', None, 'We deploy on veltra.com and aws.amazon.com', _RECORDS, {})
    assert match.record is None


async def test_step_1_rules_out_every_record_with_another_domain_and_asks_only_about_the_rest(monkeypatch):
    model = _FakeModel({'options': [{'organization': 'no match', 'confidence': 5, 'explanation': 'unrelated', 'quote': ''}]})
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)
    identity = CompanyIdentity('Vorlix', domain='vorsee.io', source='linkedin')

    match = await organization_search.search_organization('Vorlix', 'vorsee', identity, 'job text', _RECORDS, {})

    assert match.record is None and match.step == organization_search.STEP_DATABASE
    assert 'A1: Tiny Agency' in model.prompts[0]
    assert 'Veltra' not in model.prompts[0].split('=== Organizations the user applied to ===')[1]


async def test_a_record_with_another_domain_is_never_asked_about_when_none_is_left(monkeypatch):
    identity = CompanyIdentity('Vorlix', domain='vorsee.io', source='linkedin')
    records = {key: r for key, r in _RECORDS.items() if r.domains}
    match = await organization_search.search_organization('Vorlix', 'vorsee', identity, 'job text', records, {})
    assert (match.record, match.step) == (None, organization_search.STEP_RULED_OUT)


def _option(organization, confidence, quote='', explanation='because'):
    return {'organization': organization, 'confidence': confidence, 'explanation': explanation, 'quote': quote}


async def test_step_2_one_high_option_decides(monkeypatch):
    model = _FakeModel({'options': [_option('A3', 5, 'a Tiny Agency company'), _option('no match', 2)]})
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)
    match = await organization_search.search_organization(
        'Tiny Agency Labs', '', None, 'Tiny Agency Labs is a Tiny Agency company.', _RECORDS, {})
    assert (match.record.key, match.step) == ('Tiny Agency', organization_search.STEP_DATABASE)


async def test_step_2_a_no_match_option_needs_no_quote(monkeypatch):
    # 2026-11-20: deepseek/deepseek-v4.1-flash-1017 left out the quote of "no match" and the schema failed the check.
    arguments = {'options': [{'organization': 'no match', 'confidence': 5, 'explanation': 'none of them'}]}
    jsonschema.validate(arguments, organization_search._PARAMETERS)
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', _FakeModel(arguments))
    match = await organization_search.search_organization(
        'Bring It On', '', None, 'Bring It On builds competitive games.', _RECORDS, {})
    assert match is None or match.record is None


@pytest.mark.parametrize('options, says', [
    ([_option('A3', 5, 'Tiny')], 'no "no match" option'),
    ([_option('A9', 5, 'Tiny'), _option('no match', 1)], 'neither an applied organization id'),
    ([_option('A3', 7, 'Tiny'), _option('no match', 1)], 'confidence 7'),
])
async def test_step_2_an_unusable_reply_fails_the_search_and_warns_nothing(monkeypatch, options, says, caplog):
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', _FakeModel({'options': options}))
    with pytest.raises(ValueError, match=says):
        await organization_search.search_organization('Tiny Agency Labs', '', None, 'job', _RECORDS, {})
    stats: dict = {}
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', _FakeModel({'options': options}))
    assert await organization_search.warn_if_applied('Tiny Agency Labs', '', None, 'job', _RECORDS, stats, 'poster') == ''
    assert stats['organization_search_failed'] == 1 and says in caplog.text


async def test_step_2_an_option_whose_quote_is_not_in_the_input_is_dropped(monkeypatch, fake_search):
    '''Vorlix → Veltra, 2026-11-12: a match with nothing in the posting to stand on.'''
    first = {'options': [_option('A3', 5, 'Vorlix is part of Tiny Agency'), _option('no match', 2)]}
    second = {'options': [_option('no match', 4)]}
    model = _FakeModel(first, second)
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)
    monkeypatch.setattr(organization_search.evidence, 'chat_openrouter', _no_relocation)

    match = await organization_search.search_organization('Vorlix', '', None, 'Vorlix builds retail AI.', _RECORDS, {})

    # Dropped, nothing was high: the web search was run and its answer decided.
    assert (match.record, match.step) == (None, organization_search.STEP_WEB_SEARCH)


async def _no_relocation(*args, **kwargs):
    return chat_reply('{"passages": {"0": ""}}')


async def test_two_high_options_go_to_a_web_search_and_its_results_decide(monkeypatch, fake_search):
    fake_search['results'] = [{'title': 'Acme', 'url': 'https://acme.example', 'content': 'Acme is owned by Tiny Agency.'}]
    model = _FakeModel(
        {'options': [_option('A3', 4, 'Acme'), _option('no match', 4)]},
        {'options': [_option('A3', 5, 'Acme is owned by Tiny Agency'), _option('no match', 1)]})
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)

    match = await organization_search.search_organization('Acme', '', None, 'Acme is hiring.', _RECORDS, {})

    assert (match.record.key, match.step) == ('Tiny Agency', organization_search.STEP_WEB_SEARCH)
    assert fake_search['queries'] == ['Acme company']
    assert 'Acme is owned by Tiny Agency' in model.prompts[1]


async def test_still_undecided_after_the_web_search_is_no_match_and_counted(monkeypatch, fake_search):
    model = _FakeModel({'options': [_option('no match', 3)]}, {'options': [_option('no match', 3)]})
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)
    stats: dict = {}
    match = await organization_search.search_organization('Acme', '', None, 'Acme is hiring.', _RECORDS, stats)
    assert (match.record, match.step) == (None, organization_search.STEP_UNDECIDED)
    assert stats['organization_search_undecided'] == 1


async def test_an_answer_is_cached_until_the_records_change(monkeypatch):
    model = _FakeModel({'options': [_option('no match', 5)]}, {'options': [_option('no match', 5)]})
    monkeypatch.setattr(organization_search, 'chat_openrouter_tool', model)
    await organization_search.search_organization('Acme', '', None, 'job', _RECORDS, {})
    cached = await organization_search.search_organization('Acme', '', None, 'other job', _RECORDS, {})
    assert cached.cached and len(model.prompts) == 1

    changed = {**_RECORDS, 'New Co': _record('New Co')}
    fresh = await organization_search.search_organization('Acme', '', None, 'job', changed, {})
    assert not fresh.cached and len(model.prompts) == 2


# --- The applied PDF's company link (2026-11-19) ------------------------------------------------

def _pdf_with_links(path, pages: list[list[str]]):
    '''A PDF whose page n links to each LinkedIn company slug in pages[n], in order.'''
    writer = PdfWriter()
    for page_number, slugs in enumerate(pages):
        writer.add_blank_page(width=200, height=200)
        for slug in slugs:
            writer.add_annotation(page_number, Link(rect=(0, 0, 10, 10),
                                                    url=f'https://www.linkedin.com/company/{slug}/life/'))
    with path.open('wb') as handle:
        writer.write(handle)
    return path


def test_the_pdf_slug_is_the_company_linked_most_on_its_first_page(tmp_path):
    # Danforth's PDF: its own page three times on page 1, a similar job's company once, more on page 2.
    pdf = _pdf_with_links(tmp_path / 'd.pdf', [['beckman-coulter', 'danforth', 'danforth', 'danforth'],
                                               ['beckman-coulter', 'beckman-coulter', 'beckman-coulter']])
    assert applied_organizations.linkedin_slug_of_pdf(pdf) == 'danforth'
    assert applied_organizations.linkedin_slug_of_pdf(_pdf_with_links(tmp_path / 'g.pdf', [[], ['acme']])) == ''


async def test_a_page_not_cached_is_read_and_the_pdf_link_comes_before_the_saved_posting(
        monkeypatch, tmp_path, fake_search):
    _saved_posting(tmp_path, 'saved_jobs-2026Sep24', 'o.md', 'Orbiq Spain', 'AI Architect', 'wrong-page')
    company_domain._load_cache()['by_name']['Orbiq Spain'] = {'schema': 1, 'source': 'none'}
    read: list[str] = []

    async def read_page(slug, name):
        read.append(slug)
        company_domain._load_cache()['by_slug'][slug] = {
            'schema': 1, 'source': 'linkedin', 'name': {'orbiq-spain': 'Orbiq Spain', 'orbiq': 'Orbiq'}[slug],
            'website': 'https://www.orbiq.com', 'domain': 'orbiq.com',
            **({'parent_name': 'Orbiq', 'parent_slug': 'orbiq'} if slug == 'orbiq-spain' else {})}

    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', _FakeModel({'related': []}))
    records = await applied_organizations.build_records(
        {'Orbiq Spain': 'o.pdf'}, {'Orbiq Spain': ['AI Architect']}, {},
        pdf_slugs={'Orbiq Spain': ['orbiq-spain']}, read_linkedin_page=read_page)

    assert read == ['orbiq-spain'], 'never its parent page'
    assert records['Orbiq Spain'].slugs == ('orbiq-spain',)
    raw = yaml.safe_load(applied_organizations.APPLIED_ORGANIZATIONS_PATH.read_text(encoding='utf-8'))
    assert raw['Orbiq Spain']['slugs'][0] == {'slug': 'orbiq-spain', 'source': 'linkedin_page', 'via': 'applied_pdf'}


async def test_an_unreadable_page_leaves_the_record_without_page_facts(monkeypatch, fake_search):
    company_domain._load_cache()['by_name']['Acme'] = {'schema': 1, 'source': 'none'}

    async def read_page(slug, name):
        return None   # the reader logged why

    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', _FakeModel({'related': []}))
    records = await applied_organizations.build_records({'Acme': 'a.pdf'}, {}, {}, pdf_slugs={'Acme': ['acme']},
                                                        read_linkedin_page=read_page)
    assert records['Acme'].slugs == () and records['Acme'].names == ('Acme',)


async def test_a_relative_whose_quote_joins_two_table_cells_is_kept_on_its_name_in_the_cited_result(
        monkeypatch, fake_search):
    company_domain._load_cache()['by_name']['HXLTech'] = {'schema': 1, 'source': 'none'}
    fake_search['results'] = [{'title': 'HXLTech subsidiaries', 'url': 'https://www.hxltech.com/subsidiaries',
                               'content': 'HXLTech | 12 HXL Technologies Austria GmbH_English | 12 HXL Technologies '
                                          'Austria GmbH_Local |'}]
    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', _FakeModel({'related': [
        {'name': 'HXL Technologies Austria GmbH', 'relation': 'subsidiary', 'result': 1,
         'quote': '12 HXL Technologies Austria GmbH_English / 12 HXL Technologies Austria GmbH_Local'},
        {'name': 'HXL Japan', 'relation': 'subsidiary', 'result': 1, 'quote': 'HXL Japan KK'},   # in no result
    ]}))
    records = await applied_organizations.build_records({'HXLTech': 'h.pdf'}, {}, {})
    assert 'HXL Technologies Austria' in records['HXLTech'].names
    assert 'HXL Japan' not in records['HXLTech'].names


async def test_a_quote_across_table_cells_and_headings_is_found_piece_by_piece(monkeypatch, fake_search, caplog):
    company_domain._load_cache()['by_name']['KLAM'] = {'schema': 1, 'source': 'none'}
    fake_search['results'] = [{'title': 'KLAM 10-K exhibit 21', 'url': 'https://www.sec.gov/klam-ex21.htm',
                               'content': 'SUBSIDIARIES OF THE REGISTRANT\n\n| Name | Country |\n|---|---|\n'
                                          '| KLAM Systems Albania SHPK | Albania |\n\nGlobal Locations\n\n'
                                          '### KLAM Systems India Pvt. Ltd.\nNew Delhi'}]
    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', _FakeModel({'related': [
        {'name': 'KLAM Systems Albania SHPK', 'relation': 'subsidiary', 'result': 1,
         'quote': 'SUBSIDIARIES OF THE REGISTRANT | KLAM Systems Albania SHPK | Albania'},
        {'name': 'KLAM Systems India Pvt. Ltd.', 'relation': 'subsidiary', 'result': 1,
         'quote': 'Global Locations ... ### KLAM Systems India Pvt. Ltd. New Delhi'},
        {'name': 'KLAM Systems Peru', 'relation': 'subsidiary', 'result': 1,
         'quote': 'SUBSIDIARIES OF THE REGISTRANT | KLAM Systems Peru | Peru'},   # a piece in no result
    ]}))
    with caplog.at_level(logging.WARNING):
        records = await applied_organizations.build_records({'KLAM': 'k.pdf'}, {}, {})
    names = records['KLAM'].names
    assert any('Albania' in n for n in names) and any('India' in n for n in names)
    assert not any('Peru' in n for n in names)
    rejected = [r.getMessage() for r in caplog.records if 'Hallucination rejected' in r.getMessage()]
    assert len(rejected) == 1 and 'Peru' in rejected[0]


async def test_a_misspelled_relative_is_kept_as_its_cited_result_spells_it(monkeypatch, fake_search):
    company_domain._load_cache()['by_name']['KLAM'] = {'schema': 1, 'source': 'none'}
    fake_search['results'] = [{'title': 'KLAM subsidiaries', 'url': 'https://www.sec.gov/klam-ex21.htm',
                               'content': 'KLAM subsidiaries: | Maravir Technologies LLC | Delaware, USA |\n'
                                          '| Moxa SA | Uruguay |'}]
    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', _FakeModel({'related': [
        {'name': 'Maraavir Technologies LLC', 'relation': 'subsidiary', 'result': 1,
         'quote': 'KLAM subsidiaries: | Maravir Technologies LLC | Delaware, USA'},
        {'name': 'Moxi SA', 'relation': 'subsidiary', 'result': 1, 'quote': 'Moxi SA'},   # too short to respell
    ]}))
    names = (await applied_organizations.build_records({'KLAM': 'k.pdf'}, {}, {}))['KLAM'].names
    assert any(n.startswith('Maravir Technologies') for n in names)
    assert not any('Maraavir' in n for n in names)
    assert not any(n.startswith(('Moxi', 'Moxa')) for n in names)


def test_within_edits_counts_changed_added_and_dropped_letters():
    assert applied_organizations._within_edits('maravir s.a.', 'maraavir s.a.', 1)
    assert applied_organizations._within_edits('maravir s.a.', 'marevir s.a.', 1)
    assert not applied_organizations._within_edits('maravir s.a.', 'marevar s.a.', 1)
    assert not applied_organizations._within_edits('acme', 'acme systems', 1)


async def test_a_name_kept_from_one_result_lets_another_result_name_the_organization(monkeypatch, fake_search):
    company_domain._load_cache()['by_name']['FIQO'] = {'schema': 1, 'source': 'none'}
    fake_search['results'] = [
        {'title': 'EX-21', 'url': 'https://www.sec.gov/fiqo-ex21', 'content': 'Subsidiaries of Fair Ibsen Corporation: '
         'Fair Ibsen (Australia) Pty Ltd'},
        {'title': 'FIQO', 'url': 'https://en.wikipedia.org/wiki/FIQO', 'content': 'FIQO, formerly Fair Ibsen Corporation'},
    ]
    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', _FakeModel({'related': [
        {'name': 'Fair Ibsen (Australia) Pty Ltd', 'relation': 'subsidiary', 'result': 1,
         'quote': 'Fair Ibsen (Australia) Pty Ltd'},   # its result never says "FIQO"
        {'name': 'Fair Ibsen Corporation', 'relation': 'former name', 'result': 2,
         'quote': 'formerly Fair Ibsen Corporation'},
    ]}))
    records = await applied_organizations.build_records({'FIQO': 'f.pdf'}, {}, {})
    assert {'Fair Ibsen Corporation', 'Fair Ibsen (Australia) Pty Ltd'} <= set(records['FIQO'].names)


def test_parent_entries_a_file_holds_are_left_out_when_a_record_is_loaded():
    raw = {'pdf': 'b.pdf', 'names': [
        {'name': 'DMX', 'source': 'applied_record'},
        {'name': 'ZRT', 'source': 'web_search', 'relation': 'parent'},
        {'name': 'ZRT & Co. Inc.', 'source': 'web_search', 'relation': 'ultimate parent'},
        {'name': 'ZRT & Co.', 'source': 'legal_suffix_removed', 'of': 'ZRT & Co. Inc.'},
        {'name': 'DMX Helix', 'source': 'web_search', 'relation': 'subsidiary'},
        {'name': 'DMX Inc.', 'source': 'web_search', 'relation': 'parent'},   # its own legal entity
        {'name': 'Crate Parent Company', 'source': 'linkedin_parent'},
    ], 'domains': [{'domain': 'dmx.com', 'source': 'linkedin_page'},
                   {'domain': 'zrt.com', 'source': 'model_identity', 'of_parent': 'ZRT'}],
        'slugs': [{'slug': 'dmx-software', 'source': 'linkedin_page'}, {'slug': 'zrt', 'source': 'linkedin_parent'}]}
    record = applied_organizations.record_from_raw('DMX', raw)
    assert record.names == ('DMX', 'DMX Helix', 'DMX Inc.')
    assert record.domains == ('dmx.com',) and record.slugs == ('dmx-software',)


# --- Company links from recommendations (2026-11-19) --------------------------------------------

def _recommendation(root, folder, name, company, title, job_id):
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'# {title} at {company}\n\nTitle: {title}\nCompany: {company}\n'
                    f'URL: https://www.linkedin.com/jobs/view/{job_id}/\n', encoding='utf-8')


def test_recommendations_match_alike_names_and_put_a_similar_title_first(tmp_path):
    _recommendation(tmp_path, 'saved_jobs-2026Sep27', 'a.md', 'DMX Software', 'Sales Director', '1')
    _recommendation(tmp_path, 'saved_jobs-2026Sep09', 'b.md', 'DMX Software', 'Principal AI Engineer', '2')
    _recommendation(tmp_path, 'saved_jobs-2026Oct01', 'c.md', 'BMCX Holdings', 'Principal AI Engineer', '3')
    assert applied_organizations.recommendation_job_urls('DMX', ['Principal AI Engineer'], tmp_path) == [
        ('https://www.linkedin.com/jobs/view/2/', '2'), ('https://www.linkedin.com/jobs/view/1/', '1')]


async def test_a_record_with_no_other_link_takes_its_page_from_a_recommended_job(monkeypatch, tmp_path, fake_search):
    _recommendation(tmp_path, 'saved_jobs-2026Sep27', 'w.md', 'Quill AI', 'AI Engineer', '7')
    _recommendation(tmp_path, 'saved_jobs-2026Sep28', 'x.md', 'Quill AI', 'AI Engineer', '8')
    company_domain._load_cache()['by_name']['Quill'] = {'schema': 1, 'source': 'none'}
    company_domain._load_cache()['by_slug'].update({
        'quill-elsewhere': {'schema': 1, 'source': 'linkedin', 'name': 'Wanderlust Travel', 'domain': 'wl.example'},
        'quillai': {'schema': 1, 'source': 'linkedin', 'name': 'Quill AI', 'domain': 'quill.ai'}})
    read: list[str] = []

    async def read_job(url, job_id, company):
        read.append(job_id)
        return {'8': 'quill-elsewhere', '7': 'quillai'}[job_id]   # the newest links another company's page

    monkeypatch.setattr(applied_organizations, 'chat_openrouter_tool', _FakeModel({'related': []}))
    records = await applied_organizations.build_records({'Quill': 'w.pdf'}, {'Quill': ['AI Engineer']}, {},
                                                        read_job_page_slug=read_job)
    assert read == ['8', '7']
    assert records['Quill'].slugs == ('quillai',) and 'Quill AI' in records['Quill'].names
    raw = yaml.safe_load(applied_organizations.APPLIED_ORGANIZATIONS_PATH.read_text(encoding='utf-8'))
    assert raw['Quill']['slugs'][0]['via'] == 'recommendation'
    assert raw['Quill']['slugs'][0]['job_url'] == 'https://www.linkedin.com/jobs/view/7/'


# --- Company links from a web search (2026-11-20) -----------------------------------------------

async def test_a_web_search_result_page_of_the_same_name_wins_over_one_only_alike(fake_search):
    company_domain._load_cache()['by_slug'].update({
        'pierson-vue': {'schema': 1, 'source': 'linkedin', 'name': 'Pierson VUE', 'domain': 'pearsonvue.com'},
        'pierson-tours': {'schema': 1, 'source': 'linkedin', 'name': 'Tours by Pierson', 'domain': 't.example'},
        'pierson': {'schema': 1, 'source': 'linkedin', 'name': 'Pierson', 'domain': 'pierson.com'}})
    fake_search['results'] = [
        {'title': 'Pierson VUE', 'url': 'https://www.linkedin.com/company/pierson-vue/', 'content': ''},
        {'title': 'Tours', 'url': 'https://uk.linkedin.com/company/pierson-tours', 'content': ''},
        {'title': 'Pierson', 'url': 'https://www.linkedin.com/company/pierson/jobs', 'content': ''},
        {'title': 'News', 'url': 'https://example.com/pierson', 'content': ''}]
    read: list[str] = []

    async def read_page(slug, company):
        read.append(slug)

    assert await applied_organizations._slug_from_web_search('Pierson', read_page, {}) == (
        'pierson', 'https://www.linkedin.com/company/pierson/jobs')
    assert read == []   # all cached


async def test_a_web_search_takes_a_page_only_alike_when_none_is_equal_and_never_another_name(fake_search):
    company_domain._load_cache()['by_slug'].update({
        'twinlo-segment': {'schema': 1, 'source': 'linkedin', 'name': 'Twinlo Segment', 'domain': 'segment.com'},
        'other': {'schema': 1, 'source': 'linkedin', 'name': 'Acme', 'domain': 'acme.example'}})

    async def read_page(slug, company):
        pass

    fake_search['results'] = [{'title': 'x', 'url': 'https://www.linkedin.com/company/other', 'content': ''},
                              {'title': 'y', 'url': 'https://www.linkedin.com/company/twinlo-segment', 'content': ''}]
    assert (await applied_organizations._slug_from_web_search('Twinlo', read_page, {}))[0] == 'twinlo-segment'
    fake_search['results'] = [{'title': 'x', 'url': 'https://www.linkedin.com/company/other', 'content': ''}]
    assert await applied_organizations._slug_from_web_search('Twinlo', read_page, {}) == ('', '')
