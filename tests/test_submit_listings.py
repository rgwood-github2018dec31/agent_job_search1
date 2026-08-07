"""Contract tests for the batched Stage 1b listing submission.

Written BEFORE the implementation. These encode the assumptions behind batching:

1. Batching must be behaviourally identical to the per-listing path it replaces. Every
   dedup / age / already-applied rule and every audit counter must produce the same state.
2. Only listings the model actually submits may be recorded. Today `check_and_record_job`
   writes `processed_jobs/*.yaml` BEFORE the model decides whether to queue, so a listing the
   model silently judges irrelevant is permanently deduped and can never be reconsidered. In
   the 2026-08-06 run that silently buried 35 listings.
3. Silent screening must become visible in the funnel counters.
4. One tool call must handle a whole page of listings — that is the entire point (73% of
   Stage 1b turns were per-listing bookkeeping, each re-reading ~21K tokens of context).

Input data is parsed from a REAL captured LinkedIn search-results accessibility snapshot
(tests/fixtures/linkedin_search_results.md) rather than hand-written dicts, so the tests
exercise the field shapes the scraper actually encounters — duplicated titles, tracking query
strings on URLs, relative dates, missing salary/date fields.
"""

from pathlib import Path
import re

import pytest

from agentic_job_search import tools_generic as tools

FIXTURE = Path(__file__).parent / 'fixtures' / 'linkedin_search_results.md'

# The snapshot renders each listing as a link whose /url carries the job id, followed by the
# company and a few metadata generics.
_LISTING_RE = re.compile(
    r'- link "(?P<title>[^"]+)" \[ref=\w+\]:\s*\n\s*- /url: (?P<url>/jobs/view/(?P<job_id>\d+)/[^\n]*)'
)
_FIELD_RE = re.compile(r'- generic(?: \[ref=\w+\])?: (?P<val>[^\n]+)')


def parse_fixture_listings() -> list[dict]:
    """Extract listings from the real snapshot in the shape the scraper submits them."""
    raw = FIXTURE.read_text(encoding='utf-8')
    listings: list[dict] = []
    seen: set[str] = set()
    for m in _LISTING_RE.finditer(raw):
        job_id = m.group('job_id')
        if job_id in seen:
            continue
        seen.add(job_id)
        tail = raw[m.end():m.end() + 900]
        vals = [v.strip() for v in _FIELD_RE.findall(tail) if v.strip() != m.group('title')]
        listings.append({
            'job_id': job_id,
            'title': m.group('title'),
            'company': vals[0] if vals else 'Unknown',
            'url': f'https://www.linkedin.com{m.group("url")}',
            'date_posted': '',
            'snippet': ' | '.join(vals[:3]),
        })
    return listings


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point all persistent state at tmp_path and neutralise the applied-company LLM call."""
    monkeypatch.setattr(tools, 'RUN_DIR', tmp_path)
    monkeypatch.setattr(tools, 'PROCESSED_JOBS_DIR', tmp_path / 'processed_jobs')
    monkeypatch.setattr(tools, '_processed_jobs', set())
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_listing_records', {})
    monkeypatch.setattr(tools, '_applied_companies', {})

    async def no_applied_match(company):
        return None

    monkeypatch.setattr(tools, 'company_matches_applied', no_applied_match)
    return tmp_path


def snapshot_state() -> dict:
    """Everything the audit log and funnel are built from."""
    return {
        'processed': set(tools._processed_jobs),
        'candidates': [dict(c) for c in tools._candidates],
        'per_query': dict(tools._candidates_per_query),
        'check_counts': dict(tools._check_status_counts),
        'listing_records': {k: dict(v) for k, v in tools._listing_records.items()},
    }


async def run_per_listing(listings: list[dict], query: str) -> None:
    """The behaviour being replaced: check each, queue each that comes back 'new'."""
    for listing in listings:
        result = await tools.do_check_and_record_job(
            'linkedin', listing['job_id'], listing['company'], listing['title'],
            date_posted=listing['date_posted'] or None, url=listing['url'],
        )
        if result['content'][0]['text'] == 'new':
            await tools.do_queue_candidate(
                'linkedin', listing['job_id'], listing['url'], listing['title'],
                listing['company'], listing['snippet'],
                date_posted=listing['date_posted'] or None, query=query,
            )


# ---------------------------------------------------------------------------
# the fixture itself must be realistic
# ---------------------------------------------------------------------------

def test_fixture_is_a_real_snapshot_with_listings():
    listings = parse_fixture_listings()
    assert len(listings) >= 7, 'fixture should hold a realistic page of results'
    assert all(l['job_id'].isdigit() for l in listings)
    assert all(l['url'].startswith('https://www.linkedin.com/jobs/view/') for l in listings)
    # real snapshots carry tracking query strings — the batch path must tolerate them
    assert any('?' in l['url'] for l in listings)
    assert all(l['company'] and l['title'] for l in listings)


def test_fixture_carries_no_account_identifying_data():
    raw = FIXTURE.read_text(encoding='utf-8').lower()
    for leaked in ('garth', 'fullrank'):
        assert leaked not in raw


# ---------------------------------------------------------------------------
# 1. equivalence with the per-listing path
# ---------------------------------------------------------------------------

async def test_batch_matches_per_listing_exactly(isolated, monkeypatch):
    listings = parse_fixture_listings()
    query = 'Principal AI Engineer'

    await run_per_listing(listings, query)
    expected = snapshot_state()

    # reset every piece of state and replay the same input through the batch path
    monkeypatch.setattr(tools, '_processed_jobs', set())
    monkeypatch.setattr(tools, '_candidates', [])
    monkeypatch.setattr(tools, '_candidates_per_query', {})
    monkeypatch.setattr(tools, '_check_status_counts', {})
    monkeypatch.setattr(tools, '_listing_records', {})
    for stale in (isolated / 'processed_jobs').glob('*.yaml'):
        stale.unlink()

    await tools.do_submit_listings(query=query, listings=listings)
    actual = snapshot_state()

    assert actual['processed'] == expected['processed']
    assert actual['check_counts'] == expected['check_counts']
    assert actual['per_query'] == expected['per_query']
    assert actual['candidates'] == expected['candidates']
    assert actual['listing_records'] == expected['listing_records']


async def test_batch_writes_one_processed_record_per_new_listing(isolated):
    listings = parse_fixture_listings()
    await tools.do_submit_listings(query='Principal AI Engineer', listings=listings)
    written = list((isolated / 'processed_jobs').glob('*.yaml'))
    assert len(written) == len(listings)


# ---------------------------------------------------------------------------
# 2. dedup / age / already-applied rules still apply inside a batch
# ---------------------------------------------------------------------------

async def test_batch_skips_already_processed(isolated):
    listings = parse_fixture_listings()
    tools._processed_jobs.add(('linkedin', listings[0]['job_id']))

    await tools.do_submit_listings(query='q', listings=listings)

    assert tools._check_status_counts.get('already_processed') == 1
    assert listings[0]['job_id'] not in {c['job_id'] for c in tools._candidates}


async def test_batch_skips_stale_listing(isolated):
    listings = parse_fixture_listings()
    listings[0] = {**listings[0], 'date_posted': '60 days ago'}

    await tools.do_submit_listings(query='q', listings=listings)

    assert tools._check_status_counts.get('too_old') == 1
    assert listings[0]['job_id'] not in {c['job_id'] for c in tools._candidates}


async def test_batch_skips_already_applied_company(isolated, monkeypatch):
    listings = parse_fixture_listings()
    blocked = listings[0]['company']

    async def match(company):
        return 'applied.pdf' if company == blocked else None

    monkeypatch.setattr(tools, 'company_matches_applied', match)
    await tools.do_submit_listings(query='q', listings=listings)

    assert tools._check_status_counts.get('already_applied') >= 1
    assert blocked not in {c['company'] for c in tools._candidates}


# ---------------------------------------------------------------------------
# 3. screened-out listings leave no trace, and are counted
# ---------------------------------------------------------------------------

async def test_unsubmitted_listings_are_never_recorded(isolated):
    """The correctness fix. A listing the model judges irrelevant must stay reconsiderable:
    nothing about it may reach processed_jobs or _processed_jobs."""
    listings = parse_fixture_listings()
    submitted, withheld = listings[:3], listings[3:]

    await tools.do_submit_listings(query='q', listings=submitted)

    withheld_ids = {l['job_id'] for l in withheld}
    assert not withheld_ids & {job_id for _site, job_id in tools._processed_jobs}
    written = {f.name for f in (isolated / 'processed_jobs').glob('*.yaml')}
    for job_id in withheld_ids:
        assert not any(job_id in name for name in written)


async def test_screened_out_count_is_recorded_in_funnel(isolated):
    """Silent filtering was invisible in the 2026-08-06 audit; it must now be counted."""
    listings = parse_fixture_listings()
    await tools.do_submit_listings(query='q', listings=listings[:3], screened_out=8)
    assert tools._check_status_counts.get('screened_out') == 8


# ---------------------------------------------------------------------------
# 4. turn economics — the whole point of the change
# ---------------------------------------------------------------------------

async def test_one_call_handles_a_whole_page(isolated):
    listings = parse_fixture_listings()
    assert len(listings) >= 7

    result = await tools.do_submit_listings(query='q', listings=listings)

    # every listing processed by a single invocation
    assert len(tools._listing_records) == len(listings)
    # and it reports back compactly rather than per-listing
    text = result['content'][0]['text']
    assert len(text) < 300, f'batch result must stay small, got {len(text)} chars'
    assert str(len(listings)) in text


# ---------------------------------------------------------------------------
# 5. robustness against what a model actually sends
# ---------------------------------------------------------------------------

async def test_empty_batch_is_a_noop(isolated):
    result = await tools.do_submit_listings(query='q', listings=[])
    assert tools._candidates == []
    assert result['content'][0]['text']


async def test_missing_optional_fields_are_tolerated(isolated):
    """date_posted and snippet are frequently absent from a results listing."""
    await tools.do_submit_listings(query='q', listings=[
        {'job_id': '9999999999', 'title': 'Principal AI Engineer', 'company': 'Acme'},
    ])
    assert len(tools._candidates) == 1
    assert tools._candidates[0]['url'].endswith('9999999999/')


async def test_listing_missing_job_id_is_skipped_not_fatal(isolated, caplog):
    listings = parse_fixture_listings()
    bad = [{'title': 'No ID', 'company': 'Acme'}] + listings[:2]

    with caplog.at_level('WARNING'):
        await tools.do_submit_listings(query='q', listings=bad)

    assert len(tools._candidates) == 2
    assert 'job_id' in caplog.text


async def test_query_is_attributed_to_every_candidate(isolated):
    listings = parse_fixture_listings()
    await tools.do_submit_listings(query='Staff AI Engineer', listings=listings)
    assert tools._candidates_per_query == {'Staff AI Engineer': len(listings)}
