"""Read Job Pages Ahead (job_page_prefetch.JobPagePrefetcher): reads queued jobs' pages during Stage 1b."""
import asyncio

import pytest
from agentic_job_search import job_page_prefetch, linkedin_page


def _candidate(job_id):
    return {'site': 'linkedin', 'job_id': job_id, 'url': f'https://www.linkedin.com/jobs/view/{job_id}/'}


@pytest.fixture
def fake_reads(monkeypatch):
    '''A page reader: records the order of reads; `fail[job_id]` raises instead.'''
    order: list[str] = []
    fail: dict[str, Exception] = {}

    async def read(candidate):
        job_id = candidate['job_id']
        order.append(job_id)
        await asyncio.sleep(0)
        if job_id in fail:
            raise fail[job_id]
        return linkedin_page.JobPage(text=f'page {job_id}', source='test')

    monkeypatch.setattr(job_page_prefetch, 'JOB_PAGE_PREFETCH_POLL_SECONDS', 0)
    return read, order, fail


async def test_pages_are_read_ahead_in_queue_order_as_jobs_are_queued(fake_reads):
    read, order, _ = fake_reads
    queued = [_candidate('1')]
    prefetcher = job_page_prefetch.JobPagePrefetcher(lambda: queued, read, 'Signed-out')
    prefetcher.start()
    assert (await prefetcher.page(queued[0])).text == 'page 1'
    queued.append(_candidate('2'))   # queued later, while Stage 1b searches on
    assert (await prefetcher.page(queued[1])).text == 'page 2'
    await prefetcher.stop()
    assert order == ['1', '2']


async def test_a_failed_read_is_raised_for_its_job_only(fake_reads):
    read, _order, fail = fake_reads
    fail['1'] = ValueError('no title')
    queued = [_candidate('1'), _candidate('2')]
    prefetcher = job_page_prefetch.JobPagePrefetcher(lambda: queued, read, 'Signed-out')
    prefetcher.start()
    with pytest.raises(ValueError, match='no title'):
        await prefetcher.page(queued[0])
    assert (await prefetcher.page(queued[1])).text == 'page 2'
    await prefetcher.stop()


async def test_a_challenge_stops_all_reading_and_is_raised_for_every_later_job(fake_reads):
    read, order, fail = fake_reads
    fail['1'] = linkedin_page.LinkedInBlockedError('HTTP 429')
    queued = [_candidate('1'), _candidate('2'), _candidate('3')]
    prefetcher = job_page_prefetch.JobPagePrefetcher(lambda: queued, read, 'Signed-out')
    prefetcher.start()
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='429'):
        await prefetcher.page(queued[0])
    with pytest.raises(linkedin_page.LinkedInBlockedError, match='stopped earlier'):
        await prefetcher.page(queued[2])
    await prefetcher.stop()
    assert order == ['1']   # nothing was requested after the challenge


async def test_a_job_asked_for_again_is_read_again(fake_reads):
    '''Stage 2 reads a failed page a second time (a7514cd): the prefetched failure must not come back.'''
    read, order, fail = fake_reads
    fail['1'] = ValueError('Exa had no content')
    queued = [_candidate('1'), _candidate('2')]
    prefetcher = job_page_prefetch.JobPagePrefetcher(lambda: queued, read, 'Exa')
    prefetcher.start()
    with pytest.raises(ValueError, match='no content'):
        await prefetcher.page(queued[0])
    del fail['1']
    assert (await prefetcher.page(queued[0])).text == 'page 1'
    assert (await prefetcher.page(queued[1])).text == 'page 2'
    await prefetcher.stop()
    assert order.count('1') == 2 and order.count('2') == 1, 'read again once asked, and only then'


async def test_a_page_stage2_waits_for_is_read_before_the_rest(fake_reads):
    read, order, _ = fake_reads
    queued = [_candidate(str(i)) for i in range(1, 6)]
    prefetcher = job_page_prefetch.JobPagePrefetcher(lambda: queued, read, 'Exa')
    prefetcher.start()
    assert (await prefetcher.page(queued[4])).text == 'page 5'
    await prefetcher.stop()
    assert order.index('5') <= 1, 'asked for while the first read was under way, so read next'


async def test_a_page_read_but_never_asked_for_is_reported_at_stop(fake_reads, caplog):
    read, _order, _ = fake_reads
    queued = [_candidate('1'), _candidate('2')]
    prefetcher = job_page_prefetch.JobPagePrefetcher(lambda: queued, read, 'Exa')
    prefetcher.start()
    assert (await prefetcher.page(queued[0])).text == 'page 1'
    for _ in range(len(queued) * 2):
        await asyncio.sleep(0)   # let the read-ahead reach job 2
    with caplog.at_level('INFO'):
        await prefetcher.stop()
    assert any('Exa prefetch: 1 page(s) read but never reached' in r.message for r in caplog.records)
