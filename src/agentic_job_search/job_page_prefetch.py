'''Read Job Pages Ahead: Stage 2's job pages are read while Stage 1b is still searching.

A page read can be slow for reasons that have nothing to do with the job: the signed-out reader
waits about a minute between requests (linkedin_signed_out's pacer), and Exa often has no content
on its first try and is asked again PAGE_FETCH_RETRY_DELAY_SECONDS later (linkedin_exa). On
2026-11-07, 19 of those Exa retries took about 10 of Stage 2's 29 minutes. Stage 1b searches for
about 25 minutes, so reading each page as soon as Stage 1b queues its job moves that wait out of
Stage 2 entirely.

The prefetcher only reads. What the page is turned into, and what is recorded in the job's trace,
is decided in Stage 2 when it takes the result, because the job trace only exists from then on.
'''
import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from agentic_job_search.config import JOB_PAGE_PREFETCH_POLL_SECONDS
from agentic_job_search.linkedin_page import LinkedInBlockedError

logger = logging.getLogger(__name__)

PageT = TypeVar('PageT')


class JobPagePrefetcher(Generic[PageT]):
    '''Reads queued jobs' pages with `reader`, one at a time, from the moment Stage 1b queues them.

    `candidates` returns the run's live candidate list; pages are read in its order, except that a
    page Stage 2 is waiting for goes first. Each result, a page or the read's exception, is handed to
    Stage 2 once: asking for the same job again (Stage 2's second read of a failed page) reads it
    again. A LinkedInBlockedError stops all reading: it is raised for that job and for every job
    asked for after it, and nothing is read again.
    '''

    def __init__(self, candidates: Callable[[], list[dict]], reader: Callable[[dict], Awaitable[PageT]],
                 label: str) -> None:
        self._candidates = candidates
        self._reader = reader
        self._label = label                              # names the reader in log lines and errors
        self._results: dict[str, asyncio.Future] = {}    # read or being read, not yet handed over
        self._reading: set[str] = set()                  # a read has started; its result is not yet handed over
        self._waiting: list[str] = []                    # asked for by Stage 2, not yet handed over
        self._handed_over: set[str] = set()              # read again only when asked for again
        self._blocked: LinkedInBlockedError | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f'{self._label} job page prefetch')

    async def stop(self) -> None:
        '''Stop reading. Pages read but never asked for are reported.'''
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        unused = [job_id for job_id, fut in self._results.items()
                  if fut.done() and not fut.cancelled() and job_id not in self._waiting]
        for job_id in unused:
            if (ex := self._results[job_id].exception()) is not None:
                logger.info(f'{self._label} prefetch: job {job_id} was never reached in Stage 2; its read had '
                            f'failed: {type(ex).__name__}: {ex}')
        if unused:
            logger.info(f'{self._label} prefetch: {len(unused)} page(s) read but never reached in Stage 2')

    def _future(self, job_id: str) -> asyncio.Future:
        return self._results.setdefault(job_id, asyncio.get_running_loop().create_future())

    def _next(self) -> dict | None:
        by_id = {str(c.get('job_id')): c for c in self._candidates()}
        for job_id in self._waiting:
            if job_id in by_id and job_id not in self._reading:
                return by_id[job_id]
        for job_id, candidate in by_id.items():
            if job_id not in self._reading and job_id not in self._handed_over:
                return candidate
        return None

    async def _run(self) -> None:
        while True:
            candidate = self._next()
            if candidate is None:
                await asyncio.sleep(JOB_PAGE_PREFETCH_POLL_SECONDS)
                continue
            job_id = str(candidate.get('job_id'))
            self._reading.add(job_id)
            future = self._future(job_id)
            try:
                page = await self._reader(candidate)
            except LinkedInBlockedError as ex:
                self._blocked = ex
                future.set_exception(ex)
                logger.error(f'{self._label} prefetch stopped, no further reads this run: {ex}')
                return
            except Exception as ex:
                # Kept for Stage 2, which logs it with the job when it reaches the job (or stop() does).
                future.set_exception(ex)
            else:
                future.set_result(page)

    async def page(self, candidate: dict) -> PageT:
        '''The job's page, waiting for it to be read if it has not been yet. Raises the read's exception.'''
        job_id = str(candidate.get('job_id'))
        if job_id not in self._waiting:
            self._waiting.append(job_id)
        future = self._future(job_id)
        try:
            if not future.done():
                if self._blocked is not None:
                    raise LinkedInBlockedError(f'job {job_id} not read: {self._label} reading stopped earlier '
                                               f'this run ({self._blocked})')
                if self._task is None or self._task.done():
                    raise RuntimeError(f'job {job_id}: the {self._label} prefetch is not running '
                                       f'({self._task_state()})')
                await asyncio.wait({future, self._task}, return_when=asyncio.FIRST_COMPLETED)
                if not future.done():
                    if self._blocked is not None:
                        raise LinkedInBlockedError(f'job {job_id} not read: {self._label} reading stopped '
                                                   f'({self._blocked})')
                    raise RuntimeError(f'job {job_id}: the {self._label} prefetch ended before reading it '
                                       f'({self._task_state()})')
            return future.result()
        finally:
            if future.done():
                self._results.pop(job_id, None)
                self._reading.discard(job_id)
                self._waiting.remove(job_id)
                self._handed_over.add(job_id)

    def _task_state(self) -> str:
        if self._task is None:
            return 'never started'
        if self._task.cancelled():
            return 'cancelled'
        if self._task.done() and (ex := self._task.exception()) is not None:
            return f'{type(ex).__name__}: {ex}'
        return 'stopped'
