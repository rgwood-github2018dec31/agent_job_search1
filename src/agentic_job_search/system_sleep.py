'''Keep the machine awake for a run, and say so in the log when it slept anyway.

On 2026-10-14 the lid was closed at 10:05 on battery. The Mac slept, with a few brief maintenance
wakes, until about 10:50. Every call in flight died with a connection reset when it woke. One
Stage 1b query was lost, a 300 s MCP timeout "fired" 18 wall-clock minutes late, and the run log
had a 45-minute hole that read like a hung call. The power log showed the cause
(`pmset -g log`: "Entering Sleep state due to 'Clamshell Sleep'"); the run log did not.

Two things follow. A `caffeinate` assertion stops IDLE sleep for the life of the process. Nothing
a user process can do stops a closed lid on battery, so a watcher notices the wall clock jumping
past its tick and logs it as a WARNING with a run-health alert.
'''

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from agentic_job_search import tools_generic
from agentic_job_search.config import SECONDS_PER_MINUTE, SLEEP_GAP_MIN_SECONDS, SLEEP_WATCH_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

SYSTEM_SLEEP_ALERT_KIND = 'system_sleep'


def hold_awake() -> subprocess.Popen | None:
    '''Start `caffeinate -i -s -w <our pid>`, so the system does not idle-sleep during the run.

    `-w` ties the assertion to this process, so it ends with the run even if the cleanup is never
    reached. `-s` holds only on AC power. Returns None (logged) when there is nothing to start.
    '''
    if sys.platform != 'darwin':
        return None
    caffeinate = shutil.which('caffeinate')
    if caffeinate is None:
        logger.warning('caffeinate not found on PATH: the machine may idle-sleep during this run')
        return None
    try:
        process = subprocess.Popen([caffeinate, '-i', '-s', '-w', str(os.getpid())])
    except OSError as ex:
        logger.warning(f'Could not start caffeinate ({type(ex).__name__}: {ex}): the machine may idle-sleep '
                       'during this run')
        return None
    logger.info(f'Holding a no-idle-sleep assertion for the run (caffeinate pid {process.pid}); a lid '
                'closed on battery still sleeps the machine')
    return process


def release_awake(process: subprocess.Popen | None) -> None:
    '''End the assertion started by hold_awake (its own child pid, nothing else).'''
    if process is not None and process.poll() is None:
        process.terminate()


def record_gap(gap_seconds: float, resumed_at: datetime) -> None:
    '''Log one gap the process did not run through, and raise a run-health alert for it.'''
    started = datetime.fromtimestamp(resumed_at.timestamp() - gap_seconds)
    query = tools_generic._current_query or '(between queries)'
    detail = (f'{started:%H:%M:%S}–{resumed_at:%H:%M:%S} ({gap_seconds / SECONDS_PER_MINUTE:.0f} min) '
              f'during {query}')
    logger.warning(
        f'The process did not run for {gap_seconds:.0f}s ({detail}) — the machine slept (lid closed '
        'or idle) or the event loop was blocked; calls in flight across the gap may fail on resume'
    )
    tools_generic._ui_alerts.append({
        'kind': SYSTEM_SLEEP_ALERT_KIND, 'query': query, 'region': '(all)',
        'detail': detail, 'seconds': round(gap_seconds),
    })


async def watch_for_sleep(interval_seconds: float = SLEEP_WATCH_INTERVAL_SECONDS) -> None:
    '''Tick every interval; a wall-clock step longer than the tick plus SLEEP_GAP_MIN_SECONDS is a gap.

    It is judged on the wall clock, not the monotonic one: on macOS the monotonic clock can stop
    while the system sleeps, so it would not see the gap.
    '''
    last = time.time()
    while True:
        await asyncio.sleep(interval_seconds)
        now = time.time()
        if (gap := now - last - interval_seconds) >= SLEEP_GAP_MIN_SECONDS:
            record_gap(gap, datetime.fromtimestamp(now))
        last = now
