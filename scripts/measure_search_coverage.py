'''Measure Search Coverage: LinkedIn's job emails against what the pipeline listed (weekly).

Harvests LinkedIn's job emails from Gmail through a headless `claude -p` session (only the Gmail
read tools), classifies every emailed job against what the pipeline listed, replays the fresh
never-seen ones in a searched region through `main.py --job-ids`, writes
run_dir/search_coverage/coverage-{date}.md and sends one Telegram summary. agent_meta1 runs it daily
with --if-due. See Measure Search Coverage in docs/requirements.md.

Costs: the harvest is Claude Haiku 5.5, $0.14 for a week's 16 emails (2026-11-25); the replay was
$0.03 for 18 jobs that day, at most
COVERAGE_REPLAY_MAX_JOBS jobs. --no-replay costs only the harvest and the cached location lookups.

Exit status: 0 done (or not due), COVERAGE_EXIT_DEFERRED when a job search run holds the run lock
(nothing is replayed; try again later), 1 on failure.

Usage:
    uv run python scripts/measure_search_coverage.py                 # full check
    uv run python scripts/measure_search_coverage.py --if-due        # skip if one completed recently
    uv run python scripts/measure_search_coverage.py --no-harvest --no-replay   # re-measure, $0
'''

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

from agentic_job_search import preferences, tools_generic
from agentic_job_search import search_coverage as coverage
from agentic_job_search.agent import split_telegram_message
from agentic_job_search.config import (
    COVERAGE_EMAIL_SETTLE_DAYS,
    COVERAGE_EXIT_DEFERRED,
    COVERAGE_INTERVAL_DAYS,
    COVERAGE_REPLAY_MAX_JOBS,
    COVERAGE_REPLAY_TIMEOUT_SECONDS,
    COVERAGE_WINDOW_DAYS,
)
from utils_tools_n_agents_common.logging_setup import setup_logging
from utils_tools_n_agents_common.telegram_client import send_message as telegram_send_message

logger = logging.getLogger('measure_search_coverage')

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = PROJECT_DIR / 'main.py'
REPLAYS_DIR = tools_generic.RUN_DIR / 'replays'
REPLAY_DIR_GLOB = 'job-ids-*'
CLAUDE_EXECUTABLE_ENV = 'CLAUDE_EXECUTABLE'
EXIT_FAILURE = 1
STDERR_TAIL_CHARS = 4000


def is_due(today: date) -> bool:
    if not coverage.LAST_COMPLETE_PATH.exists():
        return True
    last = date.fromisoformat(coverage.LAST_COMPLETE_PATH.read_text(encoding='utf-8').strip())
    return (today - last).days >= COVERAGE_INTERVAL_DAYS


def claude_executable() -> str:
    found = os.environ.get(CLAUDE_EXECUTABLE_ENV) or shutil.which('claude')
    if not found:
        raise RuntimeError(f'No claude executable: set {CLAUDE_EXECUTABLE_ENV} or put claude on PATH '
                           f'(PATH={os.environ.get("PATH")!r})')
    return found


def harvest(window_first: date, today: date, notes: list[str]) -> None:
    '''Fetch new emails into run_dir/search_coverage/linkedin_alerts-{today}.tsv and mark them harvested.'''
    harvested = coverage.load_yaml_mapping(coverage.HARVESTED_THREADS_PATH)
    result = coverage.run_harvest(claude_executable(), window_first, set(harvested))
    for problem in result.problems:
        logger.warning(f'Harvest Alert Emails: {problem}')
    if result.problems:
        notes.append(f'harvest had {len(result.problems)} problem(s); see the run log')
    jobs: list[coverage.AlertJob] = []
    for thread_id, thread in result.threads.items():
        try:
            thread_jobs = coverage.parse_thread(thread)
        except ValueError as ex:
            logger.warning(f'Harvest Alert Emails: thread {thread_id} not parsed, left for the next harvest: {ex}')
            notes.append(f'thread {thread_id} could not be parsed')
            continue
        unlabelled = [job.job_id for job in thread_jobs if not job.title]
        if unlabelled:
            logger.warning(f'Harvest Alert Emails: thread {thread_id} has job(s) with no readable title, '
                           f'company or location: {unlabelled}')
        jobs.extend(thread_jobs)
        harvested[thread_id] = thread_jobs[0].email_date if thread_jobs else today.isoformat()
    alerts_path = coverage.SEARCH_COVERAGE_DIR / f'{coverage.ALERT_JOBS_FILENAME_PREFIX}{datetime.now():%Y%m%d-%H%M%S}.tsv'
    coverage.write_alert_jobs(alerts_path, jobs)
    coverage.save_yaml_mapping(coverage.HARVESTED_THREADS_PATH, harvested)
    logger.info(f'Harvest Alert Emails: {len(jobs)} job link(s) from {len(result.threads)} new thread(s) into {alerts_path}')


def run_replay(jobs: list[coverage.AlertJob], today: date) -> dict[str, dict]:
    '''Replay these jobs through main.py --job-ids; job id -> outcome and rating.'''
    job_ids_path = coverage.SEARCH_COVERAGE_DIR / f'replay-{today:%Y%m%d}.tsv'
    coverage.write_job_ids_file(job_ids_path, jobs)
    started = datetime.now().timestamp()
    command = [sys.executable, str(MAIN_SCRIPT), '--job-ids', str(job_ids_path)]
    logger.info(f'Replay Missed Jobs: {len(jobs)} job(s): {" ".join(command)}')
    try:
        completed = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True,
                                   timeout=COVERAGE_REPLAY_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as ex:
        raise RuntimeError(f'Replay {command} timed out after COVERAGE_REPLAY_TIMEOUT_SECONDS '
                           f'({COVERAGE_REPLAY_TIMEOUT_SECONDS} s)') from ex
    if completed.returncode != 0:
        raise RuntimeError(f'Replay {command} exited {completed.returncode}; stderr tail: '
                           f'{completed.stderr[-STDERR_TAIL_CHARS:]}; stdout tail: {completed.stdout[-STDERR_TAIL_CHARS:]}')
    replay_dirs = [path for path in REPLAYS_DIR.glob(REPLAY_DIR_GLOB) if path.stat().st_mtime >= started]
    if not replay_dirs:
        raise RuntimeError(f'Replay {command} finished but wrote no {REPLAY_DIR_GLOB} directory under {REPLAYS_DIR}')
    replay_dir = max(replay_dirs, key=lambda path: path.stat().st_mtime)
    results = coverage.read_replay_summary(replay_dir)
    for row in results.values():
        row['replay_dir'] = str(replay_dir)
        row['date'] = today.isoformat()
    logger.info(f'Replay Missed Jobs: {len(results)} result(s) in {replay_dir}')
    return results


async def send_summary(text: str) -> None:
    for index, part in enumerate(parts := split_telegram_message(text), 1):
        try:
            await telegram_send_message(part)
        except Exception as ex:
            logger.warning(f'Coverage summary: Telegram send failed (part {index} of {len(parts)}): '
                           f'{type(ex).__name__}: {ex}; the report on disk is complete')
            return


async def measure(args: argparse.Namespace) -> int:
    today = date.today()
    if args.if_due and not is_due(today):
        logger.info(f'Measure Search Coverage: not due (last check within COVERAGE_INTERVAL_DAYS, {COVERAGE_INTERVAL_DAYS})')
        return 0
    window_first, window_last = coverage.window_for(today, COVERAGE_WINDOW_DAYS, COVERAGE_EMAIL_SETTLE_DAYS)
    notes: list[str] = []
    if not args.no_harvest:
        harvest(window_first, today, notes)

    merged = coverage.merge_alert_jobs(coverage.read_alert_jobs(coverage.SEARCH_COVERAGE_DIR))
    in_window = [job for job in merged.values()
                 if window_first.isoformat() <= job.email_date <= window_last.isoformat()]
    region_names = [str(region.get('name', '')) for region in preferences.search_regions() if region.get('name')]
    record = coverage.load_pipeline_record()
    classified = await coverage.classify_jobs(sorted(in_window, key=lambda job: job.email_date), record, region_names)

    replayed = coverage.load_yaml_mapping(coverage.REPLAYED_JOBS_PATH)
    replayed_now: dict[str, dict] = {}
    deferred = False
    if not args.no_replay:
        chosen, left_out = coverage.choose_replay_jobs(classified, set(replayed), COVERAGE_REPLAY_MAX_JOBS)
        if left_out:
            logger.warning(f'Replay Missed Jobs: {left_out} never-seen job(s) beyond COVERAGE_REPLAY_MAX_JOBS '
                           f'({COVERAGE_REPLAY_MAX_JOBS}) left for the next check')
            notes.append(f'{left_out} never-seen job(s) over the replay cap, left for the next check')
        if chosen:
            if (lock := tools_generic.read_run_lock()) is not None:
                logger.warning(f'Replay Missed Jobs: deferred, a job search run holds the lock (pid {lock.get("pid")}, '
                               f'started {lock.get("started_at")}, {lock.get("mode")})')
                notes.append('replay deferred: a job search run was active')
                deferred = True
            else:
                replayed_now = run_replay(chosen, today)
                replayed.update(replayed_now)
                coverage.save_yaml_mapping(coverage.REPLAYED_JOBS_PATH, replayed)

    report = coverage.build_report(classified, replayed, coverage.latest_search_queries(),
                                   (window_first, window_last), notes)
    report_path = coverage.SEARCH_COVERAGE_DIR / f'coverage-{today:%Y%m%d}.md'
    report_path.write_text(report, encoding='utf-8')
    seen, in_region = coverage.coverage_ratio(classified)
    logger.info(f'Measure Search Coverage: {seen}/{in_region} in-region emailed jobs listed; report {report_path}')
    if deferred:
        return COVERAGE_EXIT_DEFERRED
    if not args.no_telegram:
        await send_summary(coverage.build_telegram_summary(classified, replayed_now, report_path, notes))
    if not args.no_harvest:   # --if-due counts a check that read the week's emails
        coverage.LAST_COMPLETE_PATH.write_text(today.isoformat() + '\n', encoding='utf-8')
    return 0


def main() -> int:
    setup_logging('measure_search_coverage', tools_generic.RUN_DIR, level=os.environ.get('LOG_LEVEL', 'INFO'))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--if-due', action='store_true', help='Do nothing if a check completed within COVERAGE_INTERVAL_DAYS')
    parser.add_argument('--no-harvest', action='store_true', help='Use only the emails already harvested')
    parser.add_argument('--no-replay', action='store_true', help='Classify and report only; replay nothing')
    parser.add_argument('--no-telegram', action='store_true', help='Write the report; send no summary')
    args = parser.parse_args()
    try:
        return asyncio.run(measure(args))
    except Exception as ex:
        logger.exception(f'Measure Search Coverage failed: {type(ex).__name__}: {ex}')
        return EXIT_FAILURE


if __name__ == '__main__':
    sys.exit(main())
