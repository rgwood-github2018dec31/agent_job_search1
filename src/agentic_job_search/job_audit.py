'''Write Job Audit: one job per run, rendered as a PDF of everything that decided its rating.

Every Stage 2 job leaves a Job Trace (job_trace.py). At the end of a run one job that reached the
rater is chosen at random (or the one named with --job-audit) and its trace is rendered by
scripts/job_trace_report.py, which adds the run's other artifacts around it. reportlab is not a
project dependency: the render runs as `uv run --with reportlab`, so a run never imports it and a
failed render costs the PDF, never the run.
'''

import asyncio
import logging
import random
from datetime import date
from pathlib import Path

from agentic_job_search import job_trace
from agentic_job_search import tools_generic as tools_module
from agentic_job_search.config import JOB_AUDIT_RENDER_TIMEOUT_SECONDS
from agentic_job_search.text_budget import quote_for_error

logger = logging.getLogger(__name__)

JOB_AUDITS_DIR = tools_module.RUN_DIR / 'job_audits'
RENDER_SCRIPT = tools_module.PROJECT_DIR / 'scripts' / 'job_trace_report.py'


def find_trace(candidate: dict) -> Path | None:
    '''The newest Job Trace for this job. Searched for rather than computed from today's date, so a
    run that crosses midnight still finds the traces it wrote before it.'''
    name = tools_module.job_trace_path(candidate).name
    found = sorted(tools_module.RAW_POSTINGS_DIR.glob(f'*/{name}'))
    return found[-1] if found else None


def reached_the_rater(candidate: dict) -> bool:
    '''True when the job's trace holds a rating step, i.e. it was rated, not stopped by a gate.'''
    path = find_trace(candidate)
    return path is not None and any(step['kind'] == 'rating' for step in job_trace.load(path))


def choose_job(candidates: list[dict], requested_job_id: str | None, seed: int) -> dict | None:
    '''The job to audit: the one requested (when it has a trace), else a random rated one.'''
    if requested_job_id:
        match = next((c for c in candidates if str(c.get('job_id')) == requested_job_id), None)
        if match is None or find_trace(match) is None:
            logger.warning(f'Job audit: job {requested_job_id} was not evaluated this run, so it has no '
                           f'Job Trace; choosing a rated job at random instead')
        else:
            return match
    rated = [c for c in candidates if reached_the_rater(c)]
    if not rated:
        logger.info('Job audit: no job reached the rater this run; nothing to audit')
        return None
    chosen = random.Random(seed).choice(rated)
    logger.info(f"Job audit: chose job {chosen['job_id']} ({chosen['company']}) from {len(rated)} rated "
                f'job(s), seed {seed}')
    return chosen


def audit_path_for(candidate: dict, run_date: date) -> Path:
    name = '-'.join(filter(None, [
        run_date.isoformat(), tools_module.underscorify(str(candidate.get('company') or '')),
        tools_module.underscorify(str(candidate.get('title') or '')), str(candidate.get('job_id') or ''),
    ]))
    return JOB_AUDITS_DIR / f'{name}.pdf'


def render_command(candidate: dict, run_date: date, out: Path) -> list[str]:
    trace = find_trace(candidate)
    # --date is the capture day, which names the raw posting the report reads beside the trace.
    capture_day = trace.parent.name if trace else run_date.isoformat()
    return ['uv', 'run', '--with', 'reportlab', 'python', str(RENDER_SCRIPT),
            '--job-id', str(candidate['job_id']), '--date', capture_day, '--out', str(out),
            *(['--trace', str(trace)] if trace else [])]


async def render(candidate: dict, run_date: date) -> Path | None:
    '''Render the Job Audit PDF; None (after a WARNING carrying the command and its output) on failure.'''
    out = audit_path_for(candidate, run_date)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = render_command(candidate, run_date, out)
    try:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=str(tools_module.PROJECT_DIR),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), JOB_AUDIT_RENDER_TIMEOUT_SECONDS)
        except TimeoutError:
            process.kill()   # the render's own child, started just above; never another process
            await process.wait()
            raise
    except Exception as ex:
        logger.warning(f'Job audit render failed to run {command}: {type(ex).__name__}: {ex}')
        return None
    if process.returncode != 0 or not out.is_file():
        logger.warning(
            f'Job audit render failed: {command} exited {process.returncode}; '
            f'stdout: {quote_for_error(stdout.decode(errors="replace"), "job audit render stdout")!r}; '
            f'stderr: {quote_for_error(stderr.decode(errors="replace"), "job audit render stderr")!r}')
        return None
    logger.info(f"Job audit for job {candidate['job_id']} ({candidate['company']}): {out}")
    return out
