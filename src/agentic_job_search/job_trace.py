'''Job Trace: every step one job went through in Stage 2, recorded as it happens.

`evaluate_all_candidates` opens a trace per candidate (`start`); code anywhere below it calls
`record(kind, **data)` without being handed the job, because the trace lives in a ContextVar. A
call outside any job (Stage 1b, the reference summary) records nothing. Each step is appended to
the job's file (`tools_generic.job_trace_path`, beside its raw posting) at once, so a crash mid-job
still leaves the steps that led to it. A step made after the job has finished, such as the
notification sent at the end of Stage 2, is added to the job's `JobTrace` directly.
'''

import contextvars
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar['JobTrace | None'] = contextvars.ContextVar('job_trace', default=None)


class JobTrace:
    '''The steps of one job, kept in memory and appended to its JSONL file.'''

    def __init__(self, path: Path, site: str, job_id: str):
        self.path = path
        self.site = site
        self.job_id = job_id
        self.steps: list[dict] = []
        self.started = time.monotonic()

    def add(self, kind: str, data: dict[str, Any]) -> None:
        step = {'kind': kind, 'at': datetime.now().isoformat(timespec='milliseconds'),
                'elapsed_s': round(time.monotonic() - self.started, 3), **data}
        self.steps.append(step)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(step, ensure_ascii=False, default=str) + '\n')
        except OSError as ex:
            # The in-memory copy still feeds this run's audit; only the on-disk copy is lost.
            logger.warning(f'Job trace for {self.site}/{self.job_id}: could not append a {kind!r} step to '
                           f'{self.path}: {type(ex).__name__}: {ex}')


def start(path: Path, site: str, job_id: str) -> tuple[JobTrace, contextvars.Token]:
    '''Begin recording a job; pass the token to `finish`. An existing file is replaced.'''
    if path.exists():
        path.unlink()
    trace = JobTrace(path, site, job_id)
    return trace, _current.set(trace)


def finish(token: contextvars.Token) -> None:
    _current.reset(token)


def current() -> JobTrace | None:
    return _current.get()


def record(kind: str, **data: Any) -> None:
    '''Append one step to the current job's trace; a no-op outside a job.'''
    trace = _current.get()
    if trace is not None:
        trace.add(kind, data)


def caller_name(depth: int = 2) -> str:
    """The function `depth` frames up: who made an LLM call, for a trace step's `call_site`.

    Read from the stack so the shared wrappers (`chat_openrouter`, `generate_local`) need no new
    parameter at their ~20 call sites. Depth 2 is the wrapper's own caller.
    """
    frame = sys._getframe(depth)
    return frame.f_code.co_name if frame is not None else '?'


def load(path: Path) -> list[dict]:
    '''A trace's steps from its JSONL file. A bad line raises, naming the file and line.'''
    steps = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError as ex:
            raise ValueError(f'{path}:{number} is not a JSON trace step ({ex}): {line[:200]!r}') from ex
    return steps
