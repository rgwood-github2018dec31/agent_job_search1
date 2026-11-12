'''Shadow Local Model: ask a local model every single-turn remote question too, and record both answers.

Run-scoped and off by default (`--shadow-model`). While shadow mode is on, every `chat_openrouter`
prompt asks for a final `CONFIDENCE: <1-5>` line, stripped from the reply before any caller sees it,
and every `chat_openrouter_tool` schema gains an optional `confidence` argument, cut the same way
(asked only in shadow mode since 2026-11-12; before that, on every run). The same prompt and system
message are also queued for the local model; a forced function call's prompt reaches it as a request
for the same arguments as a JSON object in text, since a local sidecar model has no function calls. One background
worker answers the queue in order, since the local server runs one model call at a time, and never
holds up the pipeline. Each pair is appended to `run_dir/shadow_calls/shadow-<run>.jsonl`; what is
still queued when the run ends is written as `pending` for `scripts/shadow_replay.py` to finish.
`scripts/shadow_report.py` compares the two models per call site and by confidence.

The point is data for deciding which tasks a local model could take over (2026-11-04): agreement
with the remote model, overall and when the local model says it is sure. Agreement is not
correctness; the remote answer is a reference, not ground truth.

Verify Source Reference: exempt — the shadow's answer is recorded for comparison and never read by
the pipeline, and a confidence level is a self-report about an answer, not a fact about the input.
'''

import asyncio
import contextlib
import contextvars
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from utils_tools_n_agents_common.mcp_client import unwrap_exception

from agentic_job_search import config as config_module
from agentic_job_search import job_trace
from agentic_job_search.config import LLM_LOCAL_PURPOSE_SHADOW
from agentic_job_search.preferences import RUN_DIR

logger = logging.getLogger(__name__)

SHADOW_CALLS_DIR = RUN_DIR / 'shadow_calls'

# Appended to every remote prompt (and so to every shadow prompt). A last line of its own so it
# can be cut off without touching the answer, whatever shape the answer has.
CONFIDENCE_INSTRUCTION = (
    '\n\nAfter your answer, on a final line of its own, write CONFIDENCE: <1-5> — how sure you are '
    'that your answer is right (1 = a guess, 5 = certain). Nothing may follow that line.'
)
CONFIDENCE_KEY = 'confidence'
CONFIDENCE_MIN, CONFIDENCE_MAX = 1, 5
CONFIDENCE_PROPERTY = {
    'type': 'integer', 'minimum': CONFIDENCE_MIN, 'maximum': CONFIDENCE_MAX,
    'description': f'How sure you are that your answer is right ({CONFIDENCE_MIN} = a guess, {CONFIDENCE_MAX} = certain)',
}
_CONFIDENCE_LINE_RE = re.compile(r'^[ \t*_`]*CONFIDENCE[ \t*_`]*:[ \t*_`]*([1-5])\b[^\n]*$', re.IGNORECASE | re.MULTILINE)

STATUS_PAIRED = 'paired'
STATUS_SHADOW_ERROR = 'shadow_error'
STATUS_PENDING = 'pending'

_queue: asyncio.Queue | None = None
_worker: asyncio.Task | None = None
_in_flight: dict | None = None
_path: Path | None = None
_counts: dict[str, int] = {}


def active() -> bool:
    '''Whether this run shadows remote calls (`--shadow-model` set).'''
    return bool(config_module.SHADOW_MODEL)


def activate(model: str, run_stamp: str) -> Path:
    '''Turn shadow mode on for this run; returns the file the pairs go to.'''
    global _path, _queue, _worker, _in_flight
    config_module.SHADOW_MODEL = model
    _path = SHADOW_CALLS_DIR / f'shadow-{run_stamp}.jsonl'
    _queue, _worker, _in_flight = None, None, None
    _counts.clear()
    _counts.update({STATUS_PAIRED: 0, STATUS_SHADOW_ERROR: 0, STATUS_PENDING: 0})
    return _path


def with_confidence_request(prompt: str) -> str:
    return prompt + CONFIDENCE_INSTRUCTION


def with_confidence_property(parameters: dict) -> dict:
    '''A function's JSON schema with an optional `confidence` argument added (the original is not changed).'''
    return {**parameters, 'properties': {**parameters.get('properties', {}), CONFIDENCE_KEY: CONFIDENCE_PROPERTY}}


def as_text_request(prompt: str, tool_name: str, parameters: dict) -> str:
    '''A forced function call's prompt as the local model gets it: the arguments asked as a JSON object
    in text, its confidence on the final line (the remote model gives it as an argument).'''
    plain = {**parameters, 'properties': {k: v for k, v in parameters.get('properties', {}).items()
                                          if k != CONFIDENCE_KEY}}
    return (f'{prompt}\n\nInstead of calling {tool_name}, reply with ONLY a JSON object holding its arguments, '
            f'matching this JSON schema: {json.dumps(plain, ensure_ascii=False)}' + CONFIDENCE_INSTRUCTION)


def split_confidence(content: str) -> tuple[str, int | None]:
    '''(the answer without its CONFIDENCE line, the level) — or (content unchanged, None) when absent.

    The LAST such line counts, and only that line is removed: a reply whose answer happens to
    mention the word elsewhere keeps it.
    '''
    matches = list(_CONFIDENCE_LINE_RE.finditer(content))
    if not matches:
        return content, None
    last = matches[-1]
    answer = (content[:last.start()] + content[last.end():]).strip()
    return answer, int(last.group(1))


def enqueue(*, call_site: str, system: str, prompt: str, temperature: float | None, remote: dict) -> None:
    '''Queue one remote call for the shadow model. Never raises: shadowing must not fail a run.'''
    global _queue, _worker
    try:
        trace = job_trace.current()
        item = {
            'at': datetime.now().isoformat(timespec='seconds'), 'call_site': call_site,
            'site': trace.site if trace else '', 'job_id': trace.job_id if trace else '',
            'system': system, 'prompt': prompt, 'temperature': temperature, 'remote': remote,
            'queued_monotonic': time.monotonic(),
        }
        if _queue is None:
            _queue = asyncio.Queue()
        if _worker is None or _worker.done():
            # An empty context: the worker must not inherit the Job Trace of whichever job happened
            # to queue the first call, or every shadow call would be recorded in that job's trace.
            _worker = asyncio.get_running_loop().create_task(_work(), context=contextvars.Context())
        _queue.put_nowait(item)
    except Exception as ex:
        logger.warning(f'Shadow: could not queue the {call_site} call for {config_module.SHADOW_MODEL}: '
                       f'{unwrap_exception(ex)}')


async def _work() -> None:
    '''Answer queued calls one at a time with the shadow model, appending each pair.'''
    global _in_flight
    from agentic_job_search import triage  # triage imports this module; resolved at run time
    while True:
        item = await _queue.get()
        _in_flight = item
        model = config_module.SHADOW_MODEL
        started = time.monotonic()
        shadow: dict = {'model': model}
        try:
            response = await triage._generate_local(
                item['prompt'], item['system'], model, None, purpose_slug=LLM_LOCAL_PURPOSE_SHADOW,
                temperature=item['temperature'])
            content, confidence = split_confidence(response)
            shadow.update(content=content, confidence=confidence)
            status = STATUS_PAIRED
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            shadow['error'] = unwrap_exception(ex)
            status = STATUS_SHADOW_ERROR
            logger.warning(f"Shadow: {model} failed on the {item['call_site']} call "
                           f"(job {item['job_id'] or '-'}): {shadow['error']}")
        shadow['seconds'] = round(time.monotonic() - started, 2)
        _in_flight = None
        _write(item, status, shadow)


def _write(item: dict, status: str, shadow: dict | None) -> None:
    record = {key: value for key, value in item.items() if key != 'queued_monotonic'}
    record['status'] = status
    record['queued_s'] = round(time.monotonic() - item['queued_monotonic'], 2)
    record['shadow'] = shadow or {'model': config_module.SHADOW_MODEL}
    _counts[status] = _counts.get(status, 0) + 1
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except OSError as ex:
        logger.warning(f"Shadow: could not append the {item['call_site']} pair to {_path}: {type(ex).__name__}: {ex}")


async def finish() -> dict[str, int] | None:
    '''Stop the worker and write whatever it had not answered as pending. None when shadow mode is off.'''
    global _worker, _in_flight
    if not active():
        return None
    if _worker is not None:
        _worker.cancel()
        # The cancellation this function asked for, not a failure.
        with contextlib.suppress(asyncio.CancelledError):
            await _worker
        _worker = None
    if _in_flight is not None:
        _write(_in_flight, STATUS_PENDING, None)
        _in_flight = None
    while _queue is not None and not _queue.empty():
        _write(_queue.get_nowait(), STATUS_PENDING, None)
    summary = dict(_counts)
    logger.info(f'Shadow ({config_module.SHADOW_MODEL}): {summary[STATUS_PAIRED]} pair(s), '
                f'{summary[STATUS_SHADOW_ERROR]} shadow error(s), {summary[STATUS_PENDING]} pending in {_path}'
                + (' — finish them with scripts/shadow_replay.py' if summary[STATUS_PENDING] else ''))
    return summary
