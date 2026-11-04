'''Answer a shadow recording's pending calls with a local model (Shadow Local Model).

A run's background worker answers only what it can before the run ends; the rest is recorded as
`pending`. This asks the local model those calls, one at a time, and writes a COMPLETE copy of the
recording beside the original (the original is never modified):

    uv run python scripts/shadow_replay.py run_dir/shadow_calls/shadow-<run>.jsonl
    uv run python scripts/shadow_replay.py <file> --errors          # also retry shadow errors
    uv run python scripts/shadow_replay.py <file> --all --model M   # every call again, with model M

Costs nothing but local compute. Then compare with scripts/shadow_report.py.
'''

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from agentic_job_search import shadow, triage
from agentic_job_search.config import LLM_LOCAL_PURPOSE_SHADOW
from utils_tools_n_agents_common.mcp_client import unwrap_exception

logger = logging.getLogger('shadow_replay')

_UNSAFE_NAME_CHARS_RE = re.compile(r'[^A-Za-z0-9._-]+')


def _needs_answer(record: dict, retry_errors: bool, redo_all: bool) -> bool:
    if redo_all:
        return True
    return record['status'] == shadow.STATUS_PENDING or (retry_errors and record['status'] == shadow.STATUS_SHADOW_ERROR)


async def _answer(record: dict, model: str) -> dict:
    started = time.monotonic()
    answered = dict(record)
    try:
        response = await triage._generate_local(
            record['prompt'], record['system'], model, None, purpose_slug=LLM_LOCAL_PURPOSE_SHADOW,
            temperature=record.get('temperature'))
        content, confidence = shadow.split_confidence(response)
        answered['shadow'] = {'model': model, 'content': content, 'confidence': confidence}
        answered['status'] = shadow.STATUS_PAIRED
    except Exception as ex:
        answered['shadow'] = {'model': model, 'error': unwrap_exception(ex)}
        answered['status'] = shadow.STATUS_SHADOW_ERROR
        logger.warning(f"{model} failed on the {record['call_site']} call (job {record.get('job_id') or '-'}): "
                       f"{answered['shadow']['error']}")
    answered['shadow']['seconds'] = round(time.monotonic() - started, 2)
    answered['replayed_at'] = datetime.now().isoformat(timespec='seconds')
    return answered


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--model', help="local model to ask (default: each record's own shadow model)")
    parser.add_argument('--errors', action='store_true', help='also retry calls whose shadow answer failed')
    parser.add_argument('--all', action='store_true', help='ask every call again (needs --model)')
    args = parser.parse_args()
    if args.all and not args.model:
        parser.error('--all needs --model')
    records = [json.loads(line) for line in args.recording.read_text(encoding='utf-8').splitlines() if line.strip()]
    todo = [i for i, r in enumerate(records) if _needs_answer(r, args.errors, args.all)]
    suffix = _UNSAFE_NAME_CHARS_RE.sub('_', args.model) if args.model else 'own-model'
    out = args.recording.with_name(f'{args.recording.stem}.replay-{suffix}-{datetime.now():%Y%m%d_%H%M%S}.jsonl')
    logger.info(f'{len(todo)} of {len(records)} call(s) to answer; writing {out}')
    for done, index in enumerate(todo, 1):
        record = records[index]
        model = args.model or record['shadow']['model']
        records[index] = await _answer(record, model)
        logger.info(f"[{done}/{len(todo)}] {record['call_site']} (job {record.get('job_id') or '-'}): "
                    f"{records[index]['status']} in {records[index]['shadow']['seconds']}s")
    with out.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    logger.info(f'Wrote {out}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname).1s %(message)s')
    asyncio.run(main())
