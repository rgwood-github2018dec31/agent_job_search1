'''Replay retained job pages through the extractor and Verify Source Reference, live.

Why this is a tracked script
----------------------------
Verify Source Reference rejects any extract field whose citation code cannot find in the page.
Too strict a matcher, or a prompt that gets sloppy quotes, turns that into silent false
rejections. A field rejected as "absent" can let a job through a gate it should have hit
(a rejected `language_requirement`), so the rate of rejections per field is the number to
watch when the extractor model, its prompt, or the matcher changes. This measures it on real pages.

It also compares each derive_* regex fallback on the raw page against the same regex on the
model's condensed description, which is what they read before 2026-10-19.

Usage:
    uv run python scripts/replay_evidence.py                       # newest capture day, 20 pages
    uv run python scripts/replay_evidence.py --date 2026-10-16 --limit 40

Needs the OpenRouter MCP tool server on :8006 and costs about a tenth of a cent per page (one
extraction plus, only for citations regex misses, one relocation call). Reads only: nothing is
saved, rated or notified. The report goes to run_dir/replays/.
'''

import argparse
import asyncio
import collections
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agentic_job_search import evidence  # noqa: E402
from agentic_job_search.agent import (  # noqa: E402
    PAGE_EXTRACTOR_INSTRUCTIONS,
    derive_agency_posting,
    derive_education_requirement,
    derive_residency_scope,
    derive_workplace_type,
    new_stage_stats,
    verify_extract,
)
from agentic_job_search.extract_openrouter import extract_job_page_openrouter  # noqa: E402

logger = logging.getLogger('replay_evidence')

RUN_DIR = Path(__file__).parent.parent / 'run_dir'
DEFAULT_REPLAY_LIMIT = 20
FRONT_MATTER_DELIMITER = '---\n'
DERIVERS = {
    'residency_scope': derive_residency_scope,
    'workplace_type': derive_workplace_type,
    'education_requirement': derive_education_requirement,
    'is_agency': derive_agency_posting,
}


def load_capture(path: Path) -> tuple[dict, str]:
    '''(front matter, page text) of one retained capture. Raises naming the file when malformed.'''
    text = path.read_text(encoding='utf-8')
    if not text.startswith(FRONT_MATTER_DELIMITER):
        raise ValueError(f'{path} has no front matter; is it a raw posting capture?')
    _, header, body = text.split(FRONT_MATTER_DELIMITER, 2)
    return yaml.safe_load(header) or {}, body


def candidate_for(header: dict, page_text: str) -> dict:
    lines = [line for line in page_text.splitlines() if line.strip()]
    return {
        'site': 'linkedin', 'job_id': str(header.get('job_id') or ''), 'url': header.get('url') or '',
        'company': header.get('company') or '', 'title': lines[1] if len(lines) > 1 else '',
        'date_posted': '', 'snippet': '',
    }


async def replay_one(path: Path) -> dict:
    header, page_text = load_capture(path)
    candidate = candidate_for(header, page_text)
    stats = new_stage_stats()
    extract = await extract_job_page_openrouter(candidate, page_text, stats, PAGE_EXTRACTOR_INSTRUCTIONS)
    if extract is None:
        return {'path': path, 'candidate': candidate, 'failed': True}
    before = dict(extract)
    derived_from_description = {name: fn(dict(before, page_text='')) for name, fn in DERIVERS.items()}
    extract['page_text'] = page_text
    statuses = await verify_extract(candidate, extract, stats)
    derived_from_page = {name: fn(extract) for name, fn in DERIVERS.items()}
    return {'path': path, 'candidate': candidate, 'failed': False, 'before': before, 'after': extract,
            'statuses': statuses, 'derived_from_description': derived_from_description,
            'derived_from_page': derived_from_page, 'cost': stats['cost']}


def report(results: list[dict]) -> str:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    lines = ['# Verify Source Reference replay', '', f'{len(results)} page(s), '
             f"${sum(r.get('cost', 0.0) for r in results):.4f}", '']
    for result in results:
        for field, status in (result.get('statuses') or {}).items():
            counts[field][status] += 1
    lines += ['## Field outcomes', '', '| field | ' + ' | '.join(
        (evidence.STATUS_REGEX, evidence.STATUS_RELOCATED, 'read_by_code', evidence.STATUS_REJECTED)) + ' |',
        '|---|---|---|---|---|']
    for field in sorted(counts):
        c = counts[field]
        lines.append(f'| {field} | {c[evidence.STATUS_REGEX]} | {c[evidence.STATUS_RELOCATED]} | '
                     f'{c["read_by_code"]} | {c[evidence.STATUS_REJECTED]} |')
    lines += ['', '## Rejections (check each: hallucination, or a citation the matcher missed?)', '']
    for result in results:
        for field, status in (result.get('statuses') or {}).items():
            if status == evidence.STATUS_REJECTED:
                quote = (result['before'].get('quotes') or {}).get(field, '')
                lines.append(f"- `{result['path'].name}` {field}={result['before'].get(field)!r} cited {quote!r}")
    lines += ['', '## derive_* on the page vs on the condensed description', '']
    for result in results:
        if result['failed']:
            lines.append(f"- `{result['path'].name}`: extraction FAILED")
            continue
        changed = {name: (result['derived_from_description'][name], result['derived_from_page'][name])
                   for name in DERIVERS if result['derived_from_description'][name] != result['derived_from_page'][name]}
        if changed:
            lines.append(f"- `{result['path'].name}`: " + ', '.join(
                f'{name} {old!r} -> {new!r}' for name, (old, new) in changed.items()))
    return '\n'.join(lines) + '\n'


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--date', help='capture day under run_dir/raw_postings (default: newest)')
    parser.add_argument('--limit', type=int, default=DEFAULT_REPLAY_LIMIT,
                        help=f'pages to replay (default {DEFAULT_REPLAY_LIMIT})')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')

    days = sorted(p for p in (RUN_DIR / 'raw_postings').iterdir() if p.is_dir())
    day = RUN_DIR / 'raw_postings' / args.date if args.date else days[-1]
    paths = sorted(day.glob('*.txt'))[:args.limit]
    if not paths:
        print(f'No captures under {day}')
        return 1
    results = []
    for number, path in enumerate(paths, start=1):
        print(f'[{number}/{len(paths)}] {path.name}', flush=True)
        results.append(await replay_one(path))
    text = report(results)
    out = RUN_DIR / 'replays' / f"evidence-{datetime.now():%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(text)
    print(f'Report written to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
