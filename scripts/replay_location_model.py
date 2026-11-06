'''Replay the Job Location model over every retained page, beside what the pipeline decided at the time.

Why this is a tracked script
----------------------------
Phase 1 of the location remodel (docs/requirements.md: Ask Location Model, Decide Location
Eligibility, Adjudicate Location). The new model changes no live behaviour until this report has been
reviewed: for each retained page it asks the model, verifies its quotes, decides in code, adjudicates
what code cannot decide, and sets the result beside the pipeline's own decision read from the run logs
(rejected by a geographic hard rule, or kept and rated). Every disagreement is listed with its reason.

`--repeat 2` asks each page twice, to measure the run-to-run variation of the verdict.

Usage:
    uv run python scripts/replay_location_model.py              # every retained page
    uv run python scripts/replay_location_model.py --limit 20 --repeat 2
    uv run python scripts/replay_location_model.py --job-ids 4478266275 4482711232

Needs the OpenRouter MCP tool server on :8006. Costs about a tenth of a cent per page and repeat.
Reads only; writes run_dir/replays/location-model-{timestamp}.json.
'''
import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from agentic_job_search import location_model
from agentic_job_search.linkedin_page import header_card
from replay_evidence import RUN_DIR, load_capture

logger = logging.getLogger('replay_location_model')

DEFAULT_REPEAT = 1
DEFAULT_CONCURRENT_PAGES = 8
CORPUS_GLOBS = ('replay_corpus/*/linkedin-*.txt', 'raw_postings/*/linkedin-*.txt')
HEADER_LOCATION_LINE = 2   # company, title, location: the capture's third non-empty line
_JOB_ID_RE = re.compile(r'linkedin-(\d+)\.txt$')
_RATING_RE = re.compile(r'Rating \[\d+/\d+\]: job (\d+): (\d)/5 (?:by hard rule \((.*?)\) —|via )')
_CAPPED_RE = re.compile(r'Rating capped: .*?: \d → \d \((.*?)\)')
_GEOGRAPHIC_RULE_RE = re.compile(r'located in|relocation|residen|sponsor|authori[sz]', re.IGNORECASE)


def captures() -> dict[str, Path]:
    '''job id -> its capture, a pinned replay_corpus copy first.'''
    found: dict[str, Path] = {}
    for pattern in CORPUS_GLOBS:
        for path in sorted(RUN_DIR.glob(pattern)):
            if path.name.endswith('.rater_posting.txt'):
                continue
            match = _JOB_ID_RE.search(path.name)
            if match and match.group(1) not in found:
                found[match.group(1)] = path
    return found


def old_decisions() -> dict[str, dict]:
    '''job id -> the pipeline's last decision on it, from every run log.'''
    decisions: dict[str, dict] = {}
    for log in sorted((RUN_DIR / 'logs').glob('run-*.log')):
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            if match := _RATING_RE.search(line):
                job_id, rating, rule = match.groups()
                geographic = bool(rule and _GEOGRAPHIC_RULE_RE.search(rule))
                decisions[job_id] = {'rating': int(rating), 'hard_rule': rule or '', 'log': log.name,
                                     'verdict': 'rejected' if geographic else 'kept'}
    return decisions


def header_of(path: Path, body: str) -> tuple[str, str]:
    html = path.with_suffix('.html')
    if html.exists():
        card = header_card(BeautifulSoup(html.read_text(encoding='utf-8'), 'lxml'))
        if card.location:
            return card.location, card.workplace
    lines = [line for line in body.splitlines() if line.strip()]
    return (lines[HEADER_LOCATION_LINE] if len(lines) > HEADER_LOCATION_LINE else ''), ''


async def replay_one(job_id: str, path: Path, profile, repeat: int, gate: asyncio.Semaphore) -> dict:
    front, body = load_capture(path)
    location, workplace = header_of(path, body)
    subject = f"job {job_id} {front.get('company') or ''}"
    runs = []
    async with gate:
        for _ in range(repeat):
            stats: dict = {}
            started = time.monotonic()
            try:
                job = await location_model.ask(body, location, workplace, subject, stats)
                decision = await location_model.decide(job, profile)
                if decision.verdict == location_model.UNDETERMINED:
                    decision.adjudicated = await location_model.adjudicate(body, job, decision, profile, subject, stats)
                runs.append({'job': asdict(job), 'decision': asdict(decision), 'cost': stats.get('cost', 0.0),
                             'seconds': round(time.monotonic() - started, 1)})
            except Exception as ex:
                logger.warning(f'{subject}: {type(ex).__name__}: {ex}')
                runs.append({'error': f'{type(ex).__name__}: {ex}', 'cost': stats.get('cost', 0.0)})
    logger.info(f'{subject}: {[r.get("decision", {}).get("verdict", "error") for r in runs]}')
    return {'job_id': job_id, 'company': front.get('company') or '', 'capture': str(path.relative_to(RUN_DIR)),
            'header_location': location, 'header_workplace': workplace, 'runs': runs}


def final_verdict(run: dict) -> str:
    decision = run.get('decision') or {}
    adjudicated = decision.get('adjudicated') or {}
    if decision.get('verdict') == location_model.UNDETERMINED:
        return f"undetermined->{adjudicated.get('verdict', '?')}"
    return decision.get('verdict', 'error')


KEPT_VERDICTS = frozenset({location_model.ELIGIBLE, location_model.ELIGIBLE_CAPPED,
                          'undetermined->eligible', 'undetermined->uncertain'})


def kept(verdict: str) -> bool:
    '''Whether the job would go on to be rated: an uncertain adjudication keeps it, with a warning.'''
    return verdict in KEPT_VERDICTS


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT)
    parser.add_argument('--job-ids', nargs='*', default=[])
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT_PAGES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname).1s %(name)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)

    pages = captures()
    if args.job_ids:
        pages = {j: pages[j] for j in args.job_ids if j in pages}
    if args.limit:
        pages = dict(list(pages.items())[:args.limit])
    decisions = old_decisions()
    profile = await location_model.load_profile()
    gate = asyncio.Semaphore(args.concurrent)
    logger.info(f'Replaying {len(pages)} page(s) x{args.repeat}')
    results = await asyncio.gather(*(replay_one(j, p, profile, args.repeat, gate) for j, p in pages.items()))

    for result in results:
        result['old'] = decisions.get(result['job_id'], {})
        verdicts = [final_verdict(r) for r in result['runs']]
        result['new_verdicts'] = verdicts
        result['stable'] = len(set(verdicts)) == 1
        old = result['old'].get('verdict')
        result['disagrees'] = bool(old) and any(kept(v) != (old == 'kept') for v in verdicts)

    out = RUN_DIR / 'replays' / f'location-model-{datetime.now():%Y%m%d-%H%M%S}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'profile': {k: sorted(v) if isinstance(v, frozenset) else list(v)
                                           for k, v in asdict(profile).items()},
                               'results': results}, indent=1, ensure_ascii=False), encoding='utf-8')
    cost = sum(r.get('cost', 0.0) for res in results for r in res['runs'])
    tally: dict[str, int] = {}
    for result in results:
        tally[result['new_verdicts'][0]] = tally.get(result['new_verdicts'][0], 0) + 1
    print(f'{len(results)} page(s), ${cost:.4f}; first-run verdicts {dict(sorted(tally.items()))}')
    print(f"unstable across repeats: {sum(not r['stable'] for r in results)}; "
          f"disagree with the pipeline: {sum(r['disagrees'] for r in results)}")
    print(f'report: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
