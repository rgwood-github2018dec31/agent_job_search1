'''Plain text vs Markdown, for the extractor and the rater, on retained job pages, live, with repeats.

Why this is a tracked script
----------------------------
Since 2026-10-30 the extractor reads the page as Markdown and the rater reads the posting as
Markdown (JobPage.markdown), on the expectation that headings, bullets and bold let a model tell
"Requirements" from "Nice to have". A single text-vs-Markdown run cannot show that: on 20 pages
(2026-10-28) the extractor changed 17 field values between the two inputs and 15 between two runs
on the same input. So every input here is run `--repeat` times, and a difference counts only
against the variation of the same input run again.

Per page:
- Extractor: plain text and Markdown, each `--repeat` times, each verified as a run verifies it
  (verify_extract). Reported per field: pages whose value differs between two runs of the same
  input (noise) and between the inputs (effect), plus rejected citations, cost and size.
- Rater: one Markdown extract, resolved as a run resolves it (resolve_listed_location,
  resolve_extract_fields), then rated `--repeat` times with the posting as plain text and as
  Markdown; the rest of the rater's input is identical. Reported: rating flips within and between
  the two forms, ratings at or above RATING_NOTIFICATION_THRESHOLD, and every page where each form's repeats
  agree with each other but not with the other form, with the rater's reasoning.

The raw rating is compared, before hard rules and caps, since those read fields, not the posting.

Usage:
    uv run python scripts/replay_markdown_ab.py                     # every retained page, 2 repeats
    uv run python scripts/replay_markdown_ab.py --date 2026-10-29 --limit 10 --repeat 3
    uv run python scripts/replay_markdown_ab.py --noise-only --extract-temperature 0   # does a setting steady the extractor?

Needs the OpenRouter MCP tool server on :8006. Costs about a tenth of a cent per extraction and
less per rating: about half a cent per page at `--repeat 2`. Reads only: nothing is saved, rated
into the job records or notified. The report and the raw results go to run_dir/replays/.
'''
import argparse
import asyncio
import collections
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from agentic_job_search import evidence, extract_openrouter
from agentic_job_search.agent import (
    EXTRACT_CITED_FIELDS,
    PAGE_EXTRACTOR_INSTRUCTIONS,
    attach_page,
    build_evaluator_prompt,
    build_reference_summary,
    format_extract_text,
    new_stage_stats,
    rate_job,
    resolve_extract_fields,
    resolve_listed_location,
    verify_extract,
)
from agentic_job_search.config import EXTRACT_RESUBMIT_UNGROUNDED, RATING_NOTIFICATION_THRESHOLD
from agentic_job_search.extract_openrouter import extract_job_page_openrouter
from agentic_job_search.linkedin_page import JobPage, header_card
from replay_evidence import RAW_POSTINGS_DIR, REPLAY_CORPUS_DIR, RUN_DIR, candidate_for, inputs_from_html, load_capture

logger = logging.getLogger('replay_markdown_ab')

FORMS = ('text', 'markdown')
DEFAULT_REPEAT = 2
DEFAULT_CONCURRENT_PAGES = 8
VALUE_SHOWN_MAX_CHARS = 80
REASONING_SHOWN_MAX_CHARS = 300


def _value(value: object) -> str:
    '''A field value as compared: empty, False and [] are the same absent value.'''
    if value in (None, False, '', [], {}):
        return ''
    return json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value).strip()


async def extract_once(path: Path, candidate: dict, plain: str, markdown: str, header, form: str) -> dict:
    '''One extraction of the page in `form`, verified as a run verifies it.'''
    extraction, evidence_stats = new_stage_stats(), new_stage_stats()
    model_input = markdown if form == 'markdown' else plain
    extract = await extract_job_page_openrouter(candidate, model_input, extraction, PAGE_EXTRACTOR_INSTRUCTIONS)
    if extract is None:
        return {'failed': True}
    attach_page(candidate, extract, JobPage(text=plain, source='dom', header=header,
                                            markdown=markdown if form == 'markdown' else ''))
    statuses = await verify_extract(candidate, extract, evidence_stats, use_llm=not EXTRACT_RESUBMIT_UNGROUNDED)
    return {
        'failed': False, 'extract': extract,
        'fields': {f: _value(extract.get(f)) for f in EXTRACT_CITED_FIELDS},
        'rejected': sorted(f for f, st in statuses.items() if st == evidence.STATUS_REJECTED),
        'cost': extraction['cost'] + evidence_stats['cost'], 'chars': len(model_input),
    }


async def rate_both(candidate: dict, extract: dict, header, evaluator_prompt: str, repeat: int) -> dict:
    '''The same extract rated `repeat` times with the posting as plain text and as Markdown.'''
    salary_stats, rating_stats = new_stage_stats(), new_stage_stats()
    extract = dict(extract)
    await resolve_listed_location(extract, header.location)
    await resolve_extract_fields(extract, salary_stats)
    texts = {
        'text': format_extract_text(candidate, dict(extract, description_markdown='')),
        'markdown': format_extract_text(candidate, extract),
    }
    ratings = {form: [] for form in FORMS}
    for _ in range(repeat):
        for form in FORMS:
            result = await rate_job(evaluator_prompt, texts[form], rating_stats)
            ratings[form].append({'rating': result.get('rating'), 'reasoning': result.get('reasoning', ''),
                                  'rated_by': result.get('rated_by')})
    return {'ratings': ratings, 'cost': rating_stats['cost'] + salary_stats['cost'],
            'chars': {form: len(text) for form, text in texts.items()},
            'same_input': texts['text'] == texts['markdown']}


async def replay_page(path: Path, evaluator_prompt: str, repeat: int, gate: asyncio.Semaphore,
                      forms: tuple[str, ...] = FORMS, rate: bool = True) -> dict | None:
    async with gate:
        try:
            plain, markdown = inputs_from_html(path)
        except FileNotFoundError as ex:
            logger.warning(f'Skipping {path.name}: {ex}')
            return None
        front, _ = load_capture(path)
        candidate = candidate_for(front, plain)
        html = path.with_suffix('.html')
        header = header_card(BeautifulSoup(html.read_text(encoding='utf-8'), 'lxml'))
        page = {'path': str(path), 'job_id': candidate['job_id'], 'company': candidate.get('company', '')}
        try:
            page['extracts'] = {form: [await extract_once(path, candidate, plain, markdown, header, form)
                                       for _ in range(repeat)] for form in forms}
            rated_from = next((r for r in page['extracts']['markdown'] if not r['failed']), None)
            page['rating'] = (await rate_both(candidate, rated_from['extract'], header, evaluator_prompt, repeat)
                              if rated_from and rate else None)
        except Exception as ex:
            logger.warning(f'{path.name}: replay failed: {type(ex).__name__}: {ex}')
            page['error'] = f'{type(ex).__name__}: {ex}'
        for runs in page.get('extracts', {}).values():
            for run in runs:
                run.pop('extract', None)
        print(f"{path.name}: done{' (error)' if 'error' in page else ''}", flush=True)
        return page


def _differs(page: dict, field: str, a: tuple[str, int], b: tuple[str, int]) -> bool:
    runs = page['extracts']
    x, y = runs[a[0]][a[1]], runs[b[0]][b[1]]
    return not x['failed'] and not y['failed'] and x['fields'][field] != y['fields'][field]


def extractor_report(pages: list[dict], repeat: int, forms: tuple[str, ...] = FORMS) -> list[str]:
    pages = [p for p in pages if 'extracts' in p]
    noise_pairs = [((form, 0), (form, i)) for form in forms for i in range(1, repeat)]
    effect_pairs = [(('text', i), ('markdown', i)) for i in range(repeat)] if 'text' in forms else noise_pairs

    def count(field: str, pairs: list) -> float:
        '''Pages differing, averaged over the pairs, so noise and effect are on the same scale.'''
        return sum(sum(_differs(p, field, a, b) for p in pages) for a, b in pairs) / len(pairs)

    lines = ['## Extractor', '',
             f'{len(pages)} pages, {repeat} runs per input. Pages whose value differs, averaged over run pairs: '
             '*same input* compares two runs of one input (noise), *text vs Markdown* compares the inputs.', '',
             '| field | same input (noise) | text vs Markdown |', '|---|---|---|']
    total_noise = total_effect = 0.0
    for field in EXTRACT_CITED_FIELDS:
        noise, effect = count(field, noise_pairs), count(field, effect_pairs)
        total_noise += noise
        total_effect += effect
        if noise or effect:
            lines.append(f'| {field} | {noise:.1f} | {effect:.1f} |')
    lines.append(f'| **all fields** | **{total_noise:.1f}** | **{total_effect:.1f}** |')
    lines += ['', f'Field values compared per pair: {len(pages) * len(EXTRACT_CITED_FIELDS)}.', '']
    for form in forms:
        runs = [r for p in pages for r in p['extracts'][form]]
        ok = [r for r in runs if not r['failed']]
        rejected = collections.Counter(f for r in ok for f in r['rejected'])
        lines.append(f"- {form}: {len(runs) - len(ok)} failed extractions, {sum(rejected.values())} rejected citations "
                     f"{dict(rejected) or ''}, ${sum(r['cost'] for r in ok):.4f}, "
                     f"{sum(r['chars'] for r in ok) / max(len(ok), 1):,.0f} chars per page")
    return lines


def rater_report(pages: list[dict], repeat: int) -> list[str]:
    rated = [p for p in pages if p.get('rating')]
    lines = ['', '## Rater', '',
             f'{len(rated)} pages, each rated {repeat} times per form from one Markdown extract; only the posting '
             'text differs between the forms.', '']

    def scores(p: dict, form: str) -> list:
        return [r['rating'] for r in p['rating']['ratings'][form]]

    for form in FORMS:
        flips = sum(len(set(scores(p, form))) > 1 for p in rated)
        notified = sum(sum(s >= RATING_NOTIFICATION_THRESHOLD for s in scores(p, form) if isinstance(s, int)) for p in rated) / repeat
        mean = sum(sum(scores(p, form)) for p in rated) / max(len(rated) * repeat, 1)
        lines.append(f'- {form}: mean {mean:.2f}; {flips} pages whose {repeat} ratings disagree; '
                     f'{notified:.1f} pages at or above {RATING_NOTIFICATION_THRESHOLD} per run')
    across = [sum(scores(p, 'text')[i] != scores(p, 'markdown')[i] for p in rated) for i in range(repeat)]
    lines.append(f'- text vs Markdown, same repeat: {sum(across) / repeat:.1f} pages differ per repeat')
    consistent = [p for p in rated if len(set(scores(p, 'text'))) == 1 and len(set(scores(p, 'markdown'))) == 1
                  and scores(p, 'text')[0] != scores(p, 'markdown')[0]]
    lines += ['', f'### Pages where each form agrees with itself and not with the other ({len(consistent)})', '']
    for p in consistent:
        lines.append(f"- job {p['job_id']} {p['company']}: text {scores(p, 'text')[0]} → Markdown "
                     f"{scores(p, 'markdown')[0]} ({p['path']})")
        for form in FORMS:
            reasoning = p['rating']['ratings'][form][0]['reasoning']
            lines.append(f'  - {form}: {reasoning[:REASONING_SHOWN_MAX_CHARS]}'
                         + (f' […cut, {len(reasoning)} chars]' if len(reasoning) > REASONING_SHOWN_MAX_CHARS else ''))
    cost = sum(p['rating']['cost'] for p in rated)
    by_model = collections.Counter(r['rated_by'] for p in rated for form in FORMS for r in p['rating']['ratings'][form])
    lines += ['', f'Rating cost ${cost:.4f}. Rated by: {dict(by_model)}.']
    return lines


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--date', action='append', help='capture day under run_dir/raw_postings; repeatable (default: all)')
    parser.add_argument('--corpus', action='store_true',
                        help='replay the pinned copies in run_dir/replay_corpus (scripts/pin_replay_corpus.py), not raw_postings')
    parser.add_argument('--limit', type=int, help='pages to replay (default: all)')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT, help=f'runs per input (default {DEFAULT_REPEAT})')
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT_PAGES,
                        help=f'pages replayed at once (default {DEFAULT_CONCURRENT_PAGES})')
    parser.add_argument('--noise-only', action='store_true',
                        help='only the Markdown extractor, repeated: its run-to-run variation, no text runs, no ratings')
    parser.add_argument('--extract-temperature', type=float,
                        help='override EXTRACT_TEMPERATURE for this replay (default: the configured one)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
    if args.repeat < 2:
        parser.error('--repeat must be at least 2: noise is measured between repeats')

    root = REPLAY_CORPUS_DIR if args.corpus else RAW_POSTINGS_DIR
    days = [root / d for d in args.date] if args.date else sorted(p for p in root.iterdir() if p.is_dir())
    paths = [p for day in days for p in sorted(day.glob('linkedin-*.txt'))
             if not p.name.endswith('.rater_posting.txt')][:args.limit]
    if not paths:
        print(f'No captures under {", ".join(map(str, days))}')
        return 1
    if args.extract_temperature is not None:
        extract_openrouter.EXTRACT_TEMPERATURE = args.extract_temperature
    forms = ('markdown',) if args.noise_only else FORMS
    evaluator_prompt = '' if args.noise_only else build_evaluator_prompt(await build_reference_summary(new_stage_stats()))
    gate = asyncio.Semaphore(args.concurrent)
    started = datetime.now()
    pages = [p for p in await asyncio.gather(*(replay_page(p, evaluator_prompt, args.repeat, gate, forms, not args.noise_only)
                                               for p in paths)) if p]
    errors = [p for p in pages if 'error' in p]
    lines = [f'# Plain text vs Markdown: extractor and rater ({started:%Y-%m-%d %H:%M})', '',
             f"{len(pages)} pages from {', '.join(d.name for d in days)}; {len(errors)} failed; "
             f'{(datetime.now() - started).total_seconds() / 60:.0f} min; extractor temperature '
             f'{extract_openrouter.EXTRACT_TEMPERATURE}.', '']
    lines += extractor_report(pages, args.repeat, forms) + ([] if args.noise_only else rater_report(pages, args.repeat))
    lines += [f"- error: {p['path']}: {p['error']}" for p in errors]
    out = RUN_DIR / 'replays' / f'markdown-ab-{started:%Y%m%d-%H%M%S}'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(pages, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('\n'.join(lines))
    print(f'Report written to {out}.md (raw results {out}.json)')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
