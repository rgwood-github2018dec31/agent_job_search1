'''Replay retained job pages through the extractor and Verify Source Reference, live.

Why this is a tracked script
----------------------------
Verify Source Reference rejects any extract field whose citation code cannot find in the page.
Too strict a matcher, or a prompt that gets sloppy quotes, turns that into silent false
rejections. A field rejected as "absent" can let a job through a gate it should have hit
(a rejected `language_requirement`), so the rate of rejections per field is the number to
watch when the extractor model, its prompt, or the matcher changes. This measures it on real pages.

It also reports each derive_* regex fallback that would change if it read the whole page instead
of the posting's own text (it reads the posting's text; see agent.regex_source_text).

Usage:
    uv run python scripts/replay_evidence.py                       # newest capture day, 20 pages
    uv run python scripts/replay_evidence.py --date 2026-10-16 --limit 40
    uv run python scripts/replay_evidence.py --input both          # plain text vs Markdown, same pages

--input markdown|both rebuild the page from the saved full .html with the pipeline's own clutter
removal (cached section verdicts only) and give the extractor Markdown
(minimal_archive.to_markdown), verified against that Markdown; `both` also runs plain text built
the same way and reports, per page, where the two extracts differ.

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
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agentic_job_search import evidence, linkedin_page, minimal_archive
from agentic_job_search.agent import (
    EXTRACT_CITED_FIELDS,
    PAGE_EXTRACTOR_INSTRUCTIONS,
    attach_page,
    derive_agency_posting,
    derive_education_requirement,
    derive_residency_scope,
    derive_workplace_type,
    new_stage_stats,
    verify_extract,
)
from agentic_job_search.extract_openrouter import extract_job_page_openrouter
from agentic_job_search.linkedin_page import HeaderCard, JobPage, header_card

logger = logging.getLogger('replay_evidence')

RUN_DIR = Path(__file__).parent.parent / 'run_dir'
DEFAULT_REPLAY_LIMIT = 20
FRONT_MATTER_DELIMITER = '---\n'
INPUT_TEXT, INPUT_MARKDOWN, INPUT_BOTH = 'text', 'markdown', 'both'
VALUE_SHOWN_MAX_CHARS = 80
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


def inputs_from_html(path: Path) -> tuple[str, str]:
    '''(plain text, Markdown) of the capture's saved full .html, de-cluttered as read_job_page does.'''
    html = path.with_suffix('.html')
    if not html.exists():
        raise FileNotFoundError(f'{html} is missing; --input markdown/both needs the full page archive')
    soup = BeautifulSoup(html.read_text(encoding='utf-8'), 'lxml')
    linkedin_page.remove_known_clutter(soup)
    linkedin_page.remove_expand_button(soup)
    linkedin_page.apply_cached_section_decisions(soup)
    main = soup.find('main') or soup.body or soup
    return linkedin_page.page_text(soup), minimal_archive.to_markdown(main)


async def replay_one(path: Path, page_text: str | None = None, markdown: str | None = None) -> dict:
    '''Extract and verify one capture. With markdown, the extractor reads it and is verified against it;
    the plain text still gives the description the derive_* regexes read, as it would in a run.'''
    header, saved_text = load_capture(path)
    page_text = saved_text if page_text is None else page_text
    model_input = markdown if markdown is not None else page_text
    candidate = candidate_for(header, page_text)
    stats = new_stage_stats()
    extract = await extract_job_page_openrouter(candidate, model_input, stats, PAGE_EXTRACTOR_INSTRUCTIONS)
    if extract is None:
        return {'path': path, 'candidate': candidate, 'failed': True, 'chars': len(model_input)}
    before = dict(extract)
    # As in a run: title, company, date and closed come off the page's header card, kept in the .html.
    html = path.with_suffix('.html')
    page_header = header_card(BeautifulSoup(html.read_text(encoding='utf-8'), 'lxml')) if html.exists() else HeaderCard()
    attach_page(candidate, extract, JobPage(text=page_text, source='dom', header=page_header))
    if markdown is not None:
        extract['page_text'] = markdown
    derived_from_description = {name: fn(dict(extract)) for name, fn in DERIVERS.items()}
    statuses = await verify_extract(candidate, extract, stats)
    # What the regex fallbacks would say if they read the whole page rather than the posting's own
    # text (they do not: see agent.regex_source_text), kept as the measurement behind that choice.
    derived_from_page = {name: fn(dict(extract, description=page_text)) for name, fn in DERIVERS.items()}
    return {'path': path, 'candidate': candidate, 'failed': False, 'before': before, 'after': extract,
            'statuses': statuses, 'derived_from_description': derived_from_description,
            'derived_from_page': derived_from_page, 'cost': stats['cost'], 'chars': len(model_input)}


def _shown(value: object) -> str:
    return repr(str(value if value is not None else '')[:VALUE_SHOWN_MAX_CHARS])


def comparison_report(pairs: list[tuple[dict, dict]]) -> str:
    '''Plain text vs Markdown on the same pages: size, cost, rejections, and every field that differs.'''
    def rejected(r: dict) -> list[str]:
        return sorted(f for f, st in (r.get('statuses') or {}).items() if st == evidence.STATUS_REJECTED)

    lines = ['# Plain text vs Markdown extractor input', '',
             '| page | chars text → md | cost text / md | rejected (text) | rejected (md) | fields differing |',
             '|---|---|---|---|---|---|']
    details = []
    totals = collections.Counter()
    for text_run, md_run in pairs:
        name = text_run['path'].name
        if text_run['failed'] or md_run['failed']:
            lines.append(f"| {name} | {text_run['chars']} → {md_run['chars']} | | text failed={text_run['failed']} "
                         f"| md failed={md_run['failed']} | |")
            continue
        differing = [f for f in EXTRACT_CITED_FIELDS if text_run['after'].get(f) != md_run['after'].get(f)]
        totals.update(cost_text=text_run['cost'], cost_md=md_run['cost'], rejected_text=len(rejected(text_run)),
                      rejected_md=len(rejected(md_run)), differing=len(differing))
        lines.append(f"| {name} | {text_run['chars']:,} → {md_run['chars']:,} | ${text_run['cost']:.4f} / "
                     f"${md_run['cost']:.4f} | {', '.join(rejected(text_run)) or '—'} | "
                     f"{', '.join(rejected(md_run)) or '—'} | {', '.join(differing) or '—'} |")
        for field in differing:
            details.append(f"- `{name}` **{field}**: text {_shown(text_run['after'].get(field))} "
                           f"({text_run['statuses'].get(field, '')}) vs md {_shown(md_run['after'].get(field))} "
                           f"({md_run['statuses'].get(field, '')})")
    lines += ['', f"Totals: cost ${totals['cost_text']:.4f} text / ${totals['cost_md']:.4f} md; "
                  f"rejections {totals['rejected_text']} text / {totals['rejected_md']} md; "
                  f"{totals['differing']} field value(s) differ.", '', '## Differing values', ''] + (details or ['None.'])
    return '\n'.join(lines) + '\n'


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
    lines += ['', '## derive_* on the whole page vs on the posting text', '']
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
    parser.add_argument('--input', choices=(INPUT_TEXT, INPUT_MARKDOWN, INPUT_BOTH), default=INPUT_TEXT,
                        help='what the extractor reads: the saved text (default), Markdown, or both to compare')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')

    days = sorted(p for p in (RUN_DIR / 'raw_postings').iterdir() if p.is_dir())
    day = RUN_DIR / 'raw_postings' / args.date if args.date else days[-1]
    paths = sorted(day.glob('*.txt'))[:args.limit]
    if not paths:
        print(f'No captures under {day}')
        return 1
    results, pairs = [], []
    for number, path in enumerate(paths, start=1):
        print(f'[{number}/{len(paths)}] {path.name}', flush=True)
        if args.input == INPUT_TEXT:
            results.append(await replay_one(path))
            continue
        try:
            plain, markdown = inputs_from_html(path)
        except FileNotFoundError as ex:
            logger.warning(f'Skipping {path.name}: {ex}')
            continue
        md_run = await replay_one(path, plain, markdown)
        results.append(md_run)
        if args.input == INPUT_BOTH:
            pairs.append((await replay_one(path, plain), md_run))
    text = (comparison_report(pairs) + '\n' if pairs else '') + report(results)
    out = RUN_DIR / 'replays' / f'evidence-{datetime.now():%Y%m%d-%H%M%S}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(text)
    print(f'Report written to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
