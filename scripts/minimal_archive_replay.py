'''Build the minimal archive (minimal_archive.py) from full job-page archives already on disk.

read_job_page writes `<job>.min.html` + `<job>.min.pdf` itself since 2026-10-29. This replays
the same steps over saved full archives: to check a change to the minimising on real pages, or to
backfill pages captured before then. The report checks every line the extractor read is still on
the minimal page.

Usage:
    uv run python scripts/minimal_archive_replay.py              # the 20 newest, into run_dir/replays/
    uv run python scripts/minimal_archive_replay.py -n 5
    uv run python scripts/minimal_archive_replay.py --in-place   # backfill beside each archive

Reads files on disk only: no browser session, no LinkedIn, no LLM call (a page section with no
cached verdict is kept, never sent to the classifier). The report goes to
run_dir/replays/minimal-archive-{timestamp}/report.md.
'''

import argparse
import html
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Comment

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agentic_job_search import linkedin_page, minimal_archive, tools_generic

logger = logging.getLogger('minimal_archive_replay')

DEFAULT_PAGE_COUNT = 20
RAW_POSTINGS_DIR = tools_generic.RUN_DIR / 'raw_postings'
REPLAYS_DIR = tools_generic.RUN_DIR / 'replays'


def newest_archives(count: int) -> list[Path]:
    full = (p for p in RAW_POSTINGS_DIR.glob('*/*.html') if not p.name.endswith(minimal_archive.MINIMAL_HTML_SUFFIX))
    return sorted(full, key=lambda p: p.stat().st_mtime)[-count:]


def archive_meta(soup: BeautifulSoup, path: Path) -> dict:
    '''The YAML header save_raw_html writes as the file's first comment.'''
    comment = soup.find(string=lambda s: isinstance(s, Comment))
    if comment is None:
        raise RuntimeError(f'{path}: no metadata comment at the top of the archive')
    try:
        meta = yaml.safe_load(str(comment))
    except yaml.YAMLError as ex:
        raise RuntimeError(f'{path}: metadata comment is not YAML: {type(ex).__name__}: {ex}') from ex
    if not isinstance(meta, dict):
        raise RuntimeError(f'{path}: metadata comment holds {type(meta).__name__}, not a mapping: {str(comment)!r}')
    return meta


def remove_cached_sections(soup: BeautifulSoup) -> list[dict]:
    '''Unknown blocks whose cached verdict is remove; unseen ones are kept, never sent to a model.'''
    main = soup.find('main')
    decisions = linkedin_page.load_section_decisions() or {}
    removed = []
    for block in linkedin_page.unknown_blocks(main) if main is not None else []:
        sig = linkedin_page.block_signature(block)
        text = linkedin_page.norm(block.get_text(' '))
        if (decisions.get(sig) or {}).get('decision') == 'remove' and not linkedin_page.mentions_pay(text):
            removed.append({'what': f'section {sig!r}', 'chars': len(text)})
            block.decompose()
    return removed


def text_lines(text: str) -> list[str]:
    return [line for line in (linkedin_page.norm(line) for line in text.splitlines()) if line]


def missing_lines(txt_path: Path, minimal_text: str) -> list[str]:
    '''Lines of the extractor's text (front matter skipped) the minimal page no longer contains.'''
    if not txt_path.exists():
        return []
    raw = txt_path.read_text(encoding='utf-8')
    body = raw.split('\n---\n', 1)[1] if raw.startswith('---\n') else raw
    joined = ' '.join(text_lines(minimal_text))
    return [line for line in text_lines(body) if line not in joined]


def process(path: Path, out_dir: Path, shell: Path) -> dict:
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'lxml')
    meta = archive_meta(soup, path)
    removed = linkedin_page.remove_known_clutter(soup) + linkedin_page.remove_expand_button(soup)
    removed += remove_cached_sections(soup)
    main = soup.find('main') or soup.body
    if main is None:
        raise RuntimeError(f'{path}: the archive has neither <main> nor <body>')
    removed += minimal_archive.minimise(main)
    if out_dir != path.parent:
        for img in main.find_all('img', src=True):
            if not str(img['src']).startswith(('http:', 'https:', 'data:')):
                img['src'] = os.path.relpath((path.parent / img['src']).resolve(), out_dir)
    html_path, pdf_path = minimal_archive.minimal_paths(out_dir / path.name)
    html_path.write_text(minimal_archive.page_html(main, meta, removed), encoding='utf-8')
    minimal_archive.render_pdf(shell, html_path, pdf_path)
    minimal_text = main.get_text('\n')
    return {
        'name': path.stem, 'orig_bytes': path.stat().st_size, 'html_bytes': html_path.stat().st_size,
        'pdf_bytes': pdf_path.stat().st_size, 'text_chars': len(' '.join(text_lines(minimal_text))),
        'removed': removed, 'missing': missing_lines(path.with_suffix('.txt'), minimal_text),
    }


def write_report(out_dir: Path, rows: list[dict], failures: list[str], stylesheet_bytes: int) -> Path:
    lines = ['# Minimal archive replay', '',
             f'Daily stylesheet the full archive needs: {stylesheet_bytes:,} bytes (the minimal page needs none).', '',
             '| page | original | min.html | pdf | text chars | removed blocks | extractor lines missing |',
             '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        missing = '<br>'.join(html.escape(m) for m in r['missing']) or '—'
        lines.append(f"| {r['name']} | {r['orig_bytes']:,} | {r['html_bytes']:,} | {r['pdf_bytes']:,} | "
                     f"{r['text_chars']:,} | {len(r['removed'])} | {missing} |")
    if rows:
        total_orig = sum(r['orig_bytes'] for r in rows)
        total_html = sum(r['html_bytes'] for r in rows)
        total_pdf = sum(r['pdf_bytes'] for r in rows)
        lines += ['', f'Totals: original {total_orig:,} → min.html {total_html:,} '
                      f'({total_orig / total_html:.0f}× smaller); PDFs {total_pdf:,}.']
    if failures:
        lines += ['', '## Failures', ''] + [f'- {f}' for f in failures]
    path = out_dir / 'report.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', type=int, default=DEFAULT_PAGE_COUNT, help='how many of the newest archives')
    parser.add_argument('--in-place', action='store_true',
                        help='backfill: write each .min.html/.min.pdf beside its archive, as read_job_page now does')
    args = parser.parse_args()
    report_dir = REPLAYS_DIR / f'minimal-archive-{datetime.now():%Y%m%d-%H%M%S}'
    report_dir.mkdir(parents=True)
    shell = minimal_archive.headless_shell()
    if shell is None:
        raise SystemExit(f'no {minimal_archive.PLAYWRIGHT_HEADLESS_SHELL_GLOB} found; '
                         f'run `npx playwright install chromium-headless-shell`')
    logger.info(f'Writing to {"each archive folder" if args.in_place else report_dir}; PDFs printed by {shell}')
    rows, failures = [], []
    for path in newest_archives(args.n):
        try:
            row = process(path, path.parent if args.in_place else report_dir, shell)
        except Exception as ex:
            logger.error(f'{path}: {type(ex).__name__}: {ex}')
            failures.append(f'{path}: {type(ex).__name__}: {ex}')
            continue
        rows.append(row)
        logger.info(f"{row['name']}: {row['orig_bytes']:,} → {row['html_bytes']:,} bytes html, "
                    f"{row['pdf_bytes']:,} pdf; {len(row['missing'])} extractor line(s) missing")
    stylesheet_bytes = max((sum(p.stat().st_size for p in d.glob('*.css'))
                            for d in RAW_POSTINGS_DIR.glob(f'*/{linkedin_page.STYLESHEET_DIR_NAME}')), default=0)
    logger.info(f'Report: {write_report(report_dir, rows, failures, stylesheet_bytes)}')


if __name__ == '__main__':
    main()
