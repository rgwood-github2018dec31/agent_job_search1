"""The minimal archive of a job page: the de-cluttered page as small self-contained HTML, and a PDF.

The full archive (`save_raw_html`) is the page before clutter removal: ~256K chars of which ~14K
is text, readable only with LinkedIn's 1.3 MB stylesheet (20 pages, 2026-10-29). This writes,
beside it, what the extractor's text came from: the page after clutter removal, with icons,
buttons and styling attributes dropped and the div nesting flattened, plus a footer naming what
was removed. The PDF is printed from it by the Chromium headless shell Playwright installs.

Audit only, like the full archive: nothing in the pipeline reads it, and every failure is logged
and the run continues.
"""
import asyncio
import html
import logging
import os
import re
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from agentic_job_search.config import (
    MINIMAL_ARCHIVE_KEEP_ATTRS,
    MINIMAL_ARCHIVE_PDF_TIMEOUT_SECONDS,
    PLAYWRIGHT_HEADLESS_SHELL_GLOB,
)

logger = logging.getLogger(__name__)

MINIMAL_HTML_SUFFIX = '.min.html'
MINIMAL_PDF_SUFFIX = '.min.pdf'
LINKEDIN_ORIGIN = 'https://www.linkedin.com'
BADGE_SEPARATOR = ' · '
STDERR_TAIL_CHARS = 2000
_PLAYWRIGHT_DEFAULT_CACHE = Path.home() / 'Library' / 'Caches' / 'ms-playwright'
_BUILD_NUMBER_RE = re.compile(r'\d+')
_KEEP_EMPTY_TAGS = ('img', 'br', 'hr')
_FLATTEN_TAGS = ('div', 'section')
# Logged once per run, not once per page.
_missing_shell_reported = False
STYLE = '''
:root { --text-base: 1rem; --text-small: 0.85rem; --text-h1: 1.5rem; --text-h2: 1.2rem;
        --sp-1: 0.25rem; --sp-2: 0.5rem; --sp-4: 1rem; --line: 70ch; --hairline: 1px; }
body { font: var(--text-base)/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif;
       max-width: var(--line); margin: var(--sp-4) auto; padding: 0 min(var(--sp-4), 3vw);
       color: #1a1a1a; background: #fff; }
h1 { font-size: var(--text-h1); margin: var(--sp-2) 0; }
h2 { font-size: var(--text-h2); margin: var(--sp-4) 0 var(--sp-2); border-bottom: var(--hairline) solid #ccc; }
h3, h4 { font-size: var(--text-base); margin: var(--sp-2) 0 var(--sp-1); }
p { margin: var(--sp-1) 0 var(--sp-2); }
img { max-height: 4rem; max-width: 12rem; }
a { color: inherit; }
.archive-meta, .archive-removed { font-size: var(--text-small); color: #555; }
.archive-meta { border-bottom: var(--hairline) solid #ccc; padding-bottom: var(--sp-2); }
.archive-removed { border-top: var(--hairline) solid #ccc; margin-top: var(--sp-4); }
'''


def _norm(text: str | None) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def _is_blank(el: Tag) -> bool:
    return not el.get_text(strip=True) and el.find('img') is None


def minimise(main: Tag) -> list[dict]:
    """Drop what only LinkedIn's CSS and scripts used, in place; flatten the div nesting.

    Kept: text, headings, lists, links, and the images localize_assets kept (the company logo).
    Returns the buttons dropped, for the removed-blocks footer.
    """
    removed = []
    for el in main.find_all('svg'):
        el.decompose()
    for img in main.find_all('img'):
        if not img.get('src') or img.get('data-removed'):
            img.decompose()
    for button in main.find_all('button'):
        text = _norm(button.get_text(' '))
        removed.append({'what': f"button {text or _norm(button.get('aria-label', ''))!r}", 'chars': len(text)})
        button.decompose()
    for el in [main, *main.find_all(True)]:
        el.attrs = {k: v for k, v in el.attrs.items() if k in MINIMAL_ARCHIVE_KEEP_ATTRS}
        if str(el.get('href', '')).startswith('/'):
            el['href'] = LINKEDIN_ORIGIN + el['href']
        if el.get('href') == '#':
            del el['href']
    for el in main.find_all('span'):
        el.unwrap()
    changed = True
    while changed:
        changed = False
        for el in main.find_all(True):
            if el.decomposed:
                continue
            if el.name not in _KEEP_EMPTY_TAGS and _is_blank(el):
                el.decompose()
                changed = True
            elif (el.name in _FLATTEN_TAGS and len(el.find_all(True, recursive=False)) == 1
                  and not ''.join(el.find_all(string=True, recursive=False)).strip()):
                el.unwrap()
                changed = True
    # Header badges end up as adjacent links ('90K EUR/yr', 'Remote', 'Full-time'); CSS spaced them.
    for link in main.find_all('a'):
        if isinstance(link.next_sibling, Tag) and link.next_sibling.name == 'a':
            link.insert_after(BADGE_SEPARATOR)
    return removed


def page_html(main: Tag, meta: dict, removed: list[dict]) -> str:
    '''The self-contained page: a provenance line, the minimised content, what was removed.'''
    url = html.escape(str(meta.get('url', '')))
    meta_line = (f'<div class="archive-meta">Archived {html.escape(str(meta.get("captured", "")))} · '
                 f'job {html.escape(str(meta.get("job_id", "")))} · <a href="{url}">{url}</a></div>')
    items = ''.join(f'<li>{html.escape(r["what"])} ({r["chars"]} chars)</li>' for r in removed)
    removed_block = f'<div class="archive-removed">Removed from the archived page:<ul>{items}</ul></div>'
    title = html.escape(f'{meta.get("company", "")} — job {meta.get("job_id", "")}')
    main.name = 'main'
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>{STYLE}</style></head>\n'
            f'<body>{meta_line}\n{main}\n{removed_block}</body></html>\n')


def headless_shell() -> Path | None:
    '''The newest Chromium headless shell Playwright installed, or None.'''
    cache = Path(os.environ.get('PLAYWRIGHT_BROWSERS_PATH') or _PLAYWRIGHT_DEFAULT_CACHE)

    def build_number(path: Path) -> int:
        found = _BUILD_NUMBER_RE.search(path.relative_to(cache).parts[0])
        return int(found.group(0)) if found else 0

    shells = sorted(cache.glob(PLAYWRIGHT_HEADLESS_SHELL_GLOB), key=build_number)
    return shells[-1] if shells else None


def render_pdf(shell: Path, html_path: Path, pdf_path: Path) -> None:
    '''Print html_path to pdf_path. Raises RuntimeError with the command, exit code and stderr.'''
    command = [str(shell), '--headless', '--disable-gpu', '--no-pdf-header-footer',
               f'--print-to-pdf={pdf_path}', html_path.resolve().as_uri()]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=MINIMAL_ARCHIVE_PDF_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as ex:
        raise RuntimeError(f'printing {html_path} to PDF failed: {type(ex).__name__}: {ex}; command {command}') from ex
    if done.returncode != 0 or not pdf_path.exists():
        stderr = done.stderr[-STDERR_TAIL_CHARS:]
        raise RuntimeError(f'printing {html_path} to PDF exited {done.returncode}, pdf written={pdf_path.exists()}; '
                           f'command {command}; stderr (last {len(stderr)} of {len(done.stderr)} chars): {stderr}')


def minimal_paths(full_html_path: Path) -> tuple[Path, Path]:
    stem = full_html_path.with_suffix('')
    return stem.with_name(stem.name + MINIMAL_HTML_SUFFIX), stem.with_name(stem.name + MINIMAL_PDF_SUFFIX)


async def save_minimal_archive(soup: BeautifulSoup, full_html_path: Path, meta: dict,
                               removed: list[dict]) -> tuple[Path | None, Path | None]:
    """Write the minimal HTML and PDF beside the full archive, from the de-cluttered soup (consumed).

    Image paths stay valid because the files sit in the full archive's folder. Never raises: a
    failure is logged and the run continues, the full archive and the extractor's text stand.
    """
    html_path, pdf_path = minimal_paths(full_html_path)
    job = meta.get('job_id', '')
    main = soup.find('main') or soup.body
    if main is None:
        logger.warning(f'Job page {job}: no <main> or <body> to archive minimally — skipped')
        return None, None
    try:
        dropped = minimise(main)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page_html(main, meta, removed + dropped), encoding='utf-8')
    except (OSError, ValueError) as ex:
        logger.warning(f'Job page {job}: minimal archive {html_path} not written: {type(ex).__name__}: {ex}')
        return None, None
    global _missing_shell_reported
    shell = headless_shell()
    if shell is None:
        if _missing_shell_reported:
            return html_path, None
        _missing_shell_reported = True
        logger.warning(f'Job page {job}: no {PLAYWRIGHT_HEADLESS_SHELL_GLOB} under PLAYWRIGHT_BROWSERS_PATH or '
                       f'{_PLAYWRIGHT_DEFAULT_CACHE} — no PDFs this run (`npx playwright install chromium-headless-shell`)')
        return html_path, None
    try:
        await asyncio.to_thread(render_pdf, shell, html_path, pdf_path)
    except RuntimeError as ex:
        logger.warning(f'Job page {job}: {ex} — the minimal HTML is kept without a PDF')
        return html_path, None
    return html_path, pdf_path
