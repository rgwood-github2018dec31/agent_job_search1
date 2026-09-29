"""Read a LinkedIn job page in code: capture it, archive it, and strip the clutter.

The extractor model used to drive the browser itself and read a full accessibility snapshot, of
which the description was ~15%; the rest was navigation, upsells, the company's social posts and
"More jobs" — other companies' listings with their salaries (measured 2026-09-24). Now code does
the reading:

1. One read-only `browser_evaluate` (`LINKEDIN_PAGE_CAPTURE_JS`) serializes the DOM after
   LinkedIn's own scripts have run.
2. The page is archived first, JS-free, as `<site>-<job_id>.html` beside the raw posting. Only
   the posting company's logo is kept (stored once per company under `company_assets/`); every
   other image — people's profile photos, other companies' logos, post images — is blanked, never
   downloaded. The stylesheet is stored once per day beside the captures.
3. Known clutter is removed: `LINKEDIN_REMOVE_TAGS`, `LINKEDIN_REMOVE_SECTION_PHRASES`,
   `LINKEDIN_REMOVE_ARIA_LABELS` — all matched on text or aria-label, never on class names.
4. Anything left outside the keep anchors (the top card, the description, the company card) is an
   UNKNOWN block: LinkedIn adds sections from time to time. Its signature is looked up in
   `linkedin_page_sections.yaml`; unseen signatures go to one `MODEL_NAME_PAGE_SECTIONS` call per
   page, and the verdicts are cached there, where the user can read and change them.

Guards decided in code, not by the model: the keep anchors are never shown to it, a block holding
a pay figure is never removed on its say-so, and any failure keeps the block. A kept block costs a
few hundred chars of noise; a wrongly removed one could cost the job's facts.
"""
import hashlib
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from http import HTTPStatus
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup, Tag

from agentic_job_search import tools_generic
from agentic_job_search.config import (
    EXTRACT_PAGE_RENDER_WAIT_SECONDS,
    EXTRACTOR_SNAPSHOT_MAX_CHARS,
    JOB_PAGE_ASSET_FETCH_TIMEOUT_SECONDS,
    LINKEDIN_PAGE_CAPTURE_JS,
    LINKEDIN_PAGE_CAPTURE_RETRIES,
    LINKEDIN_REMOVE_ARIA_LABELS,
    LINKEDIN_REMOVE_SECTION_PHRASES,
    LINKEDIN_REMOVE_TAGS,
    MODEL_NAME_PAGE_SECTIONS,
    PAGE_SECTION_SAMPLE_MAX_CHARS,
    PAGE_SECTION_SIGNATURE_MAX_WORDS,
)
from agentic_job_search.salary import compensation_context
from agentic_job_search.scrape_openrouter import parse_evaluate_result
from agentic_job_search.text_budget import normalize_whitespace, snippet, truncate_reported, truncate_reported_middle
from agentic_job_search.triage import chat_openrouter, extract_json_object

logger = logging.getLogger(__name__)

COMPANY_ASSETS_DIR = tools_generic.RUN_DIR / 'company_assets'
PAGE_SECTIONS_PATH = tools_generic.RUN_DIR / 'linkedin_page_sections.yaml'
ASSET_INDEX_NAME = 'index.yaml'
STYLESHEET_DIR_NAME = 'assets'
CONTENT_HASH_CHARS = 16

_ABOUT_JOB_ID_RE = re.compile(r'^JobDetails_AboutTheJob_')
_ABOUT_COMPANY_ID_RE = re.compile(r'^JobDetails_AboutTheCompany_')
_COMPANY_LOGO_ALT_RE = re.compile(r'^Company logo for,\s*(?P<company>.+?)\.?$')
_FOLLOWERS_RE = re.compile(r'^[\d,.]+[KkMm]?\s+followers$')
# The company card's industry / size line: 'IT Services and IT Consulting • 10001+ employees • ...'.
_EMPLOYEES_RE = re.compile(r'^[\d,.+\-–]+[KkMm]?\+?\s+employees$')
_HEADING_TAGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')
_SECTION_HEADING_TAGS = ('h1', 'h2')
_NON_CONTENT_TAGS = ('script', 'noscript', 'style', 'template')
_UNSAFE_PATH_CHARS_RE = re.compile(r'[/\\:\x00]')
_DIGITS_RE = re.compile(r'\d+')
_EXTENSION_BY_MIME = {
    'image/png': 'png', 'image/jpeg': 'jpg', 'image/gif': 'gif', 'image/webp': 'webp',
    'image/svg+xml': 'svg', 'text/css': 'css',
}
_PAGE_SECTIONS_HEADER = (
    '# LinkedIn job-page sections code could not place, and what was decided about each.\n'
    '# Keyed by signature (heading or first line, digits as N). `decision: remove` strips the\n'
    '# section before the extractor reads the page; `keep` leaves it. Written by the agent when a\n'
    '# new section appears (decided_by names the model); edit `decision` by hand to overrule it.\n'
)


@dataclass
class JobPage:
    """What the extractor reads, and how it came to be."""
    text: str
    source: str  # 'dom' (de-cluttered page) or 'snapshot' (fallback: budgeted a11y snapshot)
    removed: list[dict] = field(default_factory=list)


def norm(text: str | None) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def _own_text(el: Tag) -> str:
    return norm(''.join(el.find_all(string=True, recursive=False)))


def _contains(ancestor: Tag, el: Tag) -> bool:
    return ancestor is el or any(parent is ancestor for parent in el.parents)


# --- JS-free archive -----------------------------------------------------------------------------

def strip_js(soup: BeautifulSoup) -> dict:
    """Remove every script, script preload, inline handler and javascript: URL. Returns counts."""
    counts = {'scripts': 0, 'script_links': 0, 'handler_attrs': 0, 'js_urls': 0}
    for el in soup.find_all(['script', 'noscript']):
        el.decompose()
        counts['scripts'] += 1
    for el in soup.find_all('link'):
        rel = ' '.join(el.get('rel') or [])
        if 'modulepreload' in rel or el.get('as') == 'script':
            el.decompose()
            counts['script_links'] += 1
    for el in soup.find_all(True):
        for attr in [a for a in el.attrs if a.lower().startswith('on')]:
            del el[attr]
            counts['handler_attrs'] += 1
        for attr in ('href', 'src', 'action'):
            if str(el.get(attr, '')).strip().lower().startswith('javascript:'):
                el[attr] = '#'
                counts['js_urls'] += 1
    return counts


def company_dir_name(company: str) -> str:
    """The company's name as written, made safe as one path component."""
    return _UNSAFE_PATH_CHARS_RE.sub('_', company).strip().strip('.') or 'unknown'


def _load_asset_index(directory: Path) -> dict:
    path = directory / ASSET_INDEX_NAME
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError) as ex:
        raise RuntimeError(f'asset index {path} is unreadable: {type(ex).__name__}: {ex}') from ex


def _http_get(url: str) -> requests.Response:
    """A plain GET with no cookies or session: page assets never go through the logged-in browser."""
    return requests.get(url, timeout=JOB_PAGE_ASSET_FETCH_TIMEOUT_SECONDS)


def store_asset_once(url: str, directory: Path) -> tuple[Path, bool]:
    """The local copy of url in directory, fetched only when not already there. (path, downloaded).

    Keyed by URL in the directory's index and named by content hash, so the same bytes behind two
    URLs are one file. Fetched with plain requests — no cookies, never the logged-in browser.
    Raises RuntimeError with the URL on any failure.
    """
    index = _load_asset_index(directory)
    if url in index and (directory / index[url]).exists():
        return directory / index[url], False
    try:
        resp = _http_get(url)
    except requests.RequestException as ex:
        raise RuntimeError(f'fetching {url} failed: {type(ex).__name__}: {ex}') from ex
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f'fetching {url} returned HTTP {resp.status_code}')
    mime = resp.headers.get('content-type', '').split(';')[0].strip()
    name = f'{hashlib.sha256(resp.content).hexdigest()[:CONTENT_HASH_CHARS]}.{_EXTENSION_BY_MIME.get(mime, "bin")}'
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / name).exists():
            (directory / name).write_bytes(resp.content)
        index[url] = name
        (directory / ASSET_INDEX_NAME).write_text(yaml.safe_dump(index, allow_unicode=True), encoding='utf-8')
    except OSError as ex:
        raise RuntimeError(f'storing {url} in {directory} failed: {type(ex).__name__}: {ex}') from ex
    return directory / name, True


def posting_company(soup: BeautifulSoup) -> str | None:
    """The posting company as the page's own logo alt text names it ('Company logo for, X.')."""
    for img in soup.find_all('img'):
        if m := _COMPANY_LOGO_ALT_RE.match(norm(img.get('alt', ''))):
            return m.group('company')
    return None


def localize_assets(soup: BeautifulSoup, company: str | None, html_path: Path) -> dict:
    """Keep the posting company's logo and the stylesheet as local files; blank every other image.

    The logo is the top card's image (alt 'Company logo for, X.') and the company card's first
    image; nothing else on the page is the company's. People's photos and other companies' logos
    are never downloaded. A failed fetch blanks that image and is logged once, here.
    """
    counts = {'logos_kept': 0, 'downloaded': 0, 'images_blanked': 0, 'fetch_failures': 0, 'stylesheets': 0}
    about_company = soup.find(id=_ABOUT_COMPANY_ID_RE)
    company_card_logo = about_company.find('img') if about_company is not None else None
    logo_dir = COMPANY_ASSETS_DIR / company_dir_name(company) if company else None
    for img in soup.find_all('img'):
        src = str(img.get('src', ''))
        img.attrs.pop('srcset', None)
        is_logo = _COMPANY_LOGO_ALT_RE.match(norm(img.get('alt', ''))) or img is company_card_logo
        if logo_dir is not None and is_logo and src.startswith('http'):
            try:
                local, downloaded = store_asset_once(src, logo_dir)
                img['src'] = os.path.relpath(local, html_path.parent)
                counts['logos_kept'] += 1
                counts['downloaded'] += int(downloaded)
                continue
            except RuntimeError as ex:
                logger.warning(f'Company logo for {company!r} not kept, blanking it: {ex}')
                counts['fetch_failures'] += 1
        img['src'] = ''
        img['data-removed'] = 'image'
        counts['images_blanked'] += 1
    stylesheet_dir = html_path.parent / STYLESHEET_DIR_NAME
    for link in soup.find_all('link', rel='stylesheet'):
        href = str(link.get('href', ''))
        if not href.startswith('http'):
            continue
        try:
            local, downloaded = store_asset_once(href, stylesheet_dir)
            link['href'] = os.path.relpath(local, html_path.parent)
            counts['stylesheets'] += 1
            counts['downloaded'] += int(downloaded)
        except RuntimeError as ex:
            logger.warning(f'Stylesheet not stored, the archived page links the original: {ex}')
            counts['fetch_failures'] += 1
    return counts


def raw_html_path(candidate: dict) -> Path:
    return tools_generic.raw_posting_path(candidate).with_suffix('.html')


def save_raw_html(candidate: dict, soup: BeautifulSoup, meta: dict) -> Path | None:
    """Write the JS-free page beside the raw posting. A failure is logged, never fatal."""
    path = raw_html_path(candidate)
    header = yaml.safe_dump(meta, sort_keys=True, allow_unicode=True).replace('--', '- -')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'<!--\n{header}-->\n{soup}', encoding='utf-8')
    except OSError as ex:
        logger.warning(
            f"Could not archive the page HTML for job {candidate.get('job_id')} at {path}: "
            f'{type(ex).__name__}: {ex} — continuing without it.'
        )
        return None
    return path


# --- De-cluttering ---------------------------------------------------------------------------------

def _is_protected(el: Tag, protected: list[Tag]) -> bool:
    return any(_contains(el, p) or _contains(p, el) for p in protected)


def _section_root(el: Tag, protected: list[Tag]) -> Tag:
    """Climb from a matched element to the largest ancestor that holds only its own section."""
    node = el
    while node.parent is not None and node.parent.name not in ('main', 'body', '[document]'):
        parent = node.parent
        other_heading = any(h is not el and not _contains(node, h) for h in parent.find_all(_SECTION_HEADING_TAGS))
        if other_heading or any(_contains(parent, p) for p in protected):
            break
        node = parent
    return node


def protected_regions(soup: BeautifulSoup) -> list[Tag]:
    """What no removal rule may touch: the description, the company card's anchors and the logo.

    The logo stands in for the top card: a removal's section-root climb stops below any ancestor
    holding it, so a clutter section that shares a container with the top card ("Use AI to assess
    how you fit" can) is removed without taking the title and location with it.
    """
    about_job = soup.find(id=_ABOUT_JOB_ID_RE)
    logo = soup.find('img', alt=_COMPANY_LOGO_ALT_RE)
    return [el for el in (about_job, logo) if el is not None] + company_card_anchors(soup)


def remove_known_clutter(soup: BeautifulSoup) -> list[dict]:
    """Apply the fixed removal lists. Returns what was removed, for the log and the model's note."""
    protected = protected_regions(soup)
    removed = []
    for tag in LINKEDIN_REMOVE_TAGS:
        for el in soup.find_all(tag):
            if el.decomposed or _is_protected(el, protected):
                continue
            removed.append({'what': f'<{tag}>', 'chars': len(norm(el.get_text(' ')))})
            el.decompose()
    for label in LINKEDIN_REMOVE_ARIA_LABELS:
        for el in soup.find_all(attrs={'aria-label': label}):
            if el.decomposed or _is_protected(el, protected):
                continue
            removed.append({'what': f'aria-label {label!r}', 'chars': len(norm(el.get_text(' ')))})
            el.decompose()
    for phrase in LINKEDIN_REMOVE_SECTION_PHRASES:
        for el in soup.find_all(True):
            if el.decomposed or not _own_text(el).startswith(phrase):
                continue
            root = _section_root(el, protected)
            if _is_protected(root, protected):
                continue
            removed.append({'what': phrase, 'chars': len(norm(root.get_text(' ')))})
            root.decompose()
    return removed


def keep_anchors(main: Tag) -> list[Tag]:
    """Elements known to carry the posting's facts: the top card, the description, the company card.

    The top card has no id: it is the largest ancestor of the company logo that holds no section
    heading and not the description — so a new section beside it is still caught. The company card is anchored on its heading, its follower count and its
    description box, NOT on the whole company section, so anything new LinkedIn adds inside that
    section is still caught as an unknown block (the employee-posts carousel lived there).
    """
    anchors: list[Tag] = []
    about_job = main.find(id=_ABOUT_JOB_ID_RE)
    if about_job is not None:
        anchors.append(about_job)
    logo = main.find('img', alt=_COMPANY_LOGO_ALT_RE)
    if logo is not None and about_job is not None:
        top = logo
        while (top.parent is not None and top.parent is not main and not _contains(top.parent, about_job)
               and top.parent.find(_SECTION_HEADING_TAGS) is None):
            top = top.parent
        anchors.append(top)
    return anchors + company_card_anchors(main)


def company_card_anchors(root: Tag) -> list[Tag]:
    """The company card's heading, name/followers/industry block and description box."""
    about_company = root.find(id=_ABOUT_COMPANY_ID_RE)
    if about_company is None:
        return []
    anchors = []
    heading = about_company.find(_SECTION_HEADING_TAGS)
    if heading is not None:
        anchors.append(heading)
    for line_re in (_FOLLOWERS_RE, _EMPLOYEES_RE):
        line = about_company.find(string=lambda s, rx=line_re: bool(rx.match(norm(s))))
        if line is None:
            continue
        # The name/followers block and the industry/size line: climb to the largest ancestor that
        # still holds no heading, no description box and no other anchor line.
        node = line.parent
        while (node.parent is not None and node.parent is not about_company
               and node.parent.find(_SECTION_HEADING_TAGS) is None
               and node.parent.find(attrs={'data-testid': 'expandable-text-box'}) is None
               and not any(_contains(node.parent, a) for a in anchors)):
            node = node.parent
        anchors.append(node)
    blurb = about_company.find(attrs={'data-testid': 'expandable-text-box'})
    if blurb is not None:
        anchors.append(blurb)
    return anchors


def unknown_blocks(main: Tag) -> list[Tag]:
    """Maximal subtrees of <main> holding no keep anchor and some visible text."""
    anchors = keep_anchors(main)
    blocks: list[Tag] = []

    def walk(el: Tag) -> None:
        for child in el.find_all(True, recursive=False):
            if any(child is a for a in anchors) or child.name in _NON_CONTENT_TAGS:
                continue
            if any(_contains(child, a) for a in anchors):
                walk(child)
            elif re.search(r'[^\W\d_]', child.get_text(' ')):
                blocks.append(child)

    walk(main)
    return blocks


def block_signature(block: Tag) -> str:
    """A stable name for a block across jobs: its heading, else aria-label, else its first line."""
    heading = block.find(_HEADING_TAGS)
    text = (norm(heading.get_text(' ')) if heading is not None else '') or norm(block.get('aria-label', ''))
    if not text:
        text = next((norm(line) for line in block.get_text('\n').splitlines() if norm(line)), '')
    words = _DIGITS_RE.sub('N', text).split()
    return ' '.join(words[:PAGE_SECTION_SIGNATURE_MAX_WORDS])


def load_section_decisions() -> dict | None:
    """The cached verdicts, or None when the file is unreadable (then nothing may rewrite it)."""
    if not PAGE_SECTIONS_PATH.exists():
        return {}
    try:
        loaded = yaml.safe_load(PAGE_SECTIONS_PATH.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError) as ex:
        logger.warning(
            f'{PAGE_SECTIONS_PATH} is unreadable ({type(ex).__name__}: {ex}) — unknown page sections '
            f'are kept this run, and the file is left as it is for you to fix.'
        )
        return None
    if not isinstance(loaded, dict):
        logger.warning(f'{PAGE_SECTIONS_PATH} must hold a mapping, got {type(loaded).__name__} — left as it is.')
        return None
    return loaded


def _write_section_decisions(decisions: dict) -> None:
    try:
        PAGE_SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PAGE_SECTIONS_PATH.write_text(
            _PAGE_SECTIONS_HEADER + yaml.safe_dump(decisions, sort_keys=True, allow_unicode=True), encoding='utf-8')
    except OSError as ex:
        logger.warning(f'Could not write {PAGE_SECTIONS_PATH}: {type(ex).__name__}: {ex} — the verdicts '
                       f'apply to this page only and will be asked again.')


def _classify_prompt(candidate: dict, samples: list[tuple[str, str]]) -> str:
    listed = '\n\n'.join(f'[{i}] signature: {sig}\ntext:\n{text}' for i, (sig, text) in enumerate(samples, 1))
    return (
        f'Below are blocks of text from a LinkedIn job posting page ("{candidate.get("title", "")}" at '
        f'"{candidate.get("company", "")}"). They sit OUTSIDE the parts already known to matter (the '
        f'header card, "About the job", the company card), and nobody has classified them yet.\n\n'
        f"For each block decide whether it can be removed before a model extracts the job's facts.\n"
        f'remove=true ONLY when you are certain the block says nothing about THIS job or its '
        f'employer: advertising, Premium upsells, social posts, recommendations of other jobs, '
        f'people to contact, navigation, generic LinkedIn UI.\n'
        f'remove=false for anything about this job, its pay, location, work arrangement, '
        f'requirements, applicants or employer — and whenever you are unsure.\n\n'
        f'{listed}\n\n'
        f'Respond with ONLY a JSON object: {{"decisions": [{{"block": <number>, "remove": <true|false>, '
        f'"reason": "<a few words>"}}]}}'
    )


async def decide_unknown_blocks(soup: BeautifulSoup, candidate: dict, stage_stats: dict | None) -> list[dict]:
    """Remove unknown blocks that are cached, or newly judged, as clutter. Returns what was removed."""
    main = soup.find('main')
    if main is None:
        return []
    blocks = unknown_blocks(main)
    if not blocks:
        return []
    decisions = load_section_decisions()
    by_signature: dict[str, list[Tag]] = {}
    for block in blocks:
        by_signature.setdefault(block_signature(block), []).append(block)

    unseen = [sig for sig in by_signature if decisions is not None and sig not in decisions]
    if unseen:
        samples = [(sig, truncate_reported(
            norm(by_signature[sig][0].get_text(' ')), PAGE_SECTION_SAMPLE_MAX_CHARS,
            f'unknown page section {sig!r} shown to the classifier')) for sig in unseen]
        try:
            content, cost_usd = await chat_openrouter(_classify_prompt(candidate, samples), model=MODEL_NAME_PAGE_SECTIONS)
            if stage_stats is not None:
                stage_stats['cost'] += cost_usd
            verdicts = extract_json_object(content).get('decisions') or []
        except Exception as ex:
            logger.warning(
                f"Page-section classification failed for job {candidate.get('job_id')} "
                f'({type(ex).__name__}: {ex}) — keeping {len(unseen)} unknown section(s): {unseen}'
            )
            verdicts = []
        for verdict in verdicts:
            try:
                sig = unseen[int(verdict['block']) - 1]
            except (KeyError, ValueError, TypeError, IndexError) as ex:
                logger.warning(f'Ignoring a malformed page-section verdict {verdict!r}: {type(ex).__name__}: {ex}')
                continue
            decision = 'remove' if verdict.get('remove') is True else 'keep'
            decisions[sig] = {
                'decision': decision, 'reason': str(verdict.get('reason') or ''),
                'decided_by': MODEL_NAME_PAGE_SECTIONS, 'decided': date.today().isoformat(),
                'first_seen_job': str(candidate.get('job_id') or ''),
                'sample': snippet(norm(by_signature[sig][0].get_text(' ')), PAGE_SECTION_SAMPLE_MAX_CHARS),
            }
            logger.warning(
                f'New LinkedIn page section {sig!r} (job {candidate.get("job_id")}): {decision} — '
                f'{decisions[sig]["reason"]}. Recorded in {PAGE_SECTIONS_PATH.name}; edit it to overrule.'
            )
        if verdicts:
            _write_section_decisions(decisions)

    removed = []
    for sig, group in by_signature.items():
        entry = (decisions or {}).get(sig) or {}
        if entry.get('decision') != 'remove':
            continue
        for block in group:
            text = norm(block.get_text(' '))
            if compensation_context(text):
                logger.warning(
                    f'Page section {sig!r} is marked remove but holds pay wording — kept '
                    f"(job {candidate.get('job_id')}): {snippet(text, PAGE_SECTION_SAMPLE_MAX_CHARS)!r}")
                continue
            removed.append({'what': f'section {sig!r}', 'chars': len(text)})
            block.decompose()
    return removed


def page_text(soup: BeautifulSoup) -> str:
    """The visible text of <main>, one non-empty line per line."""
    root = soup.find('main') or soup.body or soup
    lines = (norm(line) for line in root.get_text('\n').splitlines())
    return '\n'.join(line for line in lines if line)


# --- Reading a page ----------------------------------------------------------------------------------

BrowserCall = Callable[[str, dict], Awaitable[str]]


async def _capture_dom(call: BrowserCall, job_id: str) -> dict:
    """One read-only serialization of the DOM. Raises with detail when the result is unusable."""
    raw = await call('browser_evaluate', {'function': LINKEDIN_PAGE_CAPTURE_JS})
    result = parse_evaluate_result(raw)
    if not result or not isinstance(result.get('html'), str):
        raise RuntimeError(f'job {job_id}: page capture returned no html ({len(raw)} chars of tool output)')
    utf16_units = len(result['html'].encode('utf-16-le')) // 2
    if utf16_units != result.get('chars'):
        raise RuntimeError(
            f'job {job_id}: page capture arrived with {utf16_units} UTF-16 units but the page sent '
            f'{result.get("chars")} — it was altered in transit')
    return result


async def _read_snapshot(candidate: dict, call: BrowserCall, reason: str) -> JobPage:
    """The pre-2026-09-24 path: the a11y snapshot, whitespace-collapsed, middle-cut to budget."""
    logger.warning(f"Job page {candidate.get('job_id')}: reading the accessibility snapshot instead — {reason}")
    snapshot = normalize_whitespace(await call('browser_snapshot', {}))
    budgeted = truncate_reported_middle(
        snapshot, EXTRACTOR_SNAPSHOT_MAX_CHARS, f"page snapshot for job {candidate.get('job_id')}")
    tools_generic.save_raw_posting(candidate, snapshot, 'snapshot', len(budgeted))
    return JobPage(text=budgeted, source='snapshot')


async def read_job_page(candidate: dict, call: BrowserCall, stage_stats: dict | None = None) -> JobPage:
    """Navigate to the posting and return the text the extractor should read.

    Falls back to the snapshot path, with a WARNING, when the capture fails or the description
    is not on the page — a markup change must degrade to the old behaviour, never to no page.
    """
    job_id = str(candidate.get('job_id') or '')
    await call('browser_navigate', {'url': candidate['url']})
    capture: dict | None = None
    soup: BeautifulSoup | None = None
    for attempt in range(LINKEDIN_PAGE_CAPTURE_RETRIES + 1):
        await call('browser_wait_for', {'time': EXTRACT_PAGE_RENDER_WAIT_SECONDS})
        try:
            capture = await _capture_dom(call, job_id)
        except Exception as ex:
            return await _read_snapshot(candidate, call, f'the DOM capture failed: {type(ex).__name__}: {ex}')
        soup = BeautifulSoup(capture['html'], 'lxml')
        if soup.find(id=_ABOUT_JOB_ID_RE) is not None:
            break
        logger.info(f'Job page {job_id}: no description yet (attempt {attempt + 1}), waiting again')
    else:
        return await _read_snapshot(candidate, call, 'the page has no JobDetails_AboutTheJob_ section')

    js_counts = strip_js(soup)
    company = posting_company(soup) or str(candidate.get('company') or '') or None
    html_path = raw_html_path(candidate)
    asset_counts = localize_assets(soup, company, html_path)
    save_raw_html(candidate, soup, {
        'url': str(candidate.get('url') or ''), 'job_id': job_id, 'company': company or '',
        'captured': datetime.now().isoformat(timespec='seconds'), 'dom_chars': len(capture['html']),
    })

    removed = remove_known_clutter(soup)
    removed += await decide_unknown_blocks(soup, candidate, stage_stats)
    text = page_text(soup)
    tools_generic.save_raw_posting(candidate, text, 'dom', len(text))
    removed_summary = ', '.join(f"{r['what']} ({r['chars']})" for r in removed) or 'nothing'
    logger.info(
        f"Job page {job_id}: DOM {len(capture['html']):,} chars → {len(text):,} chars of text; "
        f'removed {removed_summary}; JS {js_counts}; assets {asset_counts}'
    )
    return JobPage(text=text, source='dom', removed=removed)
