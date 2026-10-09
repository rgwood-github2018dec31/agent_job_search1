"""Reported text truncation: every cut is logged and visibly marked, never silent.

A cap on text sent to a model is a budget decision. When it bites, the log must say what was cut
and by how much, and the model must be told its input is partial. Deterministic checks (regex
gates) must not use this at all; they scan the full text.
"""
import logging
import re

from agentic_job_search.config import LOG_SNIPPET_MAX_CHARS, SNAPSHOT_HEAD_SHARE

logger = logging.getLogger(__name__)

_WHITESPACE_RUN_RE = re.compile(r'[ \t\f\v\r]+')
_BLANK_LINES_RE = re.compile(r'\n\s*\n+')


def truncation_marker(kept: int, total: int) -> str:
    return f'\n[truncated: kept {kept} of {total} chars]'


def truncate_reported(text: str, max_chars: int, what: str) -> str:
    """Return text capped at max_chars, logging a WARNING and appending a marker when it cuts.

    `what` names the text in the log line (e.g. "extract snapshot for job 123"), so a reader can
    tell which input lost content without a debugger.
    """
    if len(text) <= max_chars:
        return text
    logger.warning(
        f'Truncating {what}: {len(text)} chars exceeds the {max_chars}-char cap; '
        f'{len(text) - max_chars} chars dropped from the end'
    )
    return text[:max_chars] + truncation_marker(max_chars, len(text))


def truncate_reported_middle(text: str, max_chars: int, what: str,
                             head_share: float = SNAPSHOT_HEAD_SHARE) -> str:
    """Return text capped at max_chars by dropping from the MIDDLE, keeping a head and a tail.

    A cap that always eats the tail is not neutral: a job description keeps compensation, benefits
    and work-authorization statements at the bottom, and dropping 34,797 chars from the end of one
    LinkedIn snapshot is how a salary range stopped being checkable (job 4470298356, 2026-09-22).
    Where the fact lives at both ends, so must the budget.

    `head_share` is above half because a page's structure is front-loaded; only the trailing facts
    need rescuing. The marker names what was dropped and from where, so a reader never mistakes the
    join for the page's own text.
    """
    if len(text) <= max_chars:
        return text
    marker_template = '\n[truncated: dropped {dropped} chars from the MIDDLE; kept the first {head} and last {tail} of {total}]\n'
    # Budget the marker itself, so the result genuinely fits the cap it was given.
    reserve = len(marker_template.format(dropped=len(text), head=max_chars, tail=max_chars, total=len(text)))
    budget = max(max_chars - reserve, 0)
    head_chars = int(budget * head_share)
    tail_chars = budget - head_chars
    dropped = len(text) - head_chars - tail_chars
    logger.warning(
        f'Truncating {what}: {len(text)} chars exceeds the {max_chars}-char cap; '
        f'{dropped} chars dropped from the middle, keeping the first {head_chars} and last {tail_chars}'
    )
    marker = marker_template.format(dropped=dropped, head=head_chars, tail=tail_chars, total=len(text))
    return text[:head_chars] + marker + (text[-tail_chars:] if tail_chars else '')


def snippet(text: object, max_chars: int = LOG_SNIPPET_MAX_CHARS) -> str:
    """An excerpt of text for a log line, exception message, alert or audit cell.

    For display only: it never feeds a model or a check, so it does not log. When it cuts it says
    so and how much, so a reader never mistakes an excerpt for the whole message.
    """
    s = str(text)
    if len(s) <= max_chars:
        return s
    return f'{s[:max_chars]}… [+{len(s) - max_chars} of {len(s)} chars not shown]'


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and of blank lines; keep single line breaks."""
    return _BLANK_LINES_RE.sub('\n\n', _WHITESPACE_RUN_RE.sub(' ', text)).strip()


def pages_to_prompt(pages: list[str], name: str, max_chars: int) -> str:
    """Serialize per-page text for a prompt, reporting its size and normalizing/truncating to fit.

    The only place page lists are joined. Page count and chars per page go to DEBUG. Over the cap
    it first normalizes whitespace (a PDF of mostly padding must not push real content past the
    cap) with a WARNING, and only if still over truncates at a page-aware point with a second
    WARNING and a marker naming how many pages survived.
    """
    sizes = [len(p) for p in pages]
    total = sum(sizes)
    mean = total / len(pages) if pages else 0
    logger.debug(
        f'{name}: {len(pages)} page(s), {total} chars, {mean:.0f} chars/page (per page: {sizes})'
    )
    if total <= max_chars:
        return '\n\n'.join(pages)

    normalized = [n for n in (normalize_whitespace(p) for p in pages) if n]
    normalized_total = sum(len(p) for p in normalized)
    logger.warning(
        f'{name}: {total} chars exceeds the {max_chars}-char cap; normalized whitespace '
        f'→ {normalized_total} chars across {len(normalized)} non-empty page(s)'
    )
    joined = '\n\n'.join(normalized)
    if len(joined) <= max_chars:
        return joined

    kept_pages = 0
    running = 0
    for page in normalized:
        running += len(page)
        if running > max_chars:
            break
        kept_pages += 1
    logger.warning(
        f'{name}: still {len(joined)} chars after normalizing, over the {max_chars}-char cap; '
        f'truncating — {len(joined) - max_chars} chars dropped, pages 1–{kept_pages} of '
        f'{len(normalized)} kept whole'
    )
    return (
        joined[:max_chars]
        + f'\n[truncated: kept {max_chars} of {len(joined)} chars; pages 1–{kept_pages} of '
          f'{len(normalized)} complete]'
    )
