"""Remove regions of a LinkedIn search-page a11y snapshot that Stage 1b has no use for.

A Playwright snapshot is an indented tree, one node per line: a node's subtree is its own line plus
every following line indented deeper. It is not strict YAML (unquoted colons in text), so it is
walked by indentation rather than parsed.

Only the job detail pane is removed. The chip row, the results count and every result card stay:
the scraper sets filters from the first and must be able to see the rest.
"""
import logging
import re

from agentic_job_search.config import (
    SNAPSHOT_DETAIL_PANE_HEADING,
    SNAPSHOT_RESULT_CARD_RE,
    SNAPSHOT_RESULT_COUNT_RE,
)

logger = logging.getLogger(__name__)

_RESULT_CARD_RE = re.compile(SNAPSHOT_RESULT_CARD_RE)
_RESULT_COUNT_RE = re.compile(SNAPSHOT_RESULT_COUNT_RE)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(' '))


def _subtree_end(lines: list[str], start: int) -> int:
    """Index one past the last line of the subtree rooted at lines[start]."""
    depth = _indent(lines[start])
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > depth):
        end += 1
    return end


def _holds_search_ui(lines: list[str]) -> bool:
    """True when these lines contain a result card or the results count -- never pane content."""
    return any(_RESULT_CARD_RE.match(line.strip()) or _RESULT_COUNT_RE.search(line) for line in lines)


def _pane_span(lines: list[str], heading_index: int) -> tuple[int, int] | None:
    """The largest subtree around the pane heading that holds no search UI, or None.

    Walks up the heading's ancestors and stops at the first one that would also take a result card
    or the results count with it; the last ancestor before that is the pane. If no ancestor holds
    search UI this is not a search page (e.g. a single job's page, where the "pane" is the whole
    page), so there is nothing to separate and None is returned.
    """
    span = None
    depth = _indent(lines[heading_index])
    for i in range(heading_index - 1, -1, -1):
        if not lines[i].strip() or _indent(lines[i]) >= depth:
            continue
        depth = _indent(lines[i])
        end = _subtree_end(lines, i)
        if _holds_search_ui(lines[i:end]):
            return span
        span = (i, end)
    return None


def drop_job_detail_pane(text: str, what: str) -> str:
    """Return the snapshot with the job detail pane replaced by a one-line marker.

    Stage 1b never reads a posting, and the pane (the posting, the company profile, its posts feed)
    was most of every oversized snapshot. Returns text unchanged when there is no pane, or when the
    pane cannot be separated from the search UI -- the size cap downstream then still applies and
    warns. `what` names the snapshot in the log line.
    """
    lines = text.split('\n')
    heading_index = next((i for i, line in enumerate(lines) if SNAPSHOT_DETAIL_PANE_HEADING in line), None)
    if heading_index is None:
        return text
    span = _pane_span(lines, heading_index)
    if span is None:
        return text
    start, end = span
    removed_chars = sum(len(line) + 1 for line in lines[start:end])
    marker = (f'{" " * _indent(lines[start])}- [job detail pane omitted: {removed_chars} chars; '
              f'this search does not read postings]')
    pruned = '\n'.join([*lines[:start], marker, *lines[end:]])
    logger.debug(f'{what}: removed job detail pane, {len(text)} -> {len(pruned)} chars')
    return pruned
