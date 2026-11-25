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
    SNAPSHOT_PRUNE_LEVEL_CARDS,
    SNAPSHOT_PRUNE_LEVEL_CHROME,
    SNAPSHOT_PRUNE_LEVEL_NO_CARDS,
    SNAPSHOT_RESULT_CARD_RE,
    SNAPSHOT_RESULT_COUNT_RE,
)

logger = logging.getLogger(__name__)

_RESULT_CARD_RE = re.compile(SNAPSHOT_RESULT_CARD_RE)
_RESULT_COUNT_RE = re.compile(SNAPSHOT_RESULT_COUNT_RE)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(' '))


def _subtree_end(lines: list[str], start: int) -> int:
    '''Index one past the last line of the subtree rooted at lines[start].'''
    depth = _indent(lines[start])
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > depth):
        end += 1
    return end


def _holds_search_ui(lines: list[str]) -> bool:
    '''True when these lines contain a result card or the results count -- never pane content.'''
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
    '''Return the snapshot with the job detail pane replaced by a one-line marker.

    Stage 1b never reads a posting, and the pane (the posting, the company profile, its posts feed)
    was most of every oversized snapshot. Returns text unchanged when there is no pane, or when the
    pane cannot be separated from the search UI -- the size cap downstream then still applies and
    warns. `what` names the snapshot in the log line.
    '''
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


# --- Prune Search Snapshot (POC, 2026-11-25): what the scraper model reads of a search page -------
# Levels (SNAPSHOT_PRUNE_LEVEL_*), each including the one before:
#   chrome    drop what no search step uses: the global navigation (its iframe and banners, keeping
#             the search box), the sidebar, toasts, images and figures, `/url` lines, the
#             `[cursor=pointer]` and `[aria-hidden]` tags, and nameless `generic` wrappers (their
#             children move up a level).
#   cards     each job card becomes one line with no ref: title · company · location. Code
#             harvests the cards from the live page (harvest_listings), so the model loses no
#             listing, and with no ref it cannot click into a card (its Dismiss button).
#   no_cards  the whole results list becomes one line giving the card count.
# Not wired into Stage 1b yet: scripts/compare_scraper_models_offline.py --snapshot-prune-level.

SNAPSHOT_PRUNE_LEVELS = (SNAPSHOT_PRUNE_LEVEL_CHROME, SNAPSHOT_PRUNE_LEVEL_CARDS, SNAPSHOT_PRUNE_LEVEL_NO_CARDS)
_YAML_FENCE_RE = re.compile(r'(```yaml\n)(.*?)(\n```)', re.DOTALL)
_DROPPED_NODE_RE = re.compile(
    r'^- (iframe|complementary|img|figure|/url:|region "Toast message"|heading "\d+ notifications"'
    r'|generic "Skip navigation menu")(?=[\s:\[]|$)')
_DROPPED_TAG_RE = re.compile(r' \[(cursor=pointer|aria-hidden)\]')
_NAMELESS_WRAPPER_RE = re.compile(r'^- generic( \[ref=[^\]]+\])?:?$')
_DISMISS_RE = re.compile(r'^- button "Dismiss ')
_TEXT_LEAF_RE = re.compile(r'^- [a-z]+(?: "[^"]*")?(?: \[[^\]]*\])*: (.+)$')
# The texts a card line keeps, in page order: title, company, location.
_CARD_TEXT_FIELDS = 3
_VERIFIED_RE = re.compile(r'\s*\(Verified job\)')
_CARD_LINE_PREFIX = '- job card: '


class _Node:
    __slots__ = ('children', 'line')

    def __init__(self, line: str) -> None:
        self.line = line
        self.children: list[_Node] = []


def _tree(lines: list[str]) -> list[_Node]:
    '''Snapshot lines as a forest, by indentation; a non-node line belongs to the node above it.'''
    roots: list[_Node] = []
    stack: list[tuple[int, _Node]] = []
    for line in lines:
        if not line.strip():
            continue
        node = _Node(line.strip())
        depth = _indent(line)
        if not node.line.startswith('- ') and stack:
            stack[-1][1].line += ' ' + node.line
            continue
        while stack and stack[-1][0] >= depth:
            stack.pop()
        (stack[-1][1].children if stack else roots).append(node)
        stack.append((depth, node))
    return roots


def _render(nodes: list[_Node], depth: int = 0) -> list[str]:
    out = []
    for node in nodes:
        out.append('  ' * depth + node.line)
        out += _render(node.children, depth + 1)
    return out


def _descendants(node: _Node):
    for child in node.children:
        yield child
        yield from _descendants(child)


def _is_card(node: _Node) -> bool:
    return node.line.startswith('- button') and any(_DISMISS_RE.match(d.line) for d in _descendants(node))


def _card_line(node: _Node) -> str:
    texts = []
    for d in _descendants(node):
        if _DISMISS_RE.match(d.line):
            continue
        if (match := _TEXT_LEAF_RE.match(d.line)) and (text := _VERIFIED_RE.sub('', match.group(1)).strip()) \
                and text not in texts:
            texts.append(text)
    return f'{_CARD_LINE_PREFIX}{" · ".join(texts[:_CARD_TEXT_FIELDS])}'


def _cards_as_count(nodes: list[_Node]) -> list[_Node]:
    '''Each run of consecutive card lines replaced by one line giving its count.'''
    out: list[_Node] = []
    run = 0
    for node in [*nodes, None]:
        if node is not None and node.line.startswith(_CARD_LINE_PREFIX):
            run += 1
            continue
        if run:
            out.append(_Node(f'- [results list: {run} job cards omitted; harvest_listings reads them from the '
                             f'page. Never click inside the results list]'))
            run = 0
        if node is not None:
            out.append(node)
    return out


def _search_boxes(node: _Node) -> list[_Node]:
    return [d for d in _descendants(node) if d.line.startswith('- search')]


def _prune(nodes: list[_Node], level: str, stats: dict) -> list[_Node]:
    out: list[_Node] = []
    for node in nodes:
        if _DROPPED_NODE_RE.match(node.line):
            continue
        node.line = _DROPPED_TAG_RE.sub('', node.line)
        if node.line.startswith('- banner'):
            node.children = _search_boxes(node)
        if level != SNAPSHOT_PRUNE_LEVEL_CHROME and _is_card(node):
            stats['cards'] += 1
            out.append(_Node(_card_line(node)))
            continue
        node.children = _prune(node.children, level, stats)
        if level == SNAPSHOT_PRUNE_LEVEL_NO_CARDS:
            node.children = _cards_as_count(node.children)
        if _NAMELESS_WRAPPER_RE.match(node.line):
            out += node.children
        else:
            out.append(node)
    return out


def prune_search_snapshot(text: str, level: str, what: str) -> str:
    '''The snapshot in `text` (a tool result's ```yaml block, else the whole text) pruned to `level`.

    Run after drop_job_detail_pane. Raises ValueError on an unknown level.
    '''
    if level not in SNAPSHOT_PRUNE_LEVELS:
        raise ValueError(f'unknown snapshot prune level {level!r} for {what}; one of {SNAPSHOT_PRUNE_LEVELS}')

    def prune_block(block: str) -> str:
        stats = {'cards': 0}
        pruned = '\n'.join(_render(_prune(_tree(block.split('\n')), level, stats)))
        logger.debug(f'{what}: snapshot pruned to {level!r}, {len(block):,} -> {len(pruned):,} chars, '
                     f'{stats["cards"]} job card(s) collapsed')
        return pruned

    if _YAML_FENCE_RE.search(text):
        return _YAML_FENCE_RE.sub(lambda m: m.group(1) + prune_block(m.group(2)) + m.group(3), text)
    return prune_block(text)
