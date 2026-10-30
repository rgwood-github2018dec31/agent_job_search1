'''Copy retained job-page captures into run_dir/replay_corpus/, out of reach of retention pruning.

Why this is a tracked script
----------------------------
A run prunes run_dir/raw_postings/ to RAW_POSTINGS_RETENTION_DAYS. The replay scripts measure a change
on the same pages run after run, and on 2026-10-30 a run started during a measurement pruned 84 of its
243 pages. Pinned copies keep the {date}/{name} layout, and the replay scripts read a pinned copy in
place of the raw_postings path when one exists (replay_evidence.pinned; replay_markdown_ab.py --corpus).

Per capture it copies the page text (.txt) and the full page archive (.html), which the replays read,
plus the rater's posting (.rater_posting.txt) when there is one. It copies only: nothing is moved or
deleted, and a copy already pinned is left as it is.

Usage:
    uv run python scripts/pin_replay_corpus.py --reference run_dir/replays/markdown-ab-<ts>.json   # its pages
    uv run python scripts/pin_replay_corpus.py --all                                               # every retained capture
'''
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from replay_evidence import RAW_POSTINGS_DIR, REPLAY_CORPUS_DIR

logger = logging.getLogger('pin_replay_corpus')

CAPTURE_SUFFIXES = ('.txt', '.html', '.rater_posting.txt')
RATER_POSTING_SUFFIX = '.rater_posting.txt'


def captures(args: argparse.Namespace) -> list[Path]:
    '''The .txt captures to pin.'''
    if args.reference:
        return [Path(p['path']) for p in json.loads(args.reference.read_text(encoding='utf-8'))]
    return sorted(p for p in RAW_POSTINGS_DIR.glob('*/linkedin-*.txt') if not p.name.endswith(RATER_POSTING_SUFFIX))


def pin(capture: Path) -> str:
    '''Copy one capture's files; returns 'pinned', 'already pinned' or 'gone' (pruned before it was pinned).'''
    target = REPLAY_CORPUS_DIR / capture.parent.name
    if (target / capture.name).exists():
        return 'already pinned'
    if not capture.exists() or not capture.with_suffix('.html').exists():
        return 'gone'
    target.mkdir(parents=True, exist_ok=True)
    stem = capture.name.removesuffix('.txt')
    for suffix in CAPTURE_SUFFIXES:
        source = capture.parent / f'{stem}{suffix}'
        if source.exists():
            shutil.copy2(source, target / source.name)
    return 'pinned'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument('--reference', type=Path, help="a replay's raw results: pin the pages it replayed")
    which.add_argument('--all', action='store_true', help='pin every capture still in run_dir/raw_postings')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
    outcomes: dict[str, list[str]] = {}
    for capture in captures(args):
        outcomes.setdefault(pin(capture), []).append(f'{capture.parent.name}/{capture.name}')
    for outcome, names in sorted(outcomes.items()):
        logger.info(f'{outcome}: {len(names)}')
    if outcomes.get('gone'):
        logger.warning(f"Not pinned, already pruned from {RAW_POSTINGS_DIR}: {', '.join(outcomes['gone'])}")
    logger.info(f'Corpus: {REPLAY_CORPUS_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
