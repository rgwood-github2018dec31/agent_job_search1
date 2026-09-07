"""Replay real saved job postings through the location gate, against the LIVE classifier.

Why this is a tracked script and not a scratchpad one-off
--------------------------------------------------------
The gate turns a stated preference into an unappealable auto-reject, and the geography behind it
comes from a model. Both can drift: the classifier model changes, or the region policy does. The
triage-model swap left the same lesson in CLAUDE.md — *"Re-run that comparison before swapping this
model again"* — after 17 real postings were what proved `granite4.1:3b` had no false rejects. This
is that comparison for the location gate.

Usage:
    uv run python scripts/backtest_location_gate.py            # ~20 postings, spread across regions
    uv run python scripts/backtest_location_gate.py --all      # every saved posting
    uv run python scripts/backtest_location_gate.py --limit 40

Needs the OpenRouter MCP tool server on :8006. Reads only; nothing is saved, rated or notified.
The location cache is shared with the real runs, so a second invocation costs nothing.
"""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agentic_job_search import preferences  # noqa: E402
from agentic_job_search.agent import rejected_location  # noqa: E402
from agentic_job_search.location import classify_location  # noqa: E402

logger = logging.getLogger('backtest')

# `[ \t]*`, never `\s*`: \s matches a newline, so on a posting with an empty `Location:` field the
# group happily swallowed the line break and captured the NEXT line instead — one real saved job
# was being judged on its `Posted: Within the past 24 hours` line.
FIELD_RE = {
    'location': re.compile(r'^Location:[ \t]*(\S.*)$', re.MULTILINE),
    'relocation': re.compile(r'^Relocation required:[ \t]*(\S.*)$', re.MULTILINE),
    'local_language': re.compile(r'^Local working language:[ \t]*(\S.*)$', re.MULTILINE),
    'posting_language': re.compile(r'^Posting written in:[ \t]*(\S.*)$', re.MULTILINE),
    'workplace': re.compile(r'^Workplace:[ \t]*(\S.*)$', re.MULTILINE),
}
RATING_RE = re.compile(r'-rating_(\d)-')


def load_postings(root: Path) -> list[dict]:
    """Every saved posting, newest directory first."""
    postings = []
    for path in sorted(root.glob('saved_jobs-*/job_posting-*.md'), reverse=True):
        text = path.read_text(encoding='utf-8', errors='replace')
        record = {'path': path, 'rating': int(m.group(1)) if (m := RATING_RE.search(path.name)) else 0}
        for field, pattern in FIELD_RE.items():
            record[field] = (m.group(1).strip() if (m := pattern.search(text)) else '')
        if record['location']:
            postings.append(record)
    return postings


def spread(postings: list[dict], limit: int) -> list[dict]:
    """Pick a sample that actually exercises the gate rather than N postings from one country.

    Three things this must avoid, each of which produced a useless run while writing it:
      - newest-first, which is dominated by whatever the last run happened to surface;
      - alphabetical, which returned 22 locations all starting with A or B;
      - random, which is not reproducible across the two runs the cache check needs.
    So: dedupe to one posting per distinct location (preferring the highest-rated, since a 5 the
    gate now rejects is the finding worth seeing), then take an even STRIDE through that list.
    """
    by_location: dict[str, dict] = {}
    for record in sorted(postings, key=lambda r: -r['rating']):
        key = record['location'].lower()[:40]
        by_location.setdefault(key, record)
    distinct = sorted(by_location.values(), key=lambda r: r['location'].lower())
    if limit >= len(distinct):
        return distinct
    stride = len(distinct) / limit
    return [distinct[int(i * stride)] for i in range(limit)]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--limit', type=int, default=20, help='how many postings to replay (default 20)')
    parser.add_argument('--all', action='store_true', help='replay every saved posting')
    parser.add_argument('--match', default='', help='only postings whose filename contains this')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

    if not (preferences.excluded_locations() or preferences.rejected_regions()):
        print('locations.exclude and locations.reject_regions are both empty — the gate is inert.')
        print('Nothing to backtest. Configure them in run_dir/preferences.yaml first.')
        return 1

    root = Path(__file__).parent.parent / 'run_dir'
    postings = load_postings(root)
    if not postings:
        print(f'No saved postings found under {root}/saved_jobs-*/')
        return 1
    if args.match:
        postings = [r for r in postings if args.match.lower() in r['path'].name.lower()]
    sample = postings if (args.all or args.match) else spread(postings, args.limit)

    print(f'Replaying {len(sample)} of {len(postings)} saved postings through the location gate.')
    print(f'  reject_regions: {list(preferences.rejected_regions())}')
    print(f'  exclude:        {list(preferences.excluded_locations())}')
    print(f'  exempt:         {list(preferences.hybrid_acceptable_locations())[:6]}...\n')

    rejected = kept = 0
    flipped: list[dict] = []
    for record in sample:
        reason = await rejected_location(record['location'])
        relocation_reason = ''
        if record['relocation']:
            relocation_reason = await rejected_location(record['relocation'])
        facts = await classify_location(record['location'])

        verdict = 'REJECT' if (reason or relocation_reason) else 'keep'
        if reason or relocation_reason:
            rejected += 1
            if record['rating'] >= 4:
                flipped.append(record)
        else:
            kept += 1

        print(
            f'{verdict:7} was {record["rating"]}/5  {record["location"][:52]:52} '
            f'-> {",".join(facts.get("countries") or ["-"]):22} {",".join(facts.get("regions") or ["-"]):16} '
            f'{(reason or relocation_reason) or ""}'
        )

    print(f'\n{rejected} rejected, {kept} kept.')
    print(f'{len(flipped)} posting(s) previously rated >=4 are now rejected:')
    for record in flipped:
        print(f'  {record["rating"]}/5  {record["location"][:60]}  {record["path"].name[:70]}')
    print('\nRun again: every line should be identical and cost nothing (all cache hits).')
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
