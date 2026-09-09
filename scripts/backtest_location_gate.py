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

from agentic_job_search import preferences
from agentic_job_search.agent import (
    derive_residency_scope, rejected_location, residency_spare,
)
from agentic_job_search.location import classify_location

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
# The condensed description is everything after the header block, i.e. after the URL line.
DESCRIPTION_RE = re.compile(r'^URL:[ \t]*\S+[ \t]*$(.*)', re.MULTILINE | re.DOTALL)


def load_postings(root: Path) -> list[dict]:
    """Every saved posting, newest directory first."""
    postings = []
    for path in sorted(root.glob('saved_jobs-*/job_posting-*.md'), reverse=True):
        text = path.read_text(encoding='utf-8', errors='replace')
        record = {'path': path, 'rating': int(m.group(1)) if (m := RATING_RE.search(path.name)) else 0}
        for field, pattern in FIELD_RE.items():
            record[field] = (m.group(1).strip() if (m := pattern.search(text)) else '')
        record['description'] = (m.group(1).strip() if (m := DESCRIPTION_RE.search(text)) else '')
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

    rejected = kept = failed_open = 0
    to_reject: list[dict] = []
    to_keep: list[dict] = []
    for record in sample:
        extract = {
            'location': record['location'], 'relocation': record['relocation'],
            'description': record['description'], 'workplace_type': record['workplace'],
            # Deliberately absent: every saved posting predates the field, so this exercises
            # derive_residency_scope() alone -- the half with no live model to hide behind.
            'residency_scope': '',
        }
        scope = derive_residency_scope(extract)
        spare = residency_spare(extract)

        # The two changes push in OPPOSITE directions -- whole-word matching is monotonically
        # restrictive, the residency spare monotonically permissive -- so a single before/after
        # count nets them out to nearly nothing and proves neither. Attribute them separately.
        # The baseline must evaluate BOTH halves, exactly as apply_hard_rules does. Comparing a
        # location-only baseline against location-or-relocation made every relocation rejection
        # look like a flip caused by this change.
        legacy = await _legacy_rejected_location(record['location'])
        if not legacy and record['relocation']:
            legacy = await _legacy_rejected_location(record['relocation'])
        reason = await rejected_location(record['location'], residency_spare=spare)
        relocation_reason = ''
        if not reason and record['relocation']:
            relocation_reason = await rejected_location(record['relocation'])
        facts = await classify_location(record['location'])
        if facts.get('source') == 'error':
            failed_open += 1

        was_rejected = bool(legacy)
        now_rejected = bool(reason or relocation_reason)
        verdict = 'REJECT' if now_rejected else 'keep'
        rejected, kept = (rejected + 1, kept) if now_rejected else (rejected, kept + 1)
        if now_rejected and not was_rejected:
            to_reject.append({**record, 'why': reason or relocation_reason})
        if was_rejected and not now_rejected:
            to_keep.append({**record, 'why': f'scope={scope or "unspecified"} spare={spare}', 'legacy': legacy})

        print(
            f'{verdict:7} was {record["rating"]}/5  {record["location"][:46]:46} '
            f'{(scope or "-"):13}'
            f'-> {",".join(facts.get("countries") or ["-"]):20} {",".join(facts.get("regions") or ["-"]):16} '
            f'{(reason or relocation_reason) or ""}'
        )

    print(f'\n{rejected} rejected, {kept} kept (of {len(sample)}).')
    if failed_open:
        print(
            f'\n!! {failed_open} of {len(sample)} location(s) could not be classified — the '
            f'OpenRouter MCP server on :8006 is not answering. The gate FAILS OPEN, so every one '
            f'of those reads as "keep" here regardless of the rules. Start the server and re-run; '
            f'until then only cached locations mean anything.'
        )

    print(f'\nFLIPPED TO REJECT — the whole-word matcher ({len(to_reject)}):')
    for record in to_reject:
        print(f'  {record["rating"]}/5  {record["location"][:52]:52} {record["why"]}')
    print('  (every one of these should be a posting the old substring matcher wrongly exempted)')

    print(f'\nFLIPPED TO KEEP — the residency rule ({len(to_keep)}):')
    for record in to_keep:
        print(f'  {record["rating"]}/5  {record["location"][:52]:52} {record["why"]}  was: {record["legacy"]}')
    print('  (each must be a REMOTE posting, in an EU country, with no stated residency requirement,')
    print('   OR one that explicitly offers a multi-country area. Anything else is a bug.)')

    print('\nRun again: every line should be identical and cost nothing (all cache hits).')
    return 0


async def _legacy_rejected_location(text: str) -> str:
    """The pre-2026-09-09 gate: substring matching, no residency spare.

    Deliberately a frozen copy here rather than a mode flag in production -- a comparison harness
    owning its own baseline beats shipping a compatibility switch nobody uses.
    """
    haystack = ' '.join(str(text or '').split()).lower()
    if not haystack:
        return ''
    if not (preferences.excluded_locations() or preferences.rejected_regions()):
        return ''
    if any(token in haystack for token in preferences.hybrid_acceptable_locations()):
        return ''
    for token in preferences.excluded_locations():
        if token in haystack:
            return token
    unwanted = preferences.rejected_regions()
    facts = await classify_location(haystack)
    if facts.get('broad_area'):
        return ''
    regions = facts.get('regions') or []
    if not regions or not all(region in unwanted for region in regions):
        return ''
    countries = facts.get('countries') or []
    return f'{(countries[0] if countries else regions[0])} ({regions[0]})'


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
