'''One-time release of jobs the sponsorship hard rule rejected outside the user's sponsorship locations.

Until 2026-10-27 the "no sponsorship" patterns, which name no country, rejected a posting anywhere:
43 of the 48 sponsorship rejections on record were Canada and EU roles the user may already work in.
Each such job has a processed record, so every later run deduped it away.

A job qualifies when its saved posting is a `job without explicit visa sponsorship` auto-reject saved
within JOB_STALE_AGE_DAYS, and the fixed rule (`agent.sponsorship_rejected`) now passes it. Releasing
it moves its processed record(s) and the saved auto-reject posting to run_dir/_pending_delete/. The next
run rates it, and notifies it if it rates high enough, when a search lists it again; `--write-ids`
writes the jobs as a `--job-ids` file, to rate them now without waiting for that.

Usage:
    uv run python scripts/release_wrong_sponsorship_rejections.py                 # dry run: list them
    uv run python scripts/release_wrong_sponsorship_rejections.py --write-ids run_dir/wrong_sponsorship.tsv
    uv run python scripts/release_wrong_sponsorship_rejections.py --apply         # release them
'''

import argparse
import logging
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import agentic_job_search.tools_generic as tools
from agentic_job_search.agent import sponsorship_rejected
from agentic_job_search.config import JOB_STALE_AGE_DAYS

logger = logging.getLogger('release_wrong_sponsorship_rejections')

REJECTION_GLOB = 'saved_jobs-*/job_posting-*-auto_rejected_job_without_explicit_visa_sponsorship-*.md'
SAVED_NAME_RE = re.compile(r'job_posting-(?P<job_id>\d+)-rating_\d+-(?P<company>[^-]+)-')
SAVED_DIR_DATE_FORMAT = '%Y%b%d'
FIELD_RE = r'^{name}: (?P<value>.*)$'


def _field(text: str, name: str) -> str:
    match = re.search(FIELD_RE.format(name=re.escape(name)), text, re.MULTILINE)
    return match['value'].strip() if match else ''


def find_wrong_rejections(run_dir: Path) -> list[dict]:
    '''Recent sponsorship auto-rejects that the fixed rule would let through, with a processed record.'''
    oldest = date.today() - timedelta(days=JOB_STALE_AGE_DAYS)
    jobs = []
    for posting in sorted(run_dir.glob(REJECTION_GLOB)):
        saved_on = datetime.strptime(posting.parent.name.removeprefix('saved_jobs-'), SAVED_DIR_DATE_FORMAT).date()
        if saved_on < oldest:
            continue
        if not (name := SAVED_NAME_RE.match(posting.name)):
            raise ValueError(f'Cannot read the job id from saved posting name {posting.name!r} ({posting})')
        # The auto-reject heading itself says 'sponsorship', which would hide a posting that never does.
        text = posting.read_text(encoding='utf-8').partition('\n')[2]
        location = _field(text, 'Location')
        if sponsorship_rejected(location, text):
            logger.info(f"still rejected by the fixed rule: {name['job_id']} {name['company']} ({location})")
            continue
        records = list((run_dir / 'processed_jobs').glob(f"job_posting-linkedin-{name['job_id']}-*.yaml"))
        if not records:
            logger.info(f"no processed record, nothing to release: {name['job_id']} {name['company']}")
            continue
        jobs.append({
            'job_id': name['job_id'], 'company': _field(text, 'Company') or name['company'],
            'title': _field(text, 'Title'), 'location': location, 'saved_on': saved_on,
            'posting': posting, 'records': records,
        })
    return jobs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true', help='release the jobs (default: list them only)')
    parser.add_argument('--write-ids', type=Path, metavar='FILE', help='write the jobs as a --job-ids file')
    args = parser.parse_args()

    jobs = find_wrong_rejections(tools.RUN_DIR)
    for job in jobs:
        logger.info(f"wrongly rejected: {job['job_id']}  {job['company']} — {job['title']}  "
                    f"({job['location']}; saved {job['saved_on']})")
    logger.info(f'{len(jobs)} wrongly rejected job(s) within JOB_STALE_AGE_DAYS')

    if args.write_ids:
        args.write_ids.write_text(
            ''.join(f"{j['job_id']}\t{j['title']}\t{j['company']}\n" for j in jobs), encoding='utf-8')
        logger.info(f'Wrote {args.write_ids}')
    if not args.apply:
        logger.info('Dry run: nothing released (pass --apply)')
        return

    backup_dir = tools.RUN_DIR / '_pending_delete' / f"wrong_sponsorship_rejections-{datetime.now().strftime('%Y%b%d')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        for path in [*job['records'], job['posting']]:
            shutil.move(path, backup_dir / path.name)
        logger.info(f"{job['job_id']} {job['company']}: released for the next run")
    logger.info(f'Processed records and saved auto-rejects moved to {backup_dir}')


if __name__ == '__main__':
    main()
