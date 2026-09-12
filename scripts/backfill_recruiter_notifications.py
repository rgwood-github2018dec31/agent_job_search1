"""Seed run_dir/recruiter_notifications.yaml from saved jobs already notified.

The repost check can only suppress a second notification for a role it has seen a FIRST
notification for. Without this, the agency reposts already in flight — Archer Recruitment
advertised one pharma programme under 6+ job ids between 2026-09-02 and 09-10 — would each get one
more notification before the store had anything to compare against.

Reconstructs an entry per saved posting that (a) was rated 4 or 5, so it was notified, (b) carries
the deterministic "Posted by a recruiting agency" warning, and (c) falls inside
RECRUITER_REPOST_WINDOW_DAYS. Dry-run by default; writes only with --apply.

    uv run python scripts/backfill_recruiter_notifications.py
    uv run python scripts/backfill_recruiter_notifications.py --apply
"""

import argparse
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from agentic_job_search import tools_generic as tools
from agentic_job_search.config import RECRUITER_REPOST_WINDOW_DAYS

logger = logging.getLogger(__name__)

_AGENCY_WARNING = 'Posted by a recruiting agency'
_JOB_ID_RE = re.compile(r'job_posting-(\d+)-rating_([45])-')
# "Title: ...", "Hiring company: ...". The extract header block that format_extract_text() writes.
_FIELD_RE = re.compile(r'^([A-Z][A-Za-z -]{1,30}): (.*)$')


def _saved_dir_date(directory: Path) -> date | None:
    """The run date encoded in a `saved_jobs-2026Sep09` directory name."""
    try:
        return datetime.strptime(directory.name.removeprefix('saved_jobs-'), '%Y%b%d').date()
    except ValueError:
        return None


def _parse_saved_job(path: Path) -> tuple[dict, str] | None:
    """The extract-shaped fields and the description body of one saved job, or None."""
    text = path.read_text(encoding='utf-8')
    if _AGENCY_WARNING not in text:
        return None

    lines = text.splitlines()
    fields: dict[str, str] = {}
    last_field_line = -1
    for i, line in enumerate(lines):
        if match := _FIELD_RE.match(line):
            fields[match.group(1)] = match.group(2).strip()
            last_field_line = i
    description = '\n'.join(lines[last_field_line + 1:]).strip()
    return fields, description


def collect(run_dir: Path, today: date | None = None) -> list[dict]:
    """One record per notified agency posting inside the window, oldest first."""
    cutoff = (today or date.today()) - timedelta(days=RECRUITER_REPOST_WINDOW_DAYS)
    records: list[dict] = []

    for directory in sorted(run_dir.glob('saved_jobs-*')):
        notified = _saved_dir_date(directory)
        if notified is None or notified < cutoff:
            continue
        for path in sorted(directory.glob('job_posting-*.md')):
            match = _JOB_ID_RE.search(path.name)
            if not match:
                continue
            parsed = _parse_saved_job(path)
            if parsed is None:
                continue
            fields, description = parsed
            records.append({
                'notified': notified,
                'candidate': {'job_id': match.group(1), 'company': fields.get('Company', '')},
                'extract': {
                    'company': fields.get('Company', ''),
                    'title': fields.get('Title', ''),
                    'location': fields.get('Location', ''),
                    'salary': fields.get('Salary', ''),
                    'end_client': fields.get('Hiring company', ''),
                    'description': description,
                },
                'source': path.name,
            })

    records.sort(key=lambda r: (r['notified'], r['candidate']['job_id']))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='write the entries (default: dry run)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    records = collect(tools.RUN_DIR)
    if not records:
        logger.info(f'No notified agency postings in the last {RECRUITER_REPOST_WINDOW_DAYS} days.')
        return

    for record in records:
        logger.info(
            f"{record['notified']}  {record['candidate']['company']:<24} "
            f"job {record['candidate']['job_id']}  {record['extract']['title']}"
        )

    if not args.apply:
        logger.info(
            f'\nDry run: {len(records)} entry(ies) would be written to '
            f'{tools.RECRUITER_NOTIFICATIONS_PATH}. Re-run with --apply to write them.'
        )
        return

    # Oldest first, so the newest ends up at the head of the file like a normal run would leave it.
    for record in records:
        tools.record_recruiter_notification(
            record['candidate'], record['extract'], today=record['notified']
        )
    logger.info(f'\nWrote {len(records)} entry(ies) to {tools.RECRUITER_NOTIFICATIONS_PATH}.')


if __name__ == '__main__':
    main()
