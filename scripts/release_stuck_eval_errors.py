'''One-time release of jobs left in processed_jobs/ by an eval_error from before release_errored_job().

Until 2026-10-09 an `eval_error` in Stage 2 was terminal: Stage 1 had already written the job to
processed_jobs/, so every later run deduped it away and it was never rated. A comparison with
LinkedIn's own job-recommendation emails (2026-10-22) found such jobs still being listed weeks later,
e.g. a Toronto Lead AI Engineer posting at $200–230K that errored on 2026-09-13 on an OpenRouter
timeout and was then skipped as `already_processed` ten times.

A job is stuck when the audit logs show it ended in `eval_error`, it has a processed record, and no
saved posting exists for it. Each one is released through `release_errored_job()`, so it counts
against `EVAL_ERROR_MAX_RELEASES` like any other release, after its record is copied to
run_dir/_pending_delete/. The next run rates it if a search lists it again; `--write-ids` writes
the stuck jobs as a `--job-ids` file, to rate them now without waiting for that.

Usage:
    uv run python scripts/release_stuck_eval_errors.py                 # dry run: list them
    uv run python scripts/release_stuck_eval_errors.py --write-ids run_dir/stuck_eval_errors.tsv
    uv run python scripts/release_stuck_eval_errors.py --apply         # release them
'''

import argparse
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import agentic_job_search.tools_generic as tools

logger = logging.getLogger('release_stuck_eval_errors')

AUDIT_ROW = re.compile(r'^\| (?P<company>[^|]*) \| (?P<title>[^|]*) \| (?P<status>[a-z_]+) \|.*/jobs/view/(?P<job_id>\d+)/')
DEDUP_STATUSES = {'already_processed', 'already_applied'}  # a later skip says nothing about the evaluation


def find_stuck_jobs(run_dir: Path) -> list[dict]:
    '''Jobs whose last evaluation in the audit logs was eval_error and that were never saved.'''
    last_eval: dict[str, dict] = {}
    for audit_path in sorted((run_dir / 'audit_logs').glob('audit-*.md'), key=lambda p: p.stat().st_mtime):
        for line in audit_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if (match := AUDIT_ROW.match(line)) and match['status'] not in DEDUP_STATUSES:
                last_eval[match['job_id']] = {**match.groupdict(), 'audit': audit_path.name}
    saved_ids = {
        m[1] for p in run_dir.glob('saved_jobs-*/job_posting-*.md')
        if (m := re.match(r'job_posting-(\d+)-', p.name))
    }
    stuck = []
    for job_id, row in last_eval.items():
        if row['status'] != 'eval_error' or job_id in saved_ids:
            continue
        records = list((run_dir / 'processed_jobs').glob(f'job_posting-linkedin-{job_id}-*.yaml'))
        if records:
            stuck.append({**row, 'records': records})
    return stuck


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true', help='release the jobs (default: list them only)')
    parser.add_argument('--write-ids', type=Path, metavar='FILE', help='write the stuck jobs as a --job-ids file')
    args = parser.parse_args()

    stuck = find_stuck_jobs(tools.RUN_DIR)
    for job in stuck:
        logger.info(f"stuck: {job['job_id']}  {job['company']} — {job['title']}  (eval_error in {job['audit']})")
    logger.info(f'{len(stuck)} stuck job(s)')

    if args.write_ids:
        args.write_ids.write_text(
            ''.join(f"{j['job_id']}\t{j['title']}\t{j['company']}\n" for j in stuck), encoding='utf-8')
        logger.info(f'Wrote {args.write_ids}')
    if not args.apply:
        logger.info('Dry run: nothing released (pass --apply)')
        return

    backup_dir = tools.RUN_DIR / '_pending_delete' / f"stuck_eval_error_records-{datetime.now().strftime('%Y%b%d')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for job in stuck:
        for record in job['records']:
            shutil.copy2(record, backup_dir / record.name)
        logger.info(f"{job['job_id']} {job['company']}: {tools.release_errored_job('linkedin', job['job_id'])}")
    logger.info(f'Processed records copied to {backup_dir} before release')


if __name__ == '__main__':
    main()
