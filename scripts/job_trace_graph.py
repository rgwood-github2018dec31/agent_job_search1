#!/usr/bin/env python
'''Decision Graph (POC): every Job Trace of one date laid over Stage 2, counted.

Each node on the spine is a stage and each box beside it a way out of the pipeline there; every
edge carries the number of jobs that went that way. The path model is job_path (src); one job's
own path is drawn in its Job Audit instead (job_trace_report.py).

Usage:
    uv run --with reportlab python scripts/job_trace_graph.py                # latest date
    ... --date 2026-11-04 --out run_dir/graph.pdf

Reads only files the run already wrote; never calls a model and costs nothing.
'''

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from agentic_job_search import job_path  # noqa: E402
from agentic_job_search import job_trace as T  # noqa: E402
from job_path_drawing import path_flowables  # noqa: E402
from job_trace_report import UNICODE_FONTS_REGISTERED  # noqa: E402

logger = logging.getLogger(__name__)

RUN_DIR = PROJECT_DIR / 'run_dir'
RAW_POSTINGS_DIR = RUN_DIR / 'raw_postings'
JOB_AUDITS_DIR = RUN_DIR / 'job_audits'
TRACE_SUFFIX = '.trace.jsonl'
MARGIN = 36                 # PDF points
TITLE_SIZE, TITLE_LEADING, TEXT_SIZE, TEXT_LEADING = 14, 18, 8, 10


def latest_date() -> str:
    dates = sorted(p.name for p in RAW_POSTINGS_DIR.iterdir() if p.is_dir() and any(p.glob(f'*{TRACE_SUFFIX}')))
    if not dates:
        raise SystemExit(f'No Job Traces under {RAW_POSTINGS_DIR}')
    return dates[-1]


def load_traces(run_date: str) -> dict[str, list[dict]]:
    '''`{site}-{job_id}` -> steps, for every non-empty trace of the date.'''
    folder = RAW_POSTINGS_DIR / run_date
    paths = sorted(folder.glob(f'*{TRACE_SUFFIX}'))
    if not paths:
        raise SystemExit(f'No Job Traces in {folder}')
    traces = {p.name.removesuffix(TRACE_SUFFIX): T.load(p) for p in paths}
    return {key: steps for key, steps in traces.items() if steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--date', help='date folder under run_dir/raw_postings (default: latest with traces)')
    parser.add_argument('--out', type=Path, help='PDF path (default run_dir/job_audits/graph-{date}.pdf)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    run_date = args.date or latest_date()
    traces = load_traces(run_date)
    rules = job_path.rule_order(list(traces.values()))
    overlays = [job_path.overlay(steps, rules) for steps in traces.values()]
    outcomes = Counter(job_path.outcome(nodes) for nodes in overlays)
    without_outcome = sum(not any(s['kind'] == job_path.OUTCOME_STEP_KIND for s in steps) for steps in traces.values())

    fonts = ('TraceSans', 'TraceSansBold') if UNICODE_FONTS_REGISTERED else ('Helvetica', 'Helvetica-Bold')
    sheet = getSampleStyleSheet()
    title = ParagraphStyle('GraphTitle', parent=sheet['Title'], fontName=fonts[1], fontSize=TITLE_SIZE,
                           leading=TITLE_LEADING, alignment=0)
    text = ParagraphStyle('GraphText', parent=sheet['Normal'], fontName=fonts[0], fontSize=TEXT_SIZE,
                          leading=TEXT_LEADING)
    story = [
        Paragraph(f'Stage 2 decision graph — {run_date}', title),
        Paragraph(f'{len(overlays)} Job Traces read on {run_date} (every run that day). Numbers on the spine are '
                  "jobs reaching each stage; edge width follows them. One job's own path is in its Job Audit.", text),
        Paragraph('Outcomes: ' + ', '.join(f'{c} {o}' for o, c in outcomes.most_common()), text),
    ]
    if without_outcome:
        story.append(Paragraph(f'{without_outcome} trace(s) have no outcome step (written before 2026-11-12, or '
                               'the run stopped during the job): their exit is drawn where the trace stops, '
                               'as "outcome not recorded".', text))
    story += path_flowables(job_path.aggregate(overlays, rules), fonts, counts=True)

    out = args.out or JOB_AUDITS_DIR / f'graph-{run_date}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                      bottomMargin=MARGIN, title=f'Decision graph {run_date}').build(story)
    logger.info(f'Decision graph: {out} ({len(overlays)} traces)')


if __name__ == '__main__':
    main()
