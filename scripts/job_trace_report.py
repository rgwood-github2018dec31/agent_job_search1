#!/usr/bin/env python
'''Full-trace PDF for one evaluated job: every artifact the pipeline kept, verbatim.

Written because a summary of a run hides exactly the thing worth finding. The reference-summary
truncation (2026-10-13) was invisible in every summary and obvious in one `Run funnel` line next to
the profile it had cut. So this prints the raw material — the page text the extractor read, the
extract it returned, the profile the rater was calibrated against, the rating it gave — and then
states plainly what the pipeline does NOT retain, because that gap is the other half of the answer.

Since 2026-10-19 every Stage 2 job also leaves a Job Trace beside its raw posting
(`raw_postings/{date}/{site}-{job_id}.trace.jsonl`): every step in order, every LLM call with its
full prompt and reply, every citation check, every hard rule with its inputs, the rating before and
after caps, and the exact Telegram text. When it exists this report opens with the decision path
it records and closes with the whole trace verbatim, and the "not kept" list shrinks to match.
A run renders one of these per run by itself (job_audit.py); this script renders any other.

Usage:
    uv run --with reportlab python scripts/job_trace_report.py --job-id 4479034008
    ... --job-id 4479034008 --date 2026-10-13 --out run_dir/trace.pdf

Reads only files the run already wrote; never calls a model and costs nothing.
'''

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))

from agentic_job_search import agent as A  # noqa: E402
from agentic_job_search import config as C  # noqa: E402
from agentic_job_search import job_trace as T  # noqa: E402

RUN_DIR = PROJECT_DIR / 'run_dir'
WRAP = 108


# --- rendering helpers -------------------------------------------------------

# Standard PDF fonts are WinAnsi, so anything outside it renders as a black box. Where the
# machine has TrueType fonts covering more (macOS ships both below), they are embedded instead, so
# 'Gdańsk' prints as written; otherwise the glyphs with a readable ASCII form are mapped and cp1252
# replaces whatever is left.
_UNICODE_FONTS = {
    'TraceSans': Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
    'TraceSansBold': Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
    'TraceMono': Path('/System/Library/Fonts/Supplemental/Courier New.ttf'),
}


def _register_unicode_fonts() -> bool:
    if not all(path.is_file() for path in _UNICODE_FONTS.values()):
        return False
    for name, path in _UNICODE_FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily('TraceSans', normal='TraceSans', bold='TraceSansBold')
    return True


UNICODE_FONTS_REGISTERED = _register_unicode_fonts()
_GLYPHS = str.maketrans({
    '→': '->', '←': '<-', '≈': '~', '⭐': '[STAR]', '📍': '[LOC]', '💰': '[PAY]',
    '🗓': '[DATE]', '✅': '[OK]', '⚠': '[WARN]', '▶': '>', '·': '.', '\ufe0f': '',
})


def _plain(text: str) -> str:
    if UNICODE_FONTS_REGISTERED:
        return text.replace('\ufe0f', '')
    return text.translate(_GLYPHS).encode('cp1252', 'replace').decode('cp1252')


def _xml(text: str) -> str:
    return _plain(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def verbatim(text: str, wrap: int = WRAP) -> Preformatted:
    """A raw block: whitespace kept, long lines folded so nothing runs off the page.

    Deliberately NOT XML-escaped: reportlab's Preformatted does not parse entities, so escaping
    would put a literal `&amp;` on the page. Raw `&`, `<` and `>` render as themselves.
    """
    folded = []
    for line in _plain(text).splitlines() or ['']:
        folded.extend(textwrap.wrap(line, wrap, subsequent_indent='    ') or [''])
    return Preformatted('\n'.join(folded), VERBATIM)


def body(text: str) -> Paragraph:
    return Paragraph(_xml(text), BODY)


def note(text: str) -> Paragraph:
    return Paragraph(_xml(text), NOTE)


def h1(text: str) -> Paragraph:
    return Paragraph(_xml(text), H1)


def h2(text: str) -> Paragraph:
    return Paragraph(_xml(text), H2)


def label(text: str) -> Paragraph:
    return Paragraph(_xml(text), LABEL)


def rule() -> HRFlowable:
    return HRFlowable(width='100%', thickness=0.4, color=colors.HexColor('#999999'),
                      spaceBefore=3, spaceAfter=5)


def kv_table(rows: list[tuple[str, str]]) -> Table:
    # The tags wrap the escaped key: escaping them too printed a literal '<b>Job</b>'.
    data = [[Paragraph(f'<b>{_xml(k)}</b>', BODY), Paragraph(_xml(v), BODY)] for k, v in rows]
    table = Table(data, colWidths=[42 * mm, None])
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
    ]))
    return table


# --- artifact loading --------------------------------------------------------

def group_or(pattern: str, text: str, default: str = '?') -> str:
    '''The first capture group of `pattern` in `text`, or `default` when it does not match.

    Spelled out rather than `(re.search(...) or [None, default])[1]`: that form works only because
    `re.Match` happens to support `[1]` as `group(1)`, so the two branches of the `or` mean
    different things and neither is visible at the call site.
    '''
    match = re.search(pattern, text, re.M)
    return match.group(1) if match else default

def find_saved_job(job_id: str) -> Path | None:
    '''The Saved Job for a job id, wherever it currently lives (saved_jobs-*/ or a released dir).'''
    for pattern in (f'saved_jobs-*/job_posting-{job_id}-*.md', f'*/job_posting-{job_id}-*.md'):
        hits = sorted(RUN_DIR.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_processed_record(job_id: str) -> Path | None:
    for pattern in (f'processed_jobs/*{job_id}*.yaml', f'*/*{job_id}*.yaml'):
        hits = sorted(RUN_DIR.glob(pattern))
        if hits:
            return hits[0]
    return None


def run_log_for(run_date: str, captured: str | None = None) -> Path | None:
    """The run log of the run that produced these artifacts.

    A date can hold several runs, and the same job id can appear in more than one of them (a
    released job is re-evaluated on the next run). Taking the newest log mixes one run's log lines
    with another run's artifacts — measured: the 12:41 run's `record_listings` line was reported
    under the 11:07 run's rating. So the choice is anchored on the artifact's own capture time:
    the newest run that STARTED at or before it.
    """
    logs = sorted((RUN_DIR / 'logs').glob(f'run-{run_date}_*.log'))
    if not logs or not captured:
        return logs[0] if logs else None
    want = captured[:19].replace('T', ' ')
    chosen = None
    for path in logs:
        stamp = path.stem.split('_', 1)[1]                  # HHMMSS from run-YYYY-MM-DD_HHMMSS
        started = f'{run_date} {stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}'
        if started <= want:
            chosen = path
    return chosen or logs[0]


def audit_log_for(run_date: str, captured: str | None = None) -> Path | None:
    """The audit log for that run: the audit is written at the run's END, so the earliest one at or
    after the artifact's capture time belongs to it."""
    audits = sorted((RUN_DIR / 'audit_logs').glob(f'audit-{run_date.replace("-", "")}*.md'))
    if not audits or not captured:
        return audits[-1] if audits else None
    want = captured[:19].replace('T', '')                   # audit-YYYYMonDD-HHMMSS
    for path in audits:
        if path.stem.split('-', 2)[-1] >= want[9:]:
            return path
    return audits[-1]


def search_that_surfaced(log_path: Path | None, job_id: str) -> str:
    '''The harvest line immediately before the record_listings call that queued this job.'''
    if not log_path or not log_path.exists():
        return ''
    lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    queued = next((i for i, ln in enumerate(lines)
                   if 'record_listings:' in ln and 'queued linkedin=' in ln and job_id in ln), None)
    if queued is None:
        return ''
    for i in range(queued - 1, -1, -1):
        if 'harvested' in lines[i]:
            return lines[i]
    return ''


LOG_LINE_HEAD_CHARS = 80   # how much of a skipped line the warning quotes


def log_lines_about(log_path: Path | None, needle: str) -> list[str]:
    if not log_path or not log_path.exists():
        return []
    return [ln.rstrip('\n') for ln in log_path.read_text(encoding='utf-8', errors='replace').splitlines()
            if needle in ln]


logger = logging.getLogger(__name__)

# (file, 1-based line number) of every cost_log line that was not valid JSON. A set, because
# stage_cost() re-reads the file once per stage and each bad line must be counted once.
SKIPPED_LOG_LINES: set[tuple[str, int]] = set()


def stage_cost(run_date: str, stage: str) -> str:
    path = RUN_DIR / 'cost_log.jsonl'
    if not path.exists():
        return 'not recorded'
    lines = path.read_text(encoding='utf-8').splitlines()
    for line_number, line in reversed(list(enumerate(lines, 1))):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as ex:
            # A diagnostic must not die on a half-written last line, nor hide that it skipped one.
            if (str(path), line_number) not in SKIPPED_LOG_LINES:
                logger.warning(f'Skipping {path} line {line_number}: not valid JSON ({ex}); '
                               f'line starts {line[:LOG_LINE_HEAD_CHARS]!r}')
            SKIPPED_LOG_LINES.add((str(path), line_number))
            continue
        if str(row.get('timestamp', '')).startswith(run_date):
            stats = (row.get('stage_stats') or {}).get(stage)
            return json.dumps(stats) if stats else 'not recorded'
    return 'no cost_log row for that date'


# --- document ----------------------------------------------------------------

# --- the Job Trace --------------------------------------------------------------

def shown(path: Path) -> str:
    '''A path relative to the project when it is inside it, else absolute (a --trace elsewhere).'''
    return str(path.relative_to(PROJECT_DIR)) if path.is_relative_to(PROJECT_DIR) else str(path)


def as_text(value: object) -> str:
    '''A trace value for a VERBATIM block: strings as they are, anything else as indented JSON.'''
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=1, ensure_ascii=False, default=str)


def cell(value: object) -> str:
    return as_text(value) if value not in (None, '', [], {}) else '—'


def _reply_meta(step: dict) -> dict:
    '''Cost, provider and finish_reason from a raw MCP reply (the extract loop records it whole).'''
    if 'raw_reply' not in step:
        return {}
    try:
        data = json.loads(step['raw_reply'])
    except (TypeError, ValueError):
        return {}
    return {'cost_usd': data.get('cost_usd'), 'provider': data.get('provider'),
            'finish_reason': data.get('finish_reason')} if isinstance(data, dict) else {}


def decision_path(steps: list[dict]) -> list:
    '''The trace boiled down to what decided the rating, each line pointing at its full step.'''
    story: list = [h1('Decision path (from the Job Trace)'), rule()]
    extract = next((s['fields'] for s in steps if s['kind'] == 'extract'), {})
    statuses = extract.get('evidence') or {}
    quotes = extract.get('quotes') or {}
    rows = [('Field', 'Value', 'Cited words', 'Verification')]
    for field in C.EXTRACT_CITED_FIELDS:
        value = extract.get(field)
        if value in (None, '', False) and field not in statuses:
            continue
        rows.append((field, cell(value), cell(quotes.get(field)), statuses.get(field, 'not cited')))
    story += [label('Extracted fields, their citations, and how each was grounded'), grid(rows)]
    rules = [s for s in steps if s['kind'] == 'hard_rule']
    if rules:
        story += [label('Hard rules, in evaluation order'), grid([('Rule', 'Result', 'Inputs')] + [
            (r['rule'], f"FIRED: {r.get('reason')}" if r['fired'] else 'passed',
             cell({k: v for k, v in r.items() if k not in ('kind', 'at', 'elapsed_s', 'rule', 'fired', 'reason')}))
            for r in rules])]
    rating = next((s for s in steps if s['kind'] == 'rating'), None)
    if rating:
        triage = rating.get('triage') or {}
        story += [label('Rating'), kv_table([
            ('Triage (local)', f"{triage.get('score')} — {triage.get('reason')}" if triage else 'not run'),
            ('Rater', f"{rating['raw_rating']}/5 by {rating['result'].get('rated_by') or C.MODEL_NAME_RATING}"),
            ('Caps', rating.get('cap_reason') or 'none'),
            ('Final', f"{rating['capped_rating']}/5"),
        ])]
    rejected = [v for s in steps if s['kind'] == 'evidence' for v in s['verdicts'] if v['status'] == 'rejected']
    story += [label(f'Hallucinations rejected for this job: {len(rejected)}')]
    if rejected:
        story += [grid([('Call', 'Field', 'Value', 'Cited')] + [
            (s['call_site'], v['field'], cell(v['value']), cell(v['quote']))
            for s in steps if s['kind'] == 'evidence' for v in s['verdicts'] if v['status'] == 'rejected'])]
    calls = [dict(s, **_reply_meta(s)) for s in steps if s['kind'] == 'llm_call']
    story += [label(f'LLM calls for this job: {len(calls)} (each in full in the last section)'), grid(
        [('#', 'Call site', 'Model / served by', 'Seconds', 'Cost', 'Outcome')] + [
            (str(n), c['call_site'], f"{c.get('model')} / {c.get('provider') or c.get('transport')}",
             cell(c.get('seconds')), f"${c['cost_usd']:.5f}" if c.get('cost_usd') is not None else '—',
             'ERROR' if c.get('error') else (c.get('finish_reason') or 'ok'))
            for n, c in enumerate(calls, 1)])]
    return story


def trace_story(steps: list[dict], trace_path: Path) -> list:
    '''Every step of the trace in order, nothing summarised.'''
    story: list = [PageBreak(), h1('The Job Trace — every step, verbatim'), rule(),
                   note(f'{shown(trace_path)}: {len(steps)} step(s). Times are wall clock; '
                        'elapsed is seconds since this job started.')]
    call_number = 0
    for step in steps:
        kind = step['kind']
        heading = f"{step.get('at', '?')} (+{step.get('elapsed_s', '?')}s) — {kind}"
        body_fields = {k: v for k, v in step.items() if k not in ('kind', 'at', 'elapsed_s')}
        if kind == 'llm_call':
            call_number += 1
            heading += f" #{call_number}: {step.get('call_site')} — {step.get('model')}"
            meta = [(k, cell(body_fields.pop(k))) for k in list(body_fields)
                    if k not in ('system', 'prompt', 'messages', 'reply', 'raw_reply', 'structured_output', 'error')]
            story += [h2(heading), kv_table(meta)]
            for part in ('system', 'prompt', 'messages', 'reply', 'raw_reply', 'structured_output', 'error'):
                if body_fields.get(part) not in (None, ''):
                    story += [label(f'VERBATIM — {part}'), verbatim(as_text(body_fields[part]))]
        elif kind == 'evidence':
            story += [h2(heading + f" — {step['call_site']}"), grid(
                [('Field', 'Value', 'Cited', 'Status', 'Passage found')] + [
                    (v['field'], cell(v['value']), cell(v['quote']), v['status'], cell(v['evidence']))
                    for v in step['verdicts']])]
        elif kind == 'page_read' and step.get('text'):
            text = body_fields.pop('text')
            story += [h2(heading), kv_table([(k, cell(v)) for k, v in body_fields.items()]),
                      label('VERBATIM — the page text the extractor read'), verbatim(text)]
        else:
            story += [h2(heading), verbatim(as_text(body_fields))]
    return story


def grid(rows: list[tuple]) -> Table:
    '''A header row plus wrapped cells; long cells wrap rather than overflow the page.'''
    data = [[Paragraph(_xml(_plain(str(c))), BODY) for c in row] for row in rows]
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#bbbbbb')),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeee')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return table


def build(job_id: str, run_date: str, out: Path, trace: Path | None = None) -> Path:
    saved = find_saved_job(job_id)
    processed = find_processed_record(job_id)
    raw_txt = RUN_DIR / 'raw_postings' / run_date / f'linkedin-{job_id}.txt'
    raw_html = RUN_DIR / 'raw_postings' / run_date / f'linkedin-{job_id}.html'
    trace_path = trace or RUN_DIR / 'raw_postings' / run_date / f'linkedin-{job_id}.trace.jsonl'
    steps = T.load(trace_path) if trace_path.exists() else []
    sent = next((s for s in steps if s['kind'] == 'notification'), None)

    # The page read stamps its own capture time; that is what anchors which run this is.
    captured = None
    if raw_txt.exists():
        m = re.search(r"^captured:\s*'?([0-9T:.\- ]+)", raw_txt.read_text(encoding='utf-8', errors='replace'), re.M)
        captured = m.group(1).strip() if m else None
    log_path = run_log_for(run_date, captured)
    audit_path = audit_log_for(run_date, captured)

    md = saved.read_text(encoding='utf-8') if saved else ''
    head = md.split('\n\n', 1)[0]
    title = re.sub(r'^#\s*', '', head.splitlines()[0]) if md else f'job {job_id}'
    company = group_or(r'^Company:\s*(.+)$', md)
    url = group_or(r'^URL:\s*(.+)$', md)
    rating = group_or(r'rating (\d)/5', title)
    reasoning = group_or(r'^Reasoning:\s*(.+)$', md, '')
    triage = group_or(r'^Triage score \(local\):\s*(\d+)', md)
    pros = re.findall(r'^• (.+)$', md.split('## Warnings')[0], re.M)
    warnings = re.findall(r'^• (.+)$', md.split('## Warnings')[1].split('\n\n')[0], re.M) \
        if '## Warnings' in md else []
    estimate_line = group_or(r'^Estimated salary:\s*(.+)$', md, '')
    # Everything after the header block, i.e. the condensed extract the rater read.
    extract_text = md.split('\n\n', 1)[1] if '\n\n' in md else ''

    story: list = []
    story += [
        Paragraph(_xml(f'Full trace — {company} — job {job_id}'), TITLE),
        Paragraph(_xml(f'Every artifact the {run_date} run kept for one evaluated job, verbatim.'),
                  SUBTITLE),
        Spacer(1, 4 * mm),
        kv_table([
            ('Job', title),
            ('Company', company),
            ('Rating', f'{rating}/5 (local triage {triage}/5)'),
            ('URL', url),
            ('Run', f'{run_date}; log {log_path.name if log_path else "?"}; '
                    f'audit {audit_path.name if audit_path else "?"}'),
            ('Saved job', str(saved.relative_to(PROJECT_DIR)) if saved else 'NOT FOUND'),
            ('Processed record', str(processed.relative_to(PROJECT_DIR)) if processed else 'NOT FOUND'),
            ('Raw page text', f'{raw_txt.relative_to(PROJECT_DIR)} '
                              f'({raw_txt.stat().st_size:,} bytes)' if raw_txt.exists() else 'NOT FOUND'),
            ('Raw DOM archive', f'{raw_html.relative_to(PROJECT_DIR)} '
                                 f'({raw_html.stat().st_size:,} bytes)' if raw_html.exists() else 'NOT FOUND'),
            ('Job Trace', f'{shown(trace_path)} ({len(steps)} steps)'
                          if steps else 'NOT FOUND (written by runs from 2026-10-19 on)'),
        ]),
        Spacer(1, 3 * mm),
        note('Sections are in pipeline order. Blocks marked VERBATIM are copied byte-for-byte from '
             'the file named above them; nothing in them is paraphrased, and any marker the '
             'pipeline itself inserted (for example a truncation notice) is left visible. '
             'The final section lists what the pipeline does NOT keep.'),
        PageBreak(),
    ]
    if steps:
        story += [*decision_path(steps), PageBreak()]

    # 1. Stage 1a
    story += [h1('1. Stage 1a — the queries this run searched with'), rule()]
    if log_path:
        q = list(log_lines_about(log_path, 'search queries via'))
        story += [label('VERBATIM — the query-generation line from the run log'), verbatim('\n'.join(q) or '(not found)')]
    story += [label('VERBATIM — the query list, as the funnel recorded it'), Spacer(1, 2)]
    story += [kv_table([('Cost', stage_cost(run_date, 'query_generation'))])]

    # 2. Stage 1b
    story += [Spacer(1, 3 * mm), h1('2. Stage 1b — the search it came from'), rule()]
    queue_line = next((ln for ln in log_lines_about(log_path, 'queued linkedin=') if job_id in ln), '')
    surfacing = search_that_surfaced(log_path, job_id)
    if surfacing:
        # '... Stage 1b: "Agentic AI Engineer" / European Union — harvested 25 listing(s)'
        query_region = surfacing.split('Stage 1b:')[-1].split('—')[0].strip()
        story += [body(f'Surfaced by the search recorded on the line below — {query_region}. '
                       f'The record_listings call that followed it queued this job.')]
        story += [label('VERBATIM — the harvest that surfaced it'), verbatim(surfacing)]
    else:
        story += [note("No harvest line precedes this job's record_listings line in the selected "
                       'run log, so the search that surfaced it cannot be named.')]
    story += [label('VERBATIM — the record_listings line that queued this job'), verbatim(queue_line or '(not found)')]
    story += [label('VERBATIM — every harvest line in this run, with region and elapsed time'),
              verbatim('\n'.join(log_lines_about(log_path, 'harvested ')) or '(none)')]

    # 3. Rater inputs
    story += [PageBreak(), h1('3. The ideal-role profile the rater was calibrated against'), rule()]
    story += [body('This is the distilled profile every rating in the run is calibrated against. It '
                   'is shown here exactly as the rater received it, including any truncation marker '
                   'the pipeline added — a marker means the model produced more than the pipeline '
                   'was willing to pass on, and the text after the cut never reached the rater.')]
    profile = ''
    cache = RUN_DIR / 'reference_summary_cache.yaml'
    if cache.exists():
        import yaml
        profile = str((yaml.safe_load(cache.read_text(encoding='utf-8')) or {}).get('summary') or '')
    if not profile:
        backups = sorted(Path('/tmp').glob('ref_cache_backup_*.yaml'))
        if backups:
            import yaml
            profile = str((yaml.safe_load(backups[-1].read_text(encoding='utf-8')) or {}).get('summary') or '')
            story += [note(f'Read from {backups[-1]} — the live cache has since been invalidated.')]
    if profile:
        marker = re.search(r'\[truncated: kept (\d+) of (\d+) chars\]', profile)
        story += [kv_table([
            ('Profile length', f'{len(profile):,} chars'),
            ('Asked for', f'{C.REFERENCE_SUMMARY_MAX_CHARS:,} chars'),
            ('Truncated', f'YES — kept {marker.group(1)} of {marker.group(2)}'
                          if marker else 'no'),
        ])]
    story += [label('VERBATIM — the profile block'), verbatim(profile or '(no cached profile found)')]

    # 4. Page read
    story += [PageBreak(), h1('4. The page as the run read it'), rule()]
    if raw_txt.exists():
        story += [label(f'VERBATIM — {raw_txt.relative_to(PROJECT_DIR)}'), verbatim(raw_txt.read_text(encoding='utf-8'))]
    else:
        story += [body('The raw posting text was not retained (RAW_POSTINGS_RETENTION_DAYS).')]
    if raw_html.exists():
        html = raw_html.read_text(encoding='utf-8', errors='replace')
        story += [Spacer(1, 3 * mm), h2(f'The DOM archive ({len(html):,} chars, first 2,000 shown)'), rule()]
        story += [verbatim(html[:2000])]

    # 5. Extraction
    story += [PageBreak(), h1('5. The extraction'), rule()]
    story += [label('VERBATIM — the extractor instructions (PAGE_EXTRACTOR_INSTRUCTIONS)'),
              verbatim(A.PAGE_EXTRACTOR_INSTRUCTIONS)]
    story += [label('VERBATIM — the extract signal line from the run log'),
              verbatim('\n'.join(log_lines_about(log_path, f'Extract signal: job {job_id}')) or '(not found)')]
    story += [label('VERBATIM — the condensed extract the rater read (from the Saved Job)'),
              verbatim(extract_text or '(none)')]

    # 6. Salary
    story += [PageBreak(), h1('6. Salary: what the posting said, and what the pipeline added'), rule()]
    story += [label('VERBATIM — the Saved Job salary lines'), verbatim(estimate_line or '(none)')]
    # Since 2026-10-14 the pay note rides on the job's Rating line (`; pay: …`); older logs carry
    # a `Pay display:` line of its own.
    pay_lines = (log_lines_about(log_path, f'Pay display: job {job_id}')
                 or [ln for ln in log_lines_about(log_path, f'job {job_id}:') if 'Rating [' in ln and '; pay: ' in ln])
    story += [label('VERBATIM — the pay-display log line'),
              verbatim('\n'.join(pay_lines) or '(not found)')]
    story += [note('A model estimate is shown on the notification labelled as such, and reaches no '
                   'rating, cap or warning — it is computed after the rating and kept out of the '
                   'extract text the rater reads.')]

    # 7. Rating
    rating_lines = [ln for ln in log_lines_about(log_path, f'job {job_id}:') if 'Rating [' in ln]
    story += [PageBreak(), h1('7. The rating'), rule()]
    if not rating_lines:
        story += [note(f'No Rating line for job {job_id} in {log_path.name if log_path else "?"} — '
                       f'the artifacts above and this log may be from different runs.')]
    story += [label('VERBATIM — the Rating line from the run log'),
              verbatim('\n'.join(rating_lines) or '(not found)')]
    story += [label('VERBATIM — reasoning'), verbatim(reasoning or '(none)')]
    story += [label('VERBATIM — pros'), verbatim('\n'.join(f'* {p}' for p in pros) or '(none)')]
    story += [label('VERBATIM — warnings'), verbatim('\n'.join(f'* {w}' for w in warnings) or '(none)')]

    # 8. Notification
    story += [PageBreak(), h1('8. The Telegram message'), rule()]
    if sent is not None:
        story += [body('Exactly as sent, from the Job Trace.' if sent.get('text') else
                       f"No message was sent: {sent.get('why')}.")]
        if sent.get('text'):
            story += [kv_table([('Delivered', 'yes' if sent.get('sent') else 'NO — the send failed')]),
                      verbatim(sent['text'])]
    else:
        story += [body("Rebuilt by calling the pipeline's own format_job_notification() with the "
                       'fields the Saved Job and the run log retained. The message the user actually '
                       'received was not stored.')]
    if sent is None:
        try:
            # No salary_estimate: the Saved Job keeps only its display line, and rebuilding one from
            # it would put figures in this document that no file states. Until 2026-10-19 a
            # hard-coded EUR 75,000-115,000 estimate stood in for every job here.
            extract = {
                'title': re.search(r'^Title:\s*(.+)$', md, re.M).group(1),
                'company': company,
                'location': group_or(r'^Location:\s*(.+)$', md, ''),
                'workplace_type': group_or(r'^Workplace:\s*(.+)$', md, ''),
                'date_posted': group_or(r'^Posted:\s*(.+)$', md, ''),
                'salary': '',
            }
            candidate = {'company': company, 'title': extract['title'], 'url': url, 'job_id': job_id}
            story += [verbatim(A.format_job_notification(candidate, extract, int(rating), pros, warnings)),
                      note('The salary line above is not rebuilt; the Saved Job line in section 6 is what was shown.')]
        except Exception as ex:  # a report must not die on one missing field
            story += [body(f'Could not rebuild the message from the retained fields: {type(ex).__name__}: {ex}')]

    # 9. Cost
    story += [Spacer(1, 3 * mm), h1('9. Cost of this job'), rule()]
    story += [kv_table([
        ('Extraction stage (whole run)', stage_cost(run_date, 'extraction')),
        ('Rating stage (whole run)', stage_cost(run_date, 'rating')),
        ('Salary estimate stage (whole run)', stage_cost(run_date, 'salary_estimate')),
    ])]
    story += [note('Per-job costs are not recorded; these are the run-wide stage totals, which the '
                   'Rating and Extract lines above break down for this job.')]

    # 10. What is not retained
    story += [PageBreak(), h1('10. What the pipeline does NOT keep'), rule()]
    story += [body('Stated explicitly, because the absence is what makes a summary look complete. '
                   'Each of these was in memory during the run and is gone:')]
    not_kept_without_trace = [
        '* The search-results CARD text for this listing (title/company/location as shown in the',
        '  results list). The harvest is held in ScrapeSession._harvest and handed straight to',
        "  do_record_listings; only the counts and the audit log's per-search lines survive.",
        '* The harvest listing dicts for the whole run (_listing_records is in-memory only).',
        '* The full extract JSON — only the fields the Saved Job renders are persisted, so any',
        '  extract field with no line in the Saved Job is unrecoverable.',
        '* The rating JSON. The Saved Job keeps reasoning/pros/warnings, but not the raw object.',
        '* The exact Telegram bytes that were sent.',
        "* The rater's full prompt as sent (it is rebuilt deterministically from the profile above,",
        '  build_evaluator_instructions() and the extract text).',
        '* The search URL / filter state for the search that surfaced this job: the audit log keeps',
        '  the query, the region, the pin that was verified and the card count, not the URL.',
    ]
    not_kept_with_trace = [
        '* The search URL / filter state for the search that surfaced this job: the audit log keeps',
        '  the query, the region, the pin that was verified and the card count, not the URL.',
        '* The browser tool calls of Stage 1b (the scraper session), which happen before this job',
        '  exists as a candidate; the trace starts at the listing record that queued it.',
    ]
    story += [verbatim('\n'.join(not_kept_with_trace if steps else not_kept_without_trace))]
    if steps:
        story += trace_story(steps, trace_path)
    story += [Spacer(1, 2 * mm),
              note('Everything else in this document is verbatim from a file on disk.'),
              Spacer(1, 3 * mm),
              h2('Provenance of this document'),
              verbatim('\n'.join([
                  f'generated : {date.today().isoformat()}',
                  f'job id    : {job_id}',
                  f'run date  : {run_date}',
                  f'run log   : {log_path}',
                  f'audit log : {audit_path}',
                  'generator : scripts/job_trace_report.py',
              ]))]

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=16 * mm, rightMargin=14 * mm, topMargin=15 * mm, bottomMargin=14 * mm,
        title=f'Full trace — {company} — job {job_id}',
        author='agent_job_search',
    )
    doc.build(story)
    return out


# --- styles ------------------------------------------------------------------

_sheet = getSampleStyleSheet()
_SANS = 'TraceSans' if UNICODE_FONTS_REGISTERED else 'Helvetica'
_SANS_BOLD = 'TraceSansBold' if UNICODE_FONTS_REGISTERED else 'Helvetica-Bold'
TITLE = ParagraphStyle('T', parent=_sheet['Title'], fontName=_SANS_BOLD, fontSize=17, leading=20, spaceAfter=2,
                       textColor=colors.HexColor('#111111'))
SUBTITLE = ParagraphStyle('S', parent=_sheet['Normal'], fontName=_SANS, fontSize=9.5, leading=12,
                          textColor=colors.HexColor('#555555'))
H1 = ParagraphStyle('H1', parent=_sheet['Heading1'], fontName=_SANS_BOLD, fontSize=12.5, leading=15, spaceBefore=2,
                    spaceAfter=1, textColor=colors.HexColor('#0b3d6b'))
H2 = ParagraphStyle('H2', parent=_sheet['Heading2'], fontName=_SANS_BOLD, fontSize=10.5, leading=13, spaceBefore=4,
                    spaceAfter=1, textColor=colors.HexColor('#333333'))
BODY = ParagraphStyle('B', parent=_sheet['Normal'], fontName=_SANS, fontSize=8.8, leading=11.4)
NOTE = ParagraphStyle('N', parent=BODY, fontSize=8.2, leading=10.6,
                      textColor=colors.HexColor('#666666'), leftIndent=4)
LABEL = ParagraphStyle('L', parent=BODY, fontSize=8.2, leading=10.4, spaceBefore=4, spaceAfter=1,
                       textColor=colors.HexColor('#7a4a00'))
VERBATIM = ParagraphStyle('V', parent=_sheet['Code'], fontName='TraceMono' if UNICODE_FONTS_REGISTERED else 'Courier',
                          fontSize=7.0, leading=8.6,
                          backColor=colors.HexColor('#f6f6f4'), borderPadding=3, leftIndent=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--date', default=date.today().isoformat(),
                        help='run date, YYYY-MM-DD (default today)')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--trace', type=Path, default=None,
                        help='Job Trace file (default: raw_postings/{date}/linkedin-{job-id}.trace.jsonl)')
    args = parser.parse_args()

    out = args.out or RUN_DIR / f'job-trace-{args.job_id}-{args.date}.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'wrote {build(args.job_id, args.date, out, args.trace)}')
    if SKIPPED_LOG_LINES:
        logger.warning(f'{len(SKIPPED_LOG_LINES)} cost_log line(s) were skipped as invalid JSON, so the cost '
                       f'figures above may be missing a row: {sorted(SKIPPED_LOG_LINES)}')


if __name__ == '__main__':
    main()
