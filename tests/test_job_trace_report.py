'''scripts/job_trace_report.py: a truncated cost_log line is reported, not fatal and not silent.

The script imports reportlab at module level and it is not a project dependency (the script is
run with `uv run --with reportlab`), so a stub stands in for it here; only stage_cost is under test.
'''
import importlib.util
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'job_trace_report.py'
RUN_DATE = '2026-10-25'


@pytest.fixture
def report(monkeypatch):
    for name in ('reportlab', 'reportlab.lib', 'reportlab.lib.colors', 'reportlab.lib.pagesizes',
                 'reportlab.lib.styles', 'reportlab.lib.units', 'reportlab.pdfbase',
                 'reportlab.pdfbase.pdfmetrics', 'reportlab.pdfbase.ttfonts', 'reportlab.platypus'):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    spec = importlib.util.spec_from_file_location('job_trace_report_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SKIPPED_LOG_LINES.clear()
    return module


def test_truncated_final_line_is_skipped_counted_and_logged(report, tmp_path, monkeypatch, caplog):
    good = json.dumps({'timestamp': f'{RUN_DATE}T10:00:00', 'stage_stats': {'rating': {'cost': 0.5}}})
    truncated = '{"timestamp": "2026-10-25T11:00:00", "stage_stats": {"rat'
    (tmp_path / 'cost_log.jsonl').write_text(f'{good}\n{truncated}\n', encoding='utf-8')
    monkeypatch.setattr(report, 'RUN_DIR', tmp_path)

    with caplog.at_level(logging.WARNING):
        first = report.stage_cost(RUN_DATE, 'rating')
        second = report.stage_cost(RUN_DATE, 'extraction')   # re-reads the file: must not double count

    assert json.loads(first) == {'cost': 0.5}      # fell back to the earlier valid row
    assert second == 'not recorded'
    assert len(report.SKIPPED_LOG_LINES) == 1
    assert (str(tmp_path / 'cost_log.jsonl'), 2) in report.SKIPPED_LOG_LINES
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and 'line 2' in warned[0] and 'cost_log.jsonl' in warned[0]


def test_a_clean_log_skips_nothing(report, tmp_path, monkeypatch):
    row = json.dumps({'timestamp': RUN_DATE, 'stage_stats': {'rating': {'cost': 1}}})
    (tmp_path / 'cost_log.jsonl').write_text(row + '\n', encoding='utf-8')
    monkeypatch.setattr(report, 'RUN_DIR', tmp_path)
    assert json.loads(report.stage_cost(RUN_DATE, 'rating')) == {'cost': 1}
    assert not report.SKIPPED_LOG_LINES
