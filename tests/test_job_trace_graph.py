'''scripts/job_trace_graph.py: one unreadable trace is skipped and logged, not fatal to the date's graph.

The script imports reportlab at module level and it is not a project dependency, so stubs stand in
for it here; only the loading of traces is under test.
'''
import importlib.util
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'job_trace_graph.py'


@pytest.fixture
def graph(monkeypatch):
    for name in ('reportlab', 'reportlab.lib', 'reportlab.lib.pagesizes', 'reportlab.lib.styles',
                 'reportlab.platypus', 'job_path_drawing', 'job_trace_report'):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    spec = importlib.util.spec_from_file_location('job_trace_graph_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_unreadable_trace_is_skipped_and_named(graph, tmp_path, monkeypatch, caplog):
    folder = tmp_path / '2026-11-12'
    folder.mkdir()
    (folder / 'linkedin-1.trace.jsonl').write_text(json.dumps({'kind': 'origin'}) + '\n', encoding='utf-8')
    (folder / 'linkedin-2.trace.jsonl').write_text('{"kind": "orig', encoding='utf-8')   # a run still writing it
    monkeypatch.setattr(graph, 'RAW_POSTINGS_DIR', tmp_path)
    with caplog.at_level(logging.WARNING):
        traces = graph.load_traces('2026-11-12')
    assert list(traces) == ['linkedin-1']
    assert any('linkedin-2.trace.jsonl' in r.getMessage() for r in caplog.records)


def test_a_date_with_no_readable_trace_stops_with_the_reason(graph, tmp_path, monkeypatch):
    folder = tmp_path / '2026-11-12'
    folder.mkdir()
    (folder / 'linkedin-2.trace.jsonl').write_text('{"kind": "orig', encoding='utf-8')
    monkeypatch.setattr(graph, 'RAW_POSTINGS_DIR', tmp_path)
    with pytest.raises(SystemExit, match='No readable Job Trace'):
        graph.load_traces('2026-11-12')
