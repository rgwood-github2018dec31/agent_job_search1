'''cache_io: the fail-open YAML-mapping cache. Every test uses tmp_path, never the real run_dir.'''
import logging
from pathlib import Path

from agentic_job_search import cache_io

NOUN = 'test cache'


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_missing_file_gives_empty_mapping_without_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        assert cache_io.load(tmp_path / 'absent.yaml', NOUN) == {}
    assert _warnings(caplog) == []


def test_empty_file_gives_empty_mapping_without_a_warning(tmp_path, caplog):
    path = tmp_path / 'empty.yaml'
    path.write_text('', encoding='utf-8')
    with caplog.at_level(logging.DEBUG):
        assert cache_io.load(path, NOUN) == {}
    assert _warnings(caplog) == []


def test_unreadable_file_warns_and_is_rebuilt_empty(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'cache.yaml'
    path.write_text('a: 1\n', encoding='utf-8')

    def boom(self, *args, **kwargs):
        raise PermissionError('read denied')

    monkeypatch.setattr(Path, 'read_text', boom)
    with caplog.at_level(logging.WARNING):
        assert cache_io.load(path, NOUN) == {}
    [message] = _warnings(caplog)
    assert 'PermissionError' in message and 'read denied' in message and str(path) in message


def test_invalid_yaml_warns_and_is_rebuilt_empty(tmp_path, caplog):
    path = tmp_path / 'cache.yaml'
    path.write_text('a: [unclosed\n', encoding='utf-8')
    with caplog.at_level(logging.WARNING):
        assert cache_io.load(path, NOUN) == {}
    assert len(_warnings(caplog)) == 1


def test_non_mapping_yaml_warns_and_is_rebuilt_empty(tmp_path, caplog):
    path = tmp_path / 'cache.yaml'
    path.write_text('- one\n- two\n', encoding='utf-8')
    with caplog.at_level(logging.WARNING):
        assert cache_io.load(path, NOUN) == {}
    [message] = _warnings(caplog)
    assert 'not a mapping' in message and 'list' in message


def test_save_into_an_unwritable_target_logs_and_does_not_raise(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'sub' / 'cache.yaml'

    def boom(self, *args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(Path, 'write_text', boom)
    with caplog.at_level(logging.WARNING):
        cache_io.save(path, {'a': 1}, NOUN)   # must not raise
    [message] = _warnings(caplog)
    assert 'Could not write' in message and 'OSError' in message and 'disk full' in message and str(path) in message
    assert not path.exists()


def test_save_with_an_unmakeable_directory_logs_and_does_not_raise(tmp_path, monkeypatch, caplog):
    def boom(self, *args, **kwargs):
        raise PermissionError('mkdir denied')

    monkeypatch.setattr(Path, 'mkdir', boom)
    with caplog.at_level(logging.WARNING):
        cache_io.save(tmp_path / 'new' / 'cache.yaml', {'a': 1}, NOUN)
    [message] = _warnings(caplog)
    assert 'PermissionError' in message and 'mkdir denied' in message


def test_round_trip_returns_the_same_data_and_creates_parents(tmp_path, caplog):
    path = tmp_path / 'deep' / 'er' / 'cache.yaml'
    data = {'Málaga': {'country': 'ES', 'langs': ['es', 'en']}, 'n': 3}
    with caplog.at_level(logging.DEBUG):
        cache_io.save(path, data, NOUN)
        assert cache_io.load(path, NOUN) == data
    assert _warnings(caplog) == []


def test_a_valid_file_is_not_hidden_behind_the_empty_fallback(tmp_path):
    '''Guards against load() returning {} unconditionally: a valid file must come back non-empty.'''
    path = tmp_path / 'cache.yaml'
    path.write_text('k: v\n', encoding='utf-8')
    assert cache_io.load(path, NOUN) == {'k': 'v'}
