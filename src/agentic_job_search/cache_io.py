'''
Read and write the fail-open YAML-mapping caches.

Four modules keep a run-scoped dict on disk and all four answer the same way: a cache that cannot
be read is rebuilt as empty, and a cache that cannot be written costs a rebuild next run, never
correctness. Only the path and the noun in the message differed.

    location.py          run_dir/location_cache.yaml
    salary.py            run_dir/salary_cache.yaml
    salary_estimate.py   run_dir/salary_estimate_cache.yaml
    fx.py                run_dir/fx_rates_cache.yaml

Each keeps its own module-level dict and path constant (tests patch both), so those modules keep a
thin wrapper around these two functions; the error handling lives here once.

**Caches that must not fail open deliberately do not call this module.** A file whose contents
drive real work raises instead of rebuilding empty: `linkedin_page._load_asset_index` guards files
on disk and raises RuntimeError, and `preferences.load_preferences` / `location_review._load` raise
on a bad file. Switching one of those to `cache_io.load` would turn a hard failure into a silent
empty answer, which is the bug this module exists to avoid.
'''
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load(path: Path, noun: str) -> dict[str, Any]:
    '''Read the YAML mapping at `path`, rebuilding as empty when it is missing or unusable.

    A missing file, a read error, an empty file and a file holding something other than a mapping
    all give `{}`. A non-mapping warns, because that means the file exists but is not what this
    cache writes — worth seeing, cheap to discard.
    '''
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
    except Exception as ex:
        logger.warning(
            f'Could not read {noun} {path}: {type(ex).__name__}: {ex} — rebuilding it.'
        )
        return {}
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        logger.warning(
            f'{noun.capitalize()} {path} is not a mapping ({type(loaded).__name__}) — rebuilding it.'
        )
        return {}
    return loaded


def save(path: Path, data: dict[str, Any], noun: str) -> None:
    '''Persist `data` as YAML at `path`. Any failure warns and continues.'''
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding='utf-8'
        )
    except Exception as ex:
        logger.warning(
            f'Could not write {noun} {path}: {type(ex).__name__}: {ex} — continuing.'
        )
