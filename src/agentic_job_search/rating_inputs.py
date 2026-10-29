'''Record Rating Inputs: every input of the rating call, saved where a Job Trace can link to it.

The rater's system message is JOB_REQUIREMENTS.md, the ideal-role profile and the evaluator
instructions; its user message is the job posting. The first three are the same for every job in
a run, but JOB_REQUIREMENTS.md changes in interactive mode and the profile cache is overwritten
whenever the applied-job corpus changes, so a trace that only named them could not be checked a
week later. Each part is saved once, cut from the exact system message the rater received, under
`run_dir/rating_inputs/` and named by its md5, so an unchanged input is written once across runs.
'''

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

from agentic_job_search.cache_io import load as load_yaml_mapping

logger = logging.getLogger(__name__)

# The markers build_profile_block and _wrap_reference_summary put around each part of the system
# message. A profile that could not be summarized is the full reference block instead.
_PARTS = (
    ('job_requirements', '--- JOB_REQUIREMENTS.md ---', '--- END JOB_REQUIREMENTS.md ---'),
    ('ideal_role_profile', '--- IDEAL ROLE PROFILE', '--- END IDEAL ROLE PROFILE ---'),
    ('ideal_role_profile', '--- REFERENCE JOBS', '--- END REFERENCE JOBS ---'),
)
# Hex digits of the md5 kept in a saved file's name; enough to tell versions apart.
_NAME_HASH_CHARS = 12

_saved: dict[str, dict[str, Any]] = {}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _save(directory: Path, kind: str, text: str, header: str = '') -> dict[str, Any]:
    '''Write `text` (after an optional provenance header) as `{kind}-{md5}.md` unless it is there already.'''
    digest = _md5(text)
    path = directory / f'{kind}-{digest[:_NAME_HASH_CHARS]}.md'
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{header}{text}', encoding='utf-8')
    return {'path': str(path), 'chars': len(text), 'md5': digest}


def split_system_message(system_message: str) -> dict[str, str]:
    '''The rater's system message cut into its parts, each exactly as sent; the rest is the instructions.'''
    parts: dict[str, str] = {}
    rest = system_message
    for kind, start, end in _PARTS:
        begin = rest.find(start)
        if kind in parts or begin < 0:
            continue
        stop = rest.find(end, begin)
        stop = len(rest) if stop < 0 else stop + len(end)
        parts[kind] = rest[begin:stop]
        rest = rest[:begin] + rest[stop:]
    parts['evaluator_instructions'] = rest.strip()
    return parts


def _profile_provenance(profile: str, cache_path: Path) -> str:
    '''How the profile was made, from the reference summary cache when it holds this profile.'''
    cache = load_yaml_mapping(cache_path, 'reference summary cache')
    summary = str(cache.get('summary') or '')
    if not profile.startswith('--- IDEAL ROLE PROFILE'):
        facts = {'made_by': 'nobody: the summary failed, so the rater got the full reference jobs'}
    elif summary and summary in profile:
        facts = {
            'made_by': 'agent.build_reference_summary, from the applied jobs listed under sources',
            'provider': cache.get('provider') or '?',
            'model': cache.get('model') or 'not recorded (generated before 2026-10-29)',
            'generated_on': cache.get('generated_on') or 'not recorded (generated before 2026-10-29)',
            'asked_chars': cache.get('asked_chars') or '?',
            'returned_chars': len(summary),
            'corpus_md5': cache.get('hash') or '?',
            'cache': str(cache_path),
            'sources': cache.get('sources') or 'not recorded (generated before 2026-10-29)',
        }
    else:
        facts = {'made_by': f'unknown: {cache_path} no longer holds this profile'}
    return '---\n' + yaml.safe_dump(facts, sort_keys=False, allow_unicode=True) + '---\n\n'


def save_run_inputs(system_message: str, directory: Path, cache_path: Path) -> dict[str, dict[str, Any]]:
    '''Save each part of the rater's system message once per run; returns kind -> {path, chars, md5}.'''
    key = _md5(system_message)
    if key in _saved:
        return _saved[key]
    saved: dict[str, dict[str, Any]] = {}
    for kind, text in split_system_message(system_message).items():
        header = _profile_provenance(text, cache_path) if kind == 'ideal_role_profile' else ''
        saved[kind] = _save(directory, kind, text, header)
    saved['system_message'] = _save(directory, 'rater_system_message', system_message)
    logger.info('Rating inputs saved: ' + ', '.join(f"{kind} {Path(v['path']).name}" for kind, v in saved.items()))
    _saved[key] = saved
    return saved
