'''Compare local and remote models on Decide Company Overlap (2026-11-19), for choosing
MODEL_NAME_COMPANY_OVERLAP.

The pairs: every company of a processed job (run_dir/processed_jobs/) whose name overlaps a name of an
Applied Organization without equalling it, plus hand-made pairs. Each is labelled by hand in
run_dir/company_overlap_labels.yaml (LABELS_PATH): `same`, `different` or `excluded`. A real pair in none of them is
unlabelled: it is still asked, listed in the report to be labelled, and left out of the scores. Each model gets the production
prompt (company_overlap.build_prompt): remote models by forced function call through the OpenRouter
tool server, local models by JSON schema through the local LLM tool server. A model "skips" a pair
when it answers same at ORGANIZATION_MATCH_HIGH_CONFIDENCE or above, as Stage 1 would.

Writes run_dir/reports/company-overlap-models-<date>-<time>.md (the table) and .jsonl (every answer).
Costs money for the remote models (see the report's cost column); local models are free.

    uv run python scripts/compare_company_overlap_models.py [--local m1,m2] [--remote m1,m2] [--repeat N]
'''

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime

import yaml
from agentic_job_search import applied_organizations, company_overlap
from agentic_job_search.config import ORGANIZATION_MATCH_HIGH_CONFIDENCE
from agentic_job_search.preferences import RUN_DIR
from agentic_job_search.triage import chat_openrouter_tool, extract_json_object, generate_local

logger = logging.getLogger('compare_company_overlap_models')

LOCAL_MODELS_DEFAULT = ('granite4.1:3b', 'qwen3.5:9b-q8_0', 'gpt-oss:20b', 'qwen3.6:35b-mlx')
# anthropic/claude-opus-5.5 is left out: it refuses a forced tool_choice (HTTP 400), so it scored 0/88 on 2026-11-20.
REMOTE_MODELS_DEFAULT = ('deepseek/deepseek-v4.1-flash-1017', 'z-ai/glm-5.3-flash', 'z-ai/glm-5.3')
LOCAL_PURPOSE_SLUG = 'company-overlap-model-compare'
PERCENTILE_90 = 0.9
REPORT_STAMP_FORMAT = '%Y-%m-%d-%H%M'   # one report per run, never overwriting an earlier one
LINE_BUFFERED = 1                      # every answer reaches the .jsonl at once, kept if the run is stopped

# Hand labels (personal: they name applied companies, so they live in run_dir/, not here). `same`, `different`
# and `excluded` hold real pairs [card company, applied organization]; a real pair in none is unlabelled.
# `made_pairs` are hand-made pairs of well-known names of both kinds, so the set is not only the real pairs,
# most of which are easy "different"s.
LABELS_PATH = RUN_DIR / 'company_overlap_labels.yaml'


@dataclass(frozen=True)
class Labels:
    same: frozenset
    different: frozenset
    excluded: frozenset
    made_pairs: tuple

    def of(self, pair: tuple[str, str]) -> bool | None:
        '''True (same), False (different) or None (unlabelled) for a real pair.'''
        return True if pair in self.same else False if pair in self.different else None


def load_labels(path=LABELS_PATH) -> Labels:
    '''The hand labels; raises naming the file when it is missing or bad, or a pair is labelled twice.'''
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        lists = {name: frozenset(tuple(pair) for pair in raw.get(name) or []) for name in ('same', 'different', 'excluded')}
        made = tuple((p['company'], p['title'], p['applied'], bool(p['same'])) for p in raw.get('made_pairs') or [])
    except Exception as ex:
        raise RuntimeError(f'Could not read the hand labels {path}: {type(ex).__name__}: {ex}') from ex
    for first, second in (('same', 'different'), ('same', 'excluded'), ('different', 'excluded')):
        if both := lists[first] & lists[second]:
            raise RuntimeError(f'The hand labels {path} put {sorted(both)} in both {first!r} and {second!r}')
    return Labels(lists['same'], lists['different'], lists['excluded'], made)


def real_pairs(records: dict, labels: Labels) -> list[tuple[str, str, str, bool | None]]:
    '''(card company, title, applied organization, same? or None when unlabelled) for every processed job's
    company that overlaps a listed name, the first title seen for it.'''
    index = company_overlap.overlap_index(records)
    exact = {applied_organizations.fold_exact(n) for r in records.values() for n in r.names}
    titles: dict[str, str] = {}
    for path in sorted((RUN_DIR / 'processed_jobs').glob('*.yaml')):
        try:
            job = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        except Exception as ex:
            logger.warning(f'Could not read {path}, skipping it: {type(ex).__name__}: {ex}')
            continue
        if company := str(job.get('company') or ''):
            titles.setdefault(company, str(job.get('description') or ''))
    pairs = []
    for company, title in sorted(titles.items()):
        if applied_organizations.fold_exact(company) in exact:
            continue
        for key in company_overlap.overlapping_records(company, index):
            if (company, key) not in labels.excluded:
                pairs.append((company, title, key, labels.of((company, key))))
    return pairs


async def ask(model: str, local: bool, prompt: str, record) -> dict:
    started = time.monotonic()
    cost = 0.0
    try:
        if local:
            text = await generate_local(prompt, model=model, purpose_slug=LOCAL_PURPOSE_SLUG,
                                        output_format=company_overlap.PARAMETERS)
            arguments = extract_json_object(text)
        else:
            arguments, reply = await chat_openrouter_tool(prompt, company_overlap.TOOL_NAME,
                                                          company_overlap.TOOL_DESCRIPTION,
                                                          company_overlap.PARAMETERS, model=model)
            cost = reply.cost_usd
        verdict = company_overlap.verdict_from_arguments(arguments, record, model)
        return {'verdict': verdict.verdict, 'confidence': verdict.confidence, 'skips': verdict.skips,
                'reasoning': verdict.reasoning, 'seconds': round(time.monotonic() - started, 2), 'cost': cost}
    except Exception as ex:
        logger.warning(f'{model}: unusable answer for {prompt[:80]!r}…: {type(ex).__name__}: {ex}')
        return {'error': f'{type(ex).__name__}: {ex}', 'seconds': round(time.monotonic() - started, 2), 'cost': cost}


def table(results: dict[str, list[dict]]) -> str:
    '''Scores over the labelled pairs only; time and cost over every answer.'''
    rows = ['| Model | Where | Correct | False skips | Misses | Unusable | Unclear | Median s | p90 s | Cost $ |',
            '|---|---|---|---|---|---|---|---|---|---|']
    for model, every_answer in results.items():
        answers = [a for a in every_answer if a['same'] is not None]
        usable = [a for a in answers if 'error' not in a]
        correct = sum(1 for a in usable if a['skips'] == a['same'])
        false_skips = sum(1 for a in usable if a['skips'] and not a['same'])
        misses = sum(1 for a in usable if a['same'] and not a['skips'])
        unclear = sum(1 for a in usable if a['verdict'] == company_overlap.VERDICT_UNCLEAR)
        seconds = sorted(a['seconds'] for a in every_answer)
        p90 = seconds[min(len(seconds) - 1, int(len(seconds) * PERCENTILE_90))] if seconds else 0
        rows.append(f"| {model} | {every_answer[0]['where'] if every_answer else ''} | {correct}/{len(answers)} | {false_skips} | "
                    f'{misses} | {len(answers) - len(usable)} | {unclear} | '
                    f'{statistics.median(seconds) if seconds else 0:.1f} | {p90:.1f} | '
                    f'{sum(a["cost"] for a in every_answer):.4f} |')
    return '\n'.join(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--local', default=','.join(LOCAL_MODELS_DEFAULT))
    parser.add_argument('--remote', default=','.join(REMOTE_MODELS_DEFAULT))
    parser.add_argument('--repeat', type=int, default=1, help='ask each pair this many times (run-to-run noise)')
    args = parser.parse_args()

    raw = applied_organizations.load_records_file()
    records = {key: applied_organizations.record_from_raw(key, entry) for key, entry in raw.items()}
    labels = load_labels()
    pairs = real_pairs(records, labels) + [p for p in labels.made_pairs if p[2] in records]
    missing = [p for p in labels.made_pairs if p[2] not in records]
    if missing:
        logger.warning(f'Hand-made pairs left out, their organization has no record: {missing}')
    unlabelled = [p for p in pairs if p[3] is None]
    if unlabelled:
        logger.warning(f'{len(unlabelled)} real pair(s) are in no list of {LABELS_PATH}; they are asked and listed in '
                       f'the report, not scored: {[(p[0], p[2]) for p in unlabelled]}')
    index = company_overlap.overlap_index(records)
    logger.info(f'{len(pairs)} pairs ({sum(p[3] is True for p in pairs)} labelled same, {len(unlabelled)} unlabelled)')

    models = [(m, True) for m in args.local.split(',') if m] + [(m, False) for m in args.remote.split(',') if m]
    results: dict[str, list[dict]] = {}
    out = RUN_DIR / 'reports' / f'company-overlap-models-{datetime.now().strftime(REPORT_STAMP_FORMAT)}'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.jsonl').open('w', encoding='utf-8', buffering=LINE_BUFFERED) as log_file:
        for model, local in models:
            answers = []
            for company, title, key, same in pairs:
                overlapping = company_overlap.overlapping_records(company, index).get(key, [])
                prompt = company_overlap.build_prompt(company, title, records[key], overlapping)
                for _ in range(args.repeat):
                    answer = {'model': model, 'where': 'local' if local else 'remote', 'company': company,
                              'title': title, 'applied': key, 'same': same,
                              **await ask(model, local, prompt, records[key])}
                    answers.append(answer)
                    log_file.write(json.dumps(answer, ensure_ascii=False) + '\n')
            results[model] = answers
            logger.info(f'{model}: done\n{table({model: answers})}')

    wrong = []
    for model, answers in results.items():
        for a in answers:
            if a['same'] is not None and ('error' in a or a['skips'] != a['same']):
                wrong.append(f"- {model}: {a['company']!r} ~ {a['applied']!r} (labelled {'same' if a['same'] else 'different'}): "
                             f"{a.get('verdict', 'unusable')} {a.get('confidence', '')} — {a.get('reasoning') or a.get('error')}")
    unlabelled_lines = []
    for company, _title, key, _same in unlabelled:
        answers = [f"{model}: {a.get('verdict', 'unusable')} {a.get('confidence', '')}"
                   for model, model_answers in results.items() for a in model_answers
                   if a['company'] == company and a['applied'] == key]
        unlabelled_lines.append(f'- {company!r} ~ {key!r}: ' + '; '.join(answers))
    report = (f'# Decide Company Overlap: models compared, {date.today().isoformat()}\n\n'
              f'{len(pairs) - len(unlabelled)} labelled pairs, {sum(p[3] is True for p in pairs)} labelled same; '
              f'skip = same at {ORGANIZATION_MATCH_HIGH_CONFIDENCE}+ (ORGANIZATION_MATCH_HIGH_CONFIDENCE). '
              f'Repeat: {args.repeat}.\n\n'
              f'{table(results)}\n\n## Every wrong or unusable answer\n\n' + '\n'.join(wrong) + '\n'
              + (f'\n## Unlabelled pairs, not scored (add each to `same` or `different` in {LABELS_PATH.name})\n\n'
                 + '\n'.join(unlabelled_lines) + '\n' if unlabelled else ''))
    out.with_suffix('.md').write_text(report, encoding='utf-8')
    logger.info(f"Report: {out.with_suffix('.md')}\n{report}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname).1s %(name)s: %(message)s',
                        stream=sys.stderr)
    asyncio.run(main())
