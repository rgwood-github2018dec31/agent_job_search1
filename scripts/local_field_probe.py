'''Ask small local models for one extract field at a time, and compare with the page extractor.

Why this is a tracked script
----------------------------
The page extractor (MODEL_NAME_EXTRACTOR, a reasoning model) answers differently on the same page:
over 243 retained pages run twice at EXTRACT_TEMPERATURE (2026-10-30), `relocation`,
`residency_scope` and `language_requirement` were present in one run and absent in the other on
dozens of pages, and those fields feed the gates. A small local model without thinking, at
temperature 0, decodes greedily and should give one answer per page. This measures whether it
does, and where it disagrees with the extractor, so a person can judge which one read the page right.

Per page and model, each field in PROBE_FIELDS is asked on its own: the page's Markdown first (so
Ollama reuses it for the next field), then the field's definition from the extractor's own schema
(EXTRACT_PROPERTIES), answered as {value, quote}, decoding constrained to that schema. A value
whose quote regex cannot find on the page (evidence.find_quote) is rejected and counts as empty,
as Verify Source Reference would. Every page is asked `--repeat` times.

Reported per model and field: pages whose runs disagree (the extractor's own figure beside it,
from the `--reference` replay), agreement with the extractor, rejected citations, and every page
where the extractor's runs agree with each other and the local model's runs agree with each other
but not with the extractor, with both values and quotes.

Usage:
    uv run python scripts/local_field_probe.py --reference run_dir/replays/markdown-ab-<ts>.json
    uv run python scripts/local_field_probe.py --reference ... --model granite4.1:3b --limit 20

`--reference` is the raw results of `replay_markdown_ab.py --noise-only` (the extractor's runs on
the same pages). Needs tools_llm_local on its port; free, but keeps Ollama busy. Reads only.
'''
import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from agentic_job_search import evidence
from agentic_job_search.config import LLM_LOCAL_MCP_URL, LLM_LOCAL_SERVICE_NAME
from agentic_job_search.extract_openrouter import EXTRACT_PROPERTIES
from replay_evidence import RUN_DIR, inputs_from_html
from utils_tools_n_agents_common.mcp_client import call_mcp_tool

logger = logging.getLogger('local_field_probe')

PROBE_FIELDS = ('relocation', 'residency_scope', 'language_requirement')
DEFAULT_MODELS = ('lfm2.5:latest', 'granite4.1:3b')
DEFAULT_REPEAT = 2
PROBE_PURPOSE_SLUG = 'extract-field-probe'
PROBE_TEMPERATURE = 0.0
PROBE_MAX_TOKENS = 300
PROBE_TIMEOUT_SECONDS = 300
VALUE_SHOWN_MAX_CHARS = 100
SYSTEM = ('You read one job posting page and answer one question about it. Answer only from the page. '
          'When the page does not say, the value is an empty string.')


def comparable(field: str, value: object) -> str:
    '''A value as compared: case-folded, a language list as a sorted set (order carries nothing).'''
    text = str(value or '').strip().casefold()
    if field == 'language_requirement':
        parts = (p.strip() for p in text.replace(' or ', ',').replace(' and ', ',').split(','))
        return ', '.join(sorted({p for p in parts if p}))
    return text


def output_schema(field: str) -> dict:
    value = {'type': 'string'}
    if 'enum' in EXTRACT_PROPERTIES[field]:
        value['enum'] = EXTRACT_PROPERTIES[field]['enum']
    return {'type': 'object', 'properties': {'value': value, 'quote': {'type': 'string'}},
            'required': ['value', 'quote']}


def question(markdown: str, field: str) -> str:
    return (f'--- PAGE ---\n{markdown}\n--- END PAGE ---\n\n'
            f"Field `{field}`: {EXTRACT_PROPERTIES[field]['description']}\n"
            'Reply with JSON: "value" is the field\'s value, "" when the page does not say; "quote" is the '
            'words of the page that state it, copied exactly, "" when value is "".')


async def ask(model: str, markdown: str, field: str) -> dict:
    '''One field from one model. Raises with the model, field and reply when the reply is unusable.'''
    args = {'model': model, 'system': SYSTEM, 'prompt': question(markdown, field), 'temperature': PROBE_TEMPERATURE,
            'think': False, 'max_tokens': PROBE_MAX_TOKENS, 'output_format': output_schema(field),
            'service_name': LLM_LOCAL_SERVICE_NAME, 'purpose_slug': PROBE_PURPOSE_SLUG}
    started = time.monotonic()
    raw = await call_mcp_tool(LLM_LOCAL_MCP_URL, 'generate', args, timeout_seconds=PROBE_TIMEOUT_SECONDS)
    try:
        data = json.loads(raw)
        if data.get('error') or data.get('ok') is False:
            raise RuntimeError(f'the server reported an error: {raw}')
        answer = json.loads(data['response'])
        value, quote = str(answer['value']), str(answer['quote'])
    except (ValueError, KeyError, TypeError) as ex:
        raise RuntimeError(f'{model} {field}: unusable reply ({type(ex).__name__}: {ex}): {raw}') from ex
    found = not value.strip() or evidence.find_quote(quote, markdown) is not None
    return {'value': value if found else '', 'raw_value': value, 'quote': quote, 'rejected': not found,
            'seconds': round(time.monotonic() - started, 2), 'prompt_tokens': data.get('prompt_eval_count')}


def reference_fields(path: Path) -> dict[str, list[dict]]:
    '''job_id -> the extractor's Markdown runs (field values after verification) from a replay.'''
    pages = json.loads(path.read_text(encoding='utf-8'))
    return {p['job_id']: [r['fields'] for r in p['extracts']['markdown'] if not r['failed']]
            for p in pages if 'extracts' in p}


def report(results: list[dict], reference: dict[str, list[dict]], models: list[str], repeat: int) -> list[str]:
    lines = []
    for model in models:
        lines += ['', f'## {model}', '',
                  '| field | local runs disagree | extractor runs disagree | agree with extractor | local only | '
                  'extractor only | both, different | rejected citations | s per call |',
                  '|---|---|---|---|---|---|---|---|---|']
        disagreements = []
        for field in PROBE_FIELDS:
            counts = dict.fromkeys(('local_noise', 'ref_noise', 'agree', 'local_only', 'ref_only', 'differ', 'rejected',
                                    'incomplete'), 0)
            seconds = []
            for page in results:
                runs = page['answers'].get(model, {}).get(field) or []
                refs = reference.get(page['job_id']) or []
                if len(runs) < repeat or not refs:
                    counts['incomplete'] += 1
                    continue
                local = [comparable(field, r['value']) for r in runs]
                ref = [comparable(field, r.get(field)) for r in refs]
                counts['local_noise'] += len(set(local)) > 1
                counts['ref_noise'] += len(set(ref)) > 1
                counts['rejected'] += sum(r['rejected'] for r in runs)
                seconds += [r['seconds'] for r in runs]
                a, b = local[0], ref[0]
                key = ('agree' if a == b else 'local_only' if not b else 'ref_only' if not a else 'differ')
                counts[key] += 1
                if len(set(local)) == 1 and len(set(ref)) == 1 and a != b:
                    disagreements.append((field, page, runs[0], refs[0].get(field)))
            mean = sum(seconds) / len(seconds) if seconds else 0.0
            lines.append(f"| {field} | {counts['local_noise']} | {counts['ref_noise']} | {counts['agree']} | "
                         f"{counts['local_only']} | {counts['ref_only']} | {counts['differ']} | {counts['rejected']} | {mean:.1f} |"
                         + (f" ({counts['incomplete']} pages left out: a run failed)" if counts['incomplete'] else ''))
        lines += ['', f'### {model}: each agrees with itself, not with the extractor ({len(disagreements)})', '']
        for field, page, run, ref in disagreements:
            lines.append(f"- {field}, job {page['job_id']} ({page['path']}): local {run['value'][:VALUE_SHOWN_MAX_CHARS]!r} "
                         f"quoting {run['quote'][:VALUE_SHOWN_MAX_CHARS]!r}; extractor {str(ref or '')[:VALUE_SHOWN_MAX_CHARS]!r}")
    return lines


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True, help='raw results of replay_markdown_ab.py --noise-only')
    parser.add_argument('--model', action='append', help=f'local model; repeatable (default {", ".join(DEFAULT_MODELS)})')
    parser.add_argument('--limit', type=int, help='pages to probe (default: every page in the reference)')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT, help=f'runs per page (default {DEFAULT_REPEAT})')
    parser.add_argument('--fill', type=Path, help="an earlier probe's raw results: ask only the answers it is missing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
    models = args.model or list(DEFAULT_MODELS)
    reference = reference_fields(args.reference)
    pages = json.loads(args.reference.read_text(encoding='utf-8'))
    paths = [Path(p['path']) for p in pages if p['job_id'] in reference][:args.limit]
    started = datetime.now()
    earlier = {p['job_id']: p for p in json.loads(args.fill.read_text(encoding='utf-8'))} if args.fill else {}
    results = []
    for number, path in enumerate(paths, start=1):
        job_id = path.stem.split('-')[-1]
        page = earlier.get(job_id) or {'path': str(path), 'job_id': job_id, 'answers': {}}
        missing = any(len(page['answers'].get(m, {}).get(f) or []) < args.repeat for m in models for f in PROBE_FIELDS)
        if missing:
            _, markdown = inputs_from_html(path)
            # One model at a time over its fields, so Ollama reuses the page it just read.
            for model in models:
                answers = page['answers'].setdefault(model, {})
                for _ in range(args.repeat):
                    for field in PROBE_FIELDS:
                        if len(answers.setdefault(field, [])) >= args.repeat:
                            continue
                        try:
                            answers[field].append(await ask(model, markdown, field))
                        except Exception as ex:
                            logger.warning(f'{path.name} {model} {field}: {type(ex).__name__}: {ex}')
            print(f'[{number}/{len(paths)}] {path.name}', flush=True)
        results.append(page)
    lines = [f'# Local model per field vs the page extractor ({started:%Y-%m-%d %H:%M})', '',
             f'{len(results)} pages, {args.repeat} runs per model, reference {args.reference.name}; '
             f'{(datetime.now() - started).total_seconds() / 60:.0f} min. A count is pages; "agree" etc. compare '
             "each one's first run.", '']
    lines += report(results, reference, models, args.repeat)
    out = RUN_DIR / 'replays' / f'local-field-probe-{started:%Y%m%d-%H%M%S}'
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n'.join(lines))
    print(f'Report written to {out}.md (raw results {out}.json)')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
