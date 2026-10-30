'''Ask models for one extract field at a time, and compare them with each other and the page extractor.

Why this is a tracked script
----------------------------
The page extractor (MODEL_NAME_EXTRACTOR, a reasoning model) answers differently on the same page:
over 243 retained pages run twice at EXTRACT_TEMPERATURE (2026-10-30), `relocation`,
`residency_scope` and `language_requirement` were present in one run and absent in the other on
dozens of pages, and those fields feed the gates. A small local model without thinking, at
temperature 0, decodes greedily and should give one answer per page. This measures whether it
does, and where it disagrees with the extractor, so a person can judge which one read the page right.

Per page and model, each field in PROBE_FIELDS is asked on its own: the page's Markdown first (so
Ollama reuses it for the next field), then what code read off the page's header card (location and
workplace badge, so the model is told whether the job is remote), then the field's definition from
gate_fields.DEFINITIONS, answered as {quotes, reason, value}: the prompt Ask Gate Fields sends in a
run, so a change to it is measured here before a run relies on it. A field never answers empty:
a page that does not say gives the field's NOT_KNOWN value, so "the model looked and the page does
not say" is told apart from "no answer".

The keys come in that order, so the model copies the page's words and says why before it commits to
a value; the reason is there to tune the prompt by. A local model (a name without '/') runs on
tools_llm_local with decoding constrained to the schema; an OpenRouter model (a name with '/') is
asked for the same JSON in words and checked against the schema in code. A value whose quote regex
cannot find on the page (evidence.find_quote) is rejected and counts as not known, as Verify Source
Reference would. Every page is asked `--repeat` times.

Reported per model and field: pages whose runs disagree (the extractor's own figure beside it,
from the `--reference` replay), agreement with the extractor, rejected citations, and every page
where the extractor's runs agree with each other and the model's runs agree with each other but not
with the extractor. With several models, also every page where the models' first runs differ, with
each one's value, quote and reason.

Usage:
    uv run python scripts/local_field_probe.py --reference run_dir/replays/markdown-ab-<ts>.json --limit 30
    uv run python scripts/local_field_probe.py --reference ... --model qwen3.5:9b-q8_0 --model z-ai/glm-5.3-flash

`--reference` is the raw results of `replay_markdown_ab.py --noise-only` (the extractor's runs on
the same pages). A local model needs tools_llm_local on its port and is free; an OpenRouter model
needs tools_llm_remote_openrouter and costs money (the report gives the total). Reads only.
'''
import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from agentic_job_search import evidence
from agentic_job_search.config import LLM_LOCAL_MCP_URL, LLM_LOCAL_SERVICE_NAME, OPENROUTER_MODEL_NAME_DEEPSEEK_V41_FLASH
from agentic_job_search.gate_fields import NOT_KNOWN, SYSTEM, output_schema, question
from agentic_job_search.gate_fields import header_facts as gate_header_facts
from agentic_job_search.linkedin_page import header_card
from agentic_job_search.triage import chat_openrouter, extract_json_object
from replay_evidence import RUN_DIR, inputs_from_html
from utils_tools_n_agents_common.mcp_client import call_mcp_tool

logger = logging.getLogger('local_field_probe')

PROBE_FIELDS = ('relocation', 'residency_scope', 'language_requirement')
OPENROUTER_MODEL_NAME_GLM = 'z-ai/glm-5.3-flash'
OPENROUTER_ATTEMPTS = 2
DEFAULT_MODELS = ('qwen3.5:9b-q8_0', OPENROUTER_MODEL_NAME_DEEPSEEK_V41_FLASH, OPENROUTER_MODEL_NAME_GLM)
DEFAULT_REPEAT = 2
PROBE_PURPOSE_SLUG = 'extract-field-probe'
PROBE_TEMPERATURE = 0.0
LOCAL_MAX_TOKENS = 600
LOCAL_TIMEOUT_SECONDS = 300
VALUE_SHOWN_MAX_CHARS = 100
# The definitions, rules and prompt are Ask Gate Fields' own (gate_fields.py): this probe measures them.


def comparable(field: str, value: object) -> str:
    '''A value as compared: case-folded, NOT_KNOWN as '' (as the extractor leaves it), a list as a sorted set
    (order carries nothing): languages separated by commas, places by "; " (gate_fields.PLACES_FORMAT).'''
    text = str(value or '').strip().casefold()
    if text == NOT_KNOWN[field]:
        return ''
    if field == 'language_requirement':
        parts = (p.strip() for p in text.replace(' or ', ',').replace(' and ', ',').split(','))
        return ', '.join(sorted({p for p in parts if p}))
    if field == 'relocation':
        return '; '.join(sorted({p.strip() for p in text.split(';') if p.strip()}))
    return text


def header_facts(path: Path) -> str:
    '''What code read off the header card, as Ask Gate Fields states it to the model.'''
    header = header_card(BeautifulSoup(path.with_suffix('.html').read_text(encoding='utf-8'), 'lxml'))
    return gate_header_facts(header.location, header.workplace)


async def reply_local(model: str, prompt: str, field: str) -> tuple[dict, dict]:
    args = {'model': model, 'system': SYSTEM, 'prompt': prompt, 'temperature': PROBE_TEMPERATURE,
            'think': False, 'max_tokens': LOCAL_MAX_TOKENS, 'output_format': output_schema(field),
            'service_name': LLM_LOCAL_SERVICE_NAME, 'purpose_slug': PROBE_PURPOSE_SLUG}
    raw = await call_mcp_tool(LLM_LOCAL_MCP_URL, 'generate', args, timeout_seconds=LOCAL_TIMEOUT_SECONDS)
    try:
        data = json.loads(raw)
        if data.get('error') or data.get('ok') is False:
            raise RuntimeError(f'the server reported an error: {raw}')
        return json.loads(data['response']), {'prompt_tokens': data.get('prompt_eval_count'), 'cost_usd': 0.0}
    except (ValueError, KeyError, TypeError) as ex:
        raise RuntimeError(f'{model} {field}: unusable reply ({type(ex).__name__}: {ex}): {raw}') from ex


async def reply_openrouter(model: str, prompt: str, field: str) -> tuple[dict, dict]:
    reply = await chat_openrouter(prompt, SYSTEM, model, temperature=PROBE_TEMPERATURE)
    try:
        answer = extract_json_object(reply.content)
    except ValueError as ex:
        raise RuntimeError(f'{model} {field}: no JSON in the reply ({reply.describe()}): {ex}') from ex
    return answer, {'prompt_tokens': reply.usage.get('prompt_tokens'), 'cost_usd': reply.cost_usd,
                    'provider': reply.provider}


RELOCATE_SCHEMA = {'type': 'object', 'properties': {'passage': {'type': 'string'}}, 'required': ['passage']}


async def relocate(model: str, markdown: str, quote: str) -> str:
    '''The page's own passage behind a quote regex could not find, as a local model copies it, and only when
    regex then finds that passage; '' when the model finds none or copies it unfaithfully. The same contract
    as evidence.verify_claims, with a local model in place of MODEL_NAME_EVIDENCE.'''
    prompt = (f'--- PAGE ---\n{markdown}\n--- END PAGE ---\n\n'
              f'Another model cited these words as copied from the page: "{quote}"\n'
              'They are not on the page word for word. If a passage of the page says the same thing, copy that passage '
              'exactly as "passage"; if no passage says it, "passage" is "".')
    args = {'model': model, 'system': SYSTEM, 'prompt': prompt, 'temperature': PROBE_TEMPERATURE, 'think': False,
            'max_tokens': LOCAL_MAX_TOKENS, 'output_format': RELOCATE_SCHEMA, 'service_name': LLM_LOCAL_SERVICE_NAME,
            'purpose_slug': PROBE_PURPOSE_SLUG}
    raw = await call_mcp_tool(LLM_LOCAL_MCP_URL, 'generate', args, timeout_seconds=LOCAL_TIMEOUT_SECONDS)
    try:
        passage = str(json.loads(json.loads(raw)['response'])['passage']).strip()
    except (ValueError, KeyError, TypeError) as ex:
        raise RuntimeError(f'{model} relocating {quote!r}: unusable reply ({type(ex).__name__}: {ex}): {raw}') from ex
    return passage if passage and evidence.find_quote(passage, markdown) is not None else ''


async def ask(model: str, markdown: str, facts: str, field: str, relocate_model: str | None = None) -> dict:
    '''One field from one model. Raises with the model, field and reply when the reply is unusable. With
    `relocate_model`, a quote regex cannot find is replaced by the passage that model finds for it (relocate).'''
    started = time.monotonic()
    reply = reply_openrouter if '/' in model else reply_local
    answer, meta = await reply(model, question(markdown, facts, field), field)
    try:
        value, reason = str(answer['value']).strip(), str(answer['reason']).strip()
        quotes = answer['quotes']
        if not isinstance(quotes, list):
            raise ValueError(f'quotes is a {type(quotes).__name__}, not a list')
        quotes = [str(q) for q in quotes if str(q).strip()]
        allowed = output_schema(field)['properties']['value'].get('enum')
        if not value or (allowed and value not in allowed):
            raise ValueError(f'value {value!r} is not allowed (empty, or not one of {allowed})')
    except (KeyError, ValueError) as ex:
        raise RuntimeError(f'{model} {field}: reply does not fit the schema ({type(ex).__name__}: {ex}): {answer}') from ex
    # Each passage is checked on its own, so their order carries nothing; a stated value needs at least one.
    missing = [q for q in quotes if evidence.find_quote(q, markdown) is None]
    relocated = {}
    if relocate_model and missing and value != NOT_KNOWN[field]:
        for quote in missing:
            relocated[quote] = await relocate(relocate_model, markdown, quote)
        missing = [q for q in missing if not relocated[q]]
    found = value == NOT_KNOWN[field] or (bool(quotes) and not missing)
    return {'value': value if found else NOT_KNOWN[field], 'raw_value': value, 'quote': ' | '.join(quotes),
            'quotes_not_found': missing, 'relocated': relocated, 'reason': reason, 'basis': str(answer.get('basis') or ''), 'rejected': not found,
            'seconds': round(time.monotonic() - started, 2), **meta}


def reference_fields(path: Path) -> dict[str, list[dict]]:
    '''job_id -> the extractor's Markdown runs (field values after verification) from a replay.'''
    pages = json.loads(path.read_text(encoding='utf-8'))
    return {p['job_id']: [r['fields'] for r in p['extracts']['markdown'] if not r['failed']]
            for p in pages if 'extracts' in p}


def report(results: list[dict], reference: dict[str, list[dict]], models: list[str], repeat: int) -> list[str]:
    lines = []
    for model in models:
        cost = sum(r.get('cost_usd') or 0.0 for p in results for runs in p['answers'].get(model, {}).values() for r in runs)
        lines += ['', f'## {model} (cost ${cost:.4f})', '',
                  '| field | runs disagree | extractor runs disagree | agree with extractor | model only | '
                  'extractor only | both, different | rejected citations | s per call |',
                  '|---|---|---|---|---|---|---|---|---|']
        disagreements = []
        for field in PROBE_FIELDS:
            counts = dict.fromkeys(('noise', 'ref_noise', 'agree', 'model_only', 'ref_only', 'differ', 'rejected',
                                    'incomplete'), 0)
            seconds = []
            for page in results:
                runs = page['answers'].get(model, {}).get(field) or []
                refs = reference.get(page['job_id']) or []
                if len(runs) < repeat or not refs:
                    counts['incomplete'] += 1
                    continue
                answers = [comparable(field, r['value']) for r in runs]
                ref = [comparable(field, r.get(field)) for r in refs]
                counts['noise'] += len(set(answers)) > 1
                counts['ref_noise'] += len(set(ref)) > 1
                counts['rejected'] += sum(r['rejected'] for r in runs)
                seconds += [r['seconds'] for r in runs]
                a, b = answers[0], ref[0]
                key = ('agree' if a == b else 'model_only' if not b else 'ref_only' if not a else 'differ')
                counts[key] += 1
                if len(set(answers)) == 1 and len(set(ref)) == 1 and a != b:
                    disagreements.append((field, page, runs[0], refs[0].get(field)))
            mean = sum(seconds) / len(seconds) if seconds else 0.0
            lines.append(f"| {field} | {counts['noise']} | {counts['ref_noise']} | {counts['agree']} | "
                         f"{counts['model_only']} | {counts['ref_only']} | {counts['differ']} | {counts['rejected']} | {mean:.1f} |"
                         + (f" ({counts['incomplete']} pages left out: a run failed)" if counts['incomplete'] else ''))
        tried = [(q, passage) for page in results for runs in page['answers'].get(model, {}).values() for r in runs
                 for q, passage in (r.get('relocated') or {}).items()]
        if tried:
            lines += ['', f'Relocated quotes: {sum(bool(passage) for _, passage in tried)} of {len(tried)} found on the page:', '']
            lines += [f'- {q!r} -> {passage!r}' for q, passage in tried]
        lines += ['', f'### {model}: agrees with itself, not with the extractor ({len(disagreements)})', '']
        for field, page, run, ref in disagreements:
            lines.append(f"- {field}, job {page['job_id']} ({page['path']}): {run['value'][:VALUE_SHOWN_MAX_CHARS]!r} "
                         f"quoting {run['quote'][:VALUE_SHOWN_MAX_CHARS]!r} because {run['reason']!r}; "
                         f"extractor {str(ref or '')[:VALUE_SHOWN_MAX_CHARS]!r}")
    if len(models) > 1:
        lines += report_across_models(results, models)
    return lines


def report_across_models(results: list[dict], models: list[str]) -> list[str]:
    '''Every page where the models' first runs give different values, with each one's quote and reason.'''
    lines = ['', '## Across models (first runs)', '']
    for field in PROBE_FIELDS:
        differ = []
        for page in results:
            firsts = {m: (page['answers'].get(m, {}).get(field) or [None])[0] for m in models}
            if any(r is None for r in firsts.values()):
                continue
            if len({comparable(field, r['value']) for r in firsts.values()}) > 1:
                differ.append((page, firsts))
        lines += [f'### {field}: models differ on {len(differ)} of {len(results)} pages', '']
        for page, firsts in differ:
            lines.append(f"- job {page['job_id']}, {page.get('facts', '')}")
            for model, run in firsts.items():
                rejected = f" (said {run['raw_value']!r}; citation rejected)" if run['rejected'] else ''
                lines.append(f"  - {model}: {run['value']!r}{rejected}, "
                             f"quoting {run['quote']!r}, because {run['reason']!r}")
        lines.append('')
    return lines


async def ask_model(model: str, page: dict, markdown: str, repeat: int, relocate_model: str | None) -> None:
    '''Fill one model's missing answers for one page, its fields in turn so Ollama reuses the page.'''
    answers = page['answers'].setdefault(model, {})
    for _ in range(repeat):
        for field in PROBE_FIELDS:
            if len(answers.setdefault(field, [])) >= repeat:
                continue
            # An OpenRouter reply that fails is asked again: GLM's malformed JSON came right on the next try
            # in 9 of 9 retries, cheaply, as the provider had cached the page (2026-10-30).
            attempts = OPENROUTER_ATTEMPTS if '/' in model else 1
            for attempt in range(1, attempts + 1):
                try:
                    answers[field].append({**await ask(model, markdown, page['facts'], field, relocate_model), 'attempt': attempt})
                    break
                except Exception as ex:
                    logger.warning(f"{page['job_id']} {model} {field} (attempt {attempt} of {attempts}): "
                                   f'{type(ex).__name__}: {ex}')


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True, help='raw results of replay_markdown_ab.py --noise-only')
    parser.add_argument('--model', action='append', help=f'model; repeatable (default {", ".join(DEFAULT_MODELS)})')
    parser.add_argument('--limit', type=int, help='pages to probe (default: every page in the reference)')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT, help=f'runs per page (default {DEFAULT_REPEAT})')
    parser.add_argument('--relocate-model', help='local model that finds the page passage behind a quote regex '
                        'cannot find, as evidence.verify_claims does with MODEL_NAME_EVIDENCE (default: none, rejected)')
    parser.add_argument('--concurrent', type=int, default=1,
                        help='pages at once (default 1; more only helps OpenRouter models, Ollama serves one call at a time)')
    parser.add_argument('--fill', type=Path, help="an earlier probe's raw results: ask only the answers it is missing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
    models = args.model or list(DEFAULT_MODELS)
    reference = reference_fields(args.reference)
    pages = json.loads(args.reference.read_text(encoding='utf-8'))
    paths = [Path(p['path']) for p in pages if p['job_id'] in reference][:args.limit]
    started = datetime.now()
    earlier = {p['job_id']: p for p in json.loads(args.fill.read_text(encoding='utf-8'))} if args.fill else {}
    slots = asyncio.Semaphore(args.concurrent)

    async def probe_page(number: int, path: Path) -> dict | None:
        job_id = path.stem.split('-')[-1]
        page = earlier.get(job_id) or {'path': str(path), 'job_id': job_id, 'answers': {}}
        missing = any(len(page['answers'].get(m, {}).get(f) or []) < args.repeat for m in models for f in PROBE_FIELDS)
        if missing:
            async with slots:
                try:
                    _, markdown = inputs_from_html(path)
                except FileNotFoundError as ex:
                    # Raw postings are pruned after RAW_POSTINGS_RETENTION_DAYS; an older reference names some gone.
                    logger.warning(f'Skipping {path.name}: {ex}')
                    return None
                page['facts'] = header_facts(path)
                # Models on different servers run side by side; each one asks its fields in turn.
                await asyncio.gather(*(ask_model(m, page, markdown, args.repeat, args.relocate_model) for m in models))
            print(f'[{number}/{len(paths)}] {path.name}', flush=True)
        return page

    results = [p for p in await asyncio.gather(*(probe_page(n, p) for n, p in enumerate(paths, start=1))) if p]
    lines = [f'# One extract field at a time vs the page extractor ({started:%Y-%m-%d %H:%M})', '',
             f'{len(results)} pages, {args.repeat} runs per model, reference {args.reference.name}; '
             f'{(datetime.now() - started).total_seconds() / 60:.0f} min. A count is pages; "agree" etc. compare '
             "each one's first run; a NOT_KNOWN value counts as the extractor's empty.", '']
    lines += report(results, reference, models, args.repeat)
    out = RUN_DIR / 'replays' / f'local-field-probe-{started:%Y%m%d-%H%M%S}'
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n'.join(lines))
    print(f'Report written to {out}.md (raw results {out}.json)')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
