'''JSON in the reply text vs a forced function call, for the rater and Ask Gate Fields, on saved calls, live.

Why this is a tracked script
----------------------------
On 2026-11-12 the rater (`rate_with_openrouter`) and Ask Gate Fields (`ask_field`) moved from a JSON
object dug out of the reply text to a forced function call (`chat_openrouter_tool`): the answer
arrives as arguments checked against a schema. That changes what the model is asked to produce, so
before it decides jobs it is checked here against the same inputs.

Every sample is an `llm_call` a run recorded in a Job Trace (run_dir/raw_postings/*/*.trace.jsonl),
with its system message and prompt. The prompt's last paragraph (how to answer) is swapped for the
other form; the rest is byte for byte what the run sent. Each sample is asked once in each form with
the model the run used, and compared with the run's own answer:
- text vs the run: how much the same question in the same form moves from one ask to the next
  (the noise a difference has to beat);
- tool vs the run, and tool vs text: what the function call changes beyond that noise.
Compared: the raw rating (before hard rules and caps, which read fields, not the posting), and a gate
field's raw value with its basis. Also counted: unusable replies per form, cost and seconds.

Forms (`--forms`, each with its number of asks per sample): `text` (JSON in the reply), `tool` (the
function call as the pipeline sends it; for the rater, `reasoning` ahead of `rating`) and, for the rater
only, `tool_rating_first` (the function as first written, `rating` ahead of `reasoning`). With repeats, a
difference counts only against how much the same form moves when asked again.

First measured 2026-11-12 (`forced-tool-ab-20261112-124054`): gate fields no worse as a function call
and cheaper; DeepSeek's ratings one point higher on 6 of 40 and lower on none, against 38 of 40
unchanged in text, so the rater stayed in text and the second run asked each form again
(`--forms text=1,tool=3,tool_reasoning_first=2 --gate 0 --model deepseek/deepseek-v4.1-flash-1017`;
`forced-tool-ab-20261112-125243`). With reasoning first, the rating moved like text asked again, so
DeepSeek rates by function call since; that run's `tool` is today's `tool_rating_first` and its
`tool_reasoning_first` today's `tool`.

Usage:
    uv run python scripts/replay_forced_tool_ab.py                       # DEFAULT_RATER_SAMPLES + DEFAULT_GATE_SAMPLES
    uv run python scripts/replay_forced_tool_ab.py --rater 10 --gate 30 --since 2026-11-09
    uv run python scripts/replay_forced_tool_ab.py --gate 0 --forms text=1,tool=3,tool_rating_first=2

Needs the OpenRouter MCP tool server on :8006. Costs about a tenth of a cent per rating pair and
less per gate-field pair. Reads only: nothing is rated into the job records or notified. The report
and the raw results go to run_dir/replays/forced-tool-ab-{timestamp}/.
'''
import argparse
import asyncio
import collections
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agentic_job_search import config, gate_fields, tool_server, triage
from agentic_job_search.config import GATE_FIELDS_TEMPERATURE, LLM_OPENROUTER_MCP_URL, RATING_TEMPERATURE
from agentic_job_search.preferences import RUN_DIR
from utils_tools_n_agents_common.mcp_client import unwrap_exception

logger = logging.getLogger('replay_forced_tool_ab')

RAW_POSTINGS_DIR = RUN_DIR / 'raw_postings'
RATER, GATE = 'rate_with_openrouter', 'ask_field'
FORMS = ('text', 'tool', 'tool_rating_first')
RATER_ONLY_FORMS = ('tool_rating_first',)
DEFAULT_FORMS = 'text=1,tool=1'
DEFAULT_RATER_SAMPLES, DEFAULT_GATE_SAMPLES = 50, 150
DEFAULT_SEED = 20261112
DEFAULT_CONCURRENT_CALLS = 6
RATER_TAIL = '\n\n' + triage.RATING_JSON_INSTRUCTIONS
GATE_TAIL = '\nReply with only a JSON object'
_FIELD_MARK = 'Field `'
REASON_SHOWN_MAX_CHARS = 200


def load_samples(since: str) -> dict[str, list[dict]]:
    '''Every recorded rater and gate-field call from `since` on whose prompt has the text form's tail.'''
    found: dict[str, list[dict]] = {RATER: [], GATE: []}
    seen: set[tuple[str, str]] = set()
    unrecognised = collections.Counter()
    for path in sorted(RAW_POSTINGS_DIR.glob('*/*.trace.jsonl')):
        if path.parent.name < since:
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            step = json.loads(line)
            site = step.get('call_site')
            if step.get('kind') != 'llm_call' or site not in found or step.get('error') or not step.get('reply'):
                continue
            prompt = step['prompt']
            tail = RATER_TAIL if site == RATER else GATE_TAIL
            if tail not in prompt:
                unrecognised[site] += 1
                continue
            if (site, prompt) in seen:
                continue
            seen.add((site, prompt))
            found[site].append({'trace': str(path.relative_to(RUN_DIR)), 'site': site, 'model': step['model'],
                                'system': step.get('system') or '', 'prompt': prompt, 'run_reply': step['reply']})
    for site, count in unrecognised.items():
        logger.warning(f'{count} {site} call(s) skipped: their prompt has no recognised answer paragraph')
    return found


def prompts_for(sample: dict) -> dict[str, str]:
    '''The sample's prompt in each form: the run's prompt up to its answer paragraph, then that form's.'''
    prompt = sample['prompt']
    if sample['site'] == RATER:
        head = prompt[:prompt.index(RATER_TAIL)]
        tool = f'{head}\n\n{triage.RATING_TOOL_INSTRUCTIONS}'
        return {'text': f'{head}{RATER_TAIL}', 'tool': tool, 'tool_rating_first': tool}
    head = prompt[:prompt.index(GATE_TAIL) + 1]
    field_name = sample['field']
    return {'text': head + gate_fields.answer_request(field_name, as_tool=False),
            'tool': head + gate_fields.answer_request(field_name, as_tool=True)}


def parse_forms(spec: str) -> dict[str, int]:
    '''`text=1,tool=3` -> {'text': 1, 'tool': 3}; raises naming an unknown form or a bad count.'''
    forms = {}
    for part in spec.split(','):
        name, _, count = part.partition('=')
        if name not in FORMS or not count.isdigit() or int(count) < 1:
            raise SystemExit(f'--forms: {part!r} is not <form>=<asks> with a form of {FORMS}')
        forms[name] = int(count)
    return forms


def field_of(prompt: str) -> str:
    start = prompt.rindex(_FIELD_MARK) + len(_FIELD_MARK)
    return prompt[start:prompt.index('`', start)]


def decision(site: str, field_name: str, answer: dict) -> str:
    '''What the comparison reads: the raw rating, or a gate field's raw value with its basis.'''
    if site == RATER:
        return str(int(answer['rating']))
    parsed = gate_fields.answer_from(field_name, answer)
    return parsed.raw_value + (f' [{parsed.basis}]' if field_name == 'relocation' else '')


def run_decision(sample: dict) -> str:
    '''The run's own answer, read the way the run read it; the reason when it cannot be read.'''
    try:
        return decision(sample['site'], sample.get('field', ''), triage.extract_json_object(sample['run_reply']))
    except (ValueError, KeyError, TypeError) as ex:
        return f'unreadable: {type(ex).__name__}: {ex}'


async def ask(sample: dict, form: str, prompt: str) -> dict:
    '''One ask in `form`; {decision, reason, cost_usd, seconds} or {error, seconds}.'''
    site, field_name = sample['site'], sample.get('field', '')
    temperature = RATING_TEMPERATURE if site == RATER else GATE_FIELDS_TEMPERATURE
    started = time.monotonic()
    try:
        if form == 'text':
            reply = await triage.chat_openrouter(prompt, system=sample['system'], model=sample['model'],
                                                 temperature=temperature)
            answer = triage.extract_json_object(reply.content)
        elif site == RATER:
            parameters = (triage.RATING_TOOL_PARAMETERS_RATING_FIRST if form == 'tool_rating_first'
                          else triage.RATING_TOOL_PARAMETERS)
            answer, reply = await triage.chat_openrouter_tool(
                prompt, triage.RATING_TOOL_NAME, triage.RATING_TOOL_DESCRIPTION, parameters,
                system=sample['system'], model=sample['model'], temperature=temperature)
        else:
            answer, reply = await triage.chat_openrouter_tool(
                prompt, gate_fields.TOOL_NAME, gate_fields.TOOL_DESCRIPTION, gate_fields.output_schema(field_name),
                system=sample['system'], model=sample['model'], temperature=temperature)
        result = {'decision': decision(site, field_name, answer),
                  'reason': str(answer.get('reasoning') or answer.get('reason') or ''),
                  'cost_usd': reply.cost_usd, 'provider': reply.provider}
    except Exception as ex:
        logger.warning(f"{site} {field_name} {form} ({sample['trace']}): {unwrap_exception(ex)}")
        result = {'error': unwrap_exception(ex)}
    result['seconds'] = round(time.monotonic() - started, 2)
    return result


def _answered(row: dict, form: str) -> list[str]:
    return [a['decision'] for a in row[form] if 'decision' in a]


def _majority(decisions: list[str]) -> str:
    return collections.Counter(decisions).most_common(1)[0][0] if decisions else ''


def summarise(results: list[dict], forms: dict[str, int]) -> list[str]:
    lines = []
    for site in (RATER, GATE):
        rows = [r for r in results if r['site'] == site]
        if not rows:
            continue
        site_forms = [f for f in forms if site == RATER or f not in RATER_ONLY_FORMS]
        lines.append(f'## {site} — {len(rows)} saved calls')
        for form in site_forms:
            asks = [a for r in rows for a in r[form]]
            answered = [a for a in asks if 'decision' in a]
            seconds = sorted(a['seconds'] for a in asks)
            line = (f'- {form} ×{forms[form]}: {len(asks) - len(answered)} unusable of {len(asks)}; '
                    f'${sum(a["cost_usd"] for a in answered):.4f} in all; median {seconds[len(seconds) // 2]:.1f}s; '
                    f'differ from the run {sum(a["decision"] != r["run_decision"] for r in rows for a in r[form] if "decision" in a)}'
                    f' of {len(answered)}')
            if forms[form] > 1:
                unsteady = sum(len(set(_answered(r, form))) > 1 for r in rows)
                line += f'; samples whose own asks disagree {unsteady} of {len(rows)}'
            if site == RATER:
                deltas = collections.Counter(int(a['decision']) - int(r['run_decision'])
                                             for r in rows for a in r[form] if 'decision' in a)
                line += f"; rating minus the run's: {dict(sorted(deltas.items()))}"
            lines.append(line)
        for i, left in enumerate(site_forms):
            for right in site_forms[i + 1:]:
                both = [r for r in rows if _answered(r, left) and _answered(r, right)]
                differ = [r for r in both if _majority(_answered(r, left)) != _majority(_answered(r, right))]
                lines.append(f'- {left} vs {right} (most frequent answer of each): {len(differ)} of {len(both)} differ')
        lines.append('')
        lines.append("Samples where the forms' most frequent answers differ, with the run's answer:")
        for r in rows:
            majorities = {f: _majority(_answered(r, f)) for f in site_forms}
            if len(set(majorities.values())) > 1:
                what = r.get('field') or 'rating'
                asked = ', '.join(f'{f} {_answered(r, f)}' for f in site_forms)
                lines.append(f"  - {r['trace']} {what}: run {r['run_decision']!r}; {asked}")
        for form in site_forms:
            for r in rows:
                for a in r[form]:
                    if 'error' in a:
                        lines.append(f"  - unusable {form}: {r['trace']} {r.get('field') or 'rating'}: {a['error']}")
        lines.append('')
    return lines


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rater', type=int, default=DEFAULT_RATER_SAMPLES, help='rater calls to replay')
    parser.add_argument('--gate', type=int, default=DEFAULT_GATE_SAMPLES, help='gate-field calls to replay')
    parser.add_argument('--since', default='', help='first date folder to read (default: all)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT_CALLS)
    parser.add_argument('--forms', default=DEFAULT_FORMS, help=f'forms and asks per sample (default {DEFAULT_FORMS})')
    parser.add_argument('--model', default='', help='keep only the sampled calls the run made with this model')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    config.SHADOW_MODEL = None   # the production prompts: no confidence asked

    # A down server would otherwise turn every call into an "unusable" reply and write a report of nothing.
    down = await tool_server.probe(LLM_OPENROUTER_MCP_URL)
    if down:
        raise SystemExit(f'The OpenRouter tool server at {LLM_OPENROUTER_MCP_URL} is not answering: {down}')
    forms = parse_forms(args.forms)
    found = load_samples(args.since)
    rng = random.Random(args.seed)
    samples = []
    for site, wanted in ((RATER, args.rater), (GATE, args.gate)):
        pool = found[site]
        chosen = rng.sample(pool, min(wanted, len(pool)))
        if args.model:
            chosen = [c for c in chosen if c['model'] == args.model]
        logger.info(f'{site}: {len(chosen)} of {len(pool)} saved calls')
        samples += chosen
    for sample in samples:
        if sample['site'] == GATE:
            sample['field'] = field_of(sample['prompt'])
        sample['run_decision'] = run_decision(sample)

    limit = asyncio.Semaphore(args.concurrent)

    async def one(sample: dict) -> dict:
        prompts = prompts_for(sample)
        answers = {}
        for form, asks in forms.items():
            if sample['site'] == GATE and form in RATER_ONLY_FORMS:
                continue
            answers[form] = []
            for _ in range(asks):
                async with limit:
                    answers[form].append(await ask(sample, form, prompts[form]))
        return {**{k: v for k, v in sample.items() if k not in ('system', 'prompt', 'run_reply')}, **answers}

    started = time.monotonic()
    results = await asyncio.gather(*(one(s) for s in samples))
    out = RUN_DIR / 'replays' / f'forced-tool-ab-{datetime.now():%Y%m%d-%H%M%S}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding='utf-8')
    report = [f'# JSON text vs forced function call — {len(results)} saved calls, '
              f'{time.monotonic() - started:.0f}s', '', *summarise(results, forms)]
    (out / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report))
    logger.info(f'Report: {out / "report.md"}')


if __name__ == '__main__':
    asyncio.run(main())
