'''Two extractors, and DeepSeek per field where they disagree: does it steady the extract?

Why this is a tracked script
----------------------------
The page extractor (MODEL_NAME_EXTRACTOR, DeepSeek) gives a different value for about 8% of fields
on a second run of the same page (2026-10-30), some of them fields that feed the gates. Asked one
field at a time, with the header's location and workplace type stated and a few reading rules
(scripts/local_field_probe.py), DeepSeek and GLM agreed on all but 2 of 117 gate fields. This
measures a scheme that spends the per-field calls only where they are needed:

1. DeepSeek extracts the page as a run does (taken from `--reference`, the Markdown runs of
   `replay_markdown_ab.py --noise-only`, so they are not paid for again).
2. GLM extracts the same page with the same extractor and prompt (MODEL_NAME_EXTRACTOR swapped).
3. Every field whose two values differ is asked of DeepSeek on its own: the page, the header facts,
   gate_fields.READING_RULES and the extractor's own definition of that field
   (EXTRACT_PROPERTIES), answered as {quotes, reason, value}; the quotes are checked as
   Verify Source Reference does. That answer is the field's final value.

Run `--repeat` times per page (DeepSeek run i with GLM run i), the final extracts of two runs are
compared: the scheme's run-to-run variation, beside DeepSeek's own. Also reported: how many fields
needed a per-field call, GLM's failures, and every page whose final value still changed between runs.

Usage:
    uv run python scripts/replay_extract_arbitrate.py --reference run_dir/replays/markdown-ab-<ts>.json
    uv run python scripts/replay_extract_arbitrate.py --reference ... --limit 40

Needs the OpenRouter MCP tool server on :8006. Costs about a tenth of a cent per GLM extraction and
per per-field call: about $1-2 for every retained page at `--repeat 2`. Reads only.
'''
import argparse
import asyncio
import collections
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from agentic_job_search import evidence, extract_openrouter
from agentic_job_search.agent import EXTRACT_CITED_FIELDS
from agentic_job_search.config import OPENROUTER_MODEL_NAME_DEEPSEEK_V41_FLASH
from agentic_job_search.extract_openrouter import EXTRACT_PROPERTIES
from agentic_job_search.gate_fields import READING_RULES, SYSTEM
from agentic_job_search.linkedin_page import header_card
from agentic_job_search.triage import chat_openrouter, extract_json_object
from local_field_probe import OPENROUTER_ATTEMPTS, OPENROUTER_MODEL_NAME_GLM, header_facts
from replay_evidence import RUN_DIR, candidate_for, inputs_from_html, load_capture, pinned
from replay_markdown_ab import _value, extract_once

logger = logging.getLogger('replay_extract_arbitrate')

DEFAULT_REPEAT = 2
DEFAULT_CONCURRENT_PAGES = 16
ARBITER_MODEL = OPENROUTER_MODEL_NAME_DEEPSEEK_V41_FLASH
ARBITER_TEMPERATURE = 0.0
NOT_KNOWN = 'not_known'
VALUE_SHOWN_MAX_CHARS = 100
_PUNCTUATION_RE = re.compile(r'[\s.,;:()]+')


def same(a: str, b: str) -> bool:
    '''Two field values as the gates would read them: case, spacing and punctuation carry nothing.'''
    return _PUNCTUATION_RE.sub(' ', a.casefold()).strip() == _PUNCTUATION_RE.sub(' ', b.casefold()).strip()


def value_rule(field: str) -> str:
    spec = EXTRACT_PROPERTIES[field]
    if spec['type'] == 'boolean':
        return f'"true" or "false", or "{NOT_KNOWN}" when the page does not settle it'
    if 'enum' in spec:
        return f'one of {[v for v in spec["enum"] if v]}, or "{NOT_KNOWN}" when the page does not say'
    return f'the field\'s value, or "{NOT_KNOWN}" when the page does not say'


def question(markdown: str, facts: str, field: str) -> str:
    return (f'--- PAGE ---\n{markdown}\n--- END PAGE ---\n\n{facts}\n\n{READING_RULES}\n\n'
            f"Field `{field}`: {EXTRACT_PROPERTIES[field]['description']}\n"
            'Reply with only a JSON object with these keys, in this order: "quotes" is a list of the passages of the '
            'page that give the value, each copied exactly from one place on the page (for a header fact, the '
            "page's own words), [] when nothing on the page states it; \"reason\" is one sentence on why those "
            f'passages and the header facts give the value; "value" is {value_rule(field)}.')


async def arbitrate(markdown: str, facts: str, field: str) -> dict:
    '''DeepSeek's answer for one field, asked on its own; '' when not known or its quotes are not on the page.'''
    started = time.monotonic()
    reply = await chat_openrouter(question(markdown, facts, field), SYSTEM, ARBITER_MODEL, temperature=ARBITER_TEMPERATURE)
    try:
        answer = extract_json_object(reply.content)
        value, reason, quotes = str(answer['value']).strip(), str(answer['reason']).strip(), answer['quotes']
        if not isinstance(quotes, list):
            raise ValueError(f'quotes is a {type(quotes).__name__}, not a list')
    except (KeyError, ValueError) as ex:
        raise RuntimeError(f'{ARBITER_MODEL} {field}: unusable reply ({type(ex).__name__}: {ex}; {reply.describe()}): '
                           f'{reply.content}') from ex
    quotes = [str(q) for q in quotes if str(q).strip()]
    stated = value.casefold() not in {NOT_KNOWN, '', 'false'}
    missing = [q for q in quotes if evidence.find_quote(q, markdown) is None]
    rejected = stated and (not quotes or bool(missing))
    final = '' if not stated or rejected else ('True' if value.casefold() == 'true' else value)
    return {'value': final, 'raw_value': value, 'quotes': quotes, 'quotes_not_found': missing, 'reason': reason,
            'rejected': rejected, 'cost': reply.cost_usd, 'seconds': round(time.monotonic() - started, 2)}


async def arbitrate_retrying(markdown: str, facts: str, field: str) -> dict | None:
    for attempt in range(1, OPENROUTER_ATTEMPTS + 1):
        try:
            return {**await arbitrate(markdown, facts, field), 'attempt': attempt}
        except Exception as ex:
            logger.warning(f'{field} (attempt {attempt} of {OPENROUTER_ATTEMPTS}): {type(ex).__name__}: {ex}')
    return None


async def run_page(path: Path, deepseek_runs: list[dict], repeat: int, gate: asyncio.Semaphore) -> dict | None:
    async with gate:
        try:
            plain, markdown = inputs_from_html(path)
        except FileNotFoundError as ex:
            logger.warning(f'Skipping {path.name}: {ex}')
            return None
        front, _ = load_capture(path)
        candidate = candidate_for(front, plain)
        header = header_card(BeautifulSoup(path.with_suffix('.html').read_text(encoding='utf-8'), 'lxml'))
        facts = header_facts(path)
        page = {'path': str(path), 'job_id': candidate['job_id'], 'facts': facts}

        async def one_run(i: int) -> dict:
            deepseek = deepseek_runs[i]
            run = {'deepseek': deepseek['fields'], 'glm': None, 'glm_cost': 0.0, 'arbitrated': {}, 'final': None}
            # The extractor is GLM for the whole process (main); DeepSeek's extracts come from the reference.
            try:
                glm = await extract_once(path, candidate, plain, markdown, header, 'markdown')
            except Exception as ex:
                logger.warning(f'{path.name} run {i + 1}: GLM extraction failed: {type(ex).__name__}: {ex}')
                glm = {'failed': True, 'error': f'{type(ex).__name__}: {ex}'}
            glm.pop('extract', None)
            run['glm_cost'] = glm.get('cost', 0.0)
            if glm['failed']:
                run['final'] = dict(deepseek['fields'])
                return run
            run['glm'] = glm['fields']
            final = {f: deepseek['fields'].get(f, '') for f in EXTRACT_CITED_FIELDS}
            differ = [f for f in EXTRACT_CITED_FIELDS if not same(final[f], glm['fields'].get(f, ''))]
            answers = await asyncio.gather(*(arbitrate_retrying(markdown, facts, f) for f in differ))
            for field, answer in zip(differ, answers, strict=True):
                run['arbitrated'][field] = answer
                # A failed per-field call keeps DeepSeek's extract value, as a run without this step would.
                if answer is not None:
                    final[field] = _value(answer['value'])
            run['final'] = final
            return run

        page['runs'] = list(await asyncio.gather(*(one_run(i) for i in range(repeat))))
        print(f'{path.name}: done', flush=True)
        return page


def report(pages: list[dict]) -> list[str]:
    fields = list(EXTRACT_CITED_FIELDS)
    stats = {f: collections.Counter() for f in fields}
    changed = []
    cost = collections.Counter()
    glm_failed = 0
    for page in pages:
        runs = page['runs']
        glm_failed += sum(r['glm'] is None for r in runs)
        for r in runs:
            cost['glm'] += r['glm_cost']
            cost['arbiter'] += sum(a['cost'] for a in r['arbitrated'].values() if a)
            for field, a in r['arbitrated'].items():
                stats[field]['asked'] += 1
                stats[field]['failed'] += a is None
                stats[field]['rejected'] += bool(a and a['rejected'])
        if len(runs) < 2:
            continue
        for field in fields:
            stats[field]['pages'] += 1
            stats[field]['deepseek_noise'] += not same(runs[0]['deepseek'].get(field, ''), runs[1]['deepseek'].get(field, ''))
            if not same(runs[0]['final'].get(field, ''), runs[1]['final'].get(field, '')):
                stats[field]['final_noise'] += 1
                changed.append((page, field))
    lines = ['| field | pages | DeepSeek alone: runs differ | scheme: runs differ | per-field calls | failed | rejected |',
             '|---|---|---|---|---|---|---|']
    for field in fields:
        s = stats[field]
        lines.append(f"| {field} | {s['pages']} | {s['deepseek_noise']} | {s['final_noise']} | {s['asked']} | "
                     f"{s['failed']} | {s['rejected']} |")
    total = collections.Counter()
    for s in stats.values():
        total.update(s)
    lines += [f"| all | {total['pages']} | {total['deepseek_noise']} | {total['final_noise']} | {total['asked']} | "
              f"{total['failed']} | {total['rejected']} |", '',
              f"GLM extractions failed: {glm_failed}. Cost: GLM ${cost['glm']:.3f}, per-field DeepSeek ${cost['arbiter']:.3f} "
              '(DeepSeek extractions taken from the reference, not paid again).', '',
              f'## Final values that still changed between runs ({len(changed)})', '']
    for page, field in changed:
        lines.append(f"- {field}, job {page['job_id']}, {page['facts']}")
        for i, r in enumerate(page['runs'], start=1):
            a = r['arbitrated'].get(field)
            how = ('agreed' if field not in r['arbitrated'] else 'per-field call failed' if a is None
                   else f"per-field {a['raw_value']!r}{' (rejected)' if a['rejected'] else ''} because {a['reason']!r}")
            lines.append(f"  - run {i}: final {r['final'].get(field, '')[:VALUE_SHOWN_MAX_CHARS]!r}; DeepSeek "
                         f"{r['deepseek'].get(field, '')[:VALUE_SHOWN_MAX_CHARS]!r}, GLM "
                         f"{(r['glm'] or {}).get(field, '')[:VALUE_SHOWN_MAX_CHARS]!r}; {how}")
    return lines


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True, help='raw results of replay_markdown_ab.py --noise-only')
    parser.add_argument('--limit', type=int, help='pages (default: every page in the reference)')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT, help=f'runs per page (default {DEFAULT_REPEAT})')
    parser.add_argument('--concurrent', type=int, default=DEFAULT_CONCURRENT_PAGES,
                        help=f'pages at once (default {DEFAULT_CONCURRENT_PAGES})')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    usable = [p for p in reference if 'extracts' in p
              and sum(not r['failed'] for r in p['extracts']['markdown']) >= args.repeat][:args.limit]
    skipped = len(reference) - len(usable) if args.limit is None else 0
    # Set once, never swapped back and forth: pages run concurrently, and a swap restored by one page would
    # move another page's GLM extraction to DeepSeek mid-loop.
    extract_openrouter.MODEL_NAME_EXTRACTOR = OPENROUTER_MODEL_NAME_GLM
    gate = asyncio.Semaphore(args.concurrent)
    started = datetime.now()
    pages = [page for page in await asyncio.gather(*(run_page(pinned(Path(p['path'])), [r for r in p['extracts']['markdown'] if not r['failed']],
                                            args.repeat, gate) for p in usable)) if page]
    lines = [f'# DeepSeek and GLM extract, DeepSeek per field where they differ ({started:%Y-%m-%d %H:%M})', '',
             f'{len(pages)} pages ({skipped} in the reference left out: fewer than {args.repeat} DeepSeek runs), '
             f'{args.repeat} runs each, DeepSeek runs from {args.reference.name}; '
             f'{(datetime.now() - started).total_seconds() / 60:.0f} min. Values compared ignoring case, spacing and '
             'punctuation.', '']
    lines += report(pages)
    out = RUN_DIR / 'replays' / f'extract-arbitrate-{started:%Y%m%d-%H%M%S}'
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(pages, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('\n'.join(lines))
    print(f'Report written to {out}.md (raw results {out}.json)')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
