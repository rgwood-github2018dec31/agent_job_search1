'''Compare a shadow recording's local answers with the remote ones (Shadow Local Model).

For each call site (the function that asked), how often the local model gave the same DECISION as
the remote one, overall and by the local model's own confidence. A decision is what a caller acts
on: for a JSON reply, its numbers, booleans, nulls and short strings (free text such as reasoning
or a quote is left out, since two models never word it alike); for a plain reply, the text itself
when short. Agreement is not correctness: the remote answer is a reference, not ground truth.

    uv run python scripts/shadow_report.py run_dir/shadow_calls/shadow-<run>.jsonl [more files...]
'''

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from agentic_job_search import shadow
from agentic_job_search.triage import extract_json_object

# A string this short is a decision value (a label, a country, a currency); longer is prose.
SHORT_TEXT_MAX_CHARS = 60
SHORT_TEXT_MAX_WORDS = 4
# Keys whose value is prose whatever its length: two models never word it alike.
FREE_TEXT_KEYS = frozenset({'reasoning', 'reason', 'summary', 'quote', 'quotes', 'evidence', 'text', 'explanation'})
CONFIDENCE_SURE, CONFIDENCE_FAIRLY_SURE = 5, 4
# A reply with no confidence line counts as unsure.
CONFIDENCE_UNSURE = (1, 2, 3, None)
TOP_FIELDS_SHOWN = 3


def _norm(text: str) -> str:
    return ' '.join(text.lower().split())


def _short(value: object) -> bool:
    return isinstance(value, str) and len(value) <= SHORT_TEXT_MAX_CHARS and len(value.split()) <= SHORT_TEXT_MAX_WORDS


def decision_view(content: str) -> dict | str | None:
    '''The comparable part of a reply; None when it has none (prose only).'''
    try:
        obj = extract_json_object(content)
    except ValueError:
        text = _norm(content)
        return text if _short(text) else None
    view = {}
    for key, value in obj.items():
        if key.lower() in FREE_TEXT_KEYS:
            continue
        if value is None or isinstance(value, bool | int | float):
            view[key] = value
        elif _short(value):
            view[key] = _norm(value)
        elif isinstance(value, list) and all(_short(v) for v in value):
            view[key] = sorted(_norm(v) for v in value)
    return view or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('recordings', type=Path, nargs='+')
    args = parser.parse_args()
    by_site: dict[str, dict] = defaultdict(lambda: {
        'status': Counter(), 'agree': 0, 'compared': 0, 'prose': 0,
        'by_conf': defaultdict(lambda: [0, 0]), 'fields': Counter(), 'remote_conf': Counter()})
    for path in args.recordings:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            site = by_site[record['call_site']]
            site['status'][record['status']] += 1
            if record['status'] != shadow.STATUS_PAIRED:
                continue
            site['remote_conf'][record['remote'].get('confidence')] += 1
            remote, local = decision_view(record['remote']['content']), decision_view(record['shadow']['content'])
            if remote is None:
                site['prose'] += 1
                continue
            same = remote == local
            site['compared'] += 1
            site['agree'] += same
            bucket = site['by_conf'][record['shadow'].get('confidence')]
            bucket[0] += 1
            bucket[1] += same
            if isinstance(remote, dict) and not same:
                local_view = local if isinstance(local, dict) else {}
                site['fields'].update(k for k in remote if remote.get(k) != local_view.get(k))

    print('| call site | pairs | pending | errors | agree | local conf 5 | conf 4 | conf ≤3 | most-differing fields |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, site in sorted(by_site.items(), key=lambda item: -item[1]['status'][shadow.STATUS_PAIRED]):
        def rate(n: int, agree: int) -> str:
            return f'{agree / n:.0%} of {n}' if n else '–'
        low = [sum(site['by_conf'][c][i] for c in CONFIDENCE_UNSURE) for i in (0, 1)]
        fields = ', '.join(f'{k} ({n})' for k, n in site['fields'].most_common(TOP_FIELDS_SHOWN))
        prose = f" (+{site['prose']} prose)" if site['prose'] else ''
        print(f"| {name} | {site['status'][shadow.STATUS_PAIRED]} | {site['status'][shadow.STATUS_PENDING]} | "
              f"{site['status'][shadow.STATUS_SHADOW_ERROR]} | {rate(site['compared'], site['agree'])}{prose} | "
              f"{rate(*site['by_conf'][CONFIDENCE_SURE])} | {rate(*site['by_conf'][CONFIDENCE_FAIRLY_SURE])} | {rate(*low)} | {fields or '–'} |")
    print('\nconf ≤3 includes replies with no confidence line. "prose" = replies with nothing comparable.')


if __name__ == '__main__':
    main()
