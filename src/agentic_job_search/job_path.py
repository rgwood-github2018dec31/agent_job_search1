'''Job Path: one Job Trace laid over a fixed map of Stage 2, for the Decision Graph and the Job Audit.

A trace records only what happened, so the branches a job did NOT take come from `skeleton`, a
map of `evaluate_all_candidates` in agent.py: each node is a stage, and an exit beside it is a way
out of the pipeline there. How the job left is never inferred from where its steps stop: every
exit records an `outcome` step (`tools_generic.record_job_outcome`, 2026-11-12), and the overlay
reads it. A trace written before that has none, and says so.

A step this map does not know (a new step kind, a new LLM call site) is attached to the stage of
the step before it and named as unmapped in that stage's notes, so the map's drift shows in the
drawing instead of hiding in it.

Pure data in, data out, and no reportlab: the drawing is in scripts/job_path_drawing.py.
'''

import re
from collections import Counter
from dataclasses import dataclass, field

from agentic_job_search.job_trace import reply_meta

# Hard rules in the order apply_hard_rules evaluates them; tests/test_tools.py checks this against
# a real Stage 2 trace. A rule a trace names that is missing here is drawn after them.
HARD_RULES = ('blacklist', 'closed', 'stale', 'sponsorship', 'language', 'location', 'relocation', 'degree')

# The node an LLM call belongs to, by its call site (the caller of chat_openrouter and its kin).
CALL_SITE_NODES = {
    'decide_unknown_blocks': 'page',
    'extract_job_page_openrouter': 'extract', 'extract_job_page_direct': 'extract',
    '_find_missing_cited_passages': 'extract',
    'ask_field': 'gate',
    'classify_location': 'classify', 'resolve_salary': 'classify',
    'triage_job_fit': 'triage',
    'rate_with_openrouter': 'rate', 'rate_with_ollama': 'rate', '_rate_with_anthropic': 'rate',
    '_repost_match_openrouter': 'save', 'estimate_salary': 'save',
}
# Call sites that run inside whichever check is current (a company-name match, a company identity
# lookup, the blacklist confirmation): they belong to the node of the step before them.
CALL_SITES_OF_THE_STEP_BEFORE = frozenset({
    '_company_matches_openrouter', 'identity_for_name', '_blacklist_confirms_openrouter',
})
# The node a citation check belongs to, by its call site.
EVIDENCE_NODES = {'extract': 'extract', 'location_classify': 'classify', 'salary_classify': 'classify',
                  'triage': 'triage', 'rating_pros': 'rate', 'rating_warnings': 'rate'}
# The node of every other step kind. `outcome` is not placed: it says where the job left.
STEP_KIND_NODES = {
    'origin': 'queued', 'web_fetch': 'page', 'page_read': 'page', 'extract': 'extract',
    'page_copy_stale': 'extract', 'gate_fields': 'gate', 'classified': 'classify',
    'rating_inputs': 'rate', 'rating': 'rate', 'saved': 'save', 'notification': 'notify',
}
OUTCOME_STEP_KIND = 'outcome'
# A notification step; traces written before 2026-11-12 recorded a follow-up's under kind `follow-up`.
NOTIFICATION_STEP_KINDS = ('notification', 'follow-up')

# Outcomes (record_job_outcome) that end a job at a fixed node.
EXIT_NODE_BY_OUTCOME = {'extract_failed': 'extract', 'triaged_out': 'triage'}
EXIT_NODE_BY_APPLIED_CHECK = {'end_client': 'applied_client', 'company_identity': 'applied_domain'}
RATED_OUTCOMES = ('rated', 'recruiter_repost')

NOTE_CHARS = 150    # a note is cut here, with an ellipsis saying so (the trace keeps it whole)
SHORT_CHARS = 60    # a value quoted inside a note, a box title
TOP_REASONS = 3     # distinct exit reasons listed on an aggregate exit box

# Emoji and their joiners, which the embedded fonts have no glyph for (a Telegram text is full of them).
_EMOJI = re.compile('[\U0001F000-\U0001FFFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]')


@dataclass
class Node:
    id: str
    label: str
    exit_label: str = ''
    state: str = 'not_reached'      # taken | skipped | not_reached
    exit_state: str = ''            # taken | detour | not_taken | '' (no exit here)
    notes: list[str] = field(default_factory=list)
    exit_notes: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    count: int = 0                  # jobs that reached it (aggregate only)


def skeleton(rules: list[str]) -> list[Node]:
    '''Stage 2 for one job, every node not yet reached.'''
    return [
        Node('queued', 'Queued from search results'),
        Node('page', 'Read job page (code, read-only)'),
        Node('extract', 'Extract posting (LLM) + verify citations', 'extract failed: job skipped'),
        Node('gate', 'Ask gate fields (LLM, one per call)'),
        Node('classify', 'Classify location, salary, residency'),
        Node('applied_client', 'Already applied? (end client)', 'already applied (end client)'),
        *[Node(f'rule:{r}', f'Hard rule: {r}', f'rejected: {r}') for r in rules],
        Node('triage', 'Triage (local score)', 'triaged out'),
        Node('applied_domain', 'Already applied? (company identity)', 'already applied (company identity)'),
        Node('rate', 'Rate job fit (LLM)'),
        Node('caps', 'Rating caps (code)'),
        Node('save', 'Save job + post-rating checks'),
        Node('notify', 'Notify via Telegram', 'not notified: below the notification rating'),
        Node('notified', 'Notified'),
    ]


def rule_order(traces: list[list[dict]]) -> list[str]:
    '''HARD_RULES, then any rule a trace names that HARD_RULES lacks, in the order first seen.'''
    seen = [s['rule'] for steps in traces for s in steps if s['kind'] == 'hard_rule']
    return [*HARD_RULES, *dict.fromkeys(r for r in seen if r not in HARD_RULES)]


def cut(text: object, limit: int = NOTE_CHARS) -> str:
    '''One line of at most `limit` characters, emoji dropped; a cut ends in an ellipsis.'''
    text = ' '.join(_EMOJI.sub('', str(text)).split())
    return text if len(text) <= limit else text[:limit - 1] + '…'


def llm_note(step: dict) -> str:
    meta = {**step, **reply_meta(step)}
    usage = meta.get('usage') or {}
    tokens = ''
    if isinstance(usage, dict) and usage.get('prompt_tokens') is not None:
        tokens = f" {usage.get('prompt_tokens')}→{usage.get('completion_tokens')} tok"
    cost = f" ${meta['cost_usd']:.4f}" if isinstance(meta.get('cost_usd'), (int, float)) else ''
    model = str(meta.get('model') or '?').split('/')[-1]
    status = f" ERROR {cut(meta['error'], SHORT_CHARS)}" if meta.get('error') else ''
    unreadable = f" (raw reply unreadable: {meta['raw_reply_unreadable']})" if meta.get('raw_reply_unreadable') else ''
    return f"{step.get('call_site')}: {model} {meta.get('seconds', '?')}s{cost}{tokens}{status}{unreadable}"


def evidence_note(step: dict) -> str:
    verdicts = step.get('verdicts') or []
    statuses = Counter(v.get('status') for v in verdicts)
    text = f"citations ({step.get('call_site')}): " + ', '.join(f'{n} {s}' for s, n in statuses.items())
    rejected = [v for v in verdicts if v.get('status') == 'rejected']
    if rejected:
        text += ' — rejected: ' + '; '.join(f"{v.get('field')}={cut(v.get('value'), SHORT_CHARS)}" for v in rejected)
    return text


def step_note(step: dict) -> str | None:
    '''One line saying what a step found, or None when the node's label already says it.'''
    kind = step['kind']
    if kind == 'llm_call':
        return llm_note(step)
    if kind == 'evidence':
        return evidence_note(step)
    if kind == 'origin':
        listing = step.get('listing') or {}
        return f"{listing.get('company')} — {listing.get('title')} (position {step.get('position')})"
    if kind == 'page_read':
        if step.get('error'):
            return f"page not read: {cut(step['error'])}"
        return f"{step.get('source')}: {step.get('chars')} chars kept of {step.get('dom_chars')}"
    if kind == 'page_copy_stale':
        return f"page is a copy about {step.get('lag_days')} days old: it says {step.get('page_posted')!r}, " \
               f"the search card {step.get('card_posted')!r}"
    if kind == 'extract':
        fields = step.get('fields') or {}
        return f"location {fields.get('location')!r}, salary {fields.get('salary') or 'none'!r}"
    if kind == 'gate_fields':
        answers = step.get('answers') or {}
        return '; '.join(f"{f}={a.get('value', a.get('raw_value')) if isinstance(a, dict) else a!r}"
                         for f, a in answers.items())
    if kind == 'classified':
        geography = step.get('geography') or {}
        salary = step.get('salary_facts') or {}
        return f"countries {geography.get('countries')}, salary {salary.get('kind')}"
    if kind == 'hard_rule':
        inputs = {k: v for k, v in step.items() if k not in ('kind', 'at', 'elapsed_s', 'rule', 'fired', 'reason')}
        return cut(', '.join(f'{k}={v!r}' for k, v in inputs.items()))
    if kind == 'rating':
        triage = step.get('triage')
        return f"rated {step.get('raw_rating')}/5" + (f", triage {triage.get('score')}" if triage else '')
    if kind == 'saved':
        return f"saved as rating {step.get('rating')}: {step.get('summary')}"
    if kind in NOTIFICATION_STEP_KINDS:
        if step.get('text') is None:
            return None
        message_kind = step.get('message_kind') or kind
        return f"{message_kind} {'sent' if step.get('sent') else 'NOT delivered'}: {cut(step['text'])}"
    return None


def node_for(step: dict, previous: str) -> tuple[str, str]:
    '''(the node a step belongs to, '' or a note naming it as unmapped when the map does not know it).'''
    kind = step['kind']
    if kind == 'llm_call':
        call_site = step.get('call_site')
        if call_site in CALL_SITE_NODES:
            return CALL_SITE_NODES[call_site], ''
        if call_site in CALL_SITES_OF_THE_STEP_BEFORE:
            return previous, ''
        return previous, f'unmapped LLM call site {call_site!r} (job_path.CALL_SITE_NODES)'
    if kind == 'evidence':
        call_site = step.get('call_site')
        if call_site in EVIDENCE_NODES:
            return EVIDENCE_NODES[call_site], ''
        return previous, f'unmapped citation check {call_site!r} (job_path.EVIDENCE_NODES)'
    if kind == 'hard_rule':
        return f"rule:{step['rule']}", ''
    if kind in NOTIFICATION_STEP_KINDS:
        return 'notify', ''
    if kind in STEP_KIND_NODES:
        return STEP_KIND_NODES[kind], ''
    return previous, f'unmapped step kind {kind!r} (job_path.STEP_KIND_NODES)'


def _outcome_stated_by_other_steps(steps: list[dict]) -> dict | None:
    '''For a trace with no outcome step: the outcome when another step settles it on its own, else None.

    A fired hard rule ends the job there, and a `rating` step means it was rated. Nothing else is
    read: the end-client, triage and company-identity exits all save the job the same way, so a
    saved summary alone cannot say which check decided (the first Decision Graph guessed, wrongly).
    '''
    if fired := next((s for s in steps if s['kind'] == 'hard_rule' and s.get('fired')), None):
        return {'outcome': 'hard_ruled', 'reason': fired.get('reason')}
    if any(s['kind'] == 'rating' for s in steps):
        return {'outcome': 'rated', 'reason': ''}
    return None


def _exit(steps: list[dict], nodes: list[Node]) -> tuple[str, str, list[str]] | None:
    '''(node id, exit label or '' for the node's own, notes) of the way the job left; None when notified.'''
    last_reached = next((n.id for n in reversed(nodes) if n.steps), 'queued')
    outcome = next((s for s in reversed(steps) if s['kind'] == OUTCOME_STEP_KIND), None)
    if outcome is None:
        outcome = _outcome_stated_by_other_steps(steps)
    if outcome is None:
        # Every exit but extract-failed saves the job, so a `saved` step is how it left, not a stage
        # it reached: the exit sits at the last stage before it.
        last_reached = next((n.id for n in reversed(nodes) if n.id != 'save' and n.steps), 'queued')
        saved = next((s for s in reversed(steps) if s['kind'] == 'saved'), None)
        return last_reached, 'outcome not recorded', [
            *([f"saved as: {cut(saved.get('summary'))}"] if saved else []),
            'the trace has no outcome step: written before 2026-11-12, or the run stopped during this job']
    name, reason = outcome.get('outcome'), cut(outcome.get('reason') or '')
    if name in EXIT_NODE_BY_OUTCOME:
        return EXIT_NODE_BY_OUTCOME[name], '', [reason]
    if name == 'already_applied':
        node_id = EXIT_NODE_BY_APPLIED_CHECK.get(outcome.get('check') or '')
        return (node_id, '', [reason]) if node_id else (last_reached, 'already applied (check not recorded)', [reason])
    if name == 'hard_ruled':
        fired = next((s for s in steps if s['kind'] == 'hard_rule' and s.get('fired')), None)
        return (f"rule:{fired['rule']}", '', [reason]) if fired else (last_reached, 'rejected by a hard rule', [reason])
    if name in RATED_OUTCOMES:
        sent = [s for s in steps if s['kind'] in NOTIFICATION_STEP_KINDS]
        if not sent:
            return 'notify', 'no notification recorded', ['rated, but no notification step: Stage 2 stopped first']
        final = sent[-1]
        if final.get('text') is None:
            return 'notify', '', [cut(final.get('why'))]
        if not final.get('sent'):
            return 'notify', 'notification NOT delivered', [cut(final['text'])]
        return None
    return last_reached, name, [reason]


def overlay(steps: list[dict], rules: list[str]) -> list[Node]:
    '''Lay one trace over the skeleton: what was taken, the exit taken, what was never reached.'''
    nodes = skeleton(rules)
    by_id = {n.id: n for n in nodes}
    unmapped: dict[str, list[str]] = {}
    previous = 'queued'
    for step in steps:
        if step['kind'] == OUTCOME_STEP_KIND:
            continue
        previous, note = node_for(step, previous)
        by_id[previous].steps.append(step)
        if note:
            unmapped.setdefault(previous, []).append(note)
    for node in nodes:
        node.notes = [*(n for n in (step_note(s) for s in node.steps) if n), *unmapped.get(node.id, [])]

    exit_at = _exit(steps, nodes)
    end = [n.id for n in nodes].index(exit_at[0] if exit_at else 'notified')
    for node in nodes[:end + 1]:
        node.state = 'taken' if node.steps else 'skipped'
        if node.exit_label:
            node.exit_state = 'not_taken'
    exit_id = exit_at[0] if exit_at else ''
    for node in (by_id['applied_client'], by_id['applied_domain']):
        # These checks record no step of their own; a match shows only as the job's outcome, drawn as
        # this node's exit. Reached and not the exit, the check passed or did not apply.
        if node.state == 'skipped' and node.id != exit_id:
            node.state, node.notes = 'taken', ['passed or not applicable (the check records no step)']
    if by_id['triage'].state == 'skipped':
        by_id['triage'].notes = ['not run']

    rating = next((s for s in steps if s['kind'] == 'rating'), None)
    if rating and by_id['caps'].state != 'not_reached':
        caps = by_id['caps']
        caps.state, caps.notes = 'taken', [f"final rating {rating.get('capped_rating')}/5"]
        if rating.get('cap_reason'):
            caps.exit_state = 'detour'
            caps.exit_label = f"capped {rating.get('raw_rating')} → {rating.get('capped_rating')}"
            caps.exit_notes = [cut(rating['cap_reason'])]

    if exit_at is None:
        final = [s for s in steps if s['kind'] in NOTIFICATION_STEP_KINDS][-1]
        by_id['notified'].state = 'taken'   # the end of the path, which records no step of its own
        by_id['notified'].notes = [f"{final.get('message_kind') or final['kind']} sent"]
        return nodes
    node_id, label, notes = exit_at
    node = by_id[node_id]
    node.state = 'taken'   # the job reached the stage it left from, steps or not (an applied check has none)
    node.exit_state, node.exit_label, node.exit_notes = 'taken', label or node.exit_label, notes
    for later in nodes[end + 1:]:
        later.state, later.exit_state, later.notes = 'not_reached', '', []
    return nodes


def outcome(nodes: list[Node]) -> str:
    '''A short name for where a job ended, used to pick one example of each.'''
    exited = next((n for n in nodes if n.exit_state == 'taken'), None)
    if exited:
        # A rule's own exit is named by the rule; a trace that only stopped at a rule stage is not its rejection.
        fired_rule = exited.id.startswith('rule:') and exited.exit_label == f"rejected: {exited.id.removeprefix('rule:')}"
        return exited.id if fired_rule else exited.exit_label
    return 'notified' + (' (capped)' if any(n.exit_state == 'detour' for n in nodes) else '')


def aggregate(all_nodes: list[list[Node]], rules: list[str]) -> list[Node]:
    '''One skeleton whose notes are counts: how many jobs reached each node and took each exit.'''
    nodes = skeleton(rules)
    for i, node in enumerate(nodes):
        column = [job[i] for job in all_nodes]
        reached = sum(n.state != 'not_reached' for n in column)
        skipped = sum(n.state == 'skipped' for n in column)
        exits = [n for n in column if n.exit_state in ('taken', 'detour')]
        node.state = 'taken' if reached else 'not_reached'
        node.count = reached
        node.notes = [f'{reached} job(s) reached' + (f', {skipped} skipped it' if skipped else '')]
        if exits:
            node.exit_state = 'detour' if all(n.exit_state == 'detour' for n in exits) else 'taken'
            labels = Counter(n.exit_label for n in exits)
            mixed = node.exit_label or ' / '.join(label for label, _ in labels.most_common())
            node.exit_label = f'{len(exits)} × ' + (mixed if len(labels) > 1 else exits[0].exit_label)
            reasons = Counter(n.exit_notes[0] if n.exit_notes else n.exit_label for n in exits)
            node.exit_notes = [f'{c} × {r}' for r, c in reasons.most_common(TOP_REASONS)]
            if len(reasons) > TOP_REASONS:
                node.exit_notes.append(f'… and {len(reasons) - TOP_REASONS} other reason(s)')
        elif node.exit_label:
            node.exit_state = 'not_taken'
    return nodes


def fold_unreached(nodes: list[Node]) -> list[Node]:
    '''The stages after a job's exit as one faded node, so a rejected job's path stays short.'''
    tail = next((i for i in range(len(nodes)) if all(n.state == 'not_reached' for n in nodes[i:])), len(nodes))
    if len(nodes) - tail < 2:
        return nodes
    rest = Node('rest', f'Not reached: {len(nodes) - tail} later stages')
    rest.notes = [', '.join(n.label for n in nodes[tail:])]
    return [*nodes[:tail], rest]
