'''Check LinkedIn Session and the Telegram message split (both added 2026-11-23).'''

from contextlib import asynccontextmanager

import pytest
from agentic_job_search import agent, config, linkedin_session
from agentic_job_search.linkedin_session import SessionCheck, decide_session

# --- decide_session --------------------------------------------------------------------------------

@pytest.mark.parametrize('url', [
    'https://www.linkedin.com/uas/login?session_redirect=%2Ffeed%2F',
    'https://www.linkedin.com/authwall?trk=gf&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2F',
    'https://www.linkedin.com/checkpoint/challenge/AgH',
    'https://www.linkedin.com/login',
])
def test_redirect_to_a_login_path_is_signed_out(url):
    check = decide_session({'url': url, 'guest_links': []})
    assert not check.signed_in
    assert 'redirected' in check.reason


def test_guest_page_links_are_signed_out():
    check = decide_session({'url': 'https://www.linkedin.com/feed/', 'guest_links': ['Sign in', 'Join now']})
    assert not check.signed_in
    assert 'Sign in' in check.reason


def test_feed_without_guest_links_is_signed_in():
    check = decide_session({'url': 'https://www.linkedin.com/feed/', 'guest_links': []})
    assert check.signed_in


def test_a_capture_without_url_decides_nothing():
    with pytest.raises(ValueError, match='no url'):
        decide_session({'guest_links': []})


async def test_check_navigates_to_the_feed_and_only_reads_it():
    calls: list[tuple[str, dict]] = []

    async def call(name, args):
        calls.append((name, args))
        if name == 'browser_evaluate':
            return '### Result\n{"url": "https://www.linkedin.com/feed/", "title": "Feed", "guest_links": []}\n'
        return 'ok'

    check = await linkedin_session.check_linkedin_session(call)
    assert check.signed_in
    assert [name for name, _ in calls] == ['browser_navigate', 'browser_evaluate']
    assert calls[0][1] == {'url': config.LINKEDIN_SESSION_CHECK_URL}


# --- _linkedin_session_ok: the run gate ------------------------------------------------------------

def _patch_check(monkeypatch, outcome):
    @asynccontextmanager
    async def fake_session(url):
        yield object()

    async def fake_check(call):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(agent, 'mcp_session', fake_session)
    monkeypatch.setattr(agent, 'check_linkedin_session', fake_check)


class _Browser:
    mcp = {'url': 'http://localhost:1/mcp'}


@pytest.fixture
def sent_and_logged(monkeypatch):
    sent: list[str] = []
    logged: list[dict] = []

    async def fake_notify(text):
        sent.append(text)
        return True
    monkeypatch.setattr(agent, '_send_pipeline_notification', fake_notify)
    monkeypatch.setattr(agent, 'log_run_cost', logged.append)
    monkeypatch.setattr(agent.tools_module, '_ui_alerts', [])
    return sent, logged


async def test_signed_out_stops_the_run_with_one_short_alert(monkeypatch, sent_and_logged):
    sent, logged = sent_and_logged
    _patch_check(monkeypatch, SessionCheck(False, 'https://www.linkedin.com/uas/login', 'redirected to /uas/login'))
    assert await agent._linkedin_session_ok(_Browser(), {'scraping': {'cost': 0.0}}, 0.0) is False
    assert len(sent) == 1
    assert 'LINKEDIN SIGNED OUT' in sent[0]
    assert len(sent[0]) <= config.TELEGRAM_MESSAGE_MAX_CHARS
    assert logged[0]['status'] == 'failed_signed_out'


async def test_signed_in_continues_silently(monkeypatch, sent_and_logged):
    sent, logged = sent_and_logged
    _patch_check(monkeypatch, SessionCheck(True, 'https://www.linkedin.com/feed/', 'ok'))
    assert await agent._linkedin_session_ok(_Browser(), {}, 0.0) is True
    assert sent == [] and logged == []


async def test_unreadable_check_continues_with_an_alert(monkeypatch, sent_and_logged):
    sent, _logged = sent_and_logged
    _patch_check(monkeypatch, RuntimeError('browser_navigate timed out'))
    assert await agent._linkedin_session_ok(_Browser(), {}, 0.0) is True
    assert sent == []
    assert agent.tools_module._ui_alerts[0]['kind'] == 'session_check_failed'


# --- split_telegram_message ------------------------------------------------------------------------

def test_a_message_within_the_limit_is_unchanged():
    assert agent.split_telegram_message('short\nmessage') == ['short\nmessage']


def test_a_long_message_is_split_at_line_breaks_with_nothing_lost():
    lines = [f'line {index} ' + 'x' * 50 for index in range(200)]
    text = '\n'.join(lines)
    parts = agent.split_telegram_message(text, max_chars=1000)
    assert len(parts) > 1
    assert all(len(part) <= 1000 for part in parts)
    assert parts[0].startswith(f'(1/{len(parts)}) ')
    bodies = [part.split(') ', 1)[1] for part in parts]
    assert '\n'.join(bodies) == text


def test_a_single_line_over_the_limit_is_cut_inside_the_line():
    text = 'y' * 2500
    parts = agent.split_telegram_message(text, max_chars=1000)
    assert all(len(part) <= 1000 for part in parts)
    assert ''.join(part.split(') ', 1)[1] for part in parts) == text


async def test_send_checks_the_length_and_sends_every_part(monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_replay_dir', None)
    sent: list[str] = []

    async def fake_send(text):
        sent.append(text)
    monkeypatch.setattr(agent, 'telegram_send_message', fake_send)
    text = '\n'.join('z' * 100 for _ in range(100))   # the 2026-11-23 alert was 5799 chars
    assert await agent._send_pipeline_notification(text) is True
    assert len(sent) > 1
    assert all(len(part) <= config.TELEGRAM_MESSAGE_MAX_CHARS for part in sent)


async def test_a_failed_part_reports_failure(monkeypatch):
    monkeypatch.setattr(agent.tools_module, '_replay_dir', None)

    async def failing_send(text):
        raise RuntimeError('tools_telegram refused it')
    monkeypatch.setattr(agent, 'telegram_send_message', failing_send)
    assert await agent._send_pipeline_notification('x' * 5000) is False


# --- health_alert_block: the trimmed NEEDS ATTENTION section ---------------------------------------

def test_alerts_of_one_kind_become_one_line_naming_every_subject():
    queries = ['Staff AI Engineer', 'Principal AI Engineer', 'Senior AI Engineer']
    alerts = [f'STOPPED EARLY on {q}: used only {turns} turns (expected >= 12)' for q, turns in zip(queries, (11, 7, 9), strict=True)]
    alerts.append('ALL 3 QUERIES FAILED — no search completed')
    block = agent.health_alert_block(alerts)
    lines = block.splitlines()
    assert lines[0].startswith('⚠️ NEEDS ATTENTION (4 — grouped by kind')
    assert len(lines) == 3
    assert all(q in lines[1] for q in queries)
    assert 'STOPPED EARLY ×3' in lines[1] and 'used only 11 turns' in lines[1]
    assert lines[2] == '  - ALL 3 QUERIES FAILED — no search completed'


def test_exact_repeats_are_dropped_and_long_details_are_cut_visibly():
    detail = 'search_box missing; ' * 40
    alerts = [f'UI CHANGED on Q / Canada: {detail}', f'UI CHANGED on Q / Canada: {detail}']
    block = agent.health_alert_block(alerts)
    assert block.count('UI CHANGED') == 1
    assert 'chars not shown]' in block
    assert len(block) < len(detail)


def test_distinct_alerts_are_listed_whole():
    alerts = ['LOCAL TRIAGE DISABLED: model missing', 'BROWSER RESTARTED: once']
    assert agent.health_alert_block(alerts) == (
        '⚠️ NEEDS ATTENTION (2):\n  - LOCAL TRIAGE DISABLED: model missing\n  - BROWSER RESTARTED: once')
