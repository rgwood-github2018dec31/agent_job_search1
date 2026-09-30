import logging

from agentic_job_search import snapshot_prune

WHAT = 'browser_snapshot result for query instance1'

SEARCH_UI = '''\
- generic [ref=e1]:
  - banner [ref=e2]:
    - textbox "Describe the job you want" [ref=e3]: staff ml engineer remote
    - generic "Filter by Past week" [ref=e4]:
      - checkbox "Past week" [checked]
  - main [ref=e5]:
    - generic [ref=e6]:
      - generic [ref=e7]:
        - paragraph [ref=e8]: 99+ results
        - generic "Location Canada" [ref=e9]
      - generic [ref=e10]:
        - generic [ref=e11]:
          - button "Acme Toronto (Remote) Dismiss Staff ML Engineer job Posted 2 days ago" [ref=e12] [cursor=pointer]:
            - button "Dismiss Staff ML Engineer job" [ref=e13]
          - button "Beta Vancouver (Remote) Dismiss ML Lead job Posted 1 week ago" [ref=e14] [cursor=pointer]:
            - button "Dismiss ML Lead job" [ref=e15]
          - button "Gamma Montreal (Remote) Dismiss Principal ML job Posted 3 days ago" [ref=e16] [cursor=pointer]:
            - button "Dismiss Principal ML job" [ref=e17]'''

PANE = '''
        - generic [ref=e20]:
          - generic [ref=e21]:
            - link "Staff ML Engineer" [ref=e22]
            - heading "About the job" [level=2] [ref=e23]
            - paragraph [ref=e24]: We are hiring a Staff ML Engineer to build things.
          - heading "About the company" [level=2] [ref=e25]
          - paragraph [ref=e26]: Acme makes anvils.'''

FOOTER = '''
  - contentinfo [ref=e30]:
    - paragraph [ref=e31]: LinkedIn Corporation'''


def test_removes_the_pane_and_keeps_the_search_ui(caplog):
    snapshot = SEARCH_UI + PANE + FOOTER
    with caplog.at_level(logging.INFO, logger=snapshot_prune.logger.name):
        pruned = snapshot_prune.drop_job_detail_pane(snapshot, WHAT)

    assert 'About the job' not in pruned and 'Acme makes anvils' not in pruned
    assert '        - [job detail pane omitted: ' in pruned, 'marker sits at the pane root indent'
    assert pruned.startswith(SEARCH_UI), 'chips, count, location chip and every card kept verbatim'
    assert pruned.count('Dismiss') == snapshot.count('Dismiss')
    assert pruned.endswith(FOOTER), 'content after the pane is kept'
    messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert messages == [f'{WHAT}: removed job detail pane, {len(snapshot)} -> {len(pruned)} chars']


def test_snapshot_without_a_pane_is_unchanged_and_not_logged(caplog):
    snapshot = SEARCH_UI + FOOTER
    with caplog.at_level(logging.DEBUG, logger=snapshot_prune.logger.name):
        assert snapshot_prune.drop_job_detail_pane(snapshot, WHAT) == snapshot
    assert caplog.records == []


def test_pane_sharing_its_only_ancestor_with_the_cards_is_kept():
    """If the pane cannot be separated from the search UI, nothing is removed."""
    snapshot = '''\
- main [ref=e1]:
  - button "Acme Toronto (Remote) Dismiss Staff ML Engineer job" [ref=e2] [cursor=pointer]
  - heading "About the job" [level=2] [ref=e3]'''
    assert snapshot_prune.drop_job_detail_pane(snapshot, WHAT) == snapshot


def test_single_job_page_is_unchanged():
    """A job's own page has no search UI; the 'pane' is the whole page and must not be removed."""
    snapshot = '''\
- generic [ref=e1]:
  - banner [ref=e2]:
    - button "Dismiss Job search smarter with Premium" [ref=e3] [cursor=pointer]
  - main [ref=e4]:
    - heading "About the job" [level=2] [ref=e5]
    - paragraph [ref=e6]: We are hiring.'''
    assert snapshot_prune.drop_job_detail_pane(snapshot, WHAT) == snapshot


def test_pane_next_to_the_results_count_is_kept():
    snapshot = '''\
- main [ref=e1]:
  - paragraph [ref=e2]: 99+ results
  - heading "About the job" [level=2] [ref=e3]'''
    assert snapshot_prune.drop_job_detail_pane(snapshot, WHAT) == snapshot
