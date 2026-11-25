'''Prune Search Snapshot (POC): what each level keeps of a search page, and that no card ref survives.'''
import pytest
from agentic_job_search.config import SNAPSHOT_PRUNE_LEVEL_CARDS, SNAPSHOT_PRUNE_LEVEL_CHROME, SNAPSHOT_PRUNE_LEVEL_NO_CARDS
from agentic_job_search.snapshot_prune import prune_search_snapshot

PAGE = '''### Page
- Page URL: https://www.linkedin.com/jobs/search/
### Snapshot
```yaml
- generic [ref=e1]:
  - heading "0 notifications" [level=2] [ref=e2]
  - banner [ref=e3]:
    - button [ref=e4] [cursor=pointer]:
      - img "LinkedIn" [ref=e5]
    - search [ref=e6]:
      - textbox "Describe the job you want" [ref=e7]: Staff AI Engineer
  - navigation [ref=e8]:
    - button [ref=e9] [cursor=pointer]:
      - generic "Filter by Date posted" [ref=e10]:
        - checkbox "Date posted"
  - main [ref=e11]:
    - paragraph [ref=e12]: 99+ results
    - generic [ref=e13]:
      - button "Staff AI Engineer (Verified job) Acme Canada (Remote) Dismiss Staff AI Engineer job" [ref=e14] [cursor=pointer]:
        - figure [aria-hidden] [ref=e15]
        - paragraph [ref=e16]:
          - generic [aria-hidden] [ref=e17]: Staff AI Engineer (Verified job)
        - paragraph [ref=e18]: Acme
        - paragraph [ref=e19]: Canada (Remote)
        - button "Dismiss Staff AI Engineer job" [ref=e20]
      - button [ref=e21] [cursor=pointer]:
        - paragraph [ref=e22]: Principal Engineer
        - paragraph [ref=e23]: Jobber
        - paragraph [ref=e24]: Toronto, ON
        - button "Dismiss Principal Engineer job" [ref=e25]
    - button "Next" [ref=e26]
  - link [ref=e27] [cursor=pointer]:
    - /url: https://www.linkedin.com/feed
  - iframe [ref=e28]:
    - banner "Global Navigation" [ref=e29]
```'''


def test_chrome_keeps_the_search_box_chips_and_cards_and_drops_the_rest():
    out = prune_search_snapshot(PAGE, SNAPSHOT_PRUNE_LEVEL_CHROME, 'test')
    assert out.startswith('### Page\n- Page URL:') and out.endswith('```')
    for kept in ('textbox "Describe the job you want" [ref=e7]', 'Filter by Date posted', '99+ results',
                 '[ref=e14]', 'Dismiss Principal Engineer job', 'button "Next"'):
        assert kept in out
    for dropped in ('notifications', 'img "LinkedIn"', 'figure', '/url:', 'iframe', 'Global Navigation',
                    '[cursor=pointer]', '[aria-hidden]', '- generic [ref=e1]', '- generic [ref=e13]'):
        assert dropped not in out


def test_cards_become_one_line_each_with_no_ref():
    out = prune_search_snapshot(PAGE, SNAPSHOT_PRUNE_LEVEL_CARDS, 'test')
    assert '- job card: Staff AI Engineer · Acme · Canada (Remote)' in out
    assert '- job card: Principal Engineer · Jobber · Toronto, ON' in out
    for card_ref in ('e14', 'e20', 'e21', 'e25'):
        assert f'ref={card_ref}]' not in out
    assert 'Dismiss' not in out and '[ref=e9]' in out


def test_no_cards_leaves_one_counted_line():
    out = prune_search_snapshot(PAGE, SNAPSHOT_PRUNE_LEVEL_NO_CARDS, 'test')
    assert '[results list: 2 job cards omitted' in out and 'job card:' not in out
    assert 'button "Next" [ref=e26]' in out


def test_an_unknown_level_raises():
    with pytest.raises(ValueError, match='unknown snapshot prune level'):
        prune_search_snapshot(PAGE, 'all', 'test')
