'''Drawing of a Job Path (src/agentic_job_search/job_path.py) as reportlab flowables.

One flowable per stage, so a long path splits between stages across pages wherever platypus puts
it. Used by the Job Audit (job_trace_report.py: one job's path) and the Decision Graph
(job_trace_graph.py: every job of a date, counted). reportlab is not a project dependency, so this
lives with the scripts that run under `uv run --with reportlab`.

Sizes are PDF points (1/72 inch): a PDF page has no user font size to follow.
'''

from __future__ import annotations

from agentic_job_search.job_path import SHORT_CHARS, Node, cut, fold_unreached
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import Flowable

SPINE_SHARE = 0.47          # the stage boxes' share of the available width; exit boxes take the rest
SPINE_TO_EXIT = 30          # horizontal run of an exit edge
GAP = 14                    # vertical space above each box, where the arrow into it goes
LINE = 9.5                  # note line height
FONT_SIZE, NOTE_SIZE = 9, 7
MAX_NOTES = 5               # notes drawn in a box; the rest are counted, not hidden
BOX_TITLE_DROP = 11         # baseline of a box's title below its top
BOX_PAD_X, NOTE_INDENT = 5, 7
BOX_HEIGHT_MIN, BOX_PAD_BOTTOM = 16, 4
CORNER = 3
DASH = (3, 2)
STROKE_STRONG, STROKE_FAINT, EDGE_STRONG = 1.2, 0.6, 1.5
ARROW_EXTRA_MAX = 5         # extra width of the busiest arrow when counts are drawn
COUNT_OFFSET_X = 6
LEGEND_SWATCH_W, LEGEND_SWATCH_H, LEGEND_TEXT_X, LEGEND_ITEM_GAP = 14, 8, 18, 26
LEGEND_HEIGHT = 16

STYLE = {  # state: (stroke, fill, text)
    'taken': (colors.HexColor('#2e7d32'), colors.HexColor('#e8f5e9'), colors.black),
    'exit': (colors.HexColor('#c62828'), colors.HexColor('#ffebee'), colors.black),
    'detour': (colors.HexColor('#ef6c00'), colors.HexColor('#fff3e0'), colors.black),
    'skipped': (colors.HexColor('#757575'), colors.HexColor('#f5f5f5'), colors.HexColor('#424242')),
    'not_taken': (colors.HexColor('#bdbdbd'), colors.white, colors.HexColor('#9e9e9e')),
    'not_reached': (colors.HexColor('#e0e0e0'), colors.white, colors.HexColor('#bdbdbd')),
}
STRONG_STATES = ('taken', 'exit', 'detour')
EXIT_STYLE = {'taken': 'exit', 'detour': 'detour', 'not_taken': 'not_taken'}
LEGEND = (('taken', 'path taken'), ('exit', 'exit taken'), ('detour', 'detour (capped)'),
          ('skipped', 'skipped'), ('not_taken', 'exit not taken'), ('not_reached', 'not reached'))


def _box(canvas, x: float, y: float, w: float, h: float, state: str, dashed: bool = False) -> None:
    stroke, fill, _ = STYLE[state]
    canvas.setStrokeColor(stroke)
    canvas.setFillColor(fill)
    canvas.setLineWidth(STROKE_STRONG if state in STRONG_STATES else STROKE_FAINT)
    if dashed:
        canvas.setDash(*DASH)
    canvas.roundRect(x, y, w, h, CORNER, stroke=1, fill=1)
    canvas.setDash()


class Legend(Flowable):
    '''One line naming what each box style means.'''

    def __init__(self, fonts: tuple[str, str]):
        super().__init__()
        self.fonts = fonts

    def wrap(self, avail_width, avail_height):
        return avail_width, LEGEND_HEIGHT

    def draw(self):
        canvas, x = self.canv, 0.0
        for state, text in LEGEND:
            _box(canvas, x, LEGEND_HEIGHT - LEGEND_SWATCH_H, LEGEND_SWATCH_W, LEGEND_SWATCH_H, state,
                 dashed=state == 'not_taken')
            canvas.setFont(self.fonts[0], NOTE_SIZE)
            canvas.setFillColor(colors.black)
            canvas.drawString(x + LEGEND_TEXT_X, LEGEND_HEIGHT - NOTE_SIZE, text)
            x += LEGEND_ITEM_GAP + canvas.stringWidth(text, self.fonts[0], NOTE_SIZE)


class StageFlowable(Flowable):
    '''One stage: the arrow into it, its box with notes, and its exit box when it has one.'''

    def __init__(self, node: Node, fonts: tuple[str, str], count_in: int | None, count_max: int):
        super().__init__()
        self.node, self.fonts, self.count_in, self.count_max = node, fonts, count_in, count_max

    def _lines(self, notes: list[str], width: float) -> list[str]:
        shown = notes[:MAX_NOTES] + ([f'… {len(notes) - MAX_NOTES} more step(s)'] if len(notes) > MAX_NOTES else [])
        return [part for note in shown
                for part in simpleSplit('· ' + note, self.fonts[0], NOTE_SIZE, width - 2 * BOX_PAD_X - NOTE_INDENT)]

    def wrap(self, avail_width, avail_height):
        self.spine_w = avail_width * SPINE_SHARE
        self.exit_x = self.spine_w + SPINE_TO_EXIT
        self.exit_w = avail_width - self.exit_x
        self.notes = self._lines(self.node.notes, self.spine_w)
        drawn_exit = self.node.exit_state in ('taken', 'detour')
        self.exit_lines = self._lines(self.node.exit_notes, self.exit_w) if drawn_exit else []
        self.box_h = BOX_HEIGHT_MIN + LINE * max(len(self.notes), len(self.exit_lines)) + BOX_PAD_BOTTOM
        return avail_width, self.box_h + GAP

    def _text(self, x: float, top: float, title: str, lines: list[str], color) -> None:
        canvas = self.canv
        canvas.setFillColor(color)
        canvas.setFont(self.fonts[1], FONT_SIZE)
        canvas.drawString(x + BOX_PAD_X, top - BOX_TITLE_DROP, title)
        canvas.setFont(self.fonts[0], NOTE_SIZE)
        y = top - BOX_TITLE_DROP - LINE
        for line in lines:
            canvas.drawString(x + NOTE_INDENT, y, line)
            y -= LINE

    def draw(self):
        canvas, node = self.canv, self.node
        top, mid_x = self.box_h, self.spine_w / 2
        if self.count_in is not None:
            reached = node.state != 'not_reached'
            canvas.setStrokeColor(STYLE['taken' if reached else 'not_reached'][0])
            extra = ARROW_EXTRA_MAX * self.count_in / self.count_max if self.count_max and reached else 0
            canvas.setLineWidth(1 + extra)
            canvas.line(mid_x, top + GAP, mid_x, top)
            if self.count_max > 1:
                canvas.setFont(self.fonts[0], NOTE_SIZE)
                canvas.setFillColor(colors.black)
                canvas.drawString(mid_x + COUNT_OFFSET_X, top + (GAP - NOTE_SIZE) / 2, str(self.count_in))
        _box(canvas, 0, 0, self.spine_w, self.box_h, node.state)
        self._text(0, top, node.label, self.notes, STYLE[node.state][2])
        if not node.exit_state:
            return
        state = EXIT_STYLE[node.exit_state]
        exit_h = self.box_h if self.exit_lines else BOX_HEIGHT_MIN
        edge_y = top - BOX_HEIGHT_MIN / 2
        canvas.setStrokeColor(STYLE[state][0])
        canvas.setLineWidth(EDGE_STRONG if state != 'not_taken' else STROKE_FAINT)
        if state == 'not_taken':
            canvas.setDash(*DASH)
        canvas.line(self.spine_w, edge_y, self.exit_x, edge_y)
        canvas.setDash()
        _box(canvas, self.exit_x, top - exit_h, self.exit_w, exit_h, state, dashed=state == 'not_taken')
        self._text(self.exit_x, top, cut(node.exit_label, SHORT_CHARS), self.exit_lines, STYLE[state][2])


def path_flowables(nodes: list[Node], fonts: tuple[str, str], counts: bool = False) -> list[Flowable]:
    '''The legend and one flowable per stage. With `counts`, arrows carry each node's job count
    (an aggregate); otherwise the stages after the job's exit are folded into one.'''
    if not counts:
        nodes = fold_unreached(nodes)
    count_max = max((n.count for n in nodes), default=0) if counts else 0
    return [Legend(fonts), *(
        StageFlowable(node, fonts, (node.count if counts else 1) if i else None, count_max)
        for i, node in enumerate(nodes))]
