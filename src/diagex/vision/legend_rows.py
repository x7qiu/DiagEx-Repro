"""Page-space legend row proposals from native glyphs and adjacent printed labels.

Rows are evidence, not classifications. Model crop boundaries never define row
identity; repeated labels at different locations remain separate source rows.
"""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import median

from pydantic import BaseModel

from diagex.vision.evidence import PageEvidence, stable_evidence_id
from diagex.vision.models import BBox
from diagex.vision.vector_geometry import path_vertices


class NativeLegendRow(BaseModel):
    id: str
    page_index: int
    label: str
    bbox: BBox
    label_bbox: BBox
    source_path_ids: list[str]
    source_text_ids: list[str]


def union_boxes(boxes: list[BBox]) -> BBox:
    x, y = min(b.x for b in boxes), min(b.y for b in boxes)
    return BBox(x=x, y=y, w=max(b.x2 for b in boxes) - x, h=max(b.y2 for b in boxes) - y)


def _gap(a: BBox, b: BBox) -> tuple[float, float]:
    return max(0, a.x - b.x2, b.x - a.x2), max(0, a.y - b.y2, b.y - a.y2)


def native_legend_rows(page: PageEvidence, region: BBox | None = None) -> list[NativeLegendRow]:
    if page.is_scanned or not page.text_spans or not page.paths:
        return []
    heights = [t.bbox.h for t in page.text_spans if t.bbox.h > 0]
    if not heights:
        return []
    h = median(heights)
    rulings = []
    for path in page.paths:
        points = path_vertices(path)
        for a, b in zip(points, points[1:], strict=False):
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 18 * h:
                if abs(a[0] - b[0]) <= 1:
                    rulings.append((0, a[0], min(a[1], b[1]), max(a[1], b[1])))
                elif abs(a[1] - b[1]) <= 1:
                    rulings.append((1, a[1], min(a[0], b[0]), max(a[0], b[0])))

    def on_ruling(path):
        # Only suppress straight border/tick fragments. A real glyph can touch
        # a long demonstration pipe; that contact must not erase its contour.
        if min(path.bbox.w, path.bbox.h) > max(1, h * 0.07):
            return False
        return any(
            abs(pt[axis] - value) <= max(1, h * 0.07) and lo - 1 <= pt[1 - axis] <= hi + 1
            for pt in path.points
            for axis, value, lo, hi in rulings
        )

    paths = sorted(
        (
            p
            for p in page.paths
            if p.origin == "pdf_vector" and max(p.bbox.w, p.bbox.h) <= 18 * h and not on_ruling(p)
        ),
        key=lambda p: p.id,
    )
    # Join touching or closely spaced strokes, including disconnected motor,
    # actuator and break-mark parts. Long table rulings cannot join glyphs.
    parents = list(range(len(paths)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    grid = defaultdict(list)
    cell, gap = max(1, h * 4), h * 0.55
    for i, path in enumerate(paths):
        b = path.bbox
        keys = [
            (x, y)
            for x in range(int((b.x - gap) // cell), int((b.x2 + gap) // cell) + 1)
            for y in range(int((b.y - gap) // cell), int((b.y2 + gap) // cell) + 1)
        ]
        for j in {j for k in keys for j in grid[k]}:
            dx, dy = _gap(b, paths[j].bbox)
            if dx <= gap and dy <= gap:
                parents[root(i)] = root(j)
        for k in keys:
            grid[k].append(i)
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[root(i)].append(p)
    # Each compact glyph chooses a nearby outside label. Internal instrument
    # letters are kept as glyph context, never mistaken for row definitions.
    proposals = []
    direction_votes = defaultdict(list)
    for ps in groups.values():
        b = union_boxes([p.bbox for p in ps])
        if max(b.w, b.h) > 22 * h or min(b.w, b.h) > 15 * h or max(b.w, b.h) < h * 0.6:
            continue
        choices = []
        for t in page.text_spans:
            tb = t.bbox
            side = 1 if tb.x >= b.x2 else -1 if tb.x2 <= b.x else 0
            if not side or not t.text.strip() or not re.search(r"[A-Za-z\u4e00-\u9fff]", t.text):
                continue
            dx, dy = _gap(b, tb)
            cy = abs(tb.y + tb.h / 2 - b.y - b.h / 2)
            if dx > 16 * h or dy > h * 0.8 or cy > b.h / 2 + h:
                continue
            choices.append((dx / h + cy / h * 2, side != 1, t.id, side, t))
        if not choices:
            continue
        chosen = min(choices)
        column = round((b.x + b.w / 2) / (h * 2))
        direction_votes[column].append(chosen[3])
        proposals.append((column, ps, choices))
    anchored = defaultdict(list)
    for column, ps, choices in proposals:
        votes = direction_votes[column]
        majority = 1 if votes.count(1) >= votes.count(-1) else -1
        _, _, _, side, label = min((c for c in choices if c[3] == majority), default=min(choices))
        anchored[(label.id, side)].extend(ps)
    rows = []
    for (tid, side), ps in anchored.items():
        b = union_boxes([p.bbox for p in ps])
        if max(b.w, b.h) > 22 * h or min(b.w, b.h) > 15 * h:
            continue
        anchor = next(t for t in page.text_spans if t.id == tid)
        texts = [
            t
            for t in page.text_spans
            if b.y - h * 0.6 <= t.bbox.y + t.bbox.h / 2 <= b.y2 + h * 0.6
            and (
                abs(t.bbox.x - anchor.bbox.x) <= h * 0.7
                if side == 1
                else abs(t.bbox.x2 - anchor.bbox.x2) <= h * 0.7
            )
        ]
        if anchor not in texts:
            texts.append(anchor)
        # PDF exporters can split spaced names into separate spans on one line.
        # Extend along that baseline without crossing a column-sized gap.
        changed = True
        while changed:
            changed = False
            for t in page.text_spans:
                if t in texts:
                    continue
                if (side == 1 and t.bbox.x < b.x2) or (side == -1 and t.bbox.x2 > b.x):
                    continue
                if any(
                    abs(t.bbox.y - v.bbox.y) <= h * 0.35 and _gap(t.bbox, v.bbox)[0] <= h * 2
                    for v in texts
                ):
                    texts.append(t)
                    changed = True
        texts.sort(key=lambda t: (t.bbox.y, t.bbox.x, t.id))
        lb = union_boxes([t.bbox for t in texts])
        if region and not (
            region.x <= (b.x + b.x2) / 2 <= region.x2 and region.y <= (b.y + b.y2) / 2 <= region.y2
        ):
            continue
        pids, tids = sorted({p.id for p in ps}), sorted(t.id for t in texts)
        rows.append(
            NativeLegendRow(
                id=stable_evidence_id("legend-row", page.source_ref, page.page_index, pids, tids),
                page_index=page.page_index,
                label=" ".join(t.text.strip() for t in texts),
                bbox=b,
                label_bbox=lb,
                source_path_ids=pids,
                source_text_ids=tids,
            )
        )
    return sorted(rows, key=lambda r: (r.bbox.y, r.bbox.x, r.id))
