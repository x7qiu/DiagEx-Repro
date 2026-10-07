"""Page-local physical glyph proposals for model classification, never automatic truth.

Candidates own native strokes and stable page-space geometry. Labels cannot create
candidates or join instances. Scale-relative shape gates intentionally cover a
bounded vocabulary; open/composite symbols remain explicit model proposals.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from pydantic import BaseModel, Field

from diagex.vision.evidence import PageEvidence, stable_evidence_id
from diagex.vision.models import BBox
from diagex.vision.vector_geometry import NativeSymbol, native_symbols, path_vertices, segment_key

SYMBOL_PERCEPTION_VERSION = "2.5.0"
_INSTRUMENT_TEXT = re.compile(r"(?:[PTFLAY][A-Z]{0,4}|I|M)(?:[- ]?\d+)?")
PERCEPTION_DEPENDENT_STAGES = (
    "perception",
    "contextual",
    "topology",
    "line_evidence",
    "page_graph",
    "assembly",
    "export",
)


class SymbolCandidate(BaseModel):
    id: str
    page_index: int
    bbox: BBox
    shape: str
    source_path_ids: list[str]
    source_text_ids: list[str] = Field(default_factory=list)
    text: list[str] = Field(default_factory=list)


def _box(points: list[tuple]) -> BBox:
    xs, ys = zip(*points, strict=True)
    x, y = round(min(xs)), round(min(ys))
    return BBox(x=x, y=y, w=max(1, round(max(xs)) - x), h=max(1, round(max(ys)) - y))


def _inside(inner: BBox, outer: BBox, tolerance: float = 1) -> bool:
    return (
        inner.x >= outer.x - tolerance
        and inner.y >= outer.y - tolerance
        and inner.x2 <= outer.x2 + tolerance
        and inner.y2 <= outer.y2 + tolerance
    )


def _corners(points: list[tuple], tolerance: float) -> list[tuple]:
    """Remove only collinear polygon vertices (including a repeated closing point)."""
    points = list(points)
    if points[0] == points[-1]:
        points.pop()
    changed = True
    while changed and len(points) > 3:
        changed = False
        for i, b in enumerate(points):
            a, c = points[i - 1], points[(i + 1) % len(points)]
            distance = abs((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / max(
                math.dist(a, c), 1e-9
            )
            if (
                distance <= tolerance
                and math.dist(a, b) + math.dist(b, c) <= math.dist(a, c) + tolerance
            ):
                points.pop(i)
                changed = True
                break
    return points


def _round(points: list[tuple], box: BBox) -> bool:
    if len(points) < 8 or not 0.8 <= box.w / box.h <= 1.25:
        return False
    radii = [
        math.hypot((x - box.x - box.w / 2) / (box.w / 2), (y - box.y - box.h / 2) / (box.h / 2))
        for x, y in points
    ]
    # Source vertices are quantized to page pixels; one pixel matters for a
    # small valve seat. The allowance shrinks with physical glyph size.
    return max(abs(r - 1) for r in radii) <= 0.14 + 1.5 / min(box.w, box.h)


def _capsule(symbol: NativeSymbol, scale: float) -> bool:
    b = symbol.bbox
    length, width = max(b.w, b.h), min(b.w, b.h)
    if (
        width < scale * 0.004
        or length < scale * 0.015
        or not 1.5 <= length / width <= 14
        or symbol.area < b.w * b.h * 0.72
        or len(symbol.points) < 8
    ):
        return False
    axis = int(b.h >= b.w)
    start = b.y if axis else b.x
    middle = b.x + b.w / 2 if axis else b.y + b.h / 2
    for end in (start, start + length):
        cap = [p[1 - axis] for p in symbol.points if abs(p[axis] - end) <= max(0.8, width * 0.025)]
        if not cap or abs((min(cap) + max(cap)) / 2 - middle) > max(1, width * 0.18):
            return False
    # Opposing curved ends AND long axial walls. A rectangle or pipe loop with
    # square corners is insufficient, and vessel head faces fail the wall gate.
    diagonal = [
        p
        for _, a, z in symbol.segments
        if abs(a[0] - z[0]) > 0.3 and abs(a[1] - z[1]) > 0.3
        for p in (a, z)
    ]
    walls = [
        s
        for s in symbol.segments
        if abs(s[1][axis] - s[2][axis]) > length * 0.4
        and abs(s[1][1 - axis] - s[2][1 - axis]) < max(1, width * 0.03)
    ]
    return (
        bool(diagonal)
        and len(walls) >= 2
        and min(p[axis] for p in diagonal) < start + length * 0.2
        and max(p[axis] for p in diagonal) > start + length * 0.8
    )


def _arrow(points: list[tuple], box: BBox, tolerance: float) -> bool:
    corners = _corners(points, tolerance)
    if len(corners) != 5 or max(box.w, box.h) < min(box.w, box.h) * 1.8:
        return False
    # One axial tip centred on the short dimension, a flat tail, two shoulders.
    axis = int(box.h > box.w)
    lo, hi = (box.y, box.y2) if axis else (box.x, box.x2)
    middle = box.x + box.w / 2 if axis else box.y + box.h / 2
    short = min(box.w, box.h)
    for sign in (1, -1):
        tip = max(corners, key=lambda p: sign * p[axis])
        tail = lo if sign == 1 else hi
        if abs(tip[1 - axis] - middle) > short * 0.12:
            continue
        index = corners.index(tip)
        shoulders = (corners[index - 1], corners[(index + 1) % len(corners)])
        if (
            sum(abs(p[axis] - tail) <= tolerance for p in corners) == 2
            and abs(shoulders[0][axis] - shoulders[1][axis]) <= tolerance
            and abs(shoulders[0][1 - axis] - shoulders[1][1 - axis]) >= short * 0.85
            and abs(shoulders[0][axis] - tail) >= (hi - lo) * 0.55
        ):
            return True
    return False


def _fragment_circles(page: PageEvidence) -> list[NativeSymbol]:
    """Recover complete circles split into short chords, ignoring long pipe spurs."""
    scale = min(page.width, page.height)
    adjacency: dict[tuple, set] = defaultdict(set)
    sources: dict[frozenset, set] = defaultdict(set)
    for path in sorted(page.paths, key=lambda p: p.id):
        if path.origin != "pdf_vector" or path.visual_style in {"dashed", "dotted", "dash_dot"}:
            continue
        vs = path_vertices(path)
        if not vs:
            continue
        if path.closed and vs[-1] != vs[0]:
            vs += vs[:1]
        for a, b in zip(vs, vs[1:], strict=False):
            if not scale * 0.00015 < math.dist(a, b) < scale * 0.0052:
                continue
            a, b = tuple(round(v) for v in a), tuple(round(v) for v in b)
            if a == b:
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
            sources[frozenset((a, b))].add(segment_key(path.id, a, b))
    # Peel dangling short signal/pipe stubs before testing closed circles.
    # This removes edges; it never invents a closure across a partial arc.
    original_degree = {p: len(neighbors) for p, neighbors in adjacency.items()}
    leaves = [p for p in adjacency if len(adjacency[p]) < 2]
    while leaves:
        p = leaves.pop()
        for neighbor in list(adjacency[p]):
            adjacency[neighbor].discard(p)
            if len(adjacency[neighbor]) == 1:
                leaves.append(neighbor)
        adjacency[p].clear()
    seen, out = set(), []
    for start in sorted(adjacency):
        if start in seen:
            continue
        todo, component = [start], set()
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            component.add(p)
            todo.extend(adjacency[p] - seen)
        points = sorted(component)
        if len(points) < 12:
            continue
        b = _box(points)
        if not scale * 0.004 <= min(b.w, b.h) <= scale * 0.05 or not _round(points, b):
            continue
        if any(original_degree[p] != 2 for p in component) and not any(
            _INSTRUMENT_TEXT.fullmatch(t.text.strip())
            and _inside(t.bbox, b, max(2, min(b.w, b.h) * 0.08))
            for t in page.text_spans
        ):
            # A branched circular perimeter also occurs inside valve seats
            # and filled flow arrows. Require internal function text before
            # promoting that recovered sub-contour to an independent glyph.
            continue
        # Require a closed cycle: fitted points from a partial arc are not a glyph.
        if any(len(adjacency[p]) != 2 for p in component):
            continue
        segments = set().union(
            *(sources[frozenset((a, z))] for a in component for z in adjacency[a])
        )
        sid = stable_evidence_id("sym", page.page_index, sorted(segments))
        out.append(NativeSymbol(sid, points, segments, b, math.pi * b.w * b.h / 4))
    return out


def _closed_paths(page: PageEvidence) -> list[NativeSymbol]:
    """Keep explicit PDF contours even when touching strokes confuse face walks."""
    out = []
    for path in page.paths:
        if path.origin != "pdf_vector" or path.visual_style in {"dashed", "dotted", "dash_dot"}:
            continue
        points = path_vertices(path)
        if len(points) < 3 or not (
            path.closed or path.primitive in {"rect", "quad"} or points[0] == points[-1]
        ):
            continue
        if points[0] == points[-1]:
            points = points[:-1]
        pairs = list(zip(points, points[1:] + points[:1], strict=True))
        area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in pairs)) / 2
        segments = {segment_key(path.id, a, b) for a, b in pairs}
        out.append(
            NativeSymbol(
                stable_evidence_id("sym", page.page_index, sorted(segments)),
                points,
                segments,
                _box(points),
                area,
            )
        )
    return out


def _instrument_frame(symbol: NativeSymbol, scale: float) -> bool:
    b = symbol.bbox
    if not (
        scale * 0.004 <= min(b.w, b.h)
        and max(b.w, b.h) <= scale * 0.06
        and 0.75 <= b.w / b.h <= 1.33
    ):
        return False
    corners = _corners(symbol.points, max(0.3, scale * 0.0001))
    if len(corners) != 4:
        return False
    if symbol.area / (b.w * b.h) > 0.9:
        return True
    # A diamond requires four equal sides and vertices at the axial extrema.
    lengths = [math.dist(a, z) for a, z in zip(corners, corners[1:] + corners[:1], strict=True)]
    tips = [
        (b.x + b.w / 2, b.y),
        (b.x2, b.y + b.h / 2),
        (b.x + b.w / 2, b.y2),
        (b.x, b.y + b.h / 2),
    ]
    return min(lengths) >= max(lengths) * 0.9 and all(
        min(math.dist(p, t) for p in corners) <= max(1, min(b.w, b.h) * 0.04) for t in tips
    )


def _crossed_valves(page: PageEvidence, symbols: list[NativeSymbol]) -> list[tuple[str, BBox, set]]:
    """An X plus two native end bars is a bow-tie body, even without a centre vertex.

    This does not split piping at arbitrary crossings. Both closing bars and
    symmetric diagonals must exist locally; a bare X or pair of arrows fails.
    """
    scale = min(page.width, page.height)
    tolerance = max(0.6, scale * 0.0002)
    rows = []
    for path in sorted(page.paths, key=lambda p: p.id):
        if path.origin != "pdf_vector" or path.visual_style in {"dashed", "dotted", "dash_dot"}:
            continue
        points = path_vertices(path)
        if (
            points
            and (path.closed or path.primitive in {"rect", "quad"})
            and points[-1] != points[0]
        ):
            points += points[:1]
        for a, b in zip(points, points[1:], strict=False):
            if scale * 0.002 <= math.dist(a, b) <= scale * 0.07:
                rows.append((a, b, segment_key(path.id, a, b)))
    endpoints = defaultdict(list)
    centers = defaultdict(list)

    def key(point):
        return tuple(math.floor(v / tolerance) for v in point)

    def nearby(grid, point):
        x, y = key(point)
        return [i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in grid.get((x + dx, y + dy), [])]

    for i, (a, b, _) in enumerate(rows):
        endpoints[key(a)].append(i)
        endpoints[key(b)].append(i)
        centers[key(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))].append(i)

    def bar(a, b):
        return {
            rows[i][2]
            for i in nearby(endpoints, a)
            if min(
                max(math.dist(a, rows[i][0]), math.dist(b, rows[i][1])),
                max(math.dist(a, rows[i][1]), math.dist(b, rows[i][0])),
            )
            <= tolerance
        }

    out = []
    for i, (a, b, source) in enumerate(rows):
        midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        for j in nearby(centers, midpoint):
            if j <= i:
                continue
            c, d, other = rows[j]
            length = math.dist(a, b)
            alignment = ((b[0] - a[0]) * (d[0] - c[0]) + (b[1] - a[1]) * (d[1] - c[1])) / (
                length * math.dist(c, d)
            )
            if not 0.8 <= length / math.dist(c, d) <= 1.25 or abs(alignment) > 0.85:
                continue
            if math.dist(midpoint, ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)) > tolerance:
                continue
            box = _box([a, b, c, d])
            if not 0.35 <= box.w / box.h <= 2.9:
                continue
            pairings = [((a, c), (b, d)), ((a, d), (b, c))]
            # Closing the short ends yields opposed triangles, not top/bottom
            # walls of a rectangle with an internal X.
            ends = min(pairings, key=lambda pairs: sum(math.dist(*pair) for pair in pairs))
            caps = [bar(*pair) for pair in ends]
            other_ends = next(pairs for pairs in pairings if pairs != ends)
            if all(caps) and not all(bar(*pair) for pair in other_ends):
                out.append(("valve_body", box, {source, other} | caps[0] | caps[1]))
    # Some exporters interrupt the four diagonal legs with an explicit round
    # seat. Require that seat and four contacted legs, never bridge a blank gap.
    for seat in symbols:
        b = seat.bbox
        if max(b.w, b.h) > scale * 0.015 or not _round(seat.points, b):
            continue
        center = (b.x + b.w / 2, b.y + b.h / 2)
        radius = max(b.w, b.h) / 2
        wings = []
        for a, z, source in rows:
            length = math.dist(a, z)
            midpoint = ((a[0] + z[0]) / 2, (a[1] + z[1]) / 2)
            distance = math.dist(midpoint, center)
            if not radius * 1.5 <= length <= radius * 7 or not radius * 2 <= distance <= radius * 8:
                continue
            radial = (midpoint[0] - center[0], midpoint[1] - center[1])
            if (
                abs((z[0] - a[0]) * radial[0] + (z[1] - a[1]) * radial[1])
                > length * distance * 0.15
            ):
                continue
            legs = []
            for endpoint in (a, z):
                connected = set()
                for i in nearby(endpoints, endpoint):
                    c, d, sid = rows[i]
                    for outer, inner in ((c, d), (d, c)):
                        if (
                            math.dist(endpoint, outer) <= tolerance
                            and radius * 0.25
                            <= math.dist(inner, center)
                            <= radius * 1.35 + tolerance
                            and math.dist(outer, inner) >= length * 0.5
                        ):
                            connected.add(sid)
                legs.append(connected)
            if all(legs):
                wings.append((a, z, radial, length, distance, {source} | legs[0] | legs[1]))
        for i, a in enumerate(wings):
            for z in wings[i + 1 :]:
                if (
                    not 0.8 <= a[3] / z[3] <= 1.25
                    or not 0.7 <= a[4] / z[4] <= 1.4
                    or a[2][0] * z[2][0] + a[2][1] * z[2][1] > -0.95 * a[4] * z[4]
                ):
                    continue
                out.append(
                    ("valve_body", _box([a[0], a[1], z[0], z[1]]), a[5] | z[5] | seat.segments)
                )
    return out


def _contacted_frame(page: PageEvidence, symbol: NativeSymbol, scale: float) -> bool:
    """Bounded rectangular proposals need external ink terminating at a side.

    A border, empty table cell, or an unbroken pipe passing through the box
    does not qualify. Geometry still needs the model's semantic confirmation.
    """
    b = symbol.bbox
    if not (
        scale * 0.008 <= min(b.w, b.h)
        and max(b.w, b.h) <= scale * 0.25
        and symbol.area >= b.w * b.h * 0.95
        and len(_corners(symbol.points, 0.6)) == 4
    ):
        return False
    tolerance = max(0.6, scale * 0.0002)
    for path in page.paths:
        if path.origin != "pdf_vector" or path.id in symbol.path_ids or path.closed:
            continue
        points = path_vertices(path)
        if len(points) < 2:
            continue
        for end, other in [(points[0], points[-1]), (points[-1], points[0])]:
            if (
                b.x - tolerance <= other[0] <= b.x2 + tolerance
                and b.y - tolerance <= other[1] <= b.y2 + tolerance
            ):
                continue
            if math.dist(end, other) < min(b.w, b.h) * 0.25:
                continue
            if (
                min(abs(end[0] - b.x), abs(end[0] - b.x2)) <= tolerance
                and b.y + b.h * 0.1 < end[1] < b.y2 - b.h * 0.1
                or min(abs(end[1] - b.y), abs(end[1] - b.y2)) <= tolerance
                and b.x + b.w * 0.1 < end[0] < b.x2 - b.w * 0.1
            ):
                return True
    return False


def _open_inline_valves(page):
    """Propose a diagonal between two end bars, with external pipe contacts.

    This is only a glyph proposal. Neither service text nor a reference creates
    a valve. Two-sided pipe contacts distinguish it from a bare slash/break mark.
    Work in both orientations and exclude the pipe from the symbol bounds.
    """
    scale = min(page.width, page.height)
    tolerance = max(1.0, scale * 0.0003)
    segments = []
    for path in page.paths:
        if path.origin != "pdf_vector" or path.visual_style != "solid":
            continue
        points = path_vertices(path)
        for a, b in zip(points, points[1:], strict=False):
            if math.dist(a, b) >= scale * 0.001:
                segments.append((a, b, segment_key(path.id, a, b)))
    result, seen = [], set()
    for vertical in (False, True):

        def xy(p, vertical=vertical):
            return (p[1], p[0]) if vertical else p

        rows = [(xy(a), xy(b), key) for a, b, key in segments]
        bars = []
        for a, b, key in rows:
            if abs(a[0] - b[0]) <= tolerance and scale * 0.002 <= abs(a[1] - b[1]) <= scale * 0.025:
                bars.append(((a[0] + b[0]) / 2, min(a[1], b[1]), max(a[1], b[1]), key))
        # Spatial lookup avoids comparing every path against every end bar.
        bins = defaultdict(list)
        for bar in bars:
            bins[round(bar[0] / tolerance)].append(bar)

        def at(x, bins=bins):
            k = round(x / tolerance)
            return [
                bar for i in (k - 1, k, k + 1) for bar in bins[i] if abs(bar[0] - x) <= tolerance
            ]

        for a, b, diagonal in rows:
            if a[0] > b[0]:
                a, b = b, a
            w, h = b[0] - a[0], abs(b[1] - a[1])
            if not (scale * 0.002 <= h <= scale * 0.025 and 0.8 <= w / max(h, 1) <= 3):
                continue
            for left in at(a[0]):
                for right in at(b[0]):
                    if max(abs(left[1] - right[1]), abs(left[2] - right[2])) > tolerance:
                        continue
                    if (
                        min(
                            abs(a[1] - left[1]) + abs(b[1] - right[2]),
                            abs(a[1] - left[2]) + abs(b[1] - right[1]),
                        )
                        > 2 * tolerance
                    ):
                        continue
                    # Closed rectangular panels are not this open inline glyph.
                    if any(
                        abs(c[1] - d[1]) <= tolerance
                        and min(abs(c[1] - left[1]), abs(c[1] - left[2])) <= tolerance
                        and abs(min(c[0], d[0]) - left[0]) <= tolerance
                        and abs(max(c[0], d[0]) - right[0]) <= tolerance
                        for c, d, _ in rows
                    ):
                        continue
                    y = (left[1] + left[2]) / 2

                    def contact(x, direction, y=y, h=h, rows=rows):
                        return any(
                            abs(c[1] - y) <= tolerance
                            and abs(d[1] - y) <= tolerance
                            and min(abs(c[0] - x), abs(d[0] - x)) <= tolerance
                            and max(direction * (c[0] - x), direction * (d[0] - x)) >= h * 0.35
                            for c, d, _ in rows
                        )

                    if not contact(left[0], -1) or not contact(right[0], 1):
                        continue
                    keys = {left[3], right[3], diagonal}
                    signature = frozenset(keys)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    corners = [(left[0], left[1]), (right[0], right[2])]
                    result.append(("open_inline_valve", _box([xy(p) for p in corners]), keys))
    return result


def symbol_candidates(page: PageEvidence) -> list[SymbolCandidate]:
    """Propose separately located native glyphs without consulting detections."""
    if page.is_scanned or page.role != "pid":
        return []
    scale = min(page.width, page.height)
    symbols = native_symbols(page)
    rows: list[tuple[str, BBox, set]] = [*_crossed_valves(page, symbols), *_open_inline_valves(page)]
    byvertex: dict[tuple, list] = defaultdict(list)
    for s in symbols:
        points = _corners(s.points, max(0.25, scale * 0.00006))
        if (
            len(points) == 3
            and scale * 0.001 <= min(s.bbox.w, s.bbox.h)
            and max(s.bbox.w, s.bbox.h) <= scale * 0.025
        ):
            for p in points:
                byvertex[tuple(round(v) for v in p)].append(s)
    seen = set()
    for shared, neighbors in sorted(byvertex.items()):
        for i, a in enumerate(neighbors):
            for b in neighbors[i + 1 :]:
                pair = tuple(sorted((a.id, b.id)))
                if pair in seen or a.bbox.iou(b.bbox) > 0.05 or not 0.5 <= a.area / b.area <= 2:
                    continue
                ca = (a.bbox.x + a.bbox.w / 2 - shared[0], a.bbox.y + a.bbox.h / 2 - shared[1])
                cb = (b.bbox.x + b.bbox.w / 2 - shared[0], b.bbox.y + b.bbox.h / 2 - shared[1])
                if ca[0] * cb[0] + ca[1] * cb[1] > -0.8 * math.hypot(*ca) * math.hypot(*cb):
                    continue
                box = _box(a.points + b.points)
                if max(box.w, box.h) > 3 * min(box.w, box.h):
                    continue
                seen.add(pair)
                rows.append(("valve_body", box, a.segments | b.segments))
    for s in [*symbols, *_fragment_circles(page), *_closed_paths(page)]:
        b = s.bbox
        contained = [t for t in page.text_spans if _inside(t.bbox, b, max(2, min(b.w, b.h) * 0.08))]
        # A square is meaningful here only with instrument/function text. Tag
        # callout boxes such as QV07 must never become valve body candidates.
        instrument_text = any(_INSTRUMENT_TEXT.fullmatch(t.text.strip()) for t in contained)
        valve_callout = any(
            re.fullmatch(r"(?:[PFHTL]?SV|[PFTL]V|QV)(?:[- ]?\d+)?", t.text.strip())
            for t in contained
        )
        shape = None
        if valve_callout:
            continue
        if scale * 0.004 <= min(b.w, b.h) <= scale * 0.05 and _round(s.points, b):
            shape = "round_symbol"
        elif instrument_text and _instrument_frame(s, scale):
            shape = "instrument_frame"
        elif _arrow(s.points, b, max(0.5, scale * 0.0002)):
            shape = "continuation_arrow"
        elif _capsule(s, scale):
            shape = "capsule_body"
        elif _contacted_frame(page, s, scale):
            shape = "connected_frame"
        if shape:
            rows.append((shape, b, s.segments))
    # Same physical circle/frame may have several native faces. Consolidate
    # only nearly identical extents; never deduplicate by printed tag or proximity.
    # Head chords can form a rectangular interior panel. Shared native walls
    # establish that it belongs to the larger capsule; containment alone cannot.
    rows = [
        r
        for r in rows
        if not (
            r[0] == "connected_frame"
            and any(
                v[0] == "capsule_body"
                and _inside(r[1], v[1])
                and sum(math.dist(s[1], s[2]) for s in r[2] & v[2])
                >= 0.45 * sum(math.dist(s[1], s[2]) for s in r[2])
                for v in rows
            )
        )
    ]
    # An X contains two diagonals, but remains one bow-tie proposal.
    rows = [r for r in rows if not (r[0] == "open_inline_valve" and
            any(v[0] == "valve_body" and r[1].iou(v[1]) > .82 for v in rows))]
    # A contacted seat is part of its valve body, not another symbol instance.
    rows = [
        r
        for r in rows
        if not (r[0] == "round_symbol" and any(v[0] == "valve_body" and r[2] <= v[2] for v in rows))
    ]
    result = []
    for shape, b, segments in sorted(rows, key=lambda r: (-r[1].w * r[1].h, r[1].y, r[1].x, r[0])):
        compatible = (
            {"round_symbol", "instrument_frame", "connected_frame"}
            if shape in {"round_symbol", "instrument_frame", "connected_frame"}
            else {shape}
        )
        same = next((c for c in result if c[0] in compatible and c[1].iou(b) > 0.82), None)
        if same:
            same[2].update(segments)
            continue
        result.append([shape, b, set(segments)])
    out = []
    for shape, b, segments in result:
        texts = sorted(
            (t for t in page.text_spans if _inside(t.bbox, b, max(2, min(b.w, b.h) * 0.08))),
            key=lambda t: (t.bbox.y, t.bbox.x, t.id),
        )
        out.append(
            SymbolCandidate(
                id=stable_evidence_id("candidate", page.page_index, shape, sorted(segments)),
                page_index=page.page_index,
                bbox=b,
                shape=shape,
                source_path_ids=sorted({s[0] for s in segments}),
                source_text_ids=[t.id for t in texts],
                text=[t.text for t in texts],
            )
        )
    return sorted(out, key=lambda c: (c.bbox.y, c.bbox.x, c.id))
