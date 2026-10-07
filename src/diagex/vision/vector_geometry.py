"""Native symbol contours and arrow evidence, independent of model decisions.

PDF paths are immutable source evidence. A contour owns only its constituent
strokes; a bounding box is never a mask for everything drawn inside it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from diagex.vision.evidence import PageEvidence, PathEvidence, stable_evidence_id
from diagex.vision.models import BBox, ReconciledNode

Point = tuple[float, float]


def path_vertices(path: PathEvidence) -> list[Point]:
    if path.primitive == "curve" and len(path.points) == 4:
        # Native PDF curves contain cubic control points, not pipe vertices.
        a, b, c, d = path.points
        steps = max(
            8,
            min(
                64,
                math.ceil(
                    sum(math.dist(x, y) for x, y in zip(path.points, path.points[1:], strict=False))
                    / 8
                ),
            ),
        )
        return [
            tuple(
                (1 - t) ** 3 * a[k]
                + 3 * (1 - t) ** 2 * t * b[k]
                + 3 * (1 - t) * t * t * c[k]
                + t**3 * d[k]
                for k in (0, 1)
            )
            for t in (i / steps for i in range(steps + 1))
        ]
    return [(float(x), float(y)) for x, y in path.points]


def inside(point: Point, polygon: list[Point]) -> bool:
    x, y = point
    result = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=False):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            result = not result
    return result


def intersections(a: Point, b: Point, polygon: list[Point]) -> list[float]:
    out = []
    dx, dy = b[0] - a[0], b[1] - a[1]
    for c, d in zip(polygon, polygon[1:] + polygon[:1], strict=False):
        ex, ey = d[0] - c[0], d[1] - c[1]
        det = dx * ey - dy * ex
        if abs(det) < 1e-9:
            continue
        t = ((c[0] - a[0]) * ey - (c[1] - a[1]) * ex) / det
        u = ((c[0] - a[0]) * dy - (c[1] - a[1]) * dx) / det
        if -1e-7 <= t <= 1 + 1e-7 and -1e-7 <= u <= 1 + 1e-7:
            out.append(max(0.0, min(1.0, t)))
    return sorted(set(round(t, 8) for t in out))


def box_polygon(box: BBox) -> list[Point]:
    return [(box.x, box.y), (box.x2, box.y), (box.x2, box.y2), (box.x, box.y2)]


SCENE_VERSION = "2.0.0"


def segment_key(path_id: str, a: Point, b: Point) -> tuple:
    return (path_id, *sorted((tuple(round(v, 5) for v in a), tuple(round(v, 5) for v in b))))


@dataclass
class NativeSymbol:
    id: str
    points: list[Point]
    segments: set[tuple]
    bbox: BBox
    area: float

    @property
    def path_ids(self) -> list[str]:
        return sorted({s[0] for s in self.segments})


@dataclass
class SymbolContour:
    node_id: str
    points: list[Point]
    path_ids: list[str]
    symbol_id: str = ""
    segments: set[tuple] = field(default_factory=set)
    basis: str = "native_contour"

    @property
    def bbox(self) -> BBox:
        return _extent(self.points)


def _extent(points: list[Point]) -> BBox:
    xs, ys = zip(*points, strict=True)
    return BBox(
        x=round(min(xs)),
        y=round(min(ys)),
        w=max(1, round(max(xs) - min(xs))),
        h=max(1, round(max(ys) - min(ys))),
    )


def native_symbols(page: PageEvidence) -> list[NativeSymbol]:
    """Enumerate native faces once across a page, independently of detector crops.

    Faces are geometric hypotheses, not automatically equipment: a real piping
    loop is retained unless corroborated by symbol observations. Source strokes
    are owned individually, including when a PDF path also contains a pipe.
    """
    tolerance = max(0.6, min(page.width, page.height) * 0.00018)
    vertices: list[Point] = []
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    def vertex(point):
        key = (math.floor(point[0] / tolerance), math.floor(point[1] / tolerance))
        for x in range(key[0] - 1, key[0] + 2):
            for y in range(key[1] - 1, key[1] + 2):
                for i in grid.get((x, y), []):
                    if math.dist(point, vertices[i]) <= tolerance:
                        return i
        i = len(vertices)
        vertices.append(point)
        grid[key].append(i)
        return i

    arcs = defaultdict(list)
    geometry = []
    for path in sorted(page.paths, key=lambda p: p.id):
        if path.origin != "pdf_vector" or not path.points:
            continue
        if path.visual_style in {"dashed", "dotted", "dash_dot"}:
            continue
        points = path_vertices(path)
        if (path.closed or path.primitive in {"rect", "quad"}) and points[-1] != points[0]:
            points.append(points[0])
        for a, b in zip(points, points[1:], strict=False):
            left, right = vertex(a), vertex(b)
            if left == right:
                continue
            e = len(geometry)
            key = segment_key(path.id, a, b)
            geometry.extend([(left, right, key), (right, left, key)])
            arcs[left].append(e)
            arcs[right].append(e + 1)
    for rows in arcs.values():
        rows.sort(
            key=lambda e: math.atan2(
                vertices[geometry[e][1]][1] - vertices[geometry[e][0]][1],
                vertices[geometry[e][1]][0] - vertices[geometry[e][0]][0],
            )
        )
    visited, out = set(), {}
    for start in range(len(geometry)):
        if start in visited:
            continue
        current, walk = start, []
        while current not in visited:
            visited.add(current)
            walk.append(current)
            rows = arcs[geometry[current][1]]
            current = rows[(rows.index(current ^ 1) - 1) % len(rows)]
        if current != start:
            continue
        # Remove bridges traversed in both directions, such as dangling nozzles.
        members = set(walk)
        walk = [e for e in walk if e ^ 1 not in members]
        # Exterior walks around two symbols connected by a pipe can visit both
        # loops. Split at repeated vertices/discontinuities, never join them by
        # an artificial polygon edge after removing the bridge.
        cycles, chain, positions = [], [], {}
        for e in walk:
            left, right, _ = geometry[e]
            if chain and geometry[chain[-1]][1] != left:
                chain, positions = [], {}
            positions.setdefault(left, len(chain))
            chain.append(e)
            if right in positions:
                cut = positions[right]
                cycles.append(chain[cut:])
                chain = chain[:cut]
                positions = {geometry[v][0]: i for i, v in enumerate(chain)}
        for cycle in cycles:
            if len(cycle) < 3:
                continue
            points = [vertices[geometry[e][0]] for e in cycle]
            area = (
                abs(
                    sum(
                        a[0] * b[1] - b[0] * a[1]
                        for a, b in zip(points, points[1:] + points[:1], strict=False)
                    )
                )
                / 2
            )
            box = _extent(points)
            if min(box.w, box.h) < 4 or area < box.w * box.h * 0.2:
                continue
            if box.w > page.width * 0.7 or box.h > page.height * 0.85:
                continue
            segments = {geometry[e][2] for e in cycle}
            sid = stable_evidence_id("sym", page.page_index, sorted(segments))
            out[sid] = NativeSymbol(sid, points, segments, box, area)
    return sorted(out.values(), key=lambda s: (s.bbox.y, s.bbox.x, s.id))


def select_symbol(
    box: BBox, symbols: list[NativeSymbol], *, expandable: bool, min_cross_section: float = 0.65
) -> NativeSymbol | None:
    choices = []
    for symbol in symbols:
        b = symbol.bbox
        iw = max(0, min(box.x2, b.x2) - max(box.x, b.x))
        ih = max(0, min(box.y2, b.y2) - max(box.y, b.y))
        cover = iw * ih / max(1, box.w * box.h)
        cross_section = max(min(box.w, b.w) / max(box.w, b.w), min(box.h, b.h) / max(box.h, b.h))
        ratio = box.w * box.h / max(1, b.w * b.h)
        if b.iou(box) >= 0.3 or (
            expandable
            and cover >= 0.68
            and cross_section >= min_cross_section
            and 0.08 <= ratio <= 1.8
        ):
            choices.append(symbol)
    if not choices:
        return None
    # A vessel with head chords has smaller enclosed faces. Prefer its complete
    # exterior when the observation supports that extent, not an internal panel.
    choices.sort(key=lambda s: (s.area if expandable else s.bbox.iou(box), s.id), reverse=True)
    best = choices[0]
    if any(s.bbox.iou(best.bbox) < 0.2 and s.area >= best.area * 0.8 for s in choices[1:]):
        return None
    return best


def _hull(points: list[Point]) -> list[Point]:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = sorted(set(points))
    lower, upper = [], []
    for seq, out in [(points, lower), (list(reversed(points)), upper)]:
        for p in seq:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
    return lower[:-1] + upper[:-1]


def open_symbol(page: PageEvidence, node: ReconciledNode) -> SymbolContour | None:
    """Corroborate open glyphs with native non-collinear strokes, not a pipe alone."""
    box = node.bbox_global
    padding = max(2, min(box.w, box.h) * 0.12)
    segments = set()
    for path in page.paths:
        if path.origin != "pdf_vector" or path.visual_style in {"dashed", "dotted", "dash_dot"}:
            continue
        points = path_vertices(path)
        for a, b in zip(points, points[1:], strict=False):
            if all(
                box.x - padding <= p[0] <= box.x2 + padding
                and box.y - padding <= p[1] <= box.y2 + padding
                for p in (a, b)
            ):
                segments.add(segment_key(path.id, a, b))
    points = _hull([p for s in segments for p in s[1:]])
    if len(points) < 3:
        return None
    extent = _extent(points)
    if extent.iou(box) < 0.45 or min(extent.w / box.w, extent.h / box.h) < 0.45:
        return None
    sid = stable_evidence_id("sym", page.page_index, sorted(segments))
    return SymbolContour(
        node.id, points, sorted({s[0] for s in segments}), sid, segments, "native_open_symbol"
    )


def _repair_displaced_symbol(
    page: PageEvidence,
    node: ReconciledNode,
    nodes: list[ReconciledNode],
    symbols: list[NativeSymbol],
) -> NativeSymbol | None:
    """Recover a small localization error only with a unique, contacted native glyph.

    Similar size, overlap and a bounded translation are all required. A neighbor's
    well-localized observation reserves its glyph; proximity alone never wins.
    """
    box = node.bbox_global
    choices = []
    for symbol in symbols:
        b = symbol.bbox
        limit = max(3, min(box.w, box.h) * 0.85)
        if not (
            b.iou(box) > 0
            and 0.75 <= b.w / max(1, box.w) <= 1.33
            and 0.75 <= b.h / max(1, box.h) <= 1.33
            and max(abs(b.x + b.w / 2 - box.x - box.w / 2), abs(b.y + b.h / 2 - box.y - box.h / 2))
            <= limit
        ):
            continue
        if any(
            n.id != node.id and n.page_index == node.page_index and n.bbox_global.iou(b) >= 0.3
            for n in nodes
        ):
            continue
        # Require external ink to end on the actual contour, not merely enter
        # its box. Ignore the contour itself and strokes continuing inside it.
        contacted = False
        for path in page.paths:
            if path.origin != "pdf_vector" or path.visual_style != "solid":
                continue
            points = path_vertices(path)
            for a, z in zip(points, points[1:], strict=False):
                if segment_key(path.id, a, z) in symbol.segments or math.dist(a, z) < 8:
                    continue
                hits = intersections(a, z, symbol.points)
                middle = ((a[0] + z[0]) / 2, (a[1] + z[1]) / 2)
                if (
                    hits
                    and (hits[0] < 1e-5 or hits[-1] > 1 - 1e-5)
                    and not inside(middle, symbol.points)
                ):
                    contacted = True
                    break
            if contacted:
                break
        if contacted:
            choices.append(symbol)
    return choices[0] if len(choices) == 1 else None


def symbol_contours(
    page: PageEvidence, nodes: list[ReconciledNode], *, symbols: list[NativeSymbol] | None = None
) -> dict[str, SymbolContour]:
    symbols = native_symbols(page) if symbols is None else symbols
    out = {}
    for node in sorted(nodes, key=lambda n: n.id):
        if node.kind == "opc":
            # A continuation marker can have a real closed native boundary
            # (e.g. a five-sided arrow frame). Its local pipe contact is no
            # less physical than an equipment port. Requiring a uniquely
            # matched face avoids turning a label box or open arrow hull into
            # connectivity; this establishes neither flow nor a remote pair.
            symbol = select_symbol(node.bbox_global, symbols, expandable=False)
            if symbol:
                out[node.id] = SymbolContour(
                    node.id, symbol.points, symbol.path_ids, symbol.id, symbol.segments
                )
            continue
        attrs = node.attributes
        expandable = (
            node.kind == "equipment"
            and not attrs.get("valve_type")
            and attrs.get("equipment_class") not in {"valve", "actuator"}
        )
        symbol = select_symbol(
            node.bbox_global,
            symbols,
            expandable=expandable,
            min_cross_section=0.45 if attrs.get("equipment_class") == "vessel" else 0.65,
        )
        if symbol is None:
            symbol = _repair_displaced_symbol(page, node, nodes, symbols)
        if symbol:
            out[node.id] = SymbolContour(
                node.id, symbol.points, symbol.path_ids, symbol.id, symbol.segments
            )
        contour = open_symbol(page, node)
        if contour and (
            not symbol
            or (
                attrs.get("valve_type")
                and contour.bbox.iou(node.bbox_global) > symbol.bbox.iou(node.bbox_global) + 0.15
            )
        ):
            out[node.id] = contour
        if node.kind == "instrument" and node.id in out:
            out[node.id] = _instrument_frame(page, node, out[node.id])
    return out


def _instrument_frame(
    page: PageEvidence, node: ReconciledNode, contour: SymbolContour
) -> SymbolContour:
    """Own a corroborated enclosing instrument frame, never all ink in its box.

    CAD exporters often split a circle-in-square into separate paths. Selecting
    the circle alone leaves the square's sides available as false dashed runs.
    Require four complete sides aligned with the observed glyph's extent.
    """
    box = node.bbox_global
    if not 0.7 <= box.w / max(1, box.h) <= 1.4:
        return contour
    tol = max(2, min(box.w, box.h) * 0.07)
    sides: dict[str, list[tuple]] = {s: [] for s in ("left", "right", "top", "bottom")}
    for path in page.paths:
        if path.origin != "pdf_vector" or path.visual_style in {"dashed", "dotted", "dash_dot"}:
            continue
        points = path_vertices(path)
        if path.closed or path.primitive in {"rect", "quad"}:
            points += points[:1]
        for a, b in zip(points, points[1:], strict=False):
            key = segment_key(path.id, a, b)
            if (
                abs(a[0] - b[0]) <= 1
                and abs(min(a[1], b[1]) - box.y) <= tol
                and abs(max(a[1], b[1]) - box.y2) <= tol
            ):
                for name, x in (("left", box.x), ("right", box.x2)):
                    if abs(a[0] - x) <= tol:
                        sides[name].append(key)
            if (
                abs(a[1] - b[1]) <= 1
                and abs(min(a[0], b[0]) - box.x) <= tol
                and abs(max(a[0], b[0]) - box.x2) <= tol
            ):
                for name, y in (("top", box.y), ("bottom", box.y2)):
                    if abs(a[1] - y) <= tol:
                        sides[name].append(key)
    if not all(len(values) == 1 for values in sides.values()):
        return contour
    frame = {values[0] for values in sides.values()}
    extent = _extent([p for s in frame for p in s[1:]])
    # The observed native contour and frame must describe the same glyph.
    if extent.iou(contour.bbox) < 0.7:
        return contour
    owned = contour.segments | frame
    return SymbolContour(
        node.id,
        box_polygon(extent),
        sorted({s[0] for s in owned}),
        stable_evidence_id("sym", page.page_index, sorted(owned)),
        owned,
        contour.basis,
    )


@dataclass
class VectorArrow:
    tip: Point
    base: Point
    path_ids: list[str]
    segments: set[tuple] = field(default_factory=set)


def vector_arrows(page: PageEvidence, nodes: list[ReconciledNode]) -> list[VectorArrow]:
    """Recognise isolated native triangular arrowheads, not valve triangles.

    CAD exporters frequently split the three sides into separate primitives.
    Grouping by source drawing alone is insufficient: some drawings contain
    an entire page. Small connected line components provide a bounded test.
    """
    maximum = max(32.0, min(page.width, page.height) * 0.012)
    rows = []
    for path in page.paths:
        if path.origin != "pdf_vector" or path.primitive not in {"line", "quad"}:
            continue
        points = path_vertices(path)
        for a, b in zip(points, points[1:], strict=False):
            if 2 <= math.dist(a, b) <= maximum:
                rows.append((a, b, path.id))

    # Quantisation is used only for arrow shape discovery, never pipe joining.
    def key(p: Point) -> tuple[int, int]:
        return round(p[0]), round(p[1])

    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    sources: dict[frozenset, set[str]] = defaultdict(set)
    for a, b, pid in rows:
        a, b = key(a), key(b)
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
            sources[frozenset((a, b))].add(pid)
    out, seen = [], set()
    for a in sorted(adjacency):
        for b in sorted(adjacency[a]):
            for c in sorted(adjacency[a] & adjacency[b]):
                triangle = tuple(sorted((a, b, c)))
                if triangle in seen:
                    continue
                seen.add(triangle)
                # The short side is the base of an elongated isosceles arrow.
                sides = sorted(
                    [
                        (math.dist(a, b), a, b, c),
                        (math.dist(b, c), b, c, a),
                        (math.dist(c, a), c, a, b),
                    ]
                )
                width, p, q, tip = sides[0]
                base = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                length = math.dist(base, tip)
                if (
                    length < width * 1.1
                    or length > width * 6
                    or abs(sides[1][0] - sides[2][0]) > max(2.0, length * 0.15)
                ):
                    continue
                if any(
                    n.bbox_global.x - 3 <= tip[0] <= n.bbox_global.x2 + 3
                    and n.bbox_global.y - 3 <= tip[1] <= n.bbox_global.y2 + 3
                    for n in nodes
                ):
                    continue
                ids = set().union(
                    *(sources[frozenset((x, y))] for x, y in [(a, b), (b, c), (c, a)])
                )
                triangle_sides = {frozenset((x, y)) for x, y in [(a, b), (b, c), (c, a)]}
                segments = {
                    segment_key(pid, x, y)
                    for x, y, pid in rows
                    if frozenset((key(x), key(y))) in triangle_sides
                }
                out.append(VectorArrow(tip, base, sorted(ids), segments))
    return out


def route_direction(
    points: list[Point], arrows: list[VectorArrow], tolerance: float
) -> tuple[str, list[str]]:
    votes, ids = set(), set()
    for arrow in arrows:
        vx, vy = arrow.tip[0] - arrow.base[0], arrow.tip[1] - arrow.base[1]
        matches = []
        for a, b in zip(points, points[1:], strict=False):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            t = ((arrow.tip[0] - a[0]) * dx + (arrow.tip[1] - a[1]) * dy) / (length * length)
            distance = abs((arrow.tip[0] - a[0]) * dy - (arrow.tip[1] - a[1]) * dx) / length
            alignment = (dx * vx + dy * vy) / (length * math.hypot(vx, vy))
            if (
                -tolerance / length <= t <= 1 + tolerance / length
                and distance <= tolerance
                and abs(alignment) >= 0.96
            ):
                matches.append((distance, "forward" if alignment > 0 else "reverse"))
        if matches:
            votes.add(min(matches)[1])
            ids.update(arrow.path_ids)
    return (
        next(iter(votes)) if len(votes) == 1 else "conflicting" if votes else "unknown",
        sorted(ids),
    )


def nozzle_contact(
    page: PageEvidence, contour: SymbolContour, endpoint: Point, other: Point
) -> list[str]:
    """Support a pipe ending on a flange outside the detected equipment body.

    Require a transverse native bar and a connected local chain of nozzle
    strokes reaching the body. Proximity to the equipment alone is insufficient.
    """
    radius = max(24.0, min(page.width, page.height) * 0.008)
    length = math.dist(endpoint, other)
    if length < 12 or inside(endpoint, contour.points):
        return []
    direction = ((endpoint[0] - other[0]) / length, (endpoint[1] - other[1]) / length)
    extension = (endpoint[0] + direction[0] * radius, endpoint[1] + direction[1] * radius)
    if not intersections(endpoint, extension, contour.points):
        return []
    tolerance = max(2.0, min(page.width, page.height) * 0.0005)

    def distance(point, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(
            0.0,
            min(
                1.0,
                ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / max(dx * dx + dy * dy, 1e-9),
            ),
        )
        return math.dist(point, (a[0] + t * dx, a[1] + t * dy))

    rows = []
    for path in page.paths:
        if path.origin != "pdf_vector":
            continue
        if (
            path.bbox.x2 < endpoint[0] - radius
            or path.bbox.x > endpoint[0] + radius
            or path.bbox.y2 < endpoint[1] - radius
            or path.bbox.y > endpoint[1] + radius
        ):
            continue
        points = path_vertices(path)
        for a, b in zip(points, points[1:], strict=False):
            if segment_key(path.id, a, b) in contour.segments:
                continue
            if (
                1 <= math.dist(a, b) <= radius * 2
                and max(math.dist(a, endpoint), math.dist(b, endpoint)) <= radius * 1.5
            ):
                rows.append((a, b, path.id))
    starts = [
        i
        for i, (a, b, _) in enumerate(rows)
        if distance(endpoint, a, b) <= tolerance
        and abs((b[0] - a[0]) * direction[0] + (b[1] - a[1]) * direction[1])
        <= math.dist(a, b) * 0.2
    ]
    queue = [(i, [rows[i][2]]) for i in starts]
    visited = set(starts)
    for index, ids in queue:
        a, b, _ = rows[index]
        if any(
            inside(p, contour.points)
            or any(
                distance(p, c, d) <= tolerance
                for c, d in zip(
                    contour.points, contour.points[1:] + contour.points[:1], strict=True
                )
            )
            for p in (a, b)
        ):
            return sorted(set(ids))
        for i, (c, d, pid) in enumerate(rows):
            if (
                i not in visited
                and min(distance(a, c, d), distance(b, c, d), distance(c, a, b), distance(d, a, b))
                <= tolerance
            ):
                visited.add(i)
                queue.append((i, ids + [pid]))
    return []


def unobserved_symbols(page: PageEvidence, nodes: list[ReconciledNode]) -> list[NativeSymbol]:
    """Review hypotheses for large capsule-like native bodies missed by perception.

    This is deliberately not a classifier: curved pipe loops may have the same
    outline. Unobserved hypotheses are never allowed to certify accepted edges.
    Requiring opposing curved ends and long walls avoids text/glyph inventories.
    """
    symbols = native_symbols(page)
    assigned = {c.symbol_id for c in symbol_contours(page, nodes, symbols=symbols).values()}
    curve_ids = {p.id for p in page.paths if p.primitive == "curve"}
    out = []
    for symbol in symbols:
        b = symbol.bbox
        if symbol.id in assigned or min(b.w, b.h) < min(page.width, page.height) * 0.025:
            continue
        axis = int(b.h >= b.w)
        start, length = (b.y, b.h) if axis else (b.x, b.w)
        if max(b.w, b.h) < 1.4 * min(b.w, b.h):
            continue
        curves = [
            p
            for pid, a, z in symbol.segments
            if pid in curve_ids or (abs(a[0] - z[0]) > 0.5 and abs(a[1] - z[1]) > 0.5)
            for p in (a, z)
        ]
        if (
            not curves
            or min(p[axis] for p in curves) > start + length * 0.2
            or max(p[axis] for p in curves) < start + length * 0.8
        ):
            continue
        walls = [
            s
            for s in symbol.segments
            if s[0] not in curve_ids and abs(s[1][axis] - s[2][axis]) > length * 0.4
        ]
        if len(walls) >= 2:
            out.append(symbol)
    # Keep the outer body, not several hypotheses for its chords or panels.
    return [s for s in out if not any(t.area > s.area and t.bbox.iou(s.bbox) > 0.65 for t in out)]
