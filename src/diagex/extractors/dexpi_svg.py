"""Render a ReconciledGraph / EngineeringModel pair as a self-contained SVG.

The SVG is derived from the same ReconciledGraph the DexpiBuilder consumes, so
it is a faithful visual of what ended up in the DEXPI JSON. The DEXPI 2.0
spec defines a separate fully-populated ``Diagram`` (shapes, strokes,
NodePositions) which our conceptual-only build does not produce; this
renderer uses the ReconciledGraph's bbox_global / polyline_global directly.

Visual conventions:
  - equipment → rectangle coloured by equipment_class, label inside
  - instrument → circle with the agent's label (e.g. "FIC-101")
  - valve → rotated square (diamond) coloured by valve_type
  - opc → pentagon tagged with direction
  - edges → polylines, stroke style encodes line_type (process / signal / …)
  - low-confidence nodes → dashed outline
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from diagex.dexpi_schema import EQUIPMENT_REGISTRY, VALVE_REGISTRY
from diagex.llm.cost import format_tokens_millions, total_tokens_from_summary
from diagex.vision.models import BBox, ReconciledEdge, ReconciledGraph, ReconciledNode

# Fill colours are sourced from the DEXPI registry so adding an equipment
# class or valve type picks up a colour automatically (falls through to the
# unclassified fill when a registry entry does not specify one).
_EQUIPMENT_FILL: dict[str, str] = {s.key: s.svg_fill for s in EQUIPMENT_REGISTRY}
# The "valve" equipment_class is the router key for the valve-registry path;
# its fill is cosmetic (valves are routed into VALVE_REGISTRY for rendering).
_EQUIPMENT_FILL.setdefault("valve", "#fef3c7")

_VALVE_FILL: dict[str, str] = {s.key: s.svg_fill for s in VALVE_REGISTRY}

_LINE_STROKE = {
    "process": ("#1f2937", "none"),
    "signal_electric": ("#2563eb", "4,3"),
    "signal_pneumatic": ("#16a34a", "2,2"),
    "instrument_capillary": ("#9333ea", "1,2"),
    "electrical_power": ("#dc2626", "6,2,1,2"),
    "other": ("#6b7280", "3,3"),
}

_MARGIN_PX = 40
_MAX_DIM_PX = 4000              # cap output so a massive drawing doesn't blow up in a browser


@dataclass
class SvgRenderOptions:
    show_tags: bool = True
    show_legend_panel: bool = True
    show_title_block: bool = True
    max_dim: int = _MAX_DIM_PX
    # Manhattan-route inferred edges (those without a bootstrap polyline). On
    # by default — straight diagonals between bbox centres look distorted.
    # Bootstrap polylines stay untouched either way.
    orthogonal_inferred: bool = True


def render_graph_to_svg(
    graph: ReconciledGraph,
    *,
    title: str = "",
    options: SvgRenderOptions | None = None,
    metadata: dict | None = None,
) -> str:
    """Return a standalone SVG string for the whole ReconciledGraph."""
    opts = options or SvgRenderOptions()

    bounds = _graph_bounds(graph)
    if bounds is None:
        return _empty_svg(title)

    x_min, y_min, x_max, y_max = bounds
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)

    # Downscale to max_dim while preserving aspect ratio. `scale <= 1` always.
    scale = min(1.0, opts.max_dim / max(w, h))
    canvas_w = int(round(w * scale)) + 2 * _MARGIN_PX
    canvas_h = int(round(h * scale)) + 2 * _MARGIN_PX

    def project(x: float, y: float) -> tuple[float, float]:
        return (_MARGIN_PX + (x - x_min) * scale, _MARGIN_PX + (y - y_min) * scale)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" '
        f'viewBox="0 0 {canvas_w} {canvas_h}">'
    )
    parts.append(_defs_block())
    parts.append(_style_block())

    heading = title or graph.source_path or "diagex extraction"
    parts.append(
        f'<text class="title" x="{_MARGIN_PX}" y="{_MARGIN_PX - 12}">'
        f"{_xml_escape(heading)}</text>"
    )

    # Edges first so nodes overlay them.
    node_by_id = {n.id: n for n in graph.nodes}

    # Two-phase edge rendering: prepare (snap to perimeter, inherit line_id,
    # compute crossings) then draw (path with hops + mid-segment line-id pills).
    prepared = _prepare_edges(
        graph.edges, node_by_id,
        orthogonal_inferred=opts.orthogonal_inferred,
    )
    hops = _detect_crossings(prepared)
    nozzle_points: list[tuple[int, int, str]] = []   # (x, y, 'h'/'v') per attachment
    for pe in prepared:
        parts.append(_render_prepared_edge(pe, hops.get(pe.edge.id, []), project, nozzle_points))

    # Nozzle ticks on equipment perimeters.
    for (nx, ny, orient) in nozzle_points:
        parts.append(_nozzle_marker(nx, ny, orient, project))

    # Collect nodes + target label positions before rendering, so the label
    # collision-avoider has the full set to reason about.
    label_layout = _plan_labels(graph.nodes, project, scale) if opts.show_tags else {}
    for node in graph.nodes:
        parts.append(_render_node(node, project, scale, show_tags=opts.show_tags, label_layout=label_layout))

    if opts.show_legend_panel:
        parts.append(_legend_panel(canvas_w, canvas_h))

    if opts.show_title_block:
        parts.append(_title_block(canvas_w, canvas_h, graph=graph, heading=heading, metadata=metadata or {}))

    # Footer counts. Ghost edges (no geometry, no snapped endpoints) are edges
    # the agent recorded but the reconciler could not localise — surface them
    # explicitly so they're not silently absent from the picture.
    counts = _count_kinds(graph.nodes)
    ghost_edges = sum(
        1 for e in graph.edges
        if not e.polyline_global and not (e.from_node and e.to_node)
    )
    footer_bits = [
        f'equipment {counts["equipment"]}',
        f'instruments {counts["instrument"]}',
        f'opcs {counts["opc"]}',
        f'edges {len(graph.edges)}',
    ]
    if ghost_edges:
        footer_bits.append(f'ghost edges {ghost_edges} (no endpoints)')
    parts.append(
        f'<text class="footer" x="{_MARGIN_PX}" y="{canvas_h - 10}">'
        f"{_xml_escape(' · '.join(footer_bits))}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(
    graph: ReconciledGraph,
    path: Path,
    *,
    title: str = "",
    options: SvgRenderOptions | None = None,
    metadata: dict | None = None,
) -> Path:
    """Render `graph` and write it to `path`. Returns the path for chaining."""
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = render_graph_to_svg(graph, title=title, options=options, metadata=metadata)
    path.write_text(svg, encoding="utf-8")
    return path


def render_graph_json_to_svg(
    graph_json_path: Path,
    *,
    title: str = "",
    orthogonal_inferred: bool = True,
) -> str:
    """Convenience loader: read a graph.json from disk, render it.

    Also opportunistically harvests run metadata from a sibling `result.json`
    when present, so `diagex render-dexpi <run_dir>` gets a populated title block.
    """
    raw = json.loads(graph_json_path.read_text(encoding="utf-8"))
    graph = ReconciledGraph.model_validate(raw)
    metadata = _harvest_run_metadata(graph_json_path)
    options = SvgRenderOptions(orthogonal_inferred=orthogonal_inferred)
    return render_graph_to_svg(
        graph, title=title or graph.source_path,
        metadata=metadata, options=options,
    )


def _harvest_run_metadata(graph_json_path: Path) -> dict:
    """Peek at a sibling result.json for title-block metadata (best-effort)."""
    try:
        candidate = graph_json_path.with_name("result.json")
        if not candidate.exists():
            return {}
        r = json.loads(candidate.read_text(encoding="utf-8"))
        return {
            "run_id": r.get("run_id"),
            "model": r.get("model"),
            "effort": r.get("effort"),
            "total_tokens": total_tokens_from_summary(r.get("cost") or {}),
            "timestamp": graph_json_path.parent.name.split("_", 1)[0],
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _graph_bounds(graph: ReconciledGraph) -> tuple[int, int, int, int] | None:
    xs_lo: list[int] = []
    ys_lo: list[int] = []
    xs_hi: list[int] = []
    ys_hi: list[int] = []
    for n in graph.nodes:
        xs_lo.append(n.bbox_global.x)
        ys_lo.append(n.bbox_global.y)
        xs_hi.append(n.bbox_global.x + n.bbox_global.w)
        ys_hi.append(n.bbox_global.y + n.bbox_global.h)
    for e in graph.edges:
        for (x, y) in e.polyline_global:
            xs_lo.append(x)
            ys_lo.append(y)
            xs_hi.append(x)
            ys_hi.append(y)
    if not xs_lo:
        return None
    return (min(xs_lo), min(ys_lo), max(xs_hi), max(ys_hi))


def _render_node(
    node: ReconciledNode,
    project,
    scale: float,
    *,
    show_tags: bool,
    label_layout: dict | None = None,
) -> str:
    bb = node.bbox_global
    x0, y0 = project(bb.x, bb.y)
    w = max(2.0, bb.w * scale)
    h = max(2.0, bb.h * scale)
    cx, cy = x0 + w / 2, y0 + h / 2

    kind = node.kind
    dash = ' stroke-dasharray="3,2"' if node.confidence == "low" else ""
    label_raw = node.label or ""
    label_display = _truncate_label(label_raw)
    tooltip = _xml_escape(_tooltip_text(node))

    if kind == "equipment":
        eq_class = str(node.attributes.get("equipment_class", "unclassified_equipment"))
        if eq_class == "valve":
            return _wrap_tooltip(
                _render_valve(node, x0, y0, w, h, _xml_escape(label_display), dash,
                              layout=_label_slot(label_layout, node, cx, y0 + h)),
                tooltip,
            )
        glyph = _render_equipment_glyph(eq_class, node, x0, y0, w, h, dash)
        slot = _label_slot(label_layout, node, cx, y0 + h)
        text = ""
        if show_tags:
            sub = _equipment_subtitle(node)
            text = _label_with_subtitle(slot, _xml_escape(label_display), _xml_escape(sub))
        return _wrap_tooltip(glyph + text, tooltip)

    if kind == "instrument":
        r = max(6.0, min(w, h) / 2)
        fill = "#f3e8ff"
        shape = (
            f'<circle class="inst" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
            f'fill="{fill}"{dash} />'
        )
        # ISA-5.1: horizontal line through the bubble denotes main control
        # panel mounting (as opposed to field). Applied for controllers and
        # recorders — functions typically located in a control room.
        fn = str(node.attributes.get("instrument_function", "")).lower()
        if fn in ("controller", "indicator_controller", "recorder_controller",
                  "recorder", "alarm"):
            shape += (
                f'<line x1="{cx - r:.1f}" y1="{cy:.1f}" x2="{cx + r:.1f}" y2="{cy:.1f}" '
                f'stroke="#111827" stroke-width="1" />'
            )
        # Labels sit INSIDE the instrument bubble when they fit (tag like "FIC-101"),
        # otherwise below. Estimate fit by char count vs bubble diameter.
        fits_inside = len(label_display) * _LABEL_CHAR_PX < 2 * r - 4
        if show_tags and fits_inside:
            text = (
                f'<text class="inst-tag" x="{cx:.1f}" y="{cy + 3:.1f}" text-anchor="middle">'
                f"{_xml_escape(label_display)}</text>"
            )
        else:
            slot = _label_slot(label_layout, node, cx, cy + r)
            text = _label_text_positioned(slot, _xml_escape(label_display)) if show_tags else ""
        return _wrap_tooltip(shape + text, tooltip)

    if kind == "opc":
        direction = str(node.attributes.get("direction", "in"))
        opc_size = max(10.0, min(w, h))
        slot = _label_slot(label_layout, node, cx, cy + opc_size / 2)
        line_id = str(node.attributes.get("line_id", "")).strip()
        subtitle = _xml_escape(line_id) if line_id else ""
        return _wrap_tooltip(
            _render_opc(cx, cy, opc_size, direction, _xml_escape(label_display), dash, slot, subtitle),
            tooltip,
        )

    if kind in ("line", "connection"):
        # Line nodes shouldn't exist — edges carry geometry. Skip quietly.
        return ""

    if kind == "text":
        return _wrap_tooltip(_label_text(cx, cy, _xml_escape(label_display), classname="txt"), tooltip)

    # Fallback: thin outline box.
    slot = _label_slot(label_layout, node, cx, y0 + h)
    return _wrap_tooltip(
        (
            f'<rect class="misc" x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="#f9fafb"{dash} />'
        ) + _label_text_positioned(slot, _xml_escape(label_display)),
        tooltip,
    )


def _render_equipment_glyph(
    eq_class: str,
    node: ReconciledNode,
    x0: float,
    y0: float,
    w: float,
    h: float,
    dash: str,
) -> str:
    """Dispatch to a P&ID-style glyph per equipment_class (ISA-5.1 / ISO-10628 inspired)."""
    cx = x0 + w / 2
    cy = y0 + h / 2
    fill = _EQUIPMENT_FILL.get(eq_class, _EQUIPMENT_FILL["unclassified_equipment"])

    if eq_class == "tank":
        return _glyph_tank(x0, y0, w, h, fill, dash)
    if eq_class == "vessel":
        return _glyph_vessel(x0, y0, w, h, fill, dash)
    if eq_class == "column":
        return _glyph_column(x0, y0, w, h, fill, dash)
    if eq_class == "heat_exchanger":
        hx_type = str(node.attributes.get("heat_exchanger_type", "")).lower()
        if hx_type in ("plate", "plate_heat_exchanger", "gasketed_plate"):
            return _glyph_plate_hx(x0, y0, w, h, fill, dash)
        # Default: TEMA-style shell-and-tube.
        return _glyph_shell_tube_hx(x0, y0, w, h, fill, dash)
    if eq_class == "pump":
        pump_type = str(node.attributes.get("pump_type", "centrifugal")).lower()
        return _glyph_pump(cx, cy, min(w, h), fill, pump_type, dash)
    if eq_class == "compressor":
        return _glyph_compressor(x0, y0, w, h, fill, dash)
    if eq_class == "reactor":
        return _glyph_reactor(x0, y0, w, h, fill, dash)
    if eq_class == "agitator":
        return _glyph_agitator(cx, cy, min(w, h), fill, dash)
    if eq_class == "filter":
        return _glyph_filter(x0, y0, w, h, fill, dash)
    if eq_class == "separator":
        return _glyph_separator(x0, y0, w, h, fill, dash)
    if eq_class == "fired_heater":
        return _glyph_fired_heater(x0, y0, w, h, fill, dash)
    if eq_class == "cooling_tower":
        return _glyph_cooling_tower(x0, y0, w, h, fill, dash)
    # Unknown / unclassified: hatched rectangle so it's visually distinct from
    # a recognised class that happens to be drawn as a rectangle. Low-confidence
    # nodes already carry a dash; fall back to the 4,2 hatch when the caller
    # hasn't supplied one, to avoid double-defining the attribute.
    dash_attr = dash if dash else ' stroke-dasharray="4,2"'
    return (
        f'<rect class="eq unclassified" x="{x0:.1f}" y="{y0:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="#111827" stroke-width="1"{dash_attr} />'
    )


def _glyph_tank(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Vertical tank: rectangle with dished (elliptical) top and bottom caps."""
    cap_h = min(h * 0.18, w * 0.45)
    body_top = y0 + cap_h / 2
    body_h = h - cap_h
    cx = x0 + w / 2
    ry = cap_h / 2
    # Body rectangle, then two ellipse caps to hint at dished heads.
    return (
        f'<g class="eq tank"{dash_to_group(dash)}>'
        f'<rect x="{x0:.1f}" y="{body_top:.1f}" width="{w:.1f}" height="{body_h:.1f}" '
        f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
        f'<ellipse cx="{cx:.1f}" cy="{body_top:.1f}" rx="{w/2:.1f}" ry="{ry:.1f}" '
        f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
        f'<ellipse cx="{cx:.1f}" cy="{body_top + body_h:.1f}" rx="{w/2:.1f}" ry="{ry:.1f}" '
        f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
        f'</g>'
    )


def _glyph_vessel(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Vessel: dished caps on the short axis (horizontal vs vertical decided by aspect)."""
    if w >= h:
        # Horizontal vessel: dished caps left and right.
        cap_w = min(w * 0.18, h * 0.45)
        body_left = x0 + cap_w / 2
        body_w = w - cap_w
        rx = cap_w / 2
        cy = y0 + h / 2
        return (
            f'<g class="eq vessel">'
            f'<rect x="{body_left:.1f}" y="{y0:.1f}" width="{body_w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
            f'<ellipse cx="{body_left:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{h/2:.1f}" '
            f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
            f'<ellipse cx="{body_left + body_w:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{h/2:.1f}" '
            f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
            f'</g>'
        )
    # Otherwise treat like a tank.
    return _glyph_tank(x0, y0, w, h, fill, dash)


def _glyph_column(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Tall column: dished caps + internal tray lines."""
    body = _glyph_tank(x0, y0, w, h, fill, dash)
    # Internal horizontal tray lines — 3 to 6 evenly spaced.
    tray_count = max(3, min(6, int(h / 40)))
    cap_h = min(h * 0.18, w * 0.45)
    inner_top = y0 + cap_h
    inner_h = h - 2 * cap_h
    if inner_h <= 0 or tray_count == 0:
        return body
    step = inner_h / (tray_count + 1)
    lines = []
    for i in range(1, tray_count + 1):
        ty = inner_top + i * step
        lines.append(
            f'<line x1="{x0 + 4:.1f}" y1="{ty:.1f}" x2="{x0 + w - 4:.1f}" y2="{ty:.1f}" '
            f'stroke="#6b7280" stroke-width="0.8" />'
        )
    return body + "".join(lines)


def _glyph_shell_tube_hx(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """TEMA-style shell-and-tube: shell rectangle + tube bundle + tube-sheet divider.

    Tube-side inlet/outlet nozzles at the short ends (top of each end cap);
    shell-side inlet/outlet nozzles at the long sides.
    """
    body = (
        f'<rect class="eq heat-exchanger" x="{x0:.1f}" y="{y0:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="#111827" stroke-width="1"{dash} />'
    )
    # Tube-sheet markers (thin verticals near each end).
    ts_inset = max(4.0, w * 0.08)
    tube_sheets = (
        f'<line x1="{x0 + ts_inset:.1f}" y1="{y0:.1f}" '
        f'x2="{x0 + ts_inset:.1f}" y2="{y0 + h:.1f}" stroke="#111827" stroke-width="0.8" />'
        f'<line x1="{x0 + w - ts_inset:.1f}" y1="{y0:.1f}" '
        f'x2="{x0 + w - ts_inset:.1f}" y2="{y0 + h:.1f}" stroke="#111827" stroke-width="0.8" />'
    )
    # Tube bundle: 4-6 thin horizontal lines between the tube sheets.
    n_tubes = max(4, min(6, int(h / 10)))
    bundle_y0 = y0 + h * 0.2
    bundle_h = h * 0.6
    tube_lines: list[str] = []
    for i in range(n_tubes):
        ty = bundle_y0 + (i * bundle_h / max(1, n_tubes - 1)) if n_tubes > 1 else y0 + h / 2
        tube_lines.append(
            f'<line x1="{x0 + ts_inset:.1f}" y1="{ty:.1f}" '
            f'x2="{x0 + w - ts_inset:.1f}" y2="{ty:.1f}" '
            f'stroke="#4b5563" stroke-width="0.6" />'
        )
    tubes = "".join(tube_lines)
    # Shell-side nozzles (top-left, bottom-right is conventional for counter-current).
    stub = min(8.0, w * 0.1)
    shell_nozzles = (
        f'<line x1="{x0 + w * 0.25:.1f}" y1="{y0:.1f}" '
        f'x2="{x0 + w * 0.25:.1f}" y2="{y0 - stub:.1f}" '
        f'stroke="#111827" stroke-width="1.2" />'
        f'<line x1="{x0 + w * 0.75:.1f}" y1="{y0 + h:.1f}" '
        f'x2="{x0 + w * 0.75:.1f}" y2="{y0 + h + stub:.1f}" '
        f'stroke="#111827" stroke-width="1.2" />'
    )
    # Tube-side nozzles (on the end caps).
    cy = y0 + h / 2
    tube_nozzles = (
        f'<line x1="{x0 - stub:.1f}" y1="{cy:.1f}" x2="{x0:.1f}" y2="{cy:.1f}" '
        f'stroke="#111827" stroke-width="1.5" />'
        f'<line x1="{x0 + w:.1f}" y1="{cy:.1f}" x2="{x0 + w + stub:.1f}" y2="{cy:.1f}" '
        f'stroke="#111827" stroke-width="1.5" />'
    )
    return body + tube_sheets + tubes + shell_nozzles + tube_nozzles


def _glyph_plate_hx(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Plate-type heat exchanger: stack of V-chevron plates (standard ISA glyph)."""
    body = (
        f'<rect class="eq heat-exchanger plate" x="{x0:.1f}" y="{y0:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
        f'stroke="#111827" stroke-width="1"{dash} />'
    )
    # Chevron (V) plate stack — typical of gasketed plate HX.
    n_plates = max(3, min(6, int(h / 14)))
    chevrons: list[str] = []
    margin_x = max(4.0, w * 0.08)
    pad_y = h * 0.1
    step = (h - 2 * pad_y) / max(1, n_plates)
    for i in range(n_plates):
        py = y0 + pad_y + i * step + step / 2
        chevrons.append(
            f'<polyline fill="none" stroke="#4b5563" stroke-width="0.7" '
            f'points="{x0 + margin_x:.1f},{py - step * 0.3:.1f} '
            f'{x0 + w / 2:.1f},{py + step * 0.3:.1f} '
            f'{x0 + w - margin_x:.1f},{py - step * 0.3:.1f}" />'
        )
    # Nozzles (4 corners).
    stub = min(8.0, w * 0.1)
    nozzles = (
        f'<line x1="{x0 - stub:.1f}" y1="{y0 + h * 0.25:.1f}" '
        f'x2="{x0:.1f}" y2="{y0 + h * 0.25:.1f}" stroke="#111827" stroke-width="1.5" />'
        f'<line x1="{x0 - stub:.1f}" y1="{y0 + h * 0.75:.1f}" '
        f'x2="{x0:.1f}" y2="{y0 + h * 0.75:.1f}" stroke="#111827" stroke-width="1.5" />'
        f'<line x1="{x0 + w:.1f}" y1="{y0 + h * 0.25:.1f}" '
        f'x2="{x0 + w + stub:.1f}" y2="{y0 + h * 0.25:.1f}" stroke="#111827" stroke-width="1.5" />'
        f'<line x1="{x0 + w:.1f}" y1="{y0 + h * 0.75:.1f}" '
        f'x2="{x0 + w + stub:.1f}" y2="{y0 + h * 0.75:.1f}" stroke="#111827" stroke-width="1.5" />'
    )
    return body + "".join(chevrons) + nozzles


def _glyph_pump(cx: float, cy: float, size: float, fill: str, pump_type: str, dash: str) -> str:
    """Pump: circle body with a subtype-specific internal mark.

    Adds an ISA-5.1 "M" motor indicator above the pump and a short
    suction/discharge stub so the glyph reads as "real pump" not "graph node".
    """
    r = max(8.0, size / 2)
    body = (
        f'<circle class="eq pump" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
    )
    # Suction stub (left) and discharge stub (top) — standard centrifugal layout.
    stub_len = r * 0.6
    stubs = (
        f'<line x1="{cx - r:.1f}" y1="{cy:.1f}" x2="{cx - r - stub_len:.1f}" y2="{cy:.1f}" '
        f'stroke="#111827" stroke-width="1.5" />'
        f'<line x1="{cx:.1f}" y1="{cy - r:.1f}" x2="{cx:.1f}" y2="{cy - r - stub_len:.1f}" '
        f'stroke="#111827" stroke-width="1.5" />'
    )
    # Motor indicator above discharge (circle with "M"). Skip for very small pumps.
    motor = ""
    if r >= 12.0:
        my = cy - r - stub_len - 8
        motor = (
            f'<circle cx="{cx:.1f}" cy="{my:.1f}" r="7" fill="#ffffff" '
            f'stroke="#111827" stroke-width="1" />'
            f'<text class="actuator-mark" x="{cx:.1f}" y="{my + 3:.1f}" text-anchor="middle">M</text>'
        )
    if pump_type == "centrifugal":
        # Standard ISA centrifugal-pump glyph: triangular volute pointing up-right.
        p1 = (cx, cy - r)
        p2 = (cx + r * 0.9, cy)
        p3 = (cx, cy + r)
        mark = (
            f'<polygon points="{p1[0]:.1f},{p1[1]:.1f} '
            f'{p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}" '
            f'fill="none" stroke="#111827" stroke-width="1.2" />'
        )
    elif pump_type in ("reciprocating", "metering"):
        # Vertical piston bar.
        mark = (
            f'<rect x="{cx - r*0.2:.1f}" y="{cy - r*0.7:.1f}" '
            f'width="{r*0.4:.1f}" height="{r*1.4:.1f}" '
            f'fill="none" stroke="#111827" stroke-width="1.2" />'
        )
    elif pump_type == "rotary":
        # Two gear circles.
        mark = (
            f'<circle cx="{cx - r*0.35:.1f}" cy="{cy:.1f}" r="{r*0.3:.1f}" '
            f'fill="none" stroke="#111827" stroke-width="1.2" />'
            f'<circle cx="{cx + r*0.35:.1f}" cy="{cy:.1f}" r="{r*0.3:.1f}" '
            f'fill="none" stroke="#111827" stroke-width="1.2" />'
        )
    else:
        # Generic pump: diagonal line suggesting rotation.
        mark = (
            f'<line x1="{cx - r*0.6:.1f}" y1="{cy + r*0.6:.1f}" '
            f'x2="{cx + r*0.6:.1f}" y2="{cy - r*0.6:.1f}" '
            f'stroke="#111827" stroke-width="1.2" />'
        )
    return body + stubs + motor + mark


def _glyph_compressor(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Compressor: trapezoidal body suggesting turbine/compressor flow path."""
    inset = w * 0.15
    pts = (
        f"{x0:.1f},{y0 + h:.1f} "
        f"{x0 + inset:.1f},{y0:.1f} "
        f"{x0 + w - inset:.1f},{y0:.1f} "
        f"{x0 + w:.1f},{y0 + h:.1f}"
    )
    return (
        f'<polygon class="eq compressor" points="{pts}" fill="{fill}" '
        f'stroke="#111827" stroke-width="1"{dash} />'
    )


def dash_to_group(dash: str) -> str:
    """Propagate a stroke-dasharray attribute onto a <g> wrapper for composite glyphs."""
    return dash  # inner elements already carry the attribute; kept for future styling.


# ---------------------------------------------------------------------------
# v0.2 equipment glyphs (reactor / agitator / filter / separator / fired heater / cooling tower)
#
# First-pass shapes: distinguishable at thumbnail size, no ambitious iconography.
# Each renderer receives (x0, y0, w, h) — the reconciled bbox in SVG coords —
# and a pre-resolved ``fill`` (colour) and ``dash`` (stroke-dasharray suffix
# for low-confidence nodes). They return a single SVG snippet.


def _glyph_reactor(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Reactor: vertical vessel silhouette with an internal agitator suggestion.

    The shape hints at a CSTR — a vessel with an impeller on a shaft. Works for
    any reactor type when :data:`EQUIPMENT_REGISTRY` entry "reactor" is in play.
    """
    body = _glyph_tank(x0, y0, w, h, fill, dash)
    cx = x0 + w / 2
    shaft_top = y0 + h * 0.10
    shaft_bot = y0 + h * 0.70
    impeller_y = shaft_bot
    impeller_half_w = w * 0.22
    # Shaft line + a short horizontal impeller blade near the bottom.
    return (
        f'<g class="eq reactor">'
        f"{body}"
        f'<line x1="{cx:.1f}" y1="{shaft_top:.1f}" x2="{cx:.1f}" y2="{shaft_bot:.1f}" '
        f'stroke="#111827" stroke-width="1.5"{dash} />'
        f'<line x1="{cx - impeller_half_w:.1f}" y1="{impeller_y:.1f}" '
        f'x2="{cx + impeller_half_w:.1f}" y2="{impeller_y:.1f}" '
        f'stroke="#111827" stroke-width="2"{dash} />'
        f"</g>"
    )


def _glyph_agitator(cx: float, cy: float, size: float, fill: str, dash: str) -> str:
    """Agitator: motor-top circle with a downward shaft and impeller blade."""
    r = max(6.0, size / 3)
    shaft_top = cy - r
    shaft_bot = cy + size / 2
    blade_half = size * 0.30
    return (
        f'<g class="eq agitator">'
        f'<circle cx="{cx:.1f}" cy="{cy - r:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="#111827" stroke-width="1"{dash} />'
        f'<line x1="{cx:.1f}" y1="{shaft_top:.1f}" x2="{cx:.1f}" y2="{shaft_bot:.1f}" '
        f'stroke="#111827" stroke-width="1.5"{dash} />'
        f'<line x1="{cx - blade_half:.1f}" y1="{shaft_bot:.1f}" '
        f'x2="{cx + blade_half:.1f}" y2="{shaft_bot:.1f}" '
        f'stroke="#111827" stroke-width="2"{dash} />'
        f"</g>"
    )


def _glyph_filter(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Filter: rectangular housing with horizontal filter-element stripes."""
    stripes = 4
    stripe_pitch = h / (stripes + 1)
    lines = []
    for i in range(1, stripes + 1):
        y = y0 + i * stripe_pitch
        lines.append(
            f'<line x1="{x0 + w * 0.12:.1f}" y1="{y:.1f}" '
            f'x2="{x0 + w * 0.88:.1f}" y2="{y:.1f}" '
            f'stroke="#111827" stroke-width="1"{dash} />'
        )
    return (
        f'<g class="eq filter">'
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
        + "".join(lines)
        + "</g>"
    )


def _glyph_separator(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Separator: horizontal vessel (dished caps) with a vertical internal divider."""
    vessel = _glyph_vessel(x0, y0, w, h, fill, dash)
    cx = x0 + w / 2
    pad_top = y0 + h * 0.15
    pad_bot = y0 + h * 0.85
    divider = (
        f'<line x1="{cx:.1f}" y1="{pad_top:.1f}" x2="{cx:.1f}" y2="{pad_bot:.1f}" '
        f'stroke="#111827" stroke-width="1.5"{dash} stroke-dasharray="5,3" />'
    )
    return f'<g class="eq separator">{vessel}{divider}</g>'


def _glyph_fired_heater(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Fired heater: rectangle with upward flame arrows from the bottom."""
    body = (
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="#111827" stroke-width="1"{dash} />'
    )
    # Three small triangles along the bottom gesturing at burners.
    flames = []
    for frac in (0.25, 0.5, 0.75):
        fx = x0 + w * frac
        fy = y0 + h
        flame_w = w * 0.08
        flames.append(
            f'<polygon points="'
            f'{fx - flame_w:.1f},{fy:.1f} '
            f'{fx:.1f},{fy - h * 0.12:.1f} '
            f'{fx + flame_w:.1f},{fy:.1f}" '
            f'fill="#f97316" stroke="#7c2d12" stroke-width="0.8"{dash} />'
        )
    return f'<g class="eq fired_heater">{body}{"".join(flames)}</g>'


def _glyph_cooling_tower(x0: float, y0: float, w: float, h: float, fill: str, dash: str) -> str:
    """Cooling tower: hourglass / hyperboloid silhouette."""
    # Narrow waist at ~55% of height, open top and bottom.
    top_left_x = x0 + w * 0.05
    top_right_x = x0 + w * 0.95
    waist_left_x = x0 + w * 0.30
    waist_right_x = x0 + w * 0.70
    bot_left_x = x0 + w * 0.10
    bot_right_x = x0 + w * 0.90
    y_top = y0
    y_waist = y0 + h * 0.55
    y_bot = y0 + h
    pts = (
        f"{top_left_x:.1f},{y_top:.1f} "
        f"{waist_left_x:.1f},{y_waist:.1f} "
        f"{bot_left_x:.1f},{y_bot:.1f} "
        f"{bot_right_x:.1f},{y_bot:.1f} "
        f"{waist_right_x:.1f},{y_waist:.1f} "
        f"{top_right_x:.1f},{y_top:.1f}"
    )
    return (
        f'<polygon class="eq cooling_tower" points="{pts}" fill="{fill}" '
        f'stroke="#111827" stroke-width="1"{dash} />'
    )


def _render_valve(
    node: ReconciledNode,
    x0: float,
    y0: float,
    w: float,
    h: float,
    label: str,
    dash: str,
    layout: LabelSlot | None = None,
) -> str:
    vt = str(node.attributes.get("valve_type", "other"))
    subtype = str(node.attributes.get("subtype", "")).lower()
    fill = _VALVE_FILL.get(vt, _VALVE_FILL["other"])
    cx, cy = x0 + w / 2, y0 + h / 2
    r = max(8.0, min(w, h) / 2)

    # Safety-relief valve gets its own ISA glyph rather than a plain diamond.
    if subtype == "safety_relief_valve":
        return _glyph_safety_relief_valve(node, cx, cy, r, fill, dash, layout, label)

    # ISA-5.1 valve body: two triangles meeting at the centre (bowtie).
    # Orientation defaults to horizontal; a tall bbox makes it vertical.
    bb = node.bbox_global
    horizontal = bb.w >= bb.h
    body_svg, inner_decoration = _glyph_bowtie(cx, cy, r, fill, horizontal, dash, vt)
    # Check valves get an additional directional triangle filled solid in the
    # outlet triangle — unambiguous flow indicator.
    check_arrow = ""
    if vt == "check":
        check_arrow = _check_valve_arrow(cx, cy, r, horizontal)

    actuator = _valve_actuator(node, cx, cy - r)
    fail_badge = _valve_fail_badge(node, cx, cy + r)
    slot = layout or LabelSlot(x=cx, y=cy + r + 22, rotate=0.0)
    # Nudge label down when we added a FC/FO badge so they don't collide.
    if fail_badge and not slot.rotate:
        slot = LabelSlot(x=slot.x, y=max(slot.y, cy + r + 28), rotate=slot.rotate)
    return (
        body_svg
        + inner_decoration
        + check_arrow
        + actuator
        + fail_badge
        + _label_text_positioned(slot, label)
    )


def _glyph_bowtie(
    cx: float,
    cy: float,
    r: float,
    fill: str,
    horizontal: bool,
    dash: str,
    valve_type: str,
) -> tuple[str, str]:
    """ISA-5.1 two-triangle 'bowtie' valve body + inner marker for subtype.

    Returns (body_svg, inner_decoration). The body is one <polygon> with both
    triangles so fill/dash apply uniformly; the inner decoration carries
    subtype-specific markers (ball circle, butterfly line, globe fill).
    """
    if horizontal:
        # Two triangles meeting at centre, apex pointing inward.
        pts = (
            f"{cx - r:.1f},{cy - r * 0.6:.1f} "
            f"{cx:.1f},{cy:.1f} "
            f"{cx - r:.1f},{cy + r * 0.6:.1f} "
            f"{cx - r:.1f},{cy - r * 0.6:.1f} "
            f"M{cx + r:.1f},{cy - r * 0.6:.1f} "
            f"{cx:.1f},{cy:.1f} "
            f"{cx + r:.1f},{cy + r * 0.6:.1f} "
            f"{cx + r:.1f},{cy - r * 0.6:.1f}"
        )
    else:
        pts = (
            f"{cx - r * 0.6:.1f},{cy - r:.1f} "
            f"{cx:.1f},{cy:.1f} "
            f"{cx + r * 0.6:.1f},{cy - r:.1f} "
            f"{cx - r * 0.6:.1f},{cy - r:.1f} "
            f"M{cx - r * 0.6:.1f},{cy + r:.1f} "
            f"{cx:.1f},{cy:.1f} "
            f"{cx + r * 0.6:.1f},{cy + r:.1f} "
            f"{cx - r * 0.6:.1f},{cy + r:.1f}"
        )
    # Using <path> for the bowtie so two disjoint triangles share one fill/stroke.
    # Convert the polygon-style point list to a path `d`:
    body_d = _points_to_path_d(pts)
    body = (
        f'<path class="valve" d="{body_d}" fill="{fill}" stroke="#111827" '
        f'stroke-width="1"{dash} fill-rule="nonzero" />'
    )

    # Inner markers per valve subtype.
    deco = ""
    if valve_type == "ball":
        deco = (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.25:.1f}" '
            f'fill="none" stroke="#111827" stroke-width="1" />'
        )
    elif valve_type == "butterfly":
        if horizontal:
            deco = (
                f'<line x1="{cx - r * 0.8:.1f}" y1="{cy:.1f}" '
                f'x2="{cx + r * 0.8:.1f}" y2="{cy:.1f}" '
                f'stroke="#111827" stroke-width="1.2" />'
            )
        else:
            deco = (
                f'<line x1="{cx:.1f}" y1="{cy - r * 0.8:.1f}" '
                f'x2="{cx:.1f}" y2="{cy + r * 0.8:.1f}" '
                f'stroke="#111827" stroke-width="1.2" />'
            )
    elif valve_type == "globe":
        # Globe valves traditionally drawn with a filled body — darken the fill.
        deco = ""  # keep bowtie fill; visually distinct enough via actuator.
    return body, deco


def _points_to_path_d(pts: str) -> str:
    """Convert a 'x,y x,y M x,y x,y' style token list into SVG path commands."""
    tokens = pts.split()
    out: list[str] = []
    started = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("M"):
            coord = tok[1:]
            out.append(f"M {coord}")
            started = True
            i += 1
            continue
        if not started:
            out.append(f"M {tok}")
            started = True
        else:
            out.append(f"L {tok}")
        i += 1
    return " ".join(out)


def _check_valve_arrow(cx: float, cy: float, r: float, horizontal: bool) -> str:
    """Solid directional triangle inside the bowtie indicating flow direction."""
    if horizontal:
        ax0 = cx - r * 0.1
        ax1 = cx + r * 0.6
        pts = (
            f"{ax0:.1f},{cy - r * 0.25:.1f} "
            f"{ax1:.1f},{cy:.1f} "
            f"{ax0:.1f},{cy + r * 0.25:.1f}"
        )
    else:
        pts = (
            f"{cx - r * 0.25:.1f},{cy - r * 0.1:.1f} "
            f"{cx:.1f},{cy + r * 0.6:.1f} "
            f"{cx + r * 0.25:.1f},{cy - r * 0.1:.1f}"
        )
    return f'<polygon points="{pts}" fill="#111827" stroke="none" />'


def _valve_actuator(node: ReconciledNode, cx: float, top_y: float) -> str:
    """Draw an ISA-style actuator glyph above the valve body.

    Picks the glyph from `attributes.actuation` if present, otherwise from
    valve_type (control → pneumatic diaphragm, gate/globe/ball/butterfly → handwheel).
    """
    actuation = str(node.attributes.get("actuation", "")).lower()
    valve_type = str(node.attributes.get("valve_type", "")).lower()

    # Infer when actuation is absent.
    if not actuation:
        if valve_type == "control":
            actuation = "pneumatic"
        elif valve_type in ("gate", "globe", "ball", "butterfly"):
            actuation = "hand"

    if not actuation:
        return ""

    # Stem (short vertical line from the diamond apex to the actuator).
    stem_h = 10.0
    stem = (
        f'<line x1="{cx:.1f}" y1="{top_y:.1f}" x2="{cx:.1f}" y2="{top_y - stem_h:.1f}" '
        f'stroke="#111827" stroke-width="1.2" />'
    )
    ay = top_y - stem_h

    if actuation in ("pneumatic", "diaphragm"):
        # Domed diaphragm actuator.
        return stem + (
            f'<path d="M {cx - 8:.1f} {ay:.1f} Q {cx:.1f} {ay - 12:.1f} {cx + 8:.1f} {ay:.1f} Z" '
            f'fill="#fef9c3" stroke="#111827" stroke-width="1" />'
        )
    if actuation in ("motor", "electric_motor"):
        return stem + (
            f'<rect x="{cx - 7:.1f}" y="{ay - 12:.1f}" width="14" height="12" '
            f'fill="#dbeafe" stroke="#111827" stroke-width="1" />'
            f'<text class="actuator-mark" x="{cx:.1f}" y="{ay - 3:.1f}" text-anchor="middle">M</text>'
        )
    if actuation == "solenoid":
        return stem + (
            f'<rect x="{cx - 7:.1f}" y="{ay - 12:.1f}" width="14" height="12" '
            f'fill="#e0e7ff" stroke="#111827" stroke-width="1" />'
            f'<text class="actuator-mark" x="{cx:.1f}" y="{ay - 3:.1f}" text-anchor="middle">S</text>'
        )
    if actuation in ("hand", "hand_switch", "manual"):
        # Handwheel: circle with a cross.
        r = 6.0
        return stem + (
            f'<circle cx="{cx:.1f}" cy="{ay - r:.1f}" r="{r:.1f}" '
            f'fill="#ffffff" stroke="#111827" stroke-width="1" />'
            f'<line x1="{cx - r:.1f}" y1="{ay - r:.1f}" x2="{cx + r:.1f}" y2="{ay - r:.1f}" '
            f'stroke="#111827" stroke-width="1" />'
            f'<line x1="{cx:.1f}" y1="{ay - 2 * r:.1f}" x2="{cx:.1f}" y2="{ay:.1f}" '
            f'stroke="#111827" stroke-width="1" />'
        )
    return ""


def _valve_fail_badge(node: ReconciledNode, cx: float, bottom_y: float) -> str:
    """Render an 'FC' / 'FO' / 'FL' badge under the valve (P&ID failure position)."""
    fp_raw = str(node.attributes.get("failure_position", "")).strip()
    if not fp_raw:
        return ""
    # Normalise typical spellings: "F.C.", "FC", "fail closed", etc.
    lo = fp_raw.lower().replace(".", "").replace(" ", "_")
    if "close" in lo or lo in ("fc", "fail_close", "fail_closed"):
        code = "FC"
    elif "open" in lo or lo in ("fo", "fail_open"):
        code = "FO"
    elif "last" in lo or lo in ("fl", "fail_last", "fail_lock"):
        code = "FL"
    else:
        code = fp_raw[:3].upper()
    return (
        f'<rect x="{cx - 9:.1f}" y="{bottom_y + 2:.1f}" width="18" height="11" '
        f'fill="#ffffff" stroke="#111827" stroke-width="0.8" />'
        f'<text class="fail-code" x="{cx:.1f}" y="{bottom_y + 11:.1f}" text-anchor="middle">{code}</text>'
    )


def _glyph_safety_relief_valve(
    node: ReconciledNode,
    cx: float,
    cy: float,
    r: float,
    fill: str,
    dash: str,
    layout: LabelSlot | None,
    label: str,
) -> str:
    """ISA safety-relief valve: body diamond + spring symbol above + discharge arrow."""
    # Body (diamond).
    points = f"{cx},{cy - r} {cx + r},{cy} {cx},{cy + r} {cx - r},{cy}"
    # Spring: zig-zag above the diamond apex.
    sy = cy - r
    spring_h = 14
    spring_points = []
    for i in range(5):
        dx = (-1) ** i * 4
        spring_points.append(f"{cx + dx:.1f},{sy - i * spring_h / 4:.1f}")
    spring = (
        '<polyline fill="none" stroke="#111827" stroke-width="1" points="'
        + " ".join(spring_points)
        + '" />'
    )
    # Outlet arrow pointing up — signals "discharge to atmosphere".
    arrow_y0 = sy - spring_h - 2
    arrow_h = 8
    arrow = (
        f'<line x1="{cx:.1f}" y1="{arrow_y0:.1f}" x2="{cx:.1f}" y2="{arrow_y0 - arrow_h:.1f}" '
        f'stroke="#111827" stroke-width="1.2" />'
        f'<polygon points="{cx - 3:.1f},{arrow_y0 - arrow_h + 3:.1f} '
        f'{cx + 3:.1f},{arrow_y0 - arrow_h + 3:.1f} {cx:.1f},{arrow_y0 - arrow_h - 2:.1f}" '
        f'fill="#111827" />'
    )
    body = (
        f'<polygon class="valve srv" points="{points}" fill="{fill}"{dash} />'
    )
    # Set-pressure tag, if provided.
    sp = str(node.attributes.get("set_pressure", "")).strip()
    sp_text = ""
    if sp:
        sp_text = (
            f'<text class="srv-set" x="{cx + r + 4:.1f}" y="{cy + 3:.1f}" text-anchor="start">'
            f"{_xml_escape(sp)}</text>"
        )
    slot = layout or LabelSlot(x=cx, y=cy + r + 14, rotate=0.0)
    return body + spring + arrow + sp_text + _label_text_positioned(slot, label)


def _render_opc(
    cx: float,
    cy: float,
    size: float,
    direction: str,
    label: str,
    dash: str,
    layout: LabelSlot | None = None,
    subtitle: str = "",
) -> str:
    half = size / 2
    if direction == "out":
        pts = f"{cx - half},{cy - half} {cx + half / 2},{cy - half} {cx + half},{cy} {cx + half / 2},{cy + half} {cx - half},{cy + half}"
    else:
        pts = f"{cx + half},{cy - half} {cx - half / 2},{cy - half} {cx - half},{cy} {cx - half / 2},{cy + half} {cx + half},{cy + half}"
    slot = layout or LabelSlot(x=cx, y=cy + half + 12, rotate=0.0)
    text = _label_text_positioned(slot, label)
    if subtitle:
        sub_slot = LabelSlot(x=slot.x, y=slot.y + 12, rotate=slot.rotate)
        text += (
            f'<text class="sub" x="{sub_slot.x:.1f}" y="{sub_slot.y:.1f}" text-anchor="middle">'
            f"{subtitle}</text>"
        )
    return (
        f'<polygon class="opc" points="{pts}" fill="#fef2f2"{dash} />'
        + text
    )


# ---------------------------------------------------------------------------
# Label truncation + tooltip + equipment subtitle
# ---------------------------------------------------------------------------


_MAX_LABEL_CHARS = 24


def _truncate_label(label: str) -> str:
    """Cap the visible label; full text survives in the tooltip via _tooltip_text."""
    if not label:
        return ""
    if len(label) <= _MAX_LABEL_CHARS:
        return label
    return label[: _MAX_LABEL_CHARS - 1] + "…"


def _wrap_tooltip(svg_fragment: str, tooltip: str) -> str:
    """Wrap a node's SVG fragment in a <g> with a <title> child.

    Most browsers render <title> inside any SVG element as a native tooltip on
    hover — nearly zero rendering cost and a huge UX win for audit.
    """
    if not tooltip:
        return svg_fragment
    return f'<g class="node-wrap"><title>{tooltip}</title>{svg_fragment}</g>'


def _tooltip_text(node: ReconciledNode) -> str:
    """Build a multi-line tooltip showing label + kind + confidence + all attrs."""
    lines: list[str] = []
    lines.append(node.label or "")
    kind_line = f"kind: {node.kind}  confidence: {node.confidence}"
    lines.append(kind_line)
    if node.alternate_readings:
        lines.append(f"alt: {', '.join(node.alternate_readings[:4])}")
    if node.attributes:
        for k, v in sorted(node.attributes.items()):
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


_EQUIPMENT_SPEC_FIELDS = {
    "pump":            ["design_capacity", "design_head", "design_power"],
    "compressor":      ["design_capacity", "design_head", "design_power"],
    "heat_exchanger":  ["design_area", "design_duty", "nominal_diameter"],
    "tank":            ["capacity_ch1", "overall_height", "vessel_type"],
    "vessel":          ["capacity_ch1", "design_press_max", "vessel_type"],
    "column":          ["capacity_ch1", "overall_height"],
}


def _equipment_subtitle(node: ReconciledNode) -> str:
    """Pull one or two key specs from the attribute dict for a compact subtitle."""
    eq_class = str(node.attributes.get("equipment_class", ""))
    fields = _EQUIPMENT_SPEC_FIELDS.get(eq_class, [])
    bits: list[str] = []
    for key in fields:
        v = node.attributes.get(key)
        if v is None or v == "":
            continue
        bits.append(str(v))
        if len(bits) == 2:
            break
    return " · ".join(bits)


def _label_with_subtitle(slot: LabelSlot, label: str, subtitle: str) -> str:
    primary = _label_text_positioned(slot, label)
    if not subtitle:
        return primary
    if slot.rotate:
        # Don't stack a subtitle on a rotated label — it gets illegible.
        return primary
    sub_y = slot.y + 12
    return primary + (
        f'<text class="sub" x="{slot.x:.1f}" y="{sub_y:.1f}" text-anchor="middle">{subtitle}</text>'
    )


@dataclass
class _PreparedEdge:
    """Per-edge rendering state built once, consumed by crossing + draw passes."""

    edge: ReconciledEdge
    poly: list[tuple[int, int]]      # page-space polyline with ray-bbox attach points
    src: ReconciledNode | None
    dst: ReconciledNode | None
    line_id: str                     # effective tag (edge attr → OPC inheritance)
    stroke: str
    dash_pattern: str                # "none" for solid
    is_signal: bool
    is_inferred: bool                # loose-snap second-pass edge


def _prepare_edges(
    edges: list[ReconciledEdge],
    node_by_id: dict,
    *,
    orthogonal_inferred: bool = True,
) -> list[_PreparedEdge]:
    """Compute effective polylines (perimeter attach), inherit line_id, etc.

    `orthogonal_inferred=True` Manhattan-routes edges whose bootstrap
    `polyline_global` is empty — i.e. edges added at gt-edit time or otherwise
    synthesised. Bootstrap polylines stay verbatim either way (they reflect
    what the model traced, and any cleanup belongs to a separate pass).
    """
    out: list[_PreparedEdge] = []
    for edge in edges:
        poly = list(edge.polyline_global)
        src = node_by_id.get(edge.from_node)
        dst = node_by_id.get(edge.to_node)

        inferred = not poly
        if inferred:
            if src is None or dst is None:
                continue
            if orthogonal_inferred:
                # Manhattan route already places start/end on the perimeter
                # — no extra snap needed.
                poly = _orthogonal_route(src.bbox_global, dst.bbox_global)
            else:
                # Centre-to-centre fallback. Replace the centre points with
                # perimeter points on the ray toward the *other* endpoint —
                # using poly[0] as the snap target would be degenerate.
                src_centre = _bbox_centre_xy(
                    src.bbox_global.x, src.bbox_global.y,
                    src.bbox_global.w, src.bbox_global.h,
                )
                dst_centre = _bbox_centre_xy(
                    dst.bbox_global.x, dst.bbox_global.y,
                    dst.bbox_global.w, dst.bbox_global.h,
                )
                bb = src.bbox_global
                src_attach = _ray_bbox_exit(
                    bb.x, bb.y, bb.w, bb.h,
                    float(dst_centre[0]), float(dst_centre[1]),
                )
                bb = dst.bbox_global
                dst_attach = _ray_bbox_exit(
                    bb.x, bb.y, bb.w, bb.h,
                    float(src_centre[0]), float(src_centre[1]),
                )
                poly = [src_attach, dst_attach]
        else:
            # Bootstrap polyline — prepend / append perimeter points on rays
            # from each bbox centre toward the polyline's first / last point.
            if src is not None and poly:
                bb = src.bbox_global
                tgt = poly[0]
                poly = [_ray_bbox_exit(bb.x, bb.y, bb.w, bb.h, float(tgt[0]), float(tgt[1]))] + poly
            if dst is not None and poly:
                bb = dst.bbox_global
                tgt = poly[-1]
                poly = poly + [_ray_bbox_exit(bb.x, bb.y, bb.w, bb.h, float(tgt[0]), float(tgt[1]))]

        # Inherit line_id from the target OPC when the edge itself didn't carry one.
        line_id = str(edge.attributes.get("line_id", "")).strip()
        if not line_id and dst is not None and dst.kind == "opc":
            line_id = str(dst.attributes.get("line_id", "")).strip()
        if not line_id and src is not None and src.kind == "opc":
            line_id = str(src.attributes.get("line_id", "")).strip()

        lt = edge.line_type or "other"
        stroke, dash = _LINE_STROKE.get(lt, _LINE_STROKE["other"])
        out.append(
            _PreparedEdge(
                edge=edge,
                poly=poly,
                src=src,
                dst=dst,
                line_id=line_id,
                stroke=stroke,
                dash_pattern=dash,
                is_signal=lt.startswith("signal_") or lt == "instrument_capillary",
                is_inferred="inferred_loose" in str(edge.attributes.get("snap_note", "")),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Crossing detection (spec §7.2 — edges hopping over non-connecting pipes)
# ---------------------------------------------------------------------------


_HOP_ARC_R = 6.0                 # half-width of the hop arc in SVG pixels
_CROSSING_ENDPOINT_EPS = 8.0     # ignore "crossings" within this many px of an
                                 # endpoint of either participating segment
                                 # (those are legitimate T-junctions, not crossings)


@dataclass
class _Hop:
    """A hop point on an edge's polyline: which segment, at what fractional t."""

    seg_index: int
    t: float                     # parametric position along the segment, 0..1
    x: float                     # page-space crossing coordinate
    y: float


def _detect_crossings(prepared: list[_PreparedEdge]) -> dict[str, list[_Hop]]:
    """For each edge, return the list of hop points where other edges cross it.

    Priority: signal lines hop over process lines. Between two same-class lines
    a deterministic rule (edge.id tuple order) picks the hopper so the diagram
    is stable across re-renders.
    """
    hops: dict[str, list[_Hop]] = {pe.edge.id: [] for pe in prepared}
    n = len(prepared)
    for i in range(n):
        for j in range(i + 1, n):
            a = prepared[i]
            b = prepared[j]
            for ai, seg_a in enumerate(zip(a.poly, a.poly[1:], strict=False)):
                for bj, seg_b in enumerate(zip(b.poly, b.poly[1:], strict=False)):
                    inter = _segment_intersect(seg_a[0], seg_a[1], seg_b[0], seg_b[1])
                    if inter is None:
                        continue
                    px, py, ta, tb = inter
                    # Skip T-junction (intersection at or near a shared endpoint).
                    if _near_endpoint(px, py, a.poly, _CROSSING_ENDPOINT_EPS):
                        continue
                    if _near_endpoint(px, py, b.poly, _CROSSING_ENDPOINT_EPS):
                        continue
                    # Decide which edge hops.
                    hopper = _choose_hopper(a, b)
                    if hopper is a:
                        hops[a.edge.id].append(_Hop(seg_index=ai, t=ta, x=px, y=py))
                    else:
                        hops[b.edge.id].append(_Hop(seg_index=bj, t=tb, x=px, y=py))
    # Sort each edge's hops along its polyline (by segment index, then t).
    for lst in hops.values():
        lst.sort(key=lambda h: (h.seg_index, h.t))
    return hops


def _choose_hopper(a: _PreparedEdge, b: _PreparedEdge) -> _PreparedEdge:
    """Signal hops over process; otherwise use a stable edge.id comparison."""
    if a.is_signal and not b.is_signal:
        return a
    if b.is_signal and not a.is_signal:
        return b
    return a if a.edge.id < b.edge.id else b


def _segment_intersect(
    p0: tuple[int, int],
    p1: tuple[int, int],
    q0: tuple[int, int],
    q1: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Return (x, y, t_pq, t_rs) if the two open segments cross, else None."""
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = q0
    x4, y4 = q1
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # parallel or collinear — ignore (arguable, but hops on
                     # parallel overlaps look worse than ignoring them).
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    # Strictly inside both segments (open), not at endpoints — epsilon guard.
    eps = 1e-6
    if not (eps < t < 1 - eps and eps < u < 1 - eps):
        return None
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (px, py, t, u)


def _near_endpoint(px: float, py: float, poly: list[tuple[int, int]], eps: float) -> bool:
    for (ex, ey) in poly:
        if (px - ex) ** 2 + (py - ey) ** 2 <= eps ** 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Edge rendering (path with hop arcs + mid-segment line-id pill)
# ---------------------------------------------------------------------------


def _render_prepared_edge(
    pe: _PreparedEdge,
    hops: list[_Hop],
    project,
    nozzle_points_out: list[tuple[int, int, str]],
) -> str:
    """Emit a <path> for the edge with a small arc at each hop point.

    Also records nozzle points for `_nozzle_marker` to draw later, and appends
    a mid-segment line-id pill when `line_id` is known.
    """
    if not pe.poly:
        return ""

    dash_attr = f' stroke-dasharray="{pe.dash_pattern}"' if pe.dash_pattern != "none" else ""
    marker_attr = (
        f' marker-end="url(#arrow-{_marker_key_for(pe.edge.line_type or "other")})"'
        if pe.dst is not None and pe.edge.attributes.get("flow_direction") not in {"unknown", "conflicting"} else ""
    )
    width = "1.0" if pe.is_inferred else "1.5"
    opacity = ' opacity="0.6"' if pe.is_inferred else ""
    cross_note = ' data-cross-sheet="1"' if pe.edge.cross_sheet else ""

    d_commands = _build_path_d(pe.poly, hops, project)

    path_el = (
        f'<path class="edge" fill="none" stroke="{pe.stroke}" stroke-width="{width}" '
        f'd="{d_commands}"{dash_attr}{marker_attr}{opacity}{cross_note} />'
    )

    # Record nozzle tick points (only on endpoints that hit known equipment).
    if pe.src is not None:
        ox, oy = pe.poly[0]
        orient = _nozzle_orientation(pe.src.bbox_global, ox, oy)
        nozzle_points_out.append((ox, oy, orient))
    if pe.dst is not None:
        ox, oy = pe.poly[-1]
        orient = _nozzle_orientation(pe.dst.bbox_global, ox, oy)
        nozzle_points_out.append((ox, oy, orient))

    # Mid-segment line-id pill.
    pill = ""
    if pe.line_id:
        pill = _line_id_pill(pe.poly, pe.line_id, project)

    return path_el + pill


def _build_path_d(
    poly: list[tuple[int, int]],
    hops: list[_Hop],
    project,
) -> str:
    """Assemble the SVG `d` attribute for a polyline with small arcs at hops."""
    if not poly:
        return ""
    out: list[str] = []
    first_x, first_y = project(poly[0][0], poly[0][1])
    out.append(f"M {first_x:.1f} {first_y:.1f}")

    # Group hops by segment index for quick lookup.
    hops_by_seg: dict[int, list[_Hop]] = {}
    for h in hops:
        hops_by_seg.setdefault(h.seg_index, []).append(h)

    for i, (p, q) in enumerate(zip(poly, poly[1:], strict=False)):
        seg_hops = hops_by_seg.get(i, [])
        if not seg_hops:
            qx, qy = project(q[0], q[1])
            out.append(f"L {qx:.1f} {qy:.1f}")
            continue

        # Direction + length of this raw segment in PAGE space.
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len == 0:
            qx, qy = project(q[0], q[1])
            out.append(f"L {qx:.1f} {qy:.1f}")
            continue
        ux, uy = dx / seg_len, dy / seg_len          # unit vector along segment

        for h in sorted(seg_hops, key=lambda h: h.t):
            # Hop entry point = crossing − _HOP_ARC_R along the segment dir.
            ex_page = h.x - ux * _HOP_ARC_R
            ey_page = h.y - uy * _HOP_ARC_R
            xx_page = h.x + ux * _HOP_ARC_R
            xy_page = h.y + uy * _HOP_ARC_R
            ex, ey = project(ex_page, ey_page)
            xx, xy = project(xx_page, xy_page)
            # Sweep direction: arc curves "up" relative to segment orientation.
            # sweep-flag 0 vs 1 chooses concavity; pick 1 so the arc bulges to
            # the left of the travel direction for a consistent look.
            out.append(f"L {ex:.1f} {ey:.1f}")
            out.append(f"A {_HOP_ARC_R:.1f} {_HOP_ARC_R:.1f} 0 0 1 {xx:.1f} {xy:.1f}")
        # Finish the segment.
        qx, qy = project(q[0], q[1])
        out.append(f"L {qx:.1f} {qy:.1f}")

    return " ".join(out)


def _line_id_pill(
    poly: list[tuple[int, int]],
    line_id: str,
    project,
) -> str:
    """Render a small white rounded-rect label on the longest segment of the polyline."""
    if len(poly) < 2:
        return ""
    # Pick the longest segment in page space.
    best = 0
    best_len = 0.0
    for i, (p, q) in enumerate(zip(poly, poly[1:], strict=False)):
        d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
        if d > best_len:
            best_len = d
            best = i
    p = poly[best]
    q = poly[best + 1]
    midx_pg = (p[0] + q[0]) / 2
    midy_pg = (p[1] + q[1]) / 2
    cx, cy = project(midx_pg, midy_pg)
    text = _xml_escape(line_id)
    width = max(20.0, len(line_id) * _LABEL_CHAR_PX + 8)
    height = 12.0
    x0 = cx - width / 2
    y0 = cy - height / 2
    return (
        f'<rect class="line-id-bg" x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="3" />'
        f'<text class="line-id" x="{cx:.1f}" y="{cy + 3:.1f}" text-anchor="middle">{text}</text>'
    )


def _nozzle_orientation(bbox: BBox, px: int, py: int) -> str:
    """Given an attach point on the bbox perimeter, is the perimeter edge horizontal or vertical?"""
    # Which edge did the ray exit through? Closest-edge test.
    dl = abs(px - bbox.x)
    dr = abs(px - (bbox.x + bbox.w))
    dt = abs(py - bbox.y)
    db = abs(py - (bbox.y + bbox.h))
    mn = min(dl, dr, dt, db)
    if mn == dl or mn == dr:
        return "v"   # attach is on a vertical edge → tick is horizontal
    return "h"


def _nozzle_marker(x: int, y: int, orient: str, project) -> str:
    """Draw a short perpendicular tick on the equipment perimeter at (x, y)."""
    px, py = project(x, y)
    t = 4.0
    if orient == "h":
        # Perimeter edge is horizontal; tick is vertical.
        return (
            f'<line class="nozzle" x1="{px:.1f}" y1="{py - t:.1f}" '
            f'x2="{px:.1f}" y2="{py + t:.1f}" stroke="#111827" stroke-width="1.5" />'
        )
    return (
        f'<line class="nozzle" x1="{px - t:.1f}" y1="{py:.1f}" '
        f'x2="{px + t:.1f}" y2="{py:.1f}" stroke="#111827" stroke-width="1.5" />'
    )


def _bbox_centre_xy(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return (x + w // 2, y + h // 2)


def _orthogonal_route(src_bb: BBox, dst_bb: BBox) -> list[tuple[int, int]]:
    """Manhattan polyline between two bboxes, attach points on the perimeter.

    The dominant axis (horizontal vs vertical centre offset) decides which
    side of each bbox the pipe leaves from. A simple Z-route through a mid
    channel produces 2 points when the bboxes are co-linear on the secondary
    axis, otherwise 4. The returned start/end are exactly on the bbox
    perimeter, so `_ray_bbox_exit` can be skipped for these.
    """
    cx_s, cy_s = _bbox_centre_xy(src_bb.x, src_bb.y, src_bb.w, src_bb.h)
    cx_d, cy_d = _bbox_centre_xy(dst_bb.x, dst_bb.y, dst_bb.w, dst_bb.h)

    horiz_offset = abs(cx_d - cx_s)
    vert_offset = abs(cy_d - cy_s)

    if horiz_offset >= vert_offset:
        # Horizontal-dominant — leave / arrive on left or right perimeter.
        if cx_d > cx_s:
            x_s = src_bb.x + src_bb.w
            x_d = dst_bb.x
        else:
            x_s = src_bb.x
            x_d = dst_bb.x + dst_bb.w
        y_s, y_d = cy_s, cy_d
        if y_s == y_d:
            return [(x_s, y_s), (x_d, y_d)]
        mid_x = (x_s + x_d) // 2
        return [(x_s, y_s), (mid_x, y_s), (mid_x, y_d), (x_d, y_d)]
    else:
        # Vertical-dominant — leave / arrive on top or bottom perimeter.
        if cy_d > cy_s:
            y_s = src_bb.y + src_bb.h
            y_d = dst_bb.y
        else:
            y_s = src_bb.y
            y_d = dst_bb.y + dst_bb.h
        x_s, x_d = cx_s, cx_d
        if x_s == x_d:
            return [(x_s, y_s), (x_d, y_d)]
        mid_y = (y_s + y_d) // 2
        return [(x_s, y_s), (x_s, mid_y), (x_d, mid_y), (x_d, y_d)]


def _ray_bbox_exit(bx: int, by: int, bw: int, bh: int, tx: float, ty: float) -> tuple[int, int]:
    """Return the point on the bbox perimeter on the ray from the bbox centre to (tx, ty).

    Used so a pipe attaches to the equipment edge rather than crossing through
    the fill to its centre. If the ray has zero length (centre == target) we
    fall back to the centre itself.
    """
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    dx = tx - cx
    dy = ty - cy
    if dx == 0 and dy == 0:
        return (int(round(cx)), int(round(cy)))
    # Parametric: point = centre + t * (dx, dy). Find the smallest t > 0 where
    # the point exits the axis-aligned bbox through one of the four edges.
    t_candidates = []
    if dx > 0:
        t_candidates.append((bx + bw - cx) / dx)
    elif dx < 0:
        t_candidates.append((bx - cx) / dx)
    if dy > 0:
        t_candidates.append((by + bh - cy) / dy)
    elif dy < 0:
        t_candidates.append((by - cy) / dy)
    if not t_candidates:
        return (int(round(cx)), int(round(cy)))
    t = min(tc for tc in t_candidates if tc > 0)
    return (int(round(cx + t * dx)), int(round(cy + t * dy)))


def _marker_key_for(line_type: str) -> str:
    """Return the id suffix of the arrow marker that matches this line type."""
    if line_type == "process":
        return "process"
    if line_type in ("signal_electric", "signal_pneumatic", "instrument_capillary"):
        return "signal"
    return "other"


def _label_text(cx: float, cy: float, label: str, classname: str = "lbl") -> str:
    if not label:
        return ""
    return (
        f'<text class="{classname}" x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle">'
        f"{label}</text>"
    )


# ---------------------------------------------------------------------------
# Label collision avoidance (spec §7.2 — readable confidence-report visuals)
# ---------------------------------------------------------------------------


@dataclass
class LabelSlot:
    x: float
    y: float
    rotate: float = 0.0          # degrees; 0 means horizontal


# Heuristic label-box dimensions used for the greedy placement — we don't have
# a text-metrics library so this is an approximation good enough for collision
# avoidance at typical P&ID densities.
_LABEL_CHAR_PX = 5.8
_LABEL_LINE_HEIGHT_PX = 14
_LABEL_MAX_DROP_PX = 120        # how far we'll search below before rotating
_LABEL_ROTATE_ANGLE = 30.0


def _plan_labels(
    nodes: list[ReconciledNode],
    project,
    scale: float,
) -> dict[str, LabelSlot]:
    """Assign each node a label anchor that avoids overlapping earlier labels.

    Greedy: iterate nodes in reading order (top-to-bottom, then left-to-right),
    placing each label directly below the glyph. When the default slot overlaps
    an already-reserved label box, step down by one line-height until we find
    a free slot or exhaust `_LABEL_MAX_DROP_PX`, in which case we rotate.
    """
    occupied: list[tuple[float, float, float, float]] = []  # (x0, y0, x1, y1)
    layout: dict[str, LabelSlot] = {}

    def _anchor_for(node: ReconciledNode) -> tuple[float, float, int]:
        """Return (cx, base_y, step_direction) — step_direction is +1 for below, -1 for above.

        Vessels / tanks / columns / heat exchangers wear their tag ABOVE the
        glyph (P&ID convention — helps stack rows cleanly); pumps / compressors
        / unclassified equipment go BELOW; valves and instruments are handled
        in their own render paths.
        """
        bb = node.bbox_global
        cx, _ = project(bb.x + bb.w / 2, 0)
        _, bottom_y = project(0, bb.y + bb.h)
        _, top_y = project(0, bb.y)
        eq_class = str(node.attributes.get("equipment_class", ""))
        if node.kind == "equipment" and eq_class in ("tank", "vessel", "column", "heat_exchanger"):
            return cx, top_y - 12, -1
        return cx, bottom_y + 12, +1

    def _label_box(cx: float, cy: float, label: str) -> tuple[float, float, float, float]:
        half_w = max(8.0, len(label) * _LABEL_CHAR_PX / 2)
        return (cx - half_w, cy - _LABEL_LINE_HEIGHT_PX + 2, cx + half_w, cy + 2)

    def _overlaps(box: tuple[float, float, float, float]) -> bool:
        for ox0, oy0, ox1, oy1 in occupied:
            if box[2] < ox0 or box[0] > ox1 or box[3] < oy0 or box[1] > oy1:
                continue
            return True
        return False

    ordered = sorted(nodes, key=lambda n: (n.bbox_global.y, n.bbox_global.x))
    for n in ordered:
        if n.kind in ("line", "connection"):
            continue
        if not n.label:
            continue
        cx, base_y, step_dir = _anchor_for(n)
        # Visible display label governs the occupancy box width.
        label = _truncate_label(n.label)
        y = base_y
        max_steps = _LABEL_MAX_DROP_PX // _LABEL_LINE_HEIGHT_PX
        placed = False
        for _ in range(max_steps):
            box = _label_box(cx, y, label)
            if not _overlaps(box):
                occupied.append(box)
                layout[n.id] = LabelSlot(x=cx, y=y, rotate=0.0)
                placed = True
                break
            y += step_dir * _LABEL_LINE_HEIGHT_PX
        if not placed:
            layout[n.id] = LabelSlot(x=cx, y=base_y, rotate=_LABEL_ROTATE_ANGLE)
            # Rotated labels don't reserve a box — they dodge on a different axis.
    return layout


def _label_slot(
    layout: dict[str, LabelSlot] | None,
    node: ReconciledNode,
    default_cx: float,
    default_below_y: float,
) -> LabelSlot:
    if layout is not None and node.id in layout:
        return layout[node.id]
    return LabelSlot(x=default_cx, y=default_below_y + 12, rotate=0.0)


def _label_text_positioned(slot: LabelSlot, label: str, classname: str = "lbl") -> str:
    if not label:
        return ""
    if slot.rotate:
        return (
            f'<text class="{classname}" x="{slot.x:.1f}" y="{slot.y:.1f}" text-anchor="start" '
            f'transform="rotate({slot.rotate:.0f} {slot.x:.1f} {slot.y:.1f})">'
            f"{label}</text>"
        )
    return (
        f'<text class="{classname}" x="{slot.x:.1f}" y="{slot.y:.1f}" text-anchor="middle">'
        f"{label}</text>"
    )


def _count_kinds(nodes: Iterable[ReconciledNode]) -> dict[str, int]:
    counts = {"equipment": 0, "instrument": 0, "opc": 0, "other": 0}
    for n in nodes:
        if n.kind in counts:
            counts[n.kind] += 1
        else:
            counts["other"] += 1
    return counts


def _legend_panel(canvas_w: int, canvas_h: int) -> str:
    # Small key in the top-right showing the shape/colour vocabulary.
    rows = [
        ("rect", "#dbeafe", "equipment"),
        ("circle", "#f3e8ff", "instrument"),
        ("diamond", _VALVE_FILL["gate"], "valve"),
        ("pentagon", "#fef2f2", "off-page connector"),
        ("line", "#1f2937", "process pipe"),
        ("line-dashed", "#2563eb", "signal"),
    ]
    pad = 10
    row_h = 18
    width = 180
    height = pad * 2 + row_h * len(rows) + 8
    x = canvas_w - width - _MARGIN_PX
    y = _MARGIN_PX
    parts = [
        f'<rect class="key-bg" x="{x}" y="{y}" width="{width}" height="{height}" />',
    ]
    for i, (shape, col, name) in enumerate(rows):
        ry = y + pad + i * row_h
        sx = x + pad
        sy = ry + row_h / 2
        if shape == "rect":
            parts.append(f'<rect x="{sx}" y="{ry + 2}" width="14" height="10" fill="{col}" stroke="#111827" stroke-width="1" />')
        elif shape == "circle":
            parts.append(f'<circle cx="{sx + 7}" cy="{sy}" r="6" fill="{col}" stroke="#111827" stroke-width="1" />')
        elif shape == "diamond":
            parts.append(f'<polygon points="{sx + 7},{ry + 2} {sx + 14},{sy} {sx + 7},{ry + row_h - 4} {sx},{sy}" fill="{col}" stroke="#111827" stroke-width="1" />')
        elif shape == "pentagon":
            parts.append(f'<polygon points="{sx},{ry + 2} {sx + 10},{ry + 2} {sx + 14},{sy} {sx + 10},{ry + row_h - 4} {sx},{ry + row_h - 4}" fill="{col}" stroke="#111827" stroke-width="1" />')
        elif shape == "line":
            parts.append(f'<line x1="{sx}" y1="{sy}" x2="{sx + 14}" y2="{sy}" stroke="{col}" stroke-width="2" />')
        elif shape == "line-dashed":
            parts.append(f'<line x1="{sx}" y1="{sy}" x2="{sx + 14}" y2="{sy}" stroke="{col}" stroke-width="2" stroke-dasharray="3,2" />')
        parts.append(f'<text class="key-label" x="{sx + 22}" y="{sy + 4}">{name}</text>')
    return "\n".join(parts)


def _empty_svg(title: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="80" viewBox="0 0 400 80">'
        + _style_block()
        + f'<text class="title" x="20" y="30">{_xml_escape(title or "diagex extraction")}</text>'
        + '<text class="footer" x="20" y="55">(no nodes in graph)</text>'
        + "</svg>"
    )


def _defs_block() -> str:
    """SVG <defs> — arrow markers per line type (process, signal, other)."""
    # `context-stroke` would let one marker adapt to the host polyline's stroke
    # colour, but browser support is patchy; ship one marker per colour instead.
    markers = []
    for key, (stroke, _dash) in (
        ("process", _LINE_STROKE["process"]),
        ("signal", _LINE_STROKE["signal_electric"]),
        ("other", _LINE_STROKE["other"]),
    ):
        markers.append(
            f'<marker id="arrow-{key}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{stroke}" />'
            f'</marker>'
        )
    return f'<defs>{"".join(markers)}</defs>'


def _style_block() -> str:
    return (
        "<style>"
        'text { font-family: ui-monospace, SFMono-Regular, monospace; }'
        '.title { font-size: 14px; font-weight: 600; fill: #111827; }'
        '.footer { font-size: 10px; fill: #6b7280; }'
        '.lbl { font-size: 10px; fill: #111827; }'
        '.sub { font-size: 9px; fill: #6b7280; }'
        '.inst-tag { font-size: 10px; font-weight: 600; fill: #111827; }'
        '.txt { font-size: 10px; fill: #4b5563; }'
        '.actuator-mark { font-size: 9px; font-weight: 600; fill: #111827; }'
        '.fail-code { font-size: 8px; font-weight: 700; fill: #111827; }'
        '.srv-set { font-size: 9px; fill: #b91c1c; }'
        '.tb-title { font-size: 10px; font-weight: 700; fill: #111827; }'
        '.tb-row { font-size: 9px; fill: #374151; }'
        '.tb-bg { fill: #ffffff; stroke: #111827; stroke-width: 1; }'
        '.tb-grid { stroke: #d1d5db; stroke-width: 0.6; }'
        '.key-label { font-size: 10px; fill: #374151; }'
        '.key-bg { fill: rgba(255,255,255,0.92); stroke: #d1d5db; stroke-width: 1; }'
        '.eq, .inst, .valve, .opc, .misc { stroke: #111827; stroke-width: 1; }'
        '.node-wrap:hover { filter: brightness(0.96); }'
        '.line-id { font-size: 8px; font-weight: 600; fill: #111827; }'
        '.line-id-bg { fill: #ffffff; stroke: #6b7280; stroke-width: 0.6; }'
        '.nozzle { stroke: #111827; stroke-width: 1.5; }'
        "</style>"
    )


def _title_block(
    canvas_w: int,
    canvas_h: int,
    *,
    graph: ReconciledGraph,
    heading: str,
    metadata: dict,
) -> str:
    """Bottom-right title block with drawing metadata (ISA-5.1 / ISO 10628 convention).

    Shows: heading (drawing name/stem), run id, model, effort, tokens, timestamp,
    node/edge counts. Renders as a small bordered table.
    """
    width = 300
    rows = [
        ("diagex run",     str(metadata.get("run_id", "—"))),
        ("drawing",        heading[:46] if heading else (graph.source_path or "—")),
        ("model",          str(metadata.get("model", "—"))),
        ("effort",         str(metadata.get("effort", "—"))),
        (
            "tokens",
            format_tokens_millions(metadata["total_tokens"])
            if metadata.get("total_tokens") is not None
            else "—",
        ),
        ("timestamp",      str(metadata.get("timestamp", "—"))),
        ("nodes / edges",  f"{len(graph.nodes)} / {len(graph.edges)}"),
    ]
    header_h = 16
    row_h = 14
    height = header_h + row_h * len(rows) + 6
    x = canvas_w - width - _MARGIN_PX
    y = canvas_h - height - _MARGIN_PX - 20  # clear the footer counts text
    parts = [
        f'<rect class="tb-bg" x="{x}" y="{y}" width="{width}" height="{height}" />',
        f'<rect class="tb-bg" x="{x}" y="{y}" width="{width}" height="{header_h}" fill="#f3f4f6" />',
        f'<text class="tb-title" x="{x + 8}" y="{y + header_h - 4}">diagex · DEXPI extraction</text>',
    ]
    for i, (k, v) in enumerate(rows):
        ry = y + header_h + (i + 1) * row_h - 3
        parts.append(
            f'<line class="tb-grid" x1="{x}" y1="{y + header_h + i * row_h:.1f}" '
            f'x2="{x + width}" y2="{y + header_h + i * row_h:.1f}" />'
        )
        parts.append(
            f'<text class="tb-row" x="{x + 8}" y="{ry:.1f}">'
            f'{_xml_escape(k)}</text>'
        )
        parts.append(
            f'<text class="tb-row" x="{x + 120}" y="{ry:.1f}">'
            f'{_xml_escape(str(v))}</text>'
        )
    return "\n".join(parts)
