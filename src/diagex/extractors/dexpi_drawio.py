"""Render a ReconciledGraph as a draw.io (mxfile) document.

Sister of :mod:`diagex.extractors.dexpi_svg` — same input (a
``ReconciledGraph`` and optional run metadata) but the output is an
editable ``.drawio`` XML document that opens in diagrams.net with
each DEXPI element rendered using a real **"Process Engineering"
stencil** (``mxgraph.pid*`` shape namespace) rather than custom SVG.

Shape identifiers come from
:mod:`diagex.dexpi_schema` (``EquipmentSpec.drawio_shape`` /
``ValveSpec.drawio_shape`` / ``InstrumentSpec.drawio_mounting``) — adding
a class still happens in one place.

This is **export-only**. Reading ``.drawio`` files back into DEXPI is a
separate, much larger problem and is deliberately out of scope.

Coordinates: ``ReconciledGraph`` page-pixel space drops in directly as
mxGraph canvas units. Bootstrap polylines pass through unmodified;
inferred edges (no polyline) are Manhattan-routed by reusing
``dexpi_svg._prepare_edges`` so the two renderers stay in lockstep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as _saxutils_escape

from diagex.dexpi_schema import (
    equipment_spec_for,
    instrument_spec_for,
    valve_spec_for,
)
from diagex.extractors.dexpi_svg import _prepare_edges
from diagex.llm.cost import format_tokens_millions, total_tokens_from_summary
from diagex.vision.models import ReconciledGraph, ReconciledNode


# Labels go into value="..." attributes, so we must also escape quote chars —
# saxutils.escape only handles <, >, & by default.
def _xml_escape(s: str) -> str:
    return _saxutils_escape(s, {'"': "&quot;", "'": "&apos;"})

_FALLBACK_EQUIPMENT_STYLE = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#f9fafb;strokeColor=#111827;"
)
# OPCs render as the built-in mxgraph `step` shape — a 5-sided rectangle with
# a directional spike. The spike points the way the flow goes: east for
# outbound (flow leaves the page rightward), west for inbound. This matches
# the ISA "tag with arrow" convention that reviewers expect.
_FALLBACK_OPC_STYLE = (
    "shape=step;whiteSpace=wrap;html=1;"
    "fillColor=#fef3c7;strokeColor=#111827;fontSize=10;"
)
_INSTRUMENT_BASE_STYLE = (
    "shape=mxgraph.pid2inst.discInst;html=1;outlineConnect=0;"
    "align=center;dashed=0;aspect=fixed;"
)
_VALVE_BASE_STYLE = (
    "html=1;align=center;dashed=0;pointerEvents=1;"
    "verticalLabelPosition=bottom;verticalAlign=top;"
)
_EQUIPMENT_BASE_STYLE = (
    "html=1;dashed=0;outlineConnect=0;align=center;"
    "verticalLabelPosition=bottom;verticalAlign=top;"
)
# Low-confidence cells/edges are visually demoted via opacity rather than
# dashing — `dashed=1` is reserved for line-type semantics (signal lines,
# capillary, electrical power) so the two never collide on the canvas.
_LOW_CONF_NODE_STYLE = "opacity=55;"
_LOW_CONF_EDGE_STYLE = "opacity=45;"

# Edge styles per LineType. Matches the SVG renderer's stroke/dash palette
# so the two outputs are visually consistent.
_EDGE_STYLE: dict[str, str] = {
    "process": (
        "endArrow=classic;edgeStyle=orthogonalEdgeStyle;rounded=0;"
        "html=1;strokeColor=#1f2937;strokeWidth=1.5;"
    ),
    "signal_electric": (
        "endArrow=classic;edgeStyle=orthogonalEdgeStyle;rounded=0;"
        "html=1;strokeColor=#2563eb;strokeWidth=1.2;dashed=1;dashPattern=4 3;"
    ),
    "signal_pneumatic": (
        "endArrow=classic;edgeStyle=orthogonalEdgeStyle;rounded=0;"
        "html=1;strokeColor=#16a34a;strokeWidth=1.2;dashed=1;dashPattern=2 2;"
    ),
    "instrument_capillary": (
        "endArrow=classic;edgeStyle=orthogonalEdgeStyle;rounded=0;"
        "html=1;strokeColor=#9333ea;strokeWidth=1.2;dashed=1;dashPattern=1 2;"
    ),
    "electrical_power": (
        "endArrow=classic;edgeStyle=orthogonalEdgeStyle;rounded=0;"
        "html=1;strokeColor=#dc2626;strokeWidth=1.5;dashed=1;dashPattern=6 2 1 2;"
    ),
    "other": (
        "endArrow=classic;edgeStyle=orthogonalEdgeStyle;rounded=0;"
        "html=1;strokeColor=#6b7280;strokeWidth=1.2;dashed=1;dashPattern=3 3;"
    ),
}

# Translation from agent-emitted actuation/failure attributes to the
# parameter values drawio's pid2valves.valve stencil understands.
_ACTUATOR_MAP: dict[str, str] = {
    "pneumatic": "diaph",
    "diaphragm": "diaph",
    "balanced_diaphragm": "balDiaph",
    "motor": "motor",
    "electric_motor": "motor",
    "solenoid": "solenoid",
    "solenoid_manual_reset": "solenoidManRes",
    "hand": "man",
    "manual": "man",
    "hand_switch": "man",
    "spring": "spring",
    "weight": "weight",
    "hydraulic": "elHyd",
    "electro_hydraulic": "elHyd",
    "elhyd": "elHyd",
    "digital": "digital",
    "key": "key",
    "pilot": "pilot",
    "single_acting": "singActing",
    "double_acting": "dblActing",
    "powered": "powered",
}


_MARGIN_PX = 40


@dataclass
class DrawioRenderOptions:
    show_legend: bool = True
    show_title_block: bool = True
    orthogonal_inferred: bool = True


def render_graph_to_drawio(
    graph: ReconciledGraph,
    *,
    title: str = "",
    options: DrawioRenderOptions | None = None,
    metadata: dict | None = None,
) -> str:
    """Return a standalone ``.drawio`` XML document for ``graph``."""
    opts = options or DrawioRenderOptions()
    bounds = _graph_bounds(graph)
    if bounds is None:
        return _empty_mxfile(title)

    x_min, y_min, x_max, y_max = bounds
    page_w = max(850, (x_max - x_min) + 2 * _MARGIN_PX)
    page_h = max(1100, (y_max - y_min) + 2 * _MARGIN_PX)

    # Translate so the diagram sits inside the page with a margin.
    def project(x: float, y: float) -> tuple[float, float]:
        return (_MARGIN_PX + (x - x_min), _MARGIN_PX + (y - y_min))

    cells: list[str] = []
    next_id = _IdGen()

    node_by_id: dict[str, ReconciledNode] = {n.id: n for n in graph.nodes}
    drawio_id_for_node: dict[str, str] = {}

    # Vertices (nodes) ------------------------------------------------------
    for node in graph.nodes:
        cell_id = next_id.fresh(_prefixed_id("n", node.id))
        if node.kind in ("line", "connection"):
            continue   # geometry lives on edges
        if node.kind == "text":
            cells.append(_render_text_cell(cell_id, node, project))
        elif node.kind == "note":
            cells.append(_render_text_cell(cell_id, node, project, classname="note"))
        else:
            cells.append(_render_node_cell(cell_id, node, project))
        drawio_id_for_node[node.id] = cell_id

    # Edges -----------------------------------------------------------------
    prepared = _prepare_edges(
        graph.edges, node_by_id, orthogonal_inferred=opts.orthogonal_inferred
    )
    for pe in prepared:
        edge_cell_id = next_id.fresh(_prefixed_id("e", pe.edge.id))
        src_id = drawio_id_for_node.get(pe.edge.from_node)
        dst_id = drawio_id_for_node.get(pe.edge.to_node)
        if src_id is None or dst_id is None:
            continue       # ghost edge — already counted by other tools
        cells.append(_render_edge_cell(edge_cell_id, pe, src_id, dst_id, project))

    # Title block -----------------------------------------------------------
    if opts.show_title_block:
        cells.extend(_render_title_block(graph, title, metadata or {}, page_w, next_id))

    # Legend (line-type semantics) ------------------------------------------
    if opts.show_legend:
        cells.extend(_render_legend(graph, page_w, page_h, next_id))

    # Footer counts ---------------------------------------------------------
    cells.append(_render_footer(graph, page_w, page_h, next_id))

    return _wrap_mxfile(cells, page_w, page_h, title=title or graph.source_path)


def write_drawio(
    graph: ReconciledGraph,
    path: Path,
    *,
    title: str = "",
    options: DrawioRenderOptions | None = None,
    metadata: dict | None = None,
) -> Path:
    """Render ``graph`` to ``path`` (creates parent dirs). Returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = render_graph_to_drawio(graph, title=title, options=options, metadata=metadata)
    path.write_text(xml, encoding="utf-8")
    return path


def render_graph_json_to_drawio(
    graph_json_path: Path,
    *,
    title: str = "",
    orthogonal_inferred: bool = True,
) -> str:
    """Convenience loader: read a ``graph.json`` from disk and render it.

    Mirrors :func:`dexpi_svg.render_graph_json_to_svg`. Picks up sibling
    ``result.json`` metadata for the title block when present.
    """
    raw = json.loads(graph_json_path.read_text(encoding="utf-8"))
    graph = ReconciledGraph.model_validate(raw)
    metadata = _harvest_run_metadata(graph_json_path)
    options = DrawioRenderOptions(orthogonal_inferred=orthogonal_inferred)
    return render_graph_to_drawio(
        graph,
        title=title or graph.source_path,
        metadata=metadata,
        options=options,
    )


# ---------------------------------------------------------------------------
# Cell rendering
# ---------------------------------------------------------------------------


def _render_node_cell(
    cell_id: str, node: ReconciledNode, project
) -> str:
    bb = node.bbox_global
    x0, y0 = project(bb.x, bb.y)
    bw = max(20.0, float(bb.w))
    bh = max(20.0, float(bb.h))

    if node.kind == "equipment":
        style, x, y, w, h = _equipment_layout(node, x0, y0, bw, bh)
    elif node.kind == "instrument":
        style, x, y, w, h = _instrument_layout(node, x0, y0, bw, bh)
    elif node.kind == "opc":
        style, x, y, w, h = _opc_layout(node, x0, y0, bw, bh)
    else:
        style, x, y, w, h = _FALLBACK_EQUIPMENT_STYLE, x0, y0, bw, bh

    if node.confidence == "low":
        style = style + _LOW_CONF_NODE_STYLE

    label = _xml_escape(node.label or "")
    return _vertex_cell(cell_id, label, style, x, y, w, h)


def _fit_natural_into_bbox(
    natural_size: tuple[int, int],
    x0: float, y0: float, bw: float, bh: float,
) -> tuple[float, float, float, float]:
    """Scale ``natural_size`` to fit inside ``bw × bh`` preserving aspect, but
    never grow past the natural size; center the result within the bbox.

    Returns ``(x, y, w, h)`` in the same coord space as ``(x0, y0)``.
    """
    nat_w, nat_h = natural_size
    scale = min(bw / nat_w, bh / nat_h, 1.0)
    w = nat_w * scale
    h = nat_h * scale
    x = x0 + (bw - w) / 2.0
    y = y0 + (bh - h) / 2.0
    return x, y, w, h


def _equipment_layout(
    node: ReconciledNode, x0: float, y0: float, bw: float, bh: float,
) -> tuple[str, float, float, float, float]:
    valve_type = node.attributes.get("valve_type")
    if valve_type:
        return _valve_layout(node, x0, y0, bw, bh)
    eq_class = str(node.attributes.get("equipment_class", "")).strip().lower()
    spec = equipment_spec_for(eq_class) if eq_class else None
    shape = ""
    natural = None
    if spec is not None:
        if spec.subtype_hint_attr:
            hint = str(node.attributes.get(spec.subtype_hint_attr, "")).strip().lower()
            shape = spec.drawio_subtype_shapes.get(hint, "")
        if not shape:
            shape = spec.drawio_shape
        natural = spec.drawio_natural_size
    if not shape:
        return _FALLBACK_EQUIPMENT_STYLE, x0, y0, bw, bh

    style = f"{_EQUIPMENT_BASE_STYLE}shape={shape};"
    if natural is None:
        # Stencil meant to scale freely (tank, vessel, column, HX, …).
        return style, x0, y0, bw, bh
    style = style + "aspect=fixed;"
    x, y, w, h = _fit_natural_into_bbox(natural, x0, y0, bw, bh)
    return style, x, y, w, h


# Natural sizes for the parameterised pid2valves.valve stencil — taken from
# the diagrams.net sidebar palette: 100×60 without an actuator, 100×100 once
# an actuator is stacked above the body.
_VALVE_NATURAL_NO_ACTUATOR = (100, 60)
_VALVE_NATURAL_WITH_ACTUATOR = (100, 100)
# Pure-fitting valve aliases (rupture disc, basket strainer) live in their
# own libraries; they're tall-thin or rectangular and look fine at modest
# fixed sizes.
_FITTING_VALVE_NATURAL = (50, 60)


def _valve_layout(
    node: ReconciledNode, x0: float, y0: float, bw: float, bh: float,
) -> tuple[str, float, float, float, float]:
    vt = str(node.attributes.get("valve_type", "")).strip().lower()
    spec = valve_spec_for(vt)
    shape = spec.drawio_shape if spec is not None else ""
    if not shape:
        shape = "mxgraph.pid2valves.valve;valveType=gate"   # plain fallback

    actuator = _resolve_actuator(node, spec.drawio_default_actuator if spec else "none")
    if shape.startswith("mxgraph.pid2valves.valve"):
        if "actuator=" not in shape:
            shape = f"{shape};actuator={actuator}"
        natural = (
            _VALVE_NATURAL_WITH_ACTUATOR if actuator != "none"
            else _VALVE_NATURAL_NO_ACTUATOR
        )
    else:
        natural = _FITTING_VALVE_NATURAL

    style = f"{_VALVE_BASE_STYLE}shape={shape};aspect=fixed;"
    x, y, w, h = _fit_natural_into_bbox(natural, x0, y0, bw, bh)
    return style, x, y, w, h


# ISA-5.1 instrument bubbles are circles — drawio's pid2inst.discInst has a
# fixed 1:1 aspect already (declared aspect=fixed in the base style), so the
# only knob is "how big". 50×50 matches the sidebar default.
_INSTRUMENT_NATURAL = (50, 50)


def _instrument_layout(
    node: ReconciledNode, x0: float, y0: float, bw: float, bh: float,
) -> tuple[str, float, float, float, float]:
    fn = str(node.attributes.get("instrument_function", "")).strip().lower()
    spec = instrument_spec_for(fn)
    mounting = spec.drawio_mounting if spec is not None else "field"
    style = f"{_INSTRUMENT_BASE_STYLE}mounting={mounting};"
    x, y, w, h = _fit_natural_into_bbox(_INSTRUMENT_NATURAL, x0, y0, bw, bh)
    return style, x, y, w, h


_OPC_NATURAL = (40, 30)


def _opc_layout(
    node: ReconciledNode, x0: float, y0: float, bw: float, bh: float,
) -> tuple[str, float, float, float, float]:
    direction = str(node.attributes.get("direction", "in")).strip().lower()
    style = _FALLBACK_OPC_STYLE
    style += "direction=east;" if direction == "out" else "direction=west;"
    x, y, w, h = _fit_natural_into_bbox(_OPC_NATURAL, x0, y0, bw, bh)
    return style, x, y, w, h


def _resolve_actuator(node: ReconciledNode, default: str) -> str:
    raw = str(node.attributes.get("actuation", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return default
    return _ACTUATOR_MAP.get(raw, default)


def _render_edge_cell(
    cell_id: str,
    pe,
    src_id: str,
    dst_id: str,
    project,
) -> str:
    line_type = pe.edge.line_type or "other"
    base_style = _EDGE_STYLE.get(line_type, _EDGE_STYLE["other"])
    if pe.edge.attributes.get("flow_direction") in {"unknown", "conflicting"}:
        base_style = base_style.replace("endArrow=classic;", "endArrow=none;")
    if pe.edge.confidence == "low":
        base_style = base_style + _LOW_CONF_EDGE_STYLE
    label = _xml_escape(pe.line_id or "")

    # Project polyline waypoints. Strip the first/last because the edge
    # connects to the source/target cell endpoints — drawio computes those.
    interior_points = pe.poly[1:-1] if len(pe.poly) >= 2 else []
    points_xml = ""
    if interior_points:
        pts = []
        for x, y in interior_points:
            px, py = project(float(x), float(y))
            pts.append(f'<mxPoint x="{px:.1f}" y="{py:.1f}" />')
        points_xml = (
            '<Array as="points">' + "".join(pts) + "</Array>"
        )

    return (
        f'<mxCell id="{cell_id}" value="{label}" style="{base_style}" '
        f'edge="1" parent="1" source="{src_id}" target="{dst_id}">'
        f'<mxGeometry relative="1" as="geometry">{points_xml}</mxGeometry>'
        f"</mxCell>"
    )


def _render_text_cell(
    cell_id: str, node: ReconciledNode, project, classname: str = "text"
) -> str:
    bb = node.bbox_global
    x0, y0 = project(bb.x, bb.y)
    w = max(40.0, float(bb.w))
    h = max(16.0, float(bb.h))
    style = (
        "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;"
        "fontSize=10;strokeColor=none;fillColor=none;"
    )
    if classname == "note":
        style = style + "fontStyle=2;"
    return _vertex_cell(cell_id, _xml_escape(node.label or ""), style, x0, y0, w, h)


def _render_title_block(
    graph: ReconciledGraph,
    title: str,
    metadata: dict,
    page_w: float,
    next_id: _IdGen,
) -> list[str]:
    title_text = title or graph.source_path or "diagex extraction"
    bits = [title_text]
    if metadata.get("run_id"):
        bits.append(f"run: {metadata['run_id']}")
    if metadata.get("model"):
        bits.append(f"model: {metadata['model']}")
    if metadata.get("total_tokens") is not None:
        try:
            bits.append(
                f"tokens: {format_tokens_millions(int(metadata['total_tokens']))}"
            )
        except (TypeError, ValueError):
            pass
    label = _xml_escape("  ·  ".join(bits))
    cell_id = next_id.fresh("title")
    style = (
        "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;"
        "fontSize=12;fontStyle=1;strokeColor=none;fillColor=none;"
    )
    return [_vertex_cell(cell_id, label, style, 10, 10, page_w - 20, 24)]


# Legend rows: (LineType key, human label) — order matches reading priority.
_LEGEND_ROWS: list[tuple[str, str]] = [
    ("process", "Process"),
    ("signal_electric", "Electric signal"),
    ("signal_pneumatic", "Pneumatic signal"),
    ("instrument_capillary", "Capillary"),
    ("electrical_power", "Electrical power"),
    ("other", "Other / unknown"),
]


def _render_legend(
    graph: ReconciledGraph, page_w: float, page_h: float, next_id: _IdGen
) -> list[str]:
    """Render a small line-type / confidence legend in the bottom-left corner.

    Only entries actually present in the graph are emitted, so a process-only
    P&ID doesn't carry six unused signal-line rows.
    """
    line_types_used = {(e.line_type or "other") for e in graph.edges}
    rows = [(key, label) for key, label in _LEGEND_ROWS if key in line_types_used]
    has_low_conf = any(e.confidence == "low" for e in graph.edges) or any(
        n.confidence == "low" for n in graph.nodes
    )
    if not rows and not has_low_conf:
        return []

    box_w = 220.0
    row_h = 18.0
    pad = 10.0
    header_h = 22.0
    sample_w = 50.0
    extra = 1 if has_low_conf else 0
    box_h = header_h + (len(rows) + extra) * row_h + pad

    x0 = 10.0
    y0 = page_h - box_h - 36   # leave room for the footer line below
    out: list[str] = []

    container_id = next_id.fresh("legend")
    container_style = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;"
        "strokeColor=#9ca3af;fontSize=10;align=left;"
        "verticalAlign=top;spacingLeft=8;spacingTop=4;"
    )
    out.append(_vertex_cell(container_id, "Line types", container_style, x0, y0, box_w, box_h))

    for i, (key, label) in enumerate(rows):
        row_y = y0 + header_h + i * row_h + 2
        sample_id = next_id.fresh(f"legend-sample-{key}")
        out.append(
            _vertex_cell(sample_id, "", _legend_sample_style(key),
                         x0 + 8, row_y + 4, sample_w, 6)
        )
        text_id = next_id.fresh(f"legend-label-{key}")
        text_style = (
            "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;"
            "fontSize=10;strokeColor=none;fillColor=none;"
        )
        out.append(
            _vertex_cell(text_id, _xml_escape(label), text_style,
                         x0 + 8 + sample_w + 8, row_y, box_w - sample_w - 30, row_h - 4)
        )

    if has_low_conf:
        row_y = y0 + header_h + len(rows) * row_h + 2
        sample_id = next_id.fresh("legend-sample-lowconf")
        out.append(
            _vertex_cell(
                sample_id, "",
                "rounded=0;fillColor=#1f2937;strokeColor=none;opacity=55;",
                x0 + 8, row_y + 4, sample_w, 6,
            )
        )
        text_id = next_id.fresh("legend-label-lowconf")
        out.append(
            _vertex_cell(
                text_id, "Low-confidence (faded)",
                "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;"
                "fontSize=10;strokeColor=none;fillColor=none;",
                x0 + 8 + sample_w + 8, row_y, box_w - sample_w - 30, row_h - 4,
            )
        )

    return out


def _legend_sample_style(line_type: str) -> str:
    """Translate a LineType into a fillColor swatch style for the legend."""
    palette = {
        "process": "#1f2937",
        "signal_electric": "#2563eb",
        "signal_pneumatic": "#16a34a",
        "instrument_capillary": "#9333ea",
        "electrical_power": "#dc2626",
        "other": "#6b7280",
    }
    fill = palette.get(line_type, palette["other"])
    base = f"rounded=0;fillColor={fill};strokeColor=none;"
    # Tint dashed line-types lighter so they read as "non-process" at a glance.
    if line_type != "process":
        base = base + "opacity=70;"
    return base


def _render_footer(
    graph: ReconciledGraph, page_w: float, page_h: float, next_id: _IdGen
) -> str:
    counts = {"equipment": 0, "instrument": 0, "opc": 0}
    for n in graph.nodes:
        if n.kind in counts:
            counts[n.kind] += 1
    bits = [
        f"equipment {counts['equipment']}",
        f"instruments {counts['instrument']}",
        f"opcs {counts['opc']}",
        f"edges {len(graph.edges)}",
    ]
    label = _xml_escape("  ·  ".join(bits))
    cell_id = next_id.fresh("footer")
    style = (
        "text;html=1;align=right;verticalAlign=middle;whiteSpace=wrap;"
        "fontSize=10;strokeColor=none;fillColor=none;fontStyle=2;"
    )
    return _vertex_cell(cell_id, label, style, 10, page_h - 28, page_w - 20, 18)


# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------


def _vertex_cell(
    cell_id: str,
    value: str,
    style: str,
    x: float,
    y: float,
    w: float,
    h: float,
) -> str:
    return (
        f'<mxCell id="{cell_id}" value="{value}" style="{style}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'as="geometry" />'
        f"</mxCell>"
    )


def _wrap_mxfile(cells: list[str], page_w: float, page_h: float, *, title: str) -> str:
    diagram_name = _xml_escape(title)
    body = "".join(cells)
    return (
        '<mxfile host="diagex" type="device" compressed="false">'
        f'<diagram id="diagex-pid" name="{diagram_name}">'
        '<mxGraphModel dx="800" dy="600" grid="1" gridSize="10" '
        'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        f'pageScale="1" pageWidth="{int(page_w)}" pageHeight="{int(page_h)}" '
        'math="0" shadow="0">'
        '<root>'
        '<mxCell id="0" />'
        '<mxCell id="1" parent="0" />'
        f"{body}"
        '</root>'
        '</mxGraphModel>'
        '</diagram>'
        '</mxfile>'
    )


def _empty_mxfile(title: str) -> str:
    return _wrap_mxfile([], 850, 1100, title=title or "diagex extraction")


# ---------------------------------------------------------------------------
# Misc
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
        for x, y in e.polyline_global:
            xs_lo.append(x)
            ys_lo.append(y)
            xs_hi.append(x)
            ys_hi.append(y)
    if not xs_lo:
        return None
    return (min(xs_lo), min(ys_lo), max(xs_hi), max(ys_hi))


def _harvest_run_metadata(graph_json_path: Path) -> dict:
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
        }
    except Exception:
        return {}


_SLUG_TRANS = str.maketrans({c: "_" for c in "<>&\"' \t/\\:;,()[]{}#%?"})


def _slugify(s: str) -> str:
    out = (s or "").translate(_SLUG_TRANS)
    return out or "x"


def _prefixed_id(prefix: str, raw_id: str) -> str:
    """Build an mxCell id with a stable prefix without doubling it.

    ``ReconciledNode.id`` and ``ReconciledEdge.id`` already carry their own
    short prefix (e.g. ``n-abc123`` / ``e-abc123``). Naively wrapping again
    yields ``n-n-abc123``; this helper keeps the result clean.
    """
    slug = _slugify(raw_id)
    head = f"{prefix}-"
    if slug.startswith(head):
        return slug
    return f"{head}{slug}"


class _IdGen:
    """Allocator for unique mxCell ids — keeps human-readable prefixes."""

    def __init__(self) -> None:
        self._seen: set[str] = {"0", "1"}

    def fresh(self, base: str) -> str:
        if base not in self._seen:
            self._seen.add(base)
            return base
        n = 2
        while f"{base}-{n}" in self._seen:
            n += 1
        ident = f"{base}-{n}"
        self._seen.add(ident)
        return ident
