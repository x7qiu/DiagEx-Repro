"""Deterministic line topology for evidence-v2.

The model detects symbols; this module owns line geometry and connectivity.
Only connectable engineering nodes can become edge endpoints.  Text evidence,
notes, frames, and title blocks are never candidates for snapping.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from pydantic import BaseModel, Field

from diagex.vision.evidence import (
    PageEvidence,
    PathEvidence,
    VisualLineStyle,
    classify_pdf_dash_pattern,
    stable_evidence_id,
)
from diagex.vision.legend_models import LegendEntry, LegendPack
from diagex.vision.line_detection import LineDetectionResult, detect_lines, extract_raster_paths
from diagex.vision.models import BBox, LineType, ReconciledEdge, ReconciledNode
from diagex.vision.vector_geometry import (
    SCENE_VERSION,
    SymbolContour,
    box_polygon,
    inside,
    intersections,
    nozzle_contact,
    path_vertices,
    route_direction,
    segment_key,
    symbol_contours,
    vector_arrows,
)

TOPOLOGY_VERSION = "4.1.0"
TOPOLOGY_DEPENDENT_STAGES = ("topology", "line_evidence", "page_graph", "assembly", "export")

_CONNECTABLE_KINDS = {"equipment", "instrument", "opc"}


class LineStyleEvidence(BaseModel):
    """Derived appearance evidence for a vector line run.

    The style says what the marks look like, not what the line means. The page
    solver combines this with the drawing legend to assign an engineering
    ``LineType``.
    """

    id: str
    page_index: int
    visual_style: VisualLineStyle
    confidence: float
    source: str
    bbox: BBox
    points: list[tuple[int, int]]
    source_path_ids: list[str] = Field(default_factory=list)
    dash_lengths: list[float] = Field(default_factory=list)
    gap_lengths: list[float] = Field(default_factory=list)


class TopologyResult(BaseModel):
    page_index: int
    edges: list[ReconciledEdge] = Field(default_factory=list)
    candidate_path_ids: list[str] = Field(default_factory=list)
    used_path_ids: list[str] = Field(default_factory=list)
    unattached_node_ids: list[str] = Field(default_factory=list)
    ambiguities: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raster_fallback_used: bool = False
    line_style_evidence: list[LineStyleEvidence] = Field(default_factory=list)


class LegendLineSignature(BaseModel):
    signature: str
    visual_style: VisualLineStyle
    line_type: LineType
    legend_labels: list[str] = Field(default_factory=list)
    source_path_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class LegendLineProfile(BaseModel):
    """Project-specific mapping from PDF stroke signatures to semantics."""

    signatures: list[LegendLineSignature] = Field(default_factory=list)

    def match(self, path: PathEvidence) -> LegendLineSignature | None:
        signature = _path_signature(path)
        exact = next((item for item in self.signatures if item.signature == signature), None)
        if exact is not None:
            return exact
        # Fragment boundaries vary with route length and CAD export clipping.
        # A visual-style fallback is safe only when the project legend maps
        # that style to exactly one engineering meaning.
        style = _path_visual_style(path)
        same_style = [item for item in self.signatures if item.visual_style == style]
        line_types = {item.line_type for item in same_style}
        if len(line_types) != 1 or not same_style:
            return None
        template = max(same_style, key=lambda item: item.confidence)
        return template.model_copy(
            update={
                "signature": signature,
                "confidence": min(template.confidence, 0.72),
            }
        )


@dataclass(frozen=True)
class _Segment:
    a: tuple[float, float]
    b: tuple[float, float]
    path_id: str
    line_type: LineType
    origin: str
    visual_style: VisualLineStyle = "unknown"
    style_confidence: float = 0.0
    source_path_ids: tuple[str, ...] = ()
    legend_signature: str | None = None
    legend_labels: tuple[str, ...] = ()
    legend_confidence: float = 0.0

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)


@dataclass
class _GraphArc:
    target: int
    length: float
    path_id: str
    line_type: LineType
    visual_style: VisualLineStyle
    style_confidence: float
    source_path_ids: tuple[str, ...]
    legend_signature: str | None
    legend_labels: tuple[str, ...]
    legend_confidence: float


@dataclass
class _UnionFind:
    parent: list[int] = field(default_factory=list)
    rank: list[int] = field(default_factory=list)

    def add(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def build_page_topology(
    *,
    page: PageEvidence,
    nodes: list[ReconciledNode],
    raster_image: Any | None = None,
    legend_line_profile: LegendLineProfile | None = None,
    requested_pairs: set[frozenset[str]] | None = None,
    detected_lines: LineDetectionResult | None = None,
) -> TopologyResult:
    detected_lines = detected_lines or detect_lines(
        page=page, image=raster_image, raster_extractor=extract_raster_paths)
    if detected_lines.page_index != page.page_index:
        raise ValueError("Line evidence belongs to a different page")
    paths = detected_lines.paths
    warnings = list(detected_lines.warnings)
    raster_used = detected_lines.raster_fallback_used

    connectable = sorted(
        (
            node
            for node in nodes
            if node.page_index == page.page_index and node.kind in _CONNECTABLE_KINDS
            and node.attributes.get("representation_role") != "controller_wiring_symbol"
        ),
        key=lambda node: node.id,
    )
    contours = symbol_contours(page, connectable)
    from diagex.vision.native_hierarchy import build_native_hierarchy

    hierarchy = build_native_hierarchy(page, connectable)
    arrows = vector_arrows(page, connectable)
    owned_segments = {key for contour in contours.values() for key in contour.segments}
    owned_segments.update(key for arrow in arrows for key in arrow.segments)
    scope_segments = {
        segment_key(s[0], s[1], s[2]) for a in hierarchy.assemblies for s in a.boundary_segments
    }
    scope_segments.update(
        segment_key(s[0], s[1], s[2])
        for b in hierarchy.text_bindings
        for s in b.get("owned_segments", [])
    )
    owned_segments.update(scope_segments)
    segments, style_evidence = _candidate_segments(
        page,
        paths,
        legend_line_profile=legend_line_profile,
        owned_segments=owned_segments,
    )
    result = TopologyResult(
        page_index=page.page_index,
        candidate_path_ids=sorted({segment.path_id for segment in segments}),
        warnings=warnings,
        raster_fallback_used=raster_used,
        line_style_evidence=style_evidence,
    )
    if not segments:
        result.warnings.append("no usable line primitives were found")
        result.unattached_node_ids = sorted(
            node.id for node in nodes if node.kind in _CONNECTABLE_KINDS
        )
        return result

    graph = _build_graph(page=page, segments=segments, nodes=connectable, contours=contours)

    vertex_points, adjacency, attachments, path_origins, nozzle_ports = graph
    result.ambiguities.extend(
        _separate_unmarked_crossings(page, vertex_points, adjacency, attachments)
    )
    uncertain_curves = _cyclic_curve_paths(
        adjacency, {p.id for p in page.paths if p.primitive == "curve"}
    )
    attached_ids = {node_id for values in attachments.values() for node_id in values}
    result.unattached_node_ids = sorted(
        node.id for node in connectable if node.id not in attached_ids
    )

    node_by_id = {node.id: node for node in connectable}
    used_paths: set[str] = set()
    invalid_instances: dict[tuple[str, ...], dict[str, Any]] = {}
    for vertices in _connected_components(adjacency):
        component_edges: list[ReconciledEdge] = []
        terminals = sorted(
            {(node_id, vertex) for vertex in vertices for node_id in attachments.get(vertex, [])}
        )
        terminal_nodes = sorted({node_id for node_id, _ in terminals})
        if len(terminal_nodes) < 2:
            continue

        pair_paths: list[tuple[float, str, str, list[int], list[_GraphArc]]] = []
        for index, (left_id, left_vertex) in enumerate(terminals):
            distances, predecessors = _dijkstra(
                left_vertex, adjacency, vertices, stops=set(attachments)
            )
            for right_id, right_vertex in terminals[index + 1 :]:
                if left_id == right_id or right_vertex not in distances:
                    continue
                vertex_path, arcs = _restore_path(left_vertex, right_vertex, predecessors)
                if vertex_path:
                    pair_paths.append(
                        (distances[right_vertex], left_id, right_id, vertex_path, arcs)
                    )

        # More than two entities can share a pipe network through tees.  A
        # minimum spanning tree preserves reachability without fabricating the
        # all-pairs triangle that a naive component-to-edge conversion creates.
        terminal_uf = _UnionFind()
        terminal_index = {node_id: terminal_uf.add() for node_id in terminal_nodes}
        for _, left_id, right_id, vertex_path, arcs in sorted(
            pair_paths, key=lambda row: (row[0], row[1], row[2], row[3])
        ):
            left_root = terminal_uf.find(terminal_index[left_id])
            right_root = terminal_uf.find(terminal_index[right_id])
            if requested_pairs is not None and frozenset((left_id, right_id)) not in requested_pairs:
                continue
            if requested_pairs is None and left_root == right_root:
                continue
            path_ids = [arc.path_id for arc in arcs]
            used_paths.update(path_ids)
            line_type = _majority_line_type(arcs)
            visual_style = _majority_visual_style(arcs)
            style_confidence = _route_style_confidence(arcs, visual_style)
            source_path_ids = sorted(
                {source_id for arc in arcs for source_id in (arc.source_path_ids or (arc.path_id,))}
            )
            style_evidence_ids = sorted(
                {arc.path_id for arc in arcs if arc.path_id.startswith("sty-")}
            )
            legend_labels = sorted({label for arc in arcs for label in arc.legend_labels})
            legend_signatures = sorted(
                {arc.legend_signature for arc in arcs if arc.legend_signature}
            )
            legend_confidence = max((arc.legend_confidence for arc in arcs), default=0.0)
            polyline = [
                (int(round(vertex_points[vertex][0])), int(round(vertex_points[vertex][1])))
                for vertex in vertex_path
            ]
            confidence = (
                "high"
                if path_ids and all(path_origins.get(pid) == "pdf_vector" for pid in path_ids)
                else "medium"
            )
            score = 0.9 if confidence == "high" else 0.7
            direction, arrow_ids = route_direction(
                polyline, arrows, max(2.0, min(page.width, page.height) * 0.0008)
            )
            if direction == "reverse":
                left_id, right_id = right_id, left_id
                vertex_path = list(reversed(vertex_path))
                polyline.reverse()
                direction = "forward"
            failures = []
            if not arcs or len(set(polyline)) < 2:
                failures.append("route_has_no_length")
            if not source_path_ids:
                failures.append("missing_source_segments")
            if any(len(set(attachments[v])) != 1 for v in (vertex_path[0], vertex_path[-1])):
                failures.append("ambiguous_port_ownership")
            if direction == "conflicting":
                failures.append("conflicting_native_arrows")
            if uncertain_curves.intersection(source_path_ids):
                failures.append("curve_in_unassigned_cyclic_network")
            left_contour, right_contour = contours.get(left_id), contours.get(right_id)
            if left_contour and right_contour and left_contour.symbol_id == right_contour.symbol_id:
                failures.append("same_physical_symbol")
            for nid in (left_id, right_id):
                if nid not in contours:
                    failures.append("unsupported_physical_port")
                if node_by_id[nid].attributes.get("native_symbol_candidate"):
                    failures.append("unresolved_symbol_identity")
            ports = [
                {
                    "node_id": node_id,
                    "symbol_id": contours[node_id].symbol_id if node_id in contours else None,
                    "identity_status": "unresolved"
                    if node_by_id[node_id].attributes.get("native_symbol_candidate")
                    else "observed",
                    "point": point,
                    "basis": "native_nozzle"
                    if (node_id, tuple(point)) in nozzle_ports
                    else contours[node_id].basis
                    if node_id in contours
                    else "symbol_boundary",
                    "nozzle_path_ids": nozzle_ports.get((node_id, tuple(point)), []),
                    "outline_path_ids": contours[node_id].path_ids if node_id in contours else [],
                    "outline_segments": sorted(contours[node_id].segments)
                    if node_id in contours
                    else [],
                    "polygon": contours[node_id].points if node_id in contours else [],
                }
                for node_id, point in [(left_id, polyline[0]), (right_id, polyline[-1])]
            ]
            proof = {
                "version": TOPOLOGY_VERSION,
                "scene_version": SCENE_VERSION,
                "non_connective_segments": sorted(
                    s for s in owned_segments if s[0] in source_path_ids
                ),
                "status": "uncertain" if failures else "verified",
                "failures": failures,
                "ports": ports,
                "source_path_ids": source_path_ids,
                "uncertain_outline_path_ids": sorted(
                    uncertain_curves.intersection(source_path_ids)
                ),
                "junction_policy": "compatible_endpoints_and_tees; unsplit_crossings_do_not_join",
            }
            edge_id = _stable_edge_id(page.page_index, left_id, right_id, polyline, line_type)
            edge = ReconciledEdge(
                id=edge_id,
                from_node=left_id,
                to_node=right_id,
                line_type=line_type,
                polyline_global=polyline,
                cross_sheet=False,
                confidence=confidence,
                source_evidence_ids=source_path_ids,
                system_confidence=score,
                system_confidence_level=confidence,
                attributes={
                    "route_evidence": proof,
                    "flow_direction": direction,
                    "direction_source": "native_vector_arrow" if arrow_ids else "unknown",
                    "direction_evidence_ids": arrow_ids,
                    "topology_source": "deterministic_raster"
                    if raster_used
                    else "deterministic_vector",
                    "endpoint_labels": [
                        node_by_id[left_id].label,
                        node_by_id[right_id].label,
                    ],
                    "visual_style": visual_style,
                    "visual_style_confidence": round(style_confidence, 3),
                    "style_evidence_ids": style_evidence_ids,
                    "legend_line_labels": legend_labels,
                    "legend_line_signatures": legend_signatures,
                    "legend_line_confidence": round(legend_confidence, 3),
                },
            )
            checked_failures = route_evidence_failures(edge, page)
            if {"same_physical_symbol", "route_has_no_length"}.intersection(checked_failures):
                key = tuple(sorted((left_id, right_id)))
                conflict = invalid_instances.setdefault(
                    key,
                    {
                        "type": "physical_instance_geometry",
                        "page_index": page.page_index,
                        "node_ids": list(key),
                        "status": "unresolved",
                        "rejected_routes": [],
                    },
                )
                conflict["rejected_routes"].append(
                    {
                        "polyline_global": polyline,
                        "source_path_ids": source_path_ids,
                        "failures": checked_failures,
                    }
                )
                continue
            if checked_failures:
                edge.attributes["route_evidence"].update(
                    status="uncertain", failures=checked_failures
                )
                edge.attributes.update(
                    provisional_review_only=True,
                    requires_human_review=True,
                    review_reason="; ".join(checked_failures),
                )
                edge.system_confidence, edge.confidence, edge.system_confidence_level = (
                    0.35,
                    "low",
                    "low",
                )
            else:
                terminal_uf.union(left_root, right_root)
            component_edges.append(edge)

        result.edges.extend(_group_route_alternatives(component_edges))

    result.ambiguities.extend(invalid_instances.values())
    result.edges = sorted(
        {edge.id: edge for edge in result.edges}.values(), key=lambda edge: edge.id
    )
    result.used_path_ids = sorted(used_paths)
    used_segments = [segment for segment in segments if segment.path_id in used_paths]
    result.ambiguities.extend(_crossing_candidates(page, used_segments, nodes=connectable))
    return result


def _group_route_alternatives(edges: list[ReconciledEdge]) -> list[ReconciledEdge]:
    """One decision for competing routes within one connected native network.

    Native port coordinates distinguish separate taps on the same equipment.
    Unsupported box contacts have no physical port identity and are alternatives,
    not independent proposed connections. Every route retains its complete proof.
    """
    groups: dict[tuple, list[ReconciledEdge]] = defaultdict(list)
    accepted = []
    for edge in edges:
        if not edge.attributes.get("provisional_review_only"):
            accepted.append(edge)
            continue
        ports = edge.attributes["route_evidence"]["ports"]
        key = tuple(
            sorted((p["node_id"], tuple(p["point"]) if p.get("symbol_id") else ()) for p in ports)
        )
        groups[key].append(edge)
    for alternatives in groups.values():
        alternatives.sort(
            key=lambda e: (
                len(e.attributes["route_evidence"]["failures"]),
                sum(
                    math.dist(a, b)
                    for a, b in zip(e.polyline_global, e.polyline_global[1:], strict=False)
                ),
                e.id,
            )
        )
        representative = alternatives[0].model_copy(deep=True)
        if len(alternatives) > 1:
            representative.attributes["route_alternatives"] = [
                edge.model_dump(mode="json") for edge in alternatives
            ]
        accepted.append(representative)
    return accepted


def _candidate_segments(
    page: PageEvidence,
    paths: list[PathEvidence],
    *,
    legend_line_profile: LegendLineProfile | None = None,
    owned_segments: set[tuple] | None = None,
) -> tuple[list[_Segment], list[LineStyleEvidence]]:
    minimum = max(8.0, min(page.width, page.height) * 0.0015)
    border_margin = max(8, int(min(page.width, page.height) * 0.004))
    partly_owned = {key[0] for key in (owned_segments or set())}
    derived_paths, style_evidence, consumed_path_ids = _infer_fragmented_vector_styles(
        page,
        [path for path in paths if path.id not in partly_owned],
    )
    out: list[_Segment] = []
    for path in [*paths, *derived_paths]:
        if path.id in consumed_path_ids:
            continue
        if _is_document_structure_path(path, page):
            continue
        if path.closed or path.primitive in {"rect", "quad"}:
            continue
        if path.primitive not in {"line", "curve", "raster_line"}:
            continue
        vertices = path_vertices(path)
        for first, second in zip(vertices, vertices[1:], strict=False):
            if segment_key(path.id, first, second) in (owned_segments or set()):
                continue
            legend_match = (
                legend_line_profile.match(path) if legend_line_profile is not None else None
            )
            segment = _Segment(
                a=(float(first[0]), float(first[1])),
                b=(float(second[0]), float(second[1])),
                path_id=path.id,
                line_type=_line_type_for_path(path, legend_line_profile),
                origin=path.origin,
                visual_style=_path_visual_style(path),
                style_confidence=_path_style_confidence(path),
                source_path_ids=tuple(path.source_path_ids or [path.id]),
                legend_signature=legend_match.signature if legend_match else None,
                legend_labels=tuple(legend_match.legend_labels) if legend_match else (),
                legend_confidence=legend_match.confidence if legend_match else 0.0,
            )
            if segment.length < (0.5 if path.primitive == "curve" else minimum):
                continue
            if _is_page_frame(segment, page, border_margin):
                continue
            out.append(segment)
    return out, style_evidence


def _is_document_structure_path(path: PathEvidence, page: PageEvidence) -> bool:
    """Exclude obvious sheet furniture before it can become topology."""
    center_x = path.bbox.x + path.bbox.w / 2
    center_y = path.bbox.y + path.bbox.h / 2
    if center_x >= page.width * 0.55 and center_y >= page.height * 0.80:
        return True
    return False


def _line_type_for_path(
    path: PathEvidence,
    legend_line_profile: LegendLineProfile | None = None,
) -> LineType:
    if legend_line_profile is not None:
        matched = legend_line_profile.match(path)
        if matched is not None:
            return matched.line_type
    # Appearance is not semantics. Keep styled vector routes neutral until the
    # page solver maps them through the document's legend.
    if _path_visual_style(path) in {"dashed", "dotted", "dash_dot"}:
        return "other"
    return "process"


def learn_legend_line_profile(
    *,
    pages: list[PageEvidence],
    legend_pack: LegendPack,
) -> LegendLineProfile:
    """Learn exact PDF stroke signatures from project legend rows.

    A signature is accepted only when every matching legend sample agrees on
    the same engineering line type.  Coarse appearance alone (for example,
    merely "dashed") never resolves two project legend classes.
    """
    pages_by_index = {page.page_index: page for page in pages}
    derived_by_page = {
        page.page_index: _infer_fragmented_vector_styles(page, page.paths)[0] for page in pages
    }
    votes: dict[str, list[tuple[LineType, str, str, VisualLineStyle]]] = defaultdict(list)
    for entry in legend_pack.entries:
        line_type = _legend_entry_line_type(entry)
        if entry.kind != "line" or line_type is None or entry.source_page_index is None:
            continue
        page = pages_by_index.get(entry.source_page_index)
        if page is None:
            continue
        region = _legend_sample_region(entry, page)
        if region is None:
            continue
        for path in [*page.paths, *derived_by_page.get(page.page_index, [])]:
            if not _path_is_legend_sample(path, region):
                continue
            signature = _path_signature(path)
            if signature == "solid" and line_type != "process":
                # A short solid table border near a signal-row label is not a
                # reliable signal sample. Exact patterned signatures survive.
                continue
            votes[signature].append((line_type, entry.label, path.id, _path_visual_style(path)))

    signatures: list[LegendLineSignature] = []
    for signature, rows in sorted(votes.items()):
        line_types = {row[0] for row in rows}
        if len(line_types) != 1:
            continue
        line_type = next(iter(line_types))
        labels = sorted({row[1] for row in rows})
        path_ids = sorted({row[2] for row in rows})
        confidence = min(0.98, 0.78 + min(0.16, 0.04 * (len(path_ids) - 1)))
        signatures.append(
            LegendLineSignature(
                signature=signature,
                visual_style=rows[0][3],
                line_type=line_type,
                legend_labels=labels,
                source_path_ids=path_ids,
                confidence=confidence,
            )
        )
    return LegendLineProfile(signatures=signatures)


def _legend_entry_line_type(entry: LegendEntry) -> LineType | None:
    value = f"{entry.label} {entry.description or ''} {entry.symbol_class}".casefold()
    if any(term in value for term in ("电信号", "electric signal", "electrical signal")):
        return "signal_electric"
    if any(term in value for term in ("气压信号", "pneumatic signal", "空气信号")):
        return "signal_pneumatic"
    if any(term in value for term in ("毛细管", "capillary", "导压管", "impulse line")):
        return "instrument_capillary"
    if any(term in value for term in ("电源线", "power line", "electrical power")):
        return "electrical_power"
    if any(term in value for term in ("工艺管线", "process line", "process pipe", "物料管线")):
        return "process"
    return None


def _legend_sample_region(entry: LegendEntry, page: PageEvidence) -> BBox | None:
    anchor = entry.source_bbox or entry.source_label_bbox
    if anchor is None:
        return None
    # Legend tables often put the sample stroke well to the left of its text,
    # and fragmented PDF dashes may be represented by many tiny paths.  Search
    # the whole row rather than assuming the label and sample share one bbox.
    left = max(0, anchor.x - max(900, page.width * 0.22, anchor.w * 8))
    vertical_padding = max(12, min(28, int(anchor.h * 0.75)))
    top = max(0, anchor.y - vertical_padding)
    right = min(page.width, anchor.x2 + max(180, anchor.w * 2))
    bottom = min(page.height, anchor.y2 + vertical_padding)
    return BBox(x=left, y=top, w=max(1, right - left), h=max(1, bottom - top))


def _path_is_legend_sample(path: PathEvidence, region: BBox) -> bool:
    if path.closed or path.primitive not in {"line", "curve"} or len(path.points) < 2:
        return False
    center = (path.bbox.x + path.bbox.w / 2, path.bbox.y + path.bbox.h / 2)
    if not (region.x <= center[0] <= region.x2 and region.y <= center[1] <= region.y2):
        return False
    return max(path.bbox.w, path.bbox.h, math.dist(path.points[0], path.points[-1])) >= 30


def _path_signature(path: PathEvidence) -> str:
    style = _path_visual_style(path)
    if style == "solid":
        return "solid"
    if path.dashes and path.dashes.startswith("inferred_fragment_pattern:"):
        return f"fragment:{style}:{path.dashes.rsplit(':', 1)[-1]}"
    if path.dashes and path.dashes != "inferred_fragment_pattern":
        values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", path.dashes)]
        if values:
            width = max(float(path.stroke_width), 0.25)
            ratios = ",".join(str(round(value / width, 1)) for value in values[:8])
            return f"pdf:{style}:{ratios}"
    return f"fragment:{style}"


def _path_visual_style(path: PathEvidence) -> VisualLineStyle:
    if path.visual_style != "unknown":
        return path.visual_style
    if path.origin == "pdf_vector":
        style, _, _ = classify_pdf_dash_pattern(
            path.dashes,
            stroke_width=path.stroke_width,
        )
        return style
    return "unknown"


def _path_style_confidence(path: PathEvidence) -> float:
    if path.style_confidence is not None:
        return float(path.style_confidence)
    if path.origin == "pdf_vector":
        _, confidence, _ = classify_pdf_dash_pattern(
            path.dashes,
            stroke_width=path.stroke_width,
        )
        return confidence
    return 0.0


@dataclass(frozen=True)
class _AxisFragment:
    path: PathEvidence
    orientation: str
    coordinate: float
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


def _infer_fragmented_vector_styles(
    page: PageEvidence,
    paths: list[PathEvidence],
) -> tuple[list[PathEvidence], list[LineStyleEvidence], set[str]]:
    """Recover visible dashed runs exported as separate solid PDF strokes.

    The first implementation is intentionally conservative and limited to the
    horizontal/vertical runs used by the great majority of P&IDs. A run must
    contain at least three non-touching collinear marks with regular gaps.
    Ordinary broken pipes, text strokes, and symbol outlines therefore remain
    as their original source primitives instead of being guessed as signals.
    """
    axis_tolerance = max(1.5, min(page.width, page.height) * 0.00045)
    max_gap = max(24.0, min(page.width, page.height) * 0.012)
    groups: dict[tuple[Any, ...], list[_AxisFragment]] = defaultdict(list)

    for path in paths:
        if (
            path.origin != "pdf_vector"
            or path.primitive != "line"
            or path.closed
            or len(path.points) != 2
            or _path_visual_style(path) != "solid"
        ):
            continue
        (x1, y1), (x2, y2) = path.points
        if abs(y2 - y1) <= axis_tolerance:
            orientation = "h"
            coordinate = (y1 + y2) / 2.0
            start, end = sorted((float(x1), float(x2)))
        elif abs(x2 - x1) <= axis_tolerance:
            orientation = "v"
            coordinate = (x1 + x2) / 2.0
            start, end = sorted((float(y1), float(y2)))
        else:
            continue
        # Some CAD exporters materialise a patterned stroke as dozens of
        # one-to-six-pixel vector marks. Keep those marks here; the run-level
        # checks below reject ordinary short symbol strokes.
        if end - start < max(0.5, path.stroke_width * 0.35):
            continue
        color_key = tuple(round(float(value), 2) for value in (path.stroke_color or []))
        key = (
            orientation,
            round(coordinate / axis_tolerance),
            round(path.stroke_width * 2.0) / 2.0,
            color_key,
        )
        groups[key].append(
            _AxisFragment(
                path=path,
                orientation=orientation,
                coordinate=coordinate,
                start=start,
                end=end,
            )
        )

    derived: list[PathEvidence] = []
    evidence: list[LineStyleEvidence] = []
    consumed: set[str] = set()
    for fragments in groups.values():
        unique = _deduplicate_fragments(fragments, tolerance=axis_tolerance)
        if len(unique) < 3:
            continue
        run: list[_AxisFragment] = []
        for fragment in unique:
            if not run:
                run = [fragment]
                continue
            gap = fragment.start - run[-1].end
            if -axis_tolerance <= gap <= max_gap:
                run.append(fragment)
            else:
                _append_fragment_run(page, run, derived, evidence, consumed)
                run = [fragment]
        _append_fragment_run(page, run, derived, evidence, consumed)
    return derived, evidence, consumed


def _deduplicate_fragments(
    fragments: list[_AxisFragment],
    *,
    tolerance: float,
) -> list[_AxisFragment]:
    unique: list[_AxisFragment] = []
    for fragment in sorted(fragments, key=lambda value: (value.start, value.end, value.path.id)):
        duplicate = next(
            (
                prior
                for prior in reversed(unique[-4:])
                if abs(fragment.start - prior.start) <= tolerance
                and abs(fragment.end - prior.end) <= tolerance
            ),
            None,
        )
        if duplicate is None:
            unique.append(fragment)
    return unique


def _append_fragment_run(
    page: PageEvidence,
    run: list[_AxisFragment],
    derived: list[PathEvidence],
    evidence: list[LineStyleEvidence],
    consumed: set[str],
) -> None:
    if len(run) < 3:
        return
    dash_lengths = [item.length for item in run]
    gap_lengths = [right.start - left.end for left, right in zip(run, run[1:], strict=False)]
    span = run[-1].end - run[0].start
    minimum_span = max(80.0, min(page.width, page.height) * 0.018)
    if span < minimum_span or not gap_lengths:
        return
    positive_gaps = [max(0.0, value) for value in gap_lengths]
    gap_mean = sum(positive_gaps) / len(positive_gaps)
    gap_variance = sum((value - gap_mean) ** 2 for value in positive_gaps) / len(positive_gaps)
    gap_cv = math.sqrt(gap_variance) / gap_mean if gap_mean else 0.0
    gap_ratio = sum(positive_gaps) / span if span else 0.0
    ink_median = median(dash_lengths)
    stroke_width = median([item.path.stroke_width for item in run])
    micro_fragment_pattern = len(run) >= 8 and ink_median <= max(12.0, stroke_width * 7.0)
    if (
        gap_cv > 0.65
        or gap_ratio > 0.72
        or (gap_ratio < 0.08 and not micro_fragment_pattern)
        or (ink_median < max(5.0, stroke_width * 2.5) and not micro_fragment_pattern)
    ):
        return

    short = min(dash_lengths)
    long_threshold = (
        max(short * 2.6, short + 2.5) if micro_fragment_pattern else max(short * 2.6, short + 12.0)
    )
    long_marks = [value for value in dash_lengths if value >= long_threshold]
    short_marks = [value for value in dash_lengths if value <= short * 1.55]
    if len(long_marks) >= max(2, int(len(run) * 0.25)) and len(short_marks) >= max(
        2, int(len(run) * 0.25)
    ):
        style: VisualLineStyle = "dash_dot"
    elif ink_median <= max(8.0, stroke_width * 4.0):
        style = "dotted"
    else:
        style = "dashed"

    pattern_ratio = ink_median / max(median(gap_lengths), 0.1)
    if pattern_ratio <= 0.65:
        pattern_bucket = "short"
    elif pattern_ratio <= 1.6:
        pattern_bucket = "balanced"
    elif pattern_ratio <= 3.2:
        pattern_bucket = "long"
    else:
        pattern_bucket = "very_long"

    coordinate = median([item.coordinate for item in run])
    if run[0].orientation == "h":
        points = [
            (int(round(run[0].start)), int(round(coordinate))),
            (int(round(run[-1].end)), int(round(coordinate))),
        ]
    else:
        points = [
            (int(round(coordinate)), int(round(run[0].start))),
            (int(round(coordinate)), int(round(run[-1].end))),
        ]
    source_ids = [item.path.id for item in run]
    evidence_id = stable_evidence_id(
        "sty",
        page.page_index,
        style,
        points,
        source_ids,
    )
    confidence = max(
        0.65,
        min(0.98, 0.77 + min(0.14, (len(run) - 3) * 0.025) - min(0.12, gap_cv * 0.12)),
    )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = BBox(
        x=min(xs),
        y=min(ys),
        w=max(1, max(xs) - min(xs)),
        h=max(1, max(ys) - min(ys)),
    )
    color = run[0].path.stroke_color
    derived.append(
        PathEvidence(
            id=evidence_id,
            page_index=page.page_index,
            points=points,
            bbox=bbox,
            origin="pdf_vector",
            primitive="line",
            stroke_width=stroke_width,
            stroke_color=color,
            dashes=f"inferred_fragment_pattern:{pattern_bucket}",
            visual_style=style,
            style_confidence=confidence,
            style_source="vector_fragment_pattern",
            source_path_ids=source_ids,
        )
    )
    evidence.append(
        LineStyleEvidence(
            id=evidence_id,
            page_index=page.page_index,
            visual_style=style,
            confidence=confidence,
            source="vector_fragment_pattern",
            bbox=bbox,
            points=points,
            source_path_ids=source_ids,
            dash_lengths=[round(value, 3) for value in dash_lengths],
            gap_lengths=[round(value, 3) for value in gap_lengths],
        )
    )
    consumed.update(source_ids)


def _is_page_frame(segment: _Segment, page: PageEvidence, margin: int) -> bool:
    horizontal = abs(segment.a[1] - segment.b[1]) <= 1.5
    vertical = abs(segment.a[0] - segment.b[0]) <= 1.5
    if horizontal and segment.length > page.width * 0.65:
        return (
            min(segment.a[1], segment.b[1]) <= margin
            or max(segment.a[1], segment.b[1]) >= page.height - margin
        )
    if vertical and segment.length > page.height * 0.65:
        return (
            min(segment.a[0], segment.b[0]) <= margin
            or max(segment.a[0], segment.b[0]) >= page.width - margin
        )
    return False


def _build_graph(
    *,
    page: PageEvidence,
    segments: list[_Segment],
    nodes: list[ReconciledNode],
    contours: dict[str, SymbolContour],
) -> tuple[
    dict[int, tuple[float, float]],
    dict[int, list[_GraphArc]],
    dict[int, list[str]],
    dict[str, str],
    dict[tuple[str, tuple[int, int]], list[str]],
]:
    joint_radius = max(3.0, min(page.width, page.height) * 0.0008)
    split_marks: list[list[float]] = [[0.0, 1.0] for _ in segments]

    endpoint_grid: dict[tuple[int, int], list[tuple[int, float, tuple[float, float]]]] = (
        defaultdict(list)
    )
    cell = max(joint_radius * 2.0, 6.0)
    for index, segment in enumerate(segments):
        for t, point in ((0.0, segment.a), (1.0, segment.b)):
            endpoint_grid[_grid_key(point, cell)].append((index, t, point))

    # Filled native junction markers explicitly connect crossing interiors.
    dots = _junction_dots(page)
    for index, segment in enumerate(segments):
        for point, radius in dots:
            t, distance = _project_point_to_segment(point, segment)
            if distance <= radius and 0.0 < t < 1.0:
                split_marks[index].append(t)

    # Split a line where another line terminates on its interior (tees).
    for index, segment in enumerate(segments):
        min_x = min(segment.a[0], segment.b[0]) - joint_radius
        max_x = max(segment.a[0], segment.b[0]) + joint_radius
        min_y = min(segment.a[1], segment.b[1]) - joint_radius
        max_y = max(segment.a[1], segment.b[1]) + joint_radius
        for key in _grid_keys_for_bbox(min_x, min_y, max_x, max_y, cell):
            for other_index, _, point in endpoint_grid.get(key, []):
                if (
                    other_index == index
                    or segments[other_index].visual_style != segment.visual_style
                ):
                    continue
                t, distance = _project_point_to_segment(point, segment)
                if joint_radius >= distance and 1e-4 < t < 1 - 1e-4:
                    split_marks[index].append(t)

    node_marks: dict[str, list[tuple[int, float]]] = defaultdict(list)
    nozzle_marks = {}
    for node in nodes:
        contour = contours.get(node.id)
        polygon = contour.points if contour else box_polygon(node.bbox_global)
        for index, segment in enumerate(segments):
            # A real port is a boundary crossing into the symbol. A tangential
            # stroke running along its perimeter does not enter the symbol.
            marks = intersections(segment.a, segment.b, polygon)
            if contour and not marks:
                for t, endpoint, other in [
                    (0.0, segment.a, segment.b),
                    (1.0, segment.b, segment.a),
                ]:
                    sources = nozzle_contact(page, contour, endpoint, other)
                    if sources:
                        node_marks[node.id].append((index, t))
                        nozzle_marks[(node.id, index, t)] = sources
            for t in marks:
                delta = min(0.02, 2.0 / max(segment.length, 1.0))
                before = _point_at(segment, max(0.0, t - delta))
                after = _point_at(segment, min(1.0, t + delta))
                enters = inside(before, polygon) != inside(after, polygon)
                # A pipe may end exactly on a nozzle at the symbol boundary.
                terminates = t in {0.0, 1.0} and not inside(_point_at(segment, 0.5), polygon)
                if not enters and not terminates:
                    continue
                if contour and contour.basis == "native_open_symbol":
                    contact = _point_at(segment, t)
                    if not any(
                        _point_segment_distance(contact, a, b) <= joint_radius
                        for _, a, b in contour.segments
                    ):
                        continue
                # Exclude collinear perimeter strokes (rectangle edges, etc.).
                if not contour and _runs_on_boundary(segment, node.bbox_global):
                    continue
                node_marks[node.id].append((index, t))
                split_marks[index].append(t)

    raw_points: list[tuple[float, float]] = []
    raw_styles: list[str] = []
    raw_arcs: list[tuple[int, int, _Segment]] = []
    mark_point_index: dict[tuple[int, int], int] = {}
    for segment_index, segment in enumerate(segments):
        values = sorted(set(round(max(0.0, min(1.0, t)), 6) for t in split_marks[segment_index]))
        indexes: list[int] = []
        for t in values:
            idx = len(raw_points)
            raw_points.append(_point_at(segment, t))
            raw_styles.append(segment.visual_style)
            indexes.append(idx)
            mark_point_index[(segment_index, int(round(t * 1_000_000)))] = idx
        for left, right in zip(indexes, indexes[1:], strict=False):
            midpoint = tuple(
                (a + b) / 2 for a, b in zip(raw_points[left], raw_points[right], strict=False)
            )
            # Remove only native-contour interiors, never whole model boxes.
            if any(inside(midpoint, contour.points) for contour in contours.values()):
                continue
            if math.dist(raw_points[left], raw_points[right]) > 0.5:
                raw_arcs.append((left, right, segment))

    uf = _cluster_points(raw_points, joint_radius, groups=raw_styles)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(raw_points)):
        members[uf.find(index)].append(index)
    root_to_vertex = {root: index for index, root in enumerate(sorted(members))}
    vertex_points = {
        root_to_vertex[root]: (
            sum(raw_points[index][0] for index in values) / len(values),
            sum(raw_points[index][1] for index in values) / len(values),
        )
        for root, values in members.items()
    }
    raw_to_vertex = {index: root_to_vertex[uf.find(index)] for index in range(len(raw_points))}

    adjacency: dict[int, list[_GraphArc]] = defaultdict(list)
    path_origins: dict[str, str] = {}
    for left, right, segment in raw_arcs:
        a = raw_to_vertex[left]
        b = raw_to_vertex[right]
        if a == b:
            continue
        length = math.dist(vertex_points[a], vertex_points[b])
        arc_values = (
            length,
            segment.path_id,
            segment.line_type,
            segment.visual_style,
            segment.style_confidence,
            segment.source_path_ids,
            segment.legend_signature,
            segment.legend_labels,
            segment.legend_confidence,
        )
        adjacency[a].append(_GraphArc(b, *arc_values))
        adjacency[b].append(_GraphArc(a, *arc_values))
        path_origins[segment.path_id] = segment.origin

    attachments: dict[int, list[str]] = defaultdict(list)
    nozzle_ports = {}
    for node_id, marks in node_marks.items():
        for segment_index, t in marks:
            raw_index = mark_point_index.get((segment_index, int(round(round(t, 6) * 1_000_000))))
            if raw_index is not None:
                vertex = raw_to_vertex[raw_index]
                attachments[vertex].append(node_id)
                if (node_id, segment_index, t) in nozzle_marks:
                    point = tuple(round(v) for v in vertex_points[vertex])
                    nozzle_ports[(node_id, point)] = nozzle_marks[(node_id, segment_index, t)]

    return vertex_points, adjacency, attachments, path_origins, nozzle_ports


def _cyclic_curve_paths(adjacency: dict[int, list[_GraphArc]], curves: set[str]) -> set[str]:
    """Unowned curves in cyclic ink need review (equipment contour or pipe loop).

    Pruning terminal branches preserves ordinary open pipe bends. It does not
    assign an engineering identity to an otherwise undetected closed symbol.
    """
    neighbours = {vertex: {a.target for a in arcs} for vertex, arcs in adjacency.items()}
    leaves = [vertex for vertex, values in neighbours.items() if len(values) < 2]
    removed = set()
    for vertex in leaves:
        if vertex in removed:
            continue
        removed.add(vertex)
        for other in neighbours[vertex]:
            neighbours[other].discard(vertex)
            if len(neighbours[other]) < 2:
                leaves.append(other)
    return {
        arc.path_id
        for vertex, arcs in adjacency.items()
        if vertex not in removed
        for arc in arcs
        if arc.target not in removed and arc.path_id in curves
    }


def _junction_dots(page: PageEvidence) -> list[tuple[tuple[float, float], float]]:
    maximum = max(10.0, min(page.width, page.height) * 0.003)
    groups = defaultdict(list)
    for path in page.paths:
        if path.fill_color is not None:
            groups[
                path.source_drawing_index if path.source_drawing_index is not None else path.id
            ].append(path)
    out = []
    for paths in groups.values():
        if not any(p.closed or p.primitive in {"rect", "quad"} for p in paths) and not (
            len(paths) >= 4 and all(p.primitive == "curve" for p in paths)
        ):
            continue
        x, y = min(p.bbox.x for p in paths), min(p.bbox.y for p in paths)
        w, h = max(p.bbox.x2 for p in paths) - x, max(p.bbox.y2 for p in paths) - y
        if 1 <= min(w, h) <= max(w, h) <= maximum and min(w, h) / max(w, h) >= 0.7:
            out.append(((x + w / 2, y + h / 2), max(w, h) / 2 + 1))
    return out


def _separate_unmarked_crossings(page, points, adjacency, attachments):
    """CAD exports may split both crossing lines at the same coordinate.

    Four collinear arms without a native junction marker preserve two through
    routes, with review evidence for the unresolved cross-connection.
    """
    ambiguities = []
    dots = _junction_dots(page)
    for vertex in sorted(list(adjacency)):
        arcs = adjacency[vertex]
        if len(arcs) != 4 or vertex in attachments:
            continue
        point = points[vertex]
        if any(math.dist(point, dot) <= radius for dot, radius in dots):
            continue

        def opposite(a, b, *, point=point):
            ax, ay = points[a.target][0] - point[0], points[a.target][1] - point[1]
            bx, by = points[b.target][0] - point[0], points[b.target][1] - point[1]
            return (ax * bx + ay * by) / max(math.hypot(ax, ay) * math.hypot(bx, by), 1e-9) < -0.98

        partner = next((i for i in range(1, 4) if opposite(arcs[0], arcs[i])), None)
        if partner is None:
            continue
        straight = [arcs[0], arcs[partner]]
        other = [arc for i, arc in enumerate(arcs) if i not in {0, partner}]
        if not opposite(*other):
            continue
        clone = max(points) + 1
        points[clone] = point
        adjacency[vertex], adjacency[clone] = straight, other
        for arc in other:
            for reverse in adjacency[arc.target]:
                if reverse.target == vertex and reverse.path_id == arc.path_id:
                    reverse.target = clone
        ambiguities.append(
            {
                "type": "crossing_or_junction",
                "page_index": page.page_index,
                "x": round(point[0]),
                "y": round(point[1]),
                "status": "unresolved",
                "path_ids": sorted({a.path_id for a in arcs}),
                "reason": "four arms without native junction marker; through routes kept separate",
            }
        )
    return ambiguities


def _runs_on_boundary(segment: _Segment, bbox: BBox) -> bool:
    tolerance = 1.5
    return (
        abs(segment.a[0] - segment.b[0]) <= tolerance
        and min(abs(segment.a[0] - bbox.x), abs(segment.a[0] - bbox.x2)) <= tolerance
    ) or (
        abs(segment.a[1] - segment.b[1]) <= tolerance
        and min(abs(segment.a[1] - bbox.y), abs(segment.a[1] - bbox.y2)) <= tolerance
    )


def _cluster_points(
    points: list[tuple[float, float]], radius: float, *, groups: list[str]
) -> _UnionFind:
    uf = _UnionFind()
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cell = max(radius, 1.0)
    for index, point in enumerate(points):
        uf.add()
        key = _grid_key(point, cell)
        for x in range(key[0] - 1, key[0] + 2):
            for y in range(key[1] - 1, key[1] + 2):
                for other in grid.get((x, y), []):
                    if groups[index] == groups[other] and math.dist(point, points[other]) <= radius:
                        # Do not bridge a chain of individually close endpoints.
                        root = uf.find(other)
                        if math.dist(point, points[root]) <= radius:
                            uf.union(other, index)
        grid[key].append(index)
    return uf


def _grid_key(point: tuple[float, float], cell: float) -> tuple[int, int]:
    return int(math.floor(point[0] / cell)), int(math.floor(point[1] / cell))


def _grid_keys_for_bbox(
    x0: float, y0: float, x1: float, y1: float, cell: float
) -> list[tuple[int, int]]:
    left = int(math.floor(x0 / cell))
    right = int(math.floor(x1 / cell))
    top = int(math.floor(y0 / cell))
    bottom = int(math.floor(y1 / cell))
    return [(x, y) for x in range(left, right + 1) for y in range(top, bottom + 1)]


def _point_at(segment: _Segment, t: float) -> tuple[float, float]:
    return (
        segment.a[0] + (segment.b[0] - segment.a[0]) * t,
        segment.a[1] + (segment.b[1] - segment.a[1]) * t,
    )


def _project_point_to_segment(point: tuple[float, float], segment: _Segment) -> tuple[float, float]:
    dx = segment.b[0] - segment.a[0]
    dy = segment.b[1] - segment.a[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return 0.0, math.dist(point, segment.a)
    t = ((point[0] - segment.a[0]) * dx + (point[1] - segment.a[1]) * dy) / denom
    t = max(0.0, min(1.0, t))
    return t, math.dist(point, _point_at(segment, t))


def _connected_components(adjacency: dict[int, list[_GraphArc]]) -> list[set[int]]:
    remaining = set(adjacency)
    out: list[set[int]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[int] = set()
        while stack:
            value = stack.pop()
            if value in component:
                continue
            component.add(value)
            stack.extend(arc.target for arc in adjacency.get(value, []))
        remaining.difference_update(component)
        out.append(component)
    return out


def _dijkstra(
    source: int,
    adjacency: dict[int, list[_GraphArc]],
    allowed: set[int],
    *,
    stops: set[int],
) -> tuple[dict[int, float], dict[int, tuple[int, _GraphArc]]]:
    distances = {source: 0.0}
    previous: dict[int, tuple[int, _GraphArc]] = {}
    queue = [(0.0, source)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances.get(vertex):
            continue
        if vertex != source and vertex in stops:
            continue
        for arc in adjacency.get(vertex, []):
            if arc.target not in allowed:
                continue
            candidate = distance + arc.length
            if candidate < distances.get(arc.target, float("inf")):
                distances[arc.target] = candidate
                previous[arc.target] = (vertex, arc)
                heapq.heappush(queue, (candidate, arc.target))
    return distances, previous


def _restore_path(
    source: int,
    target: int,
    previous: dict[int, tuple[int, _GraphArc]],
) -> tuple[list[int], list[_GraphArc]]:
    vertices = [target]
    arcs: list[_GraphArc] = []
    current = target
    while current != source:
        row = previous.get(current)
        if row is None:
            return [], []
        parent, arc = row
        vertices.append(parent)
        arcs.append(arc)
        current = parent
    vertices.reverse()
    arcs.reverse()
    return vertices, arcs


def _majority_line_type(arcs: list[_GraphArc]) -> LineType:
    counts: Counter[str] = Counter()
    for arc in arcs:
        counts[arc.line_type] += max(1, int(round(arc.length)))
    return counts.most_common(1)[0][0] if counts else "process"  # type: ignore[return-value]


def _majority_visual_style(arcs: list[_GraphArc]) -> VisualLineStyle:
    counts: Counter[str] = Counter()
    for arc in arcs:
        counts[arc.visual_style] += max(1, int(round(arc.length)))
    return counts.most_common(1)[0][0] if counts else "unknown"  # type: ignore[return-value]


def _route_style_confidence(
    arcs: list[_GraphArc],
    visual_style: VisualLineStyle,
) -> float:
    weighted = [
        (max(1.0, arc.length), arc.style_confidence)
        for arc in arcs
        if arc.visual_style == visual_style and arc.style_confidence > 0
    ]
    if not weighted:
        return 0.0
    total = sum(weight for weight, _ in weighted)
    return sum(weight * confidence for weight, confidence in weighted) / total


def _stable_edge_id(
    page_index: int,
    left_id: str,
    right_id: str,
    polyline: list[tuple[int, int]],
    line_type: str,
) -> str:
    endpoints = sorted((left_id, right_id))
    payload = repr((page_index, endpoints, polyline, line_type)).encode("utf-8")
    return "e-" + hashlib.sha256(payload).hexdigest()[:16]


def _crossing_candidates(
    page: PageEvidence,
    segments: list[_Segment],
    *,
    nodes: list[ReconciledNode],
) -> list[dict[str, Any]]:
    """Return only source-supported crossings that can affect graph topology.

    Raw PDF paths contain text strokes, symbol internals, borders, and drawing
    tables.  Crossing every candidate path produced thousands of false human
    review tasks.  Callers now pass only paths actually used by an extracted
    network; this function additionally excludes symbol interiors, tiny
    strokes, same-path intersections, and spatial duplicates.
    """
    out: list[dict[str, Any]] = []
    cell = max(80.0, min(page.width, page.height) * 0.02)
    minimum_length = max(20.0, min(page.width, page.height) * 0.005)
    dedupe_radius = max(10.0, min(page.width, page.height) * 0.003)
    symbol_padding = max(8.0, min(page.width, page.height) * 0.002)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    checked: set[tuple[int, int]] = set()
    accepted_points: list[tuple[float, float]] = []
    for index, segment in enumerate(segments):
        if segment.length < minimum_length:
            continue
        keys = _grid_keys_for_bbox(
            min(segment.a[0], segment.b[0]),
            min(segment.a[1], segment.b[1]),
            max(segment.a[0], segment.b[0]),
            max(segment.a[1], segment.b[1]),
            cell,
        )
        for key in keys:
            for other_index in grid.get(key, []):
                pair = (min(index, other_index), max(index, other_index))
                if pair in checked:
                    continue
                checked.add(pair)
                other = segments[other_index]
                if other.length < minimum_length or other.path_id == segment.path_id:
                    continue
                point = _proper_intersection(segment, other)
                if point is None:
                    continue
                if any(
                    _point_inside_bbox(point, node.bbox_global, padding=symbol_padding)
                    for node in nodes
                ):
                    continue
                if any(math.dist(point, prior) <= dedupe_radius for prior in accepted_points):
                    continue
                accepted_points.append(point)
                out.append(
                    {
                        "type": "crossing_or_junction",
                        "page_index": page.page_index,
                        "x": int(round(point[0])),
                        "y": int(round(point[1])),
                        "path_ids": [segment.path_id, other.path_id],
                        "status": "unresolved",
                    }
                )
                if len(out) >= 100:
                    return out
            grid[key].append(index)
    return out


def _point_inside_bbox(point: tuple[float, float], bbox: BBox, *, padding: float) -> bool:
    return (
        bbox.x - padding <= point[0] <= bbox.x2 + padding
        and bbox.y - padding <= point[1] <= bbox.y2 + padding
    )


def _proper_intersection(left: _Segment, right: _Segment) -> tuple[float, float] | None:
    x1, y1 = left.a
    x2, y2 = left.b
    x3, y3 = right.a
    x4, y4 = right.b
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
    if not (0.03 < t < 0.97 and 0.03 < u < 0.97):
        return None
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)




def route_evidence_failures(edge: ReconciledEdge, page: PageEvidence) -> list[str]:
    """Hard acceptance contract for local routes, independent of model scores.

    Cross-sheet links use printed-reference validation in the page solver and
    assembly stage. They deliberately have no on-page pipe polyline.
    """
    if edge.cross_sheet:
        return []
    failures = []
    points = edge.polyline_global
    if edge.from_node == edge.to_node:
        failures.append("same_physical_endpoint")
    if (
        len(set(points)) < 2
        or sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False)) <= 0.5
    ):
        failures.append("route_has_no_length")
    proof = edge.attributes.get("route_evidence", {})
    if proof.get("version") != TOPOLOGY_VERSION or proof.get("status") != "verified":
        failures.extend(proof.get("failures") or ["missing_verified_port_route"])
    source_ids = set(edge.source_evidence_ids)
    native_ids = {path.id for path in page.paths}
    if not source_ids or not source_ids <= native_ids:
        failures.append("missing_source_segments")
    if set(proof.get("source_path_ids", [])) != source_ids:
        failures.append("route_provenance_mismatch")
    ports = proof.get("ports", [])
    if len(ports) != 2 or not points:
        failures.append("missing_endpoint_ports")
    else:
        for port, node_id, point in zip(
            ports, (edge.from_node, edge.to_node), (points[0], points[-1]), strict=True
        ):
            if port.get("node_id") != node_id or tuple(port.get("point", [])) != tuple(point):
                failures.append("endpoint_port_mismatch")
    if proof.get("scene_version") != SCENE_VERSION:
        failures.append("missing_native_scene")
    if (
        len(ports) == 2
        and ports[0].get("symbol_id")
        and ports[0].get("symbol_id") == ports[1].get("symbol_id")
    ):
        failures.append("same_physical_symbol")
    native = {p.id: p for p in page.paths}
    # Check exact source strokes, including foreign glyph decoration and scope
    # boundaries, rather than trusting a derived dashed run's path IDs alone.
    forbidden = {
        segment_key(s[0], s[1], s[2])
        for s in proof.get("non_connective_segments", [])
        if len(s) == 3
    }
    for port in ports:
        forbidden.update(
            segment_key(s[0], s[1], s[2])
            for s in port.get("outline_segments", [])
            if len(s) == 3
        )
    for pid, a, b in forbidden:
        if pid not in native:
            continue
        vertices = path_vertices(native[pid])
        if native[pid].closed or native[pid].primitive in {"rect", "quad"}:
            vertices += vertices[:1]
        if segment_key(pid, a, b) not in {
            segment_key(pid, x, y) for x, y in zip(vertices, vertices[1:], strict=False)
        }:
            continue
        if any(
            _positive_collinear_overlap(x, y, a, b)
            for x, y in zip(points, points[1:], strict=False)
        ):
            failures.append("route_uses_non_connective_stroke")
    tolerance = max(3.0, min(page.width, page.height) * 0.0008)
    for port in ports:
        if port.get("identity_status") == "unresolved":
            failures.append("unresolved_symbol_identity")
        records = port.get("outline_segments", [])
        owned = []
        valid = bool(records) and port.get("basis") in {
            "native_contour",
            "native_open_symbol",
            "native_nozzle",
        }
        for record in records:
            if len(record) != 3 or record[0] not in native:
                valid = False
                continue
            pid, a, b = record
            points_for_path = path_vertices(native[pid])
            if native[pid].closed or native[pid].primitive in {"rect", "quad"}:
                points_for_path = points_for_path + points_for_path[:1]
            source_keys = {
                segment_key(pid, x, y)
                for x, y in zip(points_for_path, points_for_path[1:], strict=False)
            }
            if segment_key(pid, a, b) not in source_keys:
                valid = False
            owned.append((pid, a, b))
        expected_symbol_id = stable_evidence_id(
            "sym", page.page_index, sorted({segment_key(pid, a, b) for pid, a, b in owned})
        )
        if port.get("symbol_id") != expected_symbol_id:
            valid = False
        if not valid:
            failures.append("unsupported_physical_port")
            continue
        point = port.get("point", [])
        if len(point) != 2:
            failures.append("unsupported_physical_port")
            continue
        if port.get("basis") == "native_nozzle":
            if len(points) < 2:
                failures.append("unsupported_nozzle")
                continue
            ids = port.get("nozzle_path_ids", [])
            contour = SymbolContour(
                port.get("node_id", ""),
                port.get("polygon", []),
                port.get("outline_path_ids", []),
                segments={segment_key(pid, a, b) for pid, a, b in owned},
            )
            other = (
                points[1]
                if point == list(points[0]) or tuple(point) == tuple(points[0])
                else points[-2]
            )
            supported = nozzle_contact(page, contour, tuple(point), other)
            if not ids or not set(ids) <= set(supported):
                failures.append("unsupported_nozzle")
        elif min(_point_segment_distance(point, a, b) for _, a, b in owned) > tolerance:
            failures.append("port_misses_symbol_ink")
    return sorted(set(failures))


def _positive_collinear_overlap(a, b, x, y) -> bool:
    """A crossing or endpoint touch is not traversal of non-connective ink."""
    length = math.dist(a, b)
    if length <= 1:
        return False
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    if any(abs((p[0] - a[0]) * uy - (p[1] - a[1]) * ux) > 1 for p in (x, y)):
        return False
    lo, hi = sorted((p[0] - a[0]) * ux + (p[1] - a[1]) * uy for p in (x, y))
    return min(length, hi) - max(0, lo) > 1


def _point_segment_distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(
        0, min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / max(dx * dx + dy * dy, 1e-12))
    )
    return math.dist(point, (a[0] + t * dx, a[1] + t * dy))
