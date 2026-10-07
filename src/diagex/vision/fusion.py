"""Evidence fusion, stable IDs, OPC matching, and confidence for evidence-v2."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import cache
from typing import Any

from pydantic import BaseModel, Field

from diagex.dexpi_schema import (
    EQUIPMENT_CLASS_KEYS,
    INSTRUMENT_CLASS_KEYS,
    INSTRUMENT_FUNCTION_KEYS,
    VALVE_TYPE_KEYS,
)
from diagex.vision.connection_inference import PageGraphResult
from diagex.vision.evidence import PageEvidence, TextEvidence
from diagex.vision.instance_matching import FUSION_VERSION, InstanceCluster, match_instances
from diagex.vision.legend_models import LegendPack
from diagex.vision.models import (
    BBox,
    Confidence,
    ReconciledEdge,
    ReconciledGraph,
    ReconciledNode,
)
from diagex.vision.native_text import infer_tag_semantics
from diagex.vision.reconcile import normalise_label
from diagex.vision.symbol_interpretation import DetectionRecord
from diagex.vision.text_assignment import (
    _assign_native_tags_to_nodes as _assign_native_tags_to_nodes,
)
from diagex.vision.text_assignment import (
    _bbox_centres_overlap,
    _center_distance,
    _confidence_level,
    _ordered_unique,
    assign_text,
    resolve_symbol_text,
)
from diagex.vision.text_assignment import (
    _native_entity_tag_spans as _native_entity_tag_spans,
)
from diagex.vision.text_assignment import (
    _native_tag_node_score as _native_tag_node_score,
)
from diagex.vision.text_assignment import (
    _nearby_tag_text as _nearby_tag_text,
)
from diagex.vision.topology import TopologyResult, route_evidence_failures
from diagex.vision.vector_geometry import SCENE_VERSION, unobserved_symbols

_CONF_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
_TAG_RE = re.compile(r"(?:[A-Z]{2,5}[- ]?\d{1,6}[A-Z]?|DN\s*\d+)", re.IGNORECASE)
_OPC_HINT_RE = re.compile(
    r"(?:off[- ]?page|continuation|to\s+sheet|from\s+sheet|\bDW\d{2}[- ]?\d{3,5}\b|去往|来自|接续)",
    re.IGNORECASE,
)
_EXPLICIT_OPC_REF_RE = re.compile(r"\bDW\d{2}[- ]?\d{3,5}\b", re.IGNORECASE)
_NON_CONNECTABLE_TEXT_RE = re.compile(
    r"^(?:"
    r"DN\s*\d+(?:\s*[x×/]\s*\d+)?|"
    r"\d+(?:\.\d+)?\s*(?:[\"”′]|mm|cm)|"
    r"(?:FO|FC|CSO|CSC|LO|LC)|"
    r"(?:SHEET|SHT|PAGE|REV|DWG)\s*[A-Z0-9-]+|"
    r"(?:\d{1,4}[\"”']?[- ])?[A-Z]{1,6}-\d{3,}(?:-[A-Z0-9]+){1,}"
    r")$",
    re.IGNORECASE,
)
_VALVE_STATE_RE = re.compile(r"^(FO|FC|CSO|CSC|LO|LC)$", re.IGNORECASE)
_TAG_FRAGMENT_RE = re.compile(r"^(?:[A-Z]{1,6}|\d{2,6}[A-Z]?)$", re.IGNORECASE)
_GENERIC_LABEL_KEYS = {"", "UNLABELLED", "UNLABELED", "UNKNOWN", "NONE", "NA"}
_EQUIPMENT_ALIASES = {
    "exchanger": "heat_exchanger",
    "heat_exchanger_unit": "heat_exchanger",
    "drier": "dryer",
    "blower": "fan_blower",
    "fan": "fan_blower",
    "tower": "column",
    "drum": "vessel",
    "receiver": "vessel",
    "equipment": "unclassified_equipment",
}
_VALVE_ALIASES = {
    "gate_valve": "gate",
    "globe_valve": "globe",
    "check_valve": "check",
    "check_or_non_return": "check",
    "non_return": "check",
    "ball_valve": "ball",
    "butterfly_valve": "butterfly",
    "control_valve": "control",
    "regulating_valve": "control",
    "pressure_regulating_valve": "control",
    "safety_valve": "safety_relief",
    "relief_valve": "safety_relief",
    "psv": "safety_relief",
    "three_way_valve": "three_way",
    "needle_valve": "needle",
    "plug_valve": "plug",
    "manual": "other",
    "manual_valve": "other",
    "isolation": "other",
    "isolation_valve": "other",
    "general": "other",
    "general_valve": "other",
    "block": "other",
    "block_valve": "other",
    "drain": "other",
    "inlet": "other",
    "outlet": "other",
    "unknown": "other",
    "quick_acting": "other",
    "shutdown": "other",
    "fail_open": "other",
    "fail_closed": "other",
}


class FusionResult(BaseModel):
    graph: ReconciledGraph
    ambiguities: list[dict[str, Any]] = Field(default_factory=list)
    native_text_assignment_count: int = 0


def fuse_objects(
    *,
    source_name: str,
    pages: list[PageEvidence],
    detections: list[DetectionRecord],
    per_page_status: dict[int, str],
    legend_pack: LegendPack | None = None,
    on_text_assignment=None,
    knowledge: dict | None = None,
) -> FusionResult:
    """Fuse overlapping object detections once, without constructing edges."""
    pages_by_index = {page.page_index: page for page in pages}
    accepted, rejected = _prepare_detections(detections, pages_by_index, legend_pack=legend_pack, knowledge=knowledge)
    clusters, instance_conflicts = match_instances(accepted, pages_by_index)
    nodes: list[ReconciledNode] = []
    ambiguities: list[dict[str, Any]] = list(rejected)
    assigned_native_text = 0

    for cluster in clusters:
        node, node_ambiguities, assigned_count = _node_from_cluster(cluster, pages_by_index)
        nodes.append(node)
        ambiguities.extend(node_ambiguities)
        assigned_native_text += assigned_count

    node_for_detection = {
        detection_id: node.id for node in nodes for detection_id in node.source_annotation_ids
    }
    for conflict in instance_conflicts:
        conflict["node_ids"] = sorted(
            {node_for_detection[did] for did in conflict["detection_ids"]}
        )
    ambiguities.extend(instance_conflicts)

    for page in pages:
        if page.role != "pid":
            continue
        local = [node for node in nodes if node.page_index == page.page_index]
        for symbol in unobserved_symbols(page, local):
            node_id = "n-" + symbol.id
            nodes.append(
                ReconciledNode(
                    id=node_id,
                    page_index=page.page_index,
                    kind="equipment",
                    label="unlabelled",
                    bbox_global=symbol.bbox,
                    confidence="low",
                    attributes={
                        "native_symbol_candidate": True,
                        "requires_human_review": True,
                        "physical_symbol_id": symbol.id,
                        "scene_version": SCENE_VERSION,
                        "equipment_class": "unclassified",
                    },
                    source_evidence_ids=symbol.path_ids,
                )
            )
            ambiguities.append(
                {
                    "type": "unobserved_native_symbol",
                    "status": "unresolved",
                    "page_index": page.page_index,
                    "node_ids": [node_id],
                    "source_path_ids": symbol.path_ids,
                    "reason": "Native body outline has no supported perception instance; identity requires review",
                }
            )

    assignment = _assign_text_stage(nodes, pages, legend_pack, on_text_assignment, knowledge)
    nodes = assignment.nodes
    ambiguities.extend(assignment.conflicts)
    assigned_native_text += assignment.assigned_count

    graph = ReconciledGraph(
        schema_version="0.3.0",
        source_path=source_name,
        assemblies=assignment.assemblies,
        text_bindings=assignment.text_bindings,
        nodes=sorted(
            nodes,
            key=lambda node: (node.page_index, node.bbox_global.y, node.bbox_global.x, node.id),
        ),
        conflicts=ambiguities,
        per_page_status={
            int(page): _coerce_page_status(status) for page, status in per_page_status.items()
        },
    )
    return FusionResult(
        graph=graph,
        ambiguities=ambiguities,
        native_text_assignment_count=assigned_native_text,
    )


def assemble_graph(
    *,
    objects: FusionResult,
    pages: list[PageEvidence],
    topology: list[TopologyResult],
    page_graph_results: list[PageGraphResult],
    per_page_status: dict[int, str],
) -> FusionResult:
    """Assemble validated page decisions and deterministic cross-sheet matches."""
    pages_by_index = {page.page_index: page for page in pages}
    nodes = [node.model_copy(deep=True) for node in objects.graph.nodes]
    ambiguities = [dict(value) for value in objects.ambiguities]
    synthesized_opcs, opc_enrichment = _enrich_opcs_from_native_text(nodes, pages_by_index)
    nodes.extend(synthesized_opcs)
    ambiguities.extend(opc_enrichment)
    verified_opcs = {
        node.id
        for node in nodes
        if node.kind == "opc" and node.attributes.get("native_reference_verified")
    }
    if verified_opcs:
        ambiguities = [
            value
            for value in ambiguities
            if not (
                value.get("type") == "opc_context_validation"
                and value.get("node_id") in verified_opcs
            )
        ]
    node_ids = {node.id for node in nodes}
    results_by_page = {result.page_index: result for result in page_graph_results}
    graph_edges: list[ReconciledEdge] = []

    for result in page_graph_results:
        ambiguities.extend(result.conflicts)
        ambiguities.extend(
            {
                "type": "rejected_page_graph_output",
                "page_index": result.page_index,
                "status": "resolved",
                "reason": detail,
            }
            for detail in result.diagnostics
        )
        for node in nodes:
            update = result.opc_updates.get(node.id)
            if update:
                node.attributes.update(update)

    for result in topology:
        ambiguities.extend(result.ambiguities)
        page_result = results_by_page.get(result.page_index)
        selected_edges = page_result.edges if page_result is not None else []
        if page_result is None:
            node_by_id = {node.id: node for node in nodes}
            for edge in result.edges:
                endpoints = (node_by_id.get(edge.from_node), node_by_id.get(edge.to_node))
                if all(node is not None and node.kind == "equipment" for node in endpoints):
                    selected_edges.append(edge)
                else:
                    ambiguities.append(
                        {
                            "type": "page_graph_missing",
                            "page_index": result.page_index,
                            "edge_id": edge.id,
                            "status": "unresolved",
                            "reason": "instrument/OPC topology requires a completed page-graph solve",
                        }
                    )
        for edge in selected_edges:
            if not edge.cross_sheet and not edge.attributes.get("provisional_review_only"):
                failures = route_evidence_failures(edge, pages_by_index[result.page_index])
                if failures:
                    edge = edge.model_copy(deep=True)
                    edge.attributes.update(
                        {
                            "provisional_review_only": True,
                            "requires_human_review": True,
                            "review_conflict_type": "unsupported_vector_route",
                            "review_reason": "; ".join(failures),
                        }
                    )
                    edge.system_confidence, edge.system_confidence_level = 0.35, "low"
                    edge.confidence = "low"
                    ambiguities.append(
                        {
                            "type": "unsupported_vector_route",
                            "page_index": result.page_index,
                            "edge_id": edge.id,
                            "status": "unresolved",
                            "reason": "; ".join(failures),
                        }
                    )
            if edge.from_node == edge.to_node:
                ambiguities.append(
                    {
                        "type": "rejected_self_loop",
                        "page_index": result.page_index,
                        "edge_id": edge.id,
                    }
                )
                continue
            if edge.from_node not in node_ids or edge.to_node not in node_ids:
                ambiguities.append(
                    {
                        "type": "rejected_dangling_edge",
                        "page_index": result.page_index,
                        "edge_id": edge.id,
                    }
                )
                continue
            uncertain = {n.id for n in nodes if n.attributes.get("knowledge_exception")}
            if edge.cross_sheet and uncertain.intersection({edge.from_node, edge.to_node}):
                ambiguities.append({"type": "unconfirmed_connector_relation", "edge_id": edge.id, "status": "unresolved"})
                continue
            graph_edges.append(edge)

    graph_edges = _deduplicate_edges(graph_edges)
    cross_edges, dangling_opcs, opc_ambiguities = _match_opcs(nodes, pages_by_index=pages_by_index)
    page_matched_opcs = {
        endpoint
        for edge in graph_edges
        if edge.cross_sheet
        for endpoint in (edge.from_node, edge.to_node)
    }
    if page_matched_opcs:
        dangling_opcs = [
            item for item in dangling_opcs if item.get("node_id") not in page_matched_opcs
        ]
        opc_ambiguities = [
            item
            for item in opc_ambiguities
            if not set(item.get("node_ids") or []).issubset(page_matched_opcs)
        ]
    ambiguities.extend(opc_ambiguities)
    edges = _deduplicate_edges([*graph_edges, *cross_edges])

    attached_nodes = {endpoint for edge in edges for endpoint in (edge.from_node, edge.to_node)}
    for node in nodes:
        if node.id in attached_nodes:
            score = min(1.0, float(node.system_confidence or 0.0) + 0.08)
            node.system_confidence = round(score, 3)
            node.system_confidence_level = _confidence_level(score)

    graph = ReconciledGraph(
        schema_version="0.3.0",
        source_path=objects.graph.source_path,
        assemblies=objects.graph.assemblies,
        text_bindings=objects.graph.text_bindings,
        nodes=sorted(
            nodes,
            key=lambda node: (node.page_index, node.bbox_global.y, node.bbox_global.x, node.id),
        ),
        edges=sorted(edges, key=lambda edge: edge.id),
        dangling_opcs=dangling_opcs,
        conflicts=ambiguities,
        per_page_status={
            int(page): _coerce_page_status(status) for page, status in per_page_status.items()
        },
    )
    return FusionResult(
        graph=graph,
        ambiguities=ambiguities,
        native_text_assignment_count=objects.native_text_assignment_count,
    )


def _prepare_detections(
    detections: list[DetectionRecord],
    pages_by_index: dict[int, PageEvidence],
    *,
    legend_pack: LegendPack | None = None,
    knowledge: dict | None = None,
) -> tuple[list[DetectionRecord], list[dict[str, Any]]]:
    accepted: list[DetectionRecord] = []
    annotations: list[DetectionRecord] = []
    rejected: list[dict[str, Any]] = []
    for detection in detections:
        item = detection.model_copy(deep=True)
        page = pages_by_index.get(item.page_index)
        if page is None:
            rejected.append(
                {
                    "type": "rejected_unknown_page_geometry",
                    "page_index": item.page_index,
                    "detection_id": item.id,
                    "status": "resolved",
                }
            )
            continue
        if (
            item.bbox.x < 0
            or item.bbox.y < 0
            or item.bbox.w <= 0
            or item.bbox.h <= 0
            or item.bbox.x2 > page.width
            or item.bbox.y2 > page.height
        ):
            rejected.append(
                {
                    "type": "rejected_out_of_page_geometry",
                    "page_index": item.page_index,
                    "detection_id": item.id,
                    "bbox": item.bbox.model_dump(),
                    "status": "resolved",
                }
            )
            continue
        retyped = _deterministic_tag_normalise(item, legend_pack=legend_pack)
        if retyped:
            rejected.append(retyped)
        if _is_native_text_only_detection(item, page):
            if _VALVE_STATE_RE.fullmatch(
                " ".join((item.label or item.raw_text or "").split()).strip()
            ):
                annotations.append(item)
            rejected.append(
                {
                    "type": "rejected_non_connectable_text",
                    "page_index": item.page_index,
                    "detection_id": item.id,
                    "label": item.label,
                    "bbox": item.bbox.model_dump(),
                    "status": "resolved",
                    "reason": (
                        "printed value and text-like geometry identify a line, "
                        "dimension, or valve-state annotation rather than an engineering object"
                    ),
                }
            )
            continue
        item.attributes = _normalise_taxonomy(item.attributes, kind=item.kind)
        if knowledge:
            from diagex.knowledge.resolver import resolve
            item.attributes["knowledge"] = resolve(knowledge, "symbol_interpretation", page.page_index, item.label + " connector arrow")
        if item.kind == "opc" and not _opc_has_boundary_support(item, page):
            from diagex.knowledge.candidates import connector_exception
            from diagex.knowledge.resolver import resolve
            context = resolve(knowledge, "symbol_interpretation", page.page_index, "connector arrow", legend_pack.model_dump(mode="json") if legend_pack else None)
            exception = connector_exception(item, page, context)
            if exception:
                item.attributes["knowledge_exception"] = exception
                item.attributes["opc_context_required"] = True
                item.attributes["requires_human_review"] = True
                rejected.append({"type": "unconventional_connector_placement", "status": "unresolved", "detection_id": item.id, "page_index": item.page_index, **exception})
            elif _opc_has_explicit_reference(item):
                item.attributes["opc_context_required"] = True
                item.attributes["continuation_evidence"] = "explicit_drawing_reference"
            else:
                rejected.append(
                    {
                        "type": "rejected_opc_without_boundary_evidence",
                        "page_index": item.page_index,
                        "detection_id": item.id,
                        "label": item.label,
                        "bbox": item.bbox.model_dump(),
                        "status": "resolved",
                        "reason": "OPC lacked both drawing-boundary support and an explicit continuation reference",
                    }
                )
                continue
        accepted.append(item)
    accepted, fragment_conflicts = _suppress_redundant_tag_fragments(accepted)
    rejected.extend(fragment_conflicts)
    rejected.extend(_attach_valve_state_annotations(accepted, annotations))
    return accepted, rejected


def _deterministic_tag_normalise(
    detection: DetectionRecord,
    *,
    legend_pack: LegendPack | None = None,
) -> dict[str, Any] | None:
    """Enrich or retype a node only when its printed tag is unambiguous."""
    semantics = infer_tag_semantics(detection.label or "", legend_pack)
    if semantics is None:
        return None
    attrs = detection.attributes
    original_kind = detection.kind
    can_retype_to_instrument = (
        original_kind == "equipment"
        and semantics.expected_kind == "instrument"
        and not attrs.get("valve_type")
        and attrs.get("equipment_class") in (None, "", "equipment", "unclassified_equipment")
    )
    can_retype_to_equipment = (
        original_kind == "instrument"
        and semantics.expected_kind == "equipment"
        and semantics.basis == "project_legend"
        and bool(semantics.attributes.get("valve_type"))
        and str(semantics.attributes.get("abbreviation_conflict") or "").casefold()
        not in {"true", "1", "yes"}
    )
    compatible = (
        original_kind == semantics.expected_kind
        or can_retype_to_instrument
        or can_retype_to_equipment
    )
    if not compatible:
        return {
            "type": "tag_kind_conflict",
            "page_index": detection.page_index,
            "detection_id": detection.id,
            "label": detection.label,
            "model_kind": original_kind,
            "expected_kind": semantics.expected_kind,
            "tag_semantics": semantics.model_dump(mode="json"),
            "status": "unresolved",
            "reason": "printed tag/legend semantics disagree with a specific detected object kind",
        }

    if can_retype_to_instrument:
        detection.kind = "instrument"
    elif can_retype_to_equipment:
        detection.kind = "equipment"
        if attrs.get("instrument_function") not in (None, "", "unclassified_instrument"):
            attrs = {**attrs, "model_instrument_function": attrs.get("instrument_function")}
    detection.attributes = {
        **attrs,
        **{key: value for key, value in semantics.attributes.items() if value not in (None, "")},
        "tag_prefix": semantics.prefix,
        "tag_semantics_basis": semantics.basis,
        "tag_legend_labels": semantics.legend_labels,
    }
    if can_retype_to_instrument or can_retype_to_equipment:
        detection.attributes.update(
            {
                "model_original_kind": original_kind,
                "deterministic_retype_basis": "unambiguous_tag_and_legend_semantics",
            }
        )
    if not (can_retype_to_instrument or can_retype_to_equipment):
        return None
    return {
        "type": "deterministic_kind_correction",
        "page_index": detection.page_index,
        "detection_id": detection.id,
        "label": detection.label,
        "from_kind": original_kind,
        "to_kind": detection.kind,
        "status": "resolved",
        "reason": "printed tag and project/standard legend semantics identify the node kind",
        "tag_semantics": semantics.model_dump(mode="json"),
    }


def _is_native_text_only_detection(
    detection: DetectionRecord,
    page: PageEvidence,
) -> bool:
    label = " ".join((detection.label or detection.raw_text or "").split()).strip()
    if not label:
        return False
    # A DWxx-xxxx value identifies another drawing sheet.  It can enrich an
    # OPC, but it is never an equipment or instrument identity by itself.
    if detection.kind != "opc" and _EXPLICIT_OPC_REF_RE.fullmatch(label):
        return True
    title_block_text = _is_title_block_bbox(detection.bbox, page) and any(
        normalise_label(span.text) == normalise_label(label)
        and _bbox_centres_overlap(detection.bbox, span.bbox)
        for span in page.text_spans
    )
    if _NON_CONNECTABLE_TEXT_RE.fullmatch(label) is None and not title_block_text:
        return False
    # Nominal sizes and dimensions are intrinsically document/line evidence,
    # never connectable graph entities.  Reject them even when the model bbox
    # is offset from the exact native-text box (a common crop-projection error).
    if re.fullmatch(
        r"(?:DN\s*\d+(?:\s*[x×/]\s*\d+)?|\d+(?:\.\d+)?\s*(?:[\"”′]|mm|cm))",
        label,
        re.IGNORECASE,
    ):
        return True
    if _VALVE_STATE_RE.fullmatch(label) or re.fullmatch(
        r"(?:(?:\d{1,4}[\"”']?[- ])?[A-Z]{1,6}-\d{3,}(?:-[A-Z0-9]+){1,}|"
        r"(?:SHEET|SHT|PAGE|REV|DWG)\s*[A-Z0-9-]+)",
        label,
        re.IGNORECASE,
    ):
        return True
    normalised = normalise_label(label)
    for span in page.text_spans:
        if normalise_label(span.text) != normalised:
            continue
        center_x = span.bbox.x + span.bbox.w / 2
        center_y = span.bbox.y + span.bbox.h / 2
        if not (
            detection.bbox.x <= center_x <= detection.bbox.x2
            and detection.bbox.y <= center_y <= detection.bbox.y2
        ):
            continue
        # Err on the cautious side: reject only a tight text-shaped box. A
        # larger symbol box that happens to contain this annotation survives.
        if detection.bbox.h <= max(24, span.bbox.h * 2.75) and detection.bbox.w <= max(
            40, span.bbox.w * 2.25
        ):
            return True
    return False


def _is_title_block_bbox(bbox: BBox, page: PageEvidence) -> bool:
    center_x = bbox.x + bbox.w / 2
    center_y = bbox.y + bbox.h / 2
    return center_x >= page.width * 0.55 and center_y >= page.height * 0.78




def _suppress_redundant_tag_fragments(
    detections: list[DetectionRecord],
) -> tuple[list[DetectionRecord], list[dict[str, Any]]]:
    """Drop OCR fragments only when a complete nearby tag already exists."""
    complete = [
        item
        for item in detections
        if infer_tag_semantics(item.label or "") is not None
        or bool(re.fullmatch(r"[A-Z]{1,6}[- ]?\d{2,6}[A-Z]?", item.label or "", re.I))
    ]
    kept: list[DetectionRecord] = []
    conflicts: list[dict[str, Any]] = []
    for item in detections:
        fragment = normalise_label(item.label or "")
        if not fragment or _TAG_FRAGMENT_RE.fullmatch(item.label or "") is None:
            kept.append(item)
            continue
        owner = next(
            (
                candidate
                for candidate in complete
                if candidate.id != item.id
                and candidate.page_index == item.page_index
                and fragment in normalise_label(candidate.label or "")
                and _center_distance(item.bbox, candidate.bbox)
                <= max(180.0, max(candidate.bbox.w, candidate.bbox.h) * 3.5)
            ),
            None,
        )
        if owner is None:
            kept.append(item)
            continue
        conflicts.append(
            {
                "type": "rejected_redundant_tag_fragment",
                "page_index": item.page_index,
                "detection_id": item.id,
                "label": item.label,
                "complete_detection_id": owner.id,
                "complete_label": owner.label,
                "status": "resolved",
                "reason": "nearby complete printed tag supersedes this OCR fragment",
            }
        )
    return kept, conflicts


def _attach_valve_state_annotations(
    detections: list[DetectionRecord],
    annotations: list[DetectionRecord],
) -> list[dict[str, Any]]:
    """Move FO/FC-like text onto a nearby valve instead of making a node."""
    results: list[dict[str, Any]] = []
    valves = [item for item in detections if item.attributes.get("valve_type")]
    mappings = {
        "FO": ("fail_action", "open"),
        "FC": ("fail_action", "closed"),
        "CSO": ("normal_state", "car_sealed_open"),
        "CSC": ("normal_state", "car_sealed_closed"),
        "LO": ("normal_state", "locked_open"),
        "LC": ("normal_state", "locked_closed"),
    }
    for annotation in annotations:
        code = " ".join((annotation.label or annotation.raw_text or "").split()).upper()
        key, value = mappings[code]
        candidates = sorted(
            (
                (_center_distance(annotation.bbox, valve.bbox), valve)
                for valve in valves
                if valve.page_index == annotation.page_index
            ),
            key=lambda row: (row[0], row[1].id),
        )
        limit = max(180.0, max(annotation.bbox.w, annotation.bbox.h) * 8.0)
        if not candidates or candidates[0][0] > limit:
            continue
        distance, valve = candidates[0]
        competing = len(candidates) > 1 and candidates[1][0] <= max(distance * 1.2, distance + 30)
        if competing or valve.attributes.get(key) not in (None, value):
            results.append(
                {
                    "type": "ambiguous_valve_state_annotation",
                    "page_index": annotation.page_index,
                    "detection_id": annotation.id,
                    "label": code,
                    "candidate_valve_ids": [row[1].id for row in candidates[:3]],
                    "status": "unresolved",
                    "reason": "valve-state annotation has no unique compatible nearby valve",
                }
            )
            continue
        valve.attributes[key] = value
        valve.attributes.setdefault("state_annotation_source_ids", []).append(annotation.id)
        results.append(
            {
                "type": "attached_valve_state_annotation",
                "page_index": annotation.page_index,
                "detection_id": annotation.id,
                "label": code,
                "valve_detection_id": valve.id,
                "attribute": key,
                "value": value,
                "status": "resolved",
                "reason": "project annotation was attached to the unique nearby valve",
            }
        )
    return results


def _opc_has_boundary_support(detection: DetectionRecord, page: PageEvidence) -> bool:
    margin = max(60.0, min(page.width, page.height) * 0.05)
    bbox = detection.bbox
    touches_boundary = (
        bbox.x <= margin
        or bbox.y <= margin
        or bbox.x2 >= page.width - margin
        or bbox.y2 >= page.height - margin
    )
    attrs = detection.attributes
    explicit = bool(
        attrs.get("direction") in {"in", "out"}
        or attrs.get("drawing_ref")
        or attrs.get("target_sheet")
        or attrs.get("line_id")
        or _OPC_HINT_RE.search(detection.label or detection.raw_text or "")
    )
    if touches_boundary and explicit:
        attrs["boundary_evidence"] = True
        return True
    return False


def _opc_has_explicit_reference(detection: DetectionRecord) -> bool:
    attrs = detection.attributes
    candidates = (
        detection.label,
        detection.raw_text,
        attrs.get("canonical_tag"),
        attrs.get("drawing_ref"),
        attrs.get("target_sheet"),
    )
    return any(_EXPLICIT_OPC_REF_RE.search(str(value or "")) for value in candidates)


def _node_from_cluster(
    cluster: InstanceCluster,
    pages_by_index: dict[int, PageEvidence],
) -> tuple[ReconciledNode, list[dict[str, Any]], int]:
    kind_counts = Counter(item.kind for item in cluster.detections)
    selected_kind = max(
        kind_counts,
        key=lambda kind: (
            kind_counts[kind],
            max(_CONF_RANK[item.confidence] for item in cluster.detections if item.kind == kind),
            kind == "equipment",
        ),
    )
    ordered = sorted(
        cluster.detections,
        key=lambda item: (
            item.kind == selected_kind,
            _CONF_RANK[item.confidence],
            bool(item.label),
            item.id,
        ),
        reverse=True,
    )
    canonical = ordered[0]
    fused_bbox = cluster.bbox or canonical.bbox.model_copy()
    labels = _ordered_unique(item.label for item in ordered if item.label)
    raw_readings = _ordered_unique(item.raw_text for item in ordered if item.raw_text)
    source_tiles = sorted({item.tile_id for item in ordered})

    page = pages_by_index[canonical.page_index]
    native_candidates, label_conflict, selected_label, native_resolved_label = resolve_symbol_text(
        canonical_label=canonical.label, raw_readings=raw_readings, labels=labels,
        spans=page.text_spans, bbox=fused_bbox)

    # A geometry cluster can contain two nearby printed identities (for
    # example an equipment tag and an adjacent valve/instrument tag).  Preserve
    # the conflict as evidence, but never let semantic attributes learned for
    # one identity silently contaminate the identity selected by native text.
    selected_key = normalise_label(selected_label)
    identity_contributors = [
        item
        for item in ordered
        if item.kind == selected_kind
        and (
            normalise_label(item.label or "") == selected_key
            or normalise_label(item.label or "") in _GENERIC_LABEL_KEYS
        )
    ]
    if not identity_contributors:
        identity_contributors = [canonical]
    identity_canonical = identity_contributors[0]
    text_ids = sorted({value for item in identity_contributors for value in item.source_text_ids})
    evidence_ids = sorted({item.id for item in ordered} | set(text_ids))

    attributes: dict[str, Any] = {}
    for item in identity_contributors:
        for key, value in item.attributes.items():
            if key not in attributes and value not in (None, "", [], {}):
                attributes[key] = value
    attributes["source_tiles"] = source_tiles
    attributes["fusion_evidence"] = {
        "version": FUSION_VERSION,
        "geometry_basis": cluster.geometry_basis,
        "physical_symbol_id": cluster.symbol_id,
        "scene_version": SCENE_VERSION,
        "representative_detection_id": cluster.representative_id,
        "source_path_ids": cluster.path_ids,
        "member_detection_ids": sorted(item.id for item in ordered),
    }
    evidence_ids = sorted(set(evidence_ids) | set(cluster.path_ids))
    attributes["model_confidence"] = identity_canonical.confidence
    attributes = _normalise_taxonomy(attributes, kind=selected_kind)
    for span in native_candidates:
        if span.id not in text_ids:
            text_ids.append(span.id)
            evidence_ids.append(span.id)
    text_ids.sort()
    evidence_ids = sorted(set(evidence_ids))
    if native_candidates:
        attributes["raw_text_candidates"] = [span.text for span in native_candidates]
        attributes["source_text_ids"] = text_ids

    native_agreement = native_resolved_label is not None
    score = {"high": 0.72, "medium": 0.54, "low": 0.34}[identity_canonical.confidence]
    score += min(0.14, max(0, len(ordered) - 1) * 0.07)
    score += 0.12 if native_agreement else 0.0
    score += 0.04 if raw_readings else 0.0
    score -= 0.18 if label_conflict and not native_resolved_label else 0.0
    if selected_kind == "opc" and attributes.get("opc_context_required"):
        score -= 0.14
    score = round(max(0.05, min(0.98, score)), 3)
    attributes["system_confidence"] = score
    attributes["system_confidence_evidence"] = {
        "overlapping_detections": len(ordered),
        "native_text_agreement": native_agreement,
        "label_conflict": label_conflict,
    }

    node_id = _stable_node_id(
        identity_canonical,
        selected_label,
        bbox=fused_bbox,
    )
    ambiguities: list[dict[str, Any]] = []
    if label_conflict:
        ambiguities.append(
            {
                "type": "evidence_label_conflict",
                "page_index": canonical.page_index,
                "node_id": node_id,
                "labels": labels,
                "bbox": fused_bbox.model_dump(),
                "status": "resolved" if native_resolved_label else "unresolved",
                "reason": (
                    "exact native PDF text match"
                    if native_resolved_label
                    else "overlapping observations disagree"
                ),
            }
        )
    if len(kind_counts) > 1:
        dominant = kind_counts[selected_kind] > max(
            count for kind, count in kind_counts.items() if kind != selected_kind
        )
        ambiguities.append(
            {
                "type": "evidence_kind_conflict",
                "page_index": canonical.page_index,
                "node_id": node_id,
                "kinds": dict(kind_counts),
                "selected_kind": selected_kind,
                "bbox": fused_bbox.model_dump(),
                "status": "resolved" if dominant else "unresolved",
                "reason": "dominant overlapping evidence"
                if dominant
                else "equal conflicting evidence",
            }
        )
    if selected_kind == "opc" and attributes.get("opc_context_required"):
        ambiguities.append(
            {
                "type": "opc_context_validation",
                "page_index": canonical.page_index,
                "node_id": node_id,
                "label": selected_label,
                "bbox": fused_bbox.model_dump(),
                "status": "unresolved",
                "reason": "explicit continuation reference is away from the physical PDF edge",
            }
        )

    return (
        ReconciledNode(
            id=node_id,
            kind=selected_kind,
            label=selected_label,
            bbox_global=fused_bbox,
            page_index=canonical.page_index,
            attributes=attributes,
            confidence=identity_canonical.confidence,
            source_quote=(
                identity_canonical.raw_text
                or identity_canonical.label
                or (raw_readings[0] if raw_readings else None)
            ),
            alternate_readings=[label for label in labels if label != selected_label],
            source_annotation_ids=[item.id for item in ordered],
            source_evidence_ids=evidence_ids,
            system_confidence=score,
            system_confidence_level=_confidence_level(score),
        ),
        ambiguities,
        len(native_candidates),
    )










def _normalise_taxonomy(attributes: dict[str, Any], *, kind: str) -> dict[str, Any]:
    attrs = dict(attributes)
    direction = _normalise_key(attrs.get("direction"))
    direction_aliases = {
        "to": "out",
        "outgoing": "out",
        "outlet": "out",
        "outflow": "out",
        "to_network": "out",
        "from": "in",
        "incoming": "in",
        "inlet": "in",
        "inflow": "in",
        "inbound": "in",
    }
    if direction:
        attrs["direction"] = direction_aliases.get(direction, direction)

    if kind == "equipment":
        raw_equipment = _normalise_key(attrs.get("equipment_class"))
        raw_valve = _normalise_key(attrs.get("valve_type"))
        if raw_equipment == "valve" and not raw_valve:
            raw_valve = "other"
        if raw_valve:
            normalised_valve = _VALVE_ALIASES.get(raw_valve, raw_valve)
            if normalised_valve not in VALVE_TYPE_KEYS:
                normalised_valve = "other"
            if normalised_valve != raw_valve:
                attrs.setdefault("model_valve_type", raw_valve)
            attrs["valve_type"] = normalised_valve
            attrs.pop("equipment_class", None)
        elif raw_equipment:
            normalised_equipment = _EQUIPMENT_ALIASES.get(raw_equipment, raw_equipment)
            if normalised_equipment not in EQUIPMENT_CLASS_KEYS:
                normalised_equipment = "unclassified_equipment"
            if normalised_equipment != raw_equipment:
                attrs.setdefault("model_equipment_class", raw_equipment)
            attrs["equipment_class"] = normalised_equipment

    if kind == "instrument":
        raw_class = _normalise_key(attrs.get("instrument_class"))
        raw_function = _normalise_key(attrs.get("instrument_function"))
        if raw_class and raw_class not in INSTRUMENT_CLASS_KEYS:
            attrs.setdefault("model_instrument_class", raw_class)
            attrs.pop("instrument_class", None)
            if not raw_function or raw_function == "unclassified_instrument":
                raw_function = _instrument_function_from_text(raw_class)
            variable = _measured_variable_from_text(raw_class)
            if variable and not attrs.get("measured_variable"):
                attrs["measured_variable"] = variable
        if raw_function:
            normalised_function = _instrument_function_from_text(raw_function)
            attrs["instrument_function"] = normalised_function
        elif not attrs.get("instrument_function"):
            attrs["instrument_function"] = "unclassified_instrument"
    return attrs


def _normalise_key(value: Any) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())).strip("_")


def _instrument_function_from_text(value: str) -> str:
    for key in (
        "controller",
        "transmitter",
        "indicator",
        "recorder",
        "element",
        "switch",
        "alarm",
        "valve_actuator",
    ):
        if key in value:
            return key
    return value if value in INSTRUMENT_FUNCTION_KEYS else "unclassified_instrument"


def _measured_variable_from_text(value: str) -> str | None:
    for key in ("pressure", "temperature", "flow", "level", "analysis"):
        if key in value:
            return key
    return None


def _stable_node_id(
    detection: DetectionRecord,
    label: str,
    *,
    bbox: BBox | None = None,
) -> str:
    bbox = bbox or detection.bbox
    quantized = tuple(int(round(value / 10.0) * 10) for value in (bbox.x, bbox.y, bbox.w, bbox.h))
    payload = repr(
        (detection.page_index, detection.kind, normalise_label(label), quantized)
    ).encode("utf-8")
    return "n-" + hashlib.sha256(payload).hexdigest()[:16]


def _enrich_opcs_from_native_text(
    nodes: list[ReconciledNode],
    pages_by_index: dict[int, PageEvidence],
) -> tuple[list[ReconciledNode], list[dict[str, Any]]]:
    """Verify model OPCs and recover explicit body continuation references.

    Drawing references in the sheet title block identify the current page;
    references in the drawing body identify a continuation.  This distinction
    is deterministic and does not require the connector to sit on the physical
    edge of the PDF page.
    """
    opcs_by_page: dict[int, list[ReconciledNode]] = {}
    for node in nodes:
        if node.kind == "opc":
            opcs_by_page.setdefault(node.page_index, []).append(node)

    sheet_refs = {
        page_index: _page_drawing_reference(page) for page_index, page in pages_by_index.items()
    }
    pages_for_sheet: dict[str, list[int]] = {}
    body_refs_by_page: dict[int, set[str]] = {}
    for page_index, page in pages_by_index.items():
        own_ref = sheet_refs.get(page_index)
        if own_ref:
            pages_for_sheet.setdefault(own_ref, []).append(page_index)
        body_refs_by_page[page_index] = {
            reference
            for span in _body_drawing_reference_spans(page)
            if (reference := _normalise_drawing_reference(span.text)) is not None
        }

    synthesized: list[ReconciledNode] = []
    diagnostics: list[dict[str, Any]] = []
    for page_index, page in pages_by_index.items():
        for span in _body_drawing_reference_spans(page):
            reference = _normalise_drawing_reference(span.text)
            if reference is None:
                continue
            nearby = _nearby_body_text(page, span.bbox)
            source_ids = sorted({span.id, *(value.id for value in nearby)})
            source_text = " ".join(value.text for value in [span, *nearby]).strip()
            direction = _opc_direction_from_text(source_text)
            service = _opc_service_from_text(source_text, reference)
            page_opcs = opcs_by_page.get(page_index, [])
            exact = sorted(
                (
                    node
                    for node in page_opcs
                    if _opc_visible_reference(node) == reference
                    or _normalise_drawing_reference(node.attributes.get("drawing_ref")) == reference
                ),
                key=lambda node: (
                    bool(node.attributes.get("system_generated")),
                    _center_distance(node.bbox_global, span.bbox),
                    node.id,
                ),
            )
            candidates = sorted(
                ((_center_distance(node.bbox_global, span.bbox), node) for node in page_opcs),
                key=lambda row: (row[0], row[1].id),
            )
            limit = max(320.0, min(page.width, page.height) * 0.08)
            target = (
                exact[0]
                if exact
                else (candidates[0][1] if candidates and candidates[0][0] <= limit else None)
            )
            if target is not None:
                target.attributes.update(
                    {
                        "drawing_ref": reference,
                        "native_reference_verified": True,
                        "continuation_evidence": "native_body_drawing_reference",
                    }
                )
                target.attributes.pop("opc_context_required", None)
                if direction:
                    target.attributes["direction"] = direction
                if service:
                    target.attributes.setdefault("service", service)
                _refresh_opc_label(target)
                target.attributes["source_text_ids"] = sorted(
                    set(target.attributes.get("source_text_ids") or []) | set(source_ids)
                )
                target.source_evidence_ids = sorted(
                    set(target.source_evidence_ids) | set(source_ids)
                )
                diagnostics.append(
                    {
                        "type": "native_opc_reference_assignment",
                        "page_index": page_index,
                        "node_id": target.id,
                        "drawing_ref": reference,
                        "status": "resolved",
                        "reason": "positioned body text verifies the continuation destination",
                    }
                )
                continue

            current_ref = sheet_refs.get(page_index)
            destination_pages = pages_for_sheet.get(reference, [])
            reciprocal_destination = (
                len(destination_pages) == 1
                and current_ref is not None
                and current_ref in body_refs_by_page.get(destination_pages[0], set())
            )
            explicit_glyph = _native_ref_has_explicit_connector_glyph(page, span.bbox)
            if not reciprocal_destination and not explicit_glyph:
                diagnostics.append(
                    {
                        "type": "unassigned_native_drawing_reference",
                        "page_index": page_index,
                        "source_text_id": span.id,
                        "drawing_ref": reference,
                        "status": "unresolved",
                        "reason": (
                            "body drawing reference has no detected OPC, reciprocal destination, "
                            "or explicit connector glyph"
                        ),
                    }
                )
                continue
            bbox = _bbox_union_values([span.bbox, *(value.bbox for value in nearby)])
            node_id = (
                "n-opc-"
                + hashlib.sha256(
                    repr((page_index, reference, bbox.x // 10, bbox.y // 10)).encode()
                ).hexdigest()[:12]
            )
            attributes: dict[str, Any] = {
                "drawing_ref": reference,
                "native_reference_verified": True,
                "continuation_evidence": (
                    "reciprocal_native_drawing_references"
                    if reciprocal_destination
                    else "native_body_drawing_reference_and_explicit_connector_glyph"
                ),
                "source_text_ids": source_ids,
                "system_generated": True,
            }
            if direction:
                attributes["direction"] = direction
            if service:
                attributes["service"] = service
            node = ReconciledNode(
                id=node_id,
                kind="opc",
                label=service or reference,
                bbox_global=bbox,
                page_index=page_index,
                attributes=attributes,
                confidence="medium",
                source_quote=source_text,
                source_evidence_ids=source_ids,
                system_confidence=0.78,
                system_confidence_level="medium",
            )
            synthesized.append(node)
            opcs_by_page.setdefault(page_index, []).append(node)
            diagnostics.append(
                {
                    "type": "native_opc_recovery",
                    "page_index": page_index,
                    "node_id": node_id,
                    "drawing_ref": reference,
                    "status": "resolved",
                    "reason": "native drawing reference and nearby vector continuation recovered a missed OPC",
                }
            )
    return synthesized, diagnostics


def _refresh_opc_label(node: ReconciledNode) -> None:
    """Backfill a useful label after native/page-graph OPC enrichment."""
    if node.kind != "opc" or normalise_label(node.label or "") not in _GENERIC_LABEL_KEYS:
        return
    service = " ".join(str(node.attributes.get("service") or "").split())
    reference = _normalise_drawing_reference(node.attributes.get("drawing_ref"))
    direction = str(node.attributes.get("direction") or "").casefold()
    if service and reference:
        node.label = (
            f"{reference} → {service}"
            if direction == "in"
            else f"{service} → {reference}"
            if direction == "out"
            else f"{service} · {reference}"
        )
    elif service:
        node.label = service
    elif reference:
        node.label = reference


def _body_drawing_reference_spans(page: PageEvidence) -> list[TextEvidence]:
    return [
        span
        for span in page.text_spans
        if _EXPLICIT_OPC_REF_RE.search(span.text) and not _is_title_block_bbox(span.bbox, page)
    ]


def _normalise_drawing_reference(value: Any) -> str | None:
    match = _EXPLICIT_OPC_REF_RE.search(str(value or ""))
    return re.sub(r"\s+", "", match.group(0)).upper() if match else None


def _nearby_body_text(page: PageEvidence, bbox: BBox) -> list[TextEvidence]:
    x_pad = max(260, int(page.width * 0.05))
    y_pad = max(80, int(page.height * 0.018))
    return sorted(
        (
            span
            for span in page.text_spans
            if span.bbox != bbox
            and not _is_title_block_bbox(span.bbox, page)
            and span.bbox.x < bbox.x2 + x_pad
            and span.bbox.x2 > bbox.x - x_pad
            and span.bbox.y < bbox.y2 + y_pad
            and span.bbox.y2 > bbox.y - y_pad
        ),
        key=lambda span: (_center_distance(span.bbox, bbox), span.id),
    )[:8]


def _opc_direction_from_text(value: str) -> str | None:
    text = value.casefold()
    if re.search(r"(?:来自|来?自|\bfrom\b|入口|inlet)", text):
        return "in"
    if re.search(r"(?:去往|接至|\bto\b|出口|outlet)", text):
        return "out"
    return None


def _opc_service_from_text(value: str, reference: str) -> str | None:
    text = re.sub(re.escape(reference), " ", value, flags=re.IGNORECASE)
    text = re.sub(r"(?:来自|去往|接至|\bfrom\b|\bto\b)", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip(" -–—:：")
    return text[:160] or None


def _native_ref_has_explicit_connector_glyph(page: PageEvidence, bbox: BBox) -> bool:
    """Require a compact connector-shaped mark, not merely a nearby pipe."""
    radius = max(180.0, min(page.width, page.height) * 0.045)
    for path in page.paths:
        if path.primitive not in {"line", "curve", "rect", "quad"} or len(path.points) < 2:
            continue
        size = max(path.bbox.w, path.bbox.h)
        if size < 12 or size > 180:
            continue
        has_shape = path.closed or path.primitive in {"rect", "quad"} or len(path.points) >= 3
        if has_shape and _center_distance(path.bbox, bbox) <= radius:
            return True
    return False


def _bbox_union_values(values: list[BBox]) -> BBox:
    left = min(value.x for value in values)
    top = min(value.y for value in values)
    right = max(value.x2 for value in values)
    bottom = max(value.y2 for value in values)
    return BBox(x=left, y=top, w=max(1, right - left), h=max(1, bottom - top))


def _match_opcs(
    nodes: list[ReconciledNode],
    *,
    pages_by_index: dict[int, PageEvidence],
) -> tuple[list[ReconciledEdge], list[dict[str, Any]], list[dict[str, Any]]]:
    opcs = [node for node in nodes if node.kind == "opc" and not node.attributes.get("knowledge_exception")]
    sheet_refs = {
        page_index: _page_drawing_reference(page) for page_index, page in pages_by_index.items()
    }
    matched: set[str] = set()
    edges: list[ReconciledEdge] = []
    ambiguities: list[dict[str, Any]] = []

    # A pair is safe to create without an LLM when the connector references
    # are reciprocal with the two title-block drawing numbers, the directions
    # oppose, the services do not conflict, and neither endpoint has an equally
    # supported alternative. This is the native convention used by the 2401
    # drawing and is substantially stronger than equal-label matching.
    reciprocal_candidates: list[tuple[float, ReconciledNode, ReconciledNode]] = []
    for index, left in enumerate(opcs):
        for right in opcs[index + 1 :]:
            if left.page_index == right.page_index:
                continue
            left_ref = _opc_visible_reference(left)
            right_ref = _opc_visible_reference(right)
            left_sheet = sheet_refs.get(left.page_index)
            right_sheet = sheet_refs.get(right.page_index)
            if not (
                left_ref
                and right_ref
                and left_sheet
                and right_sheet
                and left_ref == right_sheet
                and right_ref == left_sheet
            ):
                continue
            left_direction = str(left.attributes.get("direction") or "").casefold()
            right_direction = str(right.attributes.get("direction") or "").casefold()
            if (
                left_direction
                and right_direction
                and {left_direction, right_direction}
                != {
                    "in",
                    "out",
                }
            ):
                continue
            compatible, service_score = _opc_services_compatible(left, right)
            if not compatible:
                continue
            reciprocal_candidates.append((10.0 + service_score, left, right))

    selected = _maximum_weight_opc_pairs(opcs, reciprocal_candidates)
    candidates_by_node: dict[str, list[float]] = {}
    for score, left, right in reciprocal_candidates:
        candidates_by_node.setdefault(left.id, []).append(score)
        candidates_by_node.setdefault(right.id, []).append(score)
    for score, left, right in selected:
        left_scores = sorted(candidates_by_node.get(left.id, []), reverse=True)
        right_scores = sorted(candidates_by_node.get(right.id, []), reverse=True)
        unique = (len(left_scores) == 1 or left_scores[0] - left_scores[1] >= 1.0) and (
            len(right_scores) == 1 or right_scores[0] - right_scores[1] >= 1.0
        )
        if not unique:
            ambiguities.append(
                _opc_ambiguity(
                    left,
                    right,
                    score=score,
                    reason="reciprocal sheet references have a competing one-to-one candidate",
                    sheet_refs=sheet_refs,
                )
            )
            continue
        matched.update((left.id, right.id))
        explicit_opposition = {
            str(left.attributes.get("direction") or "").casefold(),
            str(right.attributes.get("direction") or "").casefold(),
        } == {"in", "out"}
        edges.append(
            _opc_edge(
                left,
                right,
                confidence="high" if explicit_opposition else "medium",
                system_confidence=0.94 if explicit_opposition else 0.84,
                attributes={
                    "match_basis": "reciprocal_title_block_references",
                    "left_sheet_ref": sheet_refs.get(left.page_index),
                    "right_sheet_ref": sheet_refs.get(right.page_index),
                    "candidate_score": round(score, 3),
                    "direction_basis": (
                        "opposing_printed_directions"
                        if explicit_opposition
                        else "reciprocal_references_without_complete_direction_text"
                    ),
                },
            )
        )

    reference_groups: dict[str, list[ReconciledNode]] = {}
    for node in opcs:
        if node.id in matched:
            continue
        reference = _opc_reference(node)
        if reference:
            reference_groups.setdefault(reference, []).append(node)
    for reference, values in sorted(reference_groups.items()):
        ordered_values = sorted(values, key=lambda node: (node.page_index, node.id))
        candidates = _opc_pair_candidates(ordered_values)
        selected_pairs = _maximum_weight_opc_pairs(ordered_values, candidates)
        candidate_table = [
            {
                "node_ids": [left.id, right.id],
                "pages": [left.page_index + 1, right.page_index + 1],
                "score": round(score, 3),
                "left_direction": left.attributes.get("direction"),
                "right_direction": right.attributes.get("direction"),
                "left_service": left.attributes.get("service"),
                "right_service": right.attributes.get("service"),
                "left_line_id": left.attributes.get("line_id"),
                "right_line_id": right.attributes.get("line_id"),
            }
            for score, left, right in candidates[:16]
        ]
        for score, left, right in selected_pairs:
            ambiguities.append(
                {
                    "type": "ambiguous_opc_evidence",
                    "reference": reference,
                    "node_ids": [left.id, right.id],
                    "page_indices": [left.page_index, right.page_index],
                    "candidate_score": round(score, 3),
                    "candidate_group_size": len(ordered_values),
                    "candidate_pairs": candidate_table,
                    "status": "unresolved",
                    "reason": (
                        "maximum-weight one-to-one candidate requires bounded visual confirmation"
                    ),
                }
            )

    dangling = [
        {
            "node_id": node.id,
            "label": node.label,
            "page_index": node.page_index,
            "reason": "no uniquely supported evidence-v2 OPC pair",
        }
        for node in opcs
        if node.id not in matched
    ]
    return edges, dangling, ambiguities


def _page_drawing_reference(page: PageEvidence) -> str | None:
    """Read the sheet's own drawing number from its lower-right title block."""
    candidates: list[tuple[float, str]] = []
    for span in page.text_spans:
        match = _EXPLICIT_OPC_REF_RE.search(span.text)
        if match is None:
            continue
        # Connector references occur in the drawing body. The sheet identifier
        # is conventionally in the lower-right title block; require that region
        # so a continuation label cannot masquerade as the current sheet.
        if span.bbox.x < page.width * 0.55 or span.bbox.y < page.height * 0.78:
            continue
        reference = re.sub(r"\s+", "", match.group(0)).upper()
        score = span.bbox.x / page.width + span.bbox.y / page.height
        candidates.append((score, reference))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1]))
    return candidates[0][1]


def _opc_visible_reference(node: ReconciledNode) -> str | None:
    """Return only a reference supported by visible node text.

    A model-only ``drawing_ref`` on an unlabelled symbol is useful evidence for
    review but is not strong enough for an automatic cross-sheet edge.
    """
    values = (
        node.label,
        node.source_quote,
        node.attributes.get("canonical_tag"),
    )
    for value in values:
        match = _EXPLICIT_OPC_REF_RE.search(str(value or ""))
        if match:
            return re.sub(r"\s+", "", match.group(0)).upper()
    if node.attributes.get("native_reference_verified"):
        return _normalise_drawing_reference(node.attributes.get("drawing_ref"))
    return None


def _opc_services_compatible(
    left: ReconciledNode,
    right: ReconciledNode,
) -> tuple[bool, float]:
    left_service = _opc_service_key(left.attributes.get("service"))
    right_service = _opc_service_key(right.attributes.get("service"))
    if not left_service or not right_service:
        return True, 0.0
    if left_service == right_service:
        return True, 2.0
    if left_service in right_service or right_service in left_service:
        return True, 1.0
    return False, 0.0


def _opc_ambiguity(
    left: ReconciledNode,
    right: ReconciledNode,
    *,
    score: float,
    reason: str,
    sheet_refs: dict[int, str | None],
) -> dict[str, Any]:
    return {
        "type": "ambiguous_opc_evidence",
        "node_ids": [left.id, right.id],
        "page_indices": [left.page_index, right.page_index],
        "candidate_score": round(score, 3),
        "sheet_references": [
            sheet_refs.get(left.page_index),
            sheet_refs.get(right.page_index),
        ],
        "status": "unresolved",
        "reason": reason,
    }


def _opc_edge(
    left: ReconciledNode,
    right: ReconciledNode,
    *,
    confidence: Confidence,
    system_confidence: float,
    attributes: dict[str, Any],
) -> ReconciledEdge:
    left_direction = str(left.attributes.get("direction") or "").casefold()
    right_direction = str(right.attributes.get("direction") or "").casefold()
    if left_direction == "out" or right_direction == "in":
        outgoing, incoming = left, right
    elif right_direction == "out" or left_direction == "in":
        outgoing, incoming = right, left
    else:
        outgoing, incoming = sorted((left, right), key=lambda node: (node.page_index, node.id))
    edge_id = "e-opc-" + hashlib.sha256(f"{outgoing.id}|{incoming.id}".encode()).hexdigest()[:12]
    return ReconciledEdge(
        id=edge_id,
        from_node=outgoing.id,
        to_node=incoming.id,
        line_type="process",
        cross_sheet=True,
        confidence=confidence,
        source_evidence_ids=sorted(set(left.source_evidence_ids + right.source_evidence_ids)),
        system_confidence=system_confidence,
        system_confidence_level=_confidence_level(system_confidence),
        attributes=attributes,
    )


def _opc_pair_candidates(
    values: list[ReconciledNode],
) -> list[tuple[float, ReconciledNode, ReconciledNode]]:
    return sorted(
        (
            (_opc_pair_score(left, right), left, right)
            for index, left in enumerate(values)
            for right in values[index + 1 :]
            if left.page_index != right.page_index
        ),
        key=lambda row: (-row[0], row[1].id, row[2].id),
    )


def _maximum_weight_opc_pairs(
    values: list[ReconciledNode],
    candidates: list[tuple[float, ReconciledNode, ReconciledNode]],
) -> list[tuple[float, ReconciledNode, ReconciledNode]]:
    """Return a deterministic globally optimal set of disjoint candidate pairs."""
    if not candidates:
        return []
    if len(values) > 16:
        # Exact subset matching is exponential. Extremely repetitive connector
        # references are already low-information, so use the same deterministic
        # score order while preserving the one-to-one invariant.
        claimed: set[str] = set()
        selected: list[tuple[float, ReconciledNode, ReconciledNode]] = []
        for score, left, right in candidates:
            if left.id in claimed or right.id in claimed:
                continue
            claimed.update((left.id, right.id))
            selected.append((score, left, right))
        return selected
    by_id = {node.id: node for node in values}
    score_by_pair = {tuple(sorted((left.id, right.id))): score for score, left, right in candidates}
    ids = tuple(sorted(by_id))

    @cache
    def solve(remaining: tuple[str, ...]) -> tuple[float, tuple[tuple[str, str], ...]]:
        if len(remaining) < 2:
            return 0.0, ()
        first = remaining[0]
        best_score, best_pairs = solve(remaining[1:])
        for offset, other in enumerate(remaining[1:], start=1):
            key = tuple(sorted((first, other)))
            pair_score = score_by_pair.get(key)
            if pair_score is None:
                continue
            rest = remaining[1:offset] + remaining[offset + 1 :]
            rest_score, rest_pairs = solve(rest)
            proposed_score = pair_score + rest_score
            proposed_pairs = tuple(sorted((key, *rest_pairs)))
            if proposed_score > best_score or (
                math.isclose(proposed_score, best_score) and proposed_pairs < best_pairs
            ):
                best_score, best_pairs = proposed_score, proposed_pairs
        return best_score, best_pairs

    _, selected = solve(ids)
    result = [(score_by_pair[pair], by_id[pair[0]], by_id[pair[1]]) for pair in selected]
    return sorted(result, key=lambda row: (-row[0], row[1].id, row[2].id))


def _opc_reference(node: ReconciledNode) -> str | None:
    values = (
        node.label,
        node.attributes.get("canonical_tag"),
        node.attributes.get("drawing_ref"),
    )
    for value in values:
        match = _EXPLICIT_OPC_REF_RE.search(str(value or ""))
        if match:
            return re.sub(r"\s+", "", match.group(0)).upper()
    return None


def _opc_pair_score(left: ReconciledNode, right: ReconciledNode) -> float:
    score = 1.0
    left_direction = str(left.attributes.get("direction") or "").casefold()
    right_direction = str(right.attributes.get("direction") or "").casefold()
    if {left_direction, right_direction} == {"in", "out"}:
        score += 1.5
    left_service = _opc_service_key(left.attributes.get("service"))
    right_service = _opc_service_key(right.attributes.get("service"))
    if left_service and right_service:
        if left_service == right_service:
            score += 2.0
        elif left_service in right_service or right_service in left_service:
            score += 1.0
    left_line = normalise_label(str(left.attributes.get("line_id") or ""))
    right_line = normalise_label(str(right.attributes.get("line_id") or ""))
    if left_line and left_line == right_line:
        score += 2.0
    return score


def _opc_service_key(value: Any) -> str:
    text = str(value or "").casefold()
    service_aliases = (
        (("instrument air", "仪表空气"), "instrument_air"),
        (("compressed air", "压缩空气"), "compressed_air"),
        (("dry air", "dried air", "干燥风", "干燥空气"), "dried_air"),
        (("purified air", "净化风", "净化空气"), "purified_air"),
        (("nitrogen", "氮气"), "nitrogen"),
        (("cooling water supply", "循环冷却水给水", "cws"), "cooling_water_supply"),
        (("cooling water return", "循环冷却水回水", "cwr"), "cooling_water_return"),
    )
    for aliases, canonical in service_aliases:
        if any(alias in text for alias in aliases):
            return canonical
    text = re.sub(r"(?:来自|去往|至|自|\bto\b|\bfrom\b)", " ", text)
    text = re.sub(r"2401[-a-z0-9/]+", " ", text)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def _deduplicate_edges(edges: list[ReconciledEdge]) -> list[ReconciledEdge]:
    selected: dict[tuple[Any, ...], ReconciledEdge] = {}
    for edge in edges:
        endpoints = sorted((edge.from_node, edge.to_node))
        poly_ends = []
        if edge.polyline_global:
            poly_ends = sorted(
                (
                    tuple(int(round(value / 10.0) * 10) for value in edge.polyline_global[0]),
                    tuple(int(round(value / 10.0) * 10) for value in edge.polyline_global[-1]),
                )
            )
        key = (
            endpoints[0],
            endpoints[1],
            str(edge.line_type or "other"),
            tuple(poly_ends),
        )
        previous = selected.get(key)
        if previous is None or len(edge.polyline_global) < len(previous.polyline_global):
            selected[key] = edge
    return list(selected.values())








def _coerce_page_status(status: str) -> str:
    return status if status in {"ok", "partial", "cost_exhausted", "error"} else "error"


def _assign_text_stage(nodes, pages, legend_pack, recorder, knowledge=None):
    request = {
        "schema_version": "1.0",
        "stage": "text_assignment",
        "backend": "geometry",
        "inputs": {
            "nodes": [n.model_dump(mode="json") for n in nodes],
            "pages": [p.model_dump(mode="json") for p in pages],
            "legend_pack": legend_pack.model_dump(mode="json") if legend_pack else None,
            "knowledge": knowledge or {},
        },
    }
    result = assign_text(
        nodes=nodes, pages=pages, legend_pack=legend_pack, knowledge=knowledge
    )
    if recorder:
        recorder(request, result.model_dump(mode="json"))
    return result
