"""Assign text to existing symbols and equipment scopes; never detect symbols."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, Field

from diagex.vision.evidence import PageEvidence, TextEvidence
from diagex.vision.legend_models import LegendPack
from diagex.vision.models import BBox, Confidence, EquipmentAssembly, ReconciledNode
from diagex.vision.native_text import infer_tag_semantics
from diagex.vision.reconcile import normalise_label

TEXT_ASSIGNMENT_VERSION = "1.0.0"
_TAG_RE = re.compile(r"(?:[A-Z]{2,5}[- ]?\d{1,6}[A-Z]?|DN\s*\d+)", re.IGNORECASE)
_EXPLICIT_OPC_REF_RE = re.compile(r"\bDW\d{2}[- ]?\d{3,5}\b", re.IGNORECASE)
_GENERIC_LABEL_KEYS = {"", "UNLABELLED", "UNLABELED", "UNKNOWN", "NONE", "NA"}


class TextAssignmentResult(BaseModel):
    nodes: list[ReconciledNode]
    assemblies: list[EquipmentAssembly] = Field(default_factory=list)
    text_bindings: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    assigned_count: int = 0
    knowledge: list[dict] = Field(default_factory=list)


def assign_text(
    *,
    nodes: list[ReconciledNode],
    pages: list[PageEvidence],
    legend_pack: LegendPack | None = None,
    knowledge: dict | None = None,
) -> TextAssignmentResult:
    from diagex.vision.native_hierarchy import apply_assembly_bindings, build_native_hierarchy

    # Assignment never mutates its upstream symbol inventory.
    nodes = [node.model_copy(deep=True) for node in nodes]
    scenes = [build_native_hierarchy(page, nodes) for page in pages if page.role == "pid"]
    result = TextAssignmentResult(
        nodes=nodes,
        assemblies=[a for s in scenes for a in s.assemblies],
        text_bindings=[b for s in scenes for b in s.text_bindings],
    )
    if knowledge:
        from diagex.knowledge.resolver import resolve
        result.knowledge = [resolve(knowledge, "text_assignment", p.page_index, "connector tag text", legend_pack.model_dump(mode="json") if legend_pack else None) for p in pages]
        result.conflicts.extend({"type": "knowledge_definition_conflict", **conflict} for ctx in result.knowledge for conflict in ctx["conflicts"])
        for node in result.nodes:
            node.attributes["text_assignment_knowledge"] = next((c for c in result.knowledge if c["page_index"] == node.page_index), {})
    excluded = set().union(*(s.excluded_node_ids for s in scenes))
    result.nodes = [n for n in nodes if n.id not in excluded]
    for scene in scenes:
        result.conflicts.extend(apply_assembly_bindings(scene, result.nodes))
    result.assigned_count = _assign_native_tags_to_nodes(
        result.nodes,
        pages_by_index={p.page_index: p for p in pages},
        legend_pack=legend_pack,
        reserved_text_ids=set().union(*(s.reserved_text_ids for s in scenes)),
    )
    return result


def resolve_symbol_text(*, canonical_label, raw_readings, labels, spans, bbox):
    label_keys = {normalise_label(label) for label in labels}
    native_candidates = [
        span for span in _nearby_tag_text(spans, bbox) if normalise_label(span.text) in label_keys
    ]
    label_conflict = len({normalise_label(label) for label in labels if normalise_label(label)}) > 1
    matches = [
        label
        for label in labels
        if any(normalise_label(s.text) == normalise_label(label) for s in native_candidates)
    ]
    selected = canonical_label or (raw_readings[0] if raw_readings else "unlabelled")
    resolved = matches[0] if len({normalise_label(label) for label in matches}) == 1 else None
    return native_candidates, label_conflict, resolved or selected, resolved


def _nearby_tag_text(spans: list[TextEvidence], bbox: BBox) -> list[TextEvidence]:
    padding = max(20, int(max(bbox.w, bbox.h) * 0.75))
    x0 = bbox.x - padding
    y0 = bbox.y - padding
    x1 = bbox.x2 + padding
    y1 = bbox.y2 + padding
    candidates = [
        span
        for span in spans
        if _TAG_RE.search(span.text)
        and span.bbox.x < x1
        and span.bbox.x2 > x0
        and span.bbox.y < y1
        and span.bbox.y2 > y0
    ]
    return sorted(candidates, key=lambda span: _center_distance(span.bbox, bbox))[:8]


def _assign_native_tags_to_nodes(
    nodes: list[ReconciledNode],
    *,
    pages_by_index: dict[int, PageEvidence],
    legend_pack: LegendPack | None,
    reserved_text_ids: set[str] | None = None,
) -> int:
    """Promote native PDF tags only for mutual-best compatible matches.

    The visual model remains responsible for finding the object.  Positioned
    vector text supplies its identity when the tag and object select each
    other as their best local match.  Competing readings are retained as
    ``label_candidates`` for human review instead of being guessed.
    """

    assigned = 0
    nodes_by_page: dict[int, list[ReconciledNode]] = {}
    for node in nodes:
        nodes_by_page.setdefault(node.page_index, []).append(node)

    for page_index, page_nodes in nodes_by_page.items():
        page = pages_by_index.get(page_index)
        if page is None:
            continue
        tags = [
            t for t in _native_entity_tag_spans(page) if t.id not in (reserved_text_ids or set())
        ]
        generic_nodes = [
            node for node in page_nodes if normalise_label(node.label or "") in _GENERIC_LABEL_KEYS
        ]
        if not tags or not generic_nodes:
            continue

        pairs: list[tuple[float, str, str, TextEvidence, ReconciledNode]] = []
        for span in tags:
            # A tag inside an already identified symbol belongs to that
            # instance. Do not offer it again to a nearby unlabelled valve.
            if any(
                normalise_label(node.label) == normalise_label(span.text)
                and _bbox_centres_overlap(node.bbox_global, span.bbox)
                for node in page_nodes
                if normalise_label(node.label or "") not in _GENERIC_LABEL_KEYS
            ):
                continue
            semantics = infer_tag_semantics(span.text, legend_pack)
            for node in generic_nodes:
                score = _native_tag_node_score(span, node, semantics=semantics)
                if score is not None:
                    pairs.append((score, span.id, node.id, span, node))

        by_span: dict[str, list[tuple[float, str, str, TextEvidence, ReconciledNode]]] = {}
        by_node: dict[str, list[tuple[float, str, str, TextEvidence, ReconciledNode]]] = {}
        for pair in pairs:
            by_span.setdefault(pair[1], []).append(pair)
            by_node.setdefault(pair[2], []).append(pair)
        for values in (*by_span.values(), *by_node.values()):
            values.sort(key=lambda value: (value[0], value[1], value[2]))

        for node in generic_nodes:
            candidates = by_node.get(node.id, [])
            if not candidates:
                continue
            labels = _ordered_unique(
                [
                    *(node.attributes.get("label_candidates") or []),
                    *(node.attributes.get("raw_text_candidates") or []),
                    *(pair[3].text for pair in candidates[:8]),
                ]
            )
            if labels:
                node.attributes["label_candidates"] = labels

        used_spans: set[str] = set()
        used_nodes: set[str] = set()
        for pair in sorted(pairs, key=lambda value: (value[0], value[1], value[2])):
            score, span_id, node_id, span, node = pair
            if span_id in used_spans or node_id in used_nodes:
                continue
            if by_span[span_id][0][2] != node_id or by_node[node_id][0][1] != span_id:
                continue
            if score > 3.0:
                continue
            semantics = infer_tag_semantics(span.text, legend_pack)
            node.label = " ".join(span.text.split())
            node.source_quote = node.source_quote or node.label
            node.attributes["canonical_tag"] = node.label
            assignment_key = (
                "native_tag_assignment" if span.origin == "pdf_text" else "raster_tag_assignment"
            )
            node.attributes[assignment_key] = "mutual_best_geometry"
            node.attributes[assignment_key + "_score"] = round(score, 3)
            if span.origin != "pdf_text":
                node.attributes["assigned_text_origin"] = span.origin
            node.attributes["source_text_ids"] = sorted(
                set(node.attributes.get("source_text_ids") or []) | {span.id}
            )
            node.source_evidence_ids = sorted(set(node.source_evidence_ids) | {span.id})
            if semantics is not None:
                for key, value in semantics.attributes.items():
                    if value not in (None, "", [], {}):
                        node.attributes.setdefault(key, value)
                node.attributes.setdefault("tag_prefix", semantics.prefix)
                node.attributes.setdefault("tag_semantics_basis", semantics.basis)
            evidence = dict(node.attributes.get("system_confidence_evidence") or {})
            agreement_key = (
                "native_text_agreement" if span.origin == "pdf_text" else "raster_text_agreement"
            )
            evidence[agreement_key] = True
            node.attributes["system_confidence_evidence"] = evidence
            score_value = min(0.98, float(node.system_confidence or 0.0) + 0.12)
            node.system_confidence = round(score_value, 3)
            node.system_confidence_level = _confidence_level(score_value)
            node.attributes["system_confidence"] = node.system_confidence
            used_spans.add(span_id)
            used_nodes.add(node_id)
            assigned += 1
    return assigned


def _native_entity_tag_spans(page: PageEvidence) -> list[TextEvidence]:
    seen: set[tuple[str, int, int]] = set()
    result: list[TextEvidence] = []
    for span in page.text_spans:
        text = " ".join(span.text.strip().split())
        if not text or _EXPLICIT_OPC_REF_RE.fullmatch(text):
            continue
        match = re.fullmatch(r"(?P<prefix>[A-Z]{1,6})[- ]?(?P<number>\d{1,6}[A-Z]?)", text, re.I)
        area = re.fullmatch(r"\d{3,6}[- ]?[A-Z]{1,4}[- ]?\d{1,5}[A-Z]?", text, re.I)
        if match is None and area is None:
            continue
        if match is not None:
            prefix = match.group("prefix")
            digits = re.sub(r"[^0-9]", "", match.group("number"))
            if prefix.upper() in {"DN", "PN", "SCH", "CL", "NO", "REV", "PAGE"}:
                continue
            if len(digits) == 1 and len(prefix) < 2:
                continue
        key = (normalise_label(text), span.bbox.x // 4, span.bbox.y // 4)
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return sorted(result, key=lambda span: (span.bbox.y, span.bbox.x, span.id))


def _native_tag_node_score(
    span: TextEvidence,
    node: ReconciledNode,
    *,
    semantics: Any,
) -> float | None:
    attrs = node.attributes or {}
    if semantics is not None and node.kind != semantics.expected_kind:
        return None
    if semantics is None:
        prefix_match = re.match(r"[A-Z]+", span.text.strip(), re.I)
        prefix = prefix_match.group(0).upper() if prefix_match else ""
        # Unknown *V tags may label a visually detected valve.  Other unknown
        # prefixes remain review evidence until a project legend defines them.
        if not (
            prefix.endswith("V") and node.kind == "equipment" and bool(attrs.get("valve_type"))
        ):
            return None

    text_scale = max(span.bbox.h, 8)
    if attrs.get("valve_type") and (
        node.bbox_global.w > max(220, text_scale * 10)
        or node.bbox_global.h > max(220, text_scale * 10)
    ):
        return None
    sx = span.bbox.x + span.bbox.w / 2
    sy = span.bbox.y + span.bbox.h / 2
    nx = node.bbox_global.x + node.bbox_global.w / 2
    ny = node.bbox_global.y + node.bbox_global.h / 2
    x_scale = max(span.bbox.w, node.bbox_global.w, text_scale * 2)
    y_scale = max(span.bbox.h, node.bbox_global.h, text_scale * 2)
    dx = abs(nx - sx) / x_scale
    dy = abs(ny - sy) / y_scale
    score = math.hypot(dx, dy)
    if score > 3.6:
        return None
    if span.bbox.iou(node.bbox_global) > 0:
        score -= 0.2
    return max(0.0, score)


def _bbox_centres_overlap(left: BBox, right: BBox) -> bool:
    center_x = right.x + right.w / 2
    center_y = right.y + right.h / 2
    return left.x <= center_x <= left.x2 and left.y <= center_y <= left.y2


def _center_distance(left: BBox, right: BBox) -> float:
    left_center = (left.x + left.w / 2, left.y + left.h / 2)
    right_center = (right.x + right.w / 2, right.y + right.h / 2)
    return math.dist(left_center, right_center)


def _ordered_unique(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        rendered = str(value or "").strip()
        if rendered and rendered not in out:
            out.append(rendered)
    return out


def _confidence_level(score: float) -> Confidence:
    if score >= 0.78:
        return "high"
    if score >= 0.52:
        return "medium"
    return "low"
