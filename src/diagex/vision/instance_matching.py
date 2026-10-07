"""Conservative, deterministic matching of physical symbol observations.

Native text is identity evidence, never a spatial merge instruction. Native
paths can corroborate a symbol footprint or a closed outline across crops;
an arbitrary connected pipe network is deliberately not a symbol footprint.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, ReconciledNode
from diagex.vision.perception import DetectionRecord
from diagex.vision.reconcile import normalise_label
from diagex.vision.vector_geometry import symbol_contours

FUSION_VERSION = "4.0.1"
FUSION_DEPENDENT_STAGES = (
    "contextual",
    "topology",
    "line_evidence",
    "page_graph",
    "assembly",
    "export",
)
_GENERIC = {"", "unlabelled", "unlabeled", "unknown", "none", "na"}
_CONFIDENCE = {"low": 0, "medium": 1, "high": 2}


@dataclass
class InstanceCluster:
    detections: list[DetectionRecord] = field(default_factory=list)
    bbox: BBox | None = None
    representative_id: str | None = None
    path_ids: list[str] = field(default_factory=list)
    geometry_basis: str = "representative_observation"
    symbol_id: str | None = None


def _tag(item: DetectionRecord) -> str:
    label = _identity_key(item.label or "")
    return "" if label.lower() in _GENERIC else label


def _identity_key(text: str) -> str:
    return re.sub(r"[-_]", "", normalise_label(text))


def _family(item: DetectionRecord) -> str:
    attrs = item.attributes
    if attrs.get("valve_type") or attrs.get("equipment_class") == "valve":
        return "valve"
    return str(attrs.get("equipment_class") or "")


def _inside(inner: BBox, outer: BBox, padding: float = 0) -> bool:
    return (
        inner.x >= outer.x - padding
        and inner.y >= outer.y - padding
        and inner.x2 <= outer.x2 + padding
        and inner.y2 <= outer.y2 + padding
    )


def _union(boxes: list[BBox]) -> BBox:
    x, y = min(b.x for b in boxes), min(b.y for b in boxes)
    return BBox(x=x, y=y, w=max(b.x2 for b in boxes) - x, h=max(b.y2 for b in boxes) - y)


def _native_identity(item: DetectionRecord, page: PageEvidence) -> tuple[str, frozenset[str]]:
    # Only an unambiguous, locally observed identity can veto a match. A tag
    # merely near a large vessel is not assigned to every object around it.
    spans = [
        span
        for span in page.text_spans
        if _inside(span.bbox, item.bbox)
        and (not _tag(item) or _identity_key(span.text) == _tag(item))
    ]
    tags = {
        _identity_key(s.text)
        for s in spans
        if any(c.isalpha() for c in s.text)
        and any(c.isdigit() for c in s.text)
        and not re.match(r"^(?:DN|PN|SCH|CL)\s*\d", s.text, re.I)
    }
    if len(tags) != 1:
        return "", frozenset()
    tag = next(iter(tags))
    return tag, frozenset(s.id for s in spans if _identity_key(s.text) == tag)


def _footprint(item: DetectionRecord, page: PageEvidence) -> frozenset[str]:
    box = item.bbox
    padding = max(2.0, min(box.w, box.h) * 0.08)
    paths = [p for p in page.paths if p.origin == "pdf_vector" and _inside(p.bbox, box, padding)]
    if not paths:
        return frozenset()
    extent = _union([p.bbox for p in paths])
    # A single pipe running through a box does not corroborate a symbol.
    if extent.w < box.w * 0.35 or extent.h < box.h * 0.35:
        return frozenset()
    if len(paths) < 2 and not (paths[0].closed or paths[0].primitive in {"rect", "quad"}):
        return frozenset()
    return frozenset(p.id for p in paths)


def _complementary_crops(a: BBox, b: BBox) -> bool:
    # A small box inside a contour is not a clipped observation of that contour.
    # Crops must cover comparable cross sections and extend each other.
    if _inside(a, b) or _inside(b, a):
        return False
    overlap_x = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    overlap_y = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return overlap_x >= 0.8 * max(a.w, b.w) or overlap_y >= 0.8 * max(a.h, b.h)


def _needs_instance_review(a: DetectionRecord, b: DetectionRecord) -> bool:
    if a.page_index != b.page_index:
        return False
    if a.bbox.iou(b.bbox) >= 0.18:
        return True
    if not (
        _tag(a)
        and _tag(a) == _tag(b)
        and _family(a) == _family(b)
        and _complementary_crops(a.bbox, b.bbox)
    ):
        return False
    gap_x = max(0, a.bbox.x - b.bbox.x2, b.bbox.x - a.bbox.x2)
    gap_y = max(0, a.bbox.y - b.bbox.y2, b.bbox.y - a.bbox.y2)
    return math.hypot(gap_x, gap_y) <= max(3, min(a.bbox.w, a.bbox.h, b.bbox.w, b.bbox.h) * 0.1)


def match_instances(
    detections: list[DetectionRecord],
    pages: dict[int, PageEvidence],
) -> tuple[list[InstanceCluster], list[dict]]:
    symbols = {d.id: None for d in detections}
    for index, page in pages.items():
        observations = [
            ReconciledNode(
                id=d.id,
                kind=d.kind,
                label=d.label,
                bbox_global=d.bbox,
                page_index=d.page_index,
                confidence=d.confidence,
                attributes=d.attributes,
            )
            for d in detections
            if d.page_index == index
        ]
        symbols.update(symbol_contours(page, observations))
    ordered = sorted(detections, key=lambda d: (d.page_index, d.bbox.y, d.bbox.x, d.id))
    footprints = {d.id: _footprint(d, pages[d.page_index]) for d in ordered}
    identities = {d.id: _native_identity(d, pages[d.page_index]) for d in ordered}
    pair_cache: dict[tuple[str, str], tuple[bool, str, list[str]]] = {}

    def match(a: DetectionRecord, b: DetectionRecord) -> tuple[bool, str, list[str]]:
        key = tuple(sorted((a.id, b.id)))
        if key in pair_cache:
            return pair_cache[key]
        result = decide(a, b)
        pair_cache[key] = result
        return result

    def decide(a: DetectionRecord, b: DetectionRecord) -> tuple[bool, str, list[str]]:
        if a.page_index != b.page_index or a.kind != b.kind:
            return False, "different_object_kinds", []
        if _family(a) and _family(b) and _family(a) != _family(b):
            return False, "different_equipment_families", []
        if _tag(a) and _tag(b) and _tag(a) != _tag(b):
            return False, "different_printed_identities", []
        ta, ia = identities[a.id]
        tb, ib = identities[b.id]
        if ta and tb and (ta != tb or not ia.intersection(ib)):
            return False, "distinct_native_text_instances", []
        sa, sb = symbols[a.id], symbols[b.id]
        if sa and sb:
            if sa.symbol_id == sb.symbol_id:
                return True, "shared_native_symbol", sa.path_ids
            if (
                sa.basis == sb.basis == "native_contour"
                and not set(sa.path_ids).intersection(sb.path_ids)
                and sa.bbox.iou(sb.bbox) < 0.2
            ):
                return False, "different_native_symbols", []
        x_ratio = min(a.bbox.w, b.bbox.w) / max(a.bbox.w, b.bbox.w, 1)
        y_ratio = min(a.bbox.h, b.bbox.h) / max(a.bbox.h, b.bbox.h, 1)
        shared = footprints[a.id] & footprints[b.id]
        total = footprints[a.id] | footprints[b.id]
        comparable = min(x_ratio, y_ratio) >= 0.65
        if comparable and a.bbox.iou(b.bbox) >= 0.55:
            return True, "overlapping_symbol_observations", sorted(shared)
        if comparable and total and len(shared) / len(total) >= 0.75:
            return True, "shared_vector_footprint", sorted(shared)
        return False, "insufficient_symbol_geometry", []

    clusters: list[InstanceCluster] = []
    for detection in ordered:
        candidates = [c for c in clusters if all(match(detection, m)[0] for m in c.detections)]
        # Two plausible clusters are not permission to join them transitively.
        if len(candidates) == 1:
            candidates[0].detections.append(detection)
        else:
            clusters.append(InstanceCluster([detection]))

    conflicts: list[dict] = []
    for index, cluster in enumerate(clusters):
        representative = max(
            cluster.detections,
            key=lambda d: (
                bool(footprints[d.id]),
                _CONFIDENCE[d.confidence],
                d.bbox.w * d.bbox.h,
                d.id,
            ),
        )
        cluster.bbox = representative.bbox.model_copy()
        cluster.representative_id = representative.id
        cluster.path_ids = sorted(footprints[representative.id])
        supported = [symbols[d.id] for d in cluster.detections if symbols[d.id]]
        if supported and len({symbol.symbol_id for symbol in supported}) == 1:
            symbol = supported[0]
            cluster.bbox = symbol.bbox.model_copy()
            cluster.path_ids = symbol.path_ids
            cluster.geometry_basis = "native_symbol"
            cluster.symbol_id = symbol.symbol_id
        for other in clusters[index + 1 :]:
            overlapping = [
                (a, b)
                for a in cluster.detections
                for b in other.detections
                if _needs_instance_review(a, b)
                or (
                    symbols[a.id]
                    and symbols[b.id]
                    and symbols[a.id].symbol_id == symbols[b.id].symbol_id
                )
            ]
            if not overlapping:
                continue
            a, b = max(overlapping, key=lambda pair: pair[0].bbox.iou(pair[1].bbox))
            conflicts.append(
                {
                    "type": "fusion_instance_uncertainty",
                    "status": "unresolved",
                    "page_index": a.page_index,
                    "detection_ids": [a.id, b.id],
                    "reason": match(a, b)[1]
                    if not match(a, b)[0]
                    else "incompatible_cluster_members",
                    "labels": [a.label, b.label],
                }
            )
    return clusters, conflicts
