"""Deterministic native evidence ownership and logical equipment scopes.

A boundary is not equipment, containment is not identity, and membership is not
connectivity. Only corroborated, captioned scopes become logical assemblies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from diagex.vision.evidence import PageEvidence, TextEvidence, stable_evidence_id
from diagex.vision.models import BBox, EquipmentAssembly, ReconciledNode
from diagex.vision.native_text import infer_tag_semantics, normalise_text_key
from diagex.vision.vector_geometry import path_vertices, segment_key, symbol_contours

_SCOPE_NOTE = re.compile(r"supplier|vendor|package|skid|battery.limit|供货|厂家|成套|界区", re.I)
_TAG = re.compile(r"(?:\d{2,6}[- ])?[A-Z]{1,6}[- ]\d{1,6}[A-Z]?(?:/[A-Z])?", re.I)


@dataclass
class Boundary:
    bbox: BBox
    segments: set[tuple]


@dataclass
class NativeHierarchy:
    assemblies: list[EquipmentAssembly] = field(default_factory=list)
    text_bindings: list[dict] = field(default_factory=list)
    excluded_node_ids: set[str] = field(default_factory=set)

    @property
    def reserved_text_ids(self) -> set[str]:
        return {sid for b in self.text_bindings for sid in b["source_text_ids"]}


def contains(outer: BBox, inner: BBox, tolerance: float = 0) -> bool:
    return (
        inner.x >= outer.x - tolerance
        and inner.y >= outer.y - tolerance
        and inner.x2 <= outer.x2 + tolerance
        and inner.y2 <= outer.y2 + tolerance
    )


def boundary_candidates(page: PageEvidence) -> list[Boundary]:
    """Recover four-sided patterned enclosures; never infer a missing side.

    The input may be native dash metadata or individually drawn dash fragments.
    Continuous solid piping loops, page borders and small symbol boxes fail the
    pattern/size gates. Source segments remain individually addressable.
    """
    rows = []
    for path in page.paths:
        if path.origin != "pdf_vector":
            continue
        points = path_vertices(path)
        if path.closed or path.primitive in {"rect", "quad"}:
            points += points[:1]
        for a, b in zip(points, points[1:], strict=False):
            if abs(a[1] - b[1]) <= 1 and abs(a[0] - b[0]) > 2:
                rows.append(
                    (
                        0,
                        (a[1] + b[1]) / 2,
                        min(a[0], b[0]),
                        max(a[0], b[0]),
                        segment_key(path.id, a, b),
                        path.visual_style,
                    )
                )
            elif abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) > 2:
                rows.append(
                    (
                        1,
                        (a[0] + b[0]) / 2,
                        min(a[1], b[1]),
                        max(a[1], b[1]),
                        segment_key(path.id, a, b),
                        path.visual_style,
                    )
                )
    lines: list[list[tuple]] = []
    for row in sorted(rows):
        if not lines or row[0] != lines[-1][0][0] or row[1] - lines[-1][0][1] > 2:
            lines.append([])
        lines[-1].append(row)
    runs = []
    gap_limit = max(5, min(page.width, page.height) * 0.015)
    for line in lines:
        chunks: list[list[tuple]] = []
        end = -float("inf")
        for row in sorted(line, key=lambda r: r[2]):
            if row[2] - end > gap_limit:
                chunks.append([])
            chunks[-1].append(row)
            end = max(end, row[3])
        for chunk in chunks:
            lo, hi = min(r[2] for r in chunk), max(r[3] for r in chunk)
            gaps = sum(b[2] - a[3] > 2 for a, b in zip(chunk, chunk[1:], strict=False))
            patterned = gaps >= 2 or any(r[5] in {"dash_dot", "dashed", "dotted"} for r in chunk)
            if hi - lo >= min(page.width, page.height) * 0.1:
                runs.append((chunk[0][0], chunk[0][1], lo, hi, patterned, {r[4] for r in chunk}))
    horizontal = [r for r in runs if r[0] == 0]
    vertical = [r for r in runs if r[0] == 1]
    out = {}
    for top in horizontal:
        for bottom in horizontal:
            if bottom[1] - top[1] < min(page.width, page.height) * 0.1:
                continue
            if abs(top[2] - bottom[2]) > 4 or abs(top[3] - bottom[3]) > 4:
                continue
            sides = []
            for x in (top[2], top[3]):
                candidates = [
                    r
                    for r in vertical
                    if abs(r[1] - x) <= 4 and abs(r[2] - top[1]) <= 4 and abs(r[3] - bottom[1]) <= 4
                ]
                if len(candidates) == 1:
                    sides.append(candidates[0])
            if len(sides) != 2 or sum(r[4] for r in [top, bottom, *sides]) < 2:
                continue
            box = BBox(
                x=round(top[2]),
                y=round(top[1]),
                w=round(top[3] - top[2]),
                h=round(bottom[1] - top[1]),
            )
            if box.w > page.width * 0.85 or box.h > page.height * 0.85:
                continue
            segments = set().union(*(r[5] for r in [top, bottom, *sides]))
            out[(box.x, box.y, box.w, box.h)] = Boundary(box, segments)
    return sorted(out.values(), key=lambda b: (b.bbox.w * b.bbox.h, b.bbox.y, b.bbox.x))


def _equipment_caption(text: str) -> bool:
    if not _TAG.fullmatch(text.strip()):
        return False
    semantics = infer_tag_semantics(text)
    return semantics is None or semantics.expected_kind == "equipment"


def _underline(page: PageEvidence, span: TextEvidence) -> set[tuple]:
    # PDF exporters can emit one underline fragment per character.
    box = span.bbox
    rows = []
    for p in page.paths:
        if p.origin != "pdf_vector":
            continue
        for a, b in zip(p.points, p.points[1:], strict=False):
            lo, hi = sorted((a[0], b[0]))
            if (
                abs(a[1] - b[1]) <= 1
                and box.y2 - box.h * 0.2 <= a[1] <= box.y2 + box.h * 0.35
                and box.x - box.h * 0.3 <= lo < hi <= box.x2 + box.h * 0.3
            ):
                rows.append((lo, hi, a[1], segment_key(p.id, a, b)))
    rows.sort()
    if (
        not rows
        or rows[0][0] > box.x + box.h * 0.3
        or max(r[1] for r in rows) < box.x2 - box.h * 0.3
    ):
        return set()
    if max(r[2] for r in rows) - min(r[2] for r in rows) > 2:
        return set()
    end = rows[0][1]
    for row in rows[1:]:
        if row[0] - end > 3:
            return set()
        end = max(end, row[1])
    return {r[3] for r in rows}


def _has_leader(page: PageEvidence, span: TextEvidence, nodes: list[ReconciledNode]) -> bool:
    # A direct native leader gives local evidence precedence over a region.
    b = span.bbox
    pad = max(3, b.h * 0.4)
    for p in page.paths:
        if p.origin != "pdf_vector" or p.closed or len(p.points) < 2:
            continue
        for a, z in [(p.points[0], p.points[-1]), (p.points[-1], p.points[0])]:
            if not (b.x - pad <= a[0] <= b.x2 + pad and b.y - pad <= a[1] <= b.y2):
                continue
            if any(
                n.bbox_global.x <= z[0] <= n.bbox_global.x2
                and n.bbox_global.y <= z[1] <= n.bbox_global.y2
                for n in nodes
            ):
                return True
    return False


def _same_family_mention(heading: str, caption: str) -> bool:
    # A/B denotes alternatives, never an instruction to merge their instances.
    if "/" in heading:
        first, suffix = heading.rsplit("/", 1)
        return normalise_text_key(caption) in {
            normalise_text_key(first),
            normalise_text_key(first[:-1] + suffix),
        }
    return normalise_text_key(heading) == normalise_text_key(caption)


def build_native_hierarchy(page: PageEvidence, nodes: list[ReconciledNode]) -> NativeHierarchy:
    local = [n for n in nodes if n.page_index == page.page_index and n.kind != "opc"]
    contours = symbol_contours(page, local)
    boundaries = boundary_candidates(page)
    captions = [t for t in page.text_spans if _equipment_caption(t.text)]
    notes = [t for t in page.text_spans if _SCOPE_NOTE.search(t.text)]
    # A specification heading has an adjacent description, outside every scope
    # and physical symbol. It is a mention even if no owner can yet be resolved.
    headings = []
    for t in captions:
        if any(contains(b.bbox, t.bbox) for b in boundaries) or any(
            contains(c.bbox, t.bbox, 2) for c in contours.values()
        ):
            continue
        descriptions = [
            s
            for s in page.text_spans
            if s.id != t.id
            and not _equipment_caption(s.text)
            and re.search(r"[A-Za-z\u4e00-\u9fff]", s.text)
            and t.bbox.y2 <= s.bbox.y <= t.bbox.y2 + t.bbox.h * 1.6
            and abs(s.bbox.x + s.bbox.w / 2 - t.bbox.x - t.bbox.w / 2) < max(t.bbox.w, s.bbox.w)
        ]
        if descriptions and _underline(page, t):
            headings.append(t)
    result = NativeHierarchy()
    reserved: set[str] = set()
    for boundary in boundaries:
        members = [
            n
            for n in local
            if contains(boundary.bbox, n.bbox_global)
            and n.id in contours
            and n.bbox_global.iou(boundary.bbox) < 0.5
        ]
        if len(members) < 2 or not notes:
            continue
        labels = [
            t
            for t in captions
            if t.id not in reserved
            and (
                contains(boundary.bbox, t.bbox)
                or (
                    boundary.bbox.y2 <= t.bbox.y <= boundary.bbox.y2 + t.bbox.h * 4
                    and boundary.bbox.x <= t.bbox.x
                    and t.bbox.x2 <= boundary.bbox.x2
                )
            )
            and _underline(page, t)
            and not any(contains(c.bbox, t.bbox, 2) for c in contours.values())
            and not _has_leader(page, t, members)
        ]
        if not labels:
            continue
        aid = stable_evidence_id("assembly", page.page_index, sorted(boundary.segments))
        # Prefer a specification above and horizontally aligned with this scope.
        aligned = [
            t
            for t in headings
            if t.bbox.y2 < boundary.bbox.y
            and boundary.bbox.x <= t.bbox.x + t.bbox.w / 2 <= boundary.bbox.x2
        ]
        heading = (
            min(
                aligned,
                key=lambda t: (
                    boundary.bbox.y - t.bbox.y2,
                    abs(t.bbox.x + t.bbox.w / 2 - boundary.bbox.x - boundary.bbox.w / 2),
                    t.id,
                ),
            )
            if aligned
            else None
        )
        names = list(dict.fromkeys(t.text.strip() for t in labels))
        # Several captions may name repeated subassemblies within a shared
        # supplier scope. Their different names do not establish a contradiction.
        conflict = (
            len(labels) == 1
            and heading is not None
            and not all(_same_family_mention(heading.text, name) for name in names)
        )
        if heading is not None and (conflict or len(labels) > 1):
            names.append(heading.text.strip())
        names = list(dict.fromkeys(names))
        status = "conflicting" if conflict else "supported" if len(labels) == 1 else "uncertain"
        source_ids = [t.id for t in labels] + ([heading.id] if heading else [])
        assembly = EquipmentAssembly(
            id=aid,
            page_index=page.page_index,
            bbox_global=boundary.bbox,
            label=names[0] if status == "supported" else None,
            label_candidates=names,
            member_node_ids=sorted(n.id for n in members),
            status=status,
            source_text_ids=source_ids,
            boundary_segments=sorted(boundary.segments),
            evidence={
                "basis": "patterned_enclosure_caption_and_scope_note",
                "note_text_ids": [t.id for t in notes],
                "caption_text_ids": [t.id for t in labels],
                "heading_text_ids": [heading.id] if heading else [],
            },
        )
        result.assemblies.append(assembly)
        for t in labels:
            reserved.add(t.id)
            result.text_bindings.append(
                {
                    "page_index": page.page_index,
                    "text": t.text,
                    "source_text_ids": [t.id],
                    "role": "assembly_label",
                    "assembly_ids": [aid],
                    "status": status,
                    "owned_segments": sorted(_underline(page, t)),
                }
            )
    # A scope-like caption with insufficient enclosure evidence must not fall
    # back to the nearest physical symbol. Keep one text-ownership candidate.
    if notes:
        for t in captions:
            if (
                t.id in reserved
                or t in headings
                or not _underline(page, t)
                or any(contains(c.bbox, t.bbox, 2) for c in contours.values())
                or _has_leader(page, t, local)
            ):
                continue
            result.text_bindings.append(
                {
                    "page_index": page.page_index,
                    "text": t.text,
                    "source_text_ids": [t.id],
                    "role": "unresolved_scope",
                    "assembly_ids": [],
                    "status": "uncertain",
                    "bbox_global": t.bbox.model_dump(),
                    "owned_segments": sorted(_underline(page, t)),
                }
            )
    for assembly in result.assemblies:
        parents = [
            a
            for a in result.assemblies
            if a.id != assembly.id and contains(a.bbox_global, assembly.bbox_global)
        ]
        if parents:
            assembly.parent_assembly_id = min(
                parents, key=lambda a: a.bbox_global.w * a.bbox_global.h
            ).id
    for t in headings:
        owners = [a.id for a in result.assemblies if t.id in a.source_text_ids]
        physical = [
            n.id
            for n in local
            if n.id in contours and normalise_text_key(n.label) == normalise_text_key(t.text)
        ]
        result.text_bindings.append(
            {
                "page_index": page.page_index,
                "text": t.text,
                "source_text_ids": [t.id],
                "role": "specification_reference",
                "assembly_ids": owners,
                "node_ids": [] if owners else physical,
                "status": "supported" if owners or physical else "uncertain",
                "owned_segments": sorted(_underline(page, t)),
            }
        )
        for n in local:
            if n.id not in contours and n.bbox_global.iou(t.bbox) >= 0.6:
                result.excluded_node_ids.add(n.id)
    return result


def _withhold_label(node: ReconciledNode, names: set[str], reason: str) -> None:
    node.attributes[reason] = node.label
    if normalise_text_key(node.source_quote) in names:
        node.source_quote = None
    node.label = "unlabelled"
    node.attributes["label_candidates"] = [
        value
        for value in node.attributes.get("label_candidates", [])
        if normalise_text_key(value) not in names
    ]
    for key in ("canonical_tag", "native_tag_assignment", "native_tag_assignment_score"):
        node.attributes.pop(key, None)
    if node.attributes.get("system_confidence_evidence", {}).get("native_text_agreement"):
        node.attributes["system_confidence_evidence"]["native_text_agreement"] = False
        score = round(max(0, (node.system_confidence or 0) - 0.12), 3)
        node.system_confidence = score
        node.system_confidence_level = (
            "high" if score >= 0.78 else "medium" if score >= 0.52 else "low"
        )
        node.attributes["system_confidence"] = score


def apply_assembly_bindings(scene: NativeHierarchy, nodes: list[ReconciledNode]) -> list[dict]:
    """Withhold inherited assembly labels without rewriting physical classes."""
    conflicts = []
    for assembly in scene.assemblies:
        members = set(assembly.member_node_ids)
        names = {normalise_text_key(t) for t in assembly.label_candidates}
        removed = []
        for n in nodes:
            if n.id not in members:
                continue
            n.attributes["assembly_ids"] = sorted(
                set(n.attributes.get("assembly_ids", [])) | {assembly.id}
            )
            if normalise_text_key(n.label) in names:
                removed.append({"node_id": n.id, "label": n.label})
                _withhold_label(n, names, "withheld_assembly_label")
        if removed:
            assembly.evidence["previous_physical_label_owners"] = removed
        if assembly.status != "supported":
            conflicts.append(
                {
                    "type": "assembly_identity_uncertainty",
                    "page_index": assembly.page_index,
                    "assembly_id": assembly.id,
                    "node_ids": assembly.member_node_ids,
                    "labels": assembly.label_candidates,
                    "source_text_ids": assembly.source_text_ids,
                    "status": "unresolved",
                    "reason": "Printed labels or assembly scope do not establish one identity",
                }
            )
    for binding in scene.text_bindings:
        if binding["role"] != "unresolved_scope":
            continue
        for n in nodes:
            refs = set(n.source_evidence_ids) | set(n.attributes.get("source_text_ids", []))
            if (
                n.page_index == binding["page_index"]
                and refs.intersection(binding["source_text_ids"])
                and normalise_text_key(n.label) == normalise_text_key(binding["text"])
            ):
                _withhold_label(n, {normalise_text_key(binding["text"])}, "withheld_scope_label")
    return conflicts
