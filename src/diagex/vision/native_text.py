"""Conservative native-PDF tag semantics and completeness inventory.

This module deliberately separates engineering tags from other positioned PDF
text.  It is deterministic: project legend entries may strengthen a standard
tag interpretation, but an unknown abbreviation never becomes a graph kind by
guesswork.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from diagex.vision.evidence import PageEvidence, TextEvidence
from diagex.vision.legend_models import LegendPack
from diagex.vision.models import BBox, ReconciledGraph, ReconciledNode

CandidateKind = Literal[
    "instrument_tag",
    "equipment_tag",
    "unknown_tag",
    "line_number",
    "drawing_reference",
    "dimension",
]
InventoryStatus = Literal["assigned", "unresolved", "ambiguous", "excluded"]

_DRAWING_REFERENCE_RE = re.compile(r"^(?:DW|DWG|SH|SHT)[A-Z0-9]*[- ]?\d", re.IGNORECASE)
_LINE_NUMBER_RE = re.compile(
    r"^(?:\d{1,3}[\"”']?[- ])?[A-Z]{1,5}-\d{3,}(?:-[A-Z0-9]+){1,}$",
    re.IGNORECASE,
)
_DIMENSION_RE = re.compile(
    r"^(?:DN\s*\d+(?:\s*[x×/]\s*\d+)?|PN\s*\d+|SCH\s*[A-Z0-9.]+|"
    r"\d+(?:\.\d+)?\s*(?:[\"”′]|mm|cm))$",
    re.IGNORECASE,
)
_AREA_EQUIPMENT_TAG_RE = re.compile(
    r"^\d{3,6}[- ]?(?P<prefix>[A-Z]{1,4})[- ]?(?P<number>\d{1,5}[A-Z]?)$",
    re.IGNORECASE,
)
_SIMPLE_TAG_RE = re.compile(
    r"^(?P<prefix>[A-Z]{1,6})[- ]?(?P<number>\d{1,6}[A-Z]?)$",
    re.IGNORECASE,
)
_EXCLUDED_PREFIXES = {"DN", "PN", "SCH", "CL", "NO", "REV", "PAGE"}

# These are tag functions, not a complete plant taxonomy.  A project legend
# can add an exact prefix.  Single-letter equipment abbreviations are omitted
# because P-001/V-001/M-001 are plant-dependent and visually ambiguous.
_STANDARD_INSTRUMENT_CODES = {
    "AI",
    "AIT",
    "AT",
    "DI",
    "DT",
    "FE",
    "FI",
    "FIC",
    "FIT",
    "FQ",
    "FT",
    "FV",
    "LG",
    "LI",
    "LIC",
    "LIT",
    "LS",
    "LT",
    "LV",
    "PI",
    "PIC",
    "PIT",
    "PS",
    "PT",
    "PV",
    "SI",
    "ST",
    "TE",
    "TI",
    "TIC",
    "TIT",
    "TS",
    "TT",
    "TV",
    "XI",
    "XT",
    "YL",
    "ZI",
    "ZT",
}
_VARIABLES = {
    "P": "pressure",
    "T": "temperature",
    "F": "flow",
    "L": "level",
    "A": "analysis",
}
_FINAL_ELEMENT_CODES = {"FV", "LV", "PV", "TV"}


class TagSemantics(BaseModel):
    tag: str
    prefix: str
    loop_number: str
    expected_kind: Literal["equipment", "instrument"]
    attributes: dict[str, Any] = Field(default_factory=dict)
    basis: Literal["project_legend", "built_in_legend", "standard_tag", "area_equipment_tag"]
    legend_labels: list[str] = Field(default_factory=list)


class NativeTextInventoryItem(BaseModel):
    id: str
    source_ids: list[str] = Field(default_factory=list)
    text: str
    normalised_text: str
    page_index: int
    bbox_global: BBox
    candidate_kind: CandidateKind
    blocking: bool
    status: InventoryStatus
    expected_kind: Literal["equipment", "instrument"] | None = None
    matched_node_ids: list[str] = Field(default_factory=list)
    matched_assembly_ids: list[str] = Field(default_factory=list)
    ownership_scope: Literal[
        "physical_symbol", "assembly_label", "specification_reference", "unresolved_scope"
    ] = "physical_symbol"
    match_method: Literal["source_reference", "exact_label"] | None = None
    reason: str
    tag_semantics: TagSemantics | None = None


class NativeTextInventory(BaseModel):
    schema_version: str = "1.0.0"
    items: list[NativeTextInventoryItem] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


def normalise_text_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def infer_tag_semantics(text: str, legend_pack: LegendPack | None = None) -> TagSemantics | None:
    """Return kind/function semantics only for an unambiguous printed tag."""
    compact = " ".join(str(text or "").strip().split())
    area = _AREA_EQUIPMENT_TAG_RE.fullmatch(compact)
    if area is not None:
        return TagSemantics(
            tag=compact,
            prefix=area.group("prefix").upper(),
            loop_number=area.group("number").upper(),
            expected_kind="equipment",
            attributes={"canonical_tag": compact},
            basis="area_equipment_tag",
        )

    match = _SIMPLE_TAG_RE.fullmatch(compact)
    if match is None:
        return None
    prefix = match.group("prefix").upper()
    # Single-digit identifiers such as QV8 are common on packaged-equipment
    # P&IDs.  Continue to exclude ambiguous single-letter codes such as A1.
    if len(re.sub(r"[^0-9]", "", match.group("number"))) == 1 and len(prefix) < 2:
        return None
    if prefix in _EXCLUDED_PREFIXES:
        return None

    legend_matches = []
    if legend_pack is not None:
        legend_matches = [
            entry
            for entry in legend_pack.entries
            if normalise_text_key(entry.label) == prefix
            and entry.kind in {"instrument", "equipment", "valve"}
        ]
    project_matches = [entry for entry in legend_matches if entry.source != "built_in"]
    selected_matches = project_matches or legend_matches
    kinds = {
        (
            "equipment"
            if entry.kind in {"equipment", "valve"}
            or entry.attributes.get("valve_type")
            or entry.attributes.get("instrument_function") == "valve_actuator"
            else "instrument"
        )
        for entry in selected_matches
    }
    if len(kinds) > 1:
        return None
    if selected_matches:
        expected_kind = next(iter(kinds))
        attrs: dict[str, Any] = {"canonical_tag": compact}
        for entry in selected_matches:
            for key, value in entry.attributes.items():
                if value not in (None, ""):
                    attrs.setdefault(key, value)
        if expected_kind == "equipment" and attrs.get("instrument_function") == "valve_actuator":
            attrs.setdefault("valve_type", "control")
        attrs.setdefault("loop_number", match.group("number").upper())
        return TagSemantics(
            tag=compact,
            prefix=prefix,
            loop_number=match.group("number").upper(),
            expected_kind=expected_kind,
            attributes=attrs,
            basis="project_legend" if project_matches else "built_in_legend",
            legend_labels=sorted({entry.label for entry in selected_matches}),
        )

    if prefix not in _STANDARD_INSTRUMENT_CODES:
        return None
    expected_kind: Literal["equipment", "instrument"] = (
        "equipment" if prefix in _FINAL_ELEMENT_CODES else "instrument"
    )
    attrs = {
        "canonical_tag": compact,
        "loop_number": match.group("number").upper(),
        "instrument_function": _instrument_function(prefix),
        "measured_variable": _VARIABLES.get(prefix[0], "other"),
    }
    if expected_kind == "equipment":
        attrs["valve_type"] = "control"
    return TagSemantics(
        tag=compact,
        prefix=prefix,
        loop_number=match.group("number").upper(),
        expected_kind=expected_kind,
        attributes=attrs,
        basis="standard_tag",
    )


def build_native_text_inventory(
    *,
    pages: list[PageEvidence],
    graph: ReconciledGraph,
    legend_pack: LegendPack | None = None,
) -> NativeTextInventory:
    nodes_by_page: dict[int, list[ReconciledNode]] = {}
    for node in graph.nodes:
        nodes_by_page.setdefault(node.page_index, []).append(node)

    items: list[NativeTextInventoryItem] = []
    seen: set[tuple[int, str, int, int]] = set()
    for page in pages:
        if page.role not in {"pid", "other"}:
            continue
        for group in _candidate_span_groups(page.text_spans):
            if _group_is_in_title_block(group, page):
                continue
            text = " ".join(span.text.strip() for span in group)
            classified = _classify_candidate(text, legend_pack)
            if classified is None:
                continue
            candidate_kind, blocking, semantics = classified
            bbox = _bbox_union(group)
            key = normalise_text_key(text)
            dedupe = (page.page_index, key, bbox.x // 8, bbox.y // 8)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            source_ids = [span.id for span in group]
            referenced: list[str] = []
            exact: list[str] = []
            for node in nodes_by_page.get(page.page_index, []):
                attrs = node.attributes or {}
                node_sources = {
                    *node.source_evidence_ids,
                    *node.source_annotation_ids,
                    *(attrs.get("source_text_ids") or []),
                }
                if node_sources.intersection(source_ids):
                    referenced.append(node.id)
                elif key in _node_text_keys(node):
                    exact.append(node.id)
            matched = sorted(set(referenced or exact))
            status, reason = _inventory_status(
                candidate_kind=candidate_kind,
                matched=matched,
                nodes=nodes_by_page.get(page.page_index, []),
                semantics=semantics,
            )
            bindings = [
                b
                for b in graph.text_bindings
                if b.get("page_index") == page.page_index
                and set(source_ids).intersection(b.get("source_text_ids", []))
            ]
            assembly_ids = sorted({aid for b in bindings for aid in b.get("assembly_ids", [])})
            scope = bindings[0]["role"] if bindings else "physical_symbol"
            if bindings:
                matched = sorted({nid for b in bindings for nid in b.get("node_ids", [])})
                owners = [a for a in graph.assemblies if a.id in assembly_ids]
                status = (
                    "ambiguous"
                    if any(a.status != "supported" for a in owners)
                    else "assigned"
                    if owners or matched
                    else "unresolved"
                )
                reason = (
                    "Native text ownership is recorded at the " + scope.replace("_", " ") + " level"
                )
                # The assembly decision owns its caption and specification mentions.
                # Do not ask for the same identity again as a physical-symbol tag.
                if owners:
                    blocking = False
            item_id = (
                "nt-"
                + hashlib.sha256(
                    repr((page.page_index, source_ids, key)).encode("utf-8")
                ).hexdigest()[:12]
            )
            items.append(
                NativeTextInventoryItem(
                    id=item_id,
                    source_ids=source_ids,
                    text=text,
                    normalised_text=key,
                    page_index=page.page_index,
                    bbox_global=bbox,
                    candidate_kind=candidate_kind,
                    blocking=blocking,
                    status=status,
                    expected_kind=semantics.expected_kind if semantics else None,
                    matched_node_ids=matched,
                    matched_assembly_ids=assembly_ids,
                    ownership_scope=scope,
                    match_method=(
                        "source_reference" if referenced else "exact_label" if exact else None
                    ),
                    reason=reason,
                    tag_semantics=semantics,
                )
            )

    items.sort(key=lambda item: (item.page_index, item.bbox_global.y, item.bbox_global.x, item.id))
    statuses = Counter(item.status for item in items)
    kinds = Counter(item.candidate_kind for item in items)
    reviewable = sum(item.blocking for item in items)
    assigned = sum(item.blocking and item.status == "assigned" for item in items)
    return NativeTextInventory(
        items=items,
        summary={
            "candidate_count": len(items),
            "reviewable_tag_count": reviewable,
            "assigned_tag_count": assigned,
            "assembly_owned_occurrence_count": sum(
                bool(item.matched_assembly_ids) for item in items
            ),
            "ambiguous_assembly_occurrence_count": sum(
                bool(item.matched_assembly_ids) and item.status == "ambiguous" for item in items
            ),
            "specification_reference_count": sum(
                item.ownership_scope == "specification_reference" for item in items
            ),
            "unresolved_tag_count": sum(
                item.blocking and item.status in {"unresolved", "ambiguous"} for item in items
            ),
            "assignment_ratio": round(assigned / reviewable, 4) if reviewable else 1.0,
            "status_counts": dict(sorted(statuses.items())),
            "candidate_kind_counts": dict(sorted(kinds.items())),
        },
    )


def _group_is_in_title_block(group: list[TextEvidence], page: PageEvidence) -> bool:
    bbox = _bbox_union(group)
    center_x = bbox.x + bbox.w / 2
    center_y = bbox.y + bbox.h / 2
    return center_x >= page.width * 0.55 and center_y >= page.height * 0.78


def _classify_candidate(
    text: str, legend_pack: LegendPack | None
) -> tuple[CandidateKind, bool, TagSemantics | None] | None:
    compact = " ".join(text.strip().split())
    key = normalise_text_key(compact)
    if not key or len(key) > 40:
        return None
    if _DIMENSION_RE.fullmatch(compact):
        return "dimension", False, None
    if _DRAWING_REFERENCE_RE.fullmatch(compact):
        return "drawing_reference", False, None
    if _LINE_NUMBER_RE.fullmatch(compact):
        return "line_number", False, None
    semantics = infer_tag_semantics(compact, legend_pack)
    if semantics is not None:
        kind: CandidateKind = (
            "instrument_tag" if semantics.expected_kind == "instrument" else "equipment_tag"
        )
        return kind, True, semantics
    match = _SIMPLE_TAG_RE.fullmatch(compact)
    if (
        match is not None
        and match.group("prefix").upper() not in _EXCLUDED_PREFIXES
        and not (
            len(re.sub(r"[^0-9]", "", match.group("number"))) == 1
            and len(match.group("prefix")) < 2
        )
    ):
        return "unknown_tag", True, None
    return None


def _instrument_function(prefix: str) -> str:
    # ISA function letters have precedence over the measured-variable letter.
    functions = prefix[1:]
    if "C" in functions:
        return "controller"
    if "T" in functions:
        return "transmitter"
    if "I" in functions:
        return "indicator"
    if "R" in functions:
        return "recorder"
    if "E" in functions:
        return "element"
    if "S" in functions:
        return "switch"
    if "A" in functions:
        return "alarm"
    if "V" in functions:
        return "valve_actuator"
    return "unclassified_instrument"


def _candidate_span_groups(spans: list[TextEvidence]) -> list[list[TextEvidence]]:
    groups: list[list[TextEvidence]] = [[span] for span in spans]
    by_line: dict[tuple[int | None, int | None], list[TextEvidence]] = {}
    for span in spans:
        by_line.setdefault((span.block_index, span.line_index), []).append(span)
    for line in by_line.values():
        ordered = sorted(line, key=lambda item: item.word_index or 0)
        for size in (2, 3):
            for start in range(len(ordered) - size + 1):
                window = ordered[start : start + size]
                if all(
                    right.bbox.x - left.bbox.x2 <= max(left.bbox.h, right.bbox.h) * 3
                    for left, right in zip(window, window[1:], strict=False)
                ):
                    groups.append(window)
    groups.extend(_spatial_tag_groups(spans))
    return groups


def _spatial_tag_groups(spans: list[TextEvidence]) -> list[list[TextEvidence]]:
    """Pair code/number halves that CAD exported as separate PDF blocks.

    Instrument bubbles commonly contain the function code above the loop
    number.  PyMuPDF correctly recovers both strings but they often have
    unrelated block and line indices, so line-based word grouping cannot join
    them.  Mutual-nearest geometric pairing avoids a combinatorial set of
    false tag candidates in dense instrument clusters.
    """

    prefixes = [
        span for span in spans if re.fullmatch(r"[A-Z]{1,6}", span.text.strip(), re.IGNORECASE)
    ]
    numbers = [
        span for span in spans if re.fullmatch(r"\d{2,6}[A-Z]?", span.text.strip(), re.IGNORECASE)
    ]
    scored: list[tuple[float, str, str, TextEvidence, TextEvidence]] = []
    for prefix in prefixes:
        for number in numbers:
            pcx = prefix.bbox.x + prefix.bbox.w / 2
            pcy = prefix.bbox.y + prefix.bbox.h / 2
            ncx = number.bbox.x + number.bbox.w / 2
            ncy = number.bbox.y + number.bbox.h / 2
            scale = max(prefix.bbox.h, number.bbox.h, 1)
            dx = abs(ncx - pcx)
            dy = ncy - pcy
            stacked = -0.4 * scale <= dy <= 3.5 * scale and dx <= 1.8 * scale
            same_line = abs(dy) <= 0.8 * scale and (
                -0.5 * scale <= number.bbox.x - prefix.bbox.x2 <= 3.0 * scale
            )
            if not (stacked or same_line):
                continue
            score = (dx / scale) + (abs(dy) / scale) + (0.0 if stacked else 0.4)
            scored.append((score, prefix.id, number.id, prefix, number))

    used_prefixes: set[str] = set()
    used_numbers: set[str] = set()
    result: list[list[TextEvidence]] = []
    for _, prefix_id, number_id, prefix, number in sorted(scored):
        if prefix_id in used_prefixes or number_id in used_numbers:
            continue
        used_prefixes.add(prefix_id)
        used_numbers.add(number_id)
        result.append([prefix, number])
    return result


def _bbox_union(spans: list[TextEvidence]) -> BBox:
    x1 = min(span.bbox.x for span in spans)
    y1 = min(span.bbox.y for span in spans)
    x2 = max(span.bbox.x2 for span in spans)
    y2 = max(span.bbox.y2 for span in spans)
    return BBox(x=x1, y=y1, w=max(1, x2 - x1), h=max(1, y2 - y1))


def _node_text_keys(node: ReconciledNode) -> set[str]:
    attrs = node.attributes or {}
    values: list[Any] = [node.label, attrs.get("canonical_tag")]
    values.extend(attrs.get("raw_text_candidates") or [])
    values.extend(node.alternate_readings or [])
    return {key for value in values if (key := normalise_text_key(value))}


def _inventory_status(
    *,
    candidate_kind: CandidateKind,
    matched: list[str],
    nodes: list[ReconciledNode],
    semantics: TagSemantics | None,
) -> tuple[InventoryStatus, str]:
    if candidate_kind in {"dimension", "line_number", "drawing_reference"}:
        return "excluded", f"{candidate_kind} is document evidence, not a connectable object"
    if not matched:
        return "unresolved", "no extracted graph node is linked to this printed tag"
    if len(matched) > 1:
        return "ambiguous", "more than one graph node is linked to this printed tag"
    if semantics is not None:
        node = next((item for item in nodes if item.id == matched[0]), None)
        if node is None or node.kind != semantics.expected_kind:
            return "unresolved", "linked node kind disagrees with tag and legend semantics"
    return "assigned", "printed tag is linked to one compatible graph node"
