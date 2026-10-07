"""Legend-assisted contextual correction for ambiguous composite P&ID symbols.

The fixed-crop detector intentionally performs a small perception task.  That
can split one physical item (for example a boxed ``S`` actuator mounted above
a valve body) into two graph nodes.  This module performs one bounded visual
comparison per affected page before topology is reconstructed.  It may merge
only an ambiguous small glyph into a nearby valve, or conservatively retype an
enumerated candidate.  It cannot invent nodes, edges, or coordinates.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from typing import Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from diagex.dexpi_schema import ACTUATION_TYPE_KEYS, EQUIPMENT_CLASS_KEYS, VALVE_TYPE_KEYS
from diagex.llm.client import LLMClient, is_malformed_tool_json_error
from diagex.llm.cost import CostTracker
from diagex.ui.progress import ProgressReporter
from diagex.vision.encode import encode_image_block
from diagex.vision.fusion import FusionResult
from diagex.vision.models import BBox, Confidence, ReconciledNode

_SHORT_GLYPH_RE = re.compile(r"^[A-Z]{1,2}$", re.IGNORECASE)
_ACTUATION_LEGEND_TERMS: dict[str, tuple[str, ...]] = {
    "manual": ("manual", "hand", "手动"),
    "solenoid": ("solenoid", "electromagnetic", "电磁"),
    "electric_motor": ("electric motor", "motor operated", "电动", "电机"),
    "pneumatic": ("pneumatic", "diaphragm", "气动", "薄膜"),
    "hydraulic": ("hydraulic", "液动", "液压"),
    "spring": ("spring", "弹簧"),
}


class ContextCandidate(BaseModel):
    id: str
    page_index: int
    node_ids: list[str]
    bbox_global: BBox


class ContextCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_ref: str
    operation: Literal["merge_composite", "retype", "keep", "uncertain"]
    primary_ref: str
    absorbed_refs: list[str] = Field(default_factory=list)
    kind: Literal["equipment", "instrument", "opc"] | None = None
    equipment_class: str | None = None
    valve_type: str | None = None
    actuation: (
        Literal[
            "manual",
            "solenoid",
            "electric_motor",
            "pneumatic",
            "hydraulic",
            "spring",
            "other",
        ]
        | None
    ) = None
    legend_refs: list[str] = Field(default_factory=list)
    confidence: Confidence = "medium"
    evidence: list[str] = Field(default_factory=list)


class ContextSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corrections: list[ContextCorrection] = Field(default_factory=list)


class ContextResolution(BaseModel):
    operation: Literal["merge_composite", "retype"]
    primary_node_id: str
    absorbed_node_ids: list[str] = Field(default_factory=list)
    kind: Literal["equipment", "instrument", "opc"] | None = None
    equipment_class: str | None = None
    valve_type: str | None = None
    actuation: str | None = None
    legend_labels: list[str] = Field(default_factory=list)
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)


class ContextualPageResult(BaseModel):
    page_index: int
    resolutions: list[ContextResolution] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    attempts: int = 0
    response: dict[str, Any] = Field(default_factory=dict)


_TOOL = {
    "name": "submit_contextual_symbol_corrections",
    "description": "Resolve only enumerated ambiguous composite symbols on one P&ID page.",
    "input_schema": ContextSubmission.model_json_schema(),
}

_SYSTEM_PROMPT = """\
Compare ambiguous P&ID objects against the supplied page context, detailed
candidate crops, and PROJECT-SPECIFIC legend symbol sheet. Return exactly one
submit_contextual_symbol_corrections call.

The project legend overrides general conventions. Use only supplied group_ref,
node_ref, and legend_ref values. Never invent nodes, edges, tags, or geometry.

Allowed decisions:
- merge_composite: a small actuator/positioner glyph is physically attached to
  one nearby valve body and together they are one engineering object. The
  valve must be primary_ref and the attached glyph belongs in absorbed_refs.
- retype: the project legend clearly proves a supplied node has the wrong kind
  or subtype. A standalone M/S group may be retyped only when the full-page
  context and a cited project-legend image distinguish a motor/actuator from an
  ordinary instrument. Do not use proximity alone.
- keep: the supplied nodes are genuinely separate objects.
- uncertain: the images or project legend do not prove a safe correction.

For merge_composite, set actuation only when a cited legend_ref visibly and
textually supports it. A box containing S above a valve commonly means a
solenoid actuator only when the project legend confirms that convention.
Symbol strokes between an actuator and valve body are not process lines.

Do not merge instrument bubbles, equipment tags, pipe-size text, nozzles,
package-interface letters, drain funnels, or separate inline valves. Prefer
uncertain over an unsupported merge. Keep evidence literal and brief.
"""


def find_contextual_candidates(
    nodes: list[ReconciledNode], *, page_index: int
) -> list[ContextCandidate]:
    """Return deterministic small-glyph/valve neighbourhoods for one page."""
    page_nodes = [node for node in nodes if node.page_index == page_index]
    valves = [node for node in page_nodes if _is_valve(node)]
    ambiguous = [node for node in page_nodes if _is_ambiguous_small_node(node)]
    candidates: list[tuple[float, ContextCandidate]] = []
    for glyph in ambiguous:
        nearby = [
            valve for valve in valves if valve.id != glyph.id and _composite_proximity(glyph, valve)
        ]
        if not nearby:
            # Isolated M/S glyphs still need the holistic page+legend pass: M
            # may be a motor, an electric actuator, or a mistaken crop, while S
            # may be a detached solenoid symbol.  Keep this as a one-node
            # retype/uncertain group; merge validation remains impossible
            # without a physically attached valve in the group.
            if glyph.attributes.get("actuation") or _is_valve(glyph):
                continue
            candidates.append(
                (
                    float("inf"),
                    ContextCandidate(
                        id=f"ctx-{page_index + 1:04d}-{len(candidates) + 1:03d}",
                        page_index=page_index,
                        node_ids=[glyph.id],
                        bbox_global=glyph.bbox_global,
                    ),
                )
            )
            continue
        nearby.sort(key=lambda valve: (_bbox_gap(glyph.bbox_global, valve.bbox_global), valve.id))
        members = [glyph, *nearby[:3]]
        bbox = _bbox_union([node.bbox_global for node in members])
        candidates.append(
            (
                _bbox_gap(glyph.bbox_global, nearby[0].bbox_global),
                ContextCandidate(
                    id=f"ctx-{page_index + 1:04d}-{len(candidates) + 1:03d}",
                    page_index=page_index,
                    node_ids=[node.id for node in members],
                    bbox_global=bbox,
                ),
            )
        )
    return [item for _, item in sorted(candidates, key=lambda row: (row[0], row[1].id))[:20]]


def resolve_contextual_page(
    *,
    client: LLMClient,
    cost_tracker: CostTracker,
    reporter: ProgressReporter,
    page_index: int,
    rendered_image: Image.Image,
    nodes: list[ReconciledNode],
    legend_entries: list[dict[str, Any]],
    step: int,
    on_attempt: Any | None = None,
    max_tokens: int = 4000,
) -> ContextualPageResult:
    """Run one constrained visual correction call for an affected page."""
    candidates = find_contextual_candidates(nodes, page_index=page_index)
    visual_legend = _visual_legend_entries(legend_entries)
    if not candidates or not visual_legend:
        return ContextualPageResult(page_index=page_index)

    page_nodes = [node for node in nodes if node.page_index == page_index]
    node_refs = {
        f"N{index:03d}": node
        for index, node in enumerate(
            sorted(
                page_nodes, key=lambda value: (value.bbox_global.y, value.bbox_global.x, value.id)
            ),
            start=1,
        )
    }
    ref_by_id = {node.id: ref for ref, node in node_refs.items()}
    legend_refs = {f"L{index:03d}": entry for index, entry in enumerate(visual_legend, start=1)}
    group_refs = {f"C{index:03d}": group for index, group in enumerate(candidates, start=1)}

    overview = _annotated_overview(rendered_image, node_refs, group_refs)
    details = _candidate_montage(rendered_image, group_refs, ref_by_id)
    legend_sheet = _legend_symbol_montage(legend_refs)
    if details is None or legend_sheet is None:
        return ContextualPageResult(
            page_index=page_index,
            diagnostics=["contextual image montage could not be constructed"],
        )

    payload = {
        "page": page_index + 1,
        "nodes": [_node_payload(ref, node) for ref, node in node_refs.items()],
        "candidate_groups": [
            {
                "group_ref": ref,
                "node_refs": [ref_by_id[node_id] for node_id in group.node_ids],
                "bbox": group.bbox_global.model_dump(),
            }
            for ref, group in group_refs.items()
        ],
        "legend_entries": [
            {
                "legend_ref": ref,
                "label": entry.get("label"),
                "kind": entry.get("kind"),
                "symbol_class": entry.get("symbol_class"),
                "description": entry.get("description"),
                "attributes": entry.get("attributes") or {},
            }
            for ref, entry in legend_refs.items()
        ],
        "image_order": [
            "annotated page overview",
            "candidate detail sheet",
            "project legend sheet",
        ],
    }
    invalid: list[str] = []
    for attempt in range(1, 3):
        if on_attempt is not None:
            on_attempt()
        recovery = (
            "\nThe previous response was malformed. Call the tool once with valid JSON only."
            if attempt == 2
            else ""
        )
        try:
            response = client.messages_create(
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            encode_image_block(overview),
                            encode_image_block(details),
                            encode_image_block(legend_sheet),
                            {
                                "type": "text",
                                "text": (
                                    "Resolve every candidate group conservatively."
                                    f"{recovery}\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": _TOOL["name"]},
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
                on_stream_delta=lambda kind, text: reporter.on_stream_delta(kind=kind, text=text),
            )
        except ValueError as exc:
            if not is_malformed_tool_json_error(exc) or attempt == 2:
                raise
            invalid.append(str(exc))
            continue
        _report_response(response, reporter)
        cost_tracker.record(
            response,
            step=step + attempt - 1,
            tile_id=f"context_page_{page_index}",
            page_index=page_index,
        )
        reporter.on_token_update(total_tokens=cost_tracker.total_tokens())
        try:
            submission = _parse_submission(response)
        except ValueError as exc:
            invalid.append(str(exc))
            if attempt == 2:
                raise ValueError(
                    "contextual symbol response was malformed after one recovery attempt"
                ) from exc
            continue
        result = _validate_submission(
            page_index=page_index,
            submission=submission,
            node_refs=node_refs,
            group_refs=group_refs,
            ref_by_id=ref_by_id,
            legend_refs=legend_refs,
        )
        result.attempts = attempt
        result.response = _response_diagnostic(response)
        result.diagnostics.extend(invalid)
        return result
    raise AssertionError("contextual recovery loop exited unexpectedly")


def apply_contextual_results(
    objects: FusionResult, results: list[ContextualPageResult]
) -> FusionResult:
    """Apply already validated resolutions before deterministic topology."""
    nodes = {node.id: node.model_copy(deep=True) for node in objects.graph.nodes}
    conflicts = [dict(value) for value in objects.ambiguities]
    removed: set[str] = set()
    replacements: dict[str, str] = {}

    for result in results:
        conflicts.extend(dict(value) for value in result.conflicts)
        for resolution in result.resolutions:
            primary = nodes.get(resolution.primary_node_id)
            if primary is None or primary.id in removed:
                continue
            if resolution.operation == "merge_composite":
                absorbed = [
                    nodes[node_id]
                    for node_id in resolution.absorbed_node_ids
                    if node_id in nodes and node_id not in removed
                ]
                if not absorbed:
                    continue
                for node in absorbed:
                    removed.add(node.id)
                    replacements[node.id] = primary.id
                primary.attributes["assembly_ids"] = sorted(
                    set(primary.attributes.get("assembly_ids", [])).union(
                        *(node.attributes.get("assembly_ids", []) for node in absorbed)
                    )
                )
                primary.source_annotation_ids = sorted(
                    set(primary.source_annotation_ids).union(
                        *(node.source_annotation_ids for node in absorbed)
                    )
                )
                primary.source_evidence_ids = sorted(
                    set(primary.source_evidence_ids).union(
                        *(node.source_evidence_ids for node in absorbed)
                    )
                )
                primary.alternate_readings = sorted(
                    set(primary.alternate_readings).union(
                        *(node.alternate_readings for node in absorbed),
                        *(node.label for node in absorbed if node.label),
                    )
                )
            if resolution.kind:
                primary.kind = resolution.kind
            if resolution.equipment_class:
                primary.attributes["equipment_class"] = resolution.equipment_class
                if resolution.equipment_class != "valve":
                    primary.attributes.pop("valve_type", None)
            if resolution.valve_type:
                primary.attributes["valve_type"] = resolution.valve_type
                primary.attributes.pop("equipment_class", None)
            if resolution.actuation:
                primary.attributes["actuation"] = resolution.actuation
            history = list(primary.attributes.get("contextual_resolutions") or [])
            history.append(
                {
                    "operation": resolution.operation,
                    "absorbed_node_ids": resolution.absorbed_node_ids,
                    "legend_labels": resolution.legend_labels,
                    "confidence": resolution.confidence,
                    "evidence": resolution.evidence,
                }
            )
            primary.attributes["contextual_resolutions"] = history

    final_nodes = sorted(
        (node for node_id, node in nodes.items() if node_id not in removed),
        key=lambda node: (node.page_index, node.bbox_global.y, node.bbox_global.x, node.id),
    )
    assemblies = [a.model_copy(deep=True) for a in objects.graph.assemblies]
    for assembly in assemblies:
        members = set()
        for nid in assembly.member_node_ids:
            while nid in replacements:
                nid = replacements[nid]
            if nid in nodes and nid not in removed:
                members.add(nid)
        assembly.member_node_ids = sorted(members)
    graph = objects.graph.model_copy(
        update={"nodes": final_nodes, "conflicts": conflicts, "assemblies": assemblies}
    )
    return FusionResult(
        graph=graph,
        ambiguities=conflicts,
        native_text_assignment_count=objects.native_text_assignment_count,
    )


def _validate_submission(
    *,
    page_index: int,
    submission: ContextSubmission,
    node_refs: dict[str, ReconciledNode],
    group_refs: dict[str, ContextCandidate],
    ref_by_id: dict[str, str],
    legend_refs: dict[str, dict[str, Any]],
) -> ContextualPageResult:
    result = ContextualPageResult(page_index=page_index)
    seen_groups: set[str] = set()
    for correction in submission.corrections:
        group = group_refs.get(correction.group_ref)
        primary = node_refs.get(correction.primary_ref)
        if group is None or primary is None or correction.group_ref in seen_groups:
            result.diagnostics.append(
                f"rejected unknown or duplicate group {correction.group_ref!r}"
            )
            continue
        seen_groups.add(correction.group_ref)
        allowed_refs = {ref_by_id[node_id] for node_id in group.node_ids}
        if correction.primary_ref not in allowed_refs or any(
            ref not in allowed_refs for ref in correction.absorbed_refs
        ):
            result.diagnostics.append(
                f"rejected out-of-group correction for {correction.group_ref}"
            )
            continue
        cited = [legend_refs[ref] for ref in correction.legend_refs if ref in legend_refs]
        legend_labels = [str(entry.get("label") or "") for entry in cited]
        if correction.operation in {"keep", "uncertain"}:
            if correction.operation == "uncertain":
                result.conflicts.append(_uncertain_conflict(group, correction, node_refs))
            continue
        if correction.confidence != "high":
            result.conflicts.append(_uncertain_conflict(group, correction, node_refs))
            continue
        if correction.equipment_class and correction.equipment_class not in EQUIPMENT_CLASS_KEYS:
            result.diagnostics.append(
                f"rejected unknown equipment class {correction.equipment_class!r}"
            )
            continue
        if correction.valve_type and correction.valve_type not in VALVE_TYPE_KEYS:
            result.diagnostics.append(f"rejected unknown valve type {correction.valve_type!r}")
            continue
        if correction.actuation and correction.actuation not in ACTUATION_TYPE_KEYS:
            result.diagnostics.append(f"rejected unknown actuation {correction.actuation!r}")
            continue
        if correction.actuation and not _legend_supports_actuation(correction.actuation, cited):
            result.conflicts.append(_uncertain_conflict(group, correction, node_refs))
            result.diagnostics.append(
                f"did not apply {correction.group_ref}: cited legend does not support {correction.actuation}"
            )
            continue
        absorbed_nodes = [node_refs[ref] for ref in correction.absorbed_refs]
        if correction.operation == "merge_composite":
            if not cited:
                result.conflicts.append(_uncertain_conflict(group, correction, node_refs))
                result.diagnostics.append(
                    f"did not apply {correction.group_ref}: composite merge did not cite a project legend symbol"
                )
                continue
            if not _is_valve(primary) or not absorbed_nodes:
                result.diagnostics.append(f"rejected unsafe merge for {correction.group_ref}")
                continue
            if any(
                not _is_ambiguous_small_node(node)
                or not _directly_attached(node.bbox_global, primary.bbox_global)
                for node in absorbed_nodes
            ):
                result.diagnostics.append(f"rejected non-local merge for {correction.group_ref}")
                continue
        result.resolutions.append(
            ContextResolution(
                operation=correction.operation,
                primary_node_id=primary.id,
                absorbed_node_ids=[node.id for node in absorbed_nodes],
                kind=correction.kind,
                equipment_class=correction.equipment_class,
                valve_type=correction.valve_type,
                actuation=correction.actuation,
                legend_labels=legend_labels,
                confidence=correction.confidence,
                evidence=correction.evidence[:6],
            )
        )
        result.conflicts.append(
            {
                "type": "contextual_symbol_resolution",
                "page_index": page_index,
                "node_id": primary.id,
                "absorbed_node_ids": [node.id for node in absorbed_nodes],
                "status": "resolved",
                "operation": correction.operation,
                "legend_labels": legend_labels,
                "reason": "; ".join(correction.evidence[:4]),
            }
        )
    return result


def _uncertain_conflict(
    group: ContextCandidate,
    correction: ContextCorrection,
    node_refs: dict[str, ReconciledNode],
) -> dict[str, Any]:
    nodes_by_ref = {ref: node for ref, node in node_refs.items()}
    selected = [
        nodes_by_ref[ref]
        for ref in (correction.primary_ref, *correction.absorbed_refs)
        if ref in nodes_by_ref
    ]
    return {
        "type": "contextual_symbol_uncertainty",
        "page_index": group.page_index,
        "node_ids": [node.id for node in selected] or list(group.node_ids),
        "candidate_operation": correction.operation,
        "primary_node_id": nodes_by_ref.get(correction.primary_ref).id
        if correction.primary_ref in nodes_by_ref
        else None,
        "absorbed_node_ids": [
            nodes_by_ref[ref].id for ref in correction.absorbed_refs if ref in nodes_by_ref
        ],
        "suggested_actuation": correction.actuation,
        "suggested_valve_type": correction.valve_type,
        "legend_refs": list(correction.legend_refs),
        "status": "unresolved",
        "reason": "; ".join(correction.evidence[:4])
        or "project legend and local context did not support a safe correction",
    }


def _legend_supports_actuation(actuation: str, entries: list[dict[str, Any]]) -> bool:
    if actuation == "other":
        return bool(entries)
    terms = _ACTUATION_LEGEND_TERMS.get(actuation, ())
    text = " ".join(
        " ".join(
            str(entry.get(key) or "").casefold() for key in ("label", "description", "symbol_class")
        )
        for entry in entries
    )
    return bool(terms) and any(term.casefold() in text for term in terms)


def _is_valve(node: ReconciledNode) -> bool:
    return node.kind == "equipment" and bool(node.attributes.get("valve_type"))


def _is_ambiguous_small_node(node: ReconciledNode) -> bool:
    attrs = node.attributes
    label = " ".join((node.label or "").split()).strip()
    short_glyph = _SHORT_GLYPH_RE.fullmatch(label)
    # Single-letter package/nozzle interface marks such as A/B are not
    # actuator candidates.  S and M are the only project-common letter glyphs
    # accepted without a more descriptive unclassified symbol.
    if short_glyph and label.upper() not in {"S", "M"}:
        return False
    unclassified = (
        attrs.get("instrument_function") == "unclassified_instrument"
        or attrs.get("equipment_class") == "unclassified_equipment"
    )
    return (
        node.kind in {"equipment", "instrument"}
        and (unclassified or bool(short_glyph))
        and max(node.bbox_global.w, node.bbox_global.h) <= 180
    )


def _composite_proximity(glyph: ReconciledNode, valve: ReconciledNode) -> bool:
    gap = _bbox_gap(glyph.bbox_global, valve.bbox_global)
    scale = max(glyph.bbox_global.w, glyph.bbox_global.h, valve.bbox_global.w, valve.bbox_global.h)
    return gap <= max(45.0, min(180.0, scale * 1.6))


def _directly_attached(glyph: BBox, valve: BBox) -> bool:
    return _bbox_gap(glyph, valve) <= max(45.0, min(100.0, max(glyph.w, glyph.h, valve.w, valve.h)))


def _bbox_gap(left: BBox, right: BBox) -> float:
    dx = max(0, left.x - right.x2, right.x - left.x2)
    dy = max(0, left.y - right.y2, right.y - left.y2)
    return math.hypot(dx, dy)


def _bbox_union(boxes: list[BBox]) -> BBox:
    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x2 for box in boxes)
    y1 = max(box.y2 for box in boxes)
    return BBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))


def _visual_legend_entries(
    entries: list[dict[str, Any]], *, limit: int = 48
) -> list[dict[str, Any]]:
    values = [entry for entry in entries if entry.get("image_b64") and entry.get("kind") != "line"]
    actuation_terms = {
        term.casefold() for terms in _ACTUATION_LEGEND_TERMS.values() for term in terms
    }

    def priority(entry: dict[str, Any]) -> tuple[int, int, str]:
        text = " ".join(
            str(entry.get(key) or "").casefold() for key in ("label", "description", "symbol_class")
        )
        is_actuator = any(term in text for term in actuation_terms)
        return (
            0 if is_actuator else 1,
            0 if entry.get("source") == "legend_extracted" else 1,
            str(entry.get("label") or ""),
        )

    values.sort(key=priority)
    return values[:limit]


def _node_payload(ref: str, node: ReconciledNode) -> dict[str, Any]:
    return {
        "node_ref": ref,
        "kind": node.kind,
        "label": node.label,
        "bbox": node.bbox_global.model_dump(),
        "attributes": {
            key: value
            for key, value in node.attributes.items()
            if key
            in {
                "equipment_class",
                "valve_type",
                "instrument_function",
                "measured_variable",
                "actuation",
                "structural_description",
            }
        },
        "source_text": node.source_quote,
    }


def _annotated_overview(
    image: Image.Image,
    node_refs: dict[str, ReconciledNode],
    group_refs: dict[str, ContextCandidate],
) -> Image.Image:
    overview = image.convert("RGB").copy()
    overview.thumbnail((1800, 1400), Image.Resampling.LANCZOS)
    sx, sy = overview.width / image.width, overview.height / image.height
    draw = ImageDraw.Draw(overview)
    candidate_ids = {node_id for group in group_refs.values() for node_id in group.node_ids}
    for ref, node in node_refs.items():
        if node.id not in candidate_ids:
            continue
        box = node.bbox_global
        xy = (box.x * sx, box.y * sy, box.x2 * sx, box.y2 * sy)
        draw.rectangle(xy, outline=(220, 38, 38), width=3)
        draw.text((xy[0], max(0, xy[1] - 12)), ref, fill=(180, 20, 20))
    for ref, group in group_refs.items():
        box = group.bbox_global
        pad = 20
        xy = (
            max(0, (box.x - pad) * sx),
            max(0, (box.y - pad) * sy),
            min(overview.width, (box.x2 + pad) * sx),
            min(overview.height, (box.y2 + pad) * sy),
        )
        draw.rectangle(xy, outline=(37, 99, 235), width=4)
        draw.text((xy[0] + 3, xy[1] + 3), ref, fill=(20, 70, 190))
    return overview


def _candidate_montage(
    image: Image.Image,
    groups: dict[str, ContextCandidate],
    ref_by_id: dict[str, str],
) -> Image.Image | None:
    cells: list[Image.Image] = []
    for group_ref, group in groups.items():
        box = group.bbox_global
        padding = max(100, min(260, max(box.w, box.h) * 2))
        bounds = (
            max(0, box.x - padding),
            max(0, box.y - padding),
            min(image.width, box.x2 + padding),
            min(image.height, box.y2 + padding),
        )
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        crop = image.crop(bounds).convert("RGB")
        crop.thumbnail((700, 430), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (720, 470), "white")
        cell.paste(crop, ((720 - crop.width) // 2, 32 + (430 - crop.height) // 2))
        caption = f"{group_ref}: " + ", ".join(ref_by_id[node_id] for node_id in group.node_ids)
        ImageDraw.Draw(cell).text((8, 8), caption, fill=(20, 20, 20))
        cells.append(cell)
    if not cells:
        return None
    columns = 2
    rows = (len(cells) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 720, rows * 470), "white")
    for index, cell in enumerate(cells):
        montage.paste(cell, ((index % columns) * 720, (index // columns) * 470))
    return montage


def _legend_symbol_montage(entries: dict[str, dict[str, Any]]) -> Image.Image | None:
    cells: list[Image.Image] = []
    for legend_ref, entry in entries.items():
        try:
            raw = base64.b64decode(str(entry.get("image_b64") or ""), validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                sample = source.convert("RGB")
        except Exception:  # noqa: BLE001 - malformed optional legend crop
            continue
        sample.thumbnail((280, 135), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (300, 170), "white")
        cell.paste(sample, ((300 - sample.width) // 2, 26 + (135 - sample.height) // 2))
        ImageDraw.Draw(cell).text((7, 6), legend_ref, fill=(20, 20, 20))
        cells.append(cell)
    if not cells:
        return None
    columns = 4
    rows = (len(cells) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 300, rows * 170), "white")
    for index, cell in enumerate(cells):
        montage.paste(cell, ((index % columns) * 300, (index // columns) * 170))
    return montage


def _parse_submission(response: Any) -> ContextSubmission:
    text_parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "tool_use":
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if name == _TOOL["name"]:
                value = (
                    block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
                )
                return ContextSubmission.model_validate(value or {})
        if block_type == "text":
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if value:
                text_parts.append(str(value))
    raw = "\n".join(text_parts).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("contextual response did not contain structured JSON")
    return ContextSubmission.model_validate_json(raw[start : end + 1])


def _report_response(response: Any, reporter: ProgressReporter) -> None:
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type not in {"text", "thinking", "reasoning"}:
            continue
        value = (
            block.get(block_type) if isinstance(block, dict) else getattr(block, block_type, None)
        )
        if value:
            reporter.on_stream_delta(
                kind="reasoning" if block_type != "text" else "model", text=str(value)
            )


def _response_diagnostic(response: Any) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for block in getattr(response, "content", None) or []:
        if isinstance(block, dict):
            content.append({key: value for key, value in block.items() if key != "source"})
        else:
            content.append(
                {
                    key: getattr(block, key)
                    for key in ("type", "name", "input", "text")
                    if getattr(block, key, None) is not None
                }
            )
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "stop_reason": getattr(response, "stop_reason", None),
        "content": content,
    }
