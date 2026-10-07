"""Bounded classification of complete native legend rows, without model navigation."""

from __future__ import annotations

import base64
import io
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from diagex.llm.client import is_non_retryable_api_error
from diagex.vision.encode import encode_image_block
from diagex.vision.legend_models import LegendEntry, LegendRegionCoverage
from diagex.vision.legend_rows import native_legend_rows, union_boxes
from diagex.vision.models import BBox

BATCH_SIZE = 12


def suspicious_legend_label(label):
    text = str(label).casefold()
    metadata = any(term in text for term in ("copyright", "property of", "unauthorized", "本图纸", "grade of qualification"))
    mixed_lines = sum(term in text for term in ("主工艺管线", "次工艺管线", "电信号线", "导压毛细管", "气压信号线")) > 1
    return metadata or mixed_lines or text.strip().startswith("=")


@dataclass
class LegendRunState:
    consecutive_failures: int = 0
    total_failures: int = 0
    stop_reason: str | None = None

    def observe(self, failed, policy):
        self.consecutive_failures = self.consecutive_failures + 1 if failed else 0
        self.total_failures += int(failed)
        if (
            self.consecutive_failures >= policy.consecutive_failure_limit
            or self.total_failures >= policy.total_failure_limit
        ):
            self.stop_reason = "Legend classification stopped after repeated failed batches; completed source rows are saved for reuse"


class RowDecision(BaseModel):
    row_id: str
    decision: Literal["accept", "reject", "uncertain"]
    kind: Literal["equipment", "instrument", "line", "valve", "connector", "other"] = "other"
    symbol_class: str = "unclassified_equipment"
    description: str = ""
    candidate_shapes: list[
        Literal[
            "round_symbol",
            "instrument_frame",
            "valve_body",
            "capsule_body",
            "connected_frame",
            "continuation_arrow",
        ]
    ] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    reason: str = ""

    @model_validator(mode="after")
    def _accept_requires_classification(self):
        if self.decision == "accept" and (
            not {"kind", "symbol_class"}.issubset(self.model_fields_set)
            or not self.symbol_class.strip()
        ):
            raise ValueError("An accepted row needs an explicit kind and symbol_class")
        if (
            self.decision == "accept"
            and self.attributes.get("symbol_role") == "actuator"
            and (self.kind != "equipment" or self.attributes.get("valve_type"))
        ):
            raise ValueError("Actuator role conflicts with valve-body classification")
        return self


_SYSTEM = """Classify the supplied complete native PDF legend rows. Python owns row
identity, printed labels, and geometry. Return exactly one submit_legend_rows
call with one decision for each row_id. Never return replacement labels or boxes.
Each image shows the glyph and its complete adjacent printed label, not a tile
fragment. Use the printed definition to interpret the symbol. Do not invent a
subtype when the label does not support it. Reject title blocks, logos, headers,
numbering explanations and other non-legend rows. Mark ambiguous rows uncertain.
Native row grouping can be wrong: if a crop combines unrelated definitions or
mixes glyphs with a title block, mark it uncertain instead of inventing one
classification for the group. Accept only a coherent graphical definition.
A drawn glyph with a caption is not a text-only abbreviation. A generic instrument
installation or DCS/PLC convention does not establish a measured variable or
instrument function; use unclassified_instrument with symbol_role=installation.
A safety valve is a valve, not merely its actuator. Pipe caps, blind flanges and
fittings are not valves. Actuator-only legend rows can show a contextual valve:
set attributes.symbol_role="actuator", kind="equipment", symbol_class="actuator"
and attributes.equipment_class="actuator". Omit valve_type: this row defines
actuation, not the contextual valve body's subtype. For a complete actuated
valve definition, retain kind="valve" and record actuation separately.
Use kind=connector for continuation conventions (including drawing-reference
arrows). This describes a physical endpoint, independently of resolving its
matching sheet. candidate_shapes names applicable glyph families, not extra
objects. An equipment row depicting a complete vessel uses capsule_body or
connected_frame; an internal P bubble must not inherit the vessel classification.
Use attributes for equipment_class, valve_type, instrument_function only when
supported by the complete printed definition."""
_TOOL = {
    "name": "submit_legend_rows",
    "description": "Classify the supplied native source rows.",
    "input_schema": {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": RowDecision.model_json_schema()}},
        "required": ["rows"],
    },
}


def classify_native_legend(
    *, page, evidence, region, client, cost_tracker, cfg, coverage, prior=None, run_state=None
):
    selected = BBox(x=region[0], y=region[1], w=region[2], h=region[3]) if region else None
    rows = native_legend_rows(evidence, selected)
    coverage_start = len(coverage) if coverage is not None else 0
    entries = []
    run_state = run_state or LegendRunState()
    prior_entries = (
        {e.source_row_id: e for e in prior.entries if e.attributes.get("row_status") == "accept"}
        if prior
        else {}
    )
    prior_rejections = (
        {
            c.source_row_id: c
            for c in prior.coverage
            if c.source_row_id and c.status == "complete" and c.entry_count == 0
        }
        if prior
        else {}
    )
    pending = []
    pending_verification = []
    previous_rejections = {}
    for row in rows:
        if row.id in prior_entries and not suspicious_legend_label(row.label):
            entries.append(prior_entries[row.id])
            if coverage is not None:
                coverage.append(
                    LegendRegionCoverage(
                        page_index=page.page_index,
                        source_row_id=row.id,
                        bbox=union_boxes([row.bbox, row.label_bbox]),
                        status="complete",
                        entry_count=1,
                        reason=f"{row.id} ({row.label}): reused completed source-row classification",
                    )
                )
        elif row.id in prior_rejections:
            if prior_rejections[row.id].verification_passes == 2:
                if coverage is not None:
                    coverage.append(prior_rejections[row.id])
            else:
                pending_verification.append(row)
                previous_rejections[row.id] = prior_rejections[row.id].reason
        else:
            pending.append(row)
    batches = deque(
        (pending[start : start + BATCH_SIZE], False) for start in range(0, len(pending), BATCH_SIZE)
    )
    batches.extend(
        (pending_verification[start : start + BATCH_SIZE], True)
        for start in range(0, len(pending_verification), BATCH_SIZE)
    )
    while batches:
        batch, verifying_rejection = batches.popleft()
        verify_next = []
        content = []
        for row in batch:
            context = union_boxes([row.bbox, row.label_bbox])
            pad = max(2, round(min(row.bbox.w, row.bbox.h) * 0.05))
            image = page.image.crop(
                (
                    max(0, context.x - pad),
                    max(0, context.y - pad),
                    min(page.width, context.x2 + pad),
                    min(page.height, context.y2 + pad),
                )
            )
            image.thumbnail((1400, 500))
            content += [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "row_id": row.id,
                            "printed_label": row.label,
                            **(
                                {
                                    "review_instruction": "This row was provisionally rejected. Independently recheck its complete printed definition and glyph. Accept a real graphical definition, including actuator-only and installation conventions. Reject metadata or numbering examples; do not accept just to fill coverage.",
                                    "previous_reason": previous_rejections.get(row.id, ""),
                                }
                                if verifying_rejection
                                else {}
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
                encode_image_block(image),
            ]
        decisions = defaultdict(list)
        raw_decisions = defaultdict(list)
        error = ""
        failure_kind = None
        response_received = False
        try:
            if run_state.stop_reason:
                raise RuntimeError(run_state.stop_reason)
            response = client.messages_create(
                system=_SYSTEM,
                messages=[{"role": "user", "content": content}],
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": _TOOL["name"]},
                max_tokens=6000,
                thinking={"type": "disabled"},
                reasoning_mode_override="disabled",
                output_config={"effort": "low"},
                time_budget_s=cfg.symbol_perception.reasoning_timeout_s,
                max_attempts=cfg.symbol_perception.transport_attempts,
            )
            response_received = True
            cost_tracker.record(
                response,
                step=len(cost_tracker.steps) + 1,
                page_index=page.page_index,
                tile_id="native_legend_rows",
            )
            for block in response.content:
                b = block if isinstance(block, dict) else block.model_dump()
                if b.get("type") != "tool_use" or b.get("name") != _TOOL["name"]:
                    continue
                for value in (b.get("input") or {}).get("rows", []):
                    if isinstance(value, dict) and isinstance(value.get("row_id"), str):
                        raw_decisions[value["row_id"]].append(value)
                    try:
                        d = RowDecision.model_validate(value)
                        decisions[d.row_id].append(d)
                    except ValidationError:
                        continue
        except Exception as exc:
            if is_non_retryable_api_error(exc):
                raise
            error = str(exc)
            failure_kind = (
                "not_inspected"
                if run_state.stop_reason
                else "contract"
                if response_received
                else "transport"
            )
        bad_rows = 0
        for row in batch:
            outcomes = decisions[row.id]
            # Identical repeats are one decision; incompatible outcomes remain
            # explicit review evidence, never a first/last-wins classification.
            unique = {d.model_dump_json(exclude={"reason"}): d for d in outcomes}
            valid = len(unique) == 1 and len(outcomes) == len(raw_decisions[row.id])
            bad_rows += int(not valid)
            decision = (
                next(iter(unique.values()))
                if valid
                else RowDecision(
                    row_id=row.id,
                    decision="uncertain",
                    reason=error or "Missing, malformed or conflicting row classification",
                )
            )
            if decision.decision == "accept" and suspicious_legend_label(row.label):
                decision = decision.model_copy(update={"decision": "uncertain", "reason": "Source grouping includes metadata, numbering examples or multiple definitions; split and review its source regions."})
            if decision.decision == "reject" and not verifying_rejection:
                # A valid source definition was previously rejected despite a
                # correct descriptive reason. Verify negatives once, without
                # changing accepted siblings or treating rejection as failure.
                verify_next.append(row)
                previous_rejections[row.id] = decision.reason
                continue
            if coverage is not None:
                coverage.append(
                    LegendRegionCoverage(
                        page_index=page.page_index,
                        source_row_id=row.id,
                        verification_passes=2 if verifying_rejection else 1,
                        bbox=union_boxes([row.bbox, row.label_bbox]),
                        status="partial" if decision.decision == "uncertain" else "complete",
                        failure_kind=(failure_kind or "contract")
                        if not valid
                        else "ambiguity"
                        if decision.decision == "uncertain"
                        else None,
                        entry_count=int(decision.decision != "reject"),
                        reason=f"{row.id} ({row.label}): {decision.reason or decision.decision}",
                    )
                )
            if decision.decision == "reject":
                continue
            pad = max(2, round(min(row.bbox.w, row.bbox.h) * 0.04))
            box = BBox(
                x=max(0, row.bbox.x - pad),
                y=max(0, row.bbox.y - pad),
                w=min(page.width, row.bbox.x2 + pad) - max(0, row.bbox.x - pad),
                h=min(page.height, row.bbox.y2 + pad) - max(0, row.bbox.y - pad),
            )
            thumbnail = page.image.crop((box.x, box.y, box.x2, box.y2))
            thumbnail.thumbnail((cfg.pid.legend_thumb_max_dim, cfg.pid.legend_thumb_max_dim))
            buf = io.BytesIO()
            thumbnail.save(buf, format="PNG")
            attrs = dict(decision.attributes)
            attrs.update(
                legend_kind="symbol",
                row_status=decision.decision,
                candidate_shapes=json.dumps(decision.candidate_shapes),
                source_path_ids=json.dumps(row.source_path_ids),
                source_text_ids=json.dumps(row.source_text_ids),
                classification_evidence=json.dumps(raw_decisions[row.id], ensure_ascii=False),
                classification_reason=decision.reason,
            )
            if verifying_rejection:
                attrs["previous_rejection_reason"] = previous_rejections.get(row.id, "")
            entries.append(
                LegendEntry(
                    label=row.label,
                    description=decision.description or decision.reason,
                    symbol_class=decision.symbol_class,
                    kind=decision.kind,
                    source="legend_extracted",
                    source_row_id=row.id,
                    source_page_index=page.page_index,
                    source_bbox=box,
                    source_label_bbox=row.label_bbox,
                    crop_method="native_text_paths",
                    crop_quality="recovered",
                    attributes=attrs,
                    image_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
                )
            )
        if not run_state.stop_reason and (
            not verifying_rejection or bad_rows >= max(1, len(batch) / 2)
        ):
            run_state.observe(bad_rows >= max(1, len(batch) / 2), cfg.symbol_perception)
        if verify_next:
            batches.appendleft((verify_next, True))
    if not rows and coverage is not None:
        coverage.append(
            LegendRegionCoverage(
                page_index=page.page_index,
                bbox=selected or BBox(x=0, y=0, w=page.width, h=page.height),
                status="partial",
                failure_kind="not_inspected",
                reason="No native graphical row correspondence; inspect source coverage",
            )
        )
    position = {r.id: i for i, r in enumerate(rows)}
    if coverage is not None:
        coverage[coverage_start:] = sorted(
            coverage[coverage_start:], key=lambda c: position.get(c.source_row_id, len(rows))
        )
    return sorted(entries, key=lambda e: position[e.source_row_id])
