"""Source-supported semantic suggestions for immutable broad observations.

This post-recognition stage does not change the benchmark inventory, geometry,
or source geometry. Validated classifications feed the normal fusion pipeline.
"""
from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from diagex.knowledge.library import supplied_trace
from diagex.llm.prompts.output_language import CHINESE_EXPLANATIONS
from diagex.vision.encode import encode_image_block
from diagex.vision.legend_context import select_legend_context
from diagex.vision.reference_evidence import KnowledgeCitation, knowledge_matches

SYSTEM = """Interpret previously recognized P&ID symbols using the original
source image and any supplied prepared legend definitions. The first image is
the source; subsequent labeled images are legend references, not more objects.
Symbol IDs and bounds are immutable. Return one result per supplied symbol_id.
Response structure is results -> decision row -> symbol. Put kind, printed_tag,
equipment_class, instrument_function and all other classification fields inside
the row's nested "symbol" object, never alongside decision or source_evidence.
For example, an interpreted row has "symbol": {"kind": "equipment",
"equipment_class": "unclassified_equipment", "structural_description": "..."}.
This is a format example, not an object to detect. For non_node or unresolved,
leave symbol null and do not emit any classification fields.
Use decision=interpreted only with a supported engineering kind/class. A reference
match is optional: when no applicable reference establishes the classification,
provide source_evidence describing the distinct physical glyph inside this exact
source box: its outline, internal strokes, and relevant visible text/attachments.
Leave citation lists empty when no supplied definition actually matches. Source
evidence, not detector category/confidence or process plausibility, must establish
the physical symbol. A tag alone is insufficient. With a reference match, cite
the visible glyph features and the reference correspondence in reason.
Preserve a clear physical symbol when its subtype or function is unknown: use
equipment_class=unclassified_equipment with structural_description, or
instrument_function=unclassified_instrument. A visible valve body may use
equipment_class=valve without an invented valve_type or actuation.
Copy printed tags only when visible;
leave absent text and unsupported subtype, actuation or function fields null.
Broad detector categories are fallible search hints, not detailed semantics.
Do not create objects, change bounds, invent native IDs, or approve graph nodes.
Use decision=non_node for visible flow-direction arrows, non-node marks, and
identity callouts whose enclosure/leader labels another physical body. A circle
containing a tag is not sufficient evidence of an independent instrument.
Use the glyph's own strokes, leader attachment and repeated local drawing convention;
neither a connecting stroke nor particular tag letters alone decides its role.
Use decision=unresolved for indistinct/incomplete ink, boxes grouping multiple
glyphs, ambiguous symbol-versus-annotation roles, or conflicting definitions.
Missing references alone do not make a clear physical symbol unresolved.
Non_node and unresolved decisions require a reason and no symbol classification. Do not suppress a real
symbol merely because the legend is incomplete. Knowledge illustrations are
definitions, never additional objects. Explicit drawing legends take precedence.
Never copy reference placeholders such as (#), (##), or (*) as actual tags.
Optional knowledge_reference_id and knowledge_variant_id must be supplied IDs;
knowledge_evidence must describe the actual source drawing features supporting
the match. For multiple references, use knowledge_citations with reference_id,
optional variant_id and drawing_evidence for each reference actually used in
the match. Omit a match when uncertain. These are signal references, not proof
of process-pipe connections. The results are model classifications with source evidence, not engineering approval. Call submit_raster_semantics once."""


class RasterSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["equipment", "instrument", "opc"]
    printed_tag: str | None = None
    equipment_class: str | None = None
    valve_type: str | None = None
    actuation: Literal["manual", "solenoid", "electric_motor", "pneumatic", "hydraulic", "spring", "other"] | None = None
    instrument_class: str | None = None
    instrument_function: str | None = None
    measured_variable: Literal["pressure", "temperature", "flow", "level", "analysis", "other"] | None = None
    loop_number: str | None = None
    connector_type: str | None = None
    opc_direction: Literal["in", "out"] | None = None
    drawing_ref: str | None = None
    service: str | None = None
    line_id: str | None = None
    structural_description: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def trim_text(cls, value):
        if isinstance(value, str):
            return " ".join(value.split()) or None
        return value

    @model_validator(mode="after")
    def classified_kind(self):
        required = {
            "equipment": (self.equipment_class, self.valve_type),
            "instrument": (self.instrument_class, self.instrument_function),
            "opc": (self.connector_type,),
        }
        if not any(required[self.kind]):
            raise ValueError("An engineering class is required for this kind")
        incompatible = {
            "equipment": (self.instrument_class, self.instrument_function, self.connector_type, self.opc_direction),
            "instrument": (self.equipment_class, self.valve_type, self.actuation, self.connector_type, self.opc_direction),
            "opc": (self.equipment_class, self.valve_type, self.actuation, self.instrument_class, self.instrument_function),
        }
        if any(incompatible[self.kind]):
            raise ValueError("Classification fields conflict with the engineering kind")
        return self


class SemanticDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol_id: str
    decision: Literal["interpreted", "non_node", "unresolved"]
    reason: str
    source_evidence: str | None = None
    legend_entry_ids: list[str]
    symbol: RasterSemantics | None = None
    knowledge_reference_id: str | None = None
    knowledge_variant_id: str | None = None
    knowledge_evidence: str | None = None
    knowledge_citations: list[KnowledgeCitation] = Field(default_factory=list)

    @field_validator("source_evidence", mode="before")
    @classmethod
    def trim_source_evidence(cls, value):
        if isinstance(value, str):
            return " ".join(value.split()) or None
        return value

    @model_validator(mode="after")
    def valid_decision(self):
        if not self.reason.strip():
            raise ValueError("Visible evidence reason is required")
        if self.decision == "interpreted":
            if self.symbol is None or not (
                self.source_evidence or self.legend_entry_ids
                or self.knowledge_reference_id or self.knowledge_citations
            ):
                raise ValueError("Interpretation requires classification and explicit source evidence or references")
        elif self.symbol is not None:
            raise ValueError("Non-node and unresolved decisions must not classify a node")
        return self


class SemanticResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[SemanticDecision]


def normalize_semantics(payload, symbol_ids, legend_entries, knowledge_context=None):
    """Keep a decision for every input, including malformed/missing responses."""
    if not isinstance(payload, dict) or set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise ValueError("Expected exactly one results array")
    if len(symbol_ids) != len(set(symbol_ids)):
        raise ValueError("Symbol IDs must be unique")
    legend = {e["legend_entry_id"]: e for e in legend_entries}
    if len(legend) != len(legend_entries):
        raise ValueError("Legend references must be unique")
    grouped, unknown = defaultdict(list), []
    for row in payload["results"]:
        identity = row.get("symbol_id") if isinstance(row, dict) else None
        if not isinstance(identity, str) or identity not in symbol_ids:
            unknown.append(row)
        else:
            grouped[identity].append(row)
    decisions = []
    for identity in symbol_ids:
        rows = grouped[identity]
        decision = {"symbol_id": identity, "decision": "unresolved", "legend_entry_ids": [],
                    "reason": "Missing decision" if not rows else "Duplicate decisions", "wire_rows": copy.deepcopy(rows)}
        if len(rows) == 1:
            try:
                parsed = SemanticDecision.model_validate(rows[0])
                if any(ref not in legend for ref in parsed.legend_entry_ids):
                    raise ValueError("Unknown legend reference")
                matches, errors = knowledge_matches(knowledge_context or {}, parsed)
                match = matches[0] if matches else None
                if parsed.decision == "interpreted" and errors:
                    decision["knowledge_match_errors"] = errors
                    raise ValueError("Invalid knowledge reference: " + errors[0]["reason"])
                if parsed.decision == "interpreted" and not (parsed.legend_entry_ids or match or parsed.source_evidence):
                    raise ValueError("Interpretation requires explicit source evidence or a supplied reference with drawing evidence")
                if parsed.decision == "interpreted" and not parsed.legend_entry_ids and match:
                    references = [e for e in knowledge_context["references"] if e["id"] in {m["reference_id"] for m in matches}]
                    if not any(e.get("reference_type") == "symbol" for e in references):
                        raise ValueError("A non-symbol convention cannot establish an engineering object")
                if parsed.symbol is not None:
                    kinds = {"equipment": {"equipment", "valve"}, "instrument": {"instrument"}, "opc": {"connector"}}
                    if parsed.legend_entry_ids and not any(legend[ref]["kind"] in kinds[parsed.symbol.kind] for ref in parsed.legend_entry_ids):
                        raise ValueError("Cited legend does not support this engineering kind")
                    markers = {s["example_marker"] for e in (knowledge_context or {}).get("references", []) for s in e.get("text_slots", [])}
                    for field in ("printed_tag", "drawing_ref", "line_id"):
                        if getattr(parsed.symbol, field) in markers:
                            setattr(parsed.symbol, field, None)
                decision.update(parsed.model_dump(mode="json", exclude_none=True))
                from diagex.vision.reference_evidence import legend_matches
                legend_used, legend_errors = legend_matches(parsed.legend_entry_ids, parsed.reason, legend_entries)
                decision["legend_matches"] = legend_used
                if legend_errors:
                    decision["legend_match_errors"] = legend_errors
                if match:
                    decision["knowledge_match"] = match
                    decision["knowledge_matches"] = matches
                if errors:
                    decision["knowledge_match_error"] = errors[0]["reason"]
                    decision["knowledge_match_errors"] = errors
            except (ValueError, TypeError) as exc:
                decision["reason"] = str(exc)
        if knowledge_context:
            decision["supplied_knowledge"] = supplied_trace(knowledge_context)
        decisions.append(decision)
    return decisions, unknown


def interpreted_detection(observation, interpretation):
    """Attach validated model semantics to an immutable source observation."""
    parsed = SemanticDecision.model_validate({k: v for k, v in interpretation.items() if k in SemanticDecision.model_fields})
    if parsed.decision != "interpreted":
        return copy.deepcopy(observation)
    semantic = parsed.symbol.model_dump(exclude_none=True)
    kind, label = semantic.pop("kind"), semantic.pop("printed_tag", "")
    if "opc_direction" in semantic:
        semantic["direction"] = semantic.pop("opc_direction")
    return {**copy.deepcopy(observation), "kind": kind, "label": label, "raw_text": label or None,
            "attributes": {**observation["attributes"], **semantic,
                           **{k: interpretation[k] for k in ("supplied_knowledge", "knowledge_match", "knowledge_matches", "knowledge_match_error", "knowledge_match_errors", "legend_matches", "legend_match_errors") if k in interpretation},
                           "recognition_method": "vlm",
                           "recognition_evidence": parsed.source_evidence or parsed.reason,
                           **({"source_evidence": parsed.source_evidence} if parsed.source_evidence else {}),
                           "legend_entry_ids": parsed.legend_entry_ids,
                           "legend_interpretation_reason": parsed.reason,
                           "requires_legend_interpretation": True}}


def interpret_raster_reviews(*, reviews, client, cost_tracker, page, tile, view_image,
                             view_info, legend_entries, policy, deadline=None, on_attempt=None, knowledge_context=None):
    output = copy.deepcopy(reviews)
    compact, blocks, _ = select_legend_context(legend_entries, [], [])
    from diagex.vision.symbol_interpretation import _bounded_knowledge_context
    knowledge_context = _bounded_knowledge_context(knowledge_context, compact)
    audit = {"legend_entries": compact, "attempts": [], "scope": "Model classifications; unresolved observations remain explicit"}
    if knowledge_context:
        audit["supplied_knowledge"] = supplied_trace(knowledge_context)
    indexed = [(f"symbol-{i}", r) for i, r in enumerate(output) if r.get("object", {}).get("kind") == "raster_symbol"]
    for identity, review in indexed:
        review["legend_interpretation"] = {"symbol_id": identity, "decision": "unresolved", "legend_entry_ids": [],
                                            "reason": "Not yet interpreted"}
    if not indexed:
        return output, audit
    from diagex.vision.symbol_interpretation import _knowledge_image_blocks
    knowledge_blocks = _knowledge_image_blocks(knowledge_context or {})
    tool = {"name": "submit_raster_semantics", "description": "Classify source-supported symbols with optional reference matches, without changing geometry",
            "input_schema": SemanticResponse.model_json_schema()}
    view = view_info.page_bbox
    stop_reason = None
    for offset in range(0, len(indexed), 24):
        chunk = indexed[offset:offset + 24]
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "Interpretation time limit reached"
        if stop_reason:
            for _, review in chunk:
                review["legend_interpretation"]["reason"] = stop_reason
                review["status"] = "unreviewed"
            continue
        symbols = [{"symbol_id": identity, "broad_category": review["object"]["attributes"]["broad_category"],
                    "bbox_in_source_view": {"x": (review["bbox"]["x"] - view.x) / view.w,
                                            "y": (review["bbox"]["y"] - view.y) / view.h,
                                            "w": review["bbox"]["w"] / view.w,
                                            "h": review["bbox"]["h"] / view.h}}
                   for identity, review in chunk]
        request = {"symbols": symbols, "legend_entries": compact, "knowledge": knowledge_context or {}}
        attempt = {"symbol_ids": [identity for identity, _ in chunk]}
        audit["attempts"].append(attempt)
        try:
            if on_attempt:
                on_attempt("submit_raster_semantics")
            response = client.messages_create(system=SYSTEM + CHINESE_EXPLANATIONS, messages=[{"role": "user", "content": [
                encode_image_block(view_image), *blocks, *knowledge_blocks, {"type": "text", "text": json.dumps(request, ensure_ascii=False)}]}],
                tools=[tool], tool_choice={"type": "tool", "name": tool["name"]}, max_tokens=policy.first_pass_tokens,
                thinking={"type": "disabled"}, reasoning_mode_override="disabled", output_config={"effort": "low"},
                time_budget_s=min(policy.request_timeout_s, max(0.1, deadline - time.monotonic())) if deadline else policy.request_timeout_s,
                max_attempts=policy.transport_attempts)
            cost_tracker.record(response, step=max((s.step for s in cost_tracker.steps), default=0) + 1,
                                tile_id=tile.id, page_index=page.page_index)
            attempt["response_id"] = response.id
            raw = [b.input for b in response.content if b.type == "tool_use" and b.name == tool["name"]]
            attempt["raw_inputs"] = copy.deepcopy(raw)
            if len(raw) != 1:
                raise ValueError("Exactly one submit_raster_semantics call required")
            decisions, unknown = normalize_semantics(raw[0], attempt["symbol_ids"], compact, knowledge_context)
            attempt["unrecognized_rows"] = unknown
            for (_, review), decision in zip(chunk, decisions, strict=True):
                review["legend_interpretation"] = decision
        except Exception as exc:  # preserve completed recognition even when this later stage fails
            stop_reason = f"Legend interpretation failed: {type(exc).__name__}: {exc}"
            attempt["error"] = stop_reason
            for _, review in chunk:
                review["legend_interpretation"]["reason"] = stop_reason
                review["status"] = "unreviewed"
    return output, audit
