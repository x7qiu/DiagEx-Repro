"""Broad raster recognition after legend preparation, without engineering typing.

These observations stay in the review inventory. Detailed legend interpretation
and graph construction are separate stages; an arrow is not equipment.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict

from diagex.llm.client import LLMClient
from diagex.vision.encode import encode_image_block
from diagex.vision.perception import (
    NormalizedBBox,
    PerceptionBatch,
    PerceptionOutcome,
    _bbox_center_is_owned,
    _project_normalized_bbox,
)
from diagex.vision.raster_detector import CLASSES

BROAD_SYSTEM = """Inspect the original P&ID image and recognize broad raster symbol
classes. This is symbol inventory, not engineering interpretation or graph design.
Use any supplied, previously prepared legend context as source evidence. Do not
invent a legend, detailed subtype, actuation, process meaning, tag, or native ID.

The broad categories are: valve (valve-body glyph), pump (pump/compressor glyph),
tank (tank/vessel body), instrumentation (instrument/function glyph), arrow
(flow-direction arrow on a line), inlet/outlet (off-page continuation glyph), and
general (another distinct equipment/component symbol not fitting those classes).
Flow-direction arrows count in this inventory even though they are not equipment
or independently approved graph nodes. Plain pipes, crossings, border strokes,
text, leader lines and annotation-only arrows are not symbol instances.

Detector hints are fallible. Inspect each hint's own source ink; correct class
and bounds. Retain small or untagged visible symbols. Do not infer existence from
engineering plausibility, detector confidence, or nearby text alone.
Return exactly one raster_results row per supplied proposal_id. Each row has
decision=symbol/reject/uncertain and a short reason describing visible evidence.
Every symbol row MUST also include broad_category, confidence=high/medium/low,
and bbox={x,y,w,h} normalized to THIS original image. A proposal_id does not
replace bbox. Reject/uncertain rows have only proposal_id, decision, and reason.
Never claim a detector ID as native evidence. Missing/invalid rows stay unresolved.

Continue searching for symbols the detector missed. Put additional symbols in
discoveries, with broad_category, confidence, bbox, and reason; do not duplicate
the supplied IDs there. Use submit_raster_symbols exactly once. No detailed
engineering objects are requested or approved by this call."""


def broad_schema():
    box = {"type": "object", "properties": {k: {"type": "number"} for k in ("x", "y", "w", "h")},
           "required": ["x", "y", "w", "h"], "additionalProperties": False}
    symbol = {"broad_category": {"type": "string", "enum": list(CLASSES)},
              "confidence": {"type": "string", "enum": ["high", "medium", "low"]}, "bbox": box,
              "reason": {"type": "string", "minLength": 1}}
    return {"type": "object", "properties": {
        "raster_results": {"type": "array", "items": {"type": "object", "properties": {
            "proposal_id": {"type": "string", "minLength": 1},
            "decision": {"type": "string", "enum": ["symbol", "reject", "uncertain"]}, **symbol},
            "required": ["proposal_id", "decision", "reason"], "additionalProperties": False}},
        "discoveries": {"type": "array", "items": {"type": "object", "properties": symbol,
            "required": list(symbol), "additionalProperties": False}}},
        "required": ["raster_results", "discoveries"], "additionalProperties": False}


def _symbol(value):
    if set(value) != {"broad_category", "confidence", "bbox", "reason"}:
        raise ValueError("Broad symbol must contain only category, confidence, bbox, and evidence reason")
    if value["broad_category"] not in CLASSES or value["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("Unknown broad class or confidence")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Visible evidence reason required")
    box = NormalizedBBox.model_validate(value["bbox"])
    return {**value, "bbox": box.model_dump()}


def normalize_broad(payload, proposal_ids):
    if not isinstance(payload, dict) or set(payload) != {"raster_results", "discoveries"}:
        raise ValueError("Expected raster_results and discoveries only")
    if not isinstance(payload["raster_results"], list) or not isinstance(payload["discoveries"], list):
        raise ValueError("Results/discoveries must be arrays")
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("Input proposal IDs must be unique")
    grouped = defaultdict(list)
    unknown = []
    for row in payload["raster_results"]:
        identity = row.get("proposal_id") if isinstance(row, dict) else None
        if not isinstance(identity, str) or identity not in proposal_ids:
            unknown.append(row)
        else:
            grouped[identity].append(row)
    decisions, observations, rejected_discoveries = [], [], []
    for identity in proposal_ids:
        rows = grouped[identity]
        audit = {"proposal_id": identity, "wire_rows": copy.deepcopy(rows), "status": "unresolved"}
        if len(rows) != 1:
            audit["reason"] = "missing decision" if not rows else "duplicate decisions"
        else:
            row = rows[0]
            try:
                decision, reason = row.get("decision"), row.get("reason")
                if decision not in {"symbol", "reject", "uncertain"} or not isinstance(reason, str) or not reason.strip():
                    raise ValueError("Decision and visible evidence required")
                if decision == "symbol":
                    symbol = _symbol({k: v for k, v in row.items() if k not in {"proposal_id", "decision"}})
                    observations.append({**symbol, "proposal_id": identity})
                elif set(row) != {"proposal_id", "decision", "reason"}:
                    raise ValueError("Reject/uncertain must not also assert a class or box")
                audit.update(status=decision, reason=reason)
            except (ValueError, TypeError) as exc:
                audit["reason"] = str(exc)
        decisions.append(audit)
    for index, row in enumerate(payload["discoveries"]):
        try:
            observations.append({**_symbol(row), "discovery_index": index})
        except (ValueError, TypeError) as exc:
            rejected_discoveries.append({"index": index, "wire_row": row, "reason": str(exc)})
    return observations, {"decisions": decisions, "unrecognized_rows": unknown,
                           "rejected_discoveries": rejected_discoveries}


class BroadRasterClient(LLMClient):
    def begin_raster_view(self, guidance):
        self.raster_proposal_ids = [r["proposal_id"] for r in guidance["raster_proposal_guidance"]["guides"]]
        self.raster_guides_omitted = guidance["raster_proposal_guidance"].get("guides_omitted_by_limit", 0)
        self.raster_review_attempts = []


def perceive_broad_raster(**kwargs):
    if kwargs.get("candidates"):
        raise ValueError("Broad raster observations must not replace supplied native candidates")
    client, policy = kwargs["client"], kwargs["policy"]
    if not hasattr(client, "raster_proposal_ids"):
        raise ValueError("Raster guidance must be prepared before broad review")
    guidance = kwargs["page_context"]["raster_proposal_guidance"]
    prompt = {"raster_guides": guidance["guides"], "guides_omitted_by_limit": guidance["guides_omitted_by_limit"],
              "legend_entries": kwargs.get("legend_summary") or [],
              "legend_scope": "Caller-prepared legend context only; empty means no legend evidence supplied",
              "coordinate_frame": "normalized_to_first_source_image"}
    tool = {"name": "submit_raster_symbols", "description": "Return broad raster observations, not engineering objects",
            "input_schema": broad_schema()}
    observations = None
    for attempt in range(1, 3):
        text = json.dumps(prompt, ensure_ascii=False)
        if attempt == 2:
            text += "\nPrevious output was not a valid tool payload. Call submit_raster_symbols with raster_results and discoveries arrays."
        response = client.messages_create(system=BROAD_SYSTEM, messages=[{"role": "user", "content": [
            encode_image_block(kwargs["view_image"]), {"type": "text", "text": text}]}],
            tools=[tool], tool_choice={"type": "tool", "name": tool["name"]}, max_tokens=policy.first_pass_tokens,
            thinking={"type": "disabled"}, reasoning_mode_override="disabled", output_config={"effort": "low"},
            time_budget_s=policy.request_timeout_s, max_attempts=policy.transport_attempts)
        kwargs["cost_tracker"].record(response, step=kwargs["step"] + attempt - 1,
                                     tile_id=kwargs["tile"].id, page_index=kwargs["page"].page_index)
        raw = [b.input for b in response.content if b.type == "tool_use" and b.name == tool["name"]]
        record = {"response_id": response.id, "proposal_ids": list(client.raster_proposal_ids), "tool_results": []}
        client.raster_review_attempts.append(record)
        try:
            if len(raw) != 1:
                raise ValueError("Exactly one submit_raster_symbols call required")
            record["tool_results"].append({"raw_input": copy.deepcopy(raw[0])})
            observations, audit = normalize_broad(raw[0], client.raster_proposal_ids)
            record["tool_results"][0].update(audit)
            break
        except (ValueError, TypeError) as exc:
            record["format_error"] = str(exc)
            if attempt == 2:
                raise
    assert observations is not None
    batch = PerceptionBatch()
    for index, row in enumerate(observations):
        box = _project_normalized_bbox(NormalizedBBox.model_validate(row["bbox"]), page=kwargs["page"], view_info=kwargs["view_info"])
        if not _bbox_center_is_owned(box, kwargs["ownership_bbox"], kwargs["page"]):
            continue
        attributes = {"broad_category": row["broad_category"], "geometry_basis": "vlm_broad_raster_observation",
                      "raster_vlm_reason": row["reason"], "requires_legend_interpretation": True,
                      "source_tile": kwargs["tile"].id}
        if "proposal_id" in row:
            attributes.update(raster_proposal_id=row["proposal_id"], raster_vlm_decision="symbol")
        batch.candidate_reviews.append({"status": "uncertain", "object_index": index, "bbox": box.model_dump(),
            "object": {"kind": "raster_symbol", "confidence": row["confidence"], "attributes": attributes},
            "reason": "Broad raster observation; detailed legend semantics and geometry require review before graph use."})
    return PerceptionOutcome(detections=[], batch=batch, attempts=attempt)
