"""Explicit VLM review of fallible raster proposals, without native identities.

Accepted responses remain geometry-review proposals in the existing perception
pipeline. This contract does not confer engineering approval or invent legends.
"""
from __future__ import annotations

import copy
from collections import defaultdict

from diagex.llm.client import LLMClient
from diagex.vision.perception import PerceivedObject

INSTRUCTION = """
Additional raster review contract: raster_proposal_guidance contains fallible
detector search hints, separate from native_symbol_candidates. Return exactly
one raster_results row for every supplied proposal_id. Copy that proposal_id
exactly; never put it in candidate_id. Use decision="symbol" only after checking
the original source ink. Include a short reason naming the visible strokes,
kind and supported classification fields, and a tight bbox normalized to this
detail image. Correct the hint's class and bounds when the image warrants it.
For decision="reject" or "uncertain", return proposal_id, decision, and reason
only. Reject based on visible non-symbol evidence, not engineering plausibility.
Clipped or ambiguous glyphs may be uncertain. Missing a legend is not evidence
that a glyph does not exist; leave detailed subtype fields unset when unsupported.
Do not repeat an accepted raster result in proposals. Continue searching the
whole image for symbols missed by the detector; return those in proposals using
the existing schema. Keep candidate_results reserved for real native candidates.
For supplied raster hints, raster_results replaces the earlier instruction to
return them through proposals; proposals is only for additional discoveries.
Raster results do not create native vector evidence or approve graph objects.
"""


def review_schema(original):
    schema = copy.deepcopy(original)
    properties = copy.deepcopy(schema["properties"]["proposals"]["items"]["properties"])
    schema["properties"]["raster_results"] = {
        "type": "array", "items": {"type": "object", "properties": {
            "proposal_id": {"type": "string", "minLength": 1},
            "decision": {"type": "string", "enum": ["symbol", "reject", "uncertain"]},
            "reason": {"type": "string", "minLength": 1}, **properties},
            "required": ["proposal_id", "decision", "reason"], "additionalProperties": False}}
    schema["required"] = [*schema.get("required", []), "raster_results"]
    return schema


def normalize_results(payload, proposal_ids):
    """Return compatible proposals and an exhaustive per-ID audit of the wire data."""
    allowed = set(proposal_ids)
    if len(allowed) != len(proposal_ids):
        raise ValueError("Input raster proposal IDs must be unique")
    if not isinstance(payload, dict) or set(payload) - {"candidate_results", "proposals", "raster_results"}:
        raise ValueError("Invalid raster review top-level response")
    raw = payload.get("raster_results")
    if not isinstance(raw, list):
        raw = []
    rows = defaultdict(list)
    unrecognized = []
    for item in raw:
        identity = item.get("proposal_id") if isinstance(item, dict) else None
        if not isinstance(identity, str) or identity not in allowed:
            unrecognized.append(item)
        else:
            rows[identity].append(item)
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list):
        raise ValueError("Discovery proposals must be an array")
    converted = {"candidate_results": payload.get("candidate_results", []), "proposals": copy.deepcopy(proposals)}
    decisions = []
    for identity in proposal_ids:
        items = rows[identity]
        audit = {"proposal_id": identity, "wire_rows": copy.deepcopy(items), "status": "unresolved"}
        if len(items) != 1:
            audit["reason"] = "missing decision" if not items else "duplicate decisions"
        else:
            item = items[0]
            decision = item.get("decision")
            reason = item.get("reason")
            try:
                if decision not in {"symbol", "reject", "uncertain"} or not isinstance(reason, str) or not reason.strip():
                    raise ValueError("Decision and visible-evidence reason required")
                classification = {k: v for k, v in item.items() if k not in {"proposal_id", "decision", "reason"}}
                if decision == "symbol":
                    # Reject any attempt to claim native identity or inject attributes.
                    fields = set(PerceivedObject.model_fields) - {"candidate_id", "attributes"}
                    if set(classification) - fields or "kind" not in classification or "bbox" not in classification:
                        raise ValueError("Accepted raster symbol requires kind/bbox and supported fields only")
                    obj = PerceivedObject.model_validate(classification)
                    if obj.bbox is None:
                        raise ValueError("Raster symbol needs its own image box")
                    value = obj.model_dump(mode="json", exclude_none=True)
                    value["attributes"] = {"raster_proposal_id": identity, "raster_vlm_decision": "symbol",
                                           "raster_vlm_reason": reason, "geometry_basis": "vlm_reviewed_raster_proposal"}
                    audit["converted_proposal_index"] = len(converted["proposals"])
                    converted["proposals"].append(value)
                elif classification:
                    raise ValueError("Reject/uncertain rows must not also classify a symbol")
                audit.update(status=decision, reason=reason)
            except (ValueError, TypeError) as exc:
                audit["reason"] = str(exc)
        decisions.append(audit)
    return converted, {"decisions": decisions, "unrecognized_rows": unrecognized,
                       "raster_results_present_as_array": isinstance(payload.get("raster_results"), list)}


class RasterReviewClient(LLMClient):
    """Use the normal metered transport, then adapt its audited review responses."""

    def begin_raster_view(self, guidance):
        self.raster_proposal_ids = [r["proposal_id"] for r in guidance["raster_proposal_guidance"]["guides"]]
        self.raster_guides_omitted = guidance["raster_proposal_guidance"].get("guides_omitted_by_limit", 0)
        self.raster_review_attempts = []

    def messages_create(self, **kwargs):
        tools = kwargs.get("tools", [])
        if len(tools) != 1 or tools[0].get("name") != "submit_pid_objects":
            return super().messages_create(**kwargs)
        if not hasattr(self, "raster_proposal_ids"):
            raise ValueError("Raster review requires source-bound view guidance")
        kwargs = {**kwargs, "tools": copy.deepcopy(tools)}
        kwargs["tools"][0]["input_schema"] = review_schema(tools[0]["input_schema"])
        system = kwargs["system"]
        kwargs["system"] = system + "\n" + INSTRUCTION if isinstance(system, str) else [*system, {"type": "text", "text": INSTRUCTION}]
        response = super().messages_create(**kwargs)
        # Billing is settled against the actual response before any adaptation.
        response = response.model_copy(deep=True)
        attempt = {"response_id": response.id, "proposal_ids": list(self.raster_proposal_ids), "tool_results": []}
        self.raster_review_attempts.append(attempt)
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_pid_objects":
                raw = copy.deepcopy(block.input)
                entry = {"raw_input": raw}
                attempt["tool_results"].append(entry)
                block.input, audit = normalize_results(raw, self.raster_proposal_ids)
                entry.update(audit)
        return response


def perceive_with_raster_review(**kwargs):
    """Keep reviewed raster hints out of the native/graph detection inventory."""
    from diagex.vision.perception import perceive_tile

    outcome = perceive_tile(**kwargs)
    objects = {o.attributes.get("raster_proposal_id"): o for o in outcome.batch.objects
               if o.attributes.get("raster_proposal_id")}
    retained = []
    for detection in outcome.detections:
        identity = detection.attributes.get("raster_proposal_id")
        if not identity:
            retained.append(detection)
            continue
        obj = objects.get(identity)
        if obj is None:
            raise ValueError("Raster detection lost its source review object")
        outcome.batch.candidate_reviews.append({
            "status": "uncertain", "bbox": detection.bbox.model_dump(),
            "object": obj.model_dump(mode="json"),
            "reason": "VLM recognized a raster proposal; source geometry and engineering interpretation still require review.",
        })
    outcome.detections = retained
    return outcome
