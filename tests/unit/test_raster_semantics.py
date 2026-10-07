import base64
import copy
import io
import json
import time
from types import SimpleNamespace

import pytest
from anthropic.types import Message
from PIL import Image

from diagex.config import SymbolPerceptionConfig
from diagex.llm.cost import CostTracker
from diagex.vision.evidence import PageEvidence
from diagex.vision.legend_context import select_legend_context
from diagex.vision.models import BBox
from diagex.vision.raster_semantics import interpret_raster_reviews, normalize_semantics


def legend():
    stream = io.BytesIO()
    Image.new("RGB", (30, 20), "black").save(stream, format="PNG")
    return [{"source_row_id": "legend-gate", "label": "Gate valve", "kind": "valve",
             "symbol_class": "gate_valve", "source": "legend_extracted",
             "image_b64": base64.b64encode(stream.getvalue()).decode(),
             "attributes": {"row_status": "accept"}}]


def observation():
    return {"status": "uncertain", "bbox": {"x": 120, "y": 70, "w": 40, "h": 20},
            "object": {"kind": "raster_symbol", "confidence": "high", "attributes": {
                "broad_category": "valve", "requires_legend_interpretation": True,
                "geometry_basis": "vlm_broad_raster_observation", "raster_proposal_id": "detector-1"}}}


def decision(identity="symbol-0"):
    return {"symbol_id": identity, "decision": "interpreted", "reason": "Opposed wedges match the gate legend",
            "legend_entry_ids": ["legend-gate"], "symbol": {"kind": "equipment", "valve_type": "gate"}}


def message(payload):
    return Message.model_validate({"id": "gen-semantic", "type": "message", "role": "assistant", "model": "test",
        "content": [{"type": "tool_use", "id": "tool1", "name": "submit_raster_semantics", "input": payload}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 50, "cost": .001}})


def arguments(client, reviews=None):
    return dict(reviews=reviews or [observation()], client=client, cost_tracker=CostTracker(),
        page=PageEvidence(page_index=0, source_ref="test", width=500, height=400, dpi=300, effective_dpi=300, role="pid", is_scanned=True),
        tile=SimpleNamespace(id="t1"), view_image=Image.new("RGB", (200, 100), "white"),
        view_info=SimpleNamespace(page_bbox=BBox(x=100, y=50, w=200, h=100)),
        legend_entries=legend(), policy=SymbolPerceptionConfig())


@pytest.mark.parametrize("mutation", ["bbox", "native_id", "attributes", "unknown_legend", "missing_legend", "wrong_kind", "conflicting_fields", "duplicate", "missing", "blank_reason"])
def test_invalid_semantics_cannot_modify_geometry_or_become_a_suggestion(mutation):
    value = decision()
    if mutation in {"bbox", "native_id", "attributes"}:
        value["symbol"][{"bbox": "bbox", "native_id": "candidate_id", "attributes": "attributes"}[mutation]] = {}
    elif mutation == "unknown_legend":
        value["legend_entry_ids"] = ["invented"]
    elif mutation == "missing_legend":
        value["legend_entry_ids"] = []
    elif mutation == "wrong_kind":
        value["symbol"] = {"kind": "instrument", "instrument_function": "transmitter"}
    elif mutation == "conflicting_fields":
        value["symbol"]["instrument_function"] = "transmitter"
    elif mutation == "blank_reason":
        value["reason"] = " "
    rows = [] if mutation == "missing" else [value, value] if mutation == "duplicate" else [value]
    compact, _, _ = select_legend_context(legend(), [], [])
    results, unknown = normalize_semantics({"results": rows}, ["symbol-0"], compact)
    assert not unknown and len(results) == 1
    assert results[0]["decision"] == "unresolved" and "symbol" not in results[0]


def test_success_includes_real_legend_crop_and_preserves_classified_geometry(tmp_path):
    calls = []
    client = SimpleNamespace(messages_create=lambda **kw: calls.append(kw) or message({"results": [decision()]}))
    args = arguments(client)
    original = copy.deepcopy(args["reviews"])
    reviews, audit = interpret_raster_reviews(**args)
    assert args["reviews"] == original and reviews[0]["bbox"] == original[0]["bbox"]
    assert reviews[0]["object"] == original[0]["object"]
    assert audit["attempts"][0]["response_id"] == "gen-semantic"
    assert len(args["cost_tracker"].steps) == 1
    blocks = calls[0]["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in blocks) == 2
    context = json.loads(blocks[-1]["text"])
    assert context["symbols"][0]["bbox_in_source_view"] == {"x": .1, "y": .2, "w": .2, "h": .2}
    from diagex.vision.raster_semantics import interpreted_detection
    from diagex.vision.symbol_interpretation import DetectionRecord
    raw = {**reviews[0]["object"], "id":"machine-1", "page_index":0,
           "tile_id":"t1", "bbox":reviews[0]["bbox"]}
    detection = DetectionRecord.model_validate(interpreted_detection(raw, reviews[0]["legend_interpretation"]))
    assert detection.kind == "equipment"
    assert detection.bbox.model_dump() == original[0]["bbox"]
    assert detection.attributes["valve_type"] == "gate"
    assert detection.attributes["legend_entry_ids"] == ["legend-gate"]
    assert detection.attributes["legend_matches"][0]["source"] == "legend_extracted"
    assert detection.attributes["recognition_method"] == "vlm"


@pytest.mark.parametrize("reason", ["deadline", "no_observations"])
def test_deadline_or_no_observations_preserves_inventory_without_api_use(reason):
    def forbidden(**kwargs):
        raise AssertionError("No API call expected")
    args = arguments(SimpleNamespace(messages_create=forbidden))
    if reason == "no_observations":
        args["reviews"] = []
    else:
        args["deadline"] = time.monotonic() - 1
    reviews, audit = interpret_raster_reviews(**args)
    assert not audit["attempts"] and not args["cost_tracker"].steps
    if reviews:
        assert reviews[0]["legend_interpretation"]["decision"] == "unresolved"
        assert reviews[0]["object"] == args["reviews"][0]["object"]


def source_decision(symbol=None):
    return {
        "symbol_id": "symbol-0", "decision": "interpreted", "legend_entry_ids": [],
        "reason": "The source supports a physical component without an applicable reference match.",
        "source_evidence": "Closed body with a cross-hatched mesh and flanged connections; FL-399 printed above.",
        "symbol": symbol or {"kind": "equipment", "equipment_class": "filter", "printed_tag": "FL-399"},
    }


def test_flattened_live_response_fields_stay_invalid_until_model_resubmits_nested_symbol():
    nested = source_decision()
    flattened = copy.deepcopy(nested)
    flattened.update(flattened.pop("symbol"))
    bad, _ = normalize_semantics({"results": [flattened]}, ["symbol-0"], [])
    assert bad[0]["decision"] == "unresolved" and "symbol" not in bad[0]
    good, _ = normalize_semantics({"results": [nested]}, ["symbol-0"], [])
    assert good[0]["decision"] == "interpreted"
    assert good[0]["symbol"] == nested["symbol"]


@pytest.mark.parametrize("reference_state", ["none", "rejected", "unmatched"])
@pytest.mark.parametrize("symbol", [
    {"kind": "equipment", "equipment_class": "filter", "printed_tag": "FL-399"},
    {"kind": "instrument", "instrument_function": "unclassified_instrument", "printed_tag": "LA 004"},
    {"kind": "equipment", "equipment_class": "unclassified_equipment", "structural_description": "Closed body with internal mesh"},
])
def test_source_supported_unmatched_symbols_preserve_geometry_and_truthful_provenance(reference_state, symbol):
    from diagex.vision.raster_semantics import interpreted_detection
    from diagex.web.evidence_origin import evidence_origin

    row = source_decision(symbol)
    if symbol["kind"] == "instrument":
        row["source_evidence"] = "Distinct hexagonal glyph with horizontal divider; LA above 004 inside the glyph."
    calls = []
    client = SimpleNamespace(messages_create=lambda **kw: calls.append(kw) or message({"results": [row]}))
    args = arguments(client)
    if reference_state == "none":
        args["legend_entries"] = []
    elif reference_state == "rejected":
        args["legend_entries"][0]["attributes"]["row_status"] = "reject"
    original = copy.deepcopy(args["reviews"])
    reviews, audit = interpret_raster_reviews(**args)
    assert len(calls) == len(audit["attempts"]) == len(args["cost_tracker"].steps) == 1
    assert original == args["reviews"] and reviews[0]["object"] == original[0]["object"]
    interpretation = reviews[0]["legend_interpretation"]
    assert interpretation["decision"] == "interpreted" and not interpretation["legend_matches"]
    assert "knowledge_match" not in interpretation
    raw = {**reviews[0]["object"], "bbox": reviews[0]["bbox"]}
    detection = interpreted_detection(raw, interpretation)
    assert detection["bbox"] == original[0]["bbox"]
    assert detection["attributes"]["raster_proposal_id"] == "detector-1"
    assert detection["attributes"]["recognition_evidence"] == row["source_evidence"]
    assert detection["label"] == symbol.get("printed_tag", "")
    origin = evidence_origin(detection)
    assert len(origin["sources"]) == 1 and origin["sources"][0]["kind"] == "other"
    assert origin["sources"][0]["evidence"] == row["source_evidence"]


@pytest.mark.parametrize("mutation", ["blank_evidence", "missing_symbol", "unknown_legend", "unknown_knowledge", "geometry"])
def test_source_fallback_does_not_rescue_invalid_interpretations(mutation):
    row = source_decision()
    if mutation == "blank_evidence":
        row["source_evidence"] = " \n "
    elif mutation == "missing_symbol":
        row.pop("symbol")
    elif mutation == "unknown_legend":
        row["legend_entry_ids"] = ["invented"]
    elif mutation == "unknown_knowledge":
        row.update(knowledge_reference_id="invented", knowledge_evidence="Visible source shape")
    else:
        row["symbol"]["bbox"] = {"x": 0, "y": 0, "w": 1, "h": 1}
    result, _ = normalize_semantics({"results": [row]}, ["symbol-0"], [])
    assert result[0]["decision"] == "unresolved" and "symbol" not in result[0]


@pytest.mark.parametrize("decision_value,reason", [
    ("non_node", "Enclosed tag and leader identify a separate visible valve body; this box is the identity callout."),
    ("non_node", "Only a flow-direction arrow on the line is visible."),
    ("unresolved", "This box groups multiple glyphs; no single coherent physical symbol."),
    ("unresolved", "Only printed text is legible; no distinct physical glyph can be established."),
    ("unresolved", "Explicit drawing definitions conflict for this ambiguous outline."),
])
def test_source_only_request_does_not_promote_uncertainty_or_non_objects(decision_value, reason):
    row = {"symbol_id": "symbol-0", "decision": decision_value, "reason": reason, "legend_entry_ids": []}
    args = arguments(SimpleNamespace(messages_create=lambda **kw: message({"results": [row]})))
    args["legend_entries"] = []
    reviews, audit = interpret_raster_reviews(**args)
    assert len(audit["attempts"]) == 1
    assert reviews[0]["legend_interpretation"]["decision"] == decision_value
    assert "symbol" not in reviews[0]["legend_interpretation"]
    assert reviews[0]["status"] == "uncertain" and reviews[0]["object"] == args["reviews"][0]["object"]


def test_source_prompt_requires_own_glyph_and_legacy_rows_are_not_promoted():
    from diagex.vision.raster_semantics import SYSTEM

    assert "identity callouts" in SYSTEM and "A tag alone is insufficient" in SYSTEM
    assert "boxes grouping multiple" in SYSTEM and "unclassified_instrument" in SYSTEM
    old = decision()
    old["legend_entry_ids"] = []
    result, _ = normalize_semantics({"results": [old]}, ["symbol-0"], [])
    assert result[0]["decision"] == "unresolved" and "symbol" not in result[0]
    compact, _, _ = select_legend_context(legend(), [], [])
    result, _ = normalize_semantics({"results": [decision()]}, ["symbol-0"], compact)
    assert result[0]["decision"] == "interpreted" and "source_evidence" not in result[0]


def test_transport_failure_retains_all_observations_and_stops_further_chunks():
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("provider unavailable")
    args = arguments(SimpleNamespace(messages_create=fail), [observation() for _ in range(49)])
    reviews, audit = interpret_raster_reviews(**args)
    assert len(calls) == 1 and len(audit["attempts"]) == 1 and len(reviews) == 49
    assert all(r["legend_interpretation"]["decision"] == "unresolved" for r in reviews)
    assert all("provider unavailable" in r["legend_interpretation"]["reason"] for r in reviews)


def test_non_node_is_a_review_suggestion_and_unknown_ids_are_audited():
    payload = {"results": [{"symbol_id": "symbol-0", "decision": "non_node", "reason": "Flow arrow on pipe", "legend_entry_ids": []}, decision("unknown")]}
    args = arguments(SimpleNamespace(messages_create=lambda **kw: message(payload)))
    args["reviews"][0]["object"]["attributes"]["broad_category"] = "arrow"
    reviews, audit = interpret_raster_reviews(**args)
    assert reviews[0]["legend_interpretation"]["decision"] == "non_node"
    assert reviews[0]["status"] == "uncertain" and reviews[0]["object"]["kind"] == "raster_symbol"
    assert len(audit["attempts"][0]["unrecognized_rows"]) == 1


def test_later_chunk_failure_preserves_completed_semantics_and_meters_success():
    calls = []
    def respond(**kw):
        calls.append(kw)
        if len(calls) == 2:
            raise RuntimeError("provider unavailable")
        return message({"results": [decision(f"symbol-{i}") for i in range(24)]})
    args = arguments(SimpleNamespace(messages_create=respond), [observation() for _ in range(49)])
    reviews, audit = interpret_raster_reviews(**args)
    assert len(calls) == len(audit["attempts"]) == 2
    assert len(args["cost_tracker"].steps) == 1
    assert [r["legend_interpretation"]["decision"] for r in reviews] == ["interpreted"] * 24 + ["unresolved"] * 25
    assert all(r["status"] == "unreviewed" for r in reviews[24:])
