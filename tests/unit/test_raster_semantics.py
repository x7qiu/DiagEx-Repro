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
from diagex.review.detection import DetectionReviewStore, write_detection_bundle
from diagex.vision.evidence import PageEvidence
from diagex.vision.legend_context import select_legend_context
from diagex.vision.legend_models import LegendPack
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


def test_success_includes_real_legend_crop_and_stays_pending_in_review(tmp_path):
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
    reviews[0].update(page_index=0, tile_id="t1")
    write_detection_bundle(tmp_path, source_hash="fixture", pages=[args["page"]], detections=[],
        legend_pack=LegendPack.model_validate({"entries": legend()}), per_page_status={0: "ok"},
        candidates=[], reviews=reviews)
    store = DetectionReviewStore(tmp_path)
    row = store.public()["symbols"][0]
    assert row["status"] == "pending" and row["origin"] == "proposal"
    assert row["detection"]["kind"] == "equipment" and row["detection"]["attributes"]["valve_type"] == "gate"
    assert row["detection"]["bbox"] == original[0]["bbox"]
    assert row["source_observation"]["object"] == original[0]["object"]
    assert store.snapshot(0, draft=True)["detections"] == []
    from diagex.vision.fusion import fuse_objects
    from diagex.vision.perception import DetectionRecord

    def apply(action, **values):
        return store.apply({"revision": store.read()["revision"], "rater": "Test reviewer",
                            "actor_type": "agent", "evidence_refs": ["source-view", "legend-gate"],
                            "action": action, **values})
    with pytest.raises(ValueError, match="Review the legend"):
        apply("save_symbol", id=row["id"], detection=row["detection"])
    apply("confirm_legends", ids=["legend-0"])
    apply("save_symbol", id=row["id"], detection=row["detection"])
    apply("coverage", page_index=0, checked=True)
    snapshot = store.snapshot(store.read()["revision"])
    fused = fuse_objects(source_name="fixture", pages=[args["page"]],
        detections=[DetectionRecord.model_validate(d) for d in snapshot["detections"]],
        per_page_status={0: "ok"}, legend_pack=LegendPack.model_validate(snapshot["legend_pack"]),
        reviewed_instances=True)
    assert len(fused.graph.nodes) == 1
    node = fused.graph.nodes[0]
    assert node.bbox_global.model_dump() == original[0]["bbox"]
    assert node.attributes["valve_type"] == "gate" and node.attributes["human_reviewed"] is False
    assert node.attributes["legend_entry_ids"] == ["legend-gate"]


@pytest.mark.parametrize("reason", ["no_legend", "rejected_legend", "deadline"])
def test_missing_legend_or_deadline_preserves_observations_without_api_use(reason):
    def forbidden(**kwargs):
        raise AssertionError("No API call expected")
    args = arguments(SimpleNamespace(messages_create=forbidden))
    if reason == "no_legend":
        args["legend_entries"] = []
    elif reason == "rejected_legend":
        args["legend_entries"][0]["attributes"]["row_status"] = "reject"
    else:
        args["deadline"] = time.monotonic() - 1
    reviews, audit = interpret_raster_reviews(**args)
    assert not audit["attempts"] and not args["cost_tracker"].steps
    assert reviews[0]["legend_interpretation"]["decision"] == "unresolved"
    assert reviews[0]["object"] == args["reviews"][0]["object"]


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
