import copy

import pytest
from anthropic.types import Message
from PIL import Image, ImageDraw

from diagex.config import SymbolPerceptionConfig
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, Tile
from diagex.vision.perception import (
    PerceptionBatch,
    _expand_candidate_results,
    _tool_input_schema,
    project_batch,
)
from diagex.vision.raster_review import (
    RasterReviewClient,
    normalize_results,
    perceive_with_raster_review,
    review_schema,
)
from diagex.vision.views import ViewInfo


def accepted(identity="d1"):
    return {"proposal_id": identity, "decision": "symbol", "reason": "Two opposing triangles form a valve body",
            "kind": "equipment", "equipment_class": "valve", "bbox": {"x": .1, "y": .2, "w": .2, "h": .1}}


def test_decisions_do_not_turn_rejections_uncertainty_or_unknown_ids_into_symbols():
    wire = {"candidate_results": [], "proposals": [], "raster_results": [accepted(),
            {"proposal_id": "d2", "decision": "reject", "reason": "Only text strokes"},
            {"proposal_id": "d3", "decision": "uncertain", "reason": "Clipped glyph"}, accepted("unknown")]}
    converted, audit = normalize_results(wire, ["d1", "d2", "d3", "d4"])
    assert len(converted["proposals"]) == 1
    assert [r["status"] for r in audit["decisions"]] == ["symbol", "reject", "uncertain", "unresolved"]
    assert audit["unrecognized_rows"][0]["proposal_id"] == "unknown"
    obj = converted["proposals"][0]
    assert "candidate_id" not in obj and obj["attributes"]["raster_proposal_id"] == "d1"


@pytest.mark.parametrize("mutation", ["duplicate", "native_id", "attributes", "invalid_bbox", "contradiction", "missing_kind"])
def test_malformed_or_conflicting_decisions_remain_unresolved(mutation):
    row = accepted()
    if mutation == "native_id":
        row["candidate_id"] = "invented-native"
    elif mutation == "attributes":
        row["attributes"] = {"approved": True}
    elif mutation == "invalid_bbox":
        row["bbox"]["x"] = 1.1
    elif mutation == "contradiction":
        row["decision"] = "reject"
    elif mutation == "missing_kind":
        del row["kind"]
    rows = [row, copy.deepcopy(row)] if mutation == "duplicate" else [row]
    converted, audit = normalize_results({"raster_results": rows}, ["d1"])
    assert not converted["proposals"] and audit["decisions"][0]["status"] == "unresolved"


def test_reviewed_raster_geometry_still_requires_review_and_never_becomes_native():
    converted, _ = normalize_results({"raster_results": [accepted()]}, ["d1"])
    batch = PerceptionBatch.model_validate(_expand_candidate_results(converted))
    box = BBox(x=100, y=50, w=200, h=100)
    page = PageEvidence(page_index=0, source_ref="drawing.png", width=500, height=400,
                        dpi=300, effective_dpi=300, is_scanned=True)
    tile = Tile(id="p0-r0-c0", page_index=0, bbox=box)
    info = ViewInfo(source_view="tile", origin=(100, 50), scale_x=1, scale_y=1,
                    view_size=(200, 100), page_bbox=box, tile_id=tile.id)
    detections = project_batch(batch=batch, page=page, tile=tile, view_info=info,
                               nearby_text_ids=set(), ownership_bbox=box, candidates=[])
    assert detections == [] and len(batch.candidate_reviews) == 1
    review = batch.candidate_reviews[0]
    assert review["bbox"] == {"x": 120, "y": 70, "w": 40, "h": 10}
    assert not review.get("candidate_id")
    assert review["object"]["attributes"]["raster_vlm_decision"] == "symbol"


def test_transport_adapter_preserves_actual_usage_and_raw_decisions(monkeypatch):
    raw = {"candidate_results": [], "raster_results": [accepted()]}
    message = Message.model_validate({"id": "gen-example", "type": "message", "role": "assistant", "model": "model",
                "content": [{"type": "tool_use", "id": "tool1", "name": "submit_pid_objects", "input": raw}],
                "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 60, "cost": .001}})
    seen = []
    monkeypatch.setattr(LLMClient, "messages_create", lambda self, **kw: seen.append(kw) or message)
    client = RasterReviewClient.__new__(RasterReviewClient)
    client.begin_raster_view({"raster_proposal_guidance": {"guides": [{"proposal_id": "d1"}]}})
    schema = _tool_input_schema()
    result = client.messages_create(system="original symbol prompt", tools=[{"name": "submit_pid_objects", "input_schema": schema}])
    assert "raster_results" not in schema["properties"]
    assert "raster_results" in seen[0]["tools"][0]["input_schema"]["required"]
    assert result.usage.model_dump() == message.usage.model_dump()
    assert message.content[0].input == raw
    assert "raster_results" not in result.content[0].input
    assert client.raster_review_attempts[0]["tool_results"][0]["raw_input"] == raw
    assert result.content[0].input["proposals"][0]["attributes"]["raster_proposal_id"] == "d1"


def test_schema_keeps_original_symbol_ontology_and_discovery_contract():
    original = _tool_input_schema()
    schema = review_schema(original)
    assert schema["properties"]["proposals"] == original["properties"]["proposals"]
    assert schema["properties"]["candidate_results"] == original["properties"]["candidate_results"]
    assert schema["properties"]["raster_results"]["items"]["properties"]["kind"] == original["properties"]["proposals"]["items"]["properties"]["kind"]


def test_full_scanned_perception_keeps_accepted_raster_output_in_review(monkeypatch):
    raw = {"candidate_results": [], "raster_results": [accepted()]}
    message = Message.model_validate({"id": "gen-example", "type": "message", "role": "assistant", "model": "model",
                "content": [{"type": "tool_use", "id": "tool1", "name": "submit_pid_objects", "input": raw}],
                "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 60, "cost": .001}})
    monkeypatch.setattr(LLMClient, "messages_create", lambda self, **kw: message)
    client = RasterReviewClient.__new__(RasterReviewClient)
    guidance = {"raster_proposal_guidance": {"guides": [{"proposal_id": "d1"}]}}
    client.begin_raster_view(guidance)
    image = Image.new("RGB", (200, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 60, 30), fill="black")
    box = BBox(x=0, y=0, w=200, h=100)
    page = PageEvidence(page_index=0, source_ref="drawing.png", width=200, height=100,
                        dpi=300, effective_dpi=300, is_scanned=True)
    tile = Tile(id="p0-r0-c0", page_index=0, bbox=box, image=image)
    info = ViewInfo(source_view="tile", origin=(0, 0), scale_x=1, scale_y=1,
                    view_size=(200, 100), page_bbox=box, tile_id=tile.id)
    outcome = perceive_with_raster_review(client=client, cost_tracker=CostTracker(), reporter=NullReporter(),
        page=page, tile=tile, view_image=image, view_info=info, ownership_bbox=box, legend_summary=[], step=1,
        page_context=guidance, candidates=[], policy=SymbolPerceptionConfig())
    assert outcome.detections == []
    assert len(outcome.batch.candidate_reviews) == 1
    review = outcome.batch.candidate_reviews[0]
    assert review["bbox"] == {"x": 20, "y": 20, "w": 40, "h": 10}
    assert review["object"]["attributes"]["raster_proposal_id"] == "d1"
    assert review["status"] == "uncertain" and "require review" in review["reason"]
