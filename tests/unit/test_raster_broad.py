import json

import pytest
from anthropic.types import Message
from PIL import Image

from diagex.config import SymbolPerceptionConfig
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, Tile
from diagex.vision.raster_broad import BroadRasterClient, normalize_broad, perceive_broad_raster
from diagex.vision.views import ViewInfo
from eval.pid2graph.raster_broad_runner import record


def symbol(**extra):
    return {"proposal_id": "d1", "decision": "symbol", "broad_category": "arrow", "confidence": "high",
            "bbox": {"x": .1, "y": .2, "w": .2, "h": .1}, "reason": "Filled triangle on the pipe points downward", **extra}


@pytest.mark.parametrize("rows", [[symbol(), symbol()], [symbol(candidate_id="fake-native")], [symbol(equipment_class="pump")]])
def test_broad_recognition_does_not_accept_duplicates_or_engineering_injections(rows):
    observations, audit = normalize_broad({"raster_results": rows, "discoveries": []}, ["d1"])
    assert observations == [] and audit["decisions"][0]["status"] == "unresolved"


def test_missing_geometry_stays_unresolved_and_unknown_ids_do_not_leak_into_output():
    row = symbol()
    del row["bbox"]
    observations, audit = normalize_broad({"raster_results": [row, symbol(proposal_id="unknown")], "discoveries": []}, ["d1"])
    assert observations == [] and len(audit["unrecognized_rows"]) == 1
    assert audit["decisions"][0]["status"] == "unresolved"


def arguments(client):
    box = BBox(x=100, y=50, w=200, h=100)
    page = PageEvidence(page_index=0, source_ref="source.png", width=500, height=400,
                        dpi=300, effective_dpi=300, is_scanned=True)
    tile = Tile(id="p0-r0-c0", page_index=0, bbox=box)
    info = ViewInfo(source_view="tile", origin=(100, 50), scale_x=1, scale_y=1,
                    view_size=(200, 100), page_bbox=box, tile_id=tile.id)
    context = {"raster_proposal_guidance": {"guides": [{"proposal_id": "d1"}], "guides_omitted_by_limit": 0}}
    client.begin_raster_view(context)
    return dict(client=client, page=page, tile=tile, view_info=info, ownership_bbox=box,
        view_image=Image.new("RGB", (200, 100), "white"), candidates=[], page_context=context,
        legend_summary=[{"id": "legend-reviewed", "label": "source definition"}],
        cost_tracker=CostTracker(), policy=SymbolPerceptionConfig(), step=1)


def response(payload):
    return Message.model_validate({"id": "gen-test", "type": "message", "role": "assistant", "model": "model",
        "content": [{"type": "tool_use", "id": "tool1", "name": "submit_raster_symbols", "input": payload}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 60}})


def test_arrow_remains_broad_review_observation_with_correct_source_geometry(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(LLMClient, "messages_create", lambda self, **kw: seen.append(kw) or response({"raster_results": [symbol()], "discoveries": []}))
    client = BroadRasterClient.__new__(BroadRasterClient)
    args = arguments(client)
    outcome = perceive_broad_raster(**args)
    assert outcome.detections == [] and outcome.batch.objects == []
    review = outcome.batch.candidate_reviews[0]
    assert review["object"]["kind"] == "raster_symbol"
    assert review["bbox"] == {"x": 120, "y": 70, "w": 40, "h": 10}
    attrs = review["object"]["attributes"]
    assert attrs["broad_category"] == "arrow" and attrs["requires_legend_interpretation"]
    prediction = record("p", "raster_symbol", attrs, review["bbox"], "high", 2, 2, "review_proposal")
    assert prediction["label"] == "arrow" and prediction["bbox"] == [240, 140, 320, 160]
    assert "candidate_id" not in review and not attrs.get("source_path_ids")
    request_context = json.loads(seen[0]["messages"][0]["content"][1]["text"])
    assert request_context["legend_entries"] == args["legend_summary"]

    args["page"].role = "pid"


def test_broad_wire_format_recovery_is_bounded_and_audited(monkeypatch):
    payloads = iter([{"wrong": []}, {"raster_results": [], "discoveries": []}])
    monkeypatch.setattr(LLMClient, "messages_create", lambda self, **kw: response(next(payloads)))
    client = BroadRasterClient.__new__(BroadRasterClient)
    outcome = perceive_broad_raster(**arguments(client))
    assert outcome.attempts == 2 and len(client.raster_review_attempts) == 2
    assert client.raster_review_attempts[0]["format_error"]
    assert client.raster_review_attempts[1]["tool_results"][0]["decisions"][0]["status"] == "unresolved"


def test_native_candidates_cannot_be_replaced_by_the_broad_raster_path():
    client = BroadRasterClient.__new__(BroadRasterClient)
    args = arguments(client)
    args["candidates"] = [object()]
    with pytest.raises(ValueError, match="native candidates"):
        perceive_broad_raster(**args)
