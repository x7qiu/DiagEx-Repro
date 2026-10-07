"""Callout guidance reaches requests without imposing tag or reference whitelists.

Mocked responses verify the response contract; recognition accuracy is measured
separately on source drawings with paired model calls.
"""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from diagex.config import SymbolPerceptionConfig
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.models import BBox, Tile
from diagex.vision.symbol_interpretation import perceive_tile
from tests.unit.test_port_topology import page
from tests.unit.test_symbol_perception import circle, symbol_candidates, valve, view
from tests.unit.test_visual_knowledge import message


def run_response(source, candidates, payload, knowledge):
    calls = []
    client = SimpleNamespace(
        messages_create=lambda **kwargs: (
            calls.append(kwargs) or message("submit_pid_objects", payload)
        )
    )
    box = BBox(x=0, y=0, w=source.width, h=source.height)
    context = {
        "identity": "valve-only-library-selection",
        "references": [{"id": "needle-valve", "version": 1,
                        "reference_type": "symbol", "source": {}}],
    } if knowledge else {}
    outcome = perceive_tile(
        client=client,
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=source,
        tile=Tile(id="test", page_index=0, bbox=box),
        view_image=Image.new("RGB", (500, 320), "white"),
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[],
        candidates=candidates,
        page_context={"knowledge": context},
        policy=SymbolPerceptionConfig(workflow="fixed"),
        reasoning_mode="disabled",
        step=1,
    )
    assert len(calls) == 1
    system = " ".join(calls[0]["system"].split())
    assert "enclosed identity callout" in system
    assert "leader attachment, repeated local drawing convention" in system
    assert "Real instrument or function bubbles remain symbols" in system
    assert "Neither a connecting stroke nor particular tag letters alone" in system
    assert "partial selection, not an inventory or whitelist" in system
    assert "Retain a visible symbol that has no reference match" in system
    model_context = json.loads(calls[0]["messages"][0]["content"][1]["text"].split("\n")[-1])
    refs = model_context["page_overview_context"]["knowledge"].get("references", [])
    assert [r["id"] for r in refs] == (["needle-valve"] if knowledge else [])
    return outcome


@pytest.mark.parametrize("knowledge", [False, True])
def test_native_callout_rejection_preserves_instruments_same_prefix_and_ambiguity(knowledge):
    source = page([
        circle("callout", 150, 150), circle("instrument", 250, 150),
        circle("same-prefix", 350, 150), circle("ambiguous", 450, 150),
        *valve("body", 550, 150),
    ])
    candidates = symbol_candidates(source)
    by_x = {round(c.bbox.x + c.bbox.w / 2): c for c in candidates}
    rows = [
        {"candidate_id": by_x[150].id, "decision": "reject_annotation",
         "reason": "Enclosed identity tag and leader identify the separate visible valve body."},
        {"candidate_id": by_x[250].id, "decision": "symbol", "kind": "instrument",
         "printed_tag": "LS 004", "recognition_evidence": "Independent instrument glyph."},
        {"candidate_id": by_x[350].id, "decision": "symbol", "kind": "instrument",
         "printed_tag": "V 008", "recognition_evidence": "Local legend defines this instrument."},
        {"candidate_id": by_x[450].id, "decision": "uncertain",
         "reason": "Cannot distinguish independent bubble from identification callout."},
        {"candidate_id": by_x[560].id, "decision": "symbol", "kind": "equipment",
         "equipment_class": "valve", "printed_tag": "XV 007",
         "recognition_evidence": "Valve body and leader support the identity association."},
    ]
    outcome = run_response(source, candidates, {"candidate_results": rows}, knowledge)
    detections = {d.label: d for d in outcome.detections}
    assert set(detections) == {"LS 004", "V 008", "XV 007"}
    assert detections["V 008"].kind == "instrument"  # No universal V-prefix rejection.
    assert detections["XV 007"].bbox == by_x[560].bbox
    reviews = {r["candidate_id"]: r for r in outcome.batch.candidate_reviews}
    assert reviews[by_x[150].id]["status"] == "reject"
    assert reviews[by_x[450].id]["status"] == "uncertain"
    assert not outcome.contract_failed


@pytest.mark.parametrize("knowledge", [False, True])
def test_raster_symbols_without_reference_match_remain_source_supported(knowledge):
    source = page([])
    source.is_scanned = True
    payload = {
        "candidate_results": [],
        "proposals": [
            {"kind": "instrument", "printed_tag": "LA 004",
             "bbox": {"x": .1, "y": .2, "w": .1, "h": .1},
             "recognition_evidence": "Hexagonal function symbol and visible tag."},
            {"kind": "equipment", "equipment_class": "filter", "printed_tag": "FL-399",
             "bbox": {"x": .4, "y": .4, "w": .15, "h": .1},
             "recognition_evidence": "Visible crosshatched filter body."},
        ],
    }
    outcome = run_response(source, [], payload, knowledge)
    assert {d.label for d in outcome.detections} == {"LA 004", "FL-399"}
    for detection in outcome.detections:
        assert "knowledge_match" not in detection.attributes
        assert "legend_matches" not in detection.attributes
        assert bool(detection.attributes.get("supplied_knowledge")) == knowledge
