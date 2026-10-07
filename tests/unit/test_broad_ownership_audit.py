import copy
import json

import pytest

from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox
from diagex.vision.raster_broad import normalize_broad
from diagex.vision.views import ViewInfo
from eval.pid2graph.broad_ownership_audit import audit_signature, replay, verify_predictions


def fixture():
    def symbol(identity, x):
        return {"proposal_id": identity, "decision": "symbol", "broad_category": "arrow",
                "confidence": "high", "bbox": {"x": x, "y": .2, "w": .1, "h": .1},
                "reason": "Visible filled triangle"}
    payload = {"raster_results": [symbol("inside", .1), symbol("outside", .8)],
               "discoveries": [{"broad_category": "arrow", "confidence": "medium",
                   "bbox": {"x": 1, "y": .2, "w": .1, "h": .1}, "reason": "Invalid bounds"}]}
    _, audit = normalize_broad(payload, ["inside", "outside"])
    raw = {"tile_id": "p0-r0-c0", "raster_review_attempts": [{"proposal_ids": ["inside", "outside"],
            "tool_results": [{"raw_input": payload, **audit}]}],
           "outcome": {"candidate_reviews": [{"object_index": 0,
               "bbox": {"x": 10, "y": 20, "w": 10, "h": 10},
               "object": {"kind": "raster_symbol", "confidence": "high",
                   "attributes": {"broad_category": "arrow", "raster_proposal_id": "inside"}}}]},
           "predictions": [{"id": "p0-r0-c0:proposal:0", "label": "arrow",
               "bbox": [20, 40, 40, 60], "confidence": .9, "disposition": "review_proposal",
               "attributes": {"broad_category": "arrow", "raster_proposal_id": "inside",
                              "equipment_class": None, "valve_type": None, "actuation": None}}]}
    page = PageEvidence(page_index=0, source_ref="fixture.png", width=100, height=100,
                        dpi=72, effective_dpi=72, is_scanned=True)
    info = ViewInfo(source_view="tile", origin=(0, 0), scale_x=1, scale_y=1,
                    view_size=(100, 100), page_bbox=BBox(x=0, y=0, w=100, h=100), tile_id="p0-r0-c0")
    return json.loads(json.dumps(raw, sort_keys=True)), dict(page=page, info=info, core=BBox(x=0, y=0, w=50, h=100))


def test_roundtrip_error_repr_does_not_change_ownership():
    raw, geometry = fixture()
    rows = replay(raw, **geometry)
    assert [(r["proposal_id"], r["owned"]) for r in rows] == [("inside", True), ("outside", False)]
    verify_predictions(raw, 2, 2)


@pytest.mark.parametrize("mutation", ["box", "class", "source_id"])
def test_replay_refuses_changed_saved_review(mutation):
    raw, geometry = fixture()
    review = raw["outcome"]["candidate_reviews"][0]
    if mutation == "box":
        review["bbox"]["x"] += 1
    elif mutation == "class":
        review["object"]["attributes"]["broad_category"] = "valve"
    else:
        review["object"]["attributes"]["raster_proposal_id"] = "outside"
    with pytest.raises(ValueError, match="saved broad reviews"):
        replay(raw, **geometry)


def test_saved_prediction_change_is_detected_after_review_parity():
    raw, geometry = fixture()
    replay(raw, **geometry)
    raw["predictions"][0]["bbox"][0] += 1
    with pytest.raises(ValueError, match="Saved predictions differ"):
        verify_predictions(raw, 2, 2)


def test_semantic_decisions_and_rejected_wire_rows_are_not_ignored():
    raw, _ = fixture()
    audit = raw["raster_review_attempts"][0]["tool_results"][0]
    changed = copy.deepcopy(audit)
    changed["decisions"][0]["status"] = "reject"
    assert audit_signature(audit) != audit_signature(changed)
    changed = copy.deepcopy(audit)
    changed["rejected_discoveries"][0]["wire_row"]["bbox"]["x"] = .5
    assert audit_signature(audit) != audit_signature(changed)


def test_multiple_accepted_payloads_cannot_be_silently_replayed():
    raw, geometry = fixture()
    raw["raster_review_attempts"].append(copy.deepcopy(raw["raster_review_attempts"][0]))
    with pytest.raises(ValueError, match="exactly one"):
        replay(raw, **geometry)
