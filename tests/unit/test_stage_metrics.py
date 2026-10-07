import pytest

from diagex.vision.stage_metrics import score_stage


def box(x=0):
    return {"x": x, "y": 0, "w": 10, "h": 10}


def test_symbol_class_errors_and_duplicates_are_not_hidden():
    truth = {
        "symbols": [{"bbox": box(), "category": "valve"}, {"bbox": box(30), "category": "pump"}]
    }
    predictions = {
        "detector": {"weights": "fixture"},
        "raster_proposals": [
            {"bbox": [0, 0, 10, 10], "label": "valve"},
            {"bbox": [0, 0, 10, 10], "label": "valve"},
            {"bbox": [30, 0, 40, 10], "label": "tank"},
        ],
    }
    result = score_stage("symbol_detection", truth, predictions)
    assert result["localization"]["tp"] == 2
    assert result["localization"]["fp"] == 1
    assert result["classification_and_localization"]["tp"] == 1
    assert result["classification_and_localization"]["fn"] == 1


def test_text_scoring_counts_wrong_missing_and_extra_characters():
    truth = {"text_spans": [{"bbox": box(), "text": "PT101"}, {"bbox": box(30), "text": "AB"}]}
    prediction = {"text_spans": [{"bbox": box(), "text": "PTI01"}, {"bbox": box(70), "text": "X"}]}
    result = score_stage("text_detection", truth, prediction)
    assert result["localization"]["tp"] == 1
    assert result["spatial_character_errors"] == 4
    assert result["spatial_character_error_rate"] == 4 / 7


def test_line_geometry_accepts_reversed_polyline_with_bounded_tolerance():
    truth = {"paths": [{"points": [[0, 0], [100, 0]]}]}
    prediction = {"paths": [{"points": [[101, 0], [1, 0]]}, {"points": [[0, 30], [100, 30]]}]}
    result = score_stage("line_detection", truth, prediction, line_tolerance=2)
    assert result["polyline_geometry"]["tp"] == 1
    assert result["polyline_geometry"]["fp"] == 1


def test_assignment_and_connections_require_correct_target_and_direction():
    assignment = score_stage(
        "text_assignment",
        {"bindings": [{"text_id": "t", "target_id": "a"}]},
        {"nodes": [{"id": "b", "attributes": {"source_text_ids": ["t"]}}]},
    )
    assert assignment["assignment"]["tp"] == 0
    assert assignment["assignment"]["fp"] == 1
    assert assignment["assignment"]["fn"] == 1
    edge = {"from_node": "a", "to_node": "b", "line_type": "process"}
    wrong = {**edge, "from_node": "b", "to_node": "a"}
    result = score_stage("connection_inference", {"edges": [edge]}, {"edges": [wrong]})
    assert result["directed_typed_edges"]["tp"] == 0


def test_invalid_metric_settings_are_rejected():
    with pytest.raises(ValueError, match="Invalid"):
        score_stage("symbol_detection", {}, {}, iou=0)
