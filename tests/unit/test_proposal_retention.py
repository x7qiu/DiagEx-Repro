import copy

import pytest

from eval.pid2graph.proposal_retention import compare
from eval.pid2graph.scoring import score_drawing


def prediction(identity, x, label):
    return {"id": identity, "bbox": [x, 0, x + 10, 10], "confidence": 0.9, "label": label}


def reports():
    truth = [prediction(str(i), i * 20, "valve") for i in range(4)]
    graph = {"nodes": truth}
    # Detector finds 0,1,2 correctly; misclassifies 3.
    detector = [prediction(str(i), i * 20, "valve" if i < 3 else "pump") for i in range(4)]
    # Hybrid retains 0, misclassifies 1, misses 2, and fixes 3.
    hybrid = [prediction(str(i), i * 20, "pump" if i == 1 else "valve") for i in [0, 1, 3]]
    result = []
    for variant, predictions in [("supervised_detector", detector), ("supervised_guidance", hybrid)]:
        cases = []
        # Repeated truth IDs across drawings must not be collapsed globally.
        for identity in ["drawing1", "drawing2"]:
            case = score_drawing(graph, predictions)
            case.update(id=identity, collection="synthetic", status="complete")
            cases.append(case)
        result.append({"manifest_sha256": "m", "panel_sha256": "p", "split": "validation",
                       "config": {"variant": variant}, "cases": cases, "summary": {}, "runtime_seconds": 1})
    return result


def test_classification_loss_localization_loss_and_recovery_are_separate():
    result = compare(*reports())
    counts = result["summary"]["counts"]
    assert counts == {"detector_true_positives": 6, "hybrid_true_positives": 4,
                      "retained_detector_truth": 2, "lost_detector_truth": 4,
                      "lost_but_still_localized": 2, "lost_without_localization": 2,
                      "recovered_beyond_detector": 2, "corrected_detector_class_errors": 2}
    assert result["summary"]["class_aware_retention"] == 1 / 3


def test_failed_attempts_stay_in_denominator_but_unfinished_drawings_are_not_complete():
    detector, hybrid = reports()
    hybrid["cases"][0]["status"] = "partial"
    assert compare(detector, hybrid)["summary"]["drawings_with_api_failures"] == 1
    hybrid["cases"][0]["status"] = "missing"
    with pytest.raises(ValueError, match="attempted coverage"):
        compare(detector, hybrid)
    altered = copy.deepcopy(detector)
    altered["panel_sha256"] = "different"
    with pytest.raises(ValueError, match="panel_sha256"):
        compare(altered, hybrid)
