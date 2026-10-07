import copy

import pytest

from diagex.vision.raster_detector import ARCHITECTURE, CLASSES
from eval.pid2graph.checkpoint_export import compatible_config, verify_report
from eval.pid2graph.data import digest, sha256


def test_missing_rpn_requires_frozen_builder_and_preserves_original(tmp_path):
    source = tmp_path / "detector.py"
    source.write_text("frozen builder")
    config = {"architecture": ARCHITECTURE, "classes": list(CLASSES), "min_size": 640,
              "source_hashes": {str(source.resolve()): sha256(source)}}
    original = copy.deepcopy(config)
    exported = compatible_config(config, detector_source=source, observed_rpn_threshold=0)
    assert exported["rpn_score_threshold"] == 0 and config == original
    source.write_text("changed builder")
    with pytest.raises(ValueError, match="exact frozen"):
        compatible_config(config, detector_source=source, observed_rpn_threshold=0)


@pytest.mark.parametrize("change,threshold,message", [
    ({"software_smoke_only": True}, 0, "smoke"),
    ({"classes": list(reversed(CLASSES))}, 0, "classes"),
    ({"min_size": 320}, 0, "transform"),
    ({"rpn_score_threshold": 0.05}, 0, "Recorded RPN"),
    ({}, 0.05, "Builder RPN"),
])
def test_incompatible_checkpoint_rejected(change, threshold, message):
    config = {"architecture": ARCHITECTURE, "classes": list(CLASSES), "min_size": 640,
              "rpn_score_threshold": 0, **change}
    with pytest.raises(ValueError, match=message):
        compatible_config(config, detector_source=__file__, observed_rpn_threshold=threshold)


def test_export_requires_same_checkpoint_and_completed_validation():
    panel = {"panel_sha256": "panel", "manifest_sha256": "manifest",
             "panels": {"validation": [{"id": "a"}, {"id": "b"}]}}
    config = {"variant": "supervised_detector", "detector_sha256": "weights", "panel_sha256": "panel"}
    report = {"split": "validation", "complete": True, "config": config, "config_sha256": digest(config),
              "manifest_sha256": "manifest", "panel_sha256": "panel",
              "cases": [{"id": "a", "status": "complete"}, {"id": "b", "status": "complete"}]}
    training = {"manifest_sha256": "manifest"}
    verify_report(report, panel, "weights", training)
    for patch in ({"split": "test"}, {"complete": False}, {"stopped": True},
                  {"cases": report["cases"][:1]}, {"config_sha256": "other"}):
        with pytest.raises(ValueError):
            verify_report({**report, **patch}, panel, "weights", training)
    with pytest.raises(ValueError, match="another checkpoint"):
        verify_report(report, panel, "other weights", training)
