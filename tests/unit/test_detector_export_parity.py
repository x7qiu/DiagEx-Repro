import json
from pathlib import Path

import pytest

from eval.pid2graph import checkpoint_export, detector, detector_parity
from eval.pid2graph.data import sha256


def fixture(tmp_path):
    original, exported = tmp_path / "original.pt", tmp_path / "exported.pt"
    original.write_bytes(b"source weights")
    exported.write_bytes(b"same tensor values with metadata")
    panel = {"panel_sha256": "panel", "manifest_sha256": "manifest"}
    config = {"detector_sha256": sha256(original)}
    proof = {"source_checkpoint": str(original), "source_checkpoint_sha256": sha256(original),
             "export_checkpoint_sha256": sha256(exported), "panel_sha256": "panel",
             "source_config": {"manifest_sha256": "manifest"}, "export_config": {"manifest_sha256": "manifest"},
             "identical_model_tensors": 1, "parity": [{"exact_forward_parity": True}],
             "exporter_sha256": sha256(checkpoint_export.__file__), "source_builder_sha256": sha256(detector.__file__),
             "production_loader_sha256": sha256(Path(detector_parity.__file__).parents[2] / "src/diagex/vision/raster_detector.py")}
    return original, exported, panel, config, proof


def test_metadata_export_requires_explicit_verified_source_binding(tmp_path):
    original, exported, panel, config, proof = fixture(tmp_path)
    assert detector_parity.checkpoint_identity(original, config, panel) == (sha256(original), None)
    with pytest.raises(ValueError, match="differs"):
        detector_parity.checkpoint_identity(exported, config, panel)
    path = tmp_path / "verification.json"
    path.write_text(json.dumps(proof))
    checksum, provenance = detector_parity.checkpoint_identity(exported, config, panel, path)
    assert checksum == sha256(exported) and provenance["source_checkpoint_sha256"] == sha256(original)
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        detector_parity.checkpoint_identity(exported, config, panel, path)


@pytest.mark.parametrize("change", [
    {"export_checkpoint_sha256": "other"}, {"panel_sha256": "test"},
    {"source_config": {"manifest_sha256": "other split"}},
    {"identical_model_tensors": 0}, {"parity": [{"exact_forward_parity": False}]},
    {"production_loader_sha256": "changed implementation"},
])
def test_invalid_export_evidence_is_rejected_before_loading_model(tmp_path, change):
    _, exported, panel, config, proof = fixture(tmp_path)
    path = tmp_path / "verification.json"
    path.write_text(json.dumps({**proof, **change}))
    with pytest.raises(ValueError):
        detector_parity.checkpoint_identity(exported, config, panel, path)
