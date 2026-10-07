import copy
import json

import pytest

from eval.pid2graph.data import digest, sha256
from eval.pid2graph.guidance import load_proposals
from eval.pid2graph.reuse_detector_validation import seed
from eval.pid2graph.selection import detector_signature


def write(path, value, seal=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if seal:
        value[seal] = digest(value)
    path.write_text(json.dumps(value))
    return value


def fixture(tmp_path, *, second_split="validation"):
    image = tmp_path / "source.png"
    image.write_bytes(b"software fixture; no image inference")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"software fixture checkpoint; never deserialized")
    rows = [{"id": name, "collection": "fixture", "image": str(image), "image_sha256": sha256(image), "size": [20, 20]}
            for name in ["first", "second"]]
    manifest = write(tmp_path / "manifest.json", {"drawings": [{**row, "split": "validation" if i == 0 else second_split}
        for i, row in enumerate(rows)]}, "manifest_sha256")
    panels = []
    for name, selected in [("source", rows[:1]), ("target", rows)]:
        panels.append(write(tmp_path / (name + ".json"), {"manifest_sha256": manifest["manifest_sha256"],
            "panels": {"validation": selected, "test": []}}, "panel_sha256"))
    config = {"variant": "supervised_detector", "model": "fixture", "split": "validation",
        "panel_sha256": panels[0]["panel_sha256"], "detector_sha256": sha256(checkpoint), "threshold": .15}
    run = tmp_path / "run"
    write(run / "config.json", config)
    record = {"id": "first", "image_sha256": rows[0]["image_sha256"], "config_sha256": digest(config),
        "detector_sha256": sha256(checkpoint), "runtime_seconds": 2.7, "reserved_or_charged_usd": 0, "errors": [],
        "predictions": [{"id": "detector-7", "label": "valve", "bbox": [1, 2, 10, 12], "confidence": .87,
                         "disposition": "review_proposal", "attributes": {"geometry_basis": "supervised_raster_proposal"}}]}
    path = run / digest("first")[:16] / "predictions.json"
    write(path, record)
    return (tmp_path / "source.json", tmp_path / "target.json", run, tmp_path / "expanded",
            checkpoint, tmp_path / "manifest.json"), path


def test_detector_reuse_preserves_exact_hints_runtime_and_source_record(tmp_path):
    args, path = fixture(tmp_path)
    original_bytes = path.read_bytes()
    result = seed(*args)
    assert result["reused_drawings"] == ["first"] and result["unattempted_drawings"] == ["second"]
    assert result["new_inference"] == result["api_calls"] == 0
    assert result["test_inputs_read"] is result["graphml_read"] is False
    old, new = json.loads(original_bytes), json.loads((args[3] / digest("first")[:16] / "predictions.json").read_text())
    assert path.read_bytes() == original_bytes
    assert {k: v for k, v in new.items() if k not in {"config_sha256", "validation_reuse"}} == {
        k: v for k, v in old.items() if k != "config_sha256"}
    assert new["validation_reuse"]["source_sha256"] == sha256(path)
    original_config = json.loads((args[2] / "config.json").read_text())
    expanded_config = json.loads((args[3] / "config.json").read_text())
    assert detector_signature(original_config) == detector_signature(expanded_config)
    assert not (args[3] / digest("second")[:16] / "predictions.json").exists()
    # A seeded directory is still incomplete; it must not appear as full hints.
    with pytest.raises(FileNotFoundError):
        load_proposals(args[3], json.loads(args[1].read_text()), "validation")
    with pytest.raises(FileExistsError):
        seed(*args)


@pytest.mark.parametrize("change", ["checkpoint", "image", "missing", "error", "bad_bounds", "nan_confidence", "duplicate_id", "approved", "test_source", "stopped"])
def test_invalid_or_unfinished_detector_records_cannot_be_seeded(tmp_path, change):
    args, path = fixture(tmp_path)
    record = json.loads(path.read_text())
    if change == "checkpoint":
        args[4].write_bytes(b"changed weights")
    elif change == "image":
        (tmp_path / "source.png").write_bytes(b"changed pixels")
    elif change == "missing":
        path.unlink()
    elif change == "test_source":
        config = json.loads((args[2] / "config.json").read_text())
        config["split"] = "test"
        write(args[2] / "config.json", config)
    elif change == "stopped":
        write(args[2] / "stop-request.json", {})
    else:
        if change == "error":
            record["errors"] = ["unfinished"]
        elif change == "bad_bounds":
            record["predictions"][0]["bbox"][2] = 21
        elif change == "nan_confidence":
            record["predictions"][0]["confidence"] = float("nan")
        elif change == "duplicate_id":
            record["predictions"].append(copy.deepcopy(record["predictions"][0]))
        else:
            record["predictions"][0]["disposition"] = "graph_eligible"
        write(path, record)
    with pytest.raises(ValueError):
        seed(*args)
    assert not args[3].exists()


def test_expansion_cannot_relabel_test_drawings_as_validation(tmp_path):
    args, _ = fixture(tmp_path, second_split="test")
    with pytest.raises(ValueError, match="frozen validation split"):
        seed(*args)
    assert not args[3].exists()
