import copy
import fcntl
import json
import subprocess
from pathlib import Path

import pytest

from eval.pid2graph import broader_detector as worker
from eval.pid2graph.data import digest, sha256


def write(path, value, seal=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if seal:
        value[seal] = digest(value)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def study(tmp_path, monkeypatch):
    training = tmp_path / "detector-epochs-v1"
    write(training / "completed.json", {"fixture": True})
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture; no image inference")
    weights = training / "epoch-001.pt"
    weights.write_bytes(b"fixture; never deserialized")
    rows = [{"id": name, "image": str(image), "image_sha256": sha256(image),
             "size": [20, 20], "collection": "fixture"} for name in ["first", "second"]]
    manifest = write(tmp_path / "manifest.json", {"drawings": [dict(r, split="validation") for r in rows]}, "manifest_sha256")
    panels = [write(tmp_path / f"panel-{name}.json", {"manifest_sha256": manifest["manifest_sha256"],
        "panels": {"validation": selected, "test": []}}, "panel_sha256")
        for name, selected in [("pilot", rows[:1]), ("broad", rows)]]
    write(tmp_path / "validation-promotion-policy.json", {
        "pilot_panel_sha256": panels[0]["panel_sha256"], "broad_panel_sha256": panels[1]["panel_sha256"],
        "selection_source_sha256": sha256(Path(worker.__file__).with_name("selection.py"))}, "policy_sha256")
    config = {"variant": "supervised_detector", "model": "fixture", "split": "validation",
        "panel_sha256": panels[0]["panel_sha256"], "detector_sha256": sha256(weights), "threshold": .15, "device": "cpu"}
    source = training / "validation-epoch-001"
    write(source / "config.json", config)
    prediction = {"id": "first", "image_sha256": sha256(image), "config_sha256": digest(config),
        "detector_sha256": sha256(weights), "runtime_seconds": 2.7, "reserved_or_charged_usd": 0, "errors": [],
        "predictions": [{"id": "detector-7", "label": "valve", "bbox": [1, 2, 10, 12], "confidence": .87,
                         "disposition": "review_proposal", "attributes": {"geometry_basis": "supervised_raster_proposal"}}]}
    write(source / digest("first")[:16] / "predictions.json", prediction)
    report = training / "validation-epoch-001.json"
    write(report, {"config": config})
    exposure = {"complete": True, "validation": [{"epoch": 1, "macro_drawing_f1": .8,
        "checkpoint": str(weights), "report": str(report), "report_sha256": sha256(report)}]}
    monkeypatch.setattr(worker, "training_report", lambda *a: copy.deepcopy(exposure))
    return tmp_path, tmp_path / "manifest.json", "fixture-python"


def test_incomplete_training_prevents_registration_and_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **kw: pytest.fail("Must not infer"))
    with pytest.raises(ValueError, match="Training is incomplete"):
        worker.execute(tmp_path, "manifest", "python")
    assert not (tmp_path / (worker.NAME + "-registration.json")).exists()
    assert json.loads((tmp_path / (worker.NAME + "-status.json")).read_text())["stage"] == "failed"


def test_registration_uses_frozen_checkpoint_and_validation_only(study):
    root, _, _ = study
    record = worker.prepare(*study)
    assert record["epoch"] == 1 and record["test_access_authorized"] is False
    assert record["command"][record["command"].index("--split") + 1] == "validation"
    assert record["command"][0] == "fixture-python"
    old = json.loads((root / "detector-epochs-v1/validation-epoch-001/config.json").read_text())
    assert {k: v for k, v in record["config"].items() if k != "panel_sha256"} == {
        k: v for k, v in old.items() if k != "panel_sha256"}
    assert worker.prepare(*study) == record


def test_failed_inference_preserves_reusable_inputs_without_scoring(study, monkeypatch):
    def fail(command, **kw):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(worker.subprocess, "run", fail)
    monkeypatch.setattr(worker, "score", lambda *a: pytest.fail("Failed inference cannot be scored"))
    with pytest.raises(subprocess.CalledProcessError):
        worker.execute(*study)
    root, _, _ = study
    record = worker.prepare(*study)
    worker.verify_reused_inputs(root, record)
    assert not (root / worker.NAME / digest("second")[:16] / "predictions.json").exists()


@pytest.mark.parametrize("change", ["prediction", "provenance"])
def test_changed_reused_records_cannot_resume(study, monkeypatch, change):
    record = worker.prepare(*study)
    root, manifest, _ = study
    worker.seed(root / "panel-pilot.json", root / "panel-broad.json", record["source_proposals"],
                root / worker.NAME, record["checkpoint"], manifest)
    path = root / worker.NAME / digest("first")[:16] / "predictions.json"
    data = json.loads(path.read_text())
    if change == "prediction":
        data["predictions"][0]["label"] = "pump"
    else:
        data["validation_reuse"]["source_sha256"] = "wrong"
    write(path, data)
    monkeypatch.setattr(worker.subprocess, "run", lambda *a, **kw: pytest.fail("Must not infer"))
    with pytest.raises(ValueError, match="hints or provenance changed"):
        worker.execute(*study)


def test_completed_run_resumes_without_new_inference_and_detects_changed_records(study, monkeypatch):
    root, _, _ = study
    calls = []
    def infer(command, **kw):
        calls.append(command)
        output = root / worker.NAME
        first = json.loads((output / digest("first")[:16] / "predictions.json").read_text())
        write(output / digest("second")[:16] / "predictions.json", dict(first, id="second", predictions=[]))
    def score(manifest, panel, output, destination, split):
        config = json.loads((output / "config.json").read_text())
        broad = json.loads(Path(panel).read_text())
        return write(destination, {"complete": True, "stopped": False, "config": config,
            "config_sha256": digest(config), "split": split, "panel_sha256": broad["panel_sha256"],
            "manifest_sha256": broad["manifest_sha256"], "cases": [{"id": r["id"]} for r in broad["panels"][split]],
            "summary": {"macro_drawing_f1": .8}})
    monkeypatch.setattr(worker.subprocess, "run", infer)
    monkeypatch.setattr(worker, "score", score)
    worker.execute(*study)
    worker.execute(*study)
    assert len(calls) == 1
    assert json.loads((root / (worker.NAME + "-status.json")).read_text())["stage"] == "broader_detector_complete"
    path = root / worker.NAME / digest("second")[:16] / "predictions.json"
    data = json.loads(path.read_text())
    data["runtime_seconds"] += 1
    write(path, data)
    with pytest.raises(ValueError, match="Cannot resume changed"):
        worker.execute(*study)


def test_duplicate_worker_cannot_overwrite_live_status(study):
    root, _, _ = study
    status = root / (worker.NAME + "-status.json")
    write(status, {"stage": "live"})
    with (root / (worker.NAME + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            worker.execute(*study)
    assert json.loads(status.read_text()) == {"stage": "live"}
