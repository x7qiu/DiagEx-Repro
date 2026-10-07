from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from diagex.web import cv
from diagex.web.server import Workbench, WorkbenchError
from tests.unit.test_web_server import _config, _pdf_bytes, _wait_for_job


def setup_job(tmp_path, monkeypatch):
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_dir=None, quality_status="needs_review")

    workbench = Workbench(_config(tmp_path), storage_dir=tmp_path / "web", runner=runner)
    monkeypatch.setattr(workbench, "_public_result", lambda result: {"workflow_stage": "detection"})
    content = _pdf_bytes()
    upload = workbench.save_upload(
        filename="drawing.pdf",
        content_type="application/pdf",
        source=io.BytesIO(content),
        length=len(content),
    )
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"fixture weights")
    body = {
        "upload_id": upload.id,
        "model_policy": "deepseek-flash",
        "cv_mode": "broad_review",
        "cv_checkpoint": str(weights),
        "cv_device": "cpu",
        "stop_after": "graph",
    }
    return workbench, body, calls


@pytest.mark.parametrize("mode,route", [("broad_review", "broad_review"), ("guided", "baseline")])
def test_cv_web_job_generates_guidance_before_vlm_and_records_selection(tmp_path, monkeypatch, mode, route):
    workbench, body, calls = setup_job(tmp_path, monkeypatch)
    body["cv_mode"] = mode
    proposal = {"format": "test source-bound proposals"}

    def prepare(source, config, selected, folder, log):
        assert not calls
        assert source.suffix == ".pdf"
        assert config.symbol_perception.workflow == "fixed"
        assert not config.scan.deskew
        assert len(selected["checkpoint_sha256"]) == 64
        return proposal

    monkeypatch.setattr(cv, "prepare", prepare)
    job = _wait_for_job(workbench, workbench.start_extraction(body).id)
    assert job.status == "succeeded", job.error
    assert calls[0]["config"].raster_proposals == proposal
    assert calls[0]["config"].raster_symbol_mode == route
    assert calls[0]["stop_after"] == "graph"
    assert job.settings["cv"]["mode"] == mode
    assert job.settings["cv"]["proposals_path"].endswith("proposals.json")
    assert "environment-secret" not in json.dumps(job.public())


def test_cv_off_clears_inherited_guidance_and_never_loads_detector(tmp_path, monkeypatch):
    workbench, body, calls = setup_job(tmp_path, monkeypatch)
    workbench.config.raster_proposals = {"wrong_source": True}
    workbench.config.raster_symbol_mode = "broad_review"
    body["cv_mode"] = "off"
    body["cv_checkpoint"] = "/does/not/exist"
    monkeypatch.setattr(cv, "prepare", lambda *a: pytest.fail("CV is disabled"))
    job = _wait_for_job(workbench, workbench.start_extraction(body).id)
    assert job.status == "succeeded", job.error
    assert calls[0]["config"].raster_proposals is None
    assert calls[0]["config"].raster_symbol_mode == "baseline"
    assert calls[0]["stop_after"] == "graph"


def test_detector_failure_stops_before_any_vlm_call(tmp_path, monkeypatch):
    workbench, body, calls = setup_job(tmp_path, monkeypatch)

    def fail(*args):
        raise RuntimeError("Torch runtime unavailable")

    monkeypatch.setattr(cv, "prepare", fail)
    job = _wait_for_job(workbench, workbench.start_extraction(body).id)
    assert job.status == "failed"
    assert "Torch runtime unavailable" in job.error
    assert calls == []


@pytest.mark.parametrize(
    "override,match",
    [
        ({"cv_mode": "wrong"}, "CV mode"),
        ({"cv_checkpoint": "/missing.pt"}, "checkpoint"),
        ({"cv_device": "wrong"}, "device"),
        ({"engine": "legacy"}, "Evidence v2"),
    ],
)
def test_invalid_cv_settings_rejected_before_job(tmp_path, monkeypatch, override, match):
    workbench, body, calls = setup_job(tmp_path, monkeypatch)
    with pytest.raises(WorkbenchError, match=match):
        workbench.start_extraction({**body, **override})
    assert not workbench.jobs and not calls




@pytest.mark.parametrize("pdf", [False, True])
def test_cv_worker_handoff_validates_source_and_preserves_renderer(tmp_path, monkeypatch, pdf):
    from diagex.vision.pdf_raster_guidance import generate
    from diagex.vision.raster_detector import RasterDetector

    workbench, body, _ = setup_job(tmp_path, monkeypatch)
    selected = cv.settings(body)
    cfg = _config(tmp_path)
    source = tmp_path / ("source.pdf" if pdf else "source.png")
    if pdf:
        source.write_bytes(_pdf_bytes())
    else:
        Image.new("RGB", (100, 100), "white").save(source)

    class Detector:
        signature = {"detector_sha256": selected["checkpoint_sha256"]}

        def predict(self, image):
            return []

        predict_file = RasterDetector.predict_file

    def execute(command, **kwargs):
        spec = json.loads((tmp_path / "worker/request.json").read_text())
        assert "environment-secret" not in json.dumps(spec)
        assert command[0] == selected["python"]
        assert kwargs["env"]["PYTHONPATH"].split(":")[0].endswith("/src")
        result = generate(source, Detector(), cfg) if pdf else Detector().predict_file(source)
        (tmp_path / "worker/proposals.json").write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(cv.subprocess, "run", execute)
    output = cv.prepare(source, cfg, selected, tmp_path / "worker", lambda text: None)
    assert output["detector"]["detector_sha256"] == selected["checkpoint_sha256"]
    assert len(output["artifact_sha256"]) == 64
    if pdf:
        assert len(output["pages"]) == 1
