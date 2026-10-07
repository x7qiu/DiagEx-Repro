from __future__ import annotations

import http.client
import io
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import fitz

from diagex.config import Config, LLMConfig
from diagex.vision.models import BBox, ReconciledGraph, ReconciledNode
from diagex.web.server import Workbench, make_handler


def _config(tmp_path: Path) -> Config:
    config = Config(
        llm=LLMConfig(
            transport="openrouter",
            model="configured/model",
            vision_model="configured/vision",
            reasoning_model="configured/reasoner",
            openrouter_api_key="environment-secret",
        )
    )
    config.runs_dir = tmp_path / "runs"
    config.pid.engine = "evidence-v2"
    return config


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page(width=160, height=100)
    page.insert_text((15, 20), "P&ID TEST")
    payload = document.tobytes()
    document.close()
    return payload


def test_web_default_and_request_use_deepseek_in_every_role(tmp_path: Path) -> None:
    from diagex.llm.model_policy import apply_production_profile
    from diagex.web.model_profiles import DEEPSEEK_FLASH_MODEL

    original = apply_production_profile(_config(tmp_path))
    workbench = Workbench(original, storage_dir=tmp_path / "web")
    public = workbench.public_config()
    assert public["model_policy"] == "deepseek-flash"
    assert public["vision_model"] == public["reasoning_model"] == DEEPSEEK_FLASH_MODEL
    cfg, settings = workbench._config_for_request({
        "model_policy": public["model_policy"],
        "vision_model": "stale/vision",
        "reasoning_model": "stale/reasoning",
    })
    assert {cfg.llm.model, cfg.llm.vision_model, cfg.llm.reasoning_model,
            cfg.llm.escalation_model} == {DEEPSEEK_FLASH_MODEL}
    assert cfg.llm.production_open_weight is False
    assert cfg.llm.transport == "openrouter"
    assert cfg.symbol_perception.workflow == "fixed"
    assert settings["model_policy"] == "deepseek-flash"
    assert original.llm.production_open_weight is True


def test_web_qwen_is_opt_in_and_custom_does_not_inherit_it(tmp_path: Path) -> None:
    from diagex.llm.model_policy import ESCALATION_MODEL, FAST_MODEL, apply_production_profile

    workbench = Workbench(apply_production_profile(_config(tmp_path)), storage_dir=tmp_path / "web")
    qwen, _ = workbench._config_for_request({"model_policy": "production-open-weight"})
    assert qwen.llm.production_open_weight is True
    assert qwen.llm.vision_model == FAST_MODEL
    assert qwen.llm.escalation_model == ESCALATION_MODEL
    custom, _ = workbench._config_for_request({
        "model_policy": "evaluation", "vision_model": "custom/vision",
        "reasoning_model": "custom/reasoning",
    })
    assert custom.llm.vision_model == "custom/vision"
    assert custom.llm.reasoning_model == custom.llm.escalation_model == "custom/reasoning"
    assert custom.llm.production_open_weight is False


def _wait_for_job(workbench: Workbench, job_id: str) -> object:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = workbench.job(job_id)
        if job.status in {"succeeded", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("background extraction did not finish")


def test_config_is_redacted_and_upload_is_safely_named(tmp_path: Path) -> None:
    workbench = Workbench(_config(tmp_path), storage_dir=tmp_path / "web")

    public = workbench.public_config()
    assert public["configured_keys"]["openrouter"] is True
    assert "environment-secret" not in json.dumps(public)

    payload = _pdf_bytes()
    upload = workbench.save_upload(
        filename="../../unsafe drawing.pdf",
        content_type="application/pdf",
        source=io.BytesIO(payload),
        length=len(payload),
    )

    assert upload.filename == "unsafe_drawing.pdf"
    assert upload.path.parent == (tmp_path / "web" / "uploads" / upload.id)
    assert upload.path.name == "unsafe_drawing.pdf"
    assert upload.path.read_bytes() == payload


def test_fresh_extraction_uses_memory_key_and_persists_only_redacted_settings(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    run_dir = tmp_path / "runs" / "drawing" / "run-1"

    def fake_runner(**kwargs: object) -> object:
        captured.update(kwargs)
        run_dir.mkdir(parents=True)
        graph = ReconciledGraph(source_path="drawing.pdf")
        (run_dir / "graph.json").write_text(graph.model_dump_json(), encoding="utf-8")
        return SimpleNamespace(
            diagram_stem="drawing",
            run_id="r-test",
            engine="evidence-v2",
            model="vision=test/vision; reasoning=test/reasoner",
            effort="high",
            quality_status="partial",
            run_dir=run_dir,
            dexpi_json_path=run_dir / "pid.dexpi.json",
            dexpi_stats={"equipment_count": 2, "instrument_count": 3},
            dexpi_issues=["review"],
            validation_issues=[],
            legend_source="cache_hit",
            legend_entry_count=10,
            cost_summary={"total_tokens": 1234, "wall_clock_s": 2.5, "retries": 1},
        )

    workbench = Workbench(
        _config(tmp_path), storage_dir=tmp_path / "web", runner=fake_runner
    )
    payload = _pdf_bytes()
    upload = workbench.save_upload(
        filename="drawing.pdf",
        content_type="application/pdf",
        source=io.BytesIO(payload),
        length=len(payload),
    )
    job = workbench.start_extraction(
        {
            "upload_id": upload.id,
            "provider": "openrouter",
            "api_key": "one-run-secret",
            "base_url": "https://openrouter.ai/api",
            "vision_model": "test/vision",
            "reasoning_model": "test/reasoner",
            "reasoning_mode": "enabled",
            "effort": "high",
            "engine": "evidence-v2",
            "fresh": True,
            "knowledge_mode": "general",
            "knowledge_sources": ["isa-5.1-2009", "sht-3101-2017"],
        }
    )
    finished = _wait_for_job(workbench, job.id)

    assert finished.status == "succeeded"
    assert captured["fresh"] is True
    request_config = captured["config"]
    assert request_config.knowledge["selected_sources"] == ["isa-5.1-2009", "sht-3101-2017"]
    assert request_config.llm.openrouter_api_key == "one-run-secret"
    public_job = finished.public()
    assert "one-run-secret" not in json.dumps(public_job)
    manifest = json.loads((run_dir / "workbench.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["fresh"] is True
    assert manifest["settings"]["knowledge"]["selected_sources"] == request_config.knowledge["selected_sources"]
    assert "one-run-secret" not in json.dumps(manifest)






def _request(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, str | None, bytes]:
    headers: dict[str, str] = {}
    if content_type:
        headers["Content-Type"] = content_type
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    return response.status, response.getheader("Content-Type"), response.read()


def test_http_dashboard_upload_job_and_removed_review_routes(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "drawing" / "run-1"

    def fake_runner(**kwargs: object) -> object:
        run_dir.mkdir(parents=True)
        graph = ReconciledGraph(
            source_path="drawing.pdf",
            nodes=[
                ReconciledNode(
                    id="n-a",
                    kind="equipment",
                    label="V-1",
                    bbox_global=BBox(x=10, y=10, w=20, h=20),
                    page_index=0,
                    attributes={"equipment_class": "vessel"},
                    confidence="high",
                )
            ],
        )
        (run_dir / "graph.json").write_text(graph.model_dump_json(), encoding="utf-8")
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "run_id": "r-test",
                    "model": "vision=test; reasoning=test",
                    "engine": "evidence-v2",
                    "quality_status": "partial",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            diagram_stem="drawing",
            run_id="r-test",
            engine="evidence-v2",
            model="vision=test; reasoning=test",
            effort="medium",
            quality_status="partial",
            run_dir=run_dir,
            dexpi_json_path=None,
            dexpi_stats={},
            dexpi_issues=[],
            validation_issues=[],
            legend_source="no_legend",
            legend_entry_count=0,
            cost_summary={"total_tokens": 10, "wall_clock_s": 0.1},
        )

    workbench = Workbench(
        _config(tmp_path), storage_dir=tmp_path / "web", runner=fake_runner
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workbench))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1], timeout=10
    )
    try:
        status, content_type, body = _request(connection, "GET", "/")
        assert status == 200
        assert content_type.startswith("text/html")
        assert b'id="startButton"' in body
        assert b'id="reviewButton"' not in body
        assert b'id="raterName"' not in body

        status, content_type, body = _request(connection, "GET", "/static/app.js")
        assert status == 200
        assert "javascript" in content_type
        assert "开始全新运行".encode() in body

        status, _, body = _request(connection, "GET", "/api/config")
        assert status == 200
        assert b"environment-secret" not in body

        source = _pdf_bytes()
        status, _, body = _request(
            connection,
            "POST",
            "/api/uploads?filename=drawing.pdf",
            source,
            "application/pdf",
        )
        assert status == 201
        upload_id = json.loads(body)["upload"]["id"]

        request_body = json.dumps(
            {
                "upload_id": upload_id,
                "provider": "openrouter",
                "api_key": "browser-secret",
                "vision_model": "test/vision",
                "reasoning_model": "test/reasoner",
                "reasoning_mode": "enabled",
                "engine": "evidence-v2",
                "fresh": True,
            }
        ).encode()
        status, _, body = _request(
            connection, "POST", "/api/extractions", request_body, "application/json"
        )
        assert status == 202
        assert b"browser-secret" not in body
        job_id = json.loads(body)["job"]["id"]
        _wait_for_job(workbench, job_id)

        status, _, body = _request(connection, "GET", f"/api/jobs/{job_id}?after=0")
        assert status == 200
        job_state = json.loads(body)
        assert job_state["status"] == "succeeded"
        assert b"browser-secret" not in body

        for route in ("/api/reviews", "/api/detection-review/actions"):
            connection.request("POST", route, body="{}", headers={"Content-Type":"application/json"})
            response = connection.getresponse()
            assert response.status == 404
            response.read()
        connection.request("GET", "/detection-review")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        workbench.close()
