from __future__ import annotations

import hashlib
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode

import fitz
import pytest

from diagex.config import Config
from diagex.web import run_viewer
from diagex.web.server import Workbench, make_handler


def put(run, name, value):
    path = run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def saved_run(tmp_path):
    cfg = Config()
    cfg.runs_dir = tmp_path / "runs"
    workbench = Workbench(cfg, storage_dir=tmp_path / "web")
    run = cfg.runs_dir / "drawing" / "run-1"
    source = tmp_path / "drawing.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=100)
        page.insert_text((15, 20), "SOURCE")
        doc.save(source)
    node = {"id": "n1", "label": "Pump", "kind": "equipment", "page_index": 0,
            "bbox_global": {"x": 10, "y": 20, "w": 30, "h": 40}}
    put(run, "graph.json", {"source_path": str(source), "nodes": [node], "edges": [],
                            "conflicts": [{"reason": "ambiguous label"}]})
    put(run, "pages.json", {"pages": [{"page_index": 0, "width": 400, "height": 200}]})
    put(run, "workbench.json", {"source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    put(run, "result.json", {"quality_status": "partial", "validation_issues": ["Missing tag"]})
    yield workbench, run, source
    workbench.close()


def fingerprint(run):
    return {str(p.relative_to(run)): (p.stat().st_mtime_ns, p.read_bytes()) for p in run.rglob("*") if p.is_file()}


def test_graph_source_and_download_are_read_only(saved_run):
    workbench, run, _ = saved_run
    before = fingerprint(run)
    value = run_viewer.details(workbench, str(run))
    assert value["read_only"] and value["source_verified"]
    assert value["nodes"][0]["label"] == "Pump"
    assert len(value["findings"]) == 2
    assert value["pages"][0]["width"] == 400
    image = run_viewer.page_image(workbench, str(run), 0)
    assert image.startswith(b"\x89PNG")
    payload, name = run_viewer.download(workbench, str(run), "graph.json")
    assert name == "graph.json" and json.loads(payload)["nodes"] == value["nodes"]
    assert fingerprint(run) == before


def test_detection_only_legend_and_candidates_without_source(saved_run):
    workbench, run, _ = saved_run
    (run / "graph.json").unlink()
    (run / "workbench.json").unlink()
    put(run, "detection.json", {"legend_pack": {"entries": [{"label": "Valve", "image_b64": "abc"}]},
        "detections": [{"id": "d1", "page_index": 0, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
        "candidates": [{"id": "c1", "page_index": 0, "bbox": {"x": 2, "y": 4, "w": 4, "h": 6}}],
        "reviews": [{"status": "uncertain", "reason": "not enough evidence"}]})
    value = run_viewer.details(workbench, str(run))
    assert not value["source_available"]
    assert value["legend"][0]["label"] == "Valve"
    assert len(value["nodes"]) == 2 and value["nodes"][1]["candidate_only"]
    assert any(v.get("reason") == "not enough evidence" for v in value["findings"] if isinstance(v, dict))
    with pytest.raises(ValueError, match="unavailable"):
        run_viewer.page_image(workbench, str(run), 0)


def test_legacy_saved_graph_and_detection_are_separate_snapshots(saved_run):
    workbench, run, _ = saved_run
    put(run, "review/state.json", {"graph": {"nodes": [{"id": "n2", "label": "Edited"}], "edges": []}})
    assert run_viewer.details(workbench, str(run))["nodes"][0]["label"] == "Pump"
    before = fingerprint(run)
    saved = run_viewer.details(workbench, str(run), "saved")
    assert saved["nodes"][0]["label"] == "Edited" and len(saved["versions"]) == 2
    assert fingerprint(run) == before
    put(run, "review/detection.json", {"symbols": [{"status": "rejected", "detection": {"id": "n1"}}],
                                       "legends": [{"status": "accepted", "entry": {"label": "Valve"}}]})
    assert run_viewer.details(workbench, str(run), "saved")["legend"][0]["saved_status"] == "accepted"
    (run / "review/state.json").unlink()
    saved = run_viewer.details(workbench, str(run), "saved")
    assert saved["edges"] == []
    assert saved["nodes"][0]["saved_status"] == "rejected"
    assert saved["legend"][0]["saved_status"] == "accepted"


def test_source_mismatch_and_missing_geometry(saved_run):
    workbench, run, source = saved_run
    (run / "pages.json").unlink()
    value = run_viewer.details(workbench, str(run))
    assert value["pages"][0]["geometry_inferred"] is True
    source.write_bytes(b"changed")
    value = run_viewer.details(workbench, str(run))
    assert not value["source_available"]
    assert value["pages"][0]["geometry_inferred"] is True


def test_source_recovered_from_legacy_session_and_evidence_precedence(saved_run):
    workbench, run, source = saved_run
    (run / "workbench.json").unlink()
    put(run, "graph.json", {"nodes": [], "edges": []})
    put(run, "review/session.json", {"source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pages": [{"page_index": 0, "width": 200, "height": 100}]})
    put(run, "evidence/page-0.json", {"page_index": 0, "width": 800, "height": 400, "rotation_deg": 1.2})
    value = run_viewer.details(workbench, str(run))
    assert value["source_verified"] and value["pages"][0]["width"] == 800
    assert run_viewer.page_image(workbench, str(run), 0).startswith(b"\x89PNG")


def test_path_escape_unknown_version_and_download_allowlist(saved_run, tmp_path):
    workbench, run, _ = saved_run
    for path in [tmp_path, run / "../../../", run.parent.parent]:
        with pytest.raises(ValueError):
            run_viewer.details(workbench, str(path))
    for name in ["../../../.env", "workbench.json", "review/state.json"]:
        with pytest.raises(ValueError):
            run_viewer.download(workbench, str(run), name)
    with pytest.raises(ValueError, match="version"):
        run_viewer.details(workbench, str(run), "saved")
    with pytest.raises(ValueError, match="page"):
        run_viewer.page_image(workbench, str(run), -1)
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":true}')
    (run / "legend.json").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        run_viewer.details(workbench, str(run))
    with pytest.raises(ValueError, match="outside"):
        run_viewer.download(workbench, str(run), "legend.json")


def test_http_viewer_routes_and_no_mutations(saved_run):
    workbench, run, _ = saved_run
    before = fingerprint(run)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workbench))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=10)
    try:
        for path, mime in [("/runs/view", "text/html"), ("/static/run.js", "javascript"),
                           ("/static/run.css", "text/css"),
                           ("/api/run-view?" + urlencode({"run_dir": str(run)}), "application/json"),
                           ("/api/run-page?" + urlencode({"run_dir": str(run), "page": 0}), "image/png"),
                           ("/api/run-artifact?" + urlencode({"run_dir": str(run), "name": "graph.json"}), "application/octet-stream")]:
            connection.request("GET", path)
            response = connection.getresponse()
            assert response.status == 200, response.read()
            assert mime in response.getheader("Content-Type")
            response.read()
        connection.request("POST", "/api/run-view", body=b"{}", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        assert fingerprint(run) == before
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
