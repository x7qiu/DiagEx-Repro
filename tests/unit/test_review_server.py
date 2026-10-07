from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import fitz

from diagex.review.core import ReviewStore
from diagex.review.server import make_handler
from diagex.vision.models import BBox, ReconciledGraph, ReconciledNode


def _store(tmp_path: Path) -> ReviewStore:
    source = tmp_path / "source.pdf"
    doc = fitz.open()
    doc.new_page(width=120, height=80)
    doc.save(source)
    doc.close()
    run = tmp_path / "run"
    run.mkdir()
    graph = ReconciledGraph(
        source_path="source.pdf",
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
    (run / "graph.json").write_text(graph.model_dump_json(), encoding="utf-8")
    return ReviewStore.open(run, source_path=source, rater="HTTP", out_dir=tmp_path / "review")


def _request(connection: http.client.HTTPConnection, method: str, path: str, body: dict | None = None):
    raw = None if body is None else json.dumps(body)
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=raw, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    return response.status, response.getheader("Content-Type"), payload


def test_http_state_assets_actions_and_path_safety(tmp_path: Path):
    store = _store(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        status, content_type, body = _request(connection, "GET", "/")
        assert status == 200
        assert content_type.startswith("text/html")
        assert b"Original source" in body
        assert b'id="languageSwitch"' in body
        assert "中文".encode() in body
        assert b'id="layoutMode"' in body
        assert b'id="toggleQueue"' in body
        assert b'id="toggleInspector"' in body
        assert b'id="workspaceView"' in body
        assert b'id="inventoryView"' in body
        assert b'id="evidenceView"' in body
        assert b'id="queueSearch"' in body
        assert b'id="nextFinding"' in body
        assert b'<option value="findings"' in body

        status, content_type, body = _request(connection, "GET", "/static/app.js")
        assert status == 200
        assert "javascript" in content_type
        assert b'diagex.review.language' in body
        assert b'diagex.review.layout' in body
        assert "复核队列".encode() in body
        assert "节点清单".encode() in body
        assert b"Related conflicts" in body
        assert b"Decision needed" in body
        assert "需要做出决定".encode() in body
        assert "工程含义".encode() in body
        assert "压力".encode() in body
        assert "变送器".encode() in body
        assert "电机".encode() in body
        assert "执行机构类型".encode() in body
        assert "电磁执行机构".encode() in body
        assert b"applyContextualMerge" in body
        assert b"applyConflictEdgeType" in body
        assert b"focusSelectionRegion" in body

        status, content_type, body = _request(connection, "GET", "/api/state")
        assert status == 200
        assert content_type.startswith("application/json")
        state = json.loads(body)
        assert state["revision"] == 0
        assert state["inventory"]["counts"]["nodes"] == 1
        assert state["inventory"]["evidence"] == []

        status, content_type, body = _request(connection, "GET", "/api/pages/0/source")
        assert status == 200
        assert content_type == "image/png"
        assert body.startswith(b"\x89PNG")

        status, _, _ = _request(connection, "GET", "/static/../server.py")
        assert status in {400, 404}

        status, _, body = _request(
            connection,
            "POST",
            "/api/action",
            {
                "expected_revision": 0,
                "target_type": "node",
                "target_id": "n-a",
                "operation": "approve",
            },
        )
        assert status == 200
        assert json.loads(body)["state"]["revision"] == 1

        status, _, body = _request(
            connection,
            "POST",
            "/api/action",
            {
                "expected_revision": 0,
                "target_type": "node",
                "target_id": "n-a",
                "operation": "approve",
            },
        )
        assert status == 409
        assert "stale review revision" in json.loads(body)["error"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
