from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import fitz
import pytest

from diagex.config import Config, LLMConfig
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.review.detection import DetectionReviewStore, ReviewConflict, write_detection_bundle
from diagex.vision.evidence import PageEvidence
from diagex.vision.legend_models import LegendEntry, LegendPack
from diagex.vision.models import BBox
from diagex.vision.perception import DetectionRecord
from diagex.web.server import Workbench, WorkbenchError


def inputs(tmp_path):
    source = tmp_path / "drawing.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=400, height=250)
        page.insert_text((15, 20), "P&ID TEST")
        page.draw_rect(fitz.Rect(40, 60, 90, 190))
        page.draw_circle((140, 80), 10)
        doc.save(source)
    page = PageEvidence(
        page_index=0,
        source_ref="drawing#1",
        width=400,
        height=250,
        dpi=72,
        effective_dpi=72,
        is_scanned=False,
        role="pid",
    )
    detection = DetectionRecord(
        id="d1",
        page_index=0,
        tile_id="t1",
        kind="equipment",
        label="V-001",
        bbox=BBox(x=40, y=60, w=50, h=130),
        confidence="high",
        attributes={"equipment_class": "vessel", "symbol_candidate_id": "c1"},
    )
    legend = LegendPack(
        entries=[LegendEntry(label="Vessel", symbol_class="vessel", kind="equipment")]
    )
    return source, page, detection, legend


def setup_store(tmp_path):
    source, page, detection, legend = inputs(tmp_path)
    run = tmp_path / "runs" / "drawing" / "run-detection"
    run.mkdir(parents=True)
    write_detection_bundle(
        run,
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        pages=[page],
        detections=[detection],
        legend_pack=legend,
        per_page_status={0: "ok"},
        reviews=[],
        candidates=[
            {"id": "c1", "page_index": 0, "bbox": detection.bbox.model_dump()},
            {"id": "c2", "page_index": 0, "bbox": {"x": 130, "y": 70, "w": 20, "h": 20}},
        ],
    )
    atomic_write_json(run / "workbench.json", {"source_path": str(source)})
    return DetectionReviewStore(run), source, page


def action(store, name, **kwargs):
    return store.apply(
        {"action": name, "revision": store.read()["revision"], "rater": "Engineer", **kwargs}
    )


def ready(store):
    action(store, "confirm_legends", ids=["legend-0"])
    action(store, "confirm_detected", ids=["d1"])
    action(store, "reject_candidates", ids=["c2"])
    action(store, "coverage", page_index=0, checked=True)


def raster_store(tmp_path, category="valve"):
    source, page, _, legend = inputs(tmp_path)
    run = tmp_path / "raster-review"
    run.mkdir()
    observation = {
        "page_index": 0, "tile_id": "t1", "status": "uncertain",
        "bbox": {"x": 40, "y": 60, "w": 30, "h": 20},
        "object": {"kind": "raster_symbol", "confidence": "high", "attributes": {
            "broad_category": category, "requires_legend_interpretation": True,
            "geometry_basis": "vlm_broad_raster_observation", "raster_proposal_id": "p1",
        }},
        "reason": "Broad raster observation; detailed legend interpretation required",
    }
    write_detection_bundle(run, source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        pages=[page], detections=[], legend_pack=legend, per_page_status={0: "ok"},
        candidates=[], reviews=[observation])
    return DetectionReviewStore(run), observation


def test_broad_review_is_pending_and_cannot_be_bulk_or_untyped_confirmed(tmp_path):
    store, original = raster_store(tmp_path)
    row = store.read()["symbols"][0]
    assert row["source_observation"] == original
    assert row["detection"]["kind"] == "raster_symbol"
    assert store.snapshot(0, draft=True)["detections"] == []
    action(store, "confirm_legends", ids=["legend-0"])
    action(store, "confirm_detected", ids=[row["id"]])
    assert store.read()["symbols"][0]["status"] == "pending"
    with pytest.raises(ValueError, match="Classify the raster symbol"):
        action(store, "save_symbol", id=row["id"], detection=row["detection"])
    row["detection"]["kind"] = "equipment"
    with pytest.raises(ValueError, match="Classify the raster symbol"):
        action(store, "save_symbol", id=row["id"], detection=row["detection"])
    with pytest.raises(ValueError, match="still need review"):
        store.snapshot(store.read()["revision"])


def test_broad_review_requires_legend_then_keeps_original_evidence_through_edits(tmp_path):
    store, original = raster_store(tmp_path)
    row = store.read()["symbols"][0]
    data = row["detection"]
    data["kind"] = "equipment"
    data["attributes"]["valve_type"] = "gate"
    with pytest.raises(ValueError, match="Review the legend"):
        action(store, "save_symbol", id=row["id"], detection=data)
    action(store, "confirm_legends", ids=["legend-0"])
    action(store, "save_symbol", id=row["id"], detection=data,
           actor_type="agent", evidence_refs=["drawing.pdf#page=1", "legend-0"])
    action(store, "coverage", page_index=0, checked=True)
    frozen = store.snapshot(store.read()["revision"])
    record = DetectionRecord.model_validate(frozen["detections"][0])
    assert record.kind == "equipment" and record.attributes["valve_type"] == "gate"
    assert record.attributes["requires_legend_interpretation"] is False
    assert record.attributes["review_origin"] == "agent"
    assert frozen["review"]["symbols"][0]["source_observation"] == original
    entry = store.read()["legends"][0]["entry"]
    entry["description"] = "Revised definition"
    action(store, "save_legend", id="legend-0", entry=entry)
    assert store.read()["symbols"][0]["status"] == "pending"
    assert store.snapshot(store.read()["revision"], draft=True)["detections"] == []


def test_flow_arrow_can_remain_pending_or_be_excluded_without_becoming_equipment(tmp_path):
    store, original = raster_store(tmp_path, "arrow")
    row = store.read()["symbols"][0]
    action(store, "save_symbol", id=row["id"], detection=row["detection"], status="pending")
    action(store, "save_symbol", id=row["id"], detection=row["detection"], status="rejected",
           note="Flow direction evidence; not a graph node")
    action(store, "confirm_legends", ids=["legend-0"])
    action(store, "coverage", page_index=0, checked=True)
    frozen = store.snapshot(store.read()["revision"])
    assert frozen["detections"] == []
    assert frozen["review"]["symbols"][0]["source_observation"] == original
    assert frozen["review"]["symbols"][0]["detection"]["kind"] == "raster_symbol"


def test_unclassified_review_edits_validate_geometry_and_undo(tmp_path):
    store, original = raster_store(tmp_path)
    row = store.read()["symbols"][0]
    row["detection"]["bbox"]["w"] = 9999
    with pytest.raises(ValueError, match="inside its source page"):
        action(store, "save_symbol", id=row["id"], detection=row["detection"], status="pending")
    action(store, "reject_candidates", ids=[row["id"]])
    action(store, "undo")
    restored = store.read()["symbols"][0]
    assert restored["status"] == "pending" and restored["source_observation"] == original


def test_review_roundtrip_snapshot_and_legend_invalidation(tmp_path):
    store, _, _ = setup_store(tmp_path)
    assert len(store.read()["symbols"]) == 2  # selected candidate is not duplicated
    with pytest.raises(ValueError, match="Review the legend"):
        action(store, "confirm_detected", ids=["d1"])
    ready(store)
    frozen = store.snapshot(store.read()["revision"])
    assert [d["id"] for d in frozen["detections"]] == ["d1"]
    assert DetectionReviewStore(store.run_dir).read() == store.read()
    entry = store.read()["legends"][0]["entry"]
    entry["symbol_class"] = "column"
    action(store, "save_legend", id="legend-0", entry=entry, status="confirmed")
    row = store.read()["symbols"][0]
    assert row["status"] == "pending" and row["stale"]
    action(store, "confirm_detected", ids=["d1"])
    assert (
        store.read()["symbols"][0]["status"] == "pending"
    )  # stale classifications require individual review
    assert frozen["legend_pack"]["entries"][0]["symbol_class"] == "vessel"
    with pytest.raises(ValueError, match="still need review"):
        store.snapshot(store.read()["revision"])


def test_edits_additions_bounds_revision_and_undo(tmp_path):
    store, _, _ = setup_store(tmp_path)
    ready(store)
    d = store.read()["symbols"][0]["detection"]
    original = dict(d["bbox"])
    d["bbox"] = {"x": 30, "y": 50, "w": 60, "h": 140}
    d["label"] = "PACKAGE-A"
    action(store, "save_symbol", id="d1", detection=d, status="confirmed")
    changed_revision = store.read()["revision"]
    with pytest.raises(ReviewConflict):
        store.apply({"action": "undo", "revision": 0, "rater": "Engineer"})
    action(store, "undo")
    assert store.read()["revision"] > changed_revision
    assert store.read()["symbols"][0]["detection"]["bbox"] == original
    d["bbox"]["w"] = 9999
    with pytest.raises(ValueError, match="inside"):
        action(store, "save_symbol", id="d1", detection=d)
    d["bbox"]["w"] = 40
    action(store, "save_symbol", detection=d, status="confirmed")
    assert len(store.snapshot(store.read()["revision"])["detections"]) == 2
    assert store.read()["symbols"][-1]["id"].startswith("manual-")


def test_bulk_only_applies_to_explicit_selection_and_page_coverage_gates_build(tmp_path):
    store, _, _ = setup_store(tmp_path)
    action(store, "confirm_legends", ids=["legend-0"])
    action(store, "confirm_detected", ids=["c2"])
    assert all(r["status"] == "pending" for r in store.read()["symbols"])
    action(store, "confirm_detected", ids=["d1"])
    action(store, "reject_candidates", ids=["c2"])
    with pytest.raises(ValueError, match="visual check"):
        store.snapshot(store.read()["revision"])


def test_workbench_detection_run_source_and_build_snapshot(tmp_path):
    store, source, _ = setup_store(tmp_path)
    ready(store)
    cfg = Config(
        runs_dir=tmp_path / "runs", llm=LLMConfig(model="fake", anthropic_api_key="unused")
    )
    workbench = Workbench(cfg)
    runs = workbench.recent_runs()
    assert runs[0]["has_detection"] and not runs[0]["has_graph"]
    url = workbench.launch_review(rater="Engineer", run_dir=str(store.run_dir), kind="detection")
    assert url.startswith("/detection-review?")
    assert workbench.detection_page(str(store.run_dir), 0).startswith(b"\x89PNG")
    with pytest.raises(WorkbenchError):
        workbench.detection_store(str(tmp_path.parent))
    with pytest.raises(ReviewConflict):
        workbench.start_extraction({"reviewed_run": str(store.run_dir), "reviewed_revision": 0})
    source.write_bytes(b"changed")
    with pytest.raises(WorkbenchError, match="original P&ID"):
        workbench.detection_store(str(store.run_dir))


@pytest.mark.parametrize("uncertain_legend,guard_stop", [(False, False), (True, False), (False, True)])
def test_detection_stops_before_fusion_and_reviewed_build_keeps_instances(
    tmp_path, monkeypatch, uncertain_legend, guard_stop
):
    from diagex.extractors import pid_evidence as pipeline
    from diagex.vision.fusion import fuse_objects as real_fuse

    source, page, detection, legend = inputs(tmp_path)
    if uncertain_legend:
        legend.entries[0].attributes["row_status"] = "uncertain"
    cfg = Config(
        runs_dir=tmp_path / "runs", llm=LLMConfig(model="fake", anthropic_api_key="unused")
    )

    class NoCalls:
        def __init__(self, *args, **kwargs):
            self.retries_total = 0

        def reset_retry_counter(self):
            pass

        def messages_create(self, **kwargs):
            raise AssertionError("Unexpected model call")

    monkeypatch.setattr(pipeline, "LLMClient", NoCalls)

    def inspect(**kw):
        if kw["run_dir"]:
            atomic_write_json(
                kw["run_dir"] / "evidence" / "page-0000.json", page.model_dump(mode="json")
            )
        return [page], []

    monkeypatch.setattr(pipeline, "_inspect_pages", inspect)
    monkeypatch.setattr(
        "diagex.extractors.pid_legend.resolve_evidence_legend",
        lambda **kw: SimpleNamespace(resolution=SimpleNamespace(pack=legend, source="test")),
    )

    def perceive(**kw):
        assert len(kw["legend_summary"]) == (0 if uncertain_legend else 1)
        if not uncertain_legend:
            assert "image_b64" in kw["legend_summary"][0]
        return [detection], {0: "partial" if guard_stop else "ok"}, {}, "Repeated invalid responses" if guard_stop else None

    monkeypatch.setattr(pipeline, "_run_perception", perceive)

    def forbidden(**kw):
        raise AssertionError("Downstream stage called during detection")

    monkeypatch.setattr(pipeline, "fuse_objects", forbidden)
    opts = dict(
        diagram=source,
        symbol_standard="isa-5.1",
        legend_path=None,
        legend_pages=None,
        legend_region=None,
        no_legend=True,
        legend_key=None,
        effort="medium",
        config=cfg,
        persist=True,
        fresh=True,
        out_path=None,
        confidence_report_path=None,
        console=None,
    )
    result = pipeline.run_pid_evidence_extract(**opts, stop_after="graph" if guard_stop else "detection")
    assert result.workflow_stage == "detection"
    assert result.dexpi_json_path is None
    assert (result.run_dir / "detection.json").is_file()
    assert not (result.run_dir / "graph.json").exists()
    if guard_stop:
        assert result.quality_status == "partial"
        manifest = json.loads((result.run_dir / "checkpoints/manifest.json").read_text())
        assert manifest["status"] == "paused"
        assert manifest["pause_reason"] == "Repeated invalid responses"
    store = DetectionReviewStore(result.run_dir)
    action(store, "confirm_legends", ids=["legend-0"])
    action(store, "confirm_detected", ids=["d1"])
    action(store, "coverage", page_index=0, checked=True)
    d = store.read()["symbols"][0]["detection"]
    d["bbox"] = {"x": 35, "y": 50, "w": 70, "h": 150}
    d["attributes"]["equipment_class"] = "column"
    action(store, "save_symbol", id="d1", detection=d, status="confirmed")
    reviewed = store.snapshot(store.read()["revision"])

    def fuse_reviewed(**kw):
        assert (
            len(kw["legend_pack"].entries) == 1
        )  # Explicit human confirmation restores eligibility.
        return real_fuse(**kw)

    monkeypatch.setattr(pipeline, "fuse_objects", fuse_reviewed)
    monkeypatch.setattr(pipeline, "_run_perception", forbidden)
    monkeypatch.setattr("diagex.extractors.pid_legend.resolve_evidence_legend", forbidden)
    monkeypatch.setattr(pipeline, "_run_contextual_resolution", forbidden)
    monkeypatch.setattr(pipeline, "_run_topology", lambda **kw: [])
    monkeypatch.setattr(pipeline, "_run_line_evidence", lambda **kw: ([], 0))
    monkeypatch.setattr(pipeline, "_run_page_graphs", lambda **kw: ([], 0))
    built = pipeline.run_pid_evidence_extract(**opts, reviewed_inputs=reviewed)
    assert built.run_dir != result.run_dir
    assert len(built.graph.nodes) == 1
    assert built.graph.nodes[0].bbox_global.model_dump() == d["bbox"]
    assert built.graph.nodes[0].attributes["equipment_class"] == "column"
    assert (built.run_dir / "reviewed.inputs.json").is_file()
    assert not (result.run_dir / "graph.json").exists()


def test_bundle_changes_are_not_silently_applied_to_an_open_review(tmp_path):
    store, _, _ = setup_store(tmp_path)
    ready(store)
    bundle = json.loads((store.run_dir / "detection.json").read_text())
    bundle["detections"][0]["label"] = "changed"
    atomic_write_json(store.run_dir / "detection.json", bundle)
    with pytest.raises(ReviewConflict, match="inputs changed"):
        store.read()
    with pytest.raises(ReviewConflict, match="inputs changed"):
        DetectionReviewStore(store.run_dir).read()


def test_detection_http_api_and_vector_detail(tmp_path):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.parse import urlencode

    from PIL import Image

    from diagex.web.server import make_handler

    store, _, _ = setup_store(tmp_path)
    cfg = Config(runs_dir=tmp_path / "runs")
    workbench = Workbench(cfg)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workbench))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    try:
        connection.request(
            "GET", "/api/detection-review?" + urlencode({"run_dir": str(store.run_dir)})
        )
        response = connection.getresponse()
        state = json.loads(response.read())
        assert response.status == 200 and state["revision"] == 0
        body = {
            "run_dir": str(store.run_dir),
            "rater": "Engineer",
            "revision": 0,
            "action": "confirm_legends",
            "ids": ["legend-0"],
        }
        for expected in (200, 409):
            connection.request(
                "POST",
                "/api/detection-review/actions",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            assert response.status == expected
        connection.request(
            "GET",
            "/api/detection-crop?"
            + urlencode({"run_dir": str(store.run_dir), "page": 0, "box": "30,50,70,150"}),
        )
        response = connection.getresponse()
        data = response.read()
        assert response.status == 200
        with Image.open(io.BytesIO(data)) as image:
            assert image.height == 1200  # native detail, not a magnified page preview
        connection.request(
            "GET",
            "/api/detection-crop?"
            + urlencode({"run_dir": str(store.run_dir), "page": 0, "box": "-1,0,10,10"}),
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 400
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        workbench.close()


def test_tiny_vector_detail_is_rendered_at_readable_resolution(tmp_path):
    from PIL import Image

    store, _, _ = setup_store(tmp_path)
    workbench = Workbench(Config(runs_dir=tmp_path / "runs"))
    # A small symbol needs a higher PDF render scale than a whole-page preview.
    data = workbench.detection_crop(str(store.run_dir), 0, [128, 68, 24, 24])
    with Image.open(io.BytesIO(data)) as image:
        assert min(image.size) >= 1000
        assert max(image.size) <= 1201
