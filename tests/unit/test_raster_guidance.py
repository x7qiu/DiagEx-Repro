import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from typer.testing import CliRunner

from diagex.cli import app
from diagex.config import Config
from diagex.extractors import pid_evidence
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import extract_page_evidence
from diagex.vision.loader import iter_pages, load
from diagex.vision.perception import PerceptionBatch, PerceptionOutcome
from diagex.vision.raster_guidance import load_guidance, validate_guidance, view_guidance
from eval.pid2graph.guidance import view_guidance as benchmark_guidance


@pytest.fixture
def source(tmp_path):
    image = tmp_path / "drawing.png"
    Image.new("RGB", (256, 128), "white").save(image)
    payload = {
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "size": [256, 128],
        "coordinate_frame": "original_image_pixels", "status": "proposals_require_interpretation",
        "detector": {"detector_sha256": "a" * 64},
        "predictions": [{"id": "detector-1", "label": "valve", "bbox": [20, 10, 60, 30],
                         "confidence": 0.8, "disposition": "review_proposal"}],
    }
    path = tmp_path / "proposals.json"
    path.write_text(json.dumps(payload))
    return image, path, payload


@pytest.mark.parametrize("change", ["hash", "size", "nan", "bounds", "duplicate", "confirmed", "frame", "pdf"])
def test_invalid_guidance_is_rejected(source, change, tmp_path):
    image, _, payload = source
    if change == "hash":
        payload["image_sha256"] = "b" * 64
    elif change == "size":
        payload["size"] = [128, 256]
    elif change == "nan":
        payload["predictions"][0]["bbox"][0] = float("nan")
    elif change == "bounds":
        payload["predictions"][0]["bbox"][2] = 257
    elif change == "duplicate":
        payload["predictions"].append(copy.deepcopy(payload["predictions"][0]))
    elif change == "confirmed":
        payload["predictions"][0]["disposition"] = "graph_eligible"
    elif change == "frame":
        payload["coordinate_frame"] = "rendered_page"
    else:
        image = tmp_path / "drawing.pdf"
        image.write_bytes(b"not a raster")
    with pytest.raises(ValueError):
        validate_guidance(payload, image)


def test_view_payload_matches_benchmark_with_clipping_and_limit(source):
    _, _, payload = source
    proposals = [{**payload["predictions"][0], "id": str(i), "confidence": i / 100}
                 for i in range(90)]
    proposals.append({**proposals[0], "id": "outside", "bbox": [200, 100, 220, 120]})
    box = SimpleNamespace(x=15, y=6, w=100, h=40)
    result = view_guidance(proposals, box, (2, 2))
    assert result == benchmark_guidance(proposals, box, (2, 2))
    hints = result["raster_proposal_guidance"]
    assert len(hints["guides"]) == 80 and hints["guides_omitted_by_limit"] == 10
    assert hints["guides"][0]["partially_visible"]
    assert "candidate_id" not in hints["guides"][0]


def test_hint_changes_invalidate_guided_cache_and_disabling_restores_baseline(source, monkeypatch):
    image, path, _ = source
    cfg = Config()
    options = dict(cfg=cfg, symbol_standard="none", vision_model="m", reasoning_model="m",
                   legend_path=None, legend_pages=None, legend_region=None, no_legend=True,
                   legend_key=None, effort="medium")
    baseline = pid_evidence._configuration_hash(**options)
    cfg.raster_proposals = load_guidance(path, image)
    guided = pid_evidence._configuration_hash(**options)
    assert guided != baseline
    cfg.raster_proposals["predictions"][0]["bbox"][0] += 1
    assert pid_evidence._configuration_hash(**options) != guided
    changed_boxes = pid_evidence._configuration_hash(**options)
    monkeypatch.setattr("diagex.vision.raster_guidance.implementation_sha256", lambda: "changed-code")
    assert pid_evidence._configuration_hash(**options) != changed_boxes
    cfg.raster_proposals = None
    assert pid_evidence._configuration_hash(**options) == baseline


@pytest.mark.parametrize("enabled", [False, True])
def test_production_perception_supplies_hints_without_creating_detections(source, monkeypatch, enabled):
    image, path, _ = source
    cfg = Config()
    # Exercise the original-image -> resized-page coordinate mapping.
    cfg.tiling.max_page_dim_px = 128
    if enabled:
        cfg.raster_proposals = load_guidance(path, image)
    diagram = load(image, cfg.tiling, cfg.scan)
    page = next(iter_pages(diagram))
    evidence = extract_page_evidence(page=page, source_path=image)
    evidence.role = "pid"  # The white fixture exercises routing, not page-role classification.
    captured = []

    def perceive(**kwargs):
        captured.append(kwargs)
        return PerceptionOutcome(detections=[], batch=PerceptionBatch())

    monkeypatch.setattr(pid_evidence, "perceive_tile", perceive)
    detections, _, _, stopped = pid_evidence._run_perception(
        source=diagram, pages=[evidence], cfg=cfg, client=None, cost=CostTracker(),
        reporter=NullReporter(), store=None, legend_summary=[], run_dir=None, prior_cost={},
    )
    assert captured and not detections and stopped is None
    for call in captured:
        assert call["candidates"] == []
        context = call["page_context"]
        assert context["coverage"] == "deterministic fixed grid"
        if enabled:
            expected = benchmark_guidance(cfg.raster_proposals["predictions"], call["view_info"].page_bbox, (2, 2))
            assert context["raster_proposal_guidance"] == expected["raster_proposal_guidance"]
        else:
            assert "raster_proposal_guidance" not in context


def test_cli_forwards_source_bound_proposals_and_rejects_legacy(source, monkeypatch):
    image, path, _ = source
    captured = []
    monkeypatch.setattr("diagex.cli.load_config", Config)
    monkeypatch.setattr("diagex.extractors.pid.run_pid_extract", lambda **kwargs: (
        captured.append(kwargs) or SimpleNamespace(to_text=lambda: "done")))
    args = ["extract-pid", str(image), "--raster-proposals", str(path)]
    assert CliRunner().invoke(app, [*args, "--engine", "legacy"]).exit_code != 0
    assert not captured
    result = CliRunner().invoke(app, [*args, "--engine", "evidence-v2"])
    assert result.exit_code == 0, result.output
    assert captured[0]["config"].raster_proposals["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_cli_broad_review_requires_guidance_and_does_not_change_default(source, monkeypatch):
    image, path, _ = source
    captured = []
    monkeypatch.setattr("diagex.cli.load_config", Config)
    monkeypatch.setattr("diagex.extractors.pid.run_pid_extract", lambda **kw: (
        captured.append(kw) or SimpleNamespace(to_text=lambda: "done")))
    base = ["extract-pid", str(image), "--engine", "evidence-v2"]
    assert CliRunner().invoke(app, [*base, "--raster-symbol-mode", "broad_review"]).exit_code != 0
    assert CliRunner().invoke(app, [*base, "--raster-symbol-mode", "invented"]).exit_code != 0
    assert not captured
    result = CliRunner().invoke(app, [*base, "--raster-proposals", str(path), "--raster-symbol-mode", "broad_review"])
    assert result.exit_code == 0, result.output
    assert captured[0]["config"].raster_symbol_mode == "broad_review"
    assert Config().raster_symbol_mode == "baseline"


def test_broad_mode_and_semantic_source_are_part_of_cache_identity(source, monkeypatch):
    image, path, _ = source
    cfg = Config(raster_proposals=load_guidance(path, image))
    options = dict(cfg=cfg, symbol_standard="none", vision_model="m", reasoning_model="m",
        legend_path=None, legend_pages=None, legend_region=None, no_legend=True, legend_key=None, effort="medium")
    baseline = pid_evidence._configuration_hash(**options)
    cfg.raster_symbol_mode = "broad_review"
    broad = pid_evidence._configuration_hash(**options)
    assert broad != baseline
    monkeypatch.setattr("diagex.vision.raster_pipeline.implementation_sha256", lambda: "changed-semantic-contract")
    assert pid_evidence._configuration_hash(**options) != broad
    cfg.raster_symbol_mode = "baseline"
    assert pid_evidence._configuration_hash(**options) == baseline


@pytest.mark.parametrize("semantic_failure", [False, True])
@pytest.mark.parametrize("semantic_basis", ["legend", "source"])
def test_production_broad_recognition_emits_only_validated_semantics(source, tmp_path, semantic_failure, semantic_basis):
    from anthropic.types import Message

    image, path, _ = source
    cfg = Config(raster_proposals=load_guidance(path, image), raster_symbol_mode="broad_review")
    diagram = load(image, cfg.tiling, cfg.scan)
    rendered = next(iter_pages(diagram))
    page = extract_page_evidence(page=rendered, source_path=image)
    page.role = "pid"
    calls = []
    def respond(**kw):
        name = kw["tools"][0]["name"]
        calls.append(name)
        if name == "submit_raster_symbols":
            payload = {"raster_results": [{"proposal_id": "detector-1", "decision": "symbol", "broad_category": "valve",
                "confidence": "high", "bbox": {"x": .1, "y": .1, "w": .2, "h": .2}, "reason": "Opposed triangles"}], "discoveries": []}
        elif name == "submit_raster_semantics":
            if semantic_failure:
                raise RuntimeError("provider unavailable")
            payload = {"results": [{"symbol_id": "symbol-0", "decision": "interpreted", "reason": "Matches the gate legend",
                "legend_entry_ids": ["gate"], "symbol": {"kind": "equipment", "valve_type": "gate"}}]}
            if semantic_basis == "source":
                payload["results"][0].update(
                    reason="Visible valve body; detailed subtype is not established.",
                    source_evidence="Two opposed triangles share a center on the pipe; no tag enclosure or leader.",
                    legend_entry_ids=[], symbol={"kind": "equipment", "equipment_class": "valve"},
                )
        else:
            raise AssertionError(name)
        return Message.model_validate({"id": "gen-" + name, "type": "message", "role": "assistant", "model": "test",
            "content": [{"type": "tool_use", "id": name, "name": name, "input": payload}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 50}})
    output = tmp_path / "production"
    output.mkdir()
    cost = CostTracker()
    detections, statuses, counts, stopped = pid_evidence._run_perception(
        source=diagram, pages=[page], cfg=cfg, client=SimpleNamespace(messages_create=respond), cost=cost,
        reporter=NullReporter(), store=None, run_dir=output, prior_cost={},
        legend_summary=[{"source_row_id": "gate", "kind": "valve", "label": "Gate valve", "symbol_class": "gate_valve", "attributes": {"row_status": "accept"}}] if semantic_basis == "legend" else [],
    )
    assert calls == ["submit_raster_symbols", "submit_raster_semantics"]
    assert stopped is None
    assert len(detections) == (0 if semantic_failure else 1)
    if detections:
        if semantic_basis == "legend":
            assert detections[0].attributes["valve_type"] == "gate"
        else:
            assert detections[0].attributes["equipment_class"] == "valve"
            assert "valve_type" not in detections[0].attributes
            assert detections[0].attributes["legend_entry_ids"] == []
            assert "knowledge_match" not in detections[0].attributes
        assert detections[0].attributes["semantic_validation"] == f"{semantic_basis}_supported_model_classification"
        assert not detections[0].attributes["requires_legend_interpretation"]
    assert counts == {"submit_pid_objects": 0, "submit_raster_symbols": 1, "submit_raster_semantics": 1}
    assert len(cost.steps) == (1 if semantic_failure else 2)
    reviews = json.loads((output / "perception.review.json").read_text())["reviews"]
    assert len(reviews) == 1 and reviews[0]["object"]["kind"] == "raster_symbol"
    assert reviews[0]["legend_interpretation"]["decision"] == ("unresolved" if semantic_failure else "interpreted")
    assert statuses[0] == ("error" if semantic_failure else "ok")


def test_broad_routing_preserves_native_candidates_without_raster_calls(monkeypatch):
    from diagex.vision import raster_pipeline
    seen = []
    native = [object()]
    expected = object()
    monkeypatch.setattr(raster_pipeline, "perceive_tile", lambda **kw: seen.append(kw) or expected)
    def forbidden(**kwargs):
        raise AssertionError("Raster interpretation must not displace native candidates")
    monkeypatch.setattr(raster_pipeline, "perceive_broad_raster", forbidden)
    assert raster_pipeline.perceive_broad_with_semantics(candidates=native) is expected
    assert seen[0]["candidates"] is native


def test_broad_public_workflow_reaches_graph_without_manual_approval(source, tmp_path, monkeypatch):
    from diagex.vision.legend_models import LegendPack
    from diagex.vision.symbol_interpretation import DetectionRecord
    image, path, _ = source
    cfg = Config(runs_dir=tmp_path / "runs", raster_proposals=load_guidance(path, image), raster_symbol_mode="broad_review")
    rendered = next(iter_pages(load(image, cfg.tiling, cfg.scan)))
    page = extract_page_evidence(page=rendered, source_path=image)
    page.role = "pid"
    monkeypatch.setattr(pid_evidence, "_inspect_pages", lambda **kw: ([page], []))
    monkeypatch.setattr("diagex.extractors.pid_legend.resolve_evidence_legend",
        lambda **kw: SimpleNamespace(resolution=SimpleNamespace(pack=LegendPack(), source="none")))
    monkeypatch.setattr(pid_evidence, "LLMClient", lambda *a, **kw: SimpleNamespace(retries_total=0, reset_retry_counter=lambda: None))
    detection = DetectionRecord(id="machine-1", page_index=0, tile_id="t1", kind="equipment", label="V-1",
        bbox={"x":20,"y":10,"w":40,"h":20}, confidence="medium", attributes={"valve_type":"gate"})
    monkeypatch.setattr(pid_evidence, "_run_perception", lambda **kw: ([detection], {0:"ok"}, {}, None))
    monkeypatch.setattr(pid_evidence, "_run_contextual_resolution", lambda **kw: ([],0))
    monkeypatch.setattr(pid_evidence, "_run_topology", lambda **kw: [])
    monkeypatch.setattr(pid_evidence, "_run_line_evidence", lambda **kw: ([],0))
    monkeypatch.setattr(pid_evidence, "_run_page_graphs", lambda **kw: ([],0))
    result = pid_evidence.run_pid_evidence_extract(diagram=image, symbol_standard="none", legend_path=None,
        legend_pages=None, legend_region=None, no_legend=True, legend_key=None, effort="medium",
        config=cfg, persist=True, fresh=True, out_path=None, confidence_report_path=None, console=None)
    assert result.workflow_stage == "graph"
    assert len(result.graph.nodes) == 1 and result.graph.nodes[0].attributes["valve_type"] == "gate"
    assert (result.run_dir / "graph.json").exists()
    assert not (result.run_dir / "review").exists()


@pytest.mark.parametrize("unsupported", ["deskew", "adaptive", "wrong_image"])
def test_public_extraction_rejects_unsupported_guidance_before_client_creation(source, monkeypatch, unsupported):
    image, path, _ = source
    cfg = Config(raster_proposals=load_guidance(path, image))
    if unsupported == "deskew":
        cfg.scan.deskew = True
    elif unsupported == "adaptive":
        cfg.symbol_perception.workflow = "adaptive"
    else:
        cfg.raster_proposals["image_sha256"] = "b" * 64

    def no_client(*args, **kwargs):
        raise AssertionError("Invalid guidance must be rejected before creating any API client")

    monkeypatch.setattr(pid_evidence, "LLMClient", no_client)
    with pytest.raises(ValueError):
        pid_evidence.run_pid_evidence_extract(
            diagram=image, symbol_standard="none", legend_path=None, legend_pages=None,
            legend_region=None, no_legend=True, legend_key=None, effort="medium", config=cfg,
            persist=False, out_path=None, confidence_report_path=None, console=None,
        )
