import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from diagex.cli import app
from diagex.config import Config
from diagex.extractors import pid_evidence
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import extract_page_evidence
from diagex.vision.loader import iter_pages, load
from diagex.vision.pdf_raster_guidance import (
    file_hash,
    generate,
    load_pdf_guidance,
    page_proposals,
    route_guided_pages,
    validate_pdf_guidance,
)
from diagex.vision.perception import PerceptionBatch, PerceptionOutcome
from diagex.vision.symbol_candidates import symbol_candidates

SOURCE = Path(__file__).resolve().parents[1] / "p-ids-public/dexpi-reference.pdf"


class FixtureDetector:
    signature = {"detector_sha256": "a" * 64, "scope": "software fixture, not a trained model"}

    def predict(self, image):
        return [{"id": "fixture-1", "label": "valve", "bbox": [10, 10, 30, 30],
                 "confidence": .5, "disposition": "review_proposal"}]


@pytest.fixture
def prepared(tmp_path):
    cfg = Config()
    cfg.tiling.max_page_dim_px = 1200
    payload = generate(SOURCE, FixtureDetector(), cfg, page_indices=[0])
    path = tmp_path / "pdf-proposals.json"
    path.write_text(json.dumps(payload))
    return cfg, payload, path


def test_existing_pdf_render_round_trip_keeps_native_evidence(prepared):
    cfg, payload, path = prepared
    loaded = load_pdf_guidance(path, SOURCE, cfg)
    page = next(iter_pages(load(SOURCE, cfg.tiling, cfg.scan)))
    before = extract_page_evidence(page=page, source_path=SOURCE)
    hints = page_proposals(loaded, page)
    after = extract_page_evidence(page=page, source_path=SOURCE)
    assert hints == payload["pages"][0]["predictions"]
    assert before.text_spans and before.paths
    assert before.model_dump() == after.model_dump()
    assert file_hash(SOURCE) == payload["source_sha256"]
    assert loaded["artifact_sha256"] == file_hash(path)


@pytest.mark.parametrize("change", ["source", "settings", "frame", "approved", "page_index", "duplicate_page", "size", "nan", "bounds", "duplicate_id", "weights", "malformed", "deskew"])
def test_mismatched_pdf_hints_are_rejected_before_use(prepared, change):
    cfg, original, _ = prepared
    payload = copy.deepcopy(original)
    page = payload["pages"][0]
    if change == "source":
        payload["source_sha256"] = "b" * 64
    elif change == "settings":
        cfg.tiling.max_page_dim_px += 1
    elif change == "frame":
        payload["coordinate_frame"] = "original_image_pixels"
    elif change == "approved":
        page["predictions"][0]["disposition"] = "graph_eligible"
    elif change == "page_index":
        page["page_index"] = 20
    elif change == "duplicate_page":
        payload["pages"].append(copy.deepcopy(page))
    elif change == "size":
        page["size"] = [0, 400]
    elif change == "nan":
        page["predictions"][0]["confidence"] = float("nan")
    elif change == "bounds":
        page["predictions"][0]["bbox"][2] = 10000
    elif change == "duplicate_id":
        page["predictions"].append(copy.deepcopy(page["predictions"][0]))
    elif change == "weights":
        payload["detector"]["detector_sha256"] = "unknown"
    elif change == "malformed":
        payload = []
    else:
        cfg.scan.deskew = True
    with pytest.raises(ValueError):
        validate_pdf_guidance(payload, SOURCE, cfg)


def test_pixel_identity_is_checked_and_absent_pages_have_no_hints(prepared):
    cfg, payload, _ = prepared
    page = next(iter_pages(load(SOURCE, cfg.tiling, cfg.scan)))
    page.image.putpixel((0, 0), (100, 20, 30))
    with pytest.raises(ValueError, match="rendered pixels"):
        page_proposals(payload, page)
    page.page_index = 1
    assert page_proposals(payload, page) is None


def test_pdf_cli_validates_and_forwards_proposals(prepared, monkeypatch):
    cfg, payload, path = prepared
    seen = []
    monkeypatch.setattr("diagex.cli.load_config", lambda: cfg)
    monkeypatch.setattr("diagex.extractors.pid.run_pid_extract", lambda **kw: seen.append(kw) or SimpleNamespace(to_text=lambda: "done"))
    result = CliRunner().invoke(app, ["extract-pid", str(SOURCE), "--engine", "evidence-v2",
                                    "--raster-proposals", str(path), "--raster-symbol-mode", "broad_review"])
    assert result.exit_code == 0, result.output
    assert seen[0]["config"].raster_proposals["source_sha256"] == payload["source_sha256"]


def test_pdf_production_hints_do_not_replace_native_candidates(prepared, tmp_path, monkeypatch):
    cfg, payload, _ = prepared
    cfg.raster_proposals, cfg.raster_symbol_mode = payload, "broad_review"
    source = load(SOURCE, cfg.tiling, cfg.scan)
    page = next(iter_pages(source))
    evidence = extract_page_evidence(page=page, source_path=SOURCE)
    evidence.role = "pid"
    native = symbol_candidates(evidence)
    assert native
    seen = []
    def native_perception(**kw):
        seen.append(kw)
        assert kw["candidates"] == native
        assert kw["page"].text_spans == evidence.text_spans
        assert kw["page"].paths == evidence.paths
        assert kw["page_context"]["raster_proposal_guidance"]["guides"][0]["proposal_id"] == "fixture-1"
        return PerceptionOutcome(detections=[], batch=PerceptionBatch(), candidates=native)
    monkeypatch.setattr("diagex.vision.raster_pipeline.perceive_tile", native_perception)
    def forbidden(**kw):
        raise AssertionError("Broad raster path must not replace existing native candidates")
    monkeypatch.setattr("diagex.vision.raster_pipeline.perceive_broad_raster", forbidden)
    result = pid_evidence._run_perception(source=source, pages=[evidence], cfg=cfg, client=None,
        cost=CostTracker(), reporter=NullReporter(), store=None, legend_summary=[], run_dir=tmp_path, prior_cost={})
    assert seen and result[3] is None


def test_scanned_first_page_reaches_public_symbol_review(tmp_path, monkeypatch):
    from diagex.vision.legend_models import LegendPack

    source = SOURCE.with_name("two-tanks.pdf")
    cfg = Config(runs_dir=tmp_path / "runs", raster_symbol_mode="broad_review")
    cfg.tiling.max_page_dim_px = 1200
    cfg.raster_proposals = generate(source, FixtureDetector(), cfg, page_indices=[0])
    rendered = next(iter_pages(load(source, cfg.tiling, cfg.scan)))
    original = extract_page_evidence(page=rendered, source_path=source)
    assert original.is_scanned and original.role == "cover"
    assert not original.paths and not original.text_spans
    seen = []
    def perception(**kw):
        seen.append(kw)
        assert kw["page"].role == "pid" and kw["page"].fail_open
        assert not kw["candidates"]
        kw["on_attempt"]("submit_raster_symbols")
        kw["on_attempt"]("submit_raster_semantics")
        return PerceptionOutcome(detections=[], batch=PerceptionBatch())
    monkeypatch.setattr("diagex.vision.raster_pipeline.perceive_broad_with_semantics", perception)
    monkeypatch.setattr(pid_evidence, "LLMClient",
        lambda *a, **kw: SimpleNamespace(retries_total=0, reset_retry_counter=lambda: None))
    monkeypatch.setattr("diagex.extractors.pid_legend.resolve_evidence_legend",
        lambda **kw: SimpleNamespace(resolution=SimpleNamespace(pack=LegendPack(), source="no_legend")))
    result = pid_evidence.run_pid_evidence_extract(diagram=source, symbol_standard="none",
        legend_path=None, legend_pages=None, legend_region=None, no_legend=True,
        legend_key=None, effort="low", config=cfg, persist=True, fresh=True,
        out_path=None, confidence_report_path=None, console=None)
    assert seen and result.workflow_stage == "detection" and not result.graph.nodes
    assert result.cost_summary["n_tool_calls"] == 2 * len(seen)
    assert result.cost_summary["tool_call_counts"]["submit_raster_semantics"] == len(seen)
    saved = json.loads((result.run_dir / "evidence/page-0001.json").read_text())
    assert saved["role"] == "pid" and saved["role_confidence"] == "low"
    assert saved["paths"] == saved["text_spans"] == []
    routing = json.loads((result.run_dir / "pdf-proposal-routing.json").read_text())
    assert routing["changes"][0]["previous_role"] == "cover"
    bundle = json.loads((result.run_dir / "detection.json").read_text())
    assert bundle["pages"][0]["role"] == "pid"


@pytest.mark.parametrize("excluded", ["legend", "unselected", "confident_cover", "not_scanned", "native"])
def test_pdf_guidance_routing_preserves_excluded_and_native_pages(prepared, excluded):
    cfg, payload, _ = prepared
    source = SOURCE.with_name("two-tanks.pdf")
    page = next(iter_pages(load(source, cfg.tiling, cfg.scan)))
    evidence = extract_page_evidence(page=page, source_path=source)
    legend_pages = None
    if excluded == "legend":
        legend_pages = [0]
    elif excluded == "unselected":
        payload = {"pages": []}
    elif excluded == "confident_cover":
        evidence.role_confidence = "high"
    elif excluded == "not_scanned":
        evidence.is_scanned = False
    else:
        native_page = next(iter_pages(load(SOURCE, cfg.tiling, cfg.scan)))
        evidence.paths = extract_page_evidence(page=native_page, source_path=SOURCE).paths
        assert evidence.paths
    before = evidence.model_dump()
    assert route_guided_pages([evidence], payload, legend_pages=legend_pages) == []
    assert evidence.model_dump() == before
