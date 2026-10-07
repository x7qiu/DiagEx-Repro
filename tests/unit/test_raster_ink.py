import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from diagex.cli import app
from diagex.config import Config
from diagex.extractors import pid_evidence
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import extract_page_evidence
from diagex.vision.loader import iter_pages, load
from diagex.vision.models import BBox
from diagex.vision.perception import DetectionRecord, PerceptionBatch, PerceptionOutcome
from diagex.vision.raster_ink import filter_observations, ink_guard, validate_source
from eval.pid2graph.postprocess import ink_guard as frozen_ink_guard


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "drawing.png"
    image = Image.new("RGB", (256, 128), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 39, 39), outline="black", width=2)
    image.save(path)
    return path


def detection(identity, box):
    return DetectionRecord(id=identity, page_index=0, tile_id="p0-r0-c0",
                           kind="equipment", label="V", bbox=BBox(**box), confidence="medium")


def test_pixel_threshold_clipping_and_rounding_match_frozen_filter(source):
    boxes = [[-10, -5, 2, 2], [19.1, 19.5, 40.2, 40.7], [200, 100, 210, 110],
             [250, 125, 300, 140], [0, 0, 0, 0], [-30, -30, -10, -10]]
    with Image.open(source) as image:
        rows = [{"id": str(i), "bbox": box} for i, box in enumerate(boxes)]
        assert ink_guard(image, rows) == frozen_ink_guard(image, rows)


def test_original_pixel_mapping_filters_detections_and_proposals_without_mutation(source):
    ink = {"x": 10, "y": 10, "w": 10, "h": 10}
    blank = {"x": 90, "y": 40, "w": 10, "h": 10}
    detections = [detection("ink", ink), detection("blank", blank)]
    reviews = [{"page_index": 0, "bbox": box, "object": {"kind": "equipment"}}
               for box in (ink, blank)]
    reviews.append({"page_index": 0, "candidate_id": "native", "bbox": blank,
                    "status": "unreviewed"})
    before = copy.deepcopy(reviews)
    kept, remaining, audit = filter_observations(
        source, [SimpleNamespace(page_index=0, width=128, height=64)], detections, reviews)
    assert [d.id for d in kept] == ["ink"]
    assert remaining == [reviews[0], reviews[2]]
    assert {r["id"] for r in audit["rejected"]} == {"detection:1", "review:1"}
    assert audit["rejected"][0]["bbox"] == [180, 80, 200, 100]
    assert audit["rejected"][1]["original"] == reviews[1]
    assert reviews == before and len(detections) == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_perception_filters_before_review_and_preserves_rejection_audit(source, monkeypatch, tmp_path, enabled):
    cfg = Config(raster_ink_filter=enabled)
    cfg.tiling.max_page_dim_px = 128
    diagram = load(source, cfg.tiling, cfg.scan)
    page = next(iter_pages(diagram))
    evidence = extract_page_evidence(page=page, source_path=source)
    evidence.role = "pid"
    blank = {"x": 90, "y": 40, "w": 10, "h": 10}

    def perceive(**kwargs):
        return PerceptionOutcome(
            detections=[detection("blank", blank)],
            batch=PerceptionBatch(candidate_reviews=[{
                "object": {"kind": "equipment", "bbox": blank}, "bbox": blank,
                "status": "review", "reason": "Unanchored proposal",
            }]),
        )

    monkeypatch.setattr(pid_evidence, "perceive_tile", perceive)
    run = tmp_path / "run"
    run.mkdir()
    detections, _, _, _ = pid_evidence._run_perception(
        source=diagram, pages=[evidence], cfg=cfg, client=None, cost=CostTracker(),
        reporter=NullReporter(), store=None, legend_summary=[], run_dir=run, prior_cost={})
    reviews = json.loads((run / "perception.review.json").read_text())["reviews"]
    if enabled:
        assert detections == [] and reviews == []
        audit = json.loads((run / "raster-ink-filter.json").read_text())
        assert len(audit["rejected"]) >= 2
    else:
        assert detections and reviews
        assert not (run / "raster-ink-filter.json").exists()


def test_filter_cache_is_separate_and_baseline_is_restored(monkeypatch):
    cfg = Config()
    options = dict(cfg=cfg, symbol_standard="none", vision_model="m", reasoning_model="m",
                   legend_path=None, legend_pages=None, legend_region=None, no_legend=True,
                   legend_key=None, effort="medium")
    baseline = pid_evidence._configuration_hash(**options)
    cfg.raster_ink_filter = True
    filtered = pid_evidence._configuration_hash(**options)
    assert filtered != baseline
    monkeypatch.setattr("diagex.vision.raster_ink.implementation_sha256", lambda: "changed")
    assert pid_evidence._configuration_hash(**options) != filtered
    cfg.raster_ink_filter = False
    assert pid_evidence._configuration_hash(**options) == baseline


def test_cli_filter_is_opt_in_and_requires_evidence_engine(source, monkeypatch):
    captured = []
    monkeypatch.setattr("diagex.cli.load_config", Config)
    monkeypatch.setattr("diagex.extractors.pid.run_pid_extract", lambda **kwargs: (
        captured.append(kwargs) or SimpleNamespace(to_text=lambda: "done")))
    args = ["extract-pid", str(source), "--raster-ink-filter"]
    assert CliRunner().invoke(app, [*args, "--engine", "legacy"]).exit_code != 0
    assert not captured
    result = CliRunner().invoke(app, [*args, "--engine", "evidence-v2"])
    assert result.exit_code == 0, result.output
    assert captured[0]["config"].raster_ink_filter


def test_filter_rejects_pdf_and_multiframe_sources(tmp_path):
    with pytest.raises(ValueError, match="single raster"):
        validate_source(tmp_path / "native.pdf")
    path = tmp_path / "pages.tiff"
    Image.new("RGB", (10, 10)).save(path, save_all=True, append_images=[Image.new("RGB", (10, 10))])
    with pytest.raises(ValueError, match="single-frame"):
        validate_source(path)
