from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from diagex.config import PidConfig
from diagex.extractors.pid_legend import (
    _annotation_to_entry,
    _dedupe,
    _load_cached,
    _thumbnail_b64,
)
from diagex.vision.evidence import PageEvidence, PathEvidence, TextEvidence
from diagex.vision.legend_models import LegendEntry, LegendPack
from diagex.vision.models import Annotation, BBox, DiagramPage


def _page_and_evidence() -> tuple[DiagramPage, PageEvidence]:
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.line((50, 80, 80, 100), fill="black", width=3)
    draw.line((80, 100, 50, 120), fill="black", width=3)
    draw.line((50, 120, 50, 80), fill="black", width=3)
    page = DiagramPage(
        page_index=0,
        image=image,
        width=400,
        height=200,
        dpi=300,
        effective_dpi=300,
        is_scanned=False,
        source_ref="test#page=0",
    )
    paths = [
        PathEvidence(
            id=f"path-{index}",
            page_index=0,
            points=points,
            bbox=bbox,
            origin="pdf_vector",
            primitive="line",
        )
        for index, (points, bbox) in enumerate(
            (
                ([(50, 80), (80, 100)], BBox(x=50, y=80, w=30, h=20)),
                ([(80, 100), (50, 120)], BBox(x=50, y=100, w=30, h=20)),
                ([(50, 120), (50, 80)], BBox(x=50, y=80, w=1, h=40)),
            )
        )
    ]
    evidence = PageEvidence(
        page_index=0,
        source_ref=page.source_ref,
        width=400,
        height=200,
        dpi=300,
        effective_dpi=300,
        is_scanned=False,
        role="legend",
        text_spans=[
            TextEvidence(
                id="label",
                text="电磁执行机构",
                bbox=BBox(x=130, y=88, w=110, h=24),
            ),
            TextEvidence(
                id="wrong",
                text="错误文字",
                bbox=BBox(x=280, y=20, w=80, h=24),
            ),
        ],
        paths=paths,
    )
    return page, evidence


def _annotation(*, legend_kind: str = "symbol") -> Annotation:
    wrong = BBox(x=280, y=20, w=80, h=24)
    return Annotation(
        page_index=0,
        kind="equipment",
        source_view="tile",
        bbox_local=wrong,
        bbox_global=wrong,
        label="电磁执行机构",
        attributes={
            "legend_kind": legend_kind,
            "legend_symbol_class": "unclassified_equipment",
        },
        confidence="high",
    )


def test_native_text_and_paths_recover_a_suspicious_model_crop() -> None:
    page, evidence = _page_and_evidence()

    entry = _annotation_to_entry(
        annotation=_annotation(),
        page=page,
        origin=(0, 0),
        cfg_pid=PidConfig(),
        page_evidence=evidence,
    )

    assert entry is not None
    assert entry.image_b64 is not None
    assert entry.crop_method == "native_text_paths"
    assert entry.crop_quality == "recovered"
    assert entry.source_page_index == 0
    assert entry.source_label_bbox == BBox(x=130, y=88, w=110, h=24)
    assert entry.source_bbox is not None
    assert entry.source_bbox.x < 100


def test_abbreviation_does_not_create_a_fake_symbol_thumbnail() -> None:
    page, evidence = _page_and_evidence()

    entry = _annotation_to_entry(
        annotation=_annotation(legend_kind="abbreviation"),
        page=page,
        origin=(0, 0),
        cfg_pid=PidConfig(),
        page_evidence=evidence,
    )

    assert entry is not None
    assert entry.image_b64 is None
    assert entry.crop_quality == "omitted_abbreviation"


def test_duplicate_pixels_for_different_labels_are_omitted() -> None:
    image = Image.new("RGB", (20, 20), "white")
    ImageDraw.Draw(image).line((2, 10, 18, 10), fill="black", width=2)
    encoded = _thumbnail_b64(image, 128)
    entries = [
        LegendEntry(
            label=label,
            symbol_class="line",
            kind="line",
            image_b64=encoded,
            source="legend_extracted",
        )
        for label in ("process line", "signal line")
    ]

    sanitized = _dedupe(entries)

    assert all(entry.image_b64 is None for entry in sanitized)
    assert all(entry.crop_quality == "rejected_duplicate" for entry in sanitized)


def test_old_or_different_extractor_cache_is_not_loaded(tmp_path: Path) -> None:
    path = tmp_path / "legend.cache.json"
    path.write_text(
        LegendPack(
            schema_version="0.1.0",
            source_hash="source",
            extractor_fingerprint="old",
        ).model_dump_json(),
        encoding="utf-8",
    )

    assert _load_cached(path, "source", "current") is None


def test_recovery_cache_requires_matching_inputs_and_verified_negative_rows(tmp_path):
    from diagex.vision.legend_models import LegendRegionCoverage

    path = tmp_path / "legend.cache.json"
    pack = LegendPack(
        source_hash="source",
        extractor_fingerprint="current",
        coverage=[
            LegendRegionCoverage(
                page_index=0,
                source_row_id="row-1",
                bbox=BBox(x=0, y=0, w=20, h=20),
                status="complete",
                entry_count=0,
                verification_passes=1,
            ),
        ],
    )
    path.write_text(pack.model_dump_json())
    assert _load_cached(path, "source", "current") is None
    assert _load_cached(path, "source", "current", allow_partial=True) == pack
    assert _load_cached(path, "changed", "current", allow_partial=True) is None
    assert _load_cached(path, "source", "changed", allow_partial=True) is None
    pack.coverage[0].verification_passes = 2
    path.write_text(pack.model_dump_json())
    assert _load_cached(path, "source", "current") == pack
    pack.coverage[0].status = "partial"
    pack.coverage[0].failure_kind = "transport"
    path.write_text(pack.model_dump_json())
    assert _load_cached(path, "source", "current") is None
    assert _load_cached(path, "source", "current", allow_partial=True) == pack


def test_all_legend_regions_are_scheduled_and_failures_keep_other_regions(monkeypatch):
    from diagex.config import Config
    from diagex.extractors import pid_legend
    from diagex.llm.cost import CostTracker
    from diagex.vision.legend_models import LegendRegionCoverage
    from diagex.vision.tiling import AspectAwareStrategy

    page, evidence = _page_and_evidence()
    cfg = Config()
    cfg.tiling.max_tokens_per_tile = 20
    planned = AspectAwareStrategy(
        max_tokens_per_tile=20,
        overlap_frac=cfg.tiling.overlap_frac,
        token_per_pixel=cfg.tiling.token_per_pixel,
    ).plan(page)
    calls = []

    class Runtime:
        def __init__(self, **kwargs):
            pass

        def run(self, *, state, view_provider, run_cfg):
            assert run_cfg.require_tile_coverage
            detail = view_provider.tiles[0]
            image, _ = view_provider.get_tile(detail.id)
            assert image.size == (state.page.width, state.page.height)
            calls.append(state.page.source_ref)
            state.tile_fetch_counts[detail.id] = 1
            annotation = _annotation()
            annotation.bbox_global = BBox(x=10, y=10, w=20, h=20)
            annotation.label = f"row-{len(calls)}"
            state.annotations.add(annotation)
            if len(calls) == 2:
                raise RuntimeError("temporary failure")
            state.completion_status = "complete"
            state.completion_reason = "finish"

    monkeypatch.setattr(pid_legend, "ReactRuntime", Runtime)
    coverage = []
    entries = pid_legend._extract_from_page(
        page=page,
        region=None,
        client=object(),
        cost_tracker=CostTracker(),
        cfg=cfg,
        page_evidence=None,
        coverage=coverage,
    )
    assert len(calls) == len(planned) > 2
    assert len(entries) == len(planned)  # Even pre-failure annotations survive.
    assert [r.status for r in coverage].count("partial") == 1
    for row, (x, y, w, h) in zip(coverage, planned, strict=True):
        assert row.bbox == BBox(x=x, y=y, w=w, h=h)
    pack = LegendPack(entries=entries, coverage=coverage)
    cached = LegendPack.model_validate_json(pack.model_dump_json())
    assert cached.merge(LegendPack()).coverage == coverage
    assert isinstance(cached.coverage[0], LegendRegionCoverage)


def test_legend_early_finish_is_partial_and_crop_origin_is_preserved(monkeypatch):
    from diagex.config import Config
    from diagex.extractors import pid_legend
    from diagex.llm.cost import CostTracker

    page, _ = _page_and_evidence()

    class Runtime:
        def __init__(self, **kwargs):
            pass

        def run(self, *, state, **kwargs):
            state.completion_status = "complete"  # No detail was inspected.

    monkeypatch.setattr(pid_legend, "ReactRuntime", Runtime)
    coverage = []
    assert (
        pid_legend._extract_from_page(
            page=page,
            region=(100, 50, 150, 100),
            client=object(),
            cost_tracker=CostTracker(),
            cfg=Config(),
            coverage=coverage,
        )
        == []
    )
    assert len(coverage) == 1 and coverage[0].status == "partial"
    assert coverage[0].bbox == BBox(x=100, y=50, w=150, h=100)
