from __future__ import annotations

from itertools import permutations

import pytest

from diagex.extractors.evidence_checkpoint import CheckpointStore, find_resumable_run_with_report
from diagex.vision.evidence import PageEvidence, PathEvidence, TextEvidence
from diagex.vision.fusion import fuse_objects
from diagex.vision.instance_matching import (
    FUSION_DEPENDENT_STAGES,
    FUSION_VERSION,
    match_instances,
)
from diagex.vision.models import BBox, ReconciledEdge, ReconciledGraph, ReconciledNode
from diagex.vision.perception import DetectionRecord
from diagex.vision.quality import assess_quality
from scripts.evaluate_fusion import fixture_audit


def page(paths=(), texts=()):
    return PageEvidence(
        page_index=0,
        source_ref="test#page=1",
        width=1000,
        height=1000,
        dpi=100,
        effective_dpi=100,
        is_scanned=False,
        paths=list(paths),
        text_spans=list(texts),
    )


def detection(id, box, label="", family="valve", kind="equipment"):
    attrs = {"valve_type": "other"} if family == "valve" else {"equipment_class": family}
    return DetectionRecord(
        id=id,
        page_index=0,
        tile_id=id,
        kind=kind,
        label=label,
        bbox=BBox(x=box[0], y=box[1], w=box[2], h=box[3]),
        confidence="high",
        attributes=attrs,
    )


def line(id, a, b):
    return PathEvidence(
        id=id,
        page_index=0,
        origin="pdf_vector",
        primitive="line",
        points=[a, b],
        bbox=BBox(
            x=min(a[0], b[0]),
            y=min(a[1], b[1]),
            w=max(1, abs(a[0] - b[0])),
            h=max(1, abs(a[1] - b[1])),
        ),
    )


def test_large_vessel_cannot_swallow_valves_or_instruments():
    vessel = detection("vessel", (300, 200, 200, 500), "V-101", "vessel")
    small = [
        detection("valve", (480, 220, 20, 20)),
        detection("gauge", (490, 200, 40, 40), "PI101", "", "instrument"),
    ]
    fused = fuse_objects(
        source_name="test", pages=[page()], detections=[vessel, *small], per_page_status={0: "ok"}
    )
    assert len(fused.graph.nodes) == 3
    node = next(n for n in fused.graph.nodes if n.label == "V-101")
    assert node.bbox_global == vessel.bbox
    assert node.attributes["equipment_class"] == "vessel"
    assert "valve_type" not in node.attributes


def test_nearby_unlabelled_valves_remain_separate():
    ds = [detection("a", (100, 100, 30, 30)), detection("b", (132, 100, 30, 30))]
    assert len(match_instances(ds, {0: page()})[0]) == 2


def test_distinct_tags_and_repeated_package_labels_remain_separate():
    ds = [
        detection("a", (100, 100, 50, 50), "PI101", "", "instrument"),
        detection("b", (104, 100, 50, 50), "PI102", "", "instrument"),
        detection("c", (400, 100, 50, 50), "PI101", "", "instrument"),
    ]
    clusters, conflicts = match_instances(ds, {0: page()})
    assert len(clusters) == 3
    assert len(conflicts) == 1
    assert conflicts[0]["reason"] == "different_printed_identities"


def test_native_tags_distinguish_unlabelled_overlapping_observations():
    ds = [detection("a", (100, 100, 100, 100)), detection("b", (120, 100, 100, 100))]
    texts = [
        TextEvidence(id="ta", text="QV01", bbox=BBox(x=101, y=110, w=15, h=12)),
        TextEvidence(id="tb", text="QV02", bbox=BBox(x=203, y=110, w=15, h=12)),
    ]
    clusters, _ = match_instances(ds, {0: page(texts=texts)})
    assert len(clusters) == 2


def test_crop_duplicates_choose_an_observed_box_and_keep_provenance():
    ds = [detection("a", (100, 100, 80, 80), "QV-001"), detection("b", (104, 102, 80, 80), "QV001")]
    fused = fuse_objects(
        source_name="test", pages=[page()], detections=ds, per_page_status={0: "ok"}
    )
    assert len(fused.graph.nodes) == 1
    node = fused.graph.nodes[0]
    assert node.bbox_global in [d.bbox for d in ds]
    assert node.attributes["fusion_evidence"]["member_detection_ids"] == ["a", "b"]


def test_cluster_checks_all_members_and_is_input_order_independent():
    ds = [
        detection("a", (100, 100, 100, 100)),
        detection("b", (125, 100, 100, 100)),
        detection("c", (150, 100, 100, 100)),
    ]
    results = []
    for order in permutations(ds):
        clusters, conflicts = match_instances(list(order), {0: page()})
        results.append([tuple(d.id for d in c.detections) for c in clusters])
        assert len(clusters) == 2
        assert conflicts
    assert all(r == results[0] for r in results)


def test_clipped_equipment_combines_only_with_a_closed_native_outline():
    ds = [
        detection("top", (100, 100, 100, 90), "V101", "vessel"),
        detection("bottom", (100, 190, 100, 110), "V101", "vessel"),
    ]
    paths = [
        line("left", (100, 100), (100, 300)),
        line("right", (200, 100), (200, 300)),
        line("top", (100, 100), (200, 100)),
        line("bottom", (100, 300), (200, 300)),
    ]
    clusters, _ = match_instances(ds, {0: page(paths)})
    assert len(clusters) == 1
    assert clusters[0].geometry_basis == "native_symbol"
    assert set(clusters[0].path_ids) == {p.id for p in paths}
    assert clusters[0].bbox.h >= 200
    assert len(match_instances(ds, {0: page(paths[:2])})[0]) == 2


def test_containment_and_a_shared_pipe_are_not_same_symbol_evidence():
    ds = [
        detection("a", (100, 100, 100, 100), "V101", "vessel"),
        detection("b", (130, 140, 30, 30), "V101", "vessel"),
    ]
    paths = [line("pipe", (0, 150), (1000, 150))]
    assert len(match_instances(ds, {0: page(paths)})[0]) == 2


def test_shared_native_symbol_footprint_supports_offset_boxes():
    paths = [line("x", (145, 120), (185, 120)), line("y", (185, 120), (185, 175))]
    ds = [detection("a", (100, 100, 100, 100)), detection("b", (140, 100, 100, 100))]
    assert ds[0].bbox.iou(ds[1].bbox) < 0.55
    clusters, _ = match_instances(ds, {0: page(paths)})
    assert len(clusters) == 1
    assert set(clusters[0].path_ids) == {"x", "y"}


def test_unproven_complementary_crops_remain_reviewable():
    ds = [
        detection("a", (100, 100, 100, 90), "V101", "vessel"),
        detection("b", (100, 195, 100, 100), "V101", "vessel"),
    ]
    clusters, conflicts = match_instances(ds, {0: page()})
    assert len(clusters) == 2
    assert conflicts[0]["reason"] == "insufficient_symbol_geometry"


def test_identified_vessel_tag_is_not_reassigned_to_neighbouring_valve():
    vessel = detection("vessel", (300, 200, 200, 500), "2401-V-001", "vessel")
    valve = detection("valve", (460, 400, 25, 25))
    text = TextEvidence(id="tag", text="2401-V-001", bbox=BBox(x=400, y=370, w=90, h=20))
    result = fuse_objects(
        source_name="test",
        pages=[page(texts=[text])],
        detections=[vessel, valve],
        per_page_status={0: "ok"},
    )
    assert sum(n.label == "2401-V-001" for n in result.graph.nodes) == 1


@pytest.mark.parametrize("stem,count", [("dexpi-reference", 13), ("tennessee1", 58)])
def test_reviewed_vector_instances_survive_crop_duplicates(stem, count):
    report = fixture_audit(stem)
    assert report["native_path_count"] > 0
    assert report["truth_instance_count"] == count
    assert report["output_instance_count"] == count
    assert report["false_merge_count"] == 0
    assert report["retained_duplicate_count"] == 0


def test_fusion_version_invalidates_only_dependent_checkpoints(tmp_path):
    store = CheckpointStore.create(
        run_dir=tmp_path,
        source_sha256="source",
        config_sha256="same-perception-config",
        run_id="test",
    )
    for stage in ("inspection", "perception", *FUSION_DEPENDENT_STAGES):
        store.write_json_artifact(stage, "item", {"saved": True})
    assert store.ensure_stage_version(
        "object_fusion", FUSION_VERSION, invalidate=FUSION_DEPENDENT_STAGES
    )
    assert all(store.is_done(s, "item") for s in ("inspection", "perception"))
    assert not any(store.is_done(s, "item") for s in FUSION_DEPENDENT_STAGES)
    assert not store.ensure_stage_version(
        "object_fusion", FUSION_VERSION, invalidate=FUSION_DEPENDENT_STAGES
    )


def test_completed_run_can_upgrade_fusion_without_repeating_perception(tmp_path):
    store = CheckpointStore.create(
        run_dir=tmp_path / "run", source_sha256="source", config_sha256="unchanged", run_id="run"
    )
    store.write_json_artifact("perception", "tile", {"detections": []})
    store.set_status("complete")
    found, report = find_resumable_run_with_report(
        runs_root=tmp_path,
        source_sha256="source",
        config_sha256="unchanged",
        required_stage_versions={"object_fusion": FUSION_VERSION},
    )
    assert found is not None
    assert found.is_done("perception", "tile")
    assert "upgrade" in report["reason"]
    found.ensure_stage_version("object_fusion", FUSION_VERSION, invalidate=FUSION_DEPENDENT_STAGES)
    found.set_status("complete")
    assert (
        find_resumable_run_with_report(
            runs_root=tmp_path,
            source_sha256="source",
            config_sha256="unchanged",
            required_stage_versions={"object_fusion": FUSION_VERSION},
        )[0]
        is None
    )


def test_upgrade_copies_raw_evidence_and_preserves_original_review(tmp_path):
    old = CheckpointStore.create(
        run_dir=tmp_path / "old", source_sha256="source", config_sha256="config", run_id="old"
    )
    old.write_json_artifact("perception", "tile", {"detections": []})
    old.write_json_artifact("topology", "page", {"edges": []})
    (old.run_dir / "evidence").mkdir()
    (old.run_dir / "evidence/page.json").write_text('{"native": true}')
    (old.run_dir / "review").mkdir()
    (old.run_dir / "review/events.jsonl").write_text('{"reviewed": true}\n')
    old.set_status("complete")
    original = {
        p.relative_to(old.run_dir): p.read_bytes() for p in old.run_dir.rglob("*") if p.is_file()
    }
    new = CheckpointStore.create(
        run_dir=tmp_path / "new", source_sha256="source", config_sha256="config", run_id="new"
    )
    new.seed_raw_evidence_from(old)
    assert new.is_done("perception", "tile")
    assert not new.is_done("topology", "page")
    assert (new.run_dir / "evidence/page.json").read_text() == '{"native": true}'
    assert not (new.run_dir / "review").exists()
    new.write_json_artifact("perception", "tile", {"changed": True})
    assert original == {
        p.relative_to(old.run_dir): p.read_bytes() for p in old.run_dir.rglob("*") if p.is_file()
    }


def test_upgrade_seeds_the_precreated_evidence_directory(tmp_path):
    from diagex.config import Config
    from diagex.extractors.pid_evidence import _prepare_v2_run_dir

    source = CheckpointStore.create(
        run_dir=tmp_path / "source", source_sha256="source",
        config_sha256="config", run_id="source",
    )
    (source.run_dir / "evidence").mkdir()
    native = source.run_dir / "evidence/page-0001.json"
    native.write_text('{"native": true}')
    source.write_json_artifact("inspection", "page-0001", {"role": "pid"})
    source.write_json_artifact("perception", "tile", {"detections": []})
    source.set_status("complete")

    run_dir, run_id = _prepare_v2_run_dir(
        Config(runs_dir=tmp_path / "runs"), "drawing", "test-model",
    )
    assert (run_dir / "evidence").is_dir()
    target = CheckpointStore.create(
        run_dir=run_dir, source_sha256="source", config_sha256="config", run_id=run_id,
    )
    target.seed_raw_evidence_from(source)

    restored = CheckpointStore.load(run_dir)
    assert restored.is_done("inspection", "page-0001")
    assert restored.is_done("perception", "tile")
    assert (run_dir / "evidence/page-0001.json").read_bytes() == native.read_bytes()
    (run_dir / "evidence/page-0001.json").write_text("changed in new run")
    assert native.read_text() == '{"native": true}'


def test_completed_upgrade_does_not_repeatedly_select_older_run(tmp_path):
    for name in ("2026-09-03", "2026-09-05"):
        store = CheckpointStore.create(
            run_dir=tmp_path / name, source_sha256="source", config_sha256="config", run_id=name
        )
        if name == "2026-09-05":
            store.ensure_stage_version("object_fusion", FUSION_VERSION)
        store.set_status("complete")
    assert (
        find_resumable_run_with_report(
            runs_root=tmp_path,
            source_sha256="source",
            config_sha256="config",
            required_stage_versions={"object_fusion": FUSION_VERSION},
        )[0]
        is None
    )


def test_quality_distinguishes_accepted_connectivity():
    nodes = [
        ReconciledNode(
            id=id,
            kind="equipment",
            label=id,
            page_index=0,
            bbox_global=BBox(x=10, y=10, w=20, h=20),
            confidence="high",
        )
        for id in ("a", "b", "c")
    ]
    edges = [
        ReconciledEdge(id="ab", from_node="a", to_node="b", line_type="process", confidence="high"),
        ReconciledEdge(
            id="bc",
            from_node="b",
            to_node="c",
            line_type="process",
            confidence="low",
            attributes={"provisional_review_only": True},
        ),
    ]
    graph = ReconciledGraph(source_path="test", nodes=nodes, edges=edges)
    metrics = assess_quality(graph=graph, pages=[page()], topology=[]).metrics
    assert metrics["accepted_edge_count"] == 1
    assert metrics["provisional_edge_count"] == 1
    assert metrics["accepted_isolated_node_count"] == 1
    assert metrics["isolated_node_count"] == 0
