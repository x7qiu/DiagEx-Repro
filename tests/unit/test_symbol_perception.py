"""Native localization contracts independent of drawing identifiers and model calls."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from PIL import Image

from diagex.extractors.evidence_checkpoint import CheckpointStore
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import TextEvidence
from diagex.vision.models import BBox, Tile
from diagex.vision.perception import (
    CandidateDisposition,
    NormalizedBBox,
    PerceivedObject,
    PerceptionBatch,
    _recover_bbox_payload,
    perceive_tile,
    project_batch,
)
from diagex.vision.symbol_candidates import (
    PERCEPTION_DEPENDENT_STAGES,
    SYMBOL_PERCEPTION_VERSION,
    symbol_candidates,
)
from diagex.vision.views import ViewInfo
from tests.unit.test_native_scene import body, rectangle
from tests.unit.test_port_topology import page, path


def circle(name, x, y, radius=18):
    points = [
        (
            round(x + radius * math.cos(i * math.pi / 16)),
            round(y + radius * math.sin(i * math.pi / 16)),
        )
        for i in range(32)
    ]
    return path(name, points + [points[0]], closed=True)


def valve(name, x, y):
    return [
        path(name + "a", [(x, y), (x + 10, y + 6), (x, y + 12), (x, y)], closed=True),
        path(
            name + "b",
            [(x + 10, y + 6), (x + 20, y), (x + 20, y + 12), (x + 10, y + 6)],
            closed=True,
        ),
    ]


def view(box):
    return ViewInfo(
        source_view="tile",
        origin=(box.x, box.y),
        scale_x=0.5,
        scale_y=0.4,
        view_size=(round(box.w * 0.5), round(box.h * 0.4)),
        page_bbox=box,
    )


def project(p, objects, candidates, core=None, current_view=None, decisions=()):
    box = BBox(x=0, y=0, w=p.width, h=p.height)
    batch = PerceptionBatch(objects=objects, candidate_decisions=list(decisions))
    out = project_batch(
        batch=batch,
        page=p,
        tile=Tile(id="tile", page_index=0, bbox=box),
        view_info=current_view or view(box),
        nearby_text_ids={t.id for t in p.text_spans},
        ownership_bbox=core,
        candidates=candidates,
    )
    return out, batch


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_native_valve_instances_are_scale_invariant_and_never_merge_by_text(scale):
    p = page(valve("left", 100, 100) + valve("right", 128, 100))
    p.width *= scale
    p.height *= scale
    for v in p.paths:
        v.points = [(x * scale, y * scale) for x, y in v.points]
        v.bbox = BBox(**{k: value * scale for k, value in v.bbox.model_dump().items()})
    p.text_spans = [
        TextEvidence(
            id="tag1",
            text="XV-1",
            bbox=BBox(x=90 * scale, y=80 * scale, w=30 * scale, h=10 * scale),
        ),
        TextEvidence(
            id="tag2",
            text="XV-1",
            bbox=BBox(x=125 * scale, y=80 * scale, w=30 * scale, h=10 * scale),
        ),
    ]
    cs = symbol_candidates(p)
    assert len(cs) == 2
    assert [c.bbox.x for c in cs] == [100 * scale, 128 * scale]
    assert all(c.shape == "valve_body" for c in cs)
    p.paths.reverse()
    assert symbol_candidates(p) == cs


def test_text_pipe_loop_border_and_valve_callouts_do_not_create_physical_candidates():
    p = page(
        [
            rectangle("border", 0, 0, 999, 799),
            rectangle("loop", 100, 200, 300, 100),
            rectangle("callout", 500, 150, 30, 25),
            circle("psv", 650, 160),
        ]
    )
    p.text_spans = [
        TextEvidence(id="title", text="K-100A", bbox=BBox(x=100, y=80, w=200, h=20)),
        TextEvidence(id="callout-text", text="QV07", bbox=BBox(x=502, y=153, w=26, h=18)),
        TextEvidence(id="psv-text", text="PSV", bbox=BBox(x=640, y=153, w=20, h=10)),
    ]
    assert symbol_candidates(p) == []


def test_full_capsule_survives_head_chord_and_neighboring_valve():
    p = page(body() + valve("valve", 620, 300))
    p.width, p.height = 2000, 1600
    cs = symbol_candidates(p)
    assert len(cs) == 2
    vessel = next(c for c in cs if c.shape == "capsule_body")
    assert vessel.bbox.h >= 398 and vessel.bbox.w == 200
    assert "chord" not in vessel.source_path_ids


def test_circle_in_square_is_one_candidate_with_both_sources():
    p = page([circle("circle", 150, 150), rectangle("frame", 132, 132, 36, 36)])
    p.text_spans = [TextEvidence(id="pi", text="PI", bbox=BBox(x=141, y=140, w=18, h=10))]
    cs = symbol_candidates(p)
    assert len(cs) == 1
    assert set(cs[0].source_path_ids) == {"circle", "frame"}


def test_native_id_ignores_even_invalid_box_and_keeps_complete_clipped_geometry():
    p = page(body() + valve("nearby", 620, 300))
    cs = symbol_candidates(p)
    vessel = next(c for c in cs if c.shape == "capsule_body")
    obj = PerceivedObject(
        kind="equipment", candidate_id=vessel.id, equipment_class="vessel", bbox={"x": 999}
    )
    crop = BBox(x=350, y=240, w=300, h=220)
    out, _ = project(p, [obj], cs, current_view=view(crop), core=crop)
    assert len(out) == 1 and out[0].bbox == vessel.bbox
    assert out[0].attributes["geometry_basis"] == "native_symbol_candidate"


def test_candidate_identity_controls_ownership_not_text_or_model_box():
    p = page([circle("left", 150, 150), circle("right", 350, 150)])
    cs = symbol_candidates(p)
    core = BBox(x=0, y=0, w=250, h=800)
    objects = [
        PerceivedObject(kind="instrument", candidate_id=c.id, printed_tag="PI-1") for c in cs
    ]
    out, _ = project(p, objects, cs, core=core)
    assert [d.bbox for d in out] == [cs[0].bbox]


def test_unknown_candidate_never_snaps_to_nearest_and_missing_candidates_stay_unreviewed():
    p = page([circle("real", 150, 150)])
    cs = symbol_candidates(p)
    out, batch = project(p, [PerceivedObject(kind="instrument", candidate_id="made-up")], cs)
    assert not out and len(batch.rejected_objects) == 1
    assert batch.candidate_reviews[0]["status"] == "unreviewed"


def test_duplicate_or_conflicting_candidate_decisions_never_emit_two_instances():
    p = page([circle("real", 150, 150)])
    cs = symbol_candidates(p)
    obj = PerceivedObject(kind="instrument", candidate_id=cs[0].id)
    for objects, decisions in [
        ([obj, obj.model_copy(update={"printed_tag": "different"})], []),
        (
            [obj],
            [CandidateDisposition(candidate_id=cs[0].id, decision="reject", reason="not a symbol")],
        ),
    ]:
        out, batch = project(p, objects, cs, decisions=decisions)
        assert not out
        assert batch.candidate_reviews[0]["status"] == "uncertain"


def test_round_symbol_cannot_become_an_off_page_connector():
    p = page([circle("real", 150, 150)])
    cs = symbol_candidates(p)
    out, batch = project(p, [PerceivedObject(kind="opc", candidate_id=cs[0].id)], cs)
    assert not out
    assert "classification contradicts" in batch.candidate_reviews[0]["reason"]


def test_unanchored_vector_proposal_is_explicitly_review_only_but_raster_is_supported():
    p = page([])
    obj = PerceivedObject(kind="equipment", bbox=NormalizedBBox(x=0.1, y=0.1, w=0.1, h=0.1))
    out, batch = project(p, [obj], [])
    assert out == []
    assert batch.candidate_reviews[0]["object"]["kind"] == "equipment"
    out, _ = project(p, [obj], None)
    assert "requires_human_review" not in out[0].attributes


def test_coordinate_roundoff_is_not_reported_as_clipping():
    p = page([])
    v = view(BBox(x=100, y=200, w=500, h=500))
    box = {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}
    _, status = _recover_bbox_payload(box, page=p, view_info=v)
    assert status == "normalized"
    recovered, status = _recover_bbox_payload(
        {"x": 0.95, "y": 0.1, "w": 0.055, "h": 0.1}, page=p, view_info=v
    )
    assert status == "normalized_clipped"
    assert recovered["w"] == pytest.approx(0.05)


def test_perception_version_invalidates_raw_and_downstream_but_retains_native_evidence(tmp_path):
    store = CheckpointStore.create(
        run_dir=tmp_path, source_sha256="source", config_sha256="config", run_id="run"
    )
    for stage in ["inspection", *PERCEPTION_DEPENDENT_STAGES]:
        store.write_json_artifact(stage, "item", {"saved": True})
    assert store.ensure_stage_version(
        "symbol_perception", SYMBOL_PERCEPTION_VERSION, invalidate=PERCEPTION_DEPENDENT_STAGES
    )
    assert store.is_done("inspection", "item")
    assert all(not store.is_done(stage, "item") for stage in PERCEPTION_DEPENDENT_STAGES)
    assert all(not store.artifact_path(stage, "item").exists() for stage in PERCEPTION_DEPENDENT_STAGES)
    assert store.artifact_path("inspection", "item").exists()
    assert not store.ensure_stage_version(
        "symbol_perception", SYMBOL_PERCEPTION_VERSION, invalidate=PERCEPTION_DEPENDENT_STAGES
    )


def test_model_selects_candidate_in_existing_single_call_without_regressing_coordinates():
    import json

    p = page([circle("gauge", 150, 150)])
    cs = symbol_candidates(p)
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            prompt = json.loads(kwargs["messages"][0]["content"][1]["text"].split("\n", 1)[1])
            candidate = prompt["native_symbol_candidates"][0]
            assert candidate["marker"] == "C1" and candidate["candidate_id"] == cs[0].id
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {
                            "objects": [
                                {
                                    "kind": "instrument",
                                    "candidate_id": candidate["candidate_id"],
                                    "printed_tag": "PI-1",
                                }
                            ]
                        },
                    }
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=10),
                model="test",
            )

    box = BBox(x=0, y=0, w=p.width, h=p.height)
    image = Image.new("RGB", (500, 320), "white")
    outcome = perceive_tile(
        client=Client(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=p,
        tile=Tile(id="tile", page_index=0, bbox=box),
        view_image=image,
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[],
        step=1,
        candidates=cs,
    )
    assert len(calls) == 1 and outcome.detections[0].bbox == cs[0].bbox
    assert "简体中文" in calls[0]["system"]
    assert outcome.detections[0].label == "PI-1"
    assert image.getextrema() == ((255, 255), (255, 255), (255, 255))
    assert outcome.batch.candidate_reviews[0]["status"] == "selected"


def test_crossed_strokes_need_both_end_bars_and_reject_a_closed_box_with_x():
    diagonals = [path("up", [(100, 100), (140, 120)]), path("down", [(100, 120), (140, 100)])]
    bars = [path("left", [(100, 100), (100, 120)]), path("right", [(140, 100), (140, 120)])]
    assert symbol_candidates(page(diagonals)) == []
    assert symbol_candidates(page(diagonals + bars[:1])) == []
    cs = symbol_candidates(page(diagonals + bars))
    assert len(cs) == 1 and cs[0].bbox == BBox(x=100, y=100, w=40, h=20)
    top_bottom = [path("top", [(100, 100), (140, 100)]), path("bottom", [(100, 120), (140, 120)])]
    assert symbol_candidates(page(diagonals + bars + top_bottom)) == []


def test_seated_valve_requires_a_native_seat_instead_of_filling_an_unproven_gap():
    wings = [
        path("top", [(90, 70), (110, 70)]),
        path("bottom", [(90, 130), (110, 130)]),
        path("a", [(90, 70), (97, 96)]),
        path("b", [(110, 70), (103, 96)]),
        path("c", [(90, 130), (97, 104)]),
        path("d", [(110, 130), (103, 104)]),
    ]
    assert not any(c.shape == "valve_body" for c in symbol_candidates(page(wings)))
    cs = symbol_candidates(page(wings + [circle("seat", 100, 100, radius=5)]))
    valve = next(c for c in cs if c.shape == "valve_body")
    assert valve.bbox == BBox(x=90, y=70, w=20, h=60)
    assert "seat" in valve.source_path_ids


def test_flow_arrow_and_half_dome_are_not_capsule_bodies():
    triangle = path("arrow", [(100, 100), (108, 150), (92, 150), (100, 100)], closed=True)
    dome = path("dome", [(200, 150), (200, 125), (250, 125), (250, 150)], "curve")
    baseline = path("baseline", [(200, 150), (250, 150)])
    assert not any(
        c.shape == "capsule_body" for c in symbol_candidates(page([triangle, dome, baseline]))
    )


def test_selected_candidate_parser_does_not_try_to_recover_ambiguous_model_coordinates():
    from diagex.vision.perception import parse_perception_response

    p = page([circle("gauge", 150, 150)])
    cs = symbol_candidates(p)
    response = SimpleNamespace(
        content=[
            {
                "type": "tool_use",
                "name": "submit_pid_objects",
                "input": {
                    "objects": [
                        {
                            "kind": "instrument",
                            "candidate_id": cs[0].id,
                            "bbox": {"x": 150, "y": 150, "w": 20, "h": 20},
                        }
                    ]
                },
            }
        ]
    )
    batch = parse_perception_response(
        response, page=p, view_info=view(BBox(x=0, y=0, w=500, h=500))
    )
    assert batch.objects[0].bbox is None
    assert batch.objects[0].candidate_id == cs[0].id


@pytest.mark.parametrize(
    "fixture_name,expected",
    [
        ("dexpi-reference", (1579, 1853, 25, 51)),
        ("dexpi-reference", (3738, 1635, 25, 50)),
        ("dexpi-reference", (3927, 1880, 51, 25)),
        ("tennessee1", (107, 201, 47, 23)),
    ],
)
def test_public_fixture_reviewed_valve_bodies_exclude_actuators_and_tag_labels(
    fixture_name, expected
):
    # Bounds independently inspected on source PDF glyphs. Existing broad
    # fixture annotations include labels and actuators, so are not body truth.
    from scripts.evaluate_symbol_perception import fixture

    _, cs = fixture(fixture_name)
    x, y, w, h = expected
    assert any(
        c["shape"] == "valve_body" and BBox(**c["bbox"]).iou(BBox(x=x, y=y, w=w, h=h)) > 0.9
        for c in cs
    )


def test_equivalent_repeated_answers_keep_one_instance_and_provenance():
    p = page([circle("real", 150, 150)])
    cs = symbol_candidates(p)
    obj = PerceivedObject(kind="instrument", candidate_id=cs[0].id, printed_tag="PI-1")
    out, batch = project(p, [obj, obj.model_copy(update={"confidence": "low"})], cs)
    assert len(out) == 1 and out[0].bbox == cs[0].bbox
    assert out[0].attributes["equivalent_observation_count"] == 2
    assert len(batch.candidate_reviews[0]["observations"]) == 2
    assert not batch.rejected_objects


def test_repeated_dispositions_preserve_all_reasons_without_inventing_conflict():
    p = page([circle("real", 150, 150)])
    cs = symbol_candidates(p)
    decisions = [
        CandidateDisposition(candidate_id=cs[0].id, decision="reject", reason=r)
        for r in ["not equipment", "decoration"]
    ]
    out, batch = project(p, [], cs, decisions=decisions)
    assert not out and batch.candidate_reviews[0]["status"] == "reject"
    assert len(batch.candidate_reviews[0]["decisions"]) == 2


@pytest.mark.parametrize("scale", [1, 3])
def test_fragment_circle_with_short_spur_has_a_closed_native_perimeter(scale):
    from diagex.vision.symbol_candidates import _fragment_circles

    outline = circle("circle", 150, 150, radius=8).points
    paths = [
        path(f"arc-{i}", [a, b]) for i, (a, b) in enumerate(zip(outline, outline[1:], strict=False))
    ]
    paths += [path("spur", [(158, 150), (161, 150)])]
    p = page(paths)
    assert _fragment_circles(p) == []  # Untagged sub-contours can be valve seats/arrows.
    p.text_spans = [
        TextEvidence(
            id="pi", text="PI", bbox=BBox(x=146 * scale, y=145 * scale, w=8 * scale, h=8 * scale)
        )
    ]
    p.width *= scale
    p.height *= scale
    for v in p.paths:
        v.points = [(x * scale, y * scale) for x, y in v.points]
    recovered = _fragment_circles(p)
    assert len(recovered) == 1
    assert "spur" not in recovered[0].path_ids
    assert recovered[0].bbox == BBox(x=142 * scale, y=142 * scale, w=16 * scale, h=16 * scale)
    p.paths = p.paths[1:]
    assert _fragment_circles(p) == []  # An open arc never becomes a whole circle.


def test_closed_frame_survives_overlapping_strokes_and_diamond_needs_native_text():
    p = page(
        [
            rectangle("frame", 120, 120, 36, 36),
            rectangle("duplicate-frame", 120, 120, 36, 36),
            path("signal", [(110, 138), (170, 138)]),
            path(
                "diamond", [(230, 120), (248, 138), (230, 156), (212, 138), (230, 120)], closed=True
            ),
        ]
    )
    p.text_spans = [
        TextEvidence(id="pi", text="PI", bbox=BBox(x=129, y=125, w=18, h=10)),
        TextEvidence(id="i", text="I", bbox=BBox(x=226, y=130, w=8, h=10)),
    ]
    cs = symbol_candidates(p)
    assert len(cs) == 2 and all(c.shape == "instrument_frame" for c in cs)
    assert set(cs[0].source_path_ids) >= {"frame", "duplicate-frame"}
    p.text_spans = p.text_spans[:1]
    assert len(symbol_candidates(p)) == 1


@pytest.mark.parametrize("repair", ["resolve", "conflict", "malformed", "unavailable"])
def test_one_targeted_retry_preserves_good_detections_and_never_loops(repair):
    import json

    p = page([circle("left", 100, 100), circle("middle", 200, 100), circle("right", 300, 100)])
    cs = symbol_candidates(p)
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            payload = json.loads(kwargs["messages"][0]["content"][1]["text"].split("\n")[-1])
            assert len([b for b in kwargs["messages"][0]["content"] if b["type"] == "image"]) == (2 if len(calls) == 1 else 4)
            if len(calls) == 1:
                objects = [
                    {"kind": "instrument", "candidate_id": cs[0].id, "printed_tag": "PI-1"},
                    {"kind": "instrument", "candidate_id": cs[1].id},
                ]
                decisions = [
                    {"candidate_id": cs[1].id, "decision": "reject", "reason": "contradiction"}
                ]
            else:
                assert [c["candidate_id"] for c in payload["native_symbol_candidates"]] == [
                    cs[1].id,
                    cs[2].id,
                ]
                if repair == "unavailable":
                    raise RuntimeError("service unavailable")
                if repair == "malformed":
                    return SimpleNamespace(
                        content=[{"type": "text", "text": "invalid answer"}], usage=None
                    )
                objects = [{"kind": "instrument", "candidate_id": cs[1].id}]
                decisions = (
                    []
                    if repair == "resolve"
                    else [
                        {
                            "candidate_id": cs[1].id,
                            "decision": "reject",
                            "reason": "still contradictory",
                        }
                    ]
                )
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {"objects": objects, "candidate_decisions": decisions},
                    }
                ],
                usage=None,
            )

    box = BBox(x=0, y=0, w=p.width, h=p.height)
    outcome = perceive_tile(
        client=Client(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=p,
        tile=Tile(id="tile", page_index=0, bbox=box),
        view_image=Image.new("RGB", (500, 320), "white"),
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[],
        step=1,
        candidates=cs,
    )
    assert len(calls) == outcome.attempts == 2
    assert outcome.detections[0].label == "PI-1"
    assert len(outcome.detections) == (2 if repair == "resolve" else 1)
    by_id = {r["candidate_id"]: r for r in outcome.batch.candidate_reviews}
    assert by_id[cs[2].id]["status"] == "unreviewed"
    if repair == "resolve":
        assert by_id[cs[1].id]["previous_review"]["status"] == "uncertain"
        assert not outcome.batch.rejected_objects


def unified_batch(rows, proposals=()):
    from diagex.vision.perception import parse_perception_response

    return parse_perception_response(
        SimpleNamespace(
            content=[
                {
                    "type": "tool_use",
                    "name": "submit_pid_objects",
                    "input": {"candidate_results": rows, "proposals": list(proposals)},
                }
            ]
        )
    )


def test_single_outcomes_retain_untagged_components_at_their_native_locations():
    p = page([circle("motor", 150, 150)] + valve("isolation", 210, 150))
    cs = symbol_candidates(p)
    batch = unified_batch(
        [
            {
                "candidate_id": c.id,
                "decision": "symbol",
                "symbol": {
                    "kind": "equipment",
                    "equipment_class": "motor" if c.shape == "round_symbol" else "valve",
                },
            }
            for c in cs
        ]
    )
    box = BBox(x=0, y=0, w=p.width, h=p.height)
    ds = project_batch(
        batch=batch,
        page=p,
        tile=Tile(id="tile", page_index=0, bbox=box),
        view_info=view(box),
        nearby_text_ids=set(),
        ownership_bbox=box,
        candidates=cs,
    )
    assert len(ds) == 2
    assert [d.bbox for d in ds] == [c.bbox for c in cs]
    assert {d.attributes["equipment_class"] for d in ds} == {"motor", "valve"}
    assert all(d.label == "" for d in ds)


@pytest.mark.parametrize(
    "bad",
    [
        {
            "decision": "non_symbol",
            "reason": "Attached to an assembly",
            "non_symbol_basis": "part_of_equipment",
        },
        {"decision": "symbol", "symbol": {"kind": "instrument", "candidate_id": "neighbor"}},
        {"decision": "symbol", "symbol": {"kind": "instrument", "bbox": {"x": 1}}},
        {"decision": "uncertain", "reason": "uncertain", "symbol": {"kind": "instrument"}},
        {"decision": "symbol", "symbol": {"kind": "instrument"}, "non_symbol_basis": "text_only"},
    ],
)
def test_invalid_outcome_cannot_overwrite_identity_or_mix_rejection_with_symbol(bad):
    batch = unified_batch(
        [
            {"candidate_id": "bad", **bad},
            {"candidate_id": "good", "decision": "symbol", "symbol": {"kind": "instrument"}},
        ]
    )
    assert [o.candidate_id for o in batch.objects] == ["good"]
    assert batch.candidate_decisions[0].decision == "uncertain"
    assert batch.candidate_decisions[0].reason.startswith("Invalid candidate result:")
    assert batch.rejected_objects[0]["candidate_id"] == "bad"


@pytest.mark.parametrize("second", ["symbol", "non_symbol"])
def test_duplicate_new_results_are_uncertain_even_if_equivalent(second):
    a = {"candidate_id": "same", "decision": "symbol", "symbol": {"kind": "instrument"}}
    b = (
        a
        if second == "symbol"
        else {
            "candidate_id": "same",
            "decision": "non_symbol",
            "reason": "line",
            "non_symbol_basis": "pipe_or_signal_line",
        }
    )
    batch = unified_batch([a, b])
    assert not batch.objects
    assert len(batch.candidate_decisions) == 1
    assert batch.candidate_decisions[0].decision == "uncertain"
    assert "duplicate outcome" in batch.candidate_decisions[0].reason


def test_proposals_cannot_be_a_second_channel_for_native_candidate_decisions():
    batch = unified_batch([], [{"candidate_id": "native", "kind": "instrument"}])
    assert not batch.objects
    assert "must not claim" in batch.rejected_objects[0]["error"]


@pytest.mark.parametrize(
    "mode, allowance, thinking",
    [
        ("auto", 6000, "disabled"),
        ("disabled", 6000, "disabled"),
        ("enabled", 6000, "disabled"),
    ],
)
def test_stage_reasoning_policy_and_one_targeted_contract_repair(mode, allowance, thinking):
    import json

    p = page([circle("left", 100, 100), circle("right", 200, 100)])
    cs = symbol_candidates(p)
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["max_tokens"] == allowance
            assert kwargs["thinking"]["type"] == thinking
            assert kwargs["reasoning_mode_override"] == "disabled"
            payload = json.loads(kwargs["messages"][0]["content"][1]["text"].split("\n")[-1])
            assert "ownership_core_normalized" not in payload
            ids = [c["candidate_id"] for c in payload["native_symbol_candidates"]]
            if len(calls) == 1:
                rows = [
                    {
                        "candidate_id": cs[0].id,
                        "decision": "symbol",
                        "symbol": {"kind": "instrument"},
                    },
                    {
                        "candidate_id": cs[1].id,
                        "decision": "non_symbol",
                        "reason": "Outside crop",
                        "non_symbol_basis": "outside_core",
                    },
                ]
            else:
                assert ids == [cs[1].id]
                rows = [
                    {
                        "candidate_id": cs[1].id,
                        "decision": "symbol",
                        "symbol": {"kind": "equipment", "equipment_class": "motor"},
                    }
                ]
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {"candidate_results": rows},
                    }
                ],
                usage=None,
            )

    box = BBox(x=0, y=0, w=p.width, h=p.height)
    result = perceive_tile(
        client=Client(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=p,
        tile=Tile(id="tile", page_index=0, bbox=box),
        view_image=Image.new("RGB", (500, 320), "white"),
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[],
        step=1,
        candidates=cs,
        reasoning_mode=mode,
    )
    assert len(calls) == 2
    assert len(result.detections) == 2
    assert not result.batch.rejected_objects
    assert all(r["status"] == "selected" for r in result.batch.candidate_reviews)


def test_reasoning_truncation_preserves_initial_uncertainty_without_a_third_call():
    p = page([circle("instrument", 150, 150)])
    cs = symbol_candidates(p)
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    content=[{"type": "tool_use", "name": "submit_pid_objects", "input": {
                        "candidate_results": [{"candidate_id": cs[0].id, "decision": "uncertain", "reason": "Glyph ambiguous"}]
                    }}],
                    usage=None,
                )
            assert kwargs["max_tokens"] == 16000
            assert kwargs["reasoning_mode_override"] == "enabled"
            return SimpleNamespace(
                content=[{"type": "thinking", "thinking": "budget exhausted"}],
                stop_reason="max_tokens",
                usage=None,
            )

    box = BBox(x=0, y=0, w=p.width, h=p.height)
    outcome = perceive_tile(
        client=Client(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=p,
        tile=Tile(id="tile", page_index=0, bbox=box),
        view_image=Image.new("RGB", (500, 320), "white"),
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[],
        step=1,
        candidates=cs,
        reasoning_mode="enabled",
    )
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 6000
    assert len(outcome.detections) == 0
    assert outcome.batch.candidate_reviews[0]["status"] == "uncertain"
    assert outcome.recovery_diagnostics[-1]["reasoning_budget_exhausted"] is True


def test_non_symbol_evidence_and_malformed_proposal_do_not_erase_valid_disposition():
    batch = unified_batch(
        [
            {
                "candidate_id": "text",
                "decision": "non_symbol",
                "reason": "Only printed words, no symbol outline",
                "non_symbol_basis": "text_only",
            }
        ],
        [{"kind": "instrument"}],
    )
    assert not batch.objects
    assert batch.candidate_decisions[0].decision == "reject"
    assert batch.candidate_decisions[0].non_symbol_basis == "text_only"
    assert batch.rejected_objects
