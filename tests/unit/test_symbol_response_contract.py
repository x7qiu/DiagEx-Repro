"""Production-path regressions for incomplete candidate rows and runaway retries."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from PIL import Image

from diagex.config import Config, LLMConfig, SymbolPerceptionConfig
from diagex.extractors.evidence_checkpoint import CheckpointStore
from diagex.extractors.pid_evidence import _run_perception
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.models import BBox, DiagramPage, Tile
from diagex.vision.perception import (
    _SUBMIT_TOOL,
    CandidateResult,
    PerceptionRunGuard,
    perceive_tile,
)
from diagex.vision.symbol_candidates import symbol_candidates
from tests.unit.test_port_topology import page
from tests.unit.test_symbol_perception import circle, unified_batch, view


def reply(rows):
    return SimpleNamespace(
        content=[
            {"type": "tool_use", "name": "submit_pid_objects", "input": {"candidate_results": rows}}
        ],
        usage=None,
    )


def test_failed_source_legend_blocks_calls_but_semantic_ambiguity_is_separate(
    tmp_path, monkeypatch
):
    from diagex.extractors.pid_evidence import _legend_prerequisite_error
    from diagex.vision.legend_models import LegendPack, LegendRegionCoverage

    box = BBox(x=0, y=0, w=500, h=500)
    failed = LegendRegionCoverage(
        page_index=0, bbox=box, status="partial", failure_kind="transport"
    )
    assert _legend_prerequisite_error(LegendPack(coverage=[failed]), "explicit_pages")
    assert (
        _legend_prerequisite_error(
            LegendPack(coverage=[failed.model_copy(update={"failure_kind": "ambiguity"})]),
            "explicit_pages",
        )
        is None
    )
    p = page([circle("symbol", 150, 150)])
    p.role = "pid"
    rendered = DiagramPage(
        page_index=0,
        source_ref="test",
        width=p.width,
        height=p.height,
        dpi=72,
        effective_dpi=72,
        is_scanned=False,
        image=Image.new("RGB", (p.width, p.height), "white"),
    )
    monkeypatch.setattr("diagex.vision.loader.iter_pages", lambda source: iter([rendered]))

    class NoCalls:
        def messages_create(self, **kwargs):
            raise AssertionError("An incomplete source legend must block paid symbol calls")

    ds, statuses, counts, reason = _run_perception(
        source=None,
        pages=[p],
        cfg=Config(),
        client=NoCalls(),
        cost=CostTracker(),
        reporter=NullReporter(),
        store=None,
        legend_summary=[],
        run_dir=tmp_path,
        prior_cost={},
        prerequisite_error="Legend source rows require recovery",
    )
    assert not ds and counts["submit_pid_objects"] == 0 and reason
    assert statuses[0] == "partial"
    reviews = json.loads((tmp_path / "perception.review.json").read_text())["reviews"]
    assert len(reviews) == 1 and reviews[0]["status"] == "unreviewed"


def test_flat_wire_schema_keeps_candidate_identity_outside_provider_conditionals():
    schema = _SUBMIT_TOOL["input_schema"]
    text = json.dumps(schema)
    for keyword in ('"oneOf"', '"anyOf"', '"allOf"', '"$ref"', '"not"'):
        assert keyword not in text
    row = schema["properties"]["candidate_results"]["items"]
    assert set(row["required"]) == {"candidate_id", "decision"}
    assert {"candidate_id", "decision", "kind", "equipment_class"} <= row["properties"].keys()
    assert "bbox" not in row["properties"] and "symbol" not in row["properties"]
    assert "non_symbol_basis" not in row["properties"]


@pytest.mark.parametrize(
    "bad",
    [
        {"decision": "symbol"},
        {"candidate_id": "bad", "decision": "symbol"},
        {
            "candidate_id": "bad",
            "decision": "non_symbol",
            "reason": "line",
            "kind": "instrument",
            "non_symbol_basis": "pipe_or_signal_line",
        },
        {
            "candidate_id": "bad",
            "decision": "symbol",
            "kind": "instrument",
            "symbol": {"kind": "instrument"},
        },
        {"candidate_id": "bad", "decision": "symbol", "kind": "equipment", "bbox": {"x": 0}},
    ],
)
def test_flat_invalid_rows_remain_rejected_without_losing_valid_siblings(bad):
    good = {
        "candidate_id": "good",
        "decision": "symbol",
        "kind": "equipment",
        "equipment_class": "motor",
    }
    with pytest.raises(ValueError):
        CandidateResult.model_validate(bad)
    batch = unified_batch([bad, good])
    assert [o.candidate_id for o in batch.objects] == ["good"]
    assert batch.rejected_objects


def test_rejection_decision_is_atomic_but_legacy_contradictions_are_still_rejected():
    row = {"candidate_id": "line", "decision": "reject_line", "reason": "Only a pipe stroke"}
    result = CandidateResult.model_validate(row)
    assert result.decision == "non_symbol" and result.non_symbol_basis == "pipe_or_signal_line"
    with pytest.raises(ValueError):
        CandidateResult.model_validate({**row, "kind": "instrument"})
    with pytest.raises(ValueError):
        CandidateResult.model_validate(
            {
                "candidate_id": "bad",
                "decision": "symbol",
                "kind": "instrument",
                "non_symbol_basis": "text_only",
            }
        )


def test_invalid_unanchored_proposal_does_not_destroy_an_empty_native_crop():
    from diagex.vision.perception import parse_perception_response

    response = SimpleNamespace(
        content=[
            {
                "type": "tool_use",
                "name": "submit_pid_objects",
                "input": {
                    "candidate_results": [],
                    "proposals": [
                        {"kind": "equipment", "bbox": {"x": 0, "y": 0.75, "w": 0, "h": 0}}
                    ],
                },
            }
        ]
    )
    batch = parse_perception_response(response)
    assert not batch.objects and not batch.candidate_decisions
    assert len(batch.rejected_objects) == 1
    assert batch.rejected_objects[0]["object"]["bbox"]["w"] == 0


@pytest.mark.parametrize(
    "label,actuation", [("solenoid_valve", "solenoid"), ("motor-operated valve", "electric_motor")]
)
def test_actuated_valve_class_alias_preserves_body_and_actuation(label, actuation):
    from diagex.vision.perception import PerceivedObject, _candidate_kind_compatible
    from diagex.vision.symbol_candidates import SymbolCandidate

    candidate = SymbolCandidate(
        id="valve",
        page_index=0,
        shape="valve_body",
        bbox=BBox(x=1, y=1, w=10, h=10),
        source_path_ids=["body"],
    )
    obj = PerceivedObject(candidate_id="valve", kind="equipment", equipment_class=label)
    assert obj.equipment_class == "valve" and obj.actuation == actuation
    assert obj.attributes["original_equipment_class"] == label
    assert _candidate_kind_compatible(candidate, obj)
    actuator = PerceivedObject(candidate_id="valve", kind="equipment", equipment_class="actuator")
    assert not _candidate_kind_compatible(candidate, actuator)


def test_geometry_conflict_gets_one_targeted_followup_without_losing_valid_sibling():
    from tests.unit.test_symbol_perception import valve

    p = page(valve("v", 150, 150) + [circle("m", 250, 150)])
    cs = symbol_candidates(p)
    vc = next(c for c in cs if c.shape == "valve_body")
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return reply(
                    [
                        {
                            "candidate_id": c.id,
                            "decision": "symbol",
                            "kind": "equipment",
                            "equipment_class": "actuator" if c.id == vc.id else "motor",
                        }
                        for c in cs
                    ]
                )
            assert kwargs["thinking"] == {"type": "adaptive"}
            prompt = json.loads(kwargs["messages"][0]["content"][1]["text"].split("\n")[-1])
            assert [c["candidate_id"] for c in prompt["native_symbol_candidates"]] == [vc.id]
            assert "geometry" in prompt["native_symbol_candidates"][0]["previous_issue"]
            return reply(
                [
                    {
                        "candidate_id": vc.id,
                        "decision": "symbol",
                        "kind": "equipment",
                        "equipment_class": "valve",
                        "actuation": "solenoid",
                    }
                ]
            )

    outcome = perceive_tile(
        client=Client(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=p,
        tile=Tile(id="t", page_index=0, bbox=BBox(x=0, y=0, w=500, h=500)),
        view_image=Image.new("RGB", (500, 500), "white"),
        view_info=view(BBox(x=0, y=0, w=500, h=500)),
        ownership_bbox=BBox(x=0, y=0, w=500, h=500),
        legend_summary=[],
        step=1,
        candidates=cs,
        reasoning_mode="enabled",
    )
    assert len(calls) == 2
    assert {d.attributes["equipment_class"] for d in outcome.detections} == {"motor", "valve"}


@pytest.mark.parametrize("mode, expected_calls", [("enabled", 2), ("disabled", 1)])
@pytest.mark.parametrize("negative", [True, False])
def test_body_or_rejected_candidate_recheck_is_bounded_and_never_auto_accepts(
    mode, expected_calls, negative
):
    from diagex.vision.symbol_candidates import SymbolCandidate

    p = page([])
    box = BBox(x=0, y=0, w=500, h=500)
    candidate = SymbolCandidate(
        id="body",
        page_index=0,
        shape="capsule_body",
        bbox=BBox(x=100, y=100, w=100, h=40),
        source_path_ids=["body-ink"],
    )
    row = (
        {
            "candidate_id": candidate.id,
            "decision": "reject_no_glyph",
            "reason": "No separate outline",
        }
        if negative
        else {
            "candidate_id": candidate.id,
            "decision": "symbol",
            "kind": "equipment",
            "equipment_class": "vessel",
            "confidence": "high",
        }
    )
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                assert kwargs["reasoning_mode_override"] == "enabled"
                content = kwargs["messages"][0]["content"]
                assert any("Enlarged source detail for body" in x.get("text", "") for x in content)
            return reply([row])

    outcome = perceive_tile(
        client=Client(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=p,
        tile=Tile(id="t", page_index=0, bbox=box),
        view_image=Image.new("RGB", (250, 200), "white"),
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[],
        step=1,
        candidates=[candidate],
        reasoning_mode=mode,
    )
    assert len(calls) == expected_calls
    assert len(outcome.detections) == (0 if negative else 1)


def test_real_sdk_stream_preserves_flat_candidate_fields_and_persists_diagnostics():
    p = page([circle("motor", 150, 150)])
    cs = symbol_candidates(p)
    payload = {
        "candidate_results": [
            {
                "candidate_id": cs[0].id,
                "decision": "symbol",
                "kind": "equipment",
                "equipment_class": "motor",
            }
        ]
    }
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/api/v1/messages"
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "test",
                    "type": "message",
                    "role": "assistant",
                    "model": "test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool",
                    "name": "submit_pid_objects",
                    "input": {},
                },
            },
        ]
        encoded = json.dumps(payload)
        events.extend(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": encoded[i : i + 7]},
            }
            for i in range(0, len(encoded), 7)
        )
        events.extend(
            [
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 12},
                },
                {"type": "message_stop"},
            ]
        )
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = LLMClient(LLMConfig(model="test", anthropic_api_key="test"))
    client._client.close()
    client._client = anthropic.Anthropic(
        api_key="test",
        base_url="https://test.invalid/api",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    diagnostics = []
    box = BBox(x=0, y=0, w=p.width, h=p.height)
    try:
        outcome = perceive_tile(
            client=client,
            cost_tracker=CostTracker(),
            reporter=NullReporter(),
            page=p,
            tile=Tile(id="tile", page_index=0, bbox=box),
            view_image=Image.new("RGB", (500, 320)),
            view_info=view(box),
            ownership_bbox=box,
            legend_summary=[],
            step=1,
            candidates=cs,
            reasoning_mode="enabled",
            on_diagnostic=diagnostics.append,
        )
    finally:
        client._client.close()
    assert len(seen) == 1 and seen[0]["stream"] is True
    assert seen[0]["thinking"] == {"type": "disabled"}
    assert outcome.detections[0].attributes["equipment_class"] == "motor"
    assert outcome.detections[0].bbox == cs[0].bbox
    assert diagnostics[-1]["response"]["content"][0]["input"] == payload
    source = diagnostics[0]["request"]["messages"][0]["content"][0]["source"]
    assert "data_sha256" in source and "data" not in source


def test_guard_ignores_empty_crops_and_distinguishes_intermittent_failures():
    guard = PerceptionRunGuard(SymbolPerceptionConfig())
    assert guard.observe(True) is None
    assert guard.observe(None) is None
    assert guard.observe(True) is None
    assert guard.observe(True)
    guard = PerceptionRunGuard(SymbolPerceptionConfig())
    for _ in range(4):
        assert guard.observe(True) is None
        assert guard.observe(False) is None
    assert guard.observe(True)


@pytest.mark.parametrize("run_limit", [0, 1800])
def test_stage_stops_and_saves_completed_work_and_pending_candidates(
    tmp_path, monkeypatch, run_limit
):
    p = page([circle(f"c{i}", i * 200 + 100, 100) for i in range(5)])
    rendered = DiagramPage(
        page_index=0,
        source_ref=p.source_ref,
        width=p.width,
        height=p.height,
        dpi=p.dpi,
        effective_dpi=p.effective_dpi,
        is_scanned=False,
        image=Image.new("RGB", (p.width, p.height)),
    )
    tiles = [
        Tile(id=f"tile{i}", page_index=0, bbox=BBox(x=i * 200, y=0, w=200, h=p.height))
        for i in range(5)
    ]
    monkeypatch.setattr("diagex.vision.loader.iter_pages", lambda _: iter([rendered]))
    monkeypatch.setattr("diagex.extractors.pid_evidence.tile", lambda *_: tiles)

    class Client:
        calls = 0

        def messages_create(self, **kwargs):
            self.calls += 1
            ids = json.loads(kwargs["messages"][0]["content"][1]["text"].split("\n")[-1])[
                "native_symbol_candidates"
            ]
            assert kwargs["reasoning_mode_override"] == "disabled"
            if self.calls == 1:
                return reply(
                    [
                        {
                            "candidate_id": ids[0]["candidate_id"],
                            "decision": "symbol",
                            "kind": "instrument",
                        }
                    ]
                )
            return reply([{"decision": "symbol"} for _ in ids])

    client = Client()
    cfg = Config(
        llm=LLMConfig(model="test", anthropic_api_key="unused", reasoning_mode="enabled"),
        symbol_perception=SymbolPerceptionConfig(run_timeout_s=run_limit),
    )
    store = CheckpointStore.create(
        run_dir=tmp_path, source_sha256="s", config_sha256="c", run_id="test"
    )
    detections, statuses, counts, stopped = _run_perception(
        source=None,
        pages=[p],
        cfg=cfg,
        client=client,
        cost=CostTracker(),
        reporter=NullReporter(),
        store=store,
        legend_summary=[],
        run_dir=tmp_path,
        prior_cost={},
    )
    assert stopped and statuses[0] == "partial"
    assert len(detections) == (1 if run_limit else 0)
    assert client.calls == counts["submit_pid_objects"] == (7 if run_limit else 0)
    assert (tmp_path / "perception.stop.json").is_file()
    audit = json.loads((tmp_path / "perception.review.json").read_text())
    assert len(audit["reviews"]) == 5
    assert audit["reviews"][-1]["status"] == "unreviewed"
    assert store.manifest.status == "partial"
    if run_limit:
        events = json.loads(
            (tmp_path / "checkpoints/perception_diagnostics/p0001__tile1.json").read_text()
        )["events"]
        assert len(events) == 4
        assert events[-1]["response"]["content"][0]["input"]["candidate_results"] == [
            {"decision": "symbol"}
        ]
