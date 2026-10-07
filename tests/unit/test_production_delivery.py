import json
import time
from types import SimpleNamespace

import pytest

from diagex.config import LLMConfig
from diagex.llm.budget import BudgetExceeded, SpendingLedger
from diagex.llm.model_policy import FAST_MODEL, validate_production_models
from diagex.vision.legend_context import select_graph_legend_context
from diagex.vision.page_graph import (
    ConnectionHypothesis,
    PageGraphSubmission,
    validate_page_graph_submission,
)
from tests.unit.test_page_graph import _node, _page


def test_budget_persists_reservations_and_counts_retries(tmp_path):
    path, prices = tmp_path / "spend.json", tmp_path / "prices.json"
    path.write_text(json.dumps({"limit_usd": 1, "category_limits": {"graph": 1}, "requests": []}))
    prices.write_text(
        json.dumps(
            {
                "models": {
                    "m": {
                        "valid_until": time.time() + 60,
                        "context_length": 100,
                        "input_per_token": 0.004,
                        "output_per_token": 0.001,
                    }
                }
            }
        )
    )
    ledger = SpendingLedger(path, prices, "graph")
    first = ledger.reserve("m", 100)
    SpendingLedger(path, prices, "graph").reserve("m", 100)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("m", 1)
    ledger.settle(first, SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=10)))
    assert json.loads(path.read_text())["requests"][1]["charged_usd"] == 0.5
    assert ledger.reserve("m", 1)
    with pytest.raises(BudgetExceeded, match="price"):
        ledger.reserve("unknown", 1)


def test_production_rejects_closed_source_in_any_stage():
    cfg = LLMConfig(model=FAST_MODEL, production_open_weight=True)
    validate_production_models(cfg)
    for field in ("model", "vision_model", "reasoning_model", "escalation_model"):
        bad = LLMConfig(model=FAST_MODEL, production_open_weight=True)
        setattr(bad, field, "anthropic/claude-opus-4.7")
        with pytest.raises(ValueError, match="evaluation-only"):
            validate_production_models(bad)


def test_graph_legend_keeps_graphical_definitions_after_large_abbreviation_table():
    entries = [
        {"kind": "other", "label": f"X{i}", "attributes": {"legend_kind": "abbreviation"}}
        for i in range(150)
    ]
    entries += [
        {"kind": "line", "label": "Electric signal", "symbol_class": "signal_electric"},
        {"kind": "valve", "label": "Check valve", "symbol_class": "check_valve"},
    ]
    selected = select_graph_legend_context(entries, ["X1"])
    assert {e["label"] for e in selected} == {"X1", "Electric signal", "Check valve"}


def test_process_hypotheses_are_not_graph_edges():
    a, b = _node("a", kind="equipment"), _node("b", kind="equipment")
    submission = PageGraphSubmission(
        hypotheses=[
            ConnectionHypothesis(
                from_ref="n1", to_ref="n2", rationale="Expected downstream cooler"
            ),
            ConnectionHypothesis(from_ref="n1", to_ref="unknown", rationale="Invalid endpoint"),
        ]
    )
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=submission,
        nodes_by_ref={"n1": a, "n2": b},
        local_node_refs={"n1": a, "n2": b},
        edges_by_ref={},
    )
    assert not result.edges
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0]["from_node"] == "a"
    assert result.hypotheses[0]["status"] == "unresolved"




def test_fresh_legend_bypasses_complete_and_partial_cache(tmp_path, monkeypatch):
    import fitz

    from diagex.config import Config
    from diagex.extractors import pid_legend
    from diagex.llm.cost import CostTracker
    from diagex.vision.legend_models import LegendEntry, LegendPack

    source = tmp_path / "drawing.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(source)

    def cached(*args, **kwargs):
        raise AssertionError("Fresh must not read either complete or partial machine cache")

    seen = []

    def extract(**kwargs):
        seen.append(kwargs["prior"])
        return [LegendEntry(label="Vessel", kind="equipment", symbol_class="vessel")]

    monkeypatch.setattr(pid_legend, "_load_cached", cached)
    monkeypatch.setattr(pid_legend, "_extract_from_page", extract)
    result = pid_legend._resolve_from_explicit_path(
        source=None,
        legend_path=source,
        symbol_standard="isa-5.1",
        builtin=LegendPack(entries=[]),
        cfg=Config(),
        client=None,
        cost_tracker=CostTracker(),
        cache_path=tmp_path / "legend-machine.json",
        extractor_fingerprint="fresh-test",
        fresh=True,
    )
    assert seen == [None]
    assert result.pack.entries[0].label == "Vessel"


def test_paused_checkpoint_resumes_only_with_matching_source_and_configuration(tmp_path):
    from diagex.extractors.evidence_checkpoint import CheckpointStore, find_resumable_run

    run = tmp_path / "run"
    store = CheckpointStore.create(
        run_dir=run, source_sha256="source", config_sha256="config", run_id="r-test"
    )
    store.write_json_artifact("perception", "tile-1", {"detections": []})
    store.manifest.pause_reason = "bounded failure guard"
    store.set_status("paused")
    found = find_resumable_run(runs_root=tmp_path, source_sha256="source", config_sha256="config")
    assert found is not None and found.is_done("perception", "tile-1")
    for source, config in [("changed", "config"), ("source", "changed")]:
        assert (
            find_resumable_run(runs_root=tmp_path, source_sha256=source, config_sha256=config)
            is None
        )


def test_failed_adaptive_reinspection_withholds_known_pressure_motor_conflict():
    from PIL import Image

    from diagex.config import SymbolPerceptionConfig
    from diagex.llm.cost import CostTracker
    from diagex.ui.progress import NullReporter
    from diagex.vision.models import BBox, Tile
    from diagex.vision.perception import perceive_tile
    from diagex.vision.symbol_candidates import symbol_candidates
    from tests.unit.test_symbol_perception import circle, page, view

    p = page([circle("pressure", 100, 100)])
    cs = symbol_candidates(p)
    cs[0].text = ["P"]
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) > 1:
                raise RuntimeError("unavailable")
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {
                            "objects": [
                                {
                                    "kind": "equipment",
                                    "candidate_id": cs[0].id,
                                    "equipment_class": "motor",
                                }
                            ]
                        },
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
        tile=Tile(id="t", page_index=0, bbox=box),
        view_image=Image.new("RGB", (500, 320), "white"),
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[],
        step=1,
        candidates=cs,
        policy=SymbolPerceptionConfig(workflow="adaptive"),
    )
    assert len(calls) == 2
    assert not outcome.detections
    assert outcome.batch.candidate_reviews[0]["status"] == "uncertain"


def test_targeted_route_query_does_not_connect_through_intermediate_symbol():
    from diagex.vision.topology import build_page_topology
    from tests.unit.test_port_topology import node, page, path

    nodes = [node("a", 100), node("b", 300), node("c", 500)]
    p = page(
        [path("left", [(140, 300), (300, 300)]), path("right", [(340, 300), (500, 300)])],
        nodes=nodes,
    )
    result = build_page_topology(page=p, nodes=nodes, requested_pairs={frozenset(("a", "c"))})
    assert not result.edges
    result = build_page_topology(page=p, nodes=nodes, requested_pairs={frozenset(("a", "b"))})
    assert result.edges
    assert all({e.from_node, e.to_node} == {"a", "b"} for e in result.edges)


def test_missing_usage_retains_full_reservation(tmp_path):
    path, prices = tmp_path / "spend.json", tmp_path / "prices.json"
    path.write_text(json.dumps({"limit_usd": 1, "category_limits": {"graph": 1}, "requests": []}))
    prices.write_text(
        json.dumps(
            {
                "models": {
                    "m": {
                        "valid_until": time.time() + 60,
                        "context_length": 100,
                        "input_per_token": 0.001,
                        "output_per_token": 0.001,
                    }
                }
            }
        )
    )
    ledger = SpendingLedger(path, prices, "graph")
    rid = ledger.reserve("m", 100)
    ledger.settle(rid, SimpleNamespace(usage=SimpleNamespace(input_tokens=10)))
    row = json.loads(path.read_text())["requests"][0]
    assert row["charged_usd"] == row["reserved_usd"] == 0.2
    assert row["status"] == "usage_unavailable"


@pytest.mark.parametrize(
    "validation,strokes,expected",
    [("accept", ["distinct outline"], 1), ("accept", [], 0), ("reject", ["pipeline"], 0)],
)
@pytest.mark.parametrize("workflow", ["fixed", "adaptive"])
def test_discovery_requires_independent_source_glyph_validation(
    monkeypatch, validation, strokes, expected, workflow
):
    from PIL import Image

    from diagex.config import SymbolPerceptionConfig
    from diagex.llm.cost import CostTracker
    from diagex.vision import perception as module
    from diagex.vision.models import BBox, Tile
    from tests.unit.test_symbol_perception import circle, page, view

    p = page([circle("source-glyph", 100, 100)])
    # Isolate the supported case: source ink exists but candidate extraction missed it.
    monkeypatch.setattr(module, "symbol_candidates", lambda page: [])
    box = BBox(x=0, y=0, w=p.width, h=p.height)
    target = BBox(x=80, y=80, w=40, h=40)
    obj = {
        "kind": "instrument",
        "bbox": {"x": 80 / p.width, "y": 80 / p.height, "w": 40 / p.width, "h": 40 / p.height},
        "printed_tag": "P1",
    }
    outcome = module.PerceptionOutcome(
        detections=[],
        batch=module.PerceptionBatch(
            candidate_reviews=[{"bbox": target.model_dump(), "object": obj}]
        ),
    )
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "validate_discoveries",
                        "input": {
                            "decisions": [
                                {
                                    "proposal_id": "0",
                                    "decision": validation,
                                    "kind": "instrument",
                                    "visible_strokes": strokes,
                                    "reason": "source inspection",
                                }
                            ]
                        },
                    }
                ],
                usage=None,
            )

    kwargs = dict(
        page=p,
        candidates=[],
        client=Client(),
        cost_tracker=CostTracker(),
        tile=Tile(id="t", page_index=0, bbox=box),
        view_info=view(box),
        ownership_bbox=box,
        region_provider=lambda *args: (Image.new("RGB", (500, 320), "white"), view(box)),
    )
    monkeypatch.setattr(module, "_perceive_tile", lambda **kw: outcome)
    module.perceive_tile(**kwargs, policy=SymbolPerceptionConfig(workflow=workflow))
    assert len(calls) == 1
    assert len(outcome.detections) == expected
    if expected:
        assert outcome.detections[0].attributes["geometry_basis"] == "validated_visual_discovery"
        assert outcome.detections[0].attributes["source_path_ids"] == ["source-glyph"]


def test_production_profile_selects_open_weights_without_legacy_model_env(monkeypatch):
    from diagex.config import load_config

    for key in (
        "DIAGEX_MODEL",
        "OPENROUTER_MODEL",
        "DIAGEX_VISION_MODEL",
        "DIAGEX_REASONING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    cfg = load_config(production_open_weight=True)
    assert cfg.llm.model == cfg.llm.vision_model == cfg.llm.reasoning_model == FAST_MODEL
    assert cfg.pid.engine == "evidence-v2"
    assert cfg.symbol_perception.workflow == "fixed"
    validate_production_models(cfg.llm)


def test_qwen_adapter_omits_unsupported_effort_parameter(monkeypatch):
    from diagex.llm.client import LLMClient

    calls = []
    client = LLMClient(
        LLMConfig(model=FAST_MODEL, transport="openrouter", openrouter_api_key="test-only")
    )

    def stream(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("captured without network")

    monkeypatch.setattr(client._client.messages, "stream", stream)
    with pytest.raises(RuntimeError, match="without network"):
        client.messages_create(
            system="test",
            messages=[],
            max_tokens=64,
            thinking={"type": "disabled"},
            reasoning_mode_override="disabled",
            output_config={"effort": "low"},
            max_attempts=1,
        )
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert "output_config" not in calls[0]
    client._client.close()


def test_targeted_inspection_requires_geometry_before_visual_model(monkeypatch):
    from PIL import Image

    from diagex.llm.cost import CostTracker
    from diagex.ui.progress import NullReporter
    from diagex.vision import page_graph as module
    from diagex.vision.topology import TopologyResult

    a, b = _node("a", kind="equipment"), _node("b", kind="equipment")
    result = module.PageGraphResult(
        page_index=0, hypotheses=[dict(from_node="a", to_node="b", status="unresolved")]
    )
    called = []
    monkeypatch.setattr(module, "build_page_topology", lambda **kw: TopologyResult(page_index=0))
    monkeypatch.setattr(module, "classify_page_line_evidence", lambda **kw: called.append(kw))
    module._inspect_proposed_routes(
        result=result,
        page=_page(),
        pages=[_page()],
        nodes=[a, b],
        rendered_image=Image.new("RGB", (100, 100)),
        client=object(),
        escalation_client=object(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        legend_line_images=[],
        on_attempt=None,
    )
    assert not called and not result.edges
    assert result.hypotheses[0]["targeted_inspection"]["source_route_found"] is False


def test_run_directory_collision_preserves_previous_artifact(tmp_path, monkeypatch):
    from diagex.config import Config
    from diagex.extractors import pid_evidence

    cfg = Config()
    cfg.runs_dir = tmp_path
    ids = iter(["r-old", "r-old", "r-new"])
    monkeypatch.setattr(pid_evidence, "_new_run_id", lambda: next(ids))
    monkeypatch.setattr(pid_evidence, "_timestamp", lambda: "fixed-time")
    old, _ = pid_evidence._prepare_v2_run_dir(cfg, "drawing", "model")
    (old / "sentinel").write_text("preserve me")
    new, run_id = pid_evidence._prepare_v2_run_dir(cfg, "drawing", "model")
    assert new != old and run_id == "r-new"
    assert (old / "sentinel").read_text() == "preserve me"
