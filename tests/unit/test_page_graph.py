from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence, PathEvidence
from diagex.vision.fusion import FusionResult, assemble_graph
from diagex.vision.models import BBox, ReconciledEdge, ReconciledGraph, ReconciledNode
from diagex.vision.page_graph import (
    CandidateDecision,
    OpcUpdate,
    PageGraphResponseFormatError,
    PageGraphSubmission,
    PageLineEvidence,
    ProposedRelation,
    VisualCandidateAssessment,
    _legend_line_montage,
    _legend_line_priority,
    classify_page_line_evidence,
    parse_line_evidence_response,
    solve_page_graph,
    validate_page_graph_submission,
)
from diagex.vision.topology import TopologyResult, build_page_topology


def _page(index: int = 0) -> PageEvidence:
    return PageEvidence(
        page_index=index,
        source_ref=f"drawing#page={index + 1}",
        width=1000,
        height=600,
        dpi=300,
        effective_dpi=300,
        is_scanned=False,
        role="pid",
        role_confidence="high",
        role_reason="test",
        fail_open=False,
        paths=[
            PathEvidence(
                id="test-pipe",
                page_index=index,
                points=[(120, 120), (500, 120)],
                bbox=BBox(x=120, y=120, w=380, h=1),
                origin="pdf_vector",
                primitive="line",
            ),
            *[
                PathEvidence(
                    id=f"body-{x}",
                    page_index=index,
                    origin="pdf_vector",
                    primitive="rect",
                    closed=True,
                    points=[(x, 100), (x + 40, 100), (x + 40, 140), (x, 140), (x, 100)],
                    bbox=BBox(x=x, y=100, w=40, h=40),
                )
                for x in (80, 500)
            ],
        ],
    )


def _node(
    node_id: str,
    *,
    kind: str,
    page: int = 0,
    direction: str | None = None,
    service: str | None = None,
) -> ReconciledNode:
    attributes = {
        key: value
        for key, value in {"direction": direction, "service": service}.items()
        if value is not None
    }
    return ReconciledNode(
        id=node_id,
        kind=kind,  # type: ignore[arg-type]
        label=node_id,
        bbox_global=BBox(x=100 + page * 20, y=100, w=40, h=40),
        page_index=page,
        confidence="high",
        attributes=attributes,
    )


def _edge(left: str, right: str) -> ReconciledEdge:
    # Semantic tests start with a physically supported native route. Structural
    # rejection, including forged proofs, is exercised in test_native_scene.
    endpoints = [
        ReconciledNode(
            id=name,
            label=name,
            kind="equipment",
            page_index=0,
            bbox_global=BBox(x=x, y=100, w=40, h=40),
            confidence="high",
        )
        for name, x in [(left, 80), (right, 500)]
    ]
    edge = build_page_topology(page=_page(), nodes=endpoints).edges[0]
    edge.id = "edge-1"
    if edge.from_node != left:
        edge.from_node, edge.to_node = edge.to_node, edge.from_node
        edge.polyline_global.reverse()
        edge.attributes["route_evidence"]["ports"].reverse()
    return edge


def test_line_evidence_ignores_harmless_provider_explanation_fields() -> None:
    response = SimpleNamespace(
        content=[
            {
                "type": "tool_use",
                "name": "submit_line_evidence",
                "input": {
                    "assessments": [
                        {
                            "candidate_ref": "e001",
                            "route_visible": "yes",
                            "endpoint_alignment": "both",
                            "observed_style": "dashed",
                            "endpoint_alignment_note": "both boxes touch the route",
                        }
                    ],
                    "summary_note": "all visible candidates assessed",
                },
            }
        ]
    )

    parsed = parse_line_evidence_response(response)

    assert len(parsed.assessments) == 1
    assert parsed.assessments[0].candidate_ref == "e001"


def _page_graph_response(*, structured: bool) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    content = (
        [
            {
                "type": "tool_use",
                "name": "submit_page_graph",
                "input": {
                    "candidate_decisions": [],
                    "new_relations": [],
                    "opc_updates": [],
                    "uncertainties": [],
                },
            }
        ]
        if structured
        else [{"type": "text", "text": "I need to inspect the diagram first."}]
    )
    return SimpleNamespace(
        id="response",
        model="reasoning-model",
        stop_reason="tool_use" if structured else "end_turn",
        usage=usage,
        content=content,
    )


@pytest.mark.parametrize(
    ("serializer_failure", "expected_phase"),
    [("prose", "response_parse"), ("malformed_tool_json", "provider_tool_json_decode")],
)
def test_page_graph_preserves_reasoning_and_recovers_serializer_once(
    serializer_failure: str,
    expected_phase: str,
) -> None:
    class RecoveringClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(reasoning_mode="enabled")
            self.requests: list[dict[str, object]] = []

        def messages_create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return _page_graph_response(structured=False)
            if len(self.requests) == 2:
                if serializer_failure == "malformed_tool_json":
                    raise ValueError("key must be a string at line 1 column 2378")
                if serializer_failure == "prose":
                    return _page_graph_response(structured=False)
            return _page_graph_response(structured=True)

    client = RecoveringClient()
    node = _node("instrument", kind="instrument")
    cost = CostTracker()
    attempts: list[None] = []

    result = solve_page_graph(
        client=client,  # type: ignore[arg-type]
        cost_tracker=cost,
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[node],
        all_nodes=[node],
        topology=TopologyResult(page_index=0),
        pages=[_page()],
        step=10,
        max_tokens=1000,
        on_attempt=lambda: attempts.append(None),
    )

    assert len(client.requests) == 3
    assert len(attempts) == 3
    assert result.format_recovery[0]["phase"] == expected_phase
    assert "tools" not in client.requests[0]
    assert client.requests[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert all(
        request["tool_choice"] == {"type": "tool", "name": "submit_page_graph"}
        for request in client.requests[1:]
    )
    assert client.requests[1]["reasoning_mode_override"] == "disabled"
    assert client.requests[1]["thinking"] == {"type": "disabled"}
    assert client.requests[1]["output_config"] == {"effort": "low"}
    serializer_text = client.requests[1]["messages"][0]["content"][-1]["text"]  # type: ignore[index]
    assert "I need to inspect the diagram first." in serializer_text
    assert result.reasoning_trace[0]["memo"] == "I need to inspect the diagram first."
    expected_cost_steps = [10, 11] if serializer_failure == "malformed_tool_json" else [10, 11, 12]
    assert [row.step for row in cost.steps] == expected_cost_steps


def test_page_graph_retains_both_invalid_responses_after_recovery_failure() -> None:
    class ProseOnlyClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(reasoning_mode="enabled")
            self.calls = 0

        def messages_create(self, **kwargs: object) -> object:
            self.calls += 1
            return _page_graph_response(structured=False)

    client = ProseOnlyClient()
    node = _node("instrument", kind="instrument")

    with pytest.raises(PageGraphResponseFormatError) as caught:
        solve_page_graph(
            client=client,  # type: ignore[arg-type]
            cost_tracker=CostTracker(),
            reporter=NullReporter(),
            page=_page(),
            rendered_image=Image.new("RGB", (1000, 600), "white"),
            nodes=[node],
            all_nodes=[node],
            topology=TopologyResult(page_index=0),
            pages=[_page()],
            step=1,
            max_tokens=1000,
        )

    assert client.calls == 3
    assert caught.value.attempts == 3
    assert [row["response"]["content"][0]["text"] for row in caught.value.diagnostics] == [
        "I need to inspect the diagram first.",
        "I need to inspect the diagram first.",
    ]


def test_page_graph_can_send_structured_evidence_without_images() -> None:
    class TextReasoningClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(reasoning_mode="enabled")
            self.request: dict[str, object] | None = None

        def messages_create(self, **kwargs: object) -> object:
            self.request = kwargs
            return _page_graph_response(structured=True)

    client = TextReasoningClient()
    node = _node("instrument", kind="instrument")

    solve_page_graph(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[node],
        all_nodes=[node],
        topology=TopologyResult(page_index=0),
        pages=[_page()],
        step=1,
        max_tokens=1000,
        include_visual_context=False,
    )

    assert client.request is not None
    content = client.request["messages"][0]["content"]  # type: ignore[index]
    assert [block["type"] for block in content] == ["text"]


def test_deepseek_separates_thinking_from_forced_serialization_tool() -> None:
    class DeepSeekClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                reasoning_mode="enabled",
                model="deepseek/deepseek-v4-flash-vision-exp",
                transport="openrouter",
            )
            self.requests: list[dict[str, object]] = []

        def messages_create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            return _page_graph_response(structured=len(self.requests) > 1)

    client = DeepSeekClient()
    node = _node("instrument", kind="instrument")

    solve_page_graph(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[node],
        all_nodes=[node],
        topology=TopologyResult(page_index=0),
        pages=[_page()],
        step=1,
        max_tokens=1000,
    )

    assert len(client.requests) == 2
    assert client.requests[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "tools" not in client.requests[0]
    assert "tool_choice" not in client.requests[0]
    assert client.requests[1]["thinking"] == {"type": "disabled"}
    assert client.requests[1]["tool_choice"] == {
        "type": "tool",
        "name": "submit_page_graph",
    }


@pytest.mark.parametrize("model", ["z-ai/glm-5.3", "z-ai/glm-5.3-flash"])
def test_mandatory_reasoning_model_keeps_low_reasoning_for_serialization(model: str) -> None:
    class GlmClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                reasoning_mode="enabled",
                model=model,
                transport="openrouter",
            )
            self.requests: list[dict[str, object]] = []

        def messages_create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            return _page_graph_response(structured=len(self.requests) > 1)

    client = GlmClient()
    node = _node("instrument", kind="instrument")

    solve_page_graph(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[node],
        all_nodes=[node],
        topology=TopologyResult(page_index=0),
        pages=[_page()],
        step=1,
        max_tokens=1000,
    )

    assert len(client.requests) == 2
    serializer = client.requests[1]
    assert serializer["reasoning_mode_override"] == "enabled"
    assert serializer["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert serializer["output_config"] == {"effort": "low"}
    assert serializer["tool_choice"] == {
        "type": "tool",
        "name": "submit_page_graph",
    }


def test_unknown_mandatory_reasoning_model_recovers_with_same_model() -> None:
    class NewlyMandatoryClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                reasoning_mode="enabled",
                model="vendor/new-mandatory-reasoner",
                transport="openrouter",
            )
            self.requests: list[dict[str, object]] = []

        def messages_create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return _page_graph_response(structured=False)
            if len(self.requests) == 2:
                raise ValueError("Reasoning is mandatory for this endpoint and cannot be disabled.")
            return _page_graph_response(structured=True)

    client = NewlyMandatoryClient()
    node = _node("instrument", kind="instrument")
    attempts: list[None] = []

    result = solve_page_graph(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[node],
        all_nodes=[node],
        topology=TopologyResult(page_index=0),
        pages=[_page()],
        step=1,
        max_tokens=1000,
        on_attempt=lambda: attempts.append(None),
    )

    assert len(client.requests) == 3
    assert len(attempts) == 3
    assert client.requests[1]["reasoning_mode_override"] == "disabled"
    assert client.requests[2]["reasoning_mode_override"] == "enabled"
    assert result.format_recovery[0]["phase"] == "reasoning_capability_recovery"


def test_page_graph_reasoning_disabled_goes_directly_to_serializer() -> None:
    class NonThinkingClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(reasoning_mode="disabled", model="local/model")
            self.requests: list[dict[str, object]] = []

        def messages_create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            return _page_graph_response(structured=True)

    client = NonThinkingClient()
    node = _node("instrument", kind="instrument")
    result = solve_page_graph(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[node],
        all_nodes=[node],
        topology=TopologyResult(page_index=0),
        pages=[_page()],
        step=1,
        max_tokens=1000,
    )

    assert len(client.requests) == 1
    assert client.requests[0]["thinking"] == {"type": "disabled"}
    assert client.requests[0]["tool_choice"] == {
        "type": "tool",
        "name": "submit_page_graph",
    }
    assert result.reasoning_trace == []


def test_page_graph_batches_ambiguities_and_merges_submissions() -> None:
    class BatchClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(reasoning_mode="enabled", model="local/reasoner")
            self.requests: list[dict[str, object]] = []

        def messages_create(self, **kwargs: object) -> object:
            self.requests.append(kwargs)
            usage = SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
            request_payload = json.loads(kwargs["messages"][0]["content"][0]["text"])  # type: ignore[index]
            if "tools" not in kwargs:
                refs = request_payload["scope"]["candidate_refs"]
                return SimpleNamespace(
                    usage=usage,
                    stop_reason="end_turn",
                    content=[
                        {
                            "type": "text",
                            "text": "\n".join(f"{ref} | keep | - | forward" for ref in refs),
                        }
                    ],
                )
            refs = request_payload["source_payload"]["scope"]["candidate_refs"]
            return SimpleNamespace(
                usage=usage,
                stop_reason="tool_use",
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_page_graph",
                        "input": {
                            "candidate_decisions": [
                                {
                                    "candidate_ref": ref,
                                    "decision": "keep",
                                    "confidence": "medium",
                                }
                                for ref in refs
                            ]
                        },
                    }
                ],
            )

    nodes = [_node(f"instrument-{index}", kind="instrument") for index in range(10)]
    edges = [
        _edge(nodes[index].id, nodes[index + 1].id).model_copy(update={"id": f"edge-{index}"})
        for index in range(9)
    ]
    client = BatchClient()
    result = solve_page_graph(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=nodes,
        all_nodes=nodes,
        topology=TopologyResult(page_index=0, edges=edges),
        pages=[_page()],
        step=1,
        max_tokens=6000,
    )

    assert len(client.requests) == 4
    reasoning_requests = [request for request in client.requests if "tools" not in request]
    assert [
        len(json.loads(request["messages"][0]["content"][0]["text"])["candidate_edges"])  # type: ignore[index]
        for request in reasoning_requests
    ] == [8, 1]
    assert len(result.reasoning_trace) == 2
    assert len(result.submission["candidate_decisions"]) == 9


def test_visual_line_evidence_is_forced_non_thinking() -> None:
    class VisualClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(reasoning_mode="disabled")
            self.request: dict[str, object] | None = None

        def messages_create(self, **kwargs: object) -> object:
            self.request = kwargs
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=20,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_line_evidence",
                        "input": {
                            "assessments": [
                                {
                                    "candidate_ref": "E001",
                                    "route_visible": "yes",
                                    "endpoint_alignment": "both",
                                    "observed_style": "solid",
                                    "arrow_direction": "none",
                                    "confidence": "high",
                                    "evidence": ["continuous solid stroke"],
                                }
                            ]
                        },
                    }
                ],
            )

    left = _node("left", kind="equipment")
    right = _node("right", kind="equipment")
    client = VisualClient()
    result = classify_page_line_evidence(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[left, right],
        topology=TopologyResult(page_index=0, edges=[_edge(left.id, right.id)]),
        step=1,
    )

    assert client.request is not None
    assert client.request["thinking"] == {"type": "disabled"}
    assert client.request["output_config"] == {"effort": "low"}
    assert client.request["tool_choice"] == {
        "type": "tool",
        "name": "submit_line_evidence",
    }
    assert result.assessments["E001"].route_visible == "yes"


def test_visual_line_evidence_recovers_generic_provider_json_syntax_error_once() -> None:
    class RecoveringVisualClient:
        def __init__(self) -> None:
            self.calls = 0

        def messages_create(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("expected `:` at line 1 column 310")
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=20,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_line_evidence",
                        "input": {
                            "assessments": [
                                {
                                    "candidate_ref": "E001",
                                    "route_visible": "uncertain",
                                    "endpoint_alignment": "one",
                                    "observed_style": "unknown",
                                    "arrow_direction": "none",
                                    "confidence": "low",
                                    "evidence": ["route partly obscured"],
                                }
                            ]
                        },
                    }
                ],
            )

    left = _node("left", kind="equipment")
    right = _node("right", kind="equipment")
    client = RecoveringVisualClient()
    result = classify_page_line_evidence(
        client=client,  # type: ignore[arg-type]
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=_page(),
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[left, right],
        topology=TopologyResult(page_index=0, edges=[_edge(left.id, right.id)]),
        step=1,
    )

    assert client.calls == 2
    assert result.format_recovery[0]["phase"] == "provider_tool_json_decode"
    assert result.assessments["E001"].route_visible == "uncertain"


def test_visual_evidence_recalculates_edge_confidence() -> None:
    left = _node("left", kind="equipment")
    right = _node("right", kind="equipment")
    candidate = _edge(left.id, right.id)
    candidate.attributes.update(
        {"topology_source": "deterministic_vector", "visual_style": "solid"}
    )
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(),
        nodes_by_ref={"N001": left, "N002": right},
        edges_by_ref={"E001": candidate},
        local_node_refs={"N001": left, "N002": right},
        visual_evidence=PageLineEvidence(
            page_index=0,
            assessments={
                "E001": VisualCandidateAssessment(
                    candidate_ref="E001",
                    route_visible="yes",
                    endpoint_alignment="both",
                    observed_style="solid",
                    confidence="high",
                )
            },
        ),
    )

    assert len(result.edges) == 1
    assert result.edges[0].system_confidence != 0.9
    evidence = result.edges[0].attributes["system_confidence_evidence"]
    assert evidence["endpoint_roles"] == ["equipment", "equipment"]


def test_page_graph_normalises_common_deepseek_schema_variants() -> None:
    submission = PageGraphSubmission.model_validate(
        {
            "candidate_decisions": [
                {
                    "candidate_ref": "E001",
                    "decision": "keep",
                    "evidence": "visible solid process line",
                }
            ],
            "new_relations": [
                {
                    "from_ref": "N001",
                    "to_ref": "N002",
                    "line_type": "signal_electric",
                    "evidence": "visible dashed line",
                }
            ],
            "opc_updates": [
                {
                    "node_ref": "N003",
                    "evidence": "printed DW02-0003",
                }
            ],
            "uncertainties": [
                {
                    "node_refs": [f"N{i:03d}" for i in range(20)],
                    "candidate_refs": "E002",
                    "reason": "ambiguous line family",
                }
            ],
        }
    )

    assert submission.candidate_decisions[0].evidence == ["visible solid process line"]
    assert submission.new_relations[0].evidence == ["visible dashed line"]
    assert submission.opc_updates[0].evidence == ["printed DW02-0003"]
    assert submission.uncertainties[0].candidate_refs == ["E002"]
    assert len(submission.uncertainties[0].node_refs) == 12


def test_line_legend_images_are_combined_into_a_compact_labelled_montage() -> None:
    source = Image.new("RGB", (240, 60), "white")
    draw = ImageDraw.Draw(source)
    for x in range(10, 230, 35):
        draw.line((x, 30, min(x + 20, 230), 30), fill="black", width=3)
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    montage = _legend_line_montage(
        [
            {
                "label": "Pneumatic signal",
                "description": "project legend sample",
                "image_b64": encoded,
            },
            {"label": "malformed", "image_b64": "not base64"},
        ]
    )

    assert montage is not None
    assert montage.size == (1120, 160)
    assert montage.getbbox() is not None


def test_line_legend_prioritises_signal_and_process_examples() -> None:
    assert _legend_line_priority({"label": "气压信号线(空气)"}) == 0
    assert _legend_line_priority({"label": "Electric signal"}) == 0
    assert _legend_line_priority({"label": "主工艺管线"}) == 1
    assert _legend_line_priority({"label": "Pipe crossing"}) == 2
    assert _legend_line_priority({"label": "Equipment insulation"}) == 3


def test_instrument_candidate_is_retyped_and_keeps_geometry() -> None:
    instrument = _node("instrument", kind="instrument")
    equipment = _node("equipment", kind="equipment")
    equipment.attributes["valve_type"] = "control_valve"
    candidate = _edge(instrument.id, equipment.id)
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            candidate_decisions=[
                CandidateDecision(
                    candidate_ref="E001",
                    decision="retype",
                    line_type="signal_pneumatic",
                    reverse=True,
                    confidence="high",
                    evidence=["visible dashed pneumatic line"],
                )
            ]
        ),
        nodes_by_ref={"N001": instrument, "N002": equipment},
        edges_by_ref={"E001": candidate},
        local_node_refs={"N001": instrument, "N002": equipment},
    )

    assert len(result.edges) == 1
    assert result.edges[0].line_type == "signal_pneumatic"
    # A model's reverse flag without an observed arrow is not flow evidence.
    assert result.edges[0].from_node == instrument.id
    assert result.edges[0].to_node == equipment.id
    assert result.edges[0].polyline_global == candidate.polyline_global
    assert result.edges[0].attributes["flow_direction"] == "unknown"
    assert result.conflicts == []


def test_generic_equipment_signal_is_retained_as_provisional_for_review() -> None:
    instrument = _node("instrument", kind="instrument")
    equipment = _node("equipment", kind="equipment")
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            candidate_decisions=[
                CandidateDecision(
                    candidate_ref="E001",
                    decision="retype",
                    line_type="signal_pneumatic",
                    confidence="high",
                )
            ]
        ),
        nodes_by_ref={"N001": instrument, "N002": equipment},
        edges_by_ref={"E001": _edge(instrument.id, equipment.id)},
        local_node_refs={"N001": instrument, "N002": equipment},
    )

    assert len(result.edges) == 1
    assert result.edges[0].attributes["provisional_review_only"] is True
    assert result.edges[0].attributes["proposed_line_type"] == "signal_pneumatic"
    assert result.conflicts[0]["type"] == "unsupported_endpoint_combination"


def test_keep_decision_retains_corroborated_same_loop_electric_classification() -> None:
    element = _node("TE-00201", kind="instrument")
    element.attributes.update(
        {
            "instrument_function": "element",
            "measured_variable": "temperature",
            "loop_number": "00201",
        }
    )
    indicator = _node("TI-00201", kind="instrument")
    indicator.attributes.update(
        {
            "instrument_function": "indicator",
            "measured_variable": "temperature",
            "loop_number": "00201",
        }
    )
    candidate = _edge(element.id, indicator.id)
    candidate.line_type = "other"
    candidate.attributes.update(
        {
            "topology_source": "deterministic_vector",
            "visual_style": "dashed",
            "visual_style_confidence": 0.91,
        }
    )
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            candidate_decisions=[
                CandidateDecision(
                    candidate_ref="E001",
                    decision="keep",
                    line_type="signal_electric",
                    confidence="high",
                    evidence=["same-loop TE to TI measurement signal"],
                )
            ]
        ),
        nodes_by_ref={"N001": element, "N002": indicator},
        edges_by_ref={"E001": candidate},
        local_node_refs={"N001": element, "N002": indicator},
        visual_evidence=PageLineEvidence(
            page_index=0,
            assessments={
                "E001": VisualCandidateAssessment(
                    candidate_ref="E001",
                    route_visible="yes",
                    endpoint_alignment="both",
                    observed_style="dashed",
                    legend_class="signal_electric",
                    confidence="high",
                    evidence=["project-legend electric signal pattern"],
                )
            },
        ),
    )

    assert result.conflicts == []
    assert result.accepted_candidate_ids == ["edge-1"]
    assert result.edges[0].line_type == "signal_electric"
    assert result.edges[0].attributes["line_type_source"] == ("page_graph_keep_classification")
    pair = result.edges[0].attributes["system_confidence_evidence"]["instrument_pair"]
    assert pair == {
        "same_loop": True,
        "loop_number": "00201",
        "functions": ["element", "indicator"],
        "measured_variables": ["temperature", "temperature"],
        "functional_direction": "forward",
        "compatible": True,
    }


def test_keep_decision_does_not_retype_uncorroborated_instrument_pair() -> None:
    element = _node("TE-00201", kind="instrument")
    element.attributes.update({"instrument_function": "element", "loop_number": "00201"})
    indicator = _node("TI-00999", kind="instrument")
    indicator.attributes.update({"instrument_function": "indicator", "loop_number": "00999"})
    candidate = _edge(element.id, indicator.id)
    candidate.line_type = "other"
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            candidate_decisions=[
                CandidateDecision(
                    candidate_ref="E001",
                    decision="keep",
                    line_type="signal_electric",
                    confidence="high",
                )
            ]
        ),
        nodes_by_ref={"N001": element, "N002": indicator},
        edges_by_ref={"E001": candidate},
        local_node_refs={"N001": element, "N002": indicator},
        visual_evidence=PageLineEvidence(
            page_index=0,
            assessments={
                "E001": VisualCandidateAssessment(
                    candidate_ref="E001",
                    route_visible="yes",
                    endpoint_alignment="both",
                    observed_style="dashed",
                    legend_class="signal_electric",
                    confidence="high",
                )
            },
        ),
    )

    assert result.edges[0].line_type == "other"
    assert result.edges[0].attributes["provisional_review_only"] is True
    assert result.edges[0].attributes["proposed_line_type"] == "signal_electric"
    assert result.conflicts[0]["type"] == "endpoint_role_uncertain"


def test_generic_equipment_electric_signal_is_withheld_from_graph() -> None:
    instrument = _node("instrument", kind="instrument")
    equipment = _node("equipment", kind="equipment")
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            new_relations=[
                ProposedRelation(
                    from_ref="N001",
                    to_ref="N002",
                    line_type="signal_electric",
                    confidence="high",
                    evidence=["model proposed a dashed signal"],
                )
            ]
        ),
        nodes_by_ref={"N001": instrument, "N002": equipment},
        edges_by_ref={},
        local_node_refs={"N001": instrument, "N002": equipment},
    )

    assert result.edges == []
    assert result.conflicts[0]["type"] == "unsupported_endpoint_combination"
    assert result.conflicts[0]["endpoint_roles"] == ["instrument", "equipment"]


def test_manual_valve_does_not_count_as_a_pneumatic_actuator_endpoint() -> None:
    instrument = _node("instrument", kind="instrument")
    valve = _node("manual-valve", kind="equipment")
    valve.attributes["valve_type"] = "gate"
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            new_relations=[
                ProposedRelation(
                    from_ref="N001",
                    to_ref="N002",
                    line_type="signal_pneumatic",
                    confidence="high",
                    evidence=["model proposed a pneumatic signal"],
                )
            ]
        ),
        nodes_by_ref={"N001": instrument, "N002": valve},
        edges_by_ref={},
        local_node_refs={"N001": instrument, "N002": valve},
    )

    assert result.edges == []
    assert result.conflicts[0]["type"] == "unsupported_endpoint_combination"
    assert result.conflicts[0]["endpoint_roles"] == ["instrument", "equipment"]


def test_missing_instrument_decision_becomes_provisional_conflict_edge() -> None:
    instrument = _node("instrument", kind="instrument")
    equipment = _node("equipment", kind="equipment")
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(),
        nodes_by_ref={"N001": instrument, "N002": equipment},
        edges_by_ref={"E001": _edge(instrument.id, equipment.id)},
        local_node_refs={"N001": instrument, "N002": equipment},
    )

    assert len(result.edges) == 1
    assert result.edges[0].attributes["provisional_review_only"] is True
    assert result.provisional_candidate_ids == ["edge-1"]
    assert result.conflicts[0]["type"] == "page_graph_uncertain_candidate"


def test_equipment_process_candidate_is_accepted_without_model_decision() -> None:
    left = _node("left", kind="equipment")
    right = _node("right", kind="equipment")
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(),
        nodes_by_ref={"N001": left, "N002": right},
        edges_by_ref={"E001": _edge(left.id, right.id)},
        local_node_refs={"N001": left, "N002": right},
    )

    assert [edge.id for edge in result.edges] == ["edge-1"]
    assert result.accepted_candidate_ids == ["edge-1"]
    assert not result.provisional_candidate_ids


def test_styled_equipment_candidate_requires_legend_backed_page_decision() -> None:
    left = _node("left", kind="equipment")
    right = _node("right", kind="equipment")
    candidate = _edge(left.id, right.id)
    candidate.line_type = "other"
    candidate.attributes.update(
        {
            "visual_style": "dashed",
            "visual_style_confidence": 0.94,
            "style_evidence_ids": ["sty-1"],
        }
    )
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(),
        nodes_by_ref={"N001": left, "N002": right},
        edges_by_ref={"E001": candidate},
        local_node_refs={"N001": left, "N002": right},
    )

    assert len(result.edges) == 1
    assert result.edges[0].attributes["provisional_review_only"] is True
    assert result.conflicts[0]["type"] == "page_graph_uncertain_candidate"


def test_unknown_refs_and_self_loops_are_rejected_individually() -> None:
    node = _node("instrument", kind="instrument")
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            candidate_decisions=[CandidateDecision(candidate_ref="E999", decision="reject")],
            new_relations=[
                ProposedRelation(
                    from_ref="N001",
                    to_ref="N001",
                    line_type="signal_electric",
                )
            ],
        ),
        nodes_by_ref={"N001": node},
        edges_by_ref={},
        local_node_refs={"N001": node},
    )

    assert result.edges == []
    assert any("unknown candidate_ref" in value for value in result.diagnostics)
    assert any("self-loop" in value for value in result.diagnostics)


def test_cross_sheet_relation_requires_opposite_opcs_and_printed_evidence() -> None:
    outgoing = _node("out", kind="opc", page=0, direction="out", service="instrument air")
    incoming = _node("in", kind="opc", page=1, direction="in", service="instrument air")
    result = validate_page_graph_submission(
        page=_page(0),
        pages=[_page(0), _page(1)],
        submission=PageGraphSubmission(
            new_relations=[
                ProposedRelation(
                    from_ref="N001",
                    to_ref="P002N001",
                    line_type="process",
                    cross_sheet=True,
                    confidence="high",
                    evidence=["instrument air continuation printed at both OPCs"],
                )
            ]
        ),
        nodes_by_ref={"N001": outgoing, "P002N001": incoming},
        edges_by_ref={},
        local_node_refs={"N001": outgoing},
    )

    assert len(result.edges) == 1
    assert result.edges[0].cross_sheet is True

    assembled = assemble_graph(
        objects=FusionResult(
            graph=ReconciledGraph(source_path="drawing.pdf", nodes=[outgoing, incoming])
        ),
        pages=[_page(0), _page(1)],
        topology=[TopologyResult(page_index=0), TopologyResult(page_index=1)],
        page_graph_results=[result],
        per_page_status={0: "ok", 1: "ok"},
    )
    assert len(assembled.graph.edges) == 1
    assert assembled.graph.dangling_opcs == []


def test_opc_update_without_literal_evidence_is_rejected() -> None:
    opc = _node("opc", kind="opc", direction="out")
    result = validate_page_graph_submission(
        page=_page(),
        pages=[_page()],
        submission=PageGraphSubmission(
            opc_updates=[OpcUpdate(node_ref="N001", service="compressed air")]
        ),
        nodes_by_ref={"N001": opc},
        edges_by_ref={},
        local_node_refs={"N001": opc},
    )

    assert result.opc_updates == {}
    assert any("no printed evidence" in value for value in result.diagnostics)
