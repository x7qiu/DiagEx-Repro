from __future__ import annotations

import base64
import io
from types import SimpleNamespace

from PIL import Image, ImageDraw

from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.contextual import (
    ContextCorrection,
    ContextSubmission,
    _validate_submission,
    apply_contextual_results,
    find_contextual_candidates,
    resolve_contextual_page,
)
from diagex.vision.fusion import FusionResult
from diagex.vision.models import BBox, ReconciledGraph, ReconciledNode


def _valve(node_id: str, x: int, y: int) -> ReconciledNode:
    return ReconciledNode(
        id=node_id,
        kind="equipment",
        label="unlabelled",
        bbox_global=BBox(x=x, y=y, w=45, h=35),
        page_index=0,
        attributes={"valve_type": "other"},
        confidence="medium",
    )


def _glyph(node_id: str, x: int, y: int) -> ReconciledNode:
    return ReconciledNode(
        id=node_id,
        kind="instrument",
        label="S",
        bbox_global=BBox(x=x, y=y, w=50, h=28),
        page_index=0,
        attributes={"instrument_function": "unclassified_instrument"},
        confidence="low",
        source_quote="S",
        source_evidence_ids=["txt-s"],
    )


def test_candidate_finder_keeps_attached_composite_and_isolated_ms_for_context() -> None:
    valve = _valve("valve", 100, 160)
    attached = _glyph("s", 98, 105)
    remote = _glyph("remote", 700, 400)
    package_interface = _glyph("interface-a", 150, 120).model_copy(
        update={"label": "A", "source_quote": "A"}
    )

    candidates = find_contextual_candidates(
        [valve, attached, remote, package_interface], page_index=0
    )

    assert len(candidates) == 2
    assert candidates[0].node_ids[0] == "s"
    assert "valve" in candidates[0].node_ids
    assert candidates[1].node_ids == ["remote"]
    assert all("interface-a" not in candidate.node_ids for candidate in candidates)


def test_already_classified_s_valve_does_not_pair_with_itself() -> None:
    combined = _valve("combined", 100, 160).model_copy(
        update={
            "label": "S",
            "attributes": {"valve_type": "other", "actuation": "solenoid"},
        }
    )

    assert find_contextual_candidates([combined], page_index=0) == []


def test_high_confidence_project_legend_merge_sets_actuation_before_topology() -> None:
    from diagex.vision.models import EquipmentAssembly

    valve = _valve("valve", 100, 160)
    glyph = _glyph("s", 98, 105)
    nodes = [valve, glyph]
    glyph.attributes["assembly_ids"] = ["assembly"]
    group = find_contextual_candidates(nodes, page_index=0)[0]
    node_refs = {"N001": glyph, "N002": valve}
    ref_by_id = {node.id: ref for ref, node in node_refs.items()}
    submission = ContextSubmission(
        corrections=[
            ContextCorrection(
                group_ref="C001",
                operation="merge_composite",
                primary_ref="N002",
                absorbed_refs=["N001"],
                valve_type="other",
                actuation="solenoid",
                legend_refs=["L001"],
                confidence="high",
                evidence=["boxed S matches the project electromagnetic-actuator legend"],
            )
        ]
    )

    result = _validate_submission(
        page_index=0,
        submission=submission,
        node_refs=node_refs,
        group_refs={"C001": group},
        ref_by_id=ref_by_id,
        legend_refs={
            "L001": {
                "label": "电磁执行机构",
                "kind": "equipment",
                "symbol_class": "unclassified_equipment",
            }
        },
    )
    fused = FusionResult(
        graph=ReconciledGraph(
            source_path="drawing.pdf",
            nodes=nodes,
            assemblies=[
                EquipmentAssembly(
                    id="assembly",
                    page_index=0,
                    bbox_global=BBox(x=50, y=50, w=300, h=300),
                    member_node_ids=["s", "valve"],
                )
            ],
        ),
        ambiguities=[],
    )
    corrected = apply_contextual_results(fused, [result])

    assert not result.diagnostics
    assert len(result.resolutions) == 1
    assert [node.id for node in corrected.graph.nodes] == ["valve"]
    corrected_valve = corrected.graph.nodes[0]
    assert corrected_valve.attributes["actuation"] == "solenoid"
    assert corrected_valve.attributes["assembly_ids"] == ["assembly"]
    assert corrected.graph.assemblies[0].member_node_ids == ["valve"]
    assert fused.graph.assemblies[0].member_node_ids == ["s", "valve"]
    assert corrected_valve.source_evidence_ids == ["txt-s"]
    assert corrected_valve.attributes["contextual_resolutions"][0]["absorbed_node_ids"] == ["s"]


def test_merge_is_not_applied_without_matching_project_legend_evidence() -> None:
    valve = _valve("valve", 100, 160)
    glyph = _glyph("s", 98, 105)
    nodes = [valve, glyph]
    group = find_contextual_candidates(nodes, page_index=0)[0]
    node_refs = {"N001": glyph, "N002": valve}
    result = _validate_submission(
        page_index=0,
        submission=ContextSubmission(
            corrections=[
                ContextCorrection(
                    group_ref="C001",
                    operation="merge_composite",
                    primary_ref="N002",
                    absorbed_refs=["N001"],
                    actuation="solenoid",
                    legend_refs=["L001"],
                    confidence="high",
                    evidence=["possible S actuator"],
                )
            ]
        ),
        node_refs=node_refs,
        group_refs={"C001": group},
        ref_by_id={node.id: ref for ref, node in node_refs.items()},
        legend_refs={"L001": {"label": "手动执行机构"}},
    )

    assert not result.resolutions
    assert result.conflicts[0]["status"] == "unresolved"
    assert any("does not support solenoid" in item for item in result.diagnostics)


def test_merge_cannot_absorb_an_unlisted_or_remote_node() -> None:
    valve = _valve("valve", 100, 160)
    glyph = _glyph("s", 98, 105)
    remote = _glyph("remote", 700, 400)
    group = find_contextual_candidates([valve, glyph, remote], page_index=0)[0]
    node_refs = {"N001": glyph, "N002": valve, "N003": remote}
    result = _validate_submission(
        page_index=0,
        submission=ContextSubmission(
            corrections=[
                ContextCorrection(
                    group_ref="C001",
                    operation="merge_composite",
                    primary_ref="N002",
                    absorbed_refs=["N003"],
                    actuation="solenoid",
                    legend_refs=["L001"],
                    confidence="high",
                )
            ]
        ),
        node_refs=node_refs,
        group_refs={"C001": group},
        ref_by_id={node.id: ref for ref, node in node_refs.items()},
        legend_refs={"L001": {"label": "电磁执行机构"}},
    )

    assert not result.resolutions
    assert any("out-of-group" in item for item in result.diagnostics)


def test_contextual_call_sends_page_details_and_saved_legend_symbol_image() -> None:
    valve = _valve("valve", 100, 160)
    glyph = _glyph("s", 98, 105)
    legend_image = Image.new("RGB", (120, 70), "white")
    draw = ImageDraw.Draw(legend_image)
    draw.rectangle((35, 5, 75, 35), outline="black", width=2)
    draw.text((50, 12), "S", fill="black")
    draw.line((55, 35, 55, 65), fill="black", width=2)
    buffer = io.BytesIO()
    legend_image.save(buffer, format="PNG")
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )

    class Client:
        def __init__(self) -> None:
            self.request: dict[str, object] | None = None

        def messages_create(self, **kwargs: object) -> object:
            self.request = kwargs
            return SimpleNamespace(
                id="ctx-1",
                model="vision-model",
                stop_reason="tool_use",
                usage=usage,
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_contextual_symbol_corrections",
                        "input": {
                            "corrections": [
                                {
                                    "group_ref": "C001",
                                    "operation": "merge_composite",
                                    "primary_ref": "N002",
                                    "absorbed_refs": ["N001"],
                                    "valve_type": "other",
                                    "actuation": "solenoid",
                                    "legend_refs": ["L001"],
                                    "confidence": "high",
                                    "evidence": ["boxed S matches L001 and is attached"],
                                }
                            ]
                        },
                    }
                ],
            )

    client = Client()
    cost = CostTracker()
    result = resolve_contextual_page(
        client=client,  # type: ignore[arg-type]
        cost_tracker=cost,
        reporter=NullReporter(),
        page_index=0,
        rendered_image=Image.new("RGB", (1000, 600), "white"),
        nodes=[valve, glyph],
        legend_entries=[
            {
                "label": "电磁执行机构",
                "kind": "equipment",
                "symbol_class": "unclassified_equipment",
                "source": "legend_extracted",
                "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        ],
        step=7,
    )

    assert len(result.resolutions) == 1
    assert result.resolutions[0].actuation == "solenoid"
    assert [row.step for row in cost.steps] == [7]
    assert client.request is not None
    content = client.request["messages"][0]["content"]  # type: ignore[index]
    assert [block["type"] for block in content[:3]] == ["image", "image", "image"]
    assert client.request["tool_choice"] == {
        "type": "tool",
        "name": "submit_contextual_symbol_corrections",
    }
