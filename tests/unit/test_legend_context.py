"""Relevant graphical legend handoff, independent of pack order and tag names."""

from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

from PIL import Image, ImageDraw

from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.legend_context import MAX_LEGEND_IMAGES, SHAPES, select_legend_context
from diagex.vision.models import BBox, Tile
from diagex.vision.perception import perceive_tile
from diagex.vision.symbol_candidates import SymbolCandidate
from tests.unit.test_port_topology import page
from tests.unit.test_symbol_perception import view


def candidate(shape):
    return SymbolCandidate(
        id=shape,
        page_index=0,
        shape=shape,
        source_path_ids=["ink"],
        bbox=BBox(x=100, y=100, w=40, h=30),
        text=["PI"],
    )


def png():
    image = Image.new("RGB", (60, 40), "white")
    ImageDraw.Draw(image).rectangle((5, 5, 55, 35), outline="black")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def definition(shape, index=0, **updates):
    return {
        "label": f"{shape} printed convention",
        "source_row_id": f"{shape}-{index}",
        "kind": "connector" if shape == "continuation_arrow" else "instrument",
        "symbol_class": "opc" if shape == "continuation_arrow" else "unclassified_instrument",
        "source": "legend_extracted",
        "image_b64": png(),
        "crop_quality": "recovered",
        "attributes": {"row_status": "accept", "candidate_shapes": json.dumps([shape])},
        **updates,
    }


def abbreviations():
    return [
        {
            "label": code,
            "kind": "instrument",
            "symbol_class": "indicator",
            "attributes": {"legend_kind": "abbreviation"},
        }
        for code in [*(f"CODE{i}" for i in range(400)), "PI", "P"]
    ]


def test_graphical_families_and_images_survive_large_abbreviation_prefix():
    entries = [*abbreviations(), *(definition(shape, i) for shape in SHAPES for i in range(20))]
    cs = [candidate(shape) for shape in SHAPES]
    selected, images, refs = select_legend_context(entries, cs, ["PI"])
    assert len(selected) == 33  # 32 graphical + the exactly matching PI abbreviation
    assert sum(b["type"] == "image" for b in images) == MAX_LEGEND_IMAGES
    assert {
        e["label"] for e in selected if e.get("attributes", {}).get("legend_kind") == "abbreviation"
    } == {"PI"}
    assert all(refs[c.id] for c in cs)
    imaged = {e["legend_entry_id"] for e in selected if e.get("has_reference_image")}
    assert all(set(refs[c.id]) & imaged for c in cs)
    assert all("image_b64" not in e for e in selected)
    assert select_legend_context(list(reversed(entries)), cs, ["PI"]) == (selected, images, refs)


def test_uncertain_or_invalid_definitions_are_not_presented_as_authoritative():
    good = definition("instrument_frame")
    bad = [
        definition("instrument_frame", 1, crop_quality="rejected_text_overlap"),
        definition("instrument_frame", 2, attributes={"row_status": "uncertain"}),
    ]
    selected, images, _ = select_legend_context([*bad, good], [candidate("instrument_frame")], [])
    assert [e["legend_entry_id"] for e in selected] == [good["source_row_id"]]
    assert len(images) == 2
    selected, images, _ = select_legend_context(
        [definition("instrument_frame", image_b64="broken")], [candidate("instrument_frame")], []
    )
    assert len(selected) == 1 and images == []


def test_repeated_label_rows_keep_distinct_reference_ids():
    selected, _, _ = select_legend_context(
        [definition("instrument_frame", i) for i in range(2)], [candidate("instrument_frame")], []
    )
    assert len(selected) == 2 and selected[0]["legend_entry_id"] != selected[1]["legend_entry_id"]


def test_capsule_retrieval_includes_connected_equipment_definition_with_shared_chinese_words():
    compressor = definition("connected_frame", kind="equipment", symbol_class="compressor", label="电动往复压缩机")
    vessels = [definition("capsule_body", i, kind="equipment", symbol_class="vessel", label="立式容器") for i in range(40)]
    selected, _, refs = select_legend_context([*vessels, compressor], [candidate("capsule_body")], ["螺杆式压缩机"])
    assert selected[0]["legend_entry_id"] == compressor["source_row_id"]
    assert selected[0]["has_reference_image"]
    assert compressor["source_row_id"] in refs["capsule_body"]


def test_explicit_actuator_legend_role_cannot_type_its_contextual_valve_body():
    from diagex.vision.legend_models import LegendEntry
    entry = definition("valve_body", kind="valve", symbol_class="unclassified_valve", attributes={"row_status": "accept", "symbol_role": "actuator", "candidate_shapes": '["valve_body"]', "valve_type": "diaphragm actuator"})
    normalized = LegendEntry.model_validate(entry)
    assert normalized.attributes["row_status"] == "uncertain"
    assert normalized.symbol_class == "unclassified_valve"
    assert "original_role_classification" in normalized.attributes
    selected, _, _ = select_legend_context([entry], [candidate("valve_body")], [])
    assert not selected
    assert LegendEntry.model_validate({**entry, "source": "customer_override"}).kind == "valve"


def test_source_continuation_image_reaches_existing_call_without_requiring_destination():
    p = page([])
    cs = [candidate("continuation_arrow")]
    source = definition("continuation_arrow")
    calls = []

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            content = kwargs["messages"][0]["content"]
            prompt = json.loads(content[1]["text"].split("\n")[-1])
            assert prompt["native_symbol_candidates"][0]["legend_entry_ids"] == [
                source["source_row_id"]
            ]
            assert prompt["legend_entries"][0]["has_reference_image"]
            assert sum(b["type"] == "image" for b in content) == 3  # original, guides, legend
            assert "Legend reference" in content[3]["text"]
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {"objects": [{"kind": "opc", "candidate_id": cs[0].id}]},
                    }
                ],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    box = BBox(x=0, y=0, w=p.width, h=p.height)
    outcome = perceive_tile(
        client=Client(),
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=p,
        tile=Tile(id="test", page_index=0, bbox=box),
        view_image=Image.new("RGB", (500, 320), "white"),
        view_info=view(box),
        ownership_bbox=box,
        legend_summary=[*abbreviations(), source],
        step=1,
        candidates=cs,
    )
    assert len(calls) == 1
    assert len(outcome.detections) == 1 and outcome.detections[0].kind == "opc"
    assert outcome.detections[0].attributes["supplied_legend_image_ids"] == [
        source["source_row_id"]
    ]
    assert outcome.detections[0].bbox == cs[0].bbox
    assert not outcome.detections[0].attributes.get("drawing_ref")
