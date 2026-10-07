"""Line and boundary definitions do not become object exemplars through shape hints."""

import copy
import json

import pytest

from diagex.vision.legend_context import (
    SHAPES,
    select_graph_legend_context,
    select_legend_context,
)
from tests.unit.test_legend_context import candidate, definition


def source_entries():
    # Routing fields transcribed from run r-9bee; glyph rendering is irrelevant
    # to this regression. The development report retains original source PNGs.
    return [
        definition(
            "continuation_arrow",
            source_row_id="legend-row-e9ba3d6b29d41566",
            label="次工艺管线",
            kind="line",
            symbol_class="continuation_arrow",
        ),
        definition(
            "continuation_arrow",
            source_row_id="legend-row-b4609dee738f6389",
            label="制造商供货/成套设备范围分界",
            kind="line",
            symbol_class="boundary_line",
            attributes={
                "row_status": "accept",
                "line_type": "scope_boundary",
                "candidate_shapes": '["continuation_arrow"]',
            },
        ),
        definition(
            "continuation_arrow",
            source_row_id="legend-row-7039a790096cb41d",
            label="续接PID图号",
            symbol_class="drawing_reference",
        ),
    ]


def test_saved_process_line_and_scope_examples_cannot_type_an_endpoint():
    entries = source_entries()
    original = copy.deepcopy(entries)
    selected, images, refs = select_legend_context(entries, [candidate("continuation_arrow")], [])
    expected = entries[2]["source_row_id"]
    assert [e["legend_entry_id"] for e in selected] == [expected]
    assert refs == {"continuation_arrow": [expected]}
    assert sum(b["type"] == "image" for b in images) == 1
    assert entries == original  # Historical records stay unchanged/readable.
    # Graph/line consumers still have the definitions, including boundary style.
    assert len(select_graph_legend_context(entries, [])) == 3


@pytest.mark.parametrize("shape", SHAPES)
def test_line_style_hints_cannot_type_any_object_family(shape):
    entry = definition(shape, kind="line", symbol_class="signal_line")
    selected, images, refs = select_legend_context([entry], [candidate(shape)], [])
    assert selected == images == []
    assert refs[shape] == []


@pytest.mark.parametrize(
    "updates",
    [
        {"reference_type": "scope_boundary"},
        {"reference_type": "line_style"},
        {"symbol_class": "boundary_line"},
        {"attributes": {"line_type": "scope_boundary"}},
        {"attributes": {"symbol_role": "line_style"}},
    ],
)
def test_explicit_nonobject_role_wins_over_erroneous_equipment_or_connector_hint(updates):
    entry = definition("connected_frame", kind="equipment", **updates)
    entry["attributes"]["candidate_shapes"] = json.dumps(list(SHAPES))
    selected, images, _ = select_legend_context([entry], [candidate(s) for s in SHAPES], [])
    assert selected == images == []


@pytest.mark.parametrize(
    "cls", ["opc", "off_page_connector", "offpage_connector", "drawing_reference"]
)
def test_legacy_line_record_with_explicit_endpoint_class_still_reaches_endpoint(cls):
    entry = definition("continuation_arrow", kind="line", symbol_class=cls, attributes={})
    selected, images, refs = select_legend_context([entry], [candidate("continuation_arrow")], [])
    assert [e["legend_entry_id"] for e in selected] == [entry["source_row_id"]]
    assert refs["continuation_arrow"] == [entry["source_row_id"]]
    assert sum(b["type"] == "image" for b in images) == 1


def test_signal_and_process_endpoint_definitions_keep_their_distinct_source_meanings():
    entries = [
        definition(
            "continuation_arrow",
            0,
            label="Drawing-to-drawing signal connector",
            symbol_class="signal_connector",
        ),
        definition(
            "continuation_arrow", 1, label="Process piping continuation", symbol_class="opc"
        ),
    ]
    selected, _, refs = select_legend_context(entries, [candidate("continuation_arrow")], [])
    assert {e["symbol_class"] for e in selected} == {"signal_connector", "opc"}
    assert len(refs["continuation_arrow"]) == 2


def test_absent_shape_hints_keep_ordinary_legacy_symbol_families():
    entry = definition("round_symbol", kind="instrument", attributes={})
    selected, _, refs = select_legend_context([entry], [candidate("instrument_frame")], [])
    assert selected and refs["instrument_frame"] == [entry["source_row_id"]]


def test_failed_previous_classification_is_audit_only_not_symbol_guidance():
    entry = definition("valve_body", kind="equipment", symbol_class="actuator")
    entry["attributes"].update(
        symbol_role="actuator",
        classification_repair_evidence=json.dumps({"previous_response": [{"kind": "valve"}]}),
    )
    selected, _, _ = select_legend_context([entry], [candidate("valve_body")], [])
    assert selected[0]["attributes"]["symbol_role"] == "actuator"
    assert "classification_repair_evidence" not in selected[0]["attributes"]
    assert "classification_repair_evidence" in entry["attributes"]
