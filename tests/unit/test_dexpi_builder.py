"""Unit tests for the Phase 2 DexpiBuilder (spec §7.2)."""

from pathlib import Path
from xml.etree import ElementTree as ET

from diagex.dexpi import xml_io
from diagex.extractors.dexpi_builder import build_dexpi, serialize_model, validate_model
from diagex.vision.models import BBox, ReconciledEdge, ReconciledGraph, ReconciledNode


def _node(
    nid: str,
    kind: str,
    label: str,
    attrs: dict,
    *,
    x: int = 0,
    y: int = 0,
    page: int = 0,
    conf: str = "high",
) -> ReconciledNode:
    return ReconciledNode(
        id=nid,
        kind=kind,  # type: ignore[arg-type]
        label=label,
        bbox_global=BBox(x=x, y=y, w=20, h=20),
        page_index=page,
        attributes=attrs,
        confidence=conf,  # type: ignore[arg-type]
    )


def _edge(
    eid: str,
    src: str,
    dst: str,
    line_type: str | None = "process",
    conf: str = "high",
) -> ReconciledEdge:
    return ReconciledEdge(
        id=eid,
        from_node=src,
        to_node=dst,
        line_type=line_type,  # type: ignore[arg-type]
        polyline_global=[(0, 0), (10, 0)],
        confidence=conf,  # type: ignore[arg-type]
    )


def test_pump_class_selection_respects_pump_type_hint():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("n1", "equipment", "P-101", {"equipment_class": "pump", "pump_type": "centrifugal"}),
            _node("n2", "equipment", "P-201", {"equipment_class": "pump", "pump_type": "reciprocating"}),
            _node("n3", "equipment", "P-301", {"equipment_class": "pump"}),  # generic
        ],
    )
    result = build_dexpi(g)
    assert result.stats["equipment_count"] == 3
    class_names = sorted(type(e).__name__ for e in result.model.conceptual_model.TaggedPlantItems)
    assert "CentrifugalPump" in class_names
    assert "ReciprocatingPump" in class_names
    assert "Pump" in class_names


def test_unclassified_equipment_escape_hatch():
    """v2 Phase C: unrecognised equipment_class falls back to ProcessEquipment with
    the LLM-supplied label captured on ``customAttributes['agent_equipment_class']``."""
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("n1", "equipment", "MYSTERY", {}, conf="low"),
            _node("n2", "equipment", "X-1", {"equipment_class": "not_in_vocabulary"}, conf="low"),
        ],
    )
    result = build_dexpi(g)
    assert result.stats["unclassified_count"] == 2
    items = result.model.conceptual_model.TaggedPlantItems
    class_names = [type(e).__name__ for e in items]
    assert all(n == "ProcessEquipment" for n in class_names)
    captured = [
        ca.value
        for e in items
        for ca in e.customAttributes
        if ca.name == "agent_equipment_class"
    ]
    assert "not_in_vocabulary" in captured
    # Issues list captures the fall-through so the confidence report can surface it.
    assert any("unclassified" in msg.lower() for msg in result.issues)


def test_valve_subtype_mapping():
    # Valves land on segments only when an edge references them — P&ID-faithful.
    nodes = [_node("t1", "equipment", "T-1", {"equipment_class": "tank"})]
    edges: list[ReconciledEdge] = []
    for i, vt in enumerate(["gate", "globe", "check", "ball", "butterfly"], start=1):
        vid = f"v{i}"
        nodes.append(_node(vid, "equipment", f"V-{i}", {"equipment_class": "valve", "valve_type": vt}))
        edges.append(_edge(f"e{i}", "t1", vid))
    g = ReconciledGraph(source_path="t.pdf", nodes=nodes, edges=edges)
    result = build_dexpi(g)
    assert result.stats["valve_count"] == 5
    classes = {
        type(s).__name__
        for network in result.model.conceptual_model.PipingNetworkSystems
        for segment in network.Segments
        for s in segment.Items
    }
    for expected in ("GateValve", "GlobeValve", "CheckValve", "BallValve", "ButterflyValve"):
        assert expected in classes


def test_control_valve_falls_back_to_operated_valve():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("t1", "equipment", "T-1", {"equipment_class": "tank"}),
            _node("v1", "equipment", "FCV-101", {"equipment_class": "valve", "valve_type": "control"}),
        ],
        edges=[_edge("e1", "t1", "v1")],
    )
    result = build_dexpi(g)
    # DEXPI 2.0 has no dedicated ControlValve — must map to OperatedValve with a custom-attr note.
    found = []
    for net in result.model.conceptual_model.PipingNetworkSystems:
        for seg in net.Segments:
            found.extend(type(s).__name__ for s in seg.Items)
    assert "OperatedValve" in found


def test_valve_shared_by_two_segments_has_one_xml_owner_and_endpoint_refs():
    graph = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("t1", "equipment", "T-1", {"equipment_class": "tank"}),
            _node(
                "v1",
                "equipment",
                "V-1",
                {"equipment_class": "valve", "valve_type": "gate"},
            ),
            _node("t2", "equipment", "T-2", {"equipment_class": "tank"}),
        ],
        edges=[_edge("e1", "t1", "v1"), _edge("e2", "v1", "t2")],
    )
    result = build_dexpi(graph)
    segments = [
        segment
        for network in result.model.conceptual_model.PipingNetworkSystems
        for segment in network.Segments
    ]
    valve = next(item for segment in segments for item in segment.Items)
    assert sum(valve in segment.Items for segment in segments) == 1
    assert any(segment.TargetItem is valve for segment in segments)
    assert any(segment.SourceItem is valve for segment in segments)

    root = ET.fromstring(xml_io.dumps(result.model))
    assert len(root.findall(f".//Object[@id='{valve.id}']")) == 1
    assert len(root.findall(f".//References[@objects='#{valve.id}']")) == 2


def test_opc_direction_flips_class():
    g_out = ReconciledGraph(
        source_path="t.pdf",
        nodes=[_node("o1", "opc", "OPC-12", {"direction": "out"})],
    )
    g_in = ReconciledGraph(
        source_path="t.pdf",
        nodes=[_node("o1", "opc", "OPC-12", {"direction": "in"})],
    )
    r_out = build_dexpi(g_out)
    r_in = build_dexpi(g_in)
    out_types = {type(s).__name__ for net in r_out.model.conceptual_model.PipingNetworkSystems
                 for seg in net.Segments for s in seg.Items}
    in_types = {type(s).__name__ for net in r_in.model.conceptual_model.PipingNetworkSystems
                for seg in net.Segments for s in seg.Items}
    assert any("OffPageConnector" in t for t in out_types)
    assert any("OffPageConnector" in t for t in in_types)


def test_dropped_edge_when_endpoint_unresolved():
    # Empty-string endpoints are now rejected at the schema layer (see
    # ReconciledEdge validator). The remaining drop path is an endpoint that
    # is non-empty but refers to a node not present in the graph.
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[_node("n1", "equipment", "T-1", {"equipment_class": "tank"})],
        edges=[_edge("e1", "n1", "n-missing")],
    )
    result = build_dexpi(g)
    assert result.stats["dropped_edges"] == 1


def test_instrument_gets_loop_number_from_attributes():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "i1", "instrument", "FIC-101",
                {"instrument_function": "controller", "measured_variable": "flow", "loop_number": "101"},
            ),
        ],
    )
    result = build_dexpi(g)
    pifs = result.model.conceptual_model.ProcessInstrumentationFunctions
    assert len(pifs) == 1
    assert pifs[0].ProcessInstrumentationFunctionNumber == "101"


def test_validate_model_is_empty_on_minimal_graph(tmp_path: Path):
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[_node("n1", "equipment", "T-1", {"equipment_class": "tank"})],
    )
    result = build_dexpi(g)
    assert validate_model(result.model) == []
    written = serialize_model(result.model, tmp_path, "pid.dexpi")
    assert written.exists()
    assert written.name == "pid.dexpi.json"


def test_instrument_signal_has_valid_source_and_target() -> None:
    graph = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("i1", "instrument", "FT-101", {"loop_number": "101"}),
            _node("i2", "instrument", "FIC-101", {"loop_number": "101"}),
        ],
        edges=[_edge("s1", "i1", "i2", "signal_electric")],
    )

    result = build_dexpi(graph)
    functions = result.model.conceptual_model.ProcessInstrumentationFunctions
    signal = functions[0].SignalConveyingFunctions[0]
    assert signal.Source is functions[0]
    assert signal.Target is functions[1]
    assert signal.SignalConveyingType.value == "ElectricalSignalConveying"
    assert result.stats["signal_count"] == 1
    assert validate_model(result.model) == []
    assert validate_model(xml_io.loads(xml_io.dumps(result.model))) == []


def test_equipment_instrument_capillary_uses_sensing_location() -> None:
    graph = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("tank", "equipment", "T-101", {"equipment_class": "tank"}),
            _node(
                "pt",
                "instrument",
                "PT-101",
                {"instrument_function": "transmitter", "loop_number": "101"},
            ),
        ],
        edges=[_edge("cap-1", "tank", "pt", "instrument_capillary")],
    )

    result = build_dexpi(graph)
    plant = result.model.conceptual_model
    instrument = plant.ProcessInstrumentationFunctions[0]
    equipment = plant.TaggedPlantItems[0]
    generating = instrument.ProcessSignalGeneratingFunctions[0]
    measuring = instrument.SignalConveyingFunctions[0]

    assert generating.sensing_location in equipment.Nozzles
    assert measuring.Source is generating
    assert measuring.Target is instrument
    assert measuring.SignalConveyingType.value == "CapillarySignalConveying"
    assert result.stats["dropped_edges"] == 0
    assert validate_model(result.model) == []
    assert validate_model(xml_io.loads(xml_io.dumps(result.model))) == []


def test_instrument_to_valve_signal_uses_actuating_function() -> None:
    graph = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("tank", "equipment", "T-1", {"equipment_class": "tank"}),
            _node(
                "valve",
                "equipment",
                "FCV-101",
                {"equipment_class": "valve", "valve_type": "control"},
            ),
            _node("controller", "instrument", "FIC-101", {"loop_number": "101"}),
        ],
        edges=[
            _edge("p1", "tank", "valve", "process"),
            _edge("s1", "controller", "valve", "signal_pneumatic"),
        ],
    )

    result = build_dexpi(graph)
    controller = result.model.conceptual_model.ProcessInstrumentationFunctions[0]
    actuating = controller.ActuatingFunctions[0]
    signal = controller.SignalConveyingFunctions[0]
    assert signal.Source is controller
    assert signal.Target is actuating
    assert signal.SignalConveyingType.value == "PneumaticSignalConveying"
    assert actuating.Systems.operated_valve_reference.Valve is not None
    assert validate_model(result.model) == []
    assert validate_model(xml_io.loads(xml_io.dumps(result.model))) == []


def test_signal_only_operated_valve_is_owned_and_xml_reference_resolves() -> None:
    graph = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "valve",
                "equipment",
                "FCV-101",
                {"equipment_class": "valve", "valve_type": "control"},
            ),
            _node("controller", "instrument", "FIC-101", {"loop_number": "101"}),
        ],
        edges=[_edge("s1", "controller", "valve", "signal_pneumatic")],
    )

    result = build_dexpi(graph)
    round_tripped = xml_io.loads(xml_io.dumps(result.model))

    assert validate_model(result.model) == []
    assert validate_model(round_tripped) == []
    assert any("signal-only or unconnected operated valve" in issue for issue in result.issues)


def test_provisional_review_edge_is_not_asserted_in_dexpi() -> None:
    graph = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("left", "equipment", "T-1", {"equipment_class": "tank"}),
            _node("right", "equipment", "P-1", {"equipment_class": "pump"}),
        ],
        edges=[
            _edge("draft", "left", "right").model_copy(
                update={"attributes": {"provisional_review_only": True}}
            )
        ],
    )

    result = build_dexpi(graph)

    assert result.stats["segment_count"] == 0
    assert result.stats["dropped_edges"] == 0
    assert any("provisional review-only" in issue for issue in result.issues)


def test_build_is_deterministic_in_stats():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("n1", "equipment", "P-101", {"equipment_class": "pump"}),
            _node("n2", "instrument", "FIC-101", {"instrument_function": "controller", "loop_number": "101"}),
            _node("n3", "opc", "OPC-3", {"direction": "out"}),
        ],
        edges=[_edge("e1", "n1", "n2")],
    )
    r1 = build_dexpi(g)
    r2 = build_dexpi(g)
    assert r1.stats == r2.stats


# ---------------------------------------------------------------------------
# Layer 2 additions (v0.2 DEXPI subset)
# ---------------------------------------------------------------------------


def _class_names_of_equipment(result) -> list[str]:
    return [type(e).__name__ for e in result.model.conceptual_model.TaggedPlantItems]


def test_new_equipment_classes_round_trip():
    """Every v0.2 equipment key that has a real DEXPI 2.0 class maps to it."""
    keys_with_pydexpi = {
        "agitator": "Agitator",
        "filter": "Filter",
        "separator": "Separator",
        "fired_heater": "Furnace",
        "cooling_tower": "CoolingTower",
        "fan_blower": "Fan",
        "turbine": "Turbine",
        "centrifuge": "Centrifuge",
    }
    nodes = [
        _node(f"n{i}", "equipment", f"E-{i}", {"equipment_class": key})
        for i, key in enumerate(keys_with_pydexpi)
    ]
    g = ReconciledGraph(source_path="t.pdf", nodes=nodes)
    result = build_dexpi(g)
    names = _class_names_of_equipment(result)
    for expected in keys_with_pydexpi.values():
        assert expected in names, f"{expected} missing from {names}"


def test_new_equipment_subtype_hints_resolve():
    cases = [
        ("filter", {"filter_type": "liquid"}, "LiquidFilter"),
        ("filter", {"filter_type": "gas"}, "GasFilter"),
        ("separator", {"separator_type": "gravity"}, "GravitationalSeparator"),
        ("separator", {"separator_type": "scrubber"}, "ScrubbingSeparator"),
        ("cooling_tower", {"cooling_tower_type": "wet"}, "WetCoolingTower"),
        ("cooling_tower", {"cooling_tower_type": "dry"}, "DryCoolingTower"),
        ("turbine", {"turbine_type": "steam"}, "SteamTurbine"),
        ("turbine", {"turbine_type": "gas"}, "GasTurbine"),
        ("fan_blower", {"fan_blower_type": "axial"}, "AxialFan"),
        ("fan_blower", {"fan_blower_type": "radial"}, "RadialFan"),
        ("fan_blower", {"fan_blower_type": "centrifugal_blower"}, "CentrifugalBlower"),
        ("centrifuge", {"centrifuge_type": "filtering"}, "FilteringCentrifuge"),
    ]
    nodes = []
    for i, (cls_key, attrs, _expected) in enumerate(cases):
        nodes.append(_node(f"n{i}", "equipment", f"E-{i}", {"equipment_class": cls_key, **attrs}))
    g = ReconciledGraph(source_path="t.pdf", nodes=nodes)
    result = build_dexpi(g)
    names = _class_names_of_equipment(result)
    for _, _, expected in cases:
        assert expected in names, f"{expected} missing from {names}"


def test_reactor_falls_back_to_process_equipment():
    """v2 Phase C: DEXPI 2.0 has no Reactor class; the builder routes it to
    ``ProcessEquipment`` and stores the LLM key on customAttributes."""
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[_node("n1", "equipment", "R-101", {"equipment_class": "reactor"})],
    )
    result = build_dexpi(g)
    items = result.model.conceptual_model.TaggedPlantItems
    assert len(items) == 1
    assert type(items[0]).__name__ == "ProcessEquipment"
    captured = [
        ca.value for ca in items[0].customAttributes if ca.name == "agent_equipment_class"
    ]
    assert captured == ["reactor"]


def test_typo_in_equipment_class_produces_suggestion():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[_node("n1", "equipment", "V-1", {"equipment_class": "tnk"})],
    )
    result = build_dexpi(g)
    assert any("did you mean 'tank'" in msg for msg in result.issues), result.issues


def test_new_valve_types_round_trip():
    valves = [
        ("safety_relief", "SafetyValveOrFitting"),
        ("needle", "NeedleValve"),
        ("plug", "PlugValve"),
        ("strainer", "Strainer"),
        ("rupture_disc", "RuptureDisc"),
    ]
    nodes = [_node("t1", "equipment", "T-1", {"equipment_class": "tank"})]
    edges = []
    for i, (vt, _) in enumerate(valves, start=1):
        vid = f"v{i}"
        nodes.append(_node(vid, "equipment", f"V-{i}", {"equipment_class": "valve", "valve_type": vt}))
        edges.append(_edge(f"e{i}", "t1", vid))
    g = ReconciledGraph(source_path="t.pdf", nodes=nodes, edges=edges)
    result = build_dexpi(g)
    classes = {
        type(s).__name__
        for net in result.model.conceptual_model.PipingNetworkSystems
        for seg in net.Segments
        for s in seg.Items
    }
    for _, expected in valves:
        assert expected in classes, f"{expected} missing from {classes}"


def test_three_way_valve_falls_back_to_operated_valve():
    """v2 Phase C: DEXPI 2.0 has no ThreeWayValve — falls back to OperatedValve
    with the LLM-supplied valve_type captured on customAttributes."""
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("t1", "equipment", "T-1", {"equipment_class": "tank"}),
            _node("v1", "equipment", "V-3W", {"equipment_class": "valve", "valve_type": "three_way"}),
        ],
        edges=[_edge("e1", "t1", "v1")],
    )
    result = build_dexpi(g)
    items = [
        s
        for net in result.model.conceptual_model.PipingNetworkSystems
        for seg in net.Segments
        for s in seg.Items
    ]
    classes = {type(s).__name__ for s in items}
    assert "OperatedValve" in classes
    # Original LLM key is preserved for downstream consumers.
    captured = [
        ca.value
        for s in items
        for ca in getattr(s, "customAttributes", [])
        if ca.name == "agent_valve_type"
    ]
    assert "three_way" in captured


def test_unknown_valve_type_produces_suggestion():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("t1", "equipment", "T-1", {"equipment_class": "tank"}),
            _node("v1", "equipment", "V-1", {"equipment_class": "valve", "valve_type": "bal"}),
        ],
        edges=[_edge("e1", "t1", "v1")],
    )
    result = build_dexpi(g)
    assert any("did you mean 'ball'" in msg for msg in result.issues), result.issues


def test_instrument_class_control_loop_upgrades_to_process_control_function():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "i1", "instrument", "FIC-101",
                {
                    "instrument_function": "controller",
                    "loop_number": "101",
                    "instrument_class": "control_loop",
                },
            ),
        ],
    )
    result = build_dexpi(g)
    pifs = result.model.conceptual_model.ProcessInstrumentationFunctions
    assert len(pifs) == 1
    assert type(pifs[0]).__name__ == "ProcessControlFunction"


def test_instrument_class_default_stays_pif():
    # No instrument_class → PIF default preserved (v0 behaviour).
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node("i1", "instrument", "FI-101", {"instrument_function": "indicator"}),
        ],
    )
    result = build_dexpi(g)
    pifs = result.model.conceptual_model.ProcessInstrumentationFunctions
    assert type(pifs[0]).__name__ == "ProcessInstrumentationFunction"


def test_unclassified_equipment_with_structural_description_preserves_attrs():
    """v2 Phase C: equipment_class and structural_description are preserved on
    the promoted ProcessEquipment via the agent_* customAttributes."""
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "n1", "equipment", "M-1",
                {
                    "equipment_class": "unclassified_equipment",
                    "structural_description": "tall vertical vessel with packed bed, 3 side nozzles",
                },
                conf="low",
            ),
        ],
    )
    result = build_dexpi(g)
    items = result.model.conceptual_model.TaggedPlantItems
    assert len(items) == 1
    assert type(items[0]).__name__ == "ProcessEquipment"
    by_name = {ca.name: ca.value for ca in items[0].customAttributes}
    assert by_name["agent_equipment_class"] == "unclassified_equipment"
    assert "tall vertical vessel" in by_name["agent_structural_description"]
    # No "review recommended" issue should fire when description is present.
    assert not any("review recommended" in msg for msg in result.issues)


def test_unclassified_equipment_without_structural_description_logs_review_issue():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "n1", "equipment", "M-1",
                {"equipment_class": "unclassified_equipment"},
                conf="low",
            ),
        ],
    )
    result = build_dexpi(g)
    items = result.model.conceptual_model.TaggedPlantItems
    assert type(items[0]).__name__ == "ProcessEquipment"
    by_name = {ca.name: ca.value for ca in items[0].customAttributes}
    assert by_name["agent_equipment_class"] == "unclassified_equipment"
    assert any(
        "without structural_description" in msg and "review recommended" in msg
        for msg in result.issues
    ), result.issues


def test_registry_entry_without_pydexpi_class_records_key_on_attribute():
    """v2 Phase C: when the registry has no concrete class (e.g. Reactor), the
    LLM key lands on customAttributes['agent_equipment_class']."""
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[_node("n1", "equipment", "R-101", {"equipment_class": "reactor"})],
    )
    result = build_dexpi(g)
    items = result.model.conceptual_model.TaggedPlantItems
    captured = [
        ca.value for ca in items[0].customAttributes if ca.name == "agent_equipment_class"
    ]
    assert captured == ["reactor"]


def test_text_node_populates_metadata_fields():
    """A 'text' title-block annotation fills DEXPI MetaData str fields.

    Recognised keys (drawing_number, project, project_no, sheet, title) map to
    their MetaData equivalents; unknown keys land in meta_data.customAttributes
    prefixed with ``agent_text_...``. No 'not mapped' issue is logged.
    """
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "n1", "text", "Sheet title block",
                {
                    "drawing_number": "51Y613",
                    "title": "INSTRUMENTATION GRIT WASHER NO. 1 P&ID",
                    "project": "San Mateo WWTP",
                    "project_no": "46T003",
                    "sheet": "107 of 229",
                    "engineer": "Brown and Caldwell",
                },
            ),
        ],
    )
    result = build_dexpi(g)
    meta = result.model.conceptual_model.meta_data
    assert meta is not None
    assert meta.DrawingNumber == "51Y613"
    assert meta.ProjectName == "San Mateo WWTP"
    assert meta.ProjectNumber == "46T003"
    assert meta.SheetNumber == "107 of 229"
    # Title goes into the MultiLanguageString DrawingSubTitle.
    assert meta.DrawingSubTitle is not None
    assert meta.DrawingSubTitle.SingleLanguageStrings[0].Value == (
        "INSTRUMENTATION GRIT WASHER NO. 1 P&ID"
    )
    # Unmapped keys land as prefixed customAttributes (plus the label).
    names = {ca.name for ca in meta.customAttributes}
    assert any(n.endswith("_engineer") for n in names)
    assert any(n.endswith("_label") for n in names)
    # Text kind must no longer produce the "not mapped to DEXPI" issue.
    assert not any("not mapped to DEXPI" in msg for msg in result.issues)
    assert result.stats.get("text_count") == 1


def test_note_node_lands_in_metadata_custom_attrs():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "n1", "note", "LCP-51-371 instrument bubbles",
                {"bubbles": "YL-51-371, HS-51-371", "notes": "Panel-mounted pilot devices."},
            ),
        ],
    )
    result = build_dexpi(g)
    meta = result.model.conceptual_model.meta_data
    assert meta is not None
    names = [ca.name for ca in meta.customAttributes]
    assert any(n.startswith("agent_note_") and n.endswith("_bubbles") for n in names)
    assert any(n.startswith("agent_note_") and n.endswith("_notes") for n in names)
    assert any(n.startswith("agent_note_") and n.endswith("_label") for n in names)
    assert not any("not mapped to DEXPI" in msg for msg in result.issues)
    assert result.stats.get("note_count") == 1


def test_opc_evidence_attributes_survive_dexpi_build():
    node = _node(
        "opc-1",
        "opc",
        "Compressed Air inlet",
        {
            "service": "compressed_air",
            "direction": "in",
            "source_equipment": "2401-V-002",
            "drawing_ref": "DW02-0003",
            "attribute_evidence": {"source_equipment": "printed_text"},
        },
    )
    node.source_quote = "压缩空气自2401-V-002"
    result = build_dexpi(ReconciledGraph(source_path="t.pdf", nodes=[node]))
    segment_items = [
        item
        for system in result.model.conceptual_model.PipingNetworkSystems
        for segment in system.Segments
        for item in segment.Items
    ]
    opc = next(item for item in segment_items if type(item).__name__ == "FlowInPipeOffPageConnector")
    attrs = {attr.name: attr.value for attr in opc.customAttributes}
    assert attrs["agent_service"] == "compressed_air"
    assert attrs["agent_source_equipment"] == "2401-V-002"
    assert attrs["agent_drawing_ref"] == "DW02-0003"
    assert attrs["agent_attribute_evidence"] == '{"source_equipment":"printed_text"}'
    assert attrs["agent_raw_text"] == "压缩空气自2401-V-002"


def test_edges_with_line_id_group_into_distinct_piping_network_systems():
    """Edges carrying distinct ``line_id`` attrs produce one PipingNetworkSystem per id."""
    nodes = [
        _node("a", "equipment", "T-1", {"equipment_class": "tank"}),
        _node("b", "equipment", "T-2", {"equipment_class": "tank"}),
        _node("c", "equipment", "T-3", {"equipment_class": "tank"}),
    ]
    e1 = ReconciledEdge(
        id="e1", from_node="a", to_node="b", line_type="process",
        polyline_global=[(0, 0), (10, 0)], confidence="high",
        attributes={"line_id": "4\"-GRT", "nominal_diameter": "4\"", "service_code": "GRT"},
    )
    e2 = ReconciledEdge(
        id="e2", from_node="b", to_node="c", line_type="process",
        polyline_global=[(0, 0), (10, 0)], confidence="high",
        attributes={"line_id": "4\"-GRT"},
    )
    e3 = ReconciledEdge(
        id="e3", from_node="a", to_node="c", line_type="process",
        polyline_global=[(0, 0), (10, 0)], confidence="high",
        attributes={"line_id": "8\"-GE", "service_code": "GE"},
    )
    g = ReconciledGraph(source_path="t.pdf", nodes=nodes, edges=[e1, e2, e3])
    result = build_dexpi(g)
    systems = result.model.conceptual_model.PipingNetworkSystems
    line_numbers = sorted(s.LineNumber for s in systems)
    assert line_numbers == ["4\"-GRT", "8\"-GE"]
    # Two segments in "4\"-GRT", one in "8\"-GE".
    by_ln = {s.LineNumber: s for s in systems}
    assert len(by_ln["4\"-GRT"].Segments) == 2
    assert len(by_ln["8\"-GE"].Segments) == 1
    # Line-level metadata (diameter/service) attached from the first edge.
    grt_attrs = {ca.name: ca.value for ca in by_ln["4\"-GRT"].customAttributes}
    assert grt_attrs.get("agent_nominal_diameter") == "4\""
    assert grt_attrs.get("agent_service_code") == "GRT"
    # No L-000 fallback issue since every edge had a line_id.
    assert not any(
        "placed in PipingNetworkSystem 'L-000'" in msg for msg in result.issues
    )
    assert result.stats["piping_network_system_count"] == 2


def test_mixed_edges_only_emit_default_bucket_message_when_needed():
    """Only edges without a line_id fall into L-000; the issue reports the count."""
    nodes = [
        _node("a", "equipment", "T-1", {"equipment_class": "tank"}),
        _node("b", "equipment", "T-2", {"equipment_class": "tank"}),
        _node("c", "equipment", "T-3", {"equipment_class": "tank"}),
    ]
    tagged = ReconciledEdge(
        id="e1", from_node="a", to_node="b", line_type="process",
        polyline_global=[(0, 0), (10, 0)], confidence="high",
        attributes={"line_id": "6\"-PW"},
    )
    untagged = ReconciledEdge(
        id="e2", from_node="b", to_node="c", line_type="process",
        polyline_global=[(0, 0), (10, 0)], confidence="high",
        attributes={},
    )
    g = ReconciledGraph(source_path="t.pdf", nodes=nodes, edges=[tagged, untagged])
    result = build_dexpi(g)
    line_numbers = sorted(
        s.LineNumber for s in result.model.conceptual_model.PipingNetworkSystems
    )
    assert line_numbers == ["6\"-PW", "L-000"]
    # Exactly one segment ended up in L-000 and the issue reflects that.
    assert any(
        "1 process segment(s) without line_id" in msg for msg in result.issues
    ), result.issues


def test_instrument_class_unknown_value_logs_issue_and_falls_back():
    g = ReconciledGraph(
        source_path="t.pdf",
        nodes=[
            _node(
                "i1", "instrument", "FIC-101",
                {"instrument_function": "controller", "instrument_class": "bogus"},
            ),
        ],
    )
    result = build_dexpi(g)
    pifs = result.model.conceptual_model.ProcessInstrumentationFunctions
    assert type(pifs[0]).__name__ == "ProcessInstrumentationFunction"
    assert any("unknown instrument_class" in msg for msg in result.issues), result.issues


def test_export_distinguishes_unknown_direction_from_proven_flow():
    from diagex.dexpi._generated.enums import PipingNetworkSegmentFlowClassification

    for direction in ["unknown", "forward"]:
        edge = _edge("route", "left", "right")
        edge.attributes["flow_direction"] = direction
        graph = ReconciledGraph(source_path="vector.pdf", nodes=[
            _node("left","equipment","A",{"equipment_class":"vessel"}),
            _node("right","equipment","B",{"equipment_class":"vessel"}),
        ], edges=[edge])
        result = build_dexpi(graph)
        segment = result.model.conceptual_model.PipingNetworkSystems[0].Segments[0]
        assert segment.FlowDirection == (
            PipingNetworkSegmentFlowClassification.SingleFlowPipingNetworkSegment
            if direction == "forward" else None
        )
        assert any(attr.name == "agent_flow_direction" and attr.value == direction for attr in segment.customAttributes)
