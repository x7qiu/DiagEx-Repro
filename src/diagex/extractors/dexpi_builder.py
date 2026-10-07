"""DEXPI builder: ``ReconciledGraph`` → diagex DEXPI 2.0 model (spec §§7.2, 7.2.1).

Pure-code conversion layer: no LLM calls, no network, no side effects beyond
the optional disk write in ``serialize_model``. Callers feed a
``ReconciledGraph`` from ``diagex.vision.models`` and receive a diagex-native
``EngineeringModel`` (with a ``PlantModel`` as its ``conceptual_model``) that
can be serialised by :mod:`diagex.dexpi.json_io`.

Migration note (DEXPI 2.0): the equipment / piping / instrumentation
collections that pyDEXPI placed on ``ConceptualModel`` now live on the
``PlantModel`` subclass instead. Field names are PascalCase
(``TaggedPlantItems``, ``PipingNetworkSystems``, ``Segments``, ``Items``, …)
matching the spec.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diagex.dexpi import json_io
from diagex.dexpi._generated.enums import (
    PipingNetworkSegmentFlowClassification,
    SignalConveyingTypeClassification,
)
from diagex.dexpi.model import customization as cu
from diagex.dexpi.model import dexpiModel as dm
from diagex.dexpi.model import equipment as eq
from diagex.dexpi.model import instrumentation as inst
from diagex.dexpi.model import piping as pp
from diagex.dexpi.model import pydantic_classes as pc
from diagex.dexpi_schema import (
    equipment_spec_for,
    instrument_class_spec_for,
    nearest_equipment_key,
    nearest_valve_key,
    valve_spec_for,
)
from diagex.vision.models import ReconciledEdge, ReconciledGraph, ReconciledNode

_MLS_LANGUAGE = "en-US"

# diagex-emitted EngineeringModel header fields. ExportDateTime is required
# but allowed to be None per the DEXPI 2.0 spec; we leave it None so callers
# can stamp it later if they want a deterministic export time.
_DIAGEX_ORIGIN = {
    "ExportDateTime": None,
    "OriginatingSystemName": "diagex",
    "OriginatingSystemVendorName": "ABB",
    "OriginatingSystemVersion": "0.1.0",
}


def _mls(text: str | None) -> pc.MultiLanguageString | None:
    """Wrap a plain string in a DEXPI ``MultiLanguageString``, or None if empty."""
    if not text:
        return None
    return pc.MultiLanguageString(
        SingleLanguageStrings=[pc.SingleLanguageString(Language=_MLS_LANGUAGE, Value=text)]
    )


_DEFAULT_LINE_NUMBER = "L-000"
_SIGNAL_LINE_TYPES = {
    "signal_electric",
    "signal_pneumatic",
    "instrument_capillary",
    "electrical_power",
}

# Attribute keys on 'text' annotations that map 1:1 to DEXPI ``MetaData`` str
# fields. The MLS field ``DrawingSubTitle`` is handled separately because it
# needs wrapping.
_METADATA_STR_FIELDS: dict[str, str] = {
    "drawing_number": "DrawingNumber",
    "drawingnumber": "DrawingNumber",
    "drawing_name": "DrawingName",
    "drawingname": "DrawingName",
    "project": "ProjectName",
    "project_name": "ProjectName",
    "projectname": "ProjectName",
    "project_no": "ProjectNumber",
    "project_number": "ProjectNumber",
    "projectnumber": "ProjectNumber",
    "sheet": "SheetNumber",
    "sheet_number": "SheetNumber",
    "sheetnumber": "SheetNumber",
    "sheet_format": "SheetFormat",
    "sheetformat": "SheetFormat",
    "revision": "RevisionNumber",
    "revision_number": "RevisionNumber",
    "file_name": "FileName",
    "filename": "FileName",
}


@dataclass
class BuildResult:
    """Output of ``build_dexpi``: model + diagnostic notes + counts."""

    model: dm.EngineeringModel
    issues: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Class-selection tables (spec §7.2.1) — registry-driven (diagex.dexpi_schema)
# ---------------------------------------------------------------------------


def _equipment_class_for(node: ReconciledNode, issues: list[str]) -> type:
    """Pick a DEXPI 2.0 equipment class from ``node.attributes['equipment_class']``."""
    raw = str(node.attributes.get("equipment_class", "")).strip().lower()
    spec = equipment_spec_for(raw)
    if spec is None:
        if raw:
            suggestion = nearest_equipment_key(raw)
            hint = f" (did you mean {suggestion!r}?)" if suggestion else ""
            issues.append(
                f"node {node.id!r}: unknown equipment_class {raw!r}{hint} "
                f"-> CustomEquipment"
            )
        return eq.CustomEquipment
    cls, _is_base = spec.resolve(node.attributes)
    if cls is None:
        return eq.CustomEquipment
    return cls


def _valve_class_for(node: ReconciledNode, issues: list[str]) -> tuple[type, bool]:
    """Pick a DEXPI 2.0 valve class. Second tuple item flags 'control intent'."""
    raw = str(node.attributes.get("valve_type", "")).strip().lower()
    spec = valve_spec_for(raw)
    if spec is None:
        if raw:
            suggestion = nearest_valve_key(raw)
            hint = f" (did you mean {suggestion!r}?)" if suggestion else ""
            issues.append(
                f"node {node.id!r}: unknown valve_type {raw!r}{hint} "
                f"-> CustomOperatedValve"
            )
        return pp.CustomOperatedValve, False
    if spec.pydexpi_class is None:
        return pp.CustomOperatedValve, spec.control_intent
    return spec.pydexpi_class, spec.control_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TAG_SUFFIX_RE = re.compile(r"(\d+)\s*$")


def _normalise_label(label: str | None) -> str:
    if not label:
        return ""
    return " ".join(label.split())


def _loop_number_from(node: ReconciledNode) -> str | None:
    explicit = node.attributes.get("loop_number") or node.attributes.get("tag_number")
    if explicit:
        return str(explicit).strip()
    match = _TAG_SUFFIX_RE.search(_normalise_label(node.label))
    if match:
        return match.group(1)
    return None


def _letter_prefix(label: str) -> str | None:
    cleaned = _normalise_label(label)
    m = re.match(r"^([A-Za-z]+)", cleaned)
    return m.group(1) if m else None


def _custom_string_attr(name: str, value: Any) -> cu.CustomStringAttribute | None:
    """Build a ``CustomStringAttribute``, returning None if the value is empty."""
    if value is None:
        return None
    text = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if isinstance(value, (dict, list, tuple))
        else str(value).strip()
    )
    if not text:
        return None
    try:
        return cu.CustomStringAttribute(name=name, value=text)
    except Exception:
        return None


def _attach_custom_attrs(target: Any, attrs: dict[str, Any], issues: list[str]) -> None:
    bag = getattr(target, "customAttributes", None)
    if bag is None:
        return
    for key in sorted(attrs.keys()):
        ca = _custom_string_attr(f"agent_{key}", attrs[key])
        if ca is not None:
            bag.append(ca)
        else:
            issues.append(f"skipped custom attr {key!r} on {type(target).__name__}")


def _attach_custom_attrs_prefixed(
    target: Any, attrs: dict[str, Any], prefix: str, issues: list[str]
) -> None:
    bag = getattr(target, "customAttributes", None)
    if bag is None:
        return
    for key in sorted(attrs.keys()):
        ca = _custom_string_attr(f"{prefix}_{key}", attrs[key])
        if ca is not None:
            bag.append(ca)
        else:
            issues.append(f"skipped custom attr {key!r} on {type(target).__name__}")


def _ensure_metadata(plant_model: pp.PlantModel) -> pc.MetaData:
    """Lazily initialise ``plant_model.meta_data`` and return it."""
    if plant_model.meta_data is None:
        plant_model.meta_data = pc.MetaData()
    return plant_model.meta_data


def _apply_text_node_to_metadata(
    node: ReconciledNode, plant_model: pp.PlantModel, issues: list[str]
) -> None:
    """Route a 'text' annotation (title-block, sheet labels) into MetaData."""
    meta = _ensure_metadata(plant_model)
    slug = _slug_for_node(node)
    prefix = f"agent_text_{slug}"
    remaining: dict[str, Any] = {}
    for key, raw_val in node.attributes.items():
        if raw_val is None:
            continue
        value = str(raw_val).strip()
        if not value:
            continue
        field_name = _METADATA_STR_FIELDS.get(key.lower())
        if field_name is not None:
            if not getattr(meta, field_name, None):
                setattr(meta, field_name, value)
            continue
        if key.lower() == "title":
            if getattr(meta, "DrawingSubTitle", None) is None:
                meta.DrawingSubTitle = _mls(value)
            continue
        remaining[key] = value
    _attach_custom_attrs_prefixed(meta, remaining, prefix, issues)
    label = _normalise_label(node.label)
    if label:
        ca = _custom_string_attr(f"{prefix}_label", label)
        if ca is not None:
            meta.customAttributes.append(ca)


def _apply_note_node_to_metadata(
    node: ReconciledNode, plant_model: pp.PlantModel, issues: list[str]
) -> None:
    """Route a free-form 'note' annotation into ``meta_data.customAttributes``."""
    meta = _ensure_metadata(plant_model)
    slug = _slug_for_node(node)
    prefix = f"agent_note_{slug}"
    _attach_custom_attrs_prefixed(meta, node.attributes, prefix, issues)
    label = _normalise_label(node.label)
    if label:
        ca = _custom_string_attr(f"{prefix}_label", label)
        if ca is not None:
            meta.customAttributes.append(ca)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug_for_node(node: ReconciledNode) -> str:
    raw = (node.id or node.label or "").lower()
    slug = _SLUG_RE.sub("_", raw).strip("_")
    return slug or "node"


# ---------------------------------------------------------------------------
# Node conversion
# ---------------------------------------------------------------------------


def _build_equipment(node: ReconciledNode, issues: list[str]) -> Any:
    """Instantiate a DEXPI 2.0 equipment or valve object for an 'equipment' node."""
    valve_type = node.attributes.get("valve_type")
    if valve_type is not None:
        cls, is_control = _valve_class_for(node, issues)
        # v2 Phase C: promote CustomOperatedValve to OperatedValve (the concrete
        # DEXPI ancestor). The LLM-supplied ``valve_type`` is preserved by the
        # downstream ``_attach_custom_attrs`` pass as ``agent_valve_type``.
        if cls is pp.CustomOperatedValve:
            cls = pp.OperatedValve
        obj = cls()
        name_attr = _custom_string_attr("agent_tag", _normalise_label(node.label))
        if name_attr is not None and hasattr(obj, "customAttributes"):
            obj.customAttributes.append(name_attr)
        if is_control:
            ca = _custom_string_attr("agent_control_intent", "control")
            if ca is not None and hasattr(obj, "customAttributes"):
                obj.customAttributes.append(ca)
        _attach_custom_attrs(obj, node.attributes, issues)
        return obj

    cls = _equipment_class_for(node, issues)
    label = _normalise_label(node.label) or None
    kwargs: dict[str, Any] = {"TagName": label}
    desc = _mls(label)
    if desc is not None:
        kwargs["EquipmentDescription"] = desc
    # v2 Phase C: promote CustomEquipment to ProcessEquipment. The LLM-supplied
    # equipment_class (and any structural_description) is preserved through
    # ``_attach_custom_attrs`` below — no extra capture step is needed.
    if cls is eq.CustomEquipment:
        # _custom_type_name_for is still called for its side effect of logging
        # a "review recommended" issue when an unclassified node has no
        # structural description. The returned string is no longer needed.
        _custom_type_name_for(node, issues)
        cls = eq.ProcessEquipment
    obj = cls(**kwargs)
    _attach_custom_attrs(obj, node.attributes, issues)
    return obj


def _custom_type_name_for(node: ReconciledNode, issues: list[str]) -> str:
    """Compose ``CustomEquipment.typeName`` for unclassified nodes."""
    raw_class = str(node.attributes.get("equipment_class", "")).strip().lower()
    desc = str(node.attributes.get("structural_description", "")).strip()

    if raw_class == "unclassified_equipment":
        if desc:
            return f"Unclassified: {desc}"
        issues.append(
            f"node {node.id!r}: unclassified_equipment without structural_description "
            f"-> CustomEquipment (review recommended)"
        )
        return "UnclassifiedEquipment"

    if raw_class:
        return raw_class

    issues.append(f"node {node.id!r}: unclassified equipment -> CustomEquipment")
    return "UnclassifiedEquipment"


def _build_instrument(node: ReconciledNode, issues: list[str]) -> inst.ProcessInstrumentationFunction:
    """Wrap an instrument node as a DEXPI 2.0 instrumentation function."""
    number = _loop_number_from(node)
    outer_cls = _instrument_outer_class_for(node, issues)
    obj = outer_cls(ProcessInstrumentationFunctionNumber=number)
    letter = _letter_prefix(node.label)
    if letter:
        ca = _custom_string_attr("agent_tag_prefix", letter)
        if ca is not None:
            obj.customAttributes.append(ca)
    tag = _normalise_label(node.label)
    if tag:
        ca = _custom_string_attr("agent_tag", tag)
        if ca is not None:
            obj.customAttributes.append(ca)
    _attach_custom_attrs(obj, node.attributes, issues)
    return obj


def _instrument_outer_class_for(node: ReconciledNode, issues: list[str]) -> type:
    """Pick the outer instrument class from ``node.attributes['instrument_class']``."""
    raw = str(node.attributes.get("instrument_class", "")).strip().lower()
    if not raw:
        return inst.ProcessInstrumentationFunction
    spec = instrument_class_spec_for(raw)
    if spec is None or spec.pydexpi_class is None:
        issues.append(
            f"node {node.id!r}: unknown instrument_class {raw!r} "
            f"-> ProcessInstrumentationFunction"
        )
        return inst.ProcessInstrumentationFunction
    return spec.pydexpi_class


def _build_opc(node: ReconciledNode, issues: list[str]) -> Any:
    """Emit a FlowInPipe/FlowOutPipe off-page connector."""
    direction = str(node.attributes.get("direction", "in")).strip().lower()
    cls = pp.FlowOutPipeOffPageConnector if direction == "out" else pp.FlowInPipeOffPageConnector
    obj = cls()
    _attach_custom_attrs(obj, node.attributes, issues)
    if node.source_quote and "raw_text" not in node.attributes:
        ca = _custom_string_attr("agent_raw_text", node.source_quote)
        if ca is not None:
            obj.customAttributes.append(ca)
    ca = _custom_string_attr("agent_opc_label", node.label)
    if ca is not None:
        obj.customAttributes.append(ca)
    return obj


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_dexpi(graph: ReconciledGraph) -> BuildResult:
    """Convert a ``ReconciledGraph`` into a diagex ``EngineeringModel``.

    Deterministic: input order drives output order; entity ``id`` fields are
    auto-generated UUIDs but the ordering of children within parents is stable.
    See spec §7.2 for the pipeline position of this step.
    """
    issues: list[str] = []
    stats = {
        "equipment_count": 0,
        "valve_count": 0,
        "instrument_count": 0,
        "segment_count": 0,
        "signal_count": 0,
        "unclassified_count": 0,
        "dropped_edges": 0,
        "opc_count": 0,
    }

    plant = dm.PlantModel()
    model = dm.EngineeringModel(conceptual_model=plant, **_DIAGEX_ORIGIN)

    # Node pass --------------------------------------------------------------
    dexpi_for_node: dict[str, Any] = {}
    valve_nodes: set[str] = set()
    opc_objects: list[Any] = []

    for node in graph.nodes:
        try:
            if node.kind == "equipment":
                obj = _build_equipment(node, issues)
                dexpi_for_node[node.id] = obj
                if isinstance(obj, pp.PipingComponent):
                    valve_nodes.add(node.id)
                    stats["valve_count"] += 1
                else:
                    plant.TaggedPlantItems.append(obj)
                    stats["equipment_count"] += 1
                    # v2 Phase C: a fallback to ProcessEquipment-as-base means the
                    # builder couldn't pick a concrete DEXPI subclass. We detect
                    # that by exact class identity (a real concrete class would
                    # be a subclass of ProcessEquipment, not the bare base).
                    if type(obj) is eq.ProcessEquipment:
                        stats["unclassified_count"] += 1
            elif node.kind == "instrument":
                obj = _build_instrument(node, issues)
                plant.ProcessInstrumentationFunctions.append(obj)
                dexpi_for_node[node.id] = obj
                stats["instrument_count"] += 1
            elif node.kind == "opc":
                obj = _build_opc(node, issues)
                dexpi_for_node[node.id] = obj
                opc_objects.append(obj)
                stats["opc_count"] += 1
            elif node.kind == "text":
                _apply_text_node_to_metadata(node, plant, issues)
                stats["text_count"] = stats.get("text_count", 0) + 1
            elif node.kind == "note":
                _apply_note_node_to_metadata(node, plant, issues)
                stats["note_count"] = stats.get("note_count", 0) + 1
            else:
                issues.append(f"node {node.id!r}: kind {node.kind!r} not mapped to DEXPI; skipped")
        except Exception as exc:  # defensive: never let one bad node kill the run
            issues.append(f"node {node.id!r}: conversion error {exc!r}; skipped")

    # Edge pass --------------------------------------------------------------
    systems_by_line: dict[str, pp.PipingNetworkSystem] = {}
    owned_valve_nodes: set[str] = set()
    default_segment_count = 0
    cross_sheet_count = 0

    edges_sorted = sorted(graph.edges, key=lambda e: e.id)

    for edge in edges_sorted:
        if edge.attributes.get("provisional_review_only"):
            issues.append(
                f"edge {edge.id!r}: provisional review-only relationship retained in "
                "graph.json but omitted from DEXPI"
            )
            continue
        if not edge.from_node or not edge.to_node:
            stats["dropped_edges"] += 1
            issues.append(f"edge {edge.id!r}: missing endpoint; dropped")
            continue
        if edge.from_node not in dexpi_for_node or edge.to_node not in dexpi_for_node:
            stats["dropped_edges"] += 1
            issues.append(
                f"edge {edge.id!r}: endpoint not found in node map; dropped"
            )
            continue
        if edge.cross_sheet:
            cross_sheet_count += 1
            _record_cross_sheet_edge(plant, edge, issues)
            continue

        line_type = edge.line_type
        if line_type in _SIGNAL_LINE_TYPES:
            _attach_signal_edge(edge, dexpi_for_node, issues, stats, plant)
            continue

        line_number = _line_number_for_edge(edge)
        network = systems_by_line.get(line_number)
        if network is None:
            network = pp.PipingNetworkSystem(LineNumber=line_number)
            _apply_line_metadata(network, edge)
            systems_by_line[line_number] = network
        segment = _build_piping_segment(
            edge,
            dexpi_for_node,
            valve_nodes,
            owned_valve_nodes=owned_valve_nodes,
        )
        network.Segments.append(segment)
        stats["segment_count"] += 1
        if line_number == _DEFAULT_LINE_NUMBER:
            default_segment_count += 1

    # A valve referenced only by an instrument signal is not encountered by a
    # process segment, but OperatedValveReference is still a reference rather
    # than ownership.  Park such valves in one fallback segment so their XML ID
    # is present and resolvable without inventing a process edge.
    unowned_valves = sorted(valve_nodes - owned_valve_nodes)
    if unowned_valves:
        host_system = systems_by_line.get(_DEFAULT_LINE_NUMBER)
        if host_system is None:
            host_system = pp.PipingNetworkSystem(LineNumber=_DEFAULT_LINE_NUMBER)
            systems_by_line[_DEFAULT_LINE_NUMBER] = host_system
        if not host_system.Segments:
            host_system.Segments.append(pp.PipingNetworkSegment())
            stats["segment_count"] += 1
        host_segment = host_system.Segments[0]
        for node_id in unowned_valves:
            host_segment.Items.append(dexpi_for_node[node_id])
            owned_valve_nodes.add(node_id)
        issues.append(
            f"{len(unowned_valves)} signal-only or unconnected operated valve(s) "
            f"placed in PipingNetworkSystem {_DEFAULT_LINE_NUMBER!r} for DEXPI ownership"
        )

    # Park OPC objects: DEXPI 2.0 has no top-level slot for free-standing
    # off-page connectors, so we dock them on the fallback L-000 system's
    # first segment (creating one if none exist).
    if opc_objects:
        host_system = systems_by_line.get(_DEFAULT_LINE_NUMBER)
        if host_system is None:
            host_system = pp.PipingNetworkSystem(LineNumber=_DEFAULT_LINE_NUMBER)
            systems_by_line[_DEFAULT_LINE_NUMBER] = host_system
        if not host_system.Segments:
            host_system.Segments.append(pp.PipingNetworkSegment())
            stats["segment_count"] += 1
        host_segment = host_system.Segments[0]
        for opc in opc_objects:
            host_segment.Items.append(opc)

    for line_number in sorted(systems_by_line.keys()):
        plant.PipingNetworkSystems.append(systems_by_line[line_number])

    stats["piping_network_system_count"] = len(systems_by_line)

    if default_segment_count:
        issues.append(
            f"{default_segment_count} process segment(s) without line_id "
            f"placed in PipingNetworkSystem {_DEFAULT_LINE_NUMBER!r}"
        )

    if cross_sheet_count:
        issues.append(f"{cross_sheet_count} cross-sheet edge(s) recorded on metaData")

    return BuildResult(model=model, issues=issues, stats=stats)


def _line_number_for_edge(edge: ReconciledEdge) -> str:
    raw = str(edge.attributes.get("line_id", "") or "").strip()
    return raw or _DEFAULT_LINE_NUMBER


def _apply_line_metadata(network: pp.PipingNetworkSystem, edge: ReconciledEdge) -> None:
    bag = getattr(network, "customAttributes", None)
    if bag is None:
        return
    for attr_key in ("nominal_diameter", "service_code"):
        value = edge.attributes.get(attr_key)
        if not value:
            continue
        ca = _custom_string_attr(f"agent_{attr_key}", value)
        if ca is not None:
            bag.append(ca)


def _build_piping_segment(
    edge: ReconciledEdge,
    dexpi_for_node: dict[str, Any],
    valve_nodes: set[str],
    *,
    owned_valve_nodes: set[str] | None = None,
) -> pp.PipingNetworkSegment:
    """Create a PipingNetworkSegment for one reconciled process edge."""
    segment = pp.PipingNetworkSegment()
    ca = _custom_string_attr("agent_edge_id", edge.id)
    if ca is not None:
        segment.customAttributes.append(ca)
    if edge.line_type:
        lt = _custom_string_attr("agent_line_type", edge.line_type)
        if lt is not None:
            segment.customAttributes.append(lt)
    direction = edge.attributes.get("flow_direction")
    if direction == "forward":
        segment.FlowDirection = PipingNetworkSegmentFlowClassification.SingleFlowPipingNetworkSegment
    if direction is not None:
        attr = _custom_string_attr("agent_flow_direction", direction)
        if attr is not None:
            segment.customAttributes.append(attr)
    source = _piping_endpoint(dexpi_for_node[edge.from_node], as_source=True)
    target = _piping_endpoint(dexpi_for_node[edge.to_node], as_source=False)
    if source is not None:
        segment.SourceItem = source
    if target is not None:
        segment.TargetItem = target

    # Items is a composition (ownership), whereas SourceItem/TargetItem are
    # references. A valve connected to two graph edges must therefore be owned
    # by exactly one segment and referenced by every incident segment. Emitting
    # it under both segments creates duplicate XML IDs and an invalid DEXPI
    # document.
    owned = owned_valve_nodes if owned_valve_nodes is not None else set()
    for endpoint_id in (edge.from_node, edge.to_node):
        if endpoint_id in valve_nodes and endpoint_id not in owned:
            segment.Items.append(dexpi_for_node[endpoint_id])
            owned.add(endpoint_id)
    return segment


def _piping_endpoint(obj: Any, *, as_source: bool) -> Any | None:
    """Return a DEXPI-connectable endpoint, adding an equipment nozzle.

    Process equipment itself is not a ``PipingSourceItem``/``PipingTargetItem``
    in DEXPI; its nozzle is. Each graph incidence therefore receives one owned
    nozzle, which the network segment references. Inline piping components are
    already valid endpoint items and need no adapter.
    """
    endpoint_type = pp.PipingSourceItem if as_source else pp.PipingTargetItem
    if isinstance(obj, endpoint_type):
        return obj
    if isinstance(obj, eq.ProcessEquipment):
        nozzle = pp.Nozzle()
        obj.Nozzles.append(nozzle)
        return nozzle
    return None


def _attach_signal_edge(
    edge: ReconciledEdge,
    dexpi_for_node: dict[str, Any],
    issues: list[str],
    stats: dict[str, int],
    plant_model: pp.PlantModel,
) -> None:
    """Serialize only signal relationships with valid DEXPI endpoints."""
    source = dexpi_for_node.get(edge.from_node)
    target = dexpi_for_node.get(edge.to_node)
    host: inst.ProcessInstrumentationFunction | None
    signal_source: Any | None = None
    signal_target: Any | None = None

    if edge.line_type == "instrument_capillary":
        instrument = next(
            (
                value
                for value in (source, target)
                if isinstance(value, inst.ProcessInstrumentationFunction)
            ),
            None,
        )
        process_endpoint = target if instrument is source else source
        sensing_location = _sensing_location(process_endpoint)
        if instrument is not None and sensing_location is not None:
            generating = inst.ProcessSignalGeneratingFunction(
                ProcessSignalGeneratingFunctionNumber=(
                    instrument.ProcessInstrumentationFunctionNumber
                ),
                SensingLocation=sensing_location,
            )
            instrument.ProcessSignalGeneratingFunctions.append(generating)
            host = instrument
            signal_source = generating
            signal_target = instrument
        else:
            host = None
    elif isinstance(source, inst.ProcessInstrumentationFunction) and isinstance(
        target, inst.ProcessInstrumentationFunction
    ):
        host = source
        signal_source = source
        signal_target = target
    else:
        instrument = next(
            (
                value
                for value in (source, target)
                if isinstance(value, inst.ProcessInstrumentationFunction)
            ),
            None,
        )
        valve = next(
            (value for value in (source, target) if isinstance(value, pp.OperatedValve)),
            None,
        )
        if instrument is not None and valve is not None:
            valve_reference = inst.OperatedValveReference(Valve=valve)
            system = inst.ActuatingSystem(OperatedValveReference=valve_reference)
            actuating = inst.ActuatingFunction(Systems=system)
            plant_model.ActuatingSystems.append(system)
            instrument.ActuatingFunctions.append(actuating)
            host = instrument
            if source is instrument:
                signal_source = instrument
                signal_target = actuating
            else:
                signal_source = actuating
                signal_target = instrument
        else:
            host = None

    if host is None or signal_source is None or signal_target is None:
        stats["dropped_edges"] += 1
        issues.append(
            f"edge {edge.id!r}: unsupported DEXPI signal endpoint combination; "
            "retained in graph.json but omitted from DEXPI"
        )
        return
    scf_cls = (
        inst.MeasuringLineFunction
        if edge.line_type == "instrument_capillary"
        else inst.SignalLineFunction
    )
    signal_types = {
        "signal_electric": SignalConveyingTypeClassification.ElectricalSignalConveying,
        "signal_pneumatic": SignalConveyingTypeClassification.PneumaticSignalConveying,
        "instrument_capillary": SignalConveyingTypeClassification.CapillarySignalConveying,
        "electrical_power": SignalConveyingTypeClassification.ElectricalSignalConveying,
    }
    scf = scf_cls(
        Source=signal_source,
        Target=signal_target,
        SignalConveyingType=signal_types.get(edge.line_type or ""),
    )
    ca = _custom_string_attr("agent_edge_id", edge.id)
    if ca is not None:
        scf.customAttributes.append(ca)
    lt = _custom_string_attr("agent_line_type", edge.line_type or "")
    if lt is not None:
        scf.customAttributes.append(lt)
    host.SignalConveyingFunctions.append(scf)
    stats["signal_count"] = stats.get("signal_count", 0) + 1


def _sensing_location(obj: Any) -> inst.SensingLocation | None:
    """Return a DEXPI sensing location for a measured process endpoint."""
    if isinstance(obj, inst.SensingLocation):
        return obj
    if isinstance(obj, eq.ProcessEquipment):
        nozzle = pp.Nozzle()
        obj.Nozzles.append(nozzle)
        return nozzle
    return None


def _record_cross_sheet_edge(
    plant_model: pp.PlantModel,
    edge: ReconciledEdge,
    issues: list[str],
) -> None:
    """Record a cross-sheet edge as a customAttribute on plant_model.meta_data."""
    if plant_model.meta_data is None:
        plant_model.meta_data = pc.MetaData()
    meta = plant_model.meta_data
    ca = _custom_string_attr(
        f"cross_sheet_edge_{edge.id}",
        f"{edge.from_node}->{edge.to_node}",
    )
    if ca is not None:
        meta.customAttributes.append(ca)
    else:
        issues.append(f"edge {edge.id!r}: cross-sheet record skipped (empty attr)")


# ---------------------------------------------------------------------------
# Serialisation + validation
# ---------------------------------------------------------------------------


def serialize_model(
    model: dm.EngineeringModel,
    out_dir: Path,
    basename: str = "pid.dexpi",
) -> Path:
    """Write the model to ``out_dir/<basename>.json`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{basename}.json"
    json_io.dump(model, target)
    return target


def validate_model(model: dm.EngineeringModel) -> list[dict]:
    """Round-trip the model through ``json_io`` and collect any validation issues."""
    problems: list[dict] = []
    try:
        text = json_io.dumps(model)
    except Exception as exc:
        problems.append({"path": "<dumps>", "msg": repr(exc)})
        return problems
    try:
        json_io.loads(text)
    except Exception as exc:
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                for err in errors():
                    loc = err.get("loc", ())
                    problems.append(
                        {
                            "path": ".".join(str(p) for p in loc),
                            "msg": err.get("msg", ""),
                        }
                    )
                return problems
            except Exception:
                pass
        problems.append({"path": "<loads>", "msg": repr(exc)})
    return problems
