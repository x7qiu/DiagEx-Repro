"""Drawing-independent physical connectivity regressions; no model calls."""

from __future__ import annotations

import pytest

from diagex.vision.evidence import PageEvidence, PathEvidence
from diagex.vision.models import BBox, ReconciledNode
from diagex.vision.page_graph import (
    CandidateDecision,
    PageGraphSubmission,
    PageLineEvidence,
    VisualCandidateAssessment,
    validate_page_graph_submission,
)
from diagex.vision.topology import build_page_topology, route_evidence_failures
from diagex.vision.vector_geometry import symbol_contours


def path(name, points, primitive="line", **kwargs):
    xs, ys = zip(*points, strict=True)
    return PathEvidence(
        id=name,
        page_index=0,
        points=points,
        bbox=BBox(x=min(xs), y=min(ys), w=max(1, max(xs) - min(xs)), h=max(1, max(ys) - min(ys))),
        origin="pdf_vector",
        primitive=primitive,
        **kwargs,
    )


def page(paths, *, nodes=()):
    # Topology fixtures include actual native endpoint glyphs. A box by itself
    # deliberately no longer certifies connectivity; new tests cover that case.
    paths = list(paths)
    for n in nodes:
        if n.attributes.get("equipment_class"):
            continue
        b = n.bbox_global
        paths.append(
            path(
                "glyph-" + n.id,
                [(b.x, b.y), (b.x2, b.y), (b.x2, b.y2), (b.x, b.y2), (b.x, b.y)],
                "rect",
                closed=True,
            )
        )
    return PageEvidence(
        page_index=0,
        source_ref="synthetic-vector",
        width=1000,
        height=800,
        dpi=100,
        effective_dpi=100,
        is_scanned=False,
        paths=paths,
    )


def node(name, x, y=280, w=40, h=40, **attributes):
    return ReconciledNode(
        id=name,
        label=name,
        kind="equipment",
        page_index=0,
        bbox_global=BBox(x=x, y=y, w=w, h=h),
        confidence="high",
        attributes=attributes,
    )


def accepted(result):
    return [e for e in result.edges if not e.attributes.get("provisional_review_only")]


def pairs(result):
    return {frozenset((e.from_node, e.to_node)) for e in accepted(result)}


def solve(p, nodes, edge, *, decision=True, arrow="none"):
    refs = {f"N{i}": n for i, n in enumerate(nodes)}
    return validate_page_graph_submission(
        page=p,
        pages=[p],
        nodes_by_ref=refs,
        local_node_refs=refs,
        edges_by_ref={"E1": edge},
        submission=PageGraphSubmission(
            candidate_decisions=[
                CandidateDecision(
                    candidate_ref="E1", decision="keep", reverse=True, confidence="high"
                )
            ]
            if decision
            else []
        ),
        visual_evidence=PageLineEvidence(
            page_index=0,
            assessments={
                "E1": VisualCandidateAssessment(
                    candidate_ref="E1",
                    route_visible="yes",
                    endpoint_alignment="both",
                    observed_style="solid",
                    confidence="high",
                    arrow_direction=arrow,
                )
            },
        ),
    )


def test_curved_vessel_wall_is_owned_and_ports_do_not_bypass_equipment():
    nodes = [
        node("in", 80),
        node("tank", 400, 150, 200, 400, equipment_class="vessel"),
        node("out", 880, 480),
    ]
    paths = [
        path("left-wall", [(400, 200), (400, 500)]),
        path("right-wall", [(600, 200), (600, 500)]),
        path("top-head", [(400, 200), (400, 133), (600, 133), (600, 200)], "curve"),
        path("bottom-head", [(600, 500), (600, 567), (400, 567), (400, 500)], "curve"),
        path("inlet", [(100, 300), (400, 300)]),
        path("outlet", [(600, 500), (900, 500)]),
        # A native internal chord must not prevent exterior face ownership.
        path("internal", [(400, 200), (600, 200)]),
    ]
    p = page(paths, nodes=nodes)
    contours = symbol_contours(p, nodes)
    assert "tank" in contours
    result = build_page_topology(page=p, nodes=nodes)
    assert pairs(result) == {frozenset(("in", "tank")), frozenset(("tank", "out"))}
    assert not {"left-wall", "right-wall", "top-head", "bottom-head", "internal"} & set(
        result.used_path_ids
    )
    for edge in accepted(result):
        assert route_evidence_failures(edge, p) == []
        tank_port = next(
            port for port in edge.attributes["route_evidence"]["ports"] if port["node_id"] == "tank"
        )
        assert tank_port["basis"] == "native_contour"


def test_inline_valve_stops_a_continuous_cad_line_and_preserves_both_ports():
    nodes = [node("left", 80), node("valve", 480, valve_type="gate"), node("right", 880)]
    p = page([path("pipe", [(100, 300), (900, 300)])], nodes=nodes)
    result = build_page_topology(page=p, nodes=nodes)
    assert pairs(result) == {frozenset(("left", "valve")), frozenset(("valve", "right"))}
    ports = [
        port["point"]
        for e in result.edges
        for port in e.attributes["route_evidence"]["ports"]
        if port["node_id"] == "valve"
    ]
    assert sorted(ports) == [(480, 300), (520, 300)]


def test_nearby_parallel_pipe_is_not_snapped_to_symbol():
    nodes = [node("left", 80), node("nearby", 480, 312), node("right", 880)]
    result = build_page_topology(
        page=page([path("pipe", [(100, 300), (900, 300)])], nodes=nodes), nodes=nodes
    )
    assert pairs(result) == {frozenset(("left", "right"))}
    assert "nearby" in result.unattached_node_ids


def test_symbol_bbox_does_not_mask_external_pipe_outside_rounded_contour():
    # The pipe traverses a corner of the contour's bounding box, outside its body.
    vessel = node("round", 400, 200, 200, 200, equipment_class="vessel")
    diamond = path(
        "outline", [(500, 200), (600, 300), (500, 400), (400, 300), (500, 200)], closed=True
    )
    p = page([diamond, path("pipe", [(100, 205), (900, 205)])])
    result = build_page_topology(
        page=p, nodes=[node("left", 80, 185), vessel, node("right", 880, 185)]
    )
    # The line actually enters the small top of the diamond: valid ports stop it.
    assert frozenset(("left", "right")) not in pairs(result)
    p = page(
        [diamond, path("pipe", [(100, 220), (440, 220)])],
        nodes=[node("left", 80, 200), node("right", 440, 200)],
    )
    result = build_page_topology(
        page=p, nodes=[node("left", 80, 200), vessel, node("right", 440, 200)]
    )
    assert pairs(result) == {frozenset(("left", "right"))}


def test_repeated_labels_in_distinct_packages_remain_separate_networks():
    nodes = [node("a1", 80), node("a2", 480), node("b1", 80, 480), node("b2", 480, 480)]
    for n in nodes:
        n.label = "REPEATED"
    p = page(
        [path("upper", [(100, 300), (500, 300)]), path("lower", [(100, 500), (500, 500)])],
        nodes=nodes,
    )
    assert pairs(build_page_topology(page=p, nodes=nodes)) == {
        frozenset(("a1", "a2")),
        frozenset(("b1", "b2")),
    }


def test_tee_connects_but_solid_and_dashed_crossing_do_not_join():
    nodes = [node("left", 80), node("right", 880), node("top", 480, 80)]
    p = page(
        [path("main", [(100, 300), (900, 300)]), path("branch", [(500, 100), (500, 300)])],
        nodes=nodes,
    )
    result = build_page_topology(page=p, nodes=nodes)
    assert len(accepted(result)) == 2
    assert {n for e in accepted(result) for n in (e.from_node, e.to_node)} == {
        "left",
        "right",
        "top",
    }
    p.paths[1].dashes = "[10 5] 0"
    result = build_page_topology(page=p, nodes=nodes)
    assert pairs(result) == {frozenset(("left", "right"))}


@pytest.mark.parametrize(
    "failure", ["single_point", "missing_sources", "missing_ports", "ambiguous_port"]
)
def test_high_model_confidence_cannot_override_missing_structural_evidence(failure):
    nodes = [node("left", 80), node("right", 880)]
    p = page([path("pipe", [(100, 300), (900, 300)])], nodes=nodes)
    edge = build_page_topology(page=p, nodes=nodes).edges[0]
    if failure == "single_point":
        edge.polyline_global = [(500, 300)]
    elif failure == "missing_sources":
        edge.source_evidence_ids = []
    elif failure == "missing_ports":
        edge.attributes["route_evidence"]["ports"] = []
    else:
        edge.attributes["route_evidence"].update(
            status="uncertain", failures=["ambiguous_port_ownership"]
        )
    result = solve(p, nodes, edge)
    assert not result.accepted_candidate_ids
    assert result.provisional_candidate_ids == [edge.id]
    assert result.edges[0].system_confidence <= 0.35
    assert result.edges[0].attributes["system_confidence_evidence"]["score"] <= 0.35
    assert result.conflicts[0]["type"] == "unsupported_vector_route"


def arrow_paths(x=500, reverse=False):
    sign = -1 if reverse else 1
    a, b, c = (x + 12 * sign, 300), (x - 12 * sign, 294), (x - 12 * sign, 306)
    return [path(f"arrow-{x}-{i}", points) for i, points in enumerate([(a, b), (b, c), (c, a)])]


@pytest.mark.parametrize("reverse", [False, True])
def test_native_arrow_sets_flow_even_when_model_reverses_it(reverse):
    nodes = [node("left", 80), node("right", 880)]
    p = page([path("pipe", [(100, 300), (900, 300)]), *arrow_paths(reverse=reverse)], nodes=nodes)
    edge = build_page_topology(page=p, nodes=nodes).edges[0]
    assert (edge.from_node, edge.to_node) == (("right", "left") if reverse else ("left", "right"))
    assert edge.attributes["flow_direction"] == "forward"
    assert edge.attributes["direction_source"] == "native_vector_arrow"
    result = solve(p, nodes, edge, arrow="reverse")
    assert result.edges[0].from_node == edge.from_node
    assert result.accepted_candidate_ids == [edge.id]


def test_conflicting_native_arrows_are_reviewable_and_unmarked_flow_is_unknown():
    nodes = [node("left", 80), node("right", 880)]
    p = page(
        [path("pipe", [(100, 300), (900, 300)]), *arrow_paths(400), *arrow_paths(600, True)],
        nodes=nodes,
    )
    result = build_page_topology(page=p, nodes=nodes)
    assert not accepted(result)
    assert "conflicting_native_arrows" in result.edges[0].attributes["route_evidence"]["failures"]
    p.paths = p.paths[:1]
    edge = build_page_topology(page=p, nodes=nodes).edges[0]
    assert solve(p, nodes, edge).edges[0].attributes["flow_direction"] == "unknown"


def test_touching_boxes_do_not_establish_a_zero_length_connection():
    p = page([path("pipe", [(100, 300), (900, 300)])])
    result = build_page_topology(page=p, nodes=[node("a", 460), node("b", 500)])
    assert not accepted(result)
    assert all(e.attributes["provisional_review_only"] for e in result.edges)


def test_processing_is_deterministic_under_input_order_changes():
    nodes = [node("left", 80), node("right", 880), node("valve", 480)]
    p = page(
        [path("a", [(100, 300), (490, 300)]), path("b", [(510, 300), (900, 300)])], nodes=nodes
    )
    first = build_page_topology(page=p, nodes=nodes)
    p.paths.reverse()
    second = build_page_topology(page=p, nodes=list(reversed(nodes)))
    assert first.model_dump() == second.model_dump()


@pytest.mark.parametrize("marked", [False, True])
def test_presplit_four_way_crossing_requires_junction_marker(marked):
    nodes = [node("left", 80), node("right", 880), node("top", 480, 80), node("bottom", 480, 480)]
    paths = [
        path("a", [(100, 300), (500, 300)]),
        path("b", [(500, 300), (900, 300)]),
        path("c", [(500, 100), (500, 300)]),
        path("d", [(500, 300), (500, 500)]),
    ]
    if marked:
        paths.append(
            path(
                "dot",
                [(497, 297), (503, 297), (503, 303), (497, 303), (497, 297)],
                "rect",
                closed=True,
                fill_color=[0, 0, 0],
            )
        )
    result = build_page_topology(page=page(paths, nodes=nodes), nodes=nodes)
    if marked:
        assert len(accepted(result)) == 3
    else:
        assert pairs(result) == {frozenset(("left", "right")), frozenset(("top", "bottom"))}
        assert result.ambiguities


def test_native_flange_strokes_establish_a_port_outside_the_body_box():
    tank = node("tank", 400, 200, 200, 300, equipment_class="vessel")
    nodes = [node("left", 80), tank]
    body = path(
        "body", [(400, 200), (600, 200), (600, 500), (400, 500), (400, 200)], "rect", closed=True
    )
    p = page(
        [
            body,
            path("pipe", [(100, 300), (380, 300)]),
            path("flange", [(380, 290), (380, 310)]),
            path("upper-neck", [(380, 290), (400, 290)]),
            path("lower-neck", [(380, 310), (400, 310)]),
        ],
        nodes=nodes,
    )
    result = build_page_topology(page=p, nodes=nodes)
    assert pairs(result) == {frozenset(("left", "tank"))}
    port = next(
        port
        for port in result.edges[0].attributes["route_evidence"]["ports"]
        if port["node_id"] == "tank"
    )
    assert port["point"] == (380, 300)
    assert port["basis"] == "native_nozzle"
    assert "flange" in port["nozzle_path_ids"]
    # The same geometric gap without nozzle evidence is not a connection.
    p.paths = p.paths[:2]
    assert not accepted(build_page_topology(page=p, nodes=nodes))


def test_open_curved_pipe_is_retained_but_unowned_cyclic_curves_need_review():
    nodes = [node("left", 80), node("right", 880)]
    curves = [path("bend", [(300, 300), (400, 200), (600, 200), (700, 300)], "curve")]
    p = page(
        [
            path("inlet", [(100, 300), (300, 300)]),
            *curves,
            path("outlet", [(700, 300), (900, 300)]),
        ],
        nodes=nodes,
    )
    assert pairs(build_page_topology(page=p, nodes=nodes)) == {frozenset(("left", "right"))}
    p.paths.append(path("return", [(300, 300), (400, 400), (600, 400), (700, 300)], "curve"))
    result = build_page_topology(page=p, nodes=nodes)
    assert not accepted(result)
    assert (
        "curve_in_unassigned_cyclic_network"
        in result.edges[0].attributes["route_evidence"]["failures"]
    )


def test_topology_upgrade_preserves_native_evidence_and_perception(tmp_path):
    from diagex.extractors.evidence_checkpoint import (
        CheckpointStore,
        find_resumable_run_with_report,
    )
    from diagex.vision.instance_matching import FUSION_VERSION
    from diagex.vision.topology import TOPOLOGY_DEPENDENT_STAGES, TOPOLOGY_VERSION

    store = CheckpointStore.create(
        run_dir=tmp_path / "run", source_sha256="source", config_sha256="same", run_id="run"
    )
    for stage in ["inspection", "perception", "contextual", *TOPOLOGY_DEPENDENT_STAGES]:
        store.write_json_artifact(stage, "saved", {"saved": True})
    store.ensure_stage_version("object_fusion", FUSION_VERSION)
    store.set_status("complete")
    found, _ = find_resumable_run_with_report(
        runs_root=tmp_path,
        source_sha256="source",
        config_sha256="same",
        required_stage_versions={
            "object_fusion": FUSION_VERSION,
            "port_topology": TOPOLOGY_VERSION,
        },
    )
    assert found is not None
    found.ensure_stage_version(
        "port_topology", TOPOLOGY_VERSION, invalidate=TOPOLOGY_DEPENDENT_STAGES
    )
    assert all(
        found.is_done(stage, "saved") for stage in ["inspection", "perception", "contextual"]
    )
    assert not any(found.is_done(stage, "saved") for stage in TOPOLOGY_DEPENDENT_STAGES)
    assert not found.ensure_stage_version(
        "port_topology", TOPOLOGY_VERSION, invalidate=TOPOLOGY_DEPENDENT_STAGES
    )


@pytest.mark.parametrize("direction", ["unknown", "forward"])
def test_graph_exports_do_not_draw_arrows_for_unknown_flow(direction):
    from xml.etree import ElementTree as ET

    from diagex.extractors.dexpi_drawio import render_graph_to_drawio
    from diagex.extractors.dexpi_svg import render_graph_to_svg
    from diagex.vision.models import ReconciledGraph

    nodes = [node("left", 80), node("right", 880)]
    p = page([path("pipe", [(100, 300), (900, 300)])], nodes=nodes)
    edge = build_page_topology(page=p, nodes=nodes).edges[0]
    edge.attributes["flow_direction"] = direction
    graph = ReconciledGraph(source_path="vector.pdf", nodes=nodes, edges=[edge])
    svg = ET.fromstring(render_graph_to_svg(graph))
    paths = [el for el in svg.iter() if el.get("class") == "edge"]
    assert paths
    assert bool(paths[0].get("marker-end")) == (direction == "forward")
    drawio = ET.fromstring(render_graph_to_drawio(graph))
    cells = [el for el in drawio.iter("mxCell") if el.get("edge") == "1"]
    assert cells
    assert ("endArrow=none" in cells[0].get("style", "")) == (direction == "unknown")


def test_completed_topology_upgrade_can_reuse_compatible_contextual_results(tmp_path):
    from diagex.extractors.evidence_checkpoint import CheckpointStore

    old = CheckpointStore.create(
        run_dir=tmp_path / "old", source_sha256="source", config_sha256="same", run_id="old"
    )
    old.write_json_artifact("contextual", "page", {"observations": ["saved-model-result"]})
    new = CheckpointStore.create(
        run_dir=tmp_path / "new", source_sha256="source", config_sha256="same", run_id="new"
    )
    new.seed_raw_evidence_from(old, include_contextual=True)
    assert new.read_json_artifact("contextual", "page") == {"observations": ["saved-model-result"]}
    assert old.is_done("contextual", "page")


@pytest.mark.parametrize("closed_marker", [True, False])
def test_closed_native_opc_preserves_local_pipe_contact_without_assuming_flow(closed_marker):
    # Source-shaped diagnostic: five independently exported strokes form the
    # closed continuation marker on 2401 page 4, with a pipe ending on its back.
    # The upstream box here is synthetic; this is not a customer recall score.
    marker_paths = [
        path("vec-0c5d8c9a8d264370", [(4375, 2335), (4375, 2376)]),
        path("vec-46c37d31215169b4", [(4651, 2376), (4672, 2356)]),
        path("vec-4752b3ebdd245c16", [(4375, 2335), (4651, 2335)]),
        path("vec-9f11624ddaa3a340", [(4651, 2335), (4672, 2356)]),
        path("vec-caa81e76a48c53a0", [(4375, 2376), (4651, 2376)]),
    ]
    upstream = node("synthetic-upstream", 3640, 2336)
    opc = node("continuation", 4375, 2335, 297, 41)
    opc.kind = "opc"
    opc.attributes = {"candidate_shape": "continuation_arrow", "direction": "out"}
    # Missing the back wall leaves an open glyph; its convex hull must not be
    # promoted to a physical boundary merely because the detector called it OPC.
    p = page(
        [path("vec-567dca0d490c2655", [(3680, 2356), (4375, 2356)]),
         *(marker_paths if closed_marker else marker_paths[1:])],
        nodes=[upstream],
    )
    p.width, p.height = 6000, 4238
    nodes = [upstream, opc]
    result = build_page_topology(page=p, nodes=nodes)
    if closed_marker:
        assert pairs(result) == {frozenset((upstream.id, opc.id))}
        edge = accepted(result)[0]
        assert edge.attributes["flow_direction"] == "unknown"
        assert edge.cross_sheet is False
        assert route_evidence_failures(edge, p) == []
        port = next(p for p in edge.attributes["route_evidence"]["ports"] if p["node_id"] == opc.id)
        assert port["basis"] == "native_contour"
        assert port["point"] == (4375, 2356)
        assert set(port["outline_path_ids"]) == {p.id for p in marker_paths}
        assert not set(port["outline_path_ids"]) & set(result.used_path_ids)
    else:
        assert opc.id not in symbol_contours(p, nodes)
        assert not accepted(result)


def test_closed_opc_marker_does_not_snap_a_nearby_parallel_pipe():
    upstream, downstream = node("left", 80), node("right", 880)
    opc = node("continuation", 400, 320, 120, 40)
    opc.kind = "opc"
    p = page([
        path("pipe", [(100, 300), (900, 300)]),
        path("closed-marker", [(400, 320), (500, 320), (520, 340), (500, 360), (400, 360), (400, 320)], closed=True),
    ], nodes=[upstream, downstream])
    result = build_page_topology(page=p, nodes=[upstream, downstream, opc])
    assert pairs(result) == {frozenset((upstream.id, downstream.id))}
    assert opc.id in result.unattached_node_ids


@pytest.mark.parametrize("ambiguous", [False, True])
def test_opc_bbox_or_ambiguous_native_faces_cannot_certify_a_port(ambiguous):
    upstream = node("upstream", 80)
    opc = node("continuation", 400, 280, 200, 40)
    opc.kind = "opc"
    paths = [path("pipe", [(100, 300), (400, 300)])]
    if ambiguous:
        for name, x in [("left-face", 400), ("right-face", 510)]:
            paths.append(path(name, [(x, 280), (x + 90, 280), (x + 90, 320), (x, 320), (x, 280)], closed=True))
    p = page(paths, nodes=[upstream])
    nodes = [upstream, opc]
    assert opc.id not in symbol_contours(p, nodes)
    assert not accepted(build_page_topology(page=p, nodes=nodes))
