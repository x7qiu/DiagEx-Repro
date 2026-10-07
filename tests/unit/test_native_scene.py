"""Physical symbol/port contracts with independent synthetic vector drawings."""

from __future__ import annotations

from itertools import permutations

import pytest

from diagex.vision.fusion import fuse_objects
from diagex.vision.instance_matching import match_instances
from diagex.vision.topology import build_page_topology, route_evidence_failures
from diagex.vision.vector_geometry import native_symbols, symbol_contours
from tests.unit.test_instance_matching import detection
from tests.unit.test_port_topology import accepted, node, page, pairs, path


def body():
    # Deliberately fragmented, with head chords and long nozzle strokes.
    return [
        path("left-wall", [(400, 200), (400, 500)]),
        path("right-wall", [(600, 200), (600, 500)]),
        path("top", [(400, 200), (400, 133), (600, 133), (600, 200)], "curve"),
        path("bottom", [(600, 500), (600, 567), (400, 567), (400, 500)], "curve"),
        path("chord", [(400, 200), (600, 200)]),
    ]


def rectangle(name, x, y, w, h):
    return path(name, [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)], "rect", closed=True)


def test_clipped_nested_observations_share_full_native_body_without_absorbing_valve():
    p = page(body())
    ds = [
        detection("upper", (395, 150, 200, 110), "V101", "vessel"),
        detection("full", (395, 150, 210, 410), "V101", "vessel"),
        detection("valve", (490, 125, 20, 30)),
    ]
    for order in permutations(ds):
        clusters, _ = match_instances(list(order), {0: p})
        vessel = next(c for c in clusters if c.detections[0].label == "V101")
        assert {d.id for d in vessel.detections} == {"upper", "full"}
        assert vessel.bbox.h >= 398
        assert len(clusters) == 2


def test_conflicting_tags_on_one_body_cannot_become_connected_equipment():
    p = page(body() + [path("inlet", [(100, 300), (400, 300)])])
    ds = [
        detection("upper", (395, 150, 200, 110), "V101", "vessel"),
        detection("full", (395, 150, 210, 410), "V102", "vessel"),
    ]
    fused = fuse_objects(source_name="test", pages=[p], detections=ds, per_page_status={0: "ok"})
    assert len(fused.graph.nodes) == 2
    assert any(c["type"] == "fusion_instance_uncertainty" for c in fused.graph.conflicts)
    result = build_page_topology(page=p, nodes=fused.graph.nodes)
    assert not accepted(result)
    assert all(e.attributes["provisional_review_only"] for e in result.edges)


def test_bbox_contact_without_native_glyph_is_provisional_even_with_verified_claim():
    p = page([path("pipe", [(100, 300), (900, 300)])])
    result = build_page_topology(page=p, nodes=[node("left", 80), node("right", 880)])
    assert result.edges and not accepted(result)
    e = result.edges[0]
    e.attributes["route_evidence"].update(status="verified", failures=[])
    assert "unsupported_physical_port" in route_evidence_failures(e, p)


def test_mixed_pdf_path_preserves_pipe_segment_but_excludes_owned_outline():
    # A single PDF drawing path closes a symbol, then continues as piping.
    mixed = path(
        "mixed",
        [
            (100, 280),
            (140, 280),
            (140, 320),
            (100, 320),
            (100, 280),
            (100, 300),
            (100, 320),
            (140, 320),
            (140, 300),
            (500, 300),
        ],
    )
    p = page([mixed, rectangle("right", 500, 280, 40, 40)])
    ns = [node("left", 100), node("right", 500)]
    result = build_page_topology(page=p, nodes=ns)
    assert pairs(result) == {frozenset(("left", "right"))}
    e = accepted(result)[0]
    assert e.polyline_global == [(140, 300), (500, 300)]
    assert "mixed" in e.source_evidence_ids
    assert not route_evidence_failures(e, p)


def test_connected_symbols_do_not_become_one_exterior_face():
    p = page(
        [
            rectangle("a", 100, 280, 40, 40),
            rectangle("b", 300, 280, 40, 40),
            path("pipe", [(140, 280), (300, 280)]),
        ]
    )
    faces = native_symbols(p)
    assert not any(s.bbox.w > 100 for s in faces)
    ds = [detection("a", (100, 280, 40, 40)), detection("b", (300, 280, 40, 40))]
    assert len(match_instances(ds, {0: p})[0]) == 2


def test_open_symbol_has_ink_supported_ports_and_owns_only_its_strokes():
    # Open U-shaped equipment: no closed contour is required.
    p = page(
        [
            path("glyph", [(400, 200), (400, 400), (600, 400), (600, 200)]),
            path("inlet", [(100, 300), (400, 300)]),
            rectangle("left", 60, 280, 40, 40),
        ]
    )
    ns = [node("left", 60), node("open", 400, 200, 200, 200, equipment_class="filter")]
    cs = symbol_contours(p, ns)
    assert cs["open"].basis == "native_open_symbol"
    result = build_page_topology(page=p, nodes=ns)
    assert pairs(result) == {frozenset(("left", "open"))}
    assert not route_evidence_failures(accepted(result)[0], p)


def test_orphan_body_is_review_candidate_and_cannot_support_accepted_bypass():
    p = page(
        body() + [path("inlet", [(100, 300), (400, 300)]), path("drain", [(500, 550), (500, 700)])]
    )
    p.role = "pid"
    fused = fuse_objects(source_name="test", pages=[p], detections=[], per_page_status={0: "ok"})
    assert len(fused.graph.nodes) == 1
    recovered = fused.graph.nodes[0]
    assert recovered.attributes["native_symbol_candidate"] is True
    assert recovered.attributes["equipment_class"] == "unclassified"
    ns = [recovered, node("left", 80), node("drain", 480, 700)]
    p.paths.extend([rectangle("left", 80, 280, 40, 40), rectangle("drain", 480, 700, 40, 40)])
    result = build_page_topology(page=p, nodes=ns)
    assert not accepted(result)
    assert result.edges
    assert all(
        "unresolved_symbol_identity" in e.attributes["route_evidence"]["failures"]
        for e in result.edges
    )


def test_real_rectangular_pipe_loop_without_symbol_evidence_is_not_owned():
    p = page([path("loop", [(200, 200), (600, 200), (600, 500), (200, 500), (200, 200)])])
    assert symbol_contours(p, []) == {}
    p.role = "pid"
    assert not fuse_objects(
        source_name="test", pages=[p], detections=[], per_page_status={0: "ok"}
    ).graph.nodes


@pytest.mark.parametrize("corruption", ["symbol_id", "ink", "boundary", "wall_route"])
def test_acceptance_independently_checks_native_port_proof(corruption):
    ns = [node("left", 80), node("right", 880)]
    p = page([path("pipe", [(100, 300), (900, 300)])], nodes=ns)
    e = accepted(build_page_topology(page=p, nodes=ns))[0]
    proof = e.attributes["route_evidence"]
    if corruption == "symbol_id":
        proof["ports"][0]["symbol_id"] = "invented"
    elif corruption == "ink":
        proof["ports"][0]["outline_segments"][0] = ["pipe", (300, 300), (400, 300)]
    elif corruption == "boundary":
        proof["ports"][0]["basis"] = "symbol_boundary"
    else:
        e.polyline_global = [(120, 300), (120, 320), (880, 300)]
    assert route_evidence_failures(e, p)


def test_flattened_native_curves_also_recover_a_missing_body():
    from diagex.vision.vector_geometry import path_vertices

    fragments = []
    for original in body():
        if original.primitive == "curve":
            points = [tuple(round(v) for v in point) for point in path_vertices(original)]
            fragments.extend(
                path(f"{original.id}-{i}", [a, b])
                for i, (a, b) in enumerate(zip(points, points[1:], strict=False))
            )
        else:
            fragments.append(original)
    p = page(fragments)
    p.role = "pid"
    fused = fuse_objects(source_name="test", pages=[p], detections=[], per_page_status={0: "ok"})
    assert len(fused.graph.nodes) == 1
    assert fused.graph.nodes[0].attributes["native_symbol_candidate"]


def test_arrow_sharing_a_pdf_path_does_not_remove_the_pipe():
    p = page(
        [
            path("mixed", [(300, 300), (320, 300), (300, 295), (300, 305), (320, 300), (900, 300)]),
            path("inlet", [(100, 300), (300, 300)]),
        ],
        nodes=[node("left", 80), node("right", 880)],
    )
    result = build_page_topology(page=p, nodes=[node("left", 80), node("right", 880)])
    assert pairs(result) == {frozenset(("left", "right"))}
    edge = accepted(result)[0]
    assert edge.attributes["flow_direction"] == "forward"
    assert "mixed" in edge.source_evidence_ids


def test_displaced_filter_recovers_unique_contacted_outline():
    left = node("filter", 173, 240, 34, 108, equipment_class="filter")
    right = node("device", 400, 250, 37, 108)
    p = page(
        [
            rectangle("filter-outline", 200, 250, 37, 108),
            path("pipe", [(237, 300), (400, 300)], visual_style="solid"),
        ],
        nodes=[right],
    )
    contours = symbol_contours(p, [left, right])
    assert contours[left.id].bbox.x == 200
    result = build_page_topology(page=p, nodes=[left, right])
    assert pairs(result) == {frozenset((left.id, right.id))}
    assert accepted(result)[0].polyline_global in [
        [(237, 300), (400, 300)],
        [(400, 300), (237, 300)],
    ]


def test_displaced_repair_does_not_borrow_neighbors_symbol_or_uncontacted_ink():
    displaced = node("uncertain", 173, 240, 34, 108, equipment_class="filter")
    owner = node("neighbor", 200, 250, 37, 108, equipment_class="filter")
    outline = rectangle("outline", 200, 250, 37, 108)
    p = page([outline, path("pipe", [(237, 300), (400, 300)], visual_style="solid")])
    assert "uncertain" not in symbol_contours(p, [displaced, owner])
    assert "uncertain" not in symbol_contours(page([outline]), [displaced])


def test_route_alternatives_preserve_native_taps_and_all_evidence():
    from diagex.vision.models import ReconciledEdge
    from diagex.vision.topology import _group_route_alternatives

    def route(name, y, *, native=False):
        return ReconciledEdge(
            id=name,
            from_node="a",
            to_node="b",
            line_type="process",
            confidence="low",
            polyline_global=[(100, y), (200, y)],
            source_evidence_ids=[name + "-ink"],
            attributes={
                "provisional_review_only": True,
                "route_evidence": {
                    "failures": [
                        "endpoint_role_uncertain" if native else "unsupported_physical_port"
                    ],
                    "ports": [
                        {"node_id": n, "symbol_id": n if native else None, "point": [x, y]}
                        for n, x in [("a", 100), ("b", 200)]
                    ],
                },
            },
        )

    alternatives = [route("one", 100), route("two", 110), route("three", 120)]
    for order in permutations(alternatives):
        grouped = _group_route_alternatives(list(order))
        assert len(grouped) == 1
        assert grouped[0].id == "one"
        assert [e["source_evidence_ids"] for e in grouped[0].attributes["route_alternatives"]] == [
            ["one-ink"],
            ["three-ink"],
            ["two-ink"],
        ]
    assert (
        len(
            _group_route_alternatives(
                [route("upper", 100, native=True), route("lower", 200, native=True)]
            )
        )
        == 2
    )


def test_same_symbol_routes_are_instance_conflicts_without_connection_edges():
    p = page(body() + [path("pipe", [(100, 300), (400, 300)])])
    ns = [
        node("first", 400, 150, 200, 400, equipment_class="vessel"),
        node("second", 400, 150, 200, 400, equipment_class="vessel"),
    ]
    result = build_page_topology(page=p, nodes=ns)
    assert not result.edges
    conflicts = [c for c in result.ambiguities if c["type"] == "physical_instance_geometry"]
    assert len(conflicts) == 1
    assert conflicts[0]["node_ids"] == ["first", "second"]
    assert "same_physical_symbol" in conflicts[0]["rejected_routes"][0]["failures"]


def test_displaced_repair_leaves_two_equally_supported_symbols_ambiguous():
    uncertain = node("uncertain", 220, 250, 37, 108, equipment_class="filter")
    p = page(
        [
            rectangle("first", 195, 250, 37, 108),
            rectangle("second", 245, 250, 37, 108),
            path("left-pipe", [(100, 300), (195, 300)], visual_style="solid"),
            path("right-pipe", [(282, 300), (400, 300)], visual_style="solid"),
        ]
    )
    assert uncertain.id not in symbol_contours(p, [uncertain])


def test_two_native_taps_between_same_devices_remain_two_connections():
    ns = [node("vessel", 200, 200, 100, 300), node("gauge", 450, 200, 40, 300)]
    p = page(
        [
            path("upper-tap", [(300, 250), (450, 250)]),
            path("lower-tap", [(300, 450), (450, 450)]),
        ],
        nodes=ns,
    )
    result = build_page_topology(page=p, nodes=ns)
    assert len(accepted(result)) == 2
    assert {tuple(e.polyline_global) for e in accepted(result)} == {
        ((450, 250), (300, 250)),
        ((450, 450), (300, 450)),
    }
