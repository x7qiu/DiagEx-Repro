"""Drawing-independent ownership contracts, with real native vector evidence."""

import math

from diagex.vision.evidence import TextEvidence
from diagex.vision.models import BBox, ReconciledGraph
from diagex.vision.native_hierarchy import apply_assembly_bindings, build_native_hierarchy
from diagex.vision.native_text import build_native_text_inventory
from diagex.vision.topology import build_page_topology
from diagex.vision.vector_geometry import segment_key, symbol_contours
from tests.unit.test_native_scene import rectangle
from tests.unit.test_port_topology import accepted, node, page, path


def text(name, value, x, y, w=100, h=20):
    return TextEvidence(
        id=name,
        text=value,
        bbox=BBox(x=x, y=y, w=w, h=h),
        page_index=0,
        block_index=0,
        line_index=0,
        word_index=0,
    )


def fixture(*, heading="K-101", note=True, boundary_style="dash_dot"):
    ns = [node("first", 240, 350, 50, 60), node("second", 600, 350, 50, 60)]
    ns[0].label = "K-101"
    p = page(
        [
            path(
                "boundary",
                [(100, 200), (800, 200), (800, 700), (100, 700), (100, 200)],
                visual_style=boundary_style,
            ),
            path("caption-line-a", [(150, 620), (200, 620)]),
            path("caption-line-b", [(200, 620), (250, 620)]),
            path("heading-line", [(400, 100), (500, 100)]),
            path("real-pipe", [(290, 380), (600, 380)]),
        ],
        nodes=ns,
    )
    p.text_spans = [
        text("caption", "K-101", 150, 600),
        text("heading", heading, 400, 80),
        text("description", "Air compressor", 380, 110, 150),
    ]
    if note:
        p.text_spans.append(text("note", "Supplier package boundary", 100, 750, 300))
    return p, ns


def test_assembly_tag_and_specification_are_owned_without_merging_symbols():
    p, ns = fixture()
    h = build_native_hierarchy(p, ns)
    assert len(h.assemblies) == 1
    a = h.assemblies[0]
    assert a.label == "K-101" and a.status == "supported"
    assert a.member_node_ids == ["first", "second"]
    assert a.source_text_ids == ["caption", "heading"]
    assert not apply_assembly_bindings(h, ns)
    assert ns[0].label == "unlabelled"
    assert [n.bbox_global.w for n in ns] == [50, 50]
    graph = ReconciledGraph(
        source_path="fixture", nodes=ns, assemblies=h.assemblies, text_bindings=h.text_bindings
    )
    inventory = build_native_text_inventory(pages=[p], graph=graph)
    occurrences = [i for i in inventory.items if i.text == "K-101"]
    assert len(occurrences) == 2
    assert all(i.matched_assembly_ids == [a.id] and not i.matched_node_ids for i in occurrences)
    result = build_page_topology(page=p, nodes=ns)
    assert len(accepted(result)) == 1
    assert accepted(result)[0].source_evidence_ids == ["real-pipe"]
    assert a.id not in {e.from_node for e in result.edges}


def test_contradictory_heading_is_one_assembly_conflict():
    p, ns = fixture(heading="K-102")
    h = build_native_hierarchy(p, ns)
    assert h.assemblies[0].label is None
    assert h.assemblies[0].label_candidates == ["K-101", "K-102"]
    conflicts = apply_assembly_bindings(h, ns)
    assert len(conflicts) == 1 and conflicts[0]["assembly_id"] == h.assemblies[0].id
    assert ns[0].label == "unlabelled"


def test_containment_and_solid_pipe_loop_do_not_establish_an_assembly():
    for args in [{"note": False}, {"boundary_style": "solid"}]:
        p, ns = fixture(**args)
        assert not build_native_hierarchy(p, ns).assemblies


def test_native_leader_takes_precedence_over_assembly_caption():
    p, ns = fixture()
    p.paths.append(path("leader", [(250, 610), (350, 610), (290, 380)]))
    assert not build_native_hierarchy(p, ns).assemblies


def test_specification_detection_is_removed_only_without_physical_glyph():
    p, ns = fixture()
    fake = node("heading-as-compressor", 400, 80, 100, 20)
    h = build_native_hierarchy(p, [*ns, fake])
    assert h.excluded_node_ids == {fake.id}
    p.paths.append(rectangle("actual-glyph", 400, 80, 100, 20))
    assert fake.id not in build_native_hierarchy(p, [*ns, fake]).excluded_node_ids


def test_repeated_packages_preserve_repeated_local_tags_and_real_crossing_pipe():
    from diagex.vision.quality import _duplicate_tag_groups

    ns = [
        node("a1", 220, 160, 40, 40),
        node("a2", 600, 160, 40, 40),
        node("b1", 220, 480, 40, 40),
        node("b2", 600, 480, 40, 40),
        node("outside", 20, 160, 40, 40),
    ]
    ns[0].label = ns[2].label = "QV01"
    p = page(
        [
            path(
                "a",
                [(100, 100), (800, 100), (800, 350), (100, 350), (100, 100)],
                visual_style="dash_dot",
            ),
            path(
                "b",
                [(100, 420), (800, 420), (800, 670), (100, 670), (100, 420)],
                visual_style="dash_dot",
            ),
            path("ua", [(150, 320), (250, 320)]),
            path("ub", [(150, 640), (250, 640)]),
            path("crossing", [(60, 180), (220, 180)]),
        ],
        nodes=ns,
    )
    p.text_spans = [
        text("ca", "PK-001A", 150, 300),
        text("cb", "PK-001B", 150, 620),
        text("note", "Vendor scope", 100, 750),
    ]
    h = build_native_hierarchy(p, ns)
    assert len(h.assemblies) == 2
    assert {tuple(a.member_node_ids) for a in h.assemblies} == {("a1", "a2"), ("b1", "b2")}
    graph = ReconciledGraph(source_path="fixture", nodes=ns, assemblies=h.assemblies)
    assert not _duplicate_tag_groups(graph)
    assert any(
        set((e.from_node, e.to_node)) == {"outside", "a1"}
        for e in accepted(build_page_topology(page=p, nodes=ns))
    )


def test_complete_instrument_frames_cannot_become_dashed_signal_runs():
    ns = [
        node("first", 500, 180, 80, 80),
        node("second", 500, 285, 80, 80),
        node("third", 500, 390, 80, 80),
    ]
    paths = []
    for n in ns:
        n.kind = "instrument"
        b = n.bbox_global
        circle = [
            (
                round(b.x + 40 + 40 * math.cos(i * math.pi / 16)),
                round(b.y + 40 + 40 * math.sin(i * math.pi / 16)),
            )
            for i in range(33)
        ]
        paths.extend(
            [
                path("circle-" + n.id, circle, closed=True),
                rectangle("frame-" + n.id, b.x, b.y, b.w, b.h),
            ]
        )
    p = page(paths)
    contours = symbol_contours(p, ns)
    for n in ns:
        b = n.bbox_global
        assert segment_key("frame-" + n.id, (b.x, b.y), (b.x, b.y2)) in contours[n.id].segments
    result = build_page_topology(page=p, nodes=ns)
    assert not result.edges


def test_nested_scopes_keep_membership_separate_from_identity():
    p, ns = fixture()
    p.paths.append(
        path(
            "outer",
            [(50, 150), (900, 150), (900, 730), (50, 730), (50, 150)],
            visual_style="dash_dot",
        )
    )
    p.paths.append(path("outer-caption", [(500, 195), (600, 195)]))
    p.text_spans.append(text("outer-label", "PK-200", 500, 175))
    h = build_native_hierarchy(p, ns)
    assert len(h.assemblies) == 2
    child = next(a for a in h.assemblies if a.label_candidates[0] == "K-101")
    parent = next(a for a in h.assemblies if a.label_candidates[0] == "PK-200")
    assert child.parent_assembly_id == parent.id


def test_incomplete_scope_withholds_nearest_tag_without_inventing_an_assembly():
    p, ns = fixture()
    p.paths = [v for v in p.paths if v.id != "boundary"]
    ns[0].source_evidence_ids = ["caption"]
    ns[0].source_quote = "K-101"
    ns[0].system_confidence = 0.8
    ns[0].attributes.update(
        canonical_tag="K-101",
        native_tag_assignment="mutual_best_geometry",
        label_candidates=["K-101"],
        system_confidence_evidence={"native_text_agreement": True},
    )
    h = build_native_hierarchy(p, ns)
    assert not h.assemblies
    assert "caption" in h.reserved_text_ids
    apply_assembly_bindings(h, ns)
    assert ns[0].label == "unlabelled" and ns[0].source_quote is None
    assert "native_tag_assignment" not in ns[0].attributes
    assert ns[0].system_confidence == ns[0].attributes["system_confidence"] == 0.68
    assert ns[0].system_confidence_level == "medium"
    graph = ReconciledGraph(source_path="fixture", nodes=ns, text_bindings=h.text_bindings)
    item = next(
        i
        for i in build_native_text_inventory(pages=[p], graph=graph).items
        if "caption" in i.source_ids
    )
    assert item.ownership_scope == "unresolved_scope"
    assert item.status == "unresolved" and item.blocking
    assert not item.matched_node_ids and not item.matched_assembly_ids


def test_multiple_captions_in_one_supplier_scope_are_not_a_printed_identity_conflict():
    p, ns = fixture(heading="PK-100A/B")
    p.paths.append(path("second-caption-line", [(500, 560), (600, 560)]))
    p.text_spans.append(text("second-caption", "K-102", 500, 540))
    h = build_native_hierarchy(p, ns)
    a = h.assemblies[0]
    assert a.status == "uncertain" and a.label is None
    assert a.label_candidates == ["K-101", "K-102", "PK-100A/B"]
    assert len(h.assemblies) == 1  # No invented division into two scopes.
