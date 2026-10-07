"""Open glyph recovery, relevant references, and Chinese explanations; offline."""

import copy

import pytest
from PIL import Image

from diagex.knowledge.resolver import knowledge_snapshot, resolve_symbol_context
from diagex.vision.models import BBox
from diagex.vision.symbol_candidates import SymbolCandidate, symbol_candidates
from diagex.vision.symbol_interpretation import PerceivedObject, _candidate_kind_compatible
from diagex.web.evidence_origin import recognition_diagnostic
from tests.unit.test_port_topology import page, path


def inline(turn=False, mirror=False, pipes=True, bars=True):
    segments = [[(100, 100), (134, 120)]]
    if bars:
        segments += [[(100, 100), (100, 120)], [(134, 100), (134, 120)]]
    if pipes:
        segments += [[(60, 110), (100, 110)], [(134, 110), (170, 110)]]

    def transform(p):
        x, y = p
        if mirror:
            x = 234 - x
        return (y, x) if turn else (x, y)

    return page(
        [
            path(str(i), [transform(p) for p in points], visual_style="solid")
            for i, points in enumerate(segments)
        ]
    )


@pytest.mark.parametrize(
    "turn,mirror", [(False, False), (False, True), (True, False), (True, True)]
)
def test_open_inline_geometry_has_tight_independent_proposal(turn, mirror):
    candidates = symbol_candidates(inline(turn, mirror))
    found = [c for c in candidates if c.shape == "open_inline_valve"]
    assert len(found) == 1
    assert sorted([found[0].bbox.w, found[0].bbox.h]) == [20, 34]
    assert len(found[0].source_path_ids) == 3  # external pipe excluded


@pytest.mark.parametrize("pipes,bars", [(False, True), (True, False), (False, False)])
def test_slashes_and_break_marks_are_not_valve_candidates(pipes, bars):
    assert not any(
        c.shape == "open_inline_valve" for c in symbol_candidates(inline(pipes=pipes, bars=bars))
    )


def test_bowtie_is_not_duplicated_as_open_valve():
    p = inline()
    p.paths.append(path("other-diagonal", [(100, 120), (134, 100)]))
    found = [c for c in symbol_candidates(p) if c.shape in {"valve_body", "open_inline_valve"}]
    assert len(found) == 1
    assert found[0].shape == "valve_body"


def test_round_connector_requires_source_arrow_and_connection_evidence():
    c = SymbolCandidate(
        id="circle",
        page_index=0,
        bbox=BBox(x=1, y=1, w=20, h=20),
        shape="round_symbol",
        source_path_ids=["p"],
    )
    assert not _candidate_kind_compatible(c, PerceivedObject(candidate_id="circle", kind="opc"))
    assert not _candidate_kind_compatible(
        c,
        PerceivedObject(candidate_id="circle", kind="opc", recognition_evidence="圆形，文字为管网"),
    )
    assert _candidate_kind_compatible(
        c,
        PerceivedObject(
            candidate_id="circle",
            kind="opc",
            recognition_evidence="圆内实心箭头与管线相接，旁边标注至管网",
        ),
    )


@pytest.fixture(scope="module")
def snapshot():
    return knowledge_snapshot("general", source_ids=["sht-3101-2017"])


def test_local_open_valve_retrieves_check_reference_and_short_crop(snapshot):
    p = inline()
    candidates = symbol_candidates(p)
    context = resolve_symbol_context(snapshot, p, candidates, [], BBox(x=0, y=0, w=300, h=300))
    assert context["reference_ids"][0] == "sht-3101-2017.t4.3-11"
    assert context["supplied_assets"][0]["asset_id"] == "tight"
    assert len(context["supplied_assets"]) <= 2 and len(context["references"]) <= 8
    assert "third-party" not in str(context["supplied_assets"])
    changed = copy.deepcopy(snapshot)
    e = next(e for e in changed["entries"] if e["id"] == context["reference_ids"][0])
    e["catalog"]["interpretation"] += " changed"
    assert (
        resolve_symbol_context(changed, p, candidates, [], BBox(x=0, y=0, w=300, h=300))["identity"]
        != context["identity"]
    )


def test_retrieval_off_and_conflicting_edition(snapshot):
    p = inline()
    c = symbol_candidates(p)
    box = BBox(x=0, y=0, w=300, h=300)
    assert resolve_symbol_context({}, p, c, [], box) == {}
    wrong = copy.deepcopy(snapshot)
    wrong["profile"] = {
        "id": "conflict",
        "version": 1,
        "context": {"standards": [{"name": "SH/T 3101", "edition": "1999"}]},
        "reference_ids": [],
    }
    assert not any(
        e.startswith("sht-") for e in resolve_symbol_context(wrong, p, c, [], box)["reference_ids"]
    )
    wrong["overrides"] = [
        {"pages": [1], "context": {"standards": [{"name": "SH/T 3101", "edition": "2017"}]}}
    ]
    assert "sht-3101-2017.t4.3-11" in resolve_symbol_context(wrong, p, c, [], box)["reference_ids"]


def test_diagnostics_distinguish_uncertainty_and_validation_without_rewriting():
    row = {"candidate_only": True, "reason": "classification contradicts native glyph geometry"}
    original = copy.deepcopy(row)
    assert recognition_diagnostic(row)["code"] == "validation_rejected"
    assert row == original
    assert recognition_diagnostic({"candidate_only": True})["code"] == "uncertain"


def test_reference_crop_is_exact_source_crop(snapshot):
    from diagex.knowledge.resolver import library_root

    entry = next(e for e in snapshot["entries"] if e["id"] == "sht-3101-2017.t4.3-11")
    assets = {a["id"]: a for a in entry["assets"]}
    with (
        Image.open(library_root() / assets["primary"]["path"]) as source,
        Image.open(library_root() / assets["tight"]["path"]) as tight,
    ):
        assert source.crop((38, 3, 86, 32)).tobytes() == tight.tobytes()
    assert "单向阀" in entry["aliases"]


def test_closed_panel_with_diagonal_is_not_open_valve():
    p = inline()
    p.paths += [
        path("top", [(100, 100), (134, 100)], visual_style="solid"),
        path("bottom", [(100, 120), (134, 120)], visual_style="solid"),
    ]
    assert not any(c.shape == "open_inline_valve" for c in symbol_candidates(p))


def test_pipeline_text_is_retained_but_not_assigned_as_device_tag():
    from diagex.vision.evidence import TextEvidence
    from diagex.vision.models import ReconciledNode
    from diagex.vision.text_assignment import assign_text

    p = inline()
    text = TextEvidence(id="line-id", text="50-PA-00901-M15B-N", bbox=BBox(x=75, y=65, w=120, h=25))
    p.text_spans = [text]
    n = ReconciledNode(
        id="valve",
        page_index=0,
        kind="equipment",
        label="unlabelled",
        bbox_global=BBox(x=100, y=100, w=34, h=20),
        attributes={"valve_type": "check"},
        confidence="medium",
    )
    result = assign_text(nodes=[n], pages=[p])
    assert result.nodes[0].label == "unlabelled"
    assert p.text_spans[0].text == "50-PA-00901-M15B-N"
