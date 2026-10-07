"""Disconnected legend components must not absorb distant drafting panels."""

import pytest

from diagex.vision.evidence import TextEvidence
from diagex.vision.legend_rows import native_legend_rows
from diagex.vision.models import BBox
from tests.unit.test_native_scene import rectangle
from tests.unit.test_port_topology import page, path


def _fixture(extra, *, second_label=False, mirror=False, scale=1):
    evidence = page([rectangle("symbol", 120, 30, 30, 30), *extra])
    evidence.text_spans = [
        TextEvidence(id="label-0", text="Definition", bbox=BBox(x=165, y=40, w=80, h=10))
    ]
    if second_label:
        evidence.paths.append(rectangle("second-symbol", 120, 80, 30, 30))
        evidence.text_spans.append(
            TextEvidence(id="label-1", text="Other definition", bbox=BBox(x=165, y=90, w=80, h=10))
        )
    for item in [*evidence.paths, *evidence.text_spans]:
        b = item.bbox
        item.bbox = BBox(
            x=(500 - b.x2 if mirror else b.x) * scale,
            y=b.y * scale,
            w=b.w * scale,
            h=b.h * scale,
        )
        if hasattr(item, "points"):
            item.points = [((500 - x if mirror else x) * scale, y * scale) for x, y in item.points]
    return evidence


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("scale", [1, 3])
def test_distant_tall_panel_does_not_merge_neighboring_definitions(mirror, scale):
    evidence = _fixture(
        [rectangle("drafting-panel", 10, 0, 20, 140)],
        second_label=True,
        mirror=mirror,
        scale=scale,
    )
    original = evidence.model_dump()
    rows = native_legend_rows(evidence)
    assert [r.label for r in rows] == ["Definition", "Other definition"]
    assert [r.source_path_ids for r in rows] == [["symbol"], ["second-symbol"]]
    assert evidence.model_dump() == original
    evidence.paths.reverse()
    evidence.text_spans.reverse()
    assert native_legend_rows(evidence) == rows


def test_distant_off_row_rule_does_not_expand_symbol_crop():
    evidence = _fixture([path("rule", [(10, 33), (70, 33)])])
    rows = native_legend_rows(evidence)
    assert len(rows) == 1
    assert rows[0].source_path_ids == ["symbol"]
    assert rows[0].bbox == BBox(x=120, y=30, w=30, h=30)


@pytest.mark.parametrize(
    "extra",
    [
        # Distant alternatives aligned with the caption remain valid.
        rectangle("alternative", 10, 30, 30, 30),
        # Nearby but disconnected tall parts can belong to an assembly.
        rectangle("actuator", 82, 0, 25, 100),
        # A line-style alternative aligned with the caption is not a border.
        path("line-alternative", [(10, 45), (70, 45)]),
    ],
)
def test_disconnected_alternatives_and_compound_parts_are_preserved(extra):
    rows = native_legend_rows(_fixture([extra]))
    assert len(rows) == 1
    assert set(rows[0].source_path_ids) == {"symbol", extra.id}


def test_without_better_local_glyph_a_tall_candidate_is_not_discarded():
    evidence = _fixture([rectangle("tall-candidate", 10, 0, 20, 140)])
    evidence.paths = [p for p in evidence.paths if p.id != "symbol"]
    rows = native_legend_rows(evidence)
    assert len(rows) == 1 and rows[0].source_path_ids == ["tall-candidate"]
