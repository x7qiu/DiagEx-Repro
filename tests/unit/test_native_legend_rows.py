"""Source row identity and bounded legend classification, without live models."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from diagex.config import Config
from diagex.extractors.pid_legend import _dedupe, _extract_from_page
from diagex.llm.cost import CostTracker
from diagex.vision.evidence import TextEvidence
from diagex.vision.legend_models import LegendEntry, LegendPack
from diagex.vision.legend_rows import native_legend_rows
from diagex.vision.models import BBox, DiagramPage
from tests.unit.test_native_scene import rectangle
from tests.unit.test_port_topology import page, path


def fixture(count=2, *, left=False):
    p = page([rectangle(f"glyph-{i}", 120, 30 + i * 50, 30, 30) for i in range(count)])
    p.height = max(200, count * 50 + 30)
    p.text_spans = [
        TextEvidence(
            id=f"label-{i}",
            text="Full definition",
            bbox=BBox(x=20 if left else 165, y=40 + i * 50, w=80, h=10),
        )
        for i in range(count)
    ]
    image = Image.new("RGB", (p.width, p.height), "white")
    draw = ImageDraw.Draw(image)
    for glyph in p.paths:
        b = glyph.bbox
        draw.rectangle((b.x, b.y, b.x2, b.y2), outline="black", width=2)
    rendered = DiagramPage(
        **{
            k: getattr(p, k)
            for k in (
                "page_index",
                "width",
                "height",
                "dpi",
                "effective_dpi",
                "is_scanned",
                "source_ref",
            )
        },
        image=image,
    )
    return rendered, p


@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("scale", [1, 3])
def test_repeated_labels_remain_distinct_complete_rows_across_clipped_regions(left, scale):
    _, p = fixture(left=left)
    for item in [*p.paths, *p.text_spans]:
        item.bbox = BBox(**{k: v * scale for k, v in item.bbox.model_dump().items()})
        if hasattr(item, "points"):
            item.points = [(x * scale, y * scale) for x, y in item.points]
    full = native_legend_rows(p)
    assert len(full) == 2 and full[0].label == full[1].label == "Full definition"
    assert full[0].id != full[1].id
    # The requested region cuts through the glyph and excludes the entire label.
    partial = native_legend_rows(p, BBox(x=130 * scale, y=35 * scale, w=10 * scale, h=20 * scale))
    assert partial == full[:1]
    p.paths.reverse()
    p.text_spans.reverse()
    assert native_legend_rows(p) == full


def test_multiline_and_spaced_labels_preserve_full_glyph_and_ignore_table_ticks():
    _, p = fixture(count=1)
    p.text_spans = [
        TextEvidence(id="a", text="Shared", bbox=BBox(x=165, y=34, w=32, h=10)),
        TextEvidence(id="b", text="display", bbox=BBox(x=205, y=34, w=35, h=10)),
        TextEvidence(id="c", text="operator position", bbox=BBox(x=165, y=48, w=80, h=10)),
    ]
    p.paths.extend([path("border", [(10, 0), (10, 200)]), path("tick", [(10, 100), (20, 100)])])
    rows = native_legend_rows(p)
    assert len(rows) == 1
    assert rows[0].label == "Shared display operator position"
    assert rows[0].source_path_ids == ["glyph-0"]
    assert rows[0].bbox == BBox(x=120, y=30, w=30, h=30)


class Client:
    def __init__(self, mode="accept"):
        self.calls = []
        self.mode = mode
        self.row_order = {}

    def messages_create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = [
            json.loads(b["text"]) for b in kwargs["messages"][0]["content"] if b["type"] == "text"
        ]
        for item in inputs:
            self.row_order.setdefault(item["row_id"], len(self.row_order))
        assert len(inputs) <= 12
        assert len(kwargs["messages"][0]["content"]) == 2 * len(inputs)
        assert kwargs["thinking"] == {"type": "disabled"}
        rows = []
        if self.mode == "failure" and len(self.calls) == 2:
            raise RuntimeError("batch unavailable")
        for i, item in enumerate(inputs):
            i = self.row_order[item["row_id"]]
            row = {
                "row_id": item["row_id"],
                "decision": "accept",
                "kind": "instrument",
                "symbol_class": "unclassified_instrument",
                "candidate_shapes": ["instrument_frame"],
                "label": "MODEL FRAGMENT",
                "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
            }
            if self.mode == "mixed":
                if i == 0:
                    row = {"row_id": item["row_id"], "decision": "reject", "reason": "title block"}
                if i == 1:
                    continue
                if i == 2:
                    rows.append({**row, "symbol_class": "vessel"})
                if i == 3:
                    row = {"row_id": item["row_id"], "decision": "accept"}
            rows.append(row)
        rows.append(
            {
                "row_id": "invented",
                "decision": "accept",
                "kind": "equipment",
                "symbol_class": "pump",
            }
        )
        return SimpleNamespace(
            content=[{"type": "tool_use", "name": "submit_legend_rows", "input": {"rows": rows}}],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def extract(count=2, mode="accept"):
    rendered, p = fixture(count)
    client = Client(mode)
    coverage = []
    entries = _extract_from_page(
        page=rendered,
        region=None,
        page_evidence=p,
        client=client,
        cost_tracker=CostTracker(),
        cfg=Config(),
        coverage=coverage,
    )
    return entries, coverage, client, p


def test_native_classifier_uses_bounded_batches_and_cannot_replace_source_identity():
    entries, coverage, client, p = extract(25)
    assert len(client.calls) == 3
    assert len(entries) == len(coverage) == 25
    assert all(
        e.label == "Full definition" and e.source_bbox.w > 30 and e.source_row_id for e in entries
    )
    assert len({e.source_row_id for e in entries}) == 25
    assert all(c.status == "complete" for c in coverage)
    # Repeated label/glyph pixels are legitimate physical rows, not bad crops.
    assert len(_dedupe(entries)) == 25
    assert all(e.image_b64 for e in _dedupe(entries))
    assert [e.source_row_id for e in entries] == [r.id for r in native_legend_rows(p)]
    assert len(LegendPack(entries=entries[:1]).merge(LegendPack(entries=entries)).entries) == 25


def test_missing_malformed_and_conflicting_classifications_remain_reviewable():
    entries, coverage, client, _ = extract(5, "mixed")
    assert len(client.calls) == 2 and len(entries) == 4
    assert [e.attributes["row_status"] for e in entries] == [
        "uncertain",
        "uncertain",
        "uncertain",
        "accept",
    ]
    assert [c.status for c in coverage] == ["complete", "partial", "partial", "partial", "complete"]
    assert len(json.loads(entries[1].attributes["classification_evidence"])) == 2
    assert json.loads(entries[2].attributes["classification_evidence"])[0]["decision"] == "accept"
    assert all(e.source_bbox and e.image_b64 for e in entries)


def test_rejected_definition_gets_one_second_review_and_completed_rows_are_reused():
    rendered, evidence = fixture(3)
    source_rows = native_legend_rows(evidence)

    class RecheckClient(Client):
        def messages_create(self, **kwargs):
            response = super().messages_create(**kwargs)
            inputs = [
                json.loads(b["text"])
                for b in kwargs["messages"][0]["content"]
                if b["type"] == "text"
            ]
            decisions = []
            for item in inputs:
                row_id = item["row_id"]
                reviewing = "review_instruction" in item
                if row_id == source_rows[1].id or (row_id == source_rows[0].id and not reviewing):
                    decisions.append({"row_id": row_id, "decision": "reject", "reason": "metadata"})
                else:
                    decisions.append(
                        {
                            "row_id": row_id,
                            "decision": "accept",
                            "kind": "equipment",
                            "symbol_class": "actuator",
                            "attributes": {"symbol_role": "actuator"},
                        }
                    )
            response.content[0]["input"]["rows"] = decisions
            return response

    client = RecheckClient()
    coverage = []
    kwargs = dict(
        page=rendered,
        region=None,
        page_evidence=evidence,
        client=client,
        cost_tracker=CostTracker(),
        cfg=Config(),
    )
    entries = _extract_from_page(**kwargs, coverage=coverage)
    assert len(client.calls) == 2
    assert {e.source_row_id for e in entries} == {source_rows[0].id, source_rows[2].id}
    assert [c.verification_passes for c in coverage] == [2, 2, 1]
    assert [c.entry_count for c in coverage] == [1, 0, 1]
    assert entries[0].attributes["previous_rejection_reason"] == "metadata"
    second_inputs = [
        json.loads(b["text"])
        for b in client.calls[1]["messages"][0]["content"]
        if b["type"] == "text"
    ]
    assert {i["row_id"] for i in second_inputs} == {source_rows[0].id, source_rows[1].id}
    assert all("review_instruction" in i for i in second_inputs)
    prior = LegendPack(entries=entries, coverage=coverage)
    reused = _extract_from_page(**kwargs, coverage=[], prior=prior)
    assert reused == entries and len(client.calls) == 2


def test_failed_batch_preserves_other_batches_without_navigation_or_retry():
    entries, coverage, client, _ = extract(25, "failure")
    assert len(client.calls) == 3 and len(entries) == 25
    assert sum(c.status == "partial" for c in coverage) == 12
    assert all("batch unavailable" in c.reason for c in coverage[12:24])


@pytest.mark.parametrize("mode,count,expected_pending", [("failure", 25, 12), ("mixed", 5, 3)])
def test_resume_reclassifies_only_unresolved_source_rows(mode, count, expected_pending):
    entries, old_coverage, _, _ = extract(count, mode)
    prior = LegendPack(entries=entries, coverage=old_coverage)
    rendered, evidence = fixture(count)
    client = Client()
    coverage = []
    recovered = _extract_from_page(
        page=rendered,
        region=None,
        client=client,
        cost_tracker=CostTracker(),
        cfg=Config(),
        page_evidence=evidence,
        coverage=coverage,
        prior=prior,
    )
    sent_ids = [
        json.loads(b["text"])["row_id"]
        for call in client.calls
        for b in call["messages"][0]["content"]
        if b["type"] == "text"
    ]
    assert len(sent_ids) == expected_pending
    completed_ids = {c.source_row_id for c in old_coverage if c.status == "complete"}
    assert not completed_ids.intersection(sent_ids)
    assert len(coverage) == count and all(c.status == "complete" for c in coverage)
    for old in entries:
        if old.attributes["row_status"] == "accept":
            assert next(e for e in recovered if e.source_row_id == old.source_row_id) == old


def test_legend_circuit_stops_failed_batches_across_pages_without_losing_rows():
    from diagex.extractors.legend_rows import LegendRunState

    rendered, evidence = fixture(60)
    state = LegendRunState()
    coverage = []

    class Unavailable:
        calls = 0

        def messages_create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("Connection error")

    client = Unavailable()
    entries = _extract_from_page(
        page=rendered,
        region=None,
        client=client,
        cost_tracker=CostTracker(),
        cfg=Config(),
        page_evidence=evidence,
        coverage=coverage,
        run_state=state,
    )
    assert client.calls == 3 and state.stop_reason
    assert len(entries) == len(coverage) == 60
    assert sum(c.failure_kind == "transport" for c in coverage) == 36
    assert sum(c.failure_kind == "not_inspected" for c in coverage) == 24
    _extract_from_page(
        page=rendered,
        region=None,
        client=client,
        cost_tracker=CostTracker(),
        cfg=Config(),
        page_evidence=evidence,
        coverage=[],
        run_state=state,
    )
    assert client.calls == 3


def test_customer_override_still_wins_over_native_row_with_same_label():
    entries, _, _, _ = extract(1)
    override = LegendEntry(
        label=entries[0].label, symbol_class="reviewed", source="customer_override"
    )
    assert LegendPack(entries=[override]).merge(LegendPack(entries=entries)).entries == [override]


def test_same_row_selects_complete_observation_and_preserves_class_conflicts():
    entries, _, _, _ = extract(1)
    complete = entries[0]
    clipped = complete.model_copy(
        update={
            "source_bbox": BBox(x=120, y=30, w=4, h=5),
            "image_b64": None,
            "crop_quality": "unavailable",
        }
    )
    assert _dedupe([clipped, complete]) == [complete]
    assert _dedupe([complete, clipped]) == [complete]
    wrong = clipped.model_copy(update={"kind": "equipment", "symbol_class": "vessel"})
    row = _dedupe([wrong, complete])[0]
    assert row.source_bbox == complete.source_bbox
    assert row.attributes["row_status"] == "uncertain"
    assert len(json.loads(row.attributes["row_conflicts"])) == 2


def test_glyph_touching_long_demonstration_pipe_is_not_a_table_tick():
    _, p = fixture(1)
    p.paths = [
        path("body", [(120, 30), (150, 45), (120, 60), (120, 30)], closed=True),
        path("long-pipe", [(0, 45), (600, 45)]),
    ]
    rows = native_legend_rows(p)
    assert len(rows) == 1 and rows[0].source_path_ids == ["body"]
