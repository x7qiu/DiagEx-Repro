"""Actual model messages share bounded legend context and truthful supply traces."""

import copy
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from diagex.config import SymbolPerceptionConfig
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.legend_context import select_legend_context
from diagex.vision.raster_broad import BroadRasterClient, perceive_broad_raster
from diagex.vision.raster_semantics import interpret_raster_reviews
from diagex.vision.symbol_interpretation import (
    PerceivedObject,
    PerceptionBatch,
    _bind_knowledge,
    _bounded_knowledge_context,
    perceive_tile,
)
from tests.unit.test_legend_context_roles import source_entries
from tests.unit.test_raster_broad import arguments, response, symbol
from tests.unit.test_visual_knowledge import drawing, message


def context(entries):
    return {
        "identity": "original-retrieval",
        "knowledge_identity": "library-version",
        "profile_id": "profile",
        "profile_version": 2,
        "core_principles": ["Explicit drawing definitions take precedence."],
        "drawing_definitions": copy.deepcopy(entries),
        "references": [
            {
                "id": "kb",
                "version": 3,
                "reference_type": "symbol",
                "source": {},
                "text_slots": [{"example_marker": "(#)"}],
                "letter_matrix": {"rows": [], "notes": ["Preserve structured rules"]},
            }
        ],
    }


def assert_bounded(supplied, compact, original):
    assert supplied["drawing_definitions"] == compact
    assert "image_b64" not in json.dumps(supplied)
    assert supplied["references"] == original["references"]
    assert supplied["core_principles"] == original["core_principles"]
    assert supplied["identity"] != original["identity"]
    assert supplied["retrieval_identity"] == original["identity"]
    assert supplied == _bounded_knowledge_context(supplied, compact)
    assert [e["legend_entry_id"] for e in compact] == ["legend-row-7039a790096cb41d"]


def test_vlm_message_does_not_reintroduce_full_legend_through_knowledge():
    page, tile, view = drawing()
    entries = source_entries()
    ctx = context(entries)
    before = copy.deepcopy(ctx)
    calls = []
    proposal = {
        "kind": "opc",
        "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1},
        "printed_tag": "FT-101",
        "legend_entry_ids": [entries[2]["source_row_id"]],
        "legend_evidence": "Source outline matches the drawing connector",
    }
    client = SimpleNamespace(
        messages_create=lambda **kw: (
            calls.append(kw)
            or message("submit_pid_objects", {"candidate_results": [], "proposals": [proposal]})
        )
    )
    result = perceive_tile(
        client=client,
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=page,
        tile=tile,
        view_image=Image.new("RGB", (500, 400), "white"),
        view_info=view,
        ownership_bbox=tile.bbox,
        legend_summary=entries,
        step=1,
        candidates=[],
        page_context={"knowledge": ctx},
        reasoning_mode="disabled",
    )
    content = calls[0]["messages"][0]["content"]
    payload = json.loads(next(b["text"] for b in content if b["type"] == "text").split("\n")[-1])
    supplied = payload["page_overview_context"]["knowledge"]
    assert_bounded(supplied, payload["legend_entries"], ctx)
    assert sum(b["type"] == "image" for b in content) == 2  # Drawing + one selected legend.
    attrs = result.detections[0].attributes
    assert attrs["supplied_knowledge"]["context_identity"] == supplied["identity"]
    assert attrs["supplied_legend_entry_ids"] == [entries[2]["source_row_id"]]
    assert attrs["legend_matches"][0]["legend_entry_id"] == entries[2]["source_row_id"]
    assert "knowledge_match" not in attrs  # Supplying a reference never asserts a match.
    assert ctx == before


def test_cv_semantic_message_and_trace_use_the_same_bounded_context():
    page, tile, view = drawing()
    entries = source_entries()
    ctx = context(entries)
    before = copy.deepcopy(ctx)
    calls = []
    client = SimpleNamespace(
        messages_create=lambda **kw: (
            calls.append(kw)
            or message(
                "submit_raster_semantics",
                {
                    "results": [
                        {
                            "symbol_id": "symbol-0",
                            "decision": "unresolved",
                            "legend_entry_ids": [],
                            "reason": "Ambiguous source glyph",
                        }
                    ]
                },
            )
        )
    )
    reviews, audit = interpret_raster_reviews(
        reviews=[
            {
                "status": "uncertain",
                "bbox": {"x": 50, "y": 80, "w": 150, "h": 40},
                "object": {
                    "kind": "raster_symbol",
                    "attributes": {"broad_category": "connector", "geometry_basis": "cv_proposal"},
                },
            }
        ],
        client=client,
        cost_tracker=CostTracker(),
        page=page,
        tile=tile,
        view_image=Image.new("RGB", (500, 400), "white"),
        view_info=view,
        legend_entries=entries,
        policy=SymbolPerceptionConfig(),
        knowledge_context=ctx,
    )
    payload = json.loads(calls[0]["messages"][0]["content"][-1]["text"])
    assert_bounded(payload["knowledge"], payload["legend_entries"], ctx)
    assert audit["supplied_knowledge"]["context_identity"] == payload["knowledge"]["identity"]
    decision = reviews[0]["legend_interpretation"]
    assert decision["decision"] == "unresolved" and "knowledge_match" not in decision
    assert decision["supplied_knowledge"] == audit["supplied_knowledge"]
    assert ctx == before


def test_cv_broad_request_also_does_not_serialize_original_legend_pack():
    calls = []
    client = BroadRasterClient.__new__(BroadRasterClient)
    client.messages_create = lambda **kw: (
        calls.append(kw) or response({"raster_results": [symbol()], "discoveries": []})
    )
    args = arguments(client)
    ctx = context(source_entries())
    compact, _, _ = select_legend_context(source_entries(), [], [])
    args["legend_summary"] = compact  # Production raster_pipeline prepares this.
    args["page_context"]["knowledge"] = ctx
    outcome = perceive_broad_raster(**args)
    payload = json.loads(calls[0]["messages"][0]["content"][-1]["text"])
    assert_bounded(payload["knowledge"], payload["legend_entries"], ctx)
    trace = client.raster_review_attempts[0]
    assert trace["supplied_knowledge"]["context_identity"] == payload["knowledge"]["identity"]
    assert trace["supplied_legend_entry_ids"] == [compact[0]["legend_entry_id"]]
    assert outcome.detections == []  # Broad observations do not create engineering nodes.


def test_disabled_knowledge_stays_disabled_and_selected_content_changes_identity():
    assert _bounded_knowledge_context({}, source_entries()) == {}
    ctx = context(source_entries())
    first = _bounded_knowledge_context(ctx, [])
    second = _bounded_knowledge_context(ctx, [{"label": "Different drawing definition"}])
    assert first["identity"] != second["identity"]


@pytest.mark.parametrize("alias", ["drawing_ref", "target_sheet", "line_id", "canonical_tag"])
def test_discarded_legacy_placeholder_does_not_reappear_in_graph_attributes(alias):
    obj = PerceivedObject(kind="opc", candidate_id="source", attributes={alias: "(#)"})
    _bind_knowledge(PerceptionBatch(objects=[obj]), context([]))
    graph_attributes = obj.graph_attributes()
    assert alias not in graph_attributes
    assert graph_attributes["discarded_reference_placeholders"][alias] == "(#)"
    assert obj.candidate_id == "source"


def test_real_legacy_drawing_reference_is_preserved():
    obj = PerceivedObject(kind="opc", candidate_id="source", attributes={"target_sheet": "D-201"})
    _bind_knowledge(PerceptionBatch(objects=[obj]), context([]))
    assert obj.drawing_ref == "D-201" and obj.graph_attributes()["drawing_ref"] == "D-201"
