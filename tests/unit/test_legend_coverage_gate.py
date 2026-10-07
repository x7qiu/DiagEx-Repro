"""Explanatory recheck failures stay visible without blocking symbols."""

import json

import pytest

from diagex.extractors.pid_evidence import _legend_prerequisite_error
from diagex.vision.legend_coverage import legend_coverage_findings
from diagex.vision.legend_models import LegendEntry, LegendPack, LegendRegionCoverage
from diagex.vision.models import BBox


def example(label="= PG 010101", prior="位号示例用于解释编号规则，而非图例符号。"):
    return LegendPack(
        entries=[LegendEntry(
            label=label, symbol_class="unclassified_equipment", source="legend_extracted",
            source_page_index=2, source_row_id="numbering-example", crop_method="native_text_paths",
            attributes={"row_status": "uncertain", "classification_evidence": "[]",
                        "previous_rejection_reason": prior},
        )],
        coverage=[LegendRegionCoverage(
            page_index=2, source_row_id="numbering-example", bbox=BBox(x=10, y=20, w=30, h=40),
            status="partial", failure_kind="contract", verification_passes=2, entry_count=1,
            reason="Return exactly one decision for this supplied row_id.",
        )],
    )


def test_missing_numbering_recheck_is_nonblocking_but_remains_uncertain():
    pack = example()
    before = pack.model_dump_json()
    assert _legend_prerequisite_error(pack, "explicit_pages") is None
    finding, = legend_coverage_findings(pack)
    assert not finding["blocks_symbol_extraction"]
    assert finding["content_role"] == "explanatory"
    assert finding["status"] == "partial" and finding["failure_kind"] == "contract"
    assert finding["page_index"] == 2 and finding["bbox"]["x"] == 10
    assert finding["explanatory_evidence"] == pack.entries[0].attributes["previous_rejection_reason"]
    assert "不作为符号定义" in finding["reason"]
    assert pack.model_dump_json() == before


@pytest.mark.parametrize("label,reason", [
    ("气缸执行机构", "not a standalone symbol"),
    ("主工艺管线 次工艺管线", "numbering example"),
    ("DCS installation", "metadata"),
    ("= PG 010101", ""),
    ("= PG 010101", "ambiguous glyph"),
    ("未知符号", "编号说明"),
])
def test_unknown_or_real_symbol_failures_still_block(label, reason):
    assert _legend_prerequisite_error(example(label, reason), "explicit_pages")


@pytest.mark.parametrize("observations", [
    "null", "invalid JSON",
    json.dumps([{"decision": "accept", "kind": "instrument"}]),
    json.dumps([{"decision": "uncertain"}]),
])
def test_conflicting_or_missing_observation_history_cannot_bypass_gate(observations):
    pack = example()
    pack.entries[0].attributes["classification_evidence"] = observations
    assert _legend_prerequisite_error(pack, "explicit_pages")


def test_whole_page_failure_and_other_rows_cannot_inherit_exemption():
    pack = example()
    pack.coverage.append(pack.coverage[0].model_copy(update={
        "source_row_id": None, "failure_kind": "not_inspected",
    }))
    assert "1 source legend" in _legend_prerequisite_error(pack, "explicit_pages")
    pack.coverage[1].source_row_id = "valve-row"
    assert _legend_prerequisite_error(pack, "explicit_pages")


def test_native_provenance_and_exact_page_are_required():
    pack = example()
    pack.entries[0].source_page_index = 0
    assert _legend_prerequisite_error(pack, "explicit_pages")
    pack.entries[0].source_page_index = 2
    pack.entries[0].crop_method = "model_bbox"
    assert _legend_prerequisite_error(pack, "explicit_pages")


def test_copyright_explanation_and_existing_ambiguity_remain_nonblocking():
    assert _legend_prerequisite_error(example("Copyright", "copyright metadata"), "explicit_pages") is None
    pack = example("Valve", "")
    pack.coverage[0].failure_kind = "ambiguity"
    assert _legend_prerequisite_error(pack, "explicit_pages") is None
    assert _legend_prerequisite_error(pack, "fallback_builtin(error=failed)")


def test_legacy_without_explanatory_evidence_keeps_gate():
    pack = example()
    pack.entries = []
    restored = LegendPack.model_validate_json(pack.model_dump_json())
    assert _legend_prerequisite_error(restored, "explicit_pages")
