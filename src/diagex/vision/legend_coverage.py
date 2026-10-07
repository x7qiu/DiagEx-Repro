"""Separate unresolved explanatory material from missing symbol definitions."""

from __future__ import annotations

import json
import re

from diagex.vision.legend_models import LegendPack

LEGEND_GATE_VERSION = "1.0.0"


def _explanatory_basis(entry) -> str | None:
    # A missing answer is not evidence of explanatory content. Require both
    # native source wording and an earlier valid rejection of this exact row.
    # A later conflicting observation must still block the gate.
    if entry.source != "legend_extracted" or entry.crop_method != "native_text_paths":
        return None
    attrs = entry.attributes
    if attrs.get("row_status") != "uncertain":
        return None
    prior = attrs.get("previous_rejection_reason", "").strip()
    if not prior:
        return None
    try:
        observations = json.loads(attrs.get("classification_evidence", "null"))
    except (TypeError, ValueError):
        return None
    if observations != []:
        return None
    label = entry.label.strip().casefold()
    reason = prior.casefold()
    numbering_label = bool(re.match(r"^[=＝]\s*\S", label)) or any(
        term in label for term in ("编号说明", "位号示例", "编号示例", "tag numbering", "tag example")
    )
    numbering_reason = any(
        term in reason for term in ("编号", "位号命名", "numbering", "tag example", "tag equivalence")
    )
    metadata_label = any(
        term in label for term in ("copyright", "property of", "版权所有", "grade of qualification")
    )
    metadata_reason = any(
        term in reason for term in ("版权", "资质", "标题栏", "copyright", "metadata", "title block")
    )
    if (numbering_label and numbering_reason) or (metadata_label and metadata_reason):
        return prior
    return None


def legend_coverage_findings(pack: LegendPack) -> list[dict]:
    """Project coverage into gate decisions without changing saved observations.

    Legacy records without sufficient evidence retain the strict gate. A
    non-blocking explanatory row remains partial and is never an interpretation
    rule; the caller's existing uncertain-entry filter still applies.
    """
    entries = {}
    for entry in pack.entries:
        if entry.source_row_id:
            entries.setdefault((entry.source_page_index, entry.source_row_id), []).append(entry)
    findings = []
    for coverage in pack.coverage:
        if coverage.status == "complete":
            continue
        matches = entries.get((coverage.page_index, coverage.source_row_id), [])
        basis = _explanatory_basis(matches[0]) if len(matches) == 1 else None
        explanatory = basis is not None
        blocking = coverage.failure_kind != "ambiguity" and not explanatory
        findings.append({
            "type": "legend_coverage",
            "label": "说明性内容待核查" if explanatory else "图例内容待核查",
            "source_row_id": coverage.source_row_id,
            "page_index": coverage.page_index,
            "bbox": coverage.bbox.model_dump(mode="json"),
            "status": coverage.status,
            "failure_kind": coverage.failure_kind,
            "blocks_symbol_extraction": blocking,
            "content_role": "explanatory" if explanatory else "unknown",
            "reason": (
                "说明性内容的复核未完成；保留原始证据，不作为符号定义，不阻止其他符号提取。"
                if explanatory else
                "图例定义尚不明确；保留待核查，不作为确定的识别依据。"
                if not blocking else
                "图例检查未完成，需处理后继续符号提取。"
            ),
            "source_reason": coverage.reason,
            "explanatory_evidence": basis,
        })
    return findings
