"""Deterministic reconstruction of native-text abbreviation tables.

Vision extraction is useful for drawn symbols, but it is a poor fit for a
vector-PDF table whose cells already exist as positioned text.  This module
finds known abbreviation-table headings, reconstructs repeated code/meaning
columns, and emits auditable rows plus project legend entries.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from diagex.vision.evidence import PageEvidence, TextEvidence
from diagex.vision.legend_models import LegendEntry, LegendPack
from diagex.vision.models import BBox

AbbreviationSection = Literal["general", "instrument_type"]

_HEADINGS: dict[AbbreviationSection, tuple[str, ...]] = {
    "general": ("通用缩写", "general abbreviations", "general abbreviation"),
    "instrument_type": (
        "仪表类型缩写",
        "instrument type abbreviations",
        "instrument abbreviations",
    ),
}
_CODE_RE = re.compile(r"^\*?[A-Z][A-Z0-9./()\-]{0,11}$")
_VARIABLES = {
    "P": "pressure",
    "T": "temperature",
    "F": "flow",
    "L": "level",
    "A": "analysis",
    "D": "density",
    "S": "speed",
    "V": "vibration",
    "Z": "position",
}


class AbbreviationRow(BaseModel):
    id: str
    section: AbbreviationSection
    section_title: str
    page_index: int
    printed_code: str
    canonical_code: str
    raw_description: str
    code_bbox: BBox
    description_bbox: BBox
    row_bbox: BBox
    source_ids: list[str] = Field(default_factory=list)
    kind: Literal["equipment", "instrument", "line", "valve", "connector", "other"]
    symbol_class: str
    attributes: dict[str, str] = Field(default_factory=dict)
    confidence: Literal["high", "medium", "low"] = "high"
    warnings: list[str] = Field(default_factory=list)


class AbbreviationInventory(BaseModel):
    schema_version: str = "1.0.0"
    rows: list[AbbreviationRow] = Field(default_factory=list)
    sections: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    def to_legend_pack(self, *, base: LegendPack) -> LegendPack:
        """Convert rows to project entries; instrument-table definitions win."""
        grouped: dict[str, list[AbbreviationRow]] = {}
        for row in self.rows:
            grouped.setdefault(row.canonical_code.casefold(), []).append(row)
        entries: list[LegendEntry] = []
        for rows in grouped.values():
            selected = sorted(
                rows,
                key=lambda row: (row.section != "instrument_type", row.page_index, row.row_bbox.y),
            )[0]
            alternatives = sorted({row.raw_description for row in rows})
            attrs = dict(selected.attributes)
            attrs.update(
                {
                    "legend_kind": "abbreviation",
                    "abbreviation_section": selected.section,
                    "printed_code": selected.printed_code,
                    "raw_description": selected.raw_description,
                    "native_text_reconstructed": "true",
                }
            )
            if len(alternatives) > 1:
                attrs["alternate_project_definitions"] = " | ".join(alternatives)
                attrs["abbreviation_conflict"] = "true"
            entries.append(
                LegendEntry(
                    label=selected.canonical_code,
                    description=selected.raw_description,
                    symbol_class=selected.symbol_class,
                    kind=selected.kind,
                    standard=None,
                    attributes=attrs,
                    source="legend_extracted",
                    source_page_index=selected.page_index,
                    source_bbox=selected.row_bbox,
                    source_label_bbox=selected.code_bbox,
                    crop_quality="omitted_abbreviation",
                )
            )
        return LegendPack(
            source_hash=base.source_hash,
            extractor_fingerprint=base.extractor_fingerprint,
            source_ref=base.source_ref,
            standard=base.standard,
            entries=entries,
            notes=f"native abbreviation tables: {len(self.rows)} rows",
        )


def extract_abbreviation_tables(pages: list[PageEvidence]) -> AbbreviationInventory:
    rows: list[AbbreviationRow] = []
    sections: list[dict[str, Any]] = []
    for page in pages:
        for section, titles in _HEADINGS.items():
            headings = [
                span
                for span in page.text_spans
                if _normalise(span.text) in {_normalise(title) for title in titles}
            ]
            for heading in headings:
                found, diagnostic = _extract_section(page, heading, section)
                rows.extend(found)
                sections.append(diagnostic)

    rows.sort(key=lambda row: (row.page_index, row.section, row.row_bbox.y, row.row_bbox.x))
    duplicate_codes = Counter(row.canonical_code for row in rows)
    conflicts = {
        code: sorted({row.raw_description for row in rows if row.canonical_code == code})
        for code, count in duplicate_codes.items()
        if count > 1
    }
    conflicts = {code: values for code, values in conflicts.items() if len(values) > 1}
    summary = {
        "section_count": len(sections),
        "row_count": len(rows),
        "general_count": sum(row.section == "general" for row in rows),
        "instrument_type_count": sum(row.section == "instrument_type" for row in rows),
        "unique_code_count": len({row.canonical_code for row in rows}),
        "conflicting_code_count": len(conflicts),
        "conflicts": conflicts,
        "warning_count": sum(len(row.warnings) for row in rows),
    }
    return AbbreviationInventory(rows=rows, sections=sections, summary=summary)


def _extract_section(
    page: PageEvidence,
    heading: TextEvidence,
    section: AbbreviationSection,
) -> tuple[list[AbbreviationRow], dict[str, Any]]:
    h = max(8, heading.bbox.h)
    if section == "general":
        left, right = heading.bbox.x - 12 * h, heading.bbox.x2 + 20 * h
    else:
        left, right = heading.bbox.x - 28 * h, heading.bbox.x2 + 32 * h
    candidates = [
        span
        for span in page.text_spans
        if heading.bbox.y2 <= span.bbox.y
        and left <= span.bbox.x <= right
        and _CODE_RE.fullmatch(_compact_code(span.text))
    ]
    clusters = _x_clusters(candidates, tolerance=max(20, int(h * 1.4)))
    dense: list[list[TextEvidence]] = []
    for cluster in clusters:
        sequence = _dense_vertical_sequence(cluster, heading, max_gap=4 * h)
        if len(sequence) >= 3:
            dense.append(sequence)
    dense.sort(key=lambda group: _median_x(group))

    rows: list[AbbreviationRow] = []
    column_xs = [_median_x(group) for group in dense]
    for index, group in enumerate(dense):
        code_x = column_xs[index]
        next_x = column_xs[index + 1] if index + 1 < len(column_xs) else None
        previous_gap = code_x - column_xs[index - 1] if index else None
        typical_gap = next_x - code_x if next_x else previous_gap or 16 * h
        x_limit = int(next_x - h) if next_x else int(code_x + typical_gap - h)
        for code_span in group:
            description_spans = _same_row_description(
                page.text_spans,
                code_span=code_span,
                x_limit=min(page.width, x_limit),
            )
            if not description_spans:
                continue
            raw_description = " ".join(span.text.strip() for span in description_spans).strip()
            if not raw_description or _CODE_RE.fullmatch(_compact_code(raw_description)):
                continue
            printed = _compact_code(code_span.text)
            canonical = printed.lstrip("*")
            desc_bbox = _union([span.bbox for span in description_spans])
            row_bbox = _union([code_span.bbox, desc_bbox])
            kind, symbol_class, attrs = _semantics(canonical, raw_description, section)
            row_id = "abbr-" + hashlib.sha256(
                f"{page.page_index}|{section}|{canonical}|{row_bbox.x}|{row_bbox.y}".encode()
            ).hexdigest()[:16]
            rows.append(
                AbbreviationRow(
                    id=row_id,
                    section=section,
                    section_title=heading.text.strip(),
                    page_index=page.page_index,
                    printed_code=printed,
                    canonical_code=canonical,
                    raw_description=raw_description,
                    code_bbox=code_span.bbox,
                    description_bbox=desc_bbox,
                    row_bbox=row_bbox,
                    source_ids=[code_span.id, *(span.id for span in description_spans)],
                    kind=kind,
                    symbol_class=symbol_class,
                    attributes=attrs,
                )
            )
    return rows, {
        "section": section,
        "title": heading.text.strip(),
        "page_index": page.page_index,
        "heading_bbox": heading.bbox.model_dump(),
        "column_count": len(dense),
        "row_count": len(rows),
        "column_x": column_xs,
        "status": "ok" if rows else "not_reconstructed",
    }


def _x_clusters(spans: list[TextEvidence], tolerance: int) -> list[list[TextEvidence]]:
    clusters: list[list[TextEvidence]] = []
    for span in sorted(spans, key=lambda item: item.bbox.x):
        match = next(
            (cluster for cluster in clusters if abs(_median_x(cluster) - span.bbox.x) <= tolerance),
            None,
        )
        if match is None:
            clusters.append([span])
        else:
            match.append(span)
    return clusters


def _dense_vertical_sequence(
    spans: list[TextEvidence], heading: TextEvidence, max_gap: int
) -> list[TextEvidence]:
    ordered = sorted(spans, key=lambda span: span.bbox.y)
    starts = [i for i, span in enumerate(ordered) if span.bbox.y <= heading.bbox.y + 7 * heading.bbox.h]
    if not starts:
        return []
    best: list[TextEvidence] = []
    for start in starts:
        current = [ordered[start]]
        for span in ordered[start + 1 :]:
            gap = span.bbox.y - current[-1].bbox.y
            if gap > max_gap:
                break
            if gap >= max(3, heading.bbox.h // 3):
                current.append(span)
        if len(current) > len(best):
            best = current
    return best


def _same_row_description(
    spans: list[TextEvidence], *, code_span: TextEvidence, x_limit: int
) -> list[TextEvidence]:
    tolerance = max(code_span.bbox.h, 8)
    candidates = [
        span
        for span in spans
        if span.id != code_span.id
        and code_span.bbox.x2 < span.bbox.x < x_limit
        and abs((span.bbox.y + span.bbox.h / 2) - (code_span.bbox.y + code_span.bbox.h / 2))
        <= tolerance * 0.7
    ]
    return sorted(candidates, key=lambda span: span.bbox.x)


def _semantics(
    code: str, description: str, section: AbbreviationSection
) -> tuple[str, str, dict[str, str]]:
    attrs: dict[str, str] = {}
    position_instrument = "阀位" in description or "valve position" in description.casefold()
    if ("阀" in description or "valve" in description.casefold()) and not position_instrument:
        lowered = description.casefold()
        valve_type = "other"
        for marker, resolved in (
            ("止回", "check"),
            ("check", "check"),
            ("球阀", "ball"),
            ("butterfly", "butterfly"),
            ("蝶阀", "butterfly"),
            ("闸阀", "gate"),
            ("截止阀", "globe"),
            ("安全阀", "safety_relief"),
            ("泄压阀", "safety_relief"),
            ("调节阀", "control"),
            ("控制阀", "control"),
            ("control valve", "control"),
        ):
            if marker in lowered:
                valve_type = resolved
                break
        attrs["valve_type"] = valve_type
        return "valve", "valve", attrs
    if section == "instrument_type":
        if code == "DEV":
            return "other", "abbreviation", attrs
        attrs["instrument_function"] = _instrument_function(code, description)
        attrs["measured_variable"] = "position" if position_instrument else _VARIABLES.get(code[:1], "other")
        return "instrument", attrs["instrument_function"], attrs
    return "other", "abbreviation", attrs


def _instrument_function(code: str, description: str = "") -> str:
    # The printed definition outranks letters in parenthesized qualifiers and
    # project-specific command codes (e.g. HST is a start command, not a transmitter).
    if "转换器" in description or "converter" in description.casefold():
        return "signal_converter"
    if "命令" in description or "选择器" in description:
        return "switch"
    if "开关或报警" in description:
        return "unclassified_instrument"
    for markers, function in (
        (("控制", "调节器", "controller"), "controller"),
        (("变送器", "transmitter"), "transmitter"),
        (("开关", "switch"), "switch"),
        (("报警", "alarm"), "alarm"),
        (("记录", "recorder"), "recorder"),
        (("指示", "压力表", "物位表", "物位计", "视镜", "indicator"), "indicator"),
        (("测量元件", "分析元件", "element"), "element"),
    ):
        if any(marker in description.casefold() for marker in markers):
            return function
    if description and any(marker in description for marker in ("探测器", "液位,温度,密度", "偏离")):
        return "unclassified_instrument"
    suffix = code.split("(", 1)[0][1:]
    if "C" in suffix:
        return "controller"
    if "T" in suffix:
        return "transmitter"
    if "S" in suffix:
        return "switch"
    if "A" in suffix:
        return "alarm"
    if "R" in suffix:
        return "recorder"
    if "I" in suffix:
        return "indicator"
    if "E" in suffix:
        return "element"
    return "unclassified_instrument"


def _compact_code(text: str) -> str:
    return re.sub(r"\s+", "", text.strip()).upper()


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _median_x(spans: list[TextEvidence]) -> int:
    values = sorted(span.bbox.x for span in spans)
    return values[len(values) // 2]


def _union(boxes: list[BBox]) -> BBox:
    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x2 for box in boxes)
    y1 = max(box.y2 for box in boxes)
    return BBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))
