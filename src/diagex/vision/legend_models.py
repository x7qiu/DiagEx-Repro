"""Legend data models shared across Phase 2 extractors (spec §7.2.2).

A `LegendEntry` is the atomic unit — one symbol with its label, class, and
optional image bytes. A `LegendPack` is the serialised collection that the
legend cache reads/writes to disk, keyed on the source plus an extractor
fingerprint so input, model, prompt-contract, and schema changes can force
re-extraction.
"""

from __future__ import annotations

import base64
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diagex.dexpi_schema import (
    EQUIPMENT_CLASS_KEYS,
    EQUIPMENT_CLASS_ROUTER_VALVE_KEY,
    INSTRUMENT_FUNCTION_KEYS,
    VALVE_TYPE_KEYS,
)
from diagex.vision.models import BBox, SourceView

SymbolStandard = Literal["isa-5.1", "iso-10628", "sama", "none"]
LEGEND_CACHE_SCHEMA_VERSION = "0.2.0"


def normalise_legend_role(entry: dict) -> dict:
    """Quarantine contradictory actuator/body definitions, including old caches.

    Applies to saved extraction results too. Preserve the original response;
    neither the claimed role nor the body class wins without source review.
    Customer overrides remain authoritative.
    """
    attrs = dict(entry.get("attributes") or {})
    if entry.get("source") != "legend_extracted" or attrs.get("symbol_role") != "actuator":
        return entry
    if entry.get("kind") != "equipment" or attrs.get("valve_type"):
        attrs.setdefault(
            "original_role_classification",
            json.dumps(
                {
                    "kind": entry.get("kind"),
                    "symbol_class": entry.get("symbol_class"),
                    "equipment_class": attrs.get("equipment_class"),
                    "valve_type": attrs.get("valve_type"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        attrs["row_status"] = "uncertain"
        attrs["classification_reason"] = (
            "Actuator role conflicts with valve-body classification; recheck the complete source definition"
        )
        return {**entry, "attributes": attrs}
    return entry


# DEXPI subset (spec §7.2.1) — the vocabulary the DexpiBuilder maps into.
# Sourced from :mod:`diagex.dexpi_schema` so the registry is the single source
# of truth; legend validation stays consistent with the phase-2 prompt and
# the DexpiBuilder dispatch table. The "valve" router key is surfaced here
# because legend entries are allowed to name it as an equipment_class
# (actuated valves drawn body-on-line); the phase-2 annotate schema routes
# valves via ``valve_type`` instead.
EquipmentClass = Literal[  # type: ignore[valid-type]
    EQUIPMENT_CLASS_KEYS + (EQUIPMENT_CLASS_ROUTER_VALVE_KEY,)  # noqa: F821
]

ValveType = Literal[VALVE_TYPE_KEYS]  # type: ignore[valid-type]  # noqa: F821

InstrumentFunction = Literal[INSTRUMENT_FUNCTION_KEYS]  # type: ignore[valid-type]  # noqa: F821


class LegendEntry(BaseModel):
    """One (label, class, optional image) triple extracted from a legend or a built-in library."""

    label: str  # agent-facing name, e.g. "FIC", "butterfly valve"

    @model_validator(mode="before")
    @classmethod
    def _explicit_symbol_role(cls, value):
        return normalise_legend_role(value) if isinstance(value, dict) else value

    description: str | None = None  # short human gloss
    symbol_class: str  # entry in the DEXPI/ISA vocabulary (§7.2.1)
    kind: Literal["equipment", "instrument", "line", "valve", "connector", "other"] = "other"
    standard: SymbolStandard | None = None  # "isa-5.1" etc.; None for customer-extracted
    image_b64: str | None = None  # optional PNG bytes, base64; None means text-only
    attributes: dict[str, str] = Field(default_factory=dict)
    source: Literal["built_in", "legend_extracted", "customer_override"] = "built_in"
    # Audit data for project-extracted thumbnails.  These fields intentionally
    # survive cache serialization so a human can trace an image back to the
    # page coordinates that produced it and future code can recrop it.
    source_page_index: int | None = None
    source_bbox: BBox | None = None
    source_label_bbox: BBox | None = None
    source_view: SourceView | None = None
    source_row_id: str | None = None
    crop_method: Literal["model_bbox", "native_text_paths"] | None = None
    crop_quality: (
        Literal[
            "accepted",
            "recovered",
            "omitted_abbreviation",
            "rejected_blank",
            "rejected_text_overlap",
            "rejected_duplicate",
            "unavailable",
        ]
        | None
    ) = None

    def image_bytes(self) -> bytes | None:
        if not self.image_b64:
            return None
        return base64.b64decode(self.image_b64)


def merge_source_row(first: LegendEntry, second: LegendEntry) -> LegendEntry:
    """Select a coherent source observation; keep class disagreements reviewable."""
    if not first.source_row_id or first.source_row_id != second.source_row_id:
        raise ValueError("Only observations of the same native source row can be combined")
    observations = [first, second]
    quality = {"accepted": 2, "recovered": 2}
    representative = max(
        observations,
        key=lambda e: (
            e.source == "customer_override",
            e.attributes.get("row_status") == "accept",
            quality.get(e.crop_quality, 0),
            bool(e.image_b64),
            (e.source_bbox.w * e.source_bbox.h) if e.source_bbox else 0,
            e.model_dump_json(),
        ),
    )
    if first.source == "customer_override" or second.source == "customer_override":
        return representative
    semantic_keys = ("equipment_class", "valve_type", "instrument_function", "symbol_role")
    signatures = {
        (e.kind, e.symbol_class, *(e.attributes.get(k) for k in semantic_keys))
        for e in observations
    }
    if len(signatures) == 1 and not any(e.attributes.get("row_conflicts") for e in observations):
        return representative
    attrs = dict(representative.attributes)
    attrs.update(
        row_status="uncertain",
        classification_reason="Conflicting observations of the same native legend row",
        row_conflicts=json.dumps(
            [e.model_dump(mode="json", exclude={"image_b64"}) for e in observations],
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return representative.model_copy(update={"attributes": attrs})


class LegendRegionCoverage(BaseModel):
    """Extraction coverage, distinct from human verification of legend rows."""

    page_index: int
    source_row_id: str | None = None
    bbox: BBox
    status: Literal["complete", "partial"]
    failure_kind: Literal["transport", "contract", "not_inspected", "ambiguity"] | None = None
    verification_passes: int = Field(default=1, ge=1, le=2)
    entry_count: int = 0
    reason: str = ""


class LegendPack(BaseModel):
    """A bundle of legend entries + metadata; what the cache serialises."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: str = LEGEND_CACHE_SCHEMA_VERSION
    source_hash: str = ""  # sha256 of the source bytes (§7.2.2)
    # Includes the extraction implementation, prompt contract, and model.
    # A matching source PDF alone is not sufficient to trust old crops.
    extractor_fingerprint: str = ""
    source_ref: str = ""  # "<stem>#pages=1,2" or "file:legend.pdf"
    standard: SymbolStandard = "isa-5.1"
    entries: list[LegendEntry] = Field(default_factory=list)
    notes: str = ""
    coverage: list[LegendRegionCoverage] = Field(default_factory=list)

    def merge(self, other: LegendPack) -> LegendPack:
        """Merge source rows by identity, preserving spatially repeated labels.

        Explicit overrides win by label; label-only fallbacks fill missing labels.

        Per spec: when built-in and extracted disagree, extracted wins. Callers
        pass `extracted.merge(built_in)` for that direction; callers building a
        presentation order call with the opposite order and use `LegendPack`
        purely for transport.
        """
        seen: set[str] = {e.label.strip().lower() for e in self.entries}
        overrides = {
            e.label.strip().lower() for e in self.entries if e.source == "customer_override"
        }
        row_ids = {e.source_row_id for e in self.entries if e.source_row_id}
        merged = list(self.entries)
        for e in other.entries:
            if e.label.strip().lower() in overrides:
                continue
            if e.source_row_id in row_ids:
                index = next(
                    i for i, row in enumerate(merged) if row.source_row_id == e.source_row_id
                )
                merged[index] = merge_source_row(merged[index], e)
                continue
            if not e.source_row_id and e.label.strip().lower() in seen:
                continue
            merged.append(e)
            seen.add(e.label.strip().lower())
            if e.source_row_id:
                row_ids.add(e.source_row_id)
        return LegendPack(
            schema_version=self.schema_version,
            source_hash=self.source_hash,
            extractor_fingerprint=self.extractor_fingerprint,
            source_ref=self.source_ref,
            standard=self.standard,
            entries=merged,
            coverage=[*self.coverage, *other.coverage],
            notes=(self.notes + (" | " if self.notes and other.notes else "") + other.notes).strip(
                " |"
            ),
        )


class LegendBudget(BaseModel):
    """Split of entries into cached few-shot vs lookup-tool fallback (spec §7.2.2)."""

    few_shot: list[LegendEntry] = Field(default_factory=list)
    lookup_only: list[LegendEntry] = Field(default_factory=list)
    budget_tokens: int = 0  # target ceiling for few_shot tokens
    used_tokens: int = 0  # rough estimate ≈ sum of entry costs
