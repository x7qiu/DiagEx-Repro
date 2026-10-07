"""Canonical Pydantic data models shared across diagex.

All coordinates are page-pixel integers at the page's effective DPI (spec §5.1);
origin top-left, x right, y down.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Identifier aliases (spec §5.6)
# ---------------------------------------------------------------------------

TileId = str         # "p{page_index}-r{row}-c{col}"
AnnotationId = str   # uuid4 hex
EntityId = str       # reconciler-assigned, stable within a document

Kind = Literal[
    "equipment",
    "instrument",
    "line",
    "connection",
    "text",
    "note",
    "opc",
]

LineType = Literal[
    "process",
    "signal_electric",
    "signal_pneumatic",
    "instrument_capillary",
    "electrical_power",
    "other",
]

Confidence = Literal["high", "medium", "low"]

SourceView = Literal["overview", "tile", "region"]


def _new_annotation_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned bounding box in page-pixel space."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


class Point(BaseModel):
    """A point on a line endpoint. Agent provides *_local; runtime projects *_global."""

    x_local: float
    y_local: float
    x_global: float = 0.0
    y_global: float = 0.0
    on_view_edge: bool = False


# ---------------------------------------------------------------------------
# Source documents (spec §5.1 / §5.6)
# ---------------------------------------------------------------------------


class DiagramPage(BaseModel):
    """One rasterised page from a DiagramSource."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_index: int
    image: Any = None            # PIL.Image.Image; materialised lazily
    width: int
    height: int
    dpi: float                   # DPI we rendered at
    effective_dpi: float         # source DPI for scans, else == dpi
    is_scanned: bool
    rotation_deg: float = 0.0
    source_ref: str              # "<pdf-stem>#page=<n>" — never absolute (spec §6.5)


class DiagramSource(BaseModel):
    """A loaded PDF or image; pages stream lazily."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    kind: Literal["pdf", "image"]
    pages: Any = None            # Iterable[DiagramPage] — kept opaque for pydantic
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tiling (spec §5.2)
# ---------------------------------------------------------------------------


class Tile(BaseModel):
    """One tile carved out of a page image."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: TileId
    page_index: int
    bbox: BBox
    image: Any = None            # PIL.Image.Image
    overlap_neighbors: list[TileId] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Annotations (spec §5.4)
# ---------------------------------------------------------------------------


class Annotation(BaseModel):
    id: AnnotationId = Field(default_factory=_new_annotation_id)
    page_index: int
    kind: Kind
    source_view: SourceView
    tile_id: TileId | None = None
    bbox_local: BBox
    bbox_global: BBox
    endpoints: list[Point] = Field(default_factory=list)
    line_type: LineType | None = None
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: Confidence
    source_quote: str | None = None


# ---------------------------------------------------------------------------
# Reconciled graph (spec §5.6)
# ---------------------------------------------------------------------------


class ReconciledNode(BaseModel):
    id: EntityId
    kind: Kind
    label: str
    bbox_global: BBox
    page_index: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: Confidence
    # Literal text read from the drawing.  Keep this separate from ``label``
    # (which may be normalised for matching) so downstream review can tell
    # evidence from interpretation.  Optional for backwards compatibility
    # with existing graph.json files.
    source_quote: str | None = None
    alternate_readings: list[str] = Field(default_factory=list)
    source_annotation_ids: list[AnnotationId] = Field(default_factory=list)
    # Evidence-v2 provenance.  Optional defaults keep legacy graph.json and
    # review sessions loadable without migration.
    source_evidence_ids: list[str] = Field(default_factory=list)
    system_confidence: float | None = None
    system_confidence_level: Confidence | None = None


class ReconciledEdge(BaseModel):
    id: EntityId
    from_node: EntityId
    to_node: EntityId
    line_type: LineType | None = None
    polyline_global: list[tuple[int, int]] = Field(default_factory=list)
    cross_sheet: bool = False
    confidence: Confidence
    source_annotation_ids: list[AnnotationId] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    system_confidence: float | None = None
    system_confidence_level: Confidence | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("from_node", "to_node")
    @classmethod
    def _endpoint_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("edge endpoint must be a non-empty EntityId; unsnappable polylines should be recorded in conflicts instead")
        return v


class EquipmentAssembly(BaseModel):
    """Logical equipment scope. Membership never implies a pipe connection."""

    id: str
    page_index: int
    bbox_global: BBox
    label: str | None = None
    label_candidates: list[str] = Field(default_factory=list)
    member_node_ids: list[str] = Field(default_factory=list)
    parent_assembly_id: str | None = None
    status: Literal["supported", "uncertain", "conflicting"] = "uncertain"
    source_text_ids: list[str] = Field(default_factory=list)
    boundary_segments: list[tuple] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReconciledGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: str = "0.1.0"
    source_path: str              # relative stem, not absolute (spec §6.5)
    nodes: list[ReconciledNode] = Field(default_factory=list)
    edges: list[ReconciledEdge] = Field(default_factory=list)
    assemblies: list[EquipmentAssembly] = Field(default_factory=list)
    text_bindings: list[dict[str, Any]] = Field(default_factory=list)
    dangling_opcs: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    per_page_status: dict[
        int, Literal["ok", "partial", "cost_exhausted", "error"]
    ] = Field(
        default_factory=dict
    )
