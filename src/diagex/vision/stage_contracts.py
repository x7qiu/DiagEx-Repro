"""Versioned, serializable requests for independently replayable extraction stages."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from diagex.vision.evidence import PageEvidence
from diagex.vision.legend_models import LegendPack
from diagex.vision.line_detection import LineDetectionResult
from diagex.vision.models import BBox, ReconciledNode, Tile
from diagex.vision.symbol_candidates import SymbolCandidate


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Asset(Contract):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class StageRequest(Contract):
    schema_version: Literal["1.0"] = "1.0"
    stage: Literal[
        "symbol_detection",
        "text_detection",
        "line_detection",
        "symbol_interpretation",
        "text_assignment",
        "connection_inference",
        "line_interpretation",
    ]
    backend: str
    inputs: dict[str, Any]
    knowledge: dict = Field(default_factory=dict)
    assets: dict[str, Asset] = Field(default_factory=dict)
    # Public model routing only. Credentials always come from the caller/environment.
    model: str | None = None
    transport: str | None = None


class PageInput(Contract):
    page: PageEvidence


class SymbolInput(PageInput):
    device: Literal["cpu", "mps", "cuda"] = "cpu"


class TextInput(PageInput):
    max_tokens: int = Field(default=6000, gt=0)


class AssignmentInput(Contract):
    knowledge: dict = Field(default_factory=dict)
    nodes: list[ReconciledNode]
    pages: list[PageEvidence]
    legend_pack: LegendPack | None = None

    @model_validator(mode="after")
    def identities(self):
        page_ids = {p.page_index for p in self.pages}
        if len(page_ids) != len(self.pages) or len({n.id for n in self.nodes}) != len(self.nodes):
            raise ValueError("Duplicate page or symbol IDs")
        if any(n.page_index not in page_ids for n in self.nodes):
            raise ValueError("Every symbol must belong to an input page")
        return self


class SymbolInterpretationInput(PageInput):
    tile: Tile
    ownership_bbox: BBox
    candidates: list[SymbolCandidate] = Field(default_factory=list)
    legend_entries: list[dict[str, Any]] = Field(default_factory=list)
    page_context: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def page_coordinates(self):
        if self.tile.page_index != self.page.page_index or self.tile.image is not None:
            raise ValueError("Tile must refer to this page; images belong in hashed assets")
        for box in (self.tile.bbox, self.ownership_bbox, *(c.bbox for c in self.candidates)):
            if box.x < 0 or box.y < 0 or box.x2 > self.page.width or box.y2 > self.page.height:
                raise ValueError("Symbol input geometry lies outside the page")
        if any(c.page_index != self.page.page_index for c in self.candidates):
            raise ValueError("Candidate belongs to another page")
        if len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate symbol candidate IDs")
        return self


class ConnectionInput(PageInput):
    knowledge_context: dict = Field(default_factory=dict)
    nodes: list[ReconciledNode]
    lines: LineDetectionResult
    # Native model classes are validated by the selected backend, avoiding an
    # import of the VLM implementation from the CV-only contracts module.
    legend_line_profile: dict[str, Any] | None = None
    topology: dict[str, Any] | None = None
    all_nodes: list[ReconciledNode] | None = None
    pages: list[PageEvidence] | None = None
    legend_entries: list[dict[str, Any]] = Field(default_factory=list)
    visual_evidence: dict[str, Any] | None = None
    max_tokens: int = Field(default=16000, gt=0)

    @model_validator(mode="after")
    def page_coordinates(self):
        if self.lines.page_index != self.page.page_index:
            raise ValueError("Line evidence belongs to another page")
        if any(n.page_index != self.page.page_index for n in self.nodes):
            raise ValueError("Local nodes belong to another page")
        return self
