"""Review-only raster observations, separate from graph-eligible detections."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from diagex.vision.models import BBox
from diagex.vision.raster_detector import CLASSES


class RasterReviewObservation(BaseModel):
    id: str
    page_index: int
    tile_id: str
    kind: Literal["raster_symbol"] = "raster_symbol"
    label: str = ""
    bbox: BBox
    confidence: Literal["high", "medium", "low"]
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_broad_evidence(self):
        if self.attributes.get("broad_category") not in CLASSES:
            raise ValueError("Raster observation requires a supported broad category")
        if self.attributes.get("requires_legend_interpretation") is not True:
            raise ValueError("Unclassified raster observations require legend interpretation")
        return self


def require_semantic_classification(detection):
    """A broad class alone cannot approve an engineering node."""
    kind = detection["kind"]
    attributes = detection.get("attributes", {})
    fields = {
        "equipment": ("equipment_class", "valve_type"),
        "instrument": ("instrument_class", "instrument_function"),
        "opc": ("connector_type",),
    }.get(kind, ())
    if not any(isinstance(attributes.get(key), str) and attributes[key].strip() for key in fields):
        raise ValueError("Classify the raster symbol using the reviewed legend before confirming it")
