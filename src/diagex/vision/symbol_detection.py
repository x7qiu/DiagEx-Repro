"""Physical symbol localization. This module never calls a language model."""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image
from pydantic import BaseModel, Field

from diagex.vision.evidence import PageEvidence
from diagex.vision.symbol_candidates import SymbolCandidate, symbol_candidates

SYMBOL_DETECTION_VERSION = "1.0.0"


class CVDetector(Protocol):
    @property
    def signature(self) -> dict[str, Any]: ...

    def predict(self, image: Image.Image) -> list[dict[str, Any]]: ...


class SymbolDetectionResult(BaseModel):
    page_index: int
    native_candidates: list[SymbolCandidate] = Field(default_factory=list)
    raster_proposals: list[dict[str, Any]] = Field(default_factory=list)
    detector: dict[str, Any] | None = None
    status: str = "proposals_require_interpretation"


def detect_symbols(*, page: PageEvidence, image=None, detector: CVDetector | None = None):
    result = SymbolDetectionResult(
        page_index=page.page_index, native_candidates=symbol_candidates(page)
    )
    if detector is not None:
        if image is None or image.size != (page.width, page.height):
            raise ValueError("Detector image must use the page coordinate frame")
        result.raster_proposals = detector.predict(image)
        result.detector = detector.signature
    return result
