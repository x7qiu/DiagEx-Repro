"""Text recognition backends, independent of symbol and graph assignment."""

from __future__ import annotations

from typing import Protocol

import fitz
from PIL import Image
from pydantic import BaseModel, Field

from diagex.vision.evidence import (
    PageEvidence,
    TextEvidence,
    _bbox_from_floats,
    _scale_factors,
    stable_evidence_id,
)
from diagex.vision.models import DiagramPage

TEXT_DETECTION_VERSION = "1.0.0"


class TextDetectionResult(BaseModel):
    page_index: int
    text_spans: list[TextEvidence] = Field(default_factory=list)
    backend: str = "native_pdf"
    warnings: list[str] = Field(default_factory=list)


class TextRecognizer(Protocol):
    def recognize(self, *, page: PageEvidence, image: Image.Image) -> list[TextEvidence]: ...


def detect_text(
    *, page: PageEvidence, image=None, recognizer: TextRecognizer | None = None
) -> TextDetectionResult:
    if recognizer is None:
        warnings = (
            ["No native text; configure an explicit raster text recognizer"]
            if not page.text_spans
            else []
        )
        return TextDetectionResult(
            page_index=page.page_index,
            text_spans=[s.model_copy(deep=True) for s in page.text_spans],
            warnings=warnings,
        )
    if image is None:
        raise ValueError("Raster text recognition requires a page image")
    spans = recognizer.recognize(page=page, image=image)
    if len({s.id for s in spans}) != len(spans):
        raise ValueError("Duplicate text evidence IDs")
    for span in spans:
        if (
            span.bbox.x < 0
            or span.bbox.y < 0
            or span.bbox.x2 > page.width
            or span.bbox.y2 > page.height
        ):
            raise ValueError("Recognized text outside the page")
    return TextDetectionResult(
        page_index=page.page_index, text_spans=spans, backend=type(recognizer).__name__
    )


def extract_pdf_text(pdf_page: fitz.Page, page: DiagramPage) -> list[TextEvidence]:
    sx, sy = _scale_factors(pdf_page, page)
    out: list[TextEvidence] = []
    words = pdf_page.get_text("words", sort=True) or []
    for raw in words:
        if len(raw) < 5:
            continue
        x0, y0, x1, y1 = (float(raw[idx]) for idx in range(4))
        text = str(raw[4]).strip()
        if not text:
            continue
        block = int(raw[5]) if len(raw) > 5 else None
        line = int(raw[6]) if len(raw) > 6 else None
        word = int(raw[7]) if len(raw) > 7 else None
        rotated = fitz.Rect(x0, y0, x1, y1) * pdf_page.rotation_matrix
        bbox = _bbox_from_floats(*rotated, sx=sx, sy=sy)
        out.append(
            TextEvidence(
                id=stable_evidence_id(
                    "txt", page.page_index, block, line, word, text, bbox.model_dump_json()
                ),
                text=text,
                bbox=bbox,
                block_index=block,
                line_index=line,
                word_index=word,
            )
        )
    return out
