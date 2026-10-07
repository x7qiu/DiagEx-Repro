"""Extract native/raster line geometry without assigning graph endpoints."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from diagex.vision.evidence import PageEvidence, PathEvidence, stable_evidence_id
from diagex.vision.models import BBox

LINE_DETECTION_VERSION = "1.0.0"


class LineDetectionResult(BaseModel):
    page_index: int
    paths: list[PathEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raster_fallback_used: bool = False


def detect_lines(*, page: PageEvidence, image=None, raster_extractor=None) -> LineDetectionResult:
    paths = sorted((p.model_copy(deep=True) for p in page.paths), key=lambda p: p.id)
    result = LineDetectionResult(page_index=page.page_index, paths=paths)
    if page.is_scanned or sum(p.primitive == "line" for p in paths) < 5:
        extra, warning = (raster_extractor or extract_raster_paths)(page=page, image=image)
        result.paths.extend(extra)
        result.raster_fallback_used = bool(extra)
        if warning:
            result.warnings.append(warning)
    return result


def extract_raster_paths(
    *, page: PageEvidence, image: Any | None
) -> tuple[list[PathEvidence], str | None]:
    if image is None:
        return [], "raster fallback requested but no rendered page image is available"
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except ImportError:
        return [], (
            "raster line extraction skipped; install the optional 'vision' extra "
            "for opencv-python-headless"
        )

    array = np.asarray(image.convert("L"))
    inverted = cv2.bitwise_not(array)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    minimum = max(20, int(min(page.width, page.height) * 0.015))
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=math.pi / 180,
        threshold=max(25, minimum // 2),
        minLineLength=minimum,
        maxLineGap=max(6, minimum // 4),
    )
    if lines is None:
        return [], "OpenCV raster fallback found no candidate lines"

    out: list[PathEvidence] = []
    # OpenCV builds expose either (N, 1, 4) or (N, 4). Normalize the
    # singleton dimension before applying the candidate limit.
    for index, raw in enumerate(np.asarray(lines).reshape(-1, 4)[:5000]):
        x1, y1, x2, y2 = (int(value) for value in raw)
        points = [(x1, y1), (x2, y2)]
        bbox = BBox(
            x=min(x1, x2),
            y=min(y1, y2),
            w=max(1, abs(x2 - x1)),
            h=max(1, abs(y2 - y1)),
        )
        out.append(
            PathEvidence(
                id=stable_evidence_id("ras", page.page_index, index, points),
                page_index=page.page_index,
                points=points,
                bbox=bbox,
                origin="raster_cv",
                primitive="raster_line",
                stroke_width=1.0,
            )
        )
    return out, None
