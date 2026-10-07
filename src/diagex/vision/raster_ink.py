"""Optional source-ink filtering for unreviewed raster symbol observations."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_source(path):
    if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        raise ValueError("Raster ink filtering requires a single raster image")
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Raster ink filtering requires a single-frame image")


def ink_guard(image, predictions, minimum_fraction=0.001):
    """Reject effectively blank source boxes; do not treat ink as semantic proof."""
    gray = image.convert("L")
    kept, rejected = [], []
    for row in predictions:
        x0, y0, x1, y1 = row["bbox"]
        box = (max(0, math.floor(x0)), max(0, math.floor(y0)),
               min(gray.width, math.ceil(x1)), min(gray.height, math.ceil(y1)))
        if box[2] <= box[0] or box[3] <= box[1]:
            fraction = 0.0
        else:
            pixels = np.asarray(gray.crop(box))
            fraction = float((pixels < 180).mean())
        record = {**row, "source_ink_fraction": fraction}
        if fraction >= minimum_fraction:
            kept.append(record)
        else:
            rejected.append({**record, "rejection_reason": "nearly_blank_source_region"})
    return kept, rejected


def filter_observations(path, pages, detections, reviews):
    """Filter the same observation inventory as the benchmark in original pixels.

    Preserve original records and all discarded observations in the returned audit.
    Native candidate decisions are outside this raster-only filter's scope.
    """
    validate_source(path)
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as source:
        image = source.convert("L")
    dimensions = {p.page_index: (p.width, p.height) for p in pages}
    rows = []

    def append(identity, box, page_index, original):
        width, height = dimensions[page_index]
        sx, sy = image.width / width, image.height / height
        rows.append({"id": identity, "bbox": [box["x"] * sx, box["y"] * sy,
                     (box["x"] + box["w"]) * sx, (box["y"] + box["h"]) * sy],
                     "original": original})

    for i, detection in enumerate(detections):
        append(f"detection:{i}", detection.bbox.model_dump(), detection.page_index,
               detection.model_dump(mode="json"))
    for i, review in enumerate(reviews):
        if review.get("object") and review.get("bbox") and not review.get("candidate_id"):
            append(f"review:{i}", review["bbox"], review["page_index"], review)
    kept, rejected = ink_guard(image, rows)
    removed = {r["id"] for r in rejected}
    audit = {"source_sha256": hashlib.sha256(raw).hexdigest(),
             "implementation_sha256": implementation_sha256(),
             "coordinate_frame": "original_image_pixels", "minimum_ink_fraction": 0.001,
             "dark_pixel_threshold": 180, "inspected": len(rows),
             "kept": [{k: v for k, v in row.items() if k != "original"} for row in kept],
             "rejected": rejected, "note": "Ink presence does not establish symbol meaning or approval."}
    return ([d for i, d in enumerate(detections) if f"detection:{i}" not in removed],
            [r for i, r in enumerate(reviews) if f"review:{i}" not in removed], audit)
