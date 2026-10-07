"""Source-bound, optional learned search hints for the existing VLM symbol stage."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image

from diagex.vision.raster_detector import CLASSES


def validate_guidance(payload, source, *, source_sha256=None):
    """Reject mismatched coordinates or approval-like records before any API use."""
    if not isinstance(payload, dict):
        raise ValueError("Raster guidance must be a JSON object")
    if Path(source).suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}:
        raise ValueError("Raster proposal guidance requires a single raster image")
    content = Path(source).read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if source_sha256 is not None and source_sha256 != actual_hash:
        raise ValueError("Source drawing changed during guidance preparation")
    with Image.open(io.BytesIO(content)) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Raster proposal guidance requires a single-frame image")
        size = list(image.size)
    if payload.get("image_sha256") != actual_hash or payload.get("size") != size:
        raise ValueError("Raster proposal source hash or dimensions do not match the drawing")
    if payload.get("coordinate_frame") != "original_image_pixels":
        raise ValueError("Raster proposals must use original image pixel coordinates")
    if payload.get("status") != "proposals_require_interpretation":
        raise ValueError("Raster guidance must contain unconfirmed proposals")
    detector = payload.get("detector", {})
    if not isinstance(detector, dict):
        raise ValueError("Raster guidance requires detector provenance")
    weight_hash = detector.get("detector_sha256", "")
    if not isinstance(weight_hash, str) or len(weight_hash) != 64 or any(c not in "0123456789abcdef" for c in weight_hash):
        raise ValueError("Raster guidance lacks a valid detector weight hash")
    if not isinstance(payload.get("predictions"), list):
        raise ValueError("Raster guidance predictions must be a list")
    identities = set()
    for row in payload["predictions"]:
        if not isinstance(row, dict):
            raise ValueError("Each raster proposal must be a JSON object")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Raster proposal IDs must be nonempty and unique")
        identities.add(identity)
        if row.get("disposition") != "review_proposal" or row.get("label") not in CLASSES:
            raise ValueError("Raster guidance requires broad-class review proposals")
        box, confidence = row.get("bbox"), row.get("confidence")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Raster proposal boxes require four pixel coordinates")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in [*box, confidence]):
            raise ValueError("Raster proposal coordinates and confidence must be finite numbers")
        a, b, c, d = box
        if not (0 <= a < c <= size[0] and 0 <= b < d <= size[1] and 0 <= confidence <= 1):
            raise ValueError("Raster proposal coordinates or confidence are out of bounds")


def load_guidance(path, source):
    content = Path(path).read_bytes()
    payload = json.loads(content)
    validate_guidance(payload, source)
    return {**payload, "artifact_sha256": hashlib.sha256(content).hexdigest()}


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def view_guidance(proposals, page_box, source_scale):
    """Map source-image coordinates into the actual VLM view, never GraphML."""
    sx, sy = source_scale
    x0, y0 = page_box.x * sx, page_box.y * sy
    width, height = page_box.w * sx, page_box.h * sy
    guides = []
    for p in sorted(proposals, key=lambda r: (-r["confidence"], r["id"])):
        a, b, c, d = p["bbox"]
        if c <= x0 or d <= y0 or a >= x0 + width or b >= y0 + height:
            continue
        left, top = max(a, x0), max(b, y0)
        right, bottom = min(c, x0 + width), min(d, y0 + height)
        guides.append({"proposal_id": p["id"], "suggested_category": p["label"],
                       "detector_confidence": p["confidence"],
                       "bbox_normalized": {"x": (left - x0) / width, "y": (top - y0) / height,
                                           "w": (right - left) / width, "h": (bottom - top) / height},
                       "partially_visible": [a, b, c, d] != [left, top, right, bottom]})
    return {"raster_proposal_guidance": {
        "origin": "supervised detector trained on separate training drawings",
        "instructions": "These locations and broad categories are fallible search hints. Verify each against original source ink, correct its class and bounds, and still search for missed symbols. They are not native candidates, ground truth, legend definitions, or accepted objects. Return visible symbols through proposals with a tight bbox in this image; do not put these proposal IDs in candidate_id.",
        "guides": guides[:80], "guides_omitted_by_limit": max(0, len(guides) - 80),
    }}
