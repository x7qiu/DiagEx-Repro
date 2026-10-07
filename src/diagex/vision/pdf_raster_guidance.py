"""Source-bound PDF page proposals using the unchanged production renderer.

The PDF stays the extraction source: rendered hints never replace native text
or vector geometry. This module reads no annotation files and makes no API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import fitz

from diagex.config import Config
from diagex.vision.loader import iter_pages, load
from diagex.vision.raster_detector import CLASSES, RasterDetector

FORMAT = "pdf_page_proposals_v1"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render_settings(cfg):
    root = Path(__file__).parent
    return {"target_dpi": cfg.tiling.target_dpi, "max_page_dim_px": cfg.tiling.max_page_dim_px,
            "scan": asdict(cfg.scan), "renderer_sources": {
                name: file_hash(root / name) for name in ("loader.py", "scan_preprocess.py")}}


def pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(json.dumps(list(rgb.size)).encode() + b"\0RGB\0" + rgb.tobytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_pdf_guidance(payload, source, cfg, *, source_sha256=None):
    if not isinstance(payload, dict) or Path(source).suffix.lower() != ".pdf" or payload.get("format") != FORMAT:
        raise ValueError("Expected PDF page proposal format and a PDF source")
    actual = file_hash(source)
    if payload.get("source_sha256") != actual or (source_sha256 is not None and actual != source_sha256):
        raise ValueError("PDF proposal source checksum differs")
    if cfg.scan.deskew or cfg.symbol_perception.workflow != "fixed":
        raise ValueError("PDF proposals require fixed inspection with deskew disabled")
    if payload.get("render_settings") != render_settings(cfg):
        raise ValueError("PDF proposal render settings or renderer source differ")
    if (payload.get("coordinate_frame") != "rendered_pdf_page_pixels"
            or payload.get("status") != "proposals_require_interpretation"
            or not isinstance(payload.get("detector"), dict)
            or not _hash(payload["detector"].get("detector_sha256"))):
        raise ValueError("PDF proposals require unconfirmed rendered-page hints with detector provenance")
    with fitz.open(source) as pdf:
        page_count = len(pdf)
    if payload.get("page_count") != page_count or not isinstance(payload.get("pages"), list) or not payload["pages"]:
        raise ValueError("PDF proposal page inventory differs or is empty")
    pages = set()
    for page in payload["pages"]:
        if not isinstance(page, dict):
            raise ValueError("PDF proposal page must be an object")
        index, size = page.get("page_index"), page.get("size")
        if type(index) is not int or not 0 <= index < page_count or index in pages:
            raise ValueError("PDF proposal page indices must be valid and unique")
        pages.add(index)
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("PDF proposal page dimensions must be positive integers")
        if not _hash(page.get("render_sha256")) or not isinstance(page.get("predictions"), list):
            raise ValueError("PDF proposal page requires rendered pixels and predictions")
        identities = set()
        for prediction in page["predictions"]:
            if not isinstance(prediction, dict):
                raise ValueError("PDF proposal prediction must be an object")
            identity = prediction.get("id")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("PDF proposal IDs must be nonempty and unique within a page")
            identities.add(identity)
            if prediction.get("disposition") != "review_proposal" or prediction.get("label") not in CLASSES:
                raise ValueError("PDF proposals require unconfirmed broad classes")
            box, confidence = prediction.get("bbox"), prediction.get("confidence")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("PDF proposal box must contain four coordinates")
            if any(type(v) not in {float, int} or not math.isfinite(v) for v in [*box, confidence]):
                raise ValueError("PDF proposal coordinates and confidence must be finite")
            if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1] and 0 <= confidence <= 1):
                raise ValueError("PDF proposal coordinates or confidence are out of bounds")


def load_pdf_guidance(path, source, cfg):
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    validate_pdf_guidance(payload, source, cfg)
    return {**payload, "artifact_sha256": hashlib.sha256(raw).hexdigest()}


def page_proposals(payload, rendered_page):
    """Verify the exact pixels before attaching any hint to an inference view."""
    row = next((p for p in payload["pages"] if p["page_index"] == rendered_page.page_index), None)
    if row is None:
        return None  # Omitted pages keep the existing baseline/native path.
    if (row["size"] != [rendered_page.width, rendered_page.height]
            or row["render_sha256"] != pixel_hash(rendered_page.image)
            or rendered_page.rotation_deg != 0):
        raise ValueError("PDF rendered pixels or coordinates differ from the detector proposal page")
    return row["predictions"]


def route_guided_pages(pages, payload, *, legend_pages=None):
    """Inspect explicitly guided scans that the sparse-first-page rule skipped.

    Empty native evidence cannot establish that a raster page is a cover.
    Preserve positive page classifications and explicit whole-page legends.
    This only changes routing metadata; source text, paths and pixels stay intact.
    """
    selected = {p["page_index"] for p in payload["pages"]}
    excluded = set(legend_pages or [])
    changes = []
    for page in pages:
        if (page.page_index not in selected or page.page_index in excluded
                or not page.is_scanned or page.text_spans or page.paths
                or page.role != "cover" or page.role_confidence == "high"):
            continue
        changes.append({"page_index": page.page_index, "previous_role": page.role,
                        "previous_reason": page.role_reason, "role": "pid"})
        page.role = "pid"
        page.role_confidence = "low"
        page.role_reason = "No native evidence; source-bound PDF guidance requires raster inspection"
        page.fail_open = True
    return changes


def generate(source, detector, cfg, *, page_indices=None):
    source = Path(source)
    if source.suffix.lower() != ".pdf" or cfg.scan.deskew or cfg.symbol_perception.workflow != "fixed":
        raise ValueError("PDF proposals require a PDF, fixed inspection and deskew disabled")
    source_hash = file_hash(source)
    with fitz.open(source) as pdf:
        page_count = len(pdf)
    selected = list(range(page_count)) if page_indices is None else list(page_indices)
    if (not selected or len(selected) != len(set(selected))
            or any(type(i) is not int or not 0 <= i < page_count for i in selected)):
        raise ValueError("Select valid, unique zero-based PDF page indices")
    pages = []
    for page in iter_pages(load(source, cfg.tiling, cfg.scan)):
        if page.page_index not in selected:
            continue
        started = time.monotonic()
        predictions = detector.predict(page.image)
        pages.append({"page_index": page.page_index, "size": [page.width, page.height],
            "render_sha256": pixel_hash(page.image), "dpi": page.dpi, "effective_dpi": page.effective_dpi,
            "is_scanned": page.is_scanned, "predictions": predictions, "runtime_seconds": time.monotonic() - started})
        if len(pages) == len(selected):
            break
    value = {"format": FORMAT, "source_sha256": source_hash, "page_count": page_count,
        "coordinate_frame": "rendered_pdf_page_pixels", "status": "proposals_require_interpretation",
        "detector": detector.signature, "render_settings": render_settings(cfg), "pages": pages,
        "omitted_page_policy": "Use existing baseline/native perception; no detector coverage is claimed"}
    if {p["page_index"] for p in pages} != set(selected):
        raise ValueError("Not all selected PDF pages were rendered")
    validate_pdf_guidance(value, source, cfg, source_sha256=source_hash)
    return value


def implementation_sha256():
    return file_hash(__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pdf", "checkpoint", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--pages", type=int, nargs="+", help="Zero-based page indices; default all pages")
    parser.add_argument("--max-page-dim", type=int)
    parser.add_argument("--dpi", type=int)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise ValueError("Choose a new proposal artifact; existing outputs are immutable")
    cfg = Config()
    if args.max_page_dim is not None:
        cfg.tiling.max_page_dim_px = args.max_page_dim
    if args.dpi is not None:
        cfg.tiling.target_dpi = args.dpi
    if min(cfg.tiling.max_page_dim_px, cfg.tiling.target_dpi) <= 0:
        raise ValueError("Render dimensions and DPI must be positive")
    detector = RasterDetector(args.checkpoint, device=args.device, expected_sha256=args.checkpoint_sha256)
    result = generate(args.pdf, detector, cfg, page_indices=args.pages)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(f"Saved {len(result['pages'])} PDF proposal pages to {output}")


if __name__ == "__main__":
    main()
