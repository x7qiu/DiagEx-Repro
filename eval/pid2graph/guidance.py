"""Advisory raster detections for the VLM; these are never native vector evidence."""
from __future__ import annotations

import json
from pathlib import Path

from .data import digest, sha256


def load_proposals(directory, panel, split):
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    if config["variant"] != "supervised_detector" or config["split"] != split or config["panel_sha256"] != panel["panel_sha256"]:
        raise ValueError("Detector proposals do not match inference panel")
    proposals, source_hashes, runtimes = {}, {}, {}
    for row in panel["panels"][split]:
        path = root / digest(row["id"])[:16] / "predictions.json"
        record = json.loads(path.read_text())
        if record["image_sha256"] != row["image_sha256"] or record["config_sha256"] != digest(config):
            raise ValueError("Detector proposal image/configuration mismatch")
        proposals[row["id"]] = record["predictions"]
        source_hashes[row["id"]] = sha256(path)
        runtimes[row["id"]] = record["runtime_seconds"]
    return proposals, {"detector_config": config, "proposal_files_sha256": source_hashes,
                       "detector_runtime_seconds": runtimes,
                       "guidance_source_sha256": sha256(__file__), "maximum_guides_per_view": 80}


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
