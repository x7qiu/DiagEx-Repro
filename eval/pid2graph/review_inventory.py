"""Replay saved broad validation observations through the production review store.

This is a software integration check, with no GraphML reads or model calls.
It does not certify semantic correctness or authorize final test access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagex.config import ScanConfig, TilingConfig
from diagex.review.detection import DetectionReviewStore, write_detection_bundle
from diagex.vision.evidence import PageEvidence
from diagex.vision.legend_models import LegendPack
from diagex.vision.loader import iter_pages, load
from diagex.vision.tiling import AspectAwareStrategy, tile

from .data import digest, save_new, sha256
from .epoch_train import sealed


def run(panel_path, run_path, output):
    panel = sealed(panel_path, "panel_sha256")
    root, output = Path(run_path), Path(output)
    cfg = json.loads((root / "config.json").read_text())
    if (cfg["split"] != "validation" or cfg["panel_sha256"] != panel["panel_sha256"]
            or cfg["variant"] != "broad_raster_recognition"):
        raise ValueError("Requires matching broad-recognition validation records")
    source_root = Path(__file__).resolve().parents[2] / "src/diagex"
    for name, expected in cfg["source_fingerprints"].items():
        if sha256(source_root / name) != expected:
            raise ValueError(f"Frozen inference source changed: {name}")
    if output.exists():
        raise ValueError("Use a new output directory to preserve previous review evidence")
    output.mkdir(parents=True)
    tc, sc = TilingConfig(**cfg["tiling"]), ScanConfig(**cfg["scan"])
    cases = []
    for row in panel["panels"]["validation"]:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Source image changed")
        case = root / digest(row["id"])[:16]
        predictions = json.loads((case / "predictions.json").read_text())
        if (predictions["config_sha256"] != digest(cfg)
                or predictions["image_sha256"] != row["image_sha256"]):
            raise ValueError("Prediction provenance differs")
        rendered = next(iter_pages(load(row["image"], tc, sc)))
        grid = tile(rendered, AspectAwareStrategy(tc.max_tokens_per_tile, tc.overlap_frac, tc.token_per_pixel))
        reviews, fingerprints = [], {}
        failures = 0
        for current in grid:
            path = case / f"{current.id}.json"
            record = json.loads(path.read_text())
            if record["config_sha256"] != digest(cfg):
                raise ValueError("Tile provenance differs")
            fingerprints[current.id] = sha256(path)
            failures += record["status"] != "complete"
            batch = record.get("outcome", {})
            if batch.get("objects"):
                raise ValueError("Broad observations unexpectedly include engineering objects")
            reviews.extend({**r, "page_index": 0, "tile_id": current.id}
                           for r in batch.get("candidate_reviews", []))
        if len(reviews) != len(predictions["predictions"]) or len(grid) != predictions["tiles"]:
            raise ValueError("Review inventory does not cover the saved predictions")
        evidence = PageEvidence(page_index=0, source_ref=row["image"], width=rendered.width,
            height=rendered.height, dpi=tc.target_dpi, effective_dpi=tc.target_dpi,
            is_scanned=True, role="pid")
        destination = output / digest(row["id"])[:16]
        destination.mkdir()
        write_detection_bundle(destination, source_hash=row["image_sha256"], pages=[evidence],
            detections=[], legend_pack=LegendPack(), candidates=[], reviews=reviews,
            per_page_status={0: "partial" if failures else "ok"})
        store = DetectionReviewStore(destination)
        state = store.public()
        if len(state["symbols"]) != len(reviews) or state["can_build"]:
            raise ValueError("Review inventory is incomplete or prematurely graph eligible")
        for actual, original in zip(state["symbols"], reviews, strict=True):
            if (actual["status"] != "pending" or actual["source_observation"] != original
                    or actual["detection"]["kind"] != "raster_symbol"
                    or actual["detection"]["bbox"] != original["bbox"]):
                raise ValueError("Review initialization changed the observation")
        snapshot = store.snapshot(0, draft=True)
        if snapshot["detections"] or len(snapshot["unresolved"]["items"]) != len(reviews):
            raise ValueError("Pending observations leaked into graph inputs or were dropped")
        cases.append({"id": row["id"], "observations": len(reviews), "tiles": len(grid),
                      "failed_tiles": failures, "tile_sha256": fingerprints})
    report = {"scope": "Software integration only; no semantic accuracy claim or final test access",
        "source_run": str(root.resolve()), "config_sha256": digest(cfg),
        "panel_sha256": panel["panel_sha256"], "drawings": len(cases),
        "observations": sum(c["observations"] for c in cases),
        "tiles": sum(c["tiles"] for c in cases), "graph_eligible": 0,
        "failed_tiles": sum(c["failed_tiles"] for c in cases), "cases": cases,
        "implementation_sha256": {name: sha256(source_root / name) for name in
                                  ("review/detection.py", "review/raster.py")},
        "verifier_sha256": sha256(__file__), "api_calls": 0}
    save_new(output / "verification.json", report)
    return {k: v for k, v in report.items() if k != "cases"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "run", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.panel, args.run, args.out), indent=2))


if __name__ == "__main__":
    main()
