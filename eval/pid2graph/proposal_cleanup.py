"""Source-only containment cleanup for learned proposals; never an acceptance step."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import digest, save_new, sha256


def remove_contained(predictions):
    def area(row):
        a, b, c, d = row["bbox"]
        return (c - a) * (d - b)

    kept, rejected = [], []
    for small in sorted(predictions, key=lambda p: (-area(p), p["id"])):
        a, b, c, d = small["bbox"]
        owner = None
        for large in kept:
            if small["label"] != large["label"] or area(small) >= 0.8 * area(large):
                continue
            if large["confidence"] < max(0.25, 0.5 * small["confidence"]):
                continue
            e, f, g, h = large["bbox"]
            overlap = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
            if overlap >= 0.9 * area(small):
                owner = large
                break
        if owner is None:
            kept.append(small)
        else:
            rejected.append({**small, "rejection_reason": "same_class_fragment_inside_larger_proposal", "retained_id": owner["id"]})
    return kept, rejected


def run(panel_path, source_dir, output):
    source_dir, output = Path(source_dir), Path(output)
    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Panel changed")
    base = json.loads((source_dir / "config.json").read_text())
    if base["panel_sha256"] != panel["panel_sha256"] or base["variant"] != "supervised_detector":
        raise ValueError("Input is not the matching detector run")
    if base["split"] == "test":
        raise ValueError("Development-only cleanup; final use requires sealed pipeline selection")
    config = {**base, "cleanup": {"source_sha256": sha256(__file__), "minimum_containment": 0.9,
               "maximum_area_ratio": 0.8, "minimum_owner_confidence": 0.25, "minimum_relative_owner_confidence": 0.5}}
    save_new(output / "config.json", config)
    for case in panel["panels"][base["split"]]:
        key = digest(case["id"])[:16]
        source = source_dir / key / "predictions.json"
        original = json.loads(source.read_text())
        if original["config_sha256"] != digest(base) or original["image_sha256"] != case["image_sha256"]:
            raise ValueError("Detector source/configuration changed")
        started = time.monotonic()
        kept, rejected = remove_contained(original["predictions"])
        elapsed = time.monotonic() - started
        save_new(output / key / "predictions.json", {**original, "predictions": kept,
                 "config_sha256": digest(config), "rejected": rejected, "runtime_seconds": original["runtime_seconds"] + elapsed,
                 "cleanup_seconds": elapsed, "source_predictions_sha256": sha256(source)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "source", "out"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    run(args.panel, args.source, args.out)
