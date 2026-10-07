"""Image-only hypotheses for cached VLM outputs; never reads annotation files."""
from __future__ import annotations

import math

import numpy as np

from .scoring import iou


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


def suppress_duplicates(predictions, threshold=0.5):
    kept, rejected = [], []
    for row in sorted(predictions, key=lambda p: (-p["confidence"], str(p["id"]))):
        owner = next((p for p in kept if p["label"] == row["label"] and iou(p["bbox"], row["bbox"]) >= threshold), None)
        if owner is None:
            kept.append(row)
        else:
            rejected.append({**row, "rejection_reason": "same_class_duplicate", "retained_id": owner["id"]})
    return kept, rejected


def transform(image, predictions, variant):
    rejected = []
    if variant in {"ink_guard", "ink_guard_nms"}:
        predictions, discarded = ink_guard(image, predictions)
        rejected.extend(discarded)
    if variant in {"nms", "ink_guard_nms"}:
        predictions, discarded = suppress_duplicates(predictions)
        rejected.extend(discarded)
    if variant not in {"ink_guard", "nms", "ink_guard_nms"}:
        raise ValueError("Unknown frozen postprocessing variant")
    return predictions, rejected


def apply_run(panel_path, source_dir, output, *, variant="ink_guard", selection=None):
    """Apply a frozen image-only hypothesis to saved development predictions."""
    import json
    import time
    from pathlib import Path

    from PIL import Image

    from .data import digest, save_new, sha256

    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Source panel changed")
    source_dir, output = Path(source_dir), Path(output)
    base = json.loads((source_dir / "config.json").read_text())
    if base["panel_sha256"] != panel["panel_sha256"]:
        raise ValueError("Source run belongs to another panel")
    if base["split"] == "test":
        from .selection import perception_signature, read_selection
        chosen = read_selection(selection, panel)
        if chosen["variant"] != variant or perception_signature(base) != chosen["inference_signature"]:
            raise ValueError("Final postprocessing input differs from selected baseline")
    if variant not in {"ink_guard", "nms", "ink_guard_nms"}:
        raise ValueError("Unknown frozen postprocessing variant")
    config = {"variant": variant, "model": base["model"], "split": base["split"],
              "panel_sha256": panel["panel_sha256"], "base_config_sha256": digest(base),
              "source_sha256": sha256(__file__), "minimum_ink_fraction": 0.001,
              "dark_pixel_threshold": 180, "same_class_nms_iou": 0.5}
    if base["split"] == "test":
        from .selection import postprocess_signature
        if postprocess_signature(config) != chosen["postprocess_signature"]:
            raise ValueError("Final postprocessing settings/source differ from selection")
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Postprocessing configuration changed")
    else:
        save_new(output / "config.json", config)
    for row in panel["panels"][base["split"]]:
        key = digest(row["id"])[:16]
        source_path = source_dir / key / "predictions.json"
        target = output / key / "predictions.json"
        if not source_path.exists():
            continue  # Missing drawings stay missing in the full-panel scorer.
        source_hash = sha256(source_path)
        if target.exists():
            if json.loads(target.read_text())["source_predictions_sha256"] != source_hash:
                raise ValueError("Cached source predictions changed")
            continue
        original = json.loads(source_path.read_text())
        if "ledger_request_ids" not in original:
            checkpoints = [json.loads(p.read_text()) for p in source_path.parent.glob("*.json") if p.name != "predictions.json"]
            original["ledger_request_ids"] = sorted({identity for r in checkpoints for identity in r.get("ledger_request_ids", [])})
        if original["config_sha256"] != digest(base) or original["image_sha256"] != row["image_sha256"]:
            raise ValueError("Source prediction configuration/image mismatch")
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Source image changed")
        started = time.monotonic()
        with Image.open(row["image"]) as image:
            predictions, rejected = transform(image, original["predictions"], variant)
        elapsed = time.monotonic() - started
        save_new(target, {**original, "config_sha256": digest(config), "predictions": predictions,
                          "rejected": rejected, "source_predictions_sha256": source_hash,
                          "runtime_seconds": original["runtime_seconds"] + elapsed,
                          "postprocessing_seconds": elapsed, "incremental_api_usd": 0})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="ink_guard", choices=["ink_guard", "nms", "ink_guard_nms"])
    parser.add_argument("--selection")
    args = parser.parse_args()
    apply_run(args.panel, args.source, args.out, variant=args.variant, selection=args.selection)
