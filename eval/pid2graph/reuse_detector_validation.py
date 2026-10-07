"""Seed broader detector validation with identical pilot hints, without inference.

Only validation metadata and previously completed source predictions are read.
The existing detector command then computes the remaining drawings. This avoids
changing hints for VLM calls that will be reused by reuse_validation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

from .data import SYMBOLS, digest, save_new, sha256
from .epoch_train import sealed
from .scoring import validate_prediction
from .selection import detector_signature


def seed(source_panel, target_panel, source_run, output, checkpoint, manifest_path):
    source_panel = sealed(source_panel, "panel_sha256")
    target_panel = sealed(target_panel, "panel_sha256")
    manifest = sealed(manifest_path, "manifest_sha256")
    source_run, output = Path(source_run), Path(output)
    if output.exists():
        raise FileExistsError("Detector validation reuse requires a new output directory")
    if (source_run / "stop-request.json").exists():
        raise ValueError("A stopped detector run cannot be reused")
    if not source_panel["manifest_sha256"] == target_panel["manifest_sha256"] == manifest["manifest_sha256"]:
        raise ValueError("Split manifest changed")
    config = json.loads((source_run / "config.json").read_text())
    weight_hash = sha256(checkpoint)
    if (config["variant"] != "supervised_detector" or config["split"] != "validation"
            or config["panel_sha256"] != source_panel["panel_sha256"]
            or config["detector_sha256"] != weight_hash):
        raise ValueError("Detector source panel, split or checkpoint differs")
    original = {r["id"]: r for r in source_panel["panels"]["validation"]}
    expanded = {r["id"]: r for r in target_panel["panels"]["validation"]}
    if (not original or len(original) != len(source_panel["panels"]["validation"])
            or len(expanded) != len(target_panel["panels"]["validation"])
            or not original.keys() <= expanded.keys()
            or any(row != expanded[identity] for identity, row in original.items())):
        raise ValueError("Target must contain the exact original validation drawings")
    frozen = {r["id"]: r for r in manifest["drawings"]}
    if len(frozen) != len(manifest["drawings"]):
        raise ValueError("Duplicate manifest identities")
    for identity, row in expanded.items():
        reference = frozen.get(identity)
        if (reference is None or reference["split"] != "validation"
                or reference["image_sha256"] != row["image_sha256"] or reference["size"] != row["size"]):
            raise ValueError("Target drawing does not belong to the frozen validation split")
    rebound = {**copy.deepcopy(config), "panel_sha256": target_panel["panel_sha256"]}
    if detector_signature(config) != detector_signature(rebound):
        raise ValueError("Validation reuse changed the detector method")
    records = []
    for identity, row in original.items():
        path = source_run / digest(identity)[:16] / "predictions.json"
        if not path.is_file():
            raise ValueError("Source detector drawing is incomplete")
        raw = json.loads(path.read_text())
        if (raw["id"] != identity or raw["image_sha256"] != row["image_sha256"]
                or sha256(row["image"]) != row["image_sha256"]
                or raw["config_sha256"] != digest(config) or raw["detector_sha256"] != weight_hash
                or raw["errors"] or raw["reserved_or_charged_usd"] != 0
                or not isinstance(raw["runtime_seconds"], (int, float))
                or not math.isfinite(raw["runtime_seconds"]) or raw["runtime_seconds"] < 0):
            raise ValueError("Source prediction provenance, completion or runtime differs")
        predictions = raw["predictions"]
        if len({p["id"] for p in predictions}) != len(predictions):
            raise ValueError("Duplicate detector proposal IDs")
        for prediction in predictions:
            validate_prediction(prediction)
            x1, y1, x2, y2 = prediction["bbox"]
            if (prediction["label"] not in SYMBOLS or prediction["disposition"] != "review_proposal"
                    or not 0 <= x1 < x2 <= row["size"][0] or not 0 <= y1 < y2 <= row["size"][1]):
                raise ValueError("Detector hint class, review status or bounds are invalid")
        copied = copy.deepcopy(raw)
        copied["config_sha256"] = digest(rebound)
        copied["validation_reuse"] = {"source_record": str(path.resolve()), "source_sha256": sha256(path),
            "source_config_sha256": digest(config), "new_inference": False}
        records.append((digest(identity)[:16], copied))
    provenance = {"source_panel_sha256": source_panel["panel_sha256"],
        "target_panel_sha256": target_panel["panel_sha256"], "manifest_sha256": manifest["manifest_sha256"],
        "source_run": str(source_run.resolve()), "source_config_sha256": digest(config),
        "target_config_sha256": digest(rebound), "checkpoint_sha256": weight_hash,
        "reused_drawings": sorted(original), "unattempted_drawings": sorted(expanded.keys() - original.keys()),
        "new_inference": 0, "api_calls": 0, "test_inputs_read": False, "graphml_read": False,
        "implementation_sha256": sha256(__file__),
        "scope": "Validation-to-validation expansion only; exact predictions and original measured runtimes preserved. Existing detector runner must compute remaining drawings before broader scoring or VLM reuse."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="detector-reuse-") as temporary:
        staging = Path(temporary) / "prepared"
        save_new(staging / "config.json", rebound)
        for name, record in records:
            save_new(staging / name / "predictions.json", record)
        save_new(staging / "validation-reuse.json", provenance)
        staging.rename(output)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-panel", "target-panel", "source-run", "out", "checkpoint", "manifest"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = seed(args.source_panel, args.target_panel, args.source_run, args.out, args.checkpoint, args.manifest)
    print(json.dumps({"reused_drawings": len(result["reused_drawings"]),
                      "unattempted_drawings": len(result["unattempted_drawings"]), "new_inference": 0}))
