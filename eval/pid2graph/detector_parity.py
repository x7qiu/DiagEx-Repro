"""Check reusable detector outputs against frozen validation predictions, without truth."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from diagex.vision.raster_detector import RasterDetector

from .data import digest, save_new, sha256


def checkpoint_identity(checkpoint, config, panel, export_verification=None):
    actual_hash = sha256(checkpoint)
    if export_verification is None:
        if actual_hash != config["detector_sha256"]:
            raise ValueError("Checkpoint differs from the reference detector")
        return actual_hash, None
    proof = json.loads(Path(export_verification).read_text())
    if (proof["source_checkpoint_sha256"] != config["detector_sha256"]
            or proof["export_checkpoint_sha256"] != actual_hash
            or sha256(proof["source_checkpoint"]) != config["detector_sha256"]):
        raise ValueError("Export provenance does not bind the source and exported checkpoints")
    if (proof["panel_sha256"] != panel["panel_sha256"]
            or proof["source_config"]["manifest_sha256"] != panel["manifest_sha256"]
            or proof["export_config"]["manifest_sha256"] != panel["manifest_sha256"]):
        raise ValueError("Export validation panel/manifest differs")
    if (proof["identical_model_tensors"] <= 0 or not proof["parity"]
            or not all(row["exact_forward_parity"] for row in proof["parity"])):
        raise ValueError("Export lacks tensor and forward verification")
    from . import checkpoint_export, detector
    if (proof["exporter_sha256"] != sha256(checkpoint_export.__file__)
            or proof["source_builder_sha256"] != sha256(detector.__file__)
            or proof["production_loader_sha256"] != sha256(Path(__file__).parents[2] / "src/diagex/vision/raster_detector.py")):
        raise ValueError("Export verification implementation changed")
    return actual_hash, {"path": str(Path(export_verification).resolve()),
                         "sha256": sha256(export_verification), "source_checkpoint_sha256": config["detector_sha256"],
                         "export_checkpoint_sha256": actual_hash}


def run(panel_path, reference, checkpoint, output, *, export_verification=None):
    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Panel changed")
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise ValueError("Use a new parity output directory")
    config = json.loads((reference / "config.json").read_text())
    if config["split"] != "validation" or config["panel_sha256"] != panel["panel_sha256"]:
        raise ValueError("Only the matching validation run can be used for parity")
    checkpoint_hash, export_provenance = checkpoint_identity(checkpoint, config, panel, export_verification)
    detector = RasterDetector(checkpoint, device=config["device"], expected_sha256=checkpoint_hash)
    started = time.monotonic()
    cases = []
    for row in panel["panels"]["validation"]:
        path = reference / digest(row["id"])[:16] / "predictions.json"
        expected = json.loads(path.read_text())
        actual = detector.predict_file(row["image"])
        if actual["image_sha256"] != row["image_sha256"] or expected["image_sha256"] != row["image_sha256"]:
            raise ValueError("Image changed")
        if expected["config_sha256"] != digest(config):
            raise ValueError("Reference prediction configuration changed")
        equal = actual["predictions"] == expected["predictions"]
        save_new(output / digest(row["id"])[:16] / "predictions.json", actual)
        cases.append({"id": row["id"], "exact_prediction_parity": equal,
                      "predictions": len(actual["predictions"]), "reference_sha256": sha256(path),
                      "actual_predictions_sha256": digest(actual["predictions"]),
                      "expected_predictions_sha256": digest(expected["predictions"])})
        print(row["id"], "exact parity" if equal else "MISMATCH", len(actual["predictions"]), flush=True)
    report = {"all_equal": all(r["exact_prediction_parity"] for r in cases), "cases": cases,
              "panel_sha256": panel["panel_sha256"], "reference_config": config,
              "reusable_detector": detector.signature, "runtime_seconds": time.monotonic() - started,
              "export_provenance": export_provenance,
              "ground_truth_read": False, "test_images_read": False, "api_calls": 0}
    save_new(output / "report.json", report)
    if not report["all_equal"]:
        raise AssertionError("Reusable detector differs from the frozen validation method")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "reference", "checkpoint", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--export-verification", help="Verified unchanged-tensor export of the reference checkpoint")
    args = parser.parse_args()
    run(args.panel, args.reference, args.checkpoint, args.out, export_verification=args.export_verification)
