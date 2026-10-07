"""Export a validated detector checkpoint without changing its model tensors.

An export is a compatibility artifact, not a model-selection decision. Only
validation images are read for parity; no GraphML or test input is opened.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

from diagex.vision.raster_detector import ARCHITECTURE, CLASSES, RasterDetector

from .data import digest, save_new, sha256
from .epoch_train import sealed


def compatible_config(config, *, detector_source, observed_rpn_threshold):
    """Recover omitted metadata only when the archived builder is verified."""
    if config.get("software_smoke_only"):
        raise ValueError("Software smoke checkpoints are excluded")
    if (config.get("architecture") != ARCHITECTURE
            or tuple(config.get("classes", [])) != CLASSES or config.get("min_size") != 640):
        raise ValueError("Unsupported architecture, classes or transform")
    if observed_rpn_threshold != 0:
        raise ValueError("Builder RPN threshold differs from the production method")
    if "rpn_score_threshold" not in config:
        source = Path(detector_source).resolve()
        if config.get("source_hashes", {}).get(str(source)) != sha256(source):
            raise ValueError("Missing RPN metadata requires the exact frozen detector source")
    elif config["rpn_score_threshold"] != observed_rpn_threshold:
        raise ValueError("Recorded RPN metadata differs from the builder")
    result = copy.deepcopy(config)
    result["rpn_score_threshold"] = observed_rpn_threshold
    return result


def verify_report(report, panel, checkpoint_hash, config):
    if report.get("split") != "validation" or not report.get("complete") or report.get("stopped"):
        raise ValueError("A completed validation report is required")
    if report["config_sha256"] != digest(report["config"]):
        raise ValueError("Validation configuration checksum differs")
    if (report["config"]["variant"] != "supervised_detector"
            or report["config"]["detector_sha256"] != checkpoint_hash):
        raise ValueError("Validation report belongs to another checkpoint")
    if (report["panel_sha256"] != panel["panel_sha256"]
            or report["config"]["panel_sha256"] != panel["panel_sha256"]
            or report["manifest_sha256"] != panel["manifest_sha256"]
            or config["manifest_sha256"] != panel["manifest_sha256"]):
        raise ValueError("Validation panel or manifest differs")
    expected = {r["id"] for r in panel["panels"]["validation"]}
    cases = report["cases"]
    if (not expected or len(cases) != len(expected) or {r["id"] for r in cases} != expected
            or any(r["status"] != "complete" for r in cases)):
        raise ValueError("Incomplete validation drawing coverage")


def export(checkpoint, report_path, panel_path, output):
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    from . import detector

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new export directory")
    panel = sealed(panel_path, "panel_sha256")
    report = json.loads(Path(report_path).read_text())
    checkpoint_hash = sha256(checkpoint)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading")
    verify_report(report, panel, checkpoint_hash, weights["config"])
    torch.set_num_threads(4)
    reference = detector.build_model(pretrained=False).cpu().eval()
    reference.load_state_dict(weights["model"])
    config = compatible_config(weights["config"], detector_source=detector.__file__,
                               observed_rpn_threshold=reference.rpn.score_thresh)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".pending-", dir=output.parent))
    try:
        target = stage / "detector.pt"
        payload = {k: weights[k] for k in ("model", "step", "epoch") if k in weights}
        payload["config"] = config
        torch.save(payload, target)
        exported = RasterDetector(target, device="cpu", expected_sha256=sha256(target))
        original_state, exported_state = reference.state_dict(), exported.model.state_dict()
        if (original_state.keys() != exported_state.keys()
                or any(not torch.equal(v, exported_state[k]) for k, v in original_state.items())):
            raise AssertionError("Export changed model tensors")
        # Include each collection, using ordinary source crops with no truth.
        rows = {}
        for row in panel["panels"]["validation"]:
            rows.setdefault(row["collection"], row)
        parity = []
        with torch.inference_mode():
            for row in rows.values():
                source = Path(row["image"])
                if sha256(source) != row["image_sha256"]:
                    raise ValueError("Validation image changed")
                with Image.open(source) as image:
                    image = image.convert("RGB")
                    x, y = max(0, (image.width - 768) // 2), max(0, (image.height - 768) // 2)
                    crop = image.crop((x, y, min(image.width, x + 768), min(image.height, y + 768)))
                tensor = pil_to_tensor(crop).float().div_(255)
                before, after = reference([tensor])[0], exported.model([tensor])[0]
                exact = before.keys() == after.keys() and all(torch.equal(v, after[k]) for k, v in before.items())
                if not exact:
                    raise AssertionError("Export changed CPU forward predictions")
                parity.append({"id": row["id"], "image_sha256": row["image_sha256"],
                               "crop": [x, y, x + crop.width, y + crop.height],
                               "predictions": len(before["boxes"]), "exact_forward_parity": exact})
        proof = {"source_checkpoint": str(checkpoint), "source_checkpoint_sha256": checkpoint_hash,
                 "export_checkpoint_sha256": sha256(target), "source_config": weights["config"],
                 "export_config": config, "validation_report_sha256": sha256(report_path),
                 "panel_sha256": panel["panel_sha256"], "identical_model_tensors": len(original_state),
                 "source_builder_sha256": sha256(detector.__file__), "exporter_sha256": sha256(__file__),
                 "production_loader_sha256": sha256(Path(__file__).parents[2] / "src/diagex/vision/raster_detector.py"),
                 "parity": parity, "ground_truth_read": False, "test_images_read": False, "api_calls": 0,
                 "selection_status": "Compatibility verification only; does not select or deploy a model",
                 "scope": "Exact state-dict equality and CPU forward parity on one validation crop per collection. Full-drawing integration parity remains a separate check."}
        save_new(stage / "verification.json", proof)
        stage.rename(output)
        return proof
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "report", "panel", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    proof = export(args.checkpoint, args.report, args.panel, args.out)
    print(json.dumps({k: proof[k] for k in ("export_checkpoint_sha256", "identical_model_tensors", "parity")}, indent=2))


if __name__ == "__main__":
    main()
