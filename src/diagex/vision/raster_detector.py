"""Optional learned raster proposals; VLM interpretation remains a separate stage.

This module accepts images and trusted local weights only, never GraphML. Torch
is imported when the detector is constructed, so ordinary DiagEx use does not
require the optional learning runtime. Outputs are in original image pixels.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

from PIL import Image

CLASSES = ("general", "valve", "pump", "tank", "instrumentation", "arrow", "inlet/outlet")
ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RasterDetector:
    """The frozen proposal method, reusable across images without reloading weights."""

    def __init__(self, checkpoint, *, device="cpu", expected_sha256=None):
        try:
            import torch
            from torch import nn
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.ops.misc import FrozenBatchNorm2d
        except ImportError as exc:
            raise RuntimeError(
                "Raster proposals require the optional torch/torchvision runtime. "
                "See docs/PID2GRAPH_DETECTOR.md for the validated environment."
            ) from exc
        self.checkpoint_sha256 = _sha256(checkpoint)
        if expected_sha256 and self.checkpoint_sha256 != expected_sha256:
            raise ValueError("Raster detector checkpoint checksum changed")
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        training = weights["config"]
        if training["architecture"] != ARCHITECTURE or tuple(training["classes"]) != CLASSES:
            raise ValueError("Unsupported raster detector architecture or class mapping")
        if training["min_size"] != 640 or training["rpn_score_threshold"] != 0:
            raise ValueError("Checkpoint transform/RPN settings differ from the validated method")
        torch.set_num_threads(4)
        self.device = device
        self.training_manifest_sha256 = training["manifest_sha256"]
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, min_size=640, max_size=640,
            box_detections_per_img=100, box_score_thresh=0.05, rpn_score_thresh=0.0,
        )

        def freeze_norms(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=child.eps))
                else:
                    freeze_norms(child)

        freeze_norms(model)
        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
        self.model = model.to(device).eval()
        self.model.load_state_dict(weights["model"])

    @property
    def signature(self):
        return {"architecture": ARCHITECTURE, "detector_sha256": self.checkpoint_sha256,
                "training_manifest_sha256": self.training_manifest_sha256,
                "threshold": 0.15, "crop_size": 768, "stride": 576,
                "class_aware_nms_iou": 0.4, "device": self.device,
                "rpn_score_threshold": 0.0, "source_sha256": _sha256(__file__)}

    def predict(self, image):
        import torch
        from torchvision.ops import batched_nms
        from torchvision.transforms.functional import pil_to_tensor

        image = image.convert("RGB")
        boxes, scores, labels = [], [], []

        def origins(length):
            return sorted(set([*range(0, max(1, length - 768 + 1), 576), max(0, length - 768)]))

        with torch.inference_mode():
            for y in origins(image.height):
                for x in origins(image.width):
                    crop = image.crop((x, y, min(image.width, x + 768), min(image.height, y + 768)))
                    prediction = self.model([pil_to_tensor(crop).float().div_(255).to(self.device)])[0]
                    keep = prediction["scores"] >= 0.15
                    b = prediction["boxes"][keep].cpu()
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                    boxes.append(b)
                    scores.append(prediction["scores"][keep].cpu())
                    labels.append(prediction["labels"][keep].cpu())
        boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        indices = batched_nms(boxes, scores, labels, 0.4)
        return [{"id": f"detector-{int(i)}", "bbox": boxes[i].tolist(),
                 "label": CLASSES[int(labels[i]) - 1], "confidence": float(scores[i]),
                 "disposition": "review_proposal",
                 "attributes": {"geometry_basis": "supervised_raster_proposal"}} for i in indices]

    def predict_file(self, source):
        started = time.monotonic()
        content = Path(source).read_bytes()
        # Hash and decode the same bytes even if another process edits the file.
        with Image.open(io.BytesIO(content)) as image:
            size = list(image.size)
            predictions = self.predict(image)
        return {"image_sha256": hashlib.sha256(content).hexdigest(), "size": size,
                "coordinate_frame": "original_image_pixels", "detector": self.signature,
                "predictions": predictions, "runtime_seconds": time.monotonic() - started,
                "status": "proposals_require_interpretation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise ValueError("Output already exists; choose a new proposal artifact")
    detector = RasterDetector(args.checkpoint, device=args.device, expected_sha256=args.checkpoint_sha256)
    result = detector.predict_file(args.image)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(f"Saved {len(result['predictions'])} raster proposals to {output}")


if __name__ == "__main__":
    main()
