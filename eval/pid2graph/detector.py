"""Train-only supervised proposals on local hardware; source-only inference.

This detector proposes broad symbol categories. It does not interpret legends,
read tags, establish engineering facts, or replace VLM interpretation.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from PIL import Image

from .data import SYMBOLS, digest, read_graph, save_new, sha256


def prepare_training(manifest_path, output, *, crop_size=768, crops_per_drawing=5):
    manifest = json.loads(Path(manifest_path).read_text())
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Training data output must be new")
    (output / "images").mkdir(parents=True)
    sources = [r for r in manifest["drawings"] if r["split"] == "train"]
    samples = []
    for index, row in enumerate(sources):
        root = Path(manifest["dataset_root"])
        image_path, graph_path = root / row["image"], root / row["graph"]
        if sha256(image_path) != row["image_sha256"] or sha256(graph_path) != row["graph_sha256"]:
            raise ValueError("Training source changed")
        nodes = [n for n in read_graph(graph_path)["nodes"] if n["label"] in SYMBOLS]
        rng = random.Random(digest(["train-crops-v1", row["id"]]))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        classes = sorted({n["label"] for n in nodes})
        for crop_index in range(crops_per_drawing):
            if classes and crop_index < crops_per_drawing - 1:
                category = rng.choice(classes)
                anchor = rng.choice([n for n in nodes if n["label"] == category])
                cx = (anchor["bbox"][0] + anchor["bbox"][2]) / 2
                cy = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
                x = round(cx - crop_size * rng.uniform(0.25, 0.75))
                y = round(cy - crop_size * rng.uniform(0.25, 0.75))
            else:
                x = rng.randint(0, max(0, image.width - crop_size))
                y = rng.randint(0, max(0, image.height - crop_size))
            x = max(0, min(x, image.width - crop_size))
            y = max(0, min(y, image.height - crop_size))
            width, height = min(crop_size, image.width), min(crop_size, image.height)
            boxes, labels = [], []
            for n in nodes:
                a, b, c, d = n["bbox"]
                clipped = [max(a, x), max(b, y), min(c, x + width), min(d, y + height)]
                area = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
                if area >= 0.5 * (c - a) * (d - b) and area > 4:
                    boxes.append([clipped[0] - x, clipped[1] - y, clipped[2] - x, clipped[3] - y])
                    labels.append(SYMBOLS.index(n["label"]) + 1)
            name = f"{digest(row['id'])[:16]}-{crop_index}.png"
            path = output / "images" / name
            image.crop((x, y, x + width, y + height)).save(path)
            samples.append({"image": f"images/{name}", "source_id": row["id"], "source_group": row["group"],
                            "image_sha256": sha256(path), "origin": [x, y], "boxes": boxes, "labels": labels})
        if (index + 1) % 20 == 0:
            print(f"Prepared {index + 1}/{len(sources)} training drawings", flush=True)
    result = {"manifest_sha256": manifest["manifest_sha256"], "classes": SYMBOLS,
              "sources": [{k: r[k] for k in ("id", "group", "image_sha256", "graph_sha256", "split")} for r in sources],
              "crop_size": crop_size, "crops_per_drawing": crops_per_drawing, "samples": samples,
              "clipped_objects": "Include clipped boxes retaining at least half their area; tiny fragments are omitted in training only"}
    result["training_sha256"] = digest(result)
    save_new(output / "training.json", result)
    print(f"Prepared {len(samples)} source-traced training crops", flush=True)


def build_model(*, pretrained, num_classes=8, min_size=640):
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None, min_size=min_size, max_size=min_size,
        box_detections_per_img=100, box_score_thresh=0.05, rpn_score_thresh=0.0,
    )
    if not pretrained:
        # Torchvision uses trainable BatchNorm when weights=None, but the COCO
        # model uses FrozenBatchNorm. Reconstruct the trained architecture
        # without downloading/requiring its initialization weights at inference.
        from torch import nn
        from torchvision.ops.misc import FrozenBatchNorm2d

        def freeze_norms(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=child.eps))
                else:
                    freeze_norms(child)

        freeze_norms(model)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return model


def train(training_path, output, *, steps=1000, batch_size=2, device="mps", seed=20260914):
    import torch
    import torchvision
    from torchvision.transforms.functional import pil_to_tensor
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    data = json.loads(Path(training_path).read_text())
    if digest({k: v for k, v in data.items() if k != "training_sha256"}) != data["training_sha256"]:
        raise ValueError("Training manifest changed")
    if any(r["split"] != "train" for r in data["sources"]):
        raise ValueError("Non-training source in training manifest")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new training output; do not overwrite weights")
    output.mkdir(parents=True)
    config = {"architecture": "fasterrcnn_mobilenet_v3_large_320_fpn", "initial_weights": "COCO_V1",
              "classes": list(SYMBOLS), "manifest_sha256": data["manifest_sha256"],
              "training_sha256": data["training_sha256"], "steps": steps, "batch_size": batch_size,
              "seed": seed, "device": device, "min_size": 640, "learning_rate": 0.005,
              "rpn_score_threshold": 0.0,
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "source_sha256": sha256(__file__), "augmentation": "seeded horizontal flip at probability 0.5"}
    save_new(output / "config.json", config)
    model = build_model(pretrained=True).to(device).train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    samples = data["samples"]
    order = list(range(len(samples)))
    rng.shuffle(order)
    cursor = 0
    started = time.monotonic()
    root = Path(training_path).parent
    with (output / "losses.jsonl").open("x") as log:
        for step in range(1, steps + 1):
            images, targets = [], []
            for _ in range(batch_size):
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                row = samples[order[cursor]]
                cursor += 1
                path = root / row["image"]
                if sha256(path) != row["image_sha256"]:
                    raise ValueError("Training crop changed")
                with Image.open(path) as source:
                    image = source.convert("RGB")
                boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
                if rng.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if len(boxes):
                        boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
                images.append(pil_to_tensor(image).float().div_(255).to(device))
                targets.append({"boxes": boxes.to(device), "labels": torch.tensor(row["labels"], dtype=torch.int64, device=device)})
            warmup = min(1.0, step / 100)
            decay = 0.1 if step > steps * 0.8 else 1.0
            for group in optimizer.param_groups:
                group["lr"] = 0.005 * warmup * decay
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "elapsed_seconds": time.monotonic() - started,
                   "loss": float(loss.detach().cpu()), **{k: float(v.detach().cpu()) for k, v in losses.items()}}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 20 == 0 or step == 1:
                print(json.dumps(row), flush=True)
            if step % 200 == 0 or step == steps:
                path = output / f"step-{step:04d}.pt"
                torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "config": config, "step": step}, path)
                save_new(output / f"step-{step:04d}.sha256.json", {"sha256": sha256(path)})
    save_new(output / "completed.json", {"steps": steps, "elapsed_seconds": time.monotonic() - started,
                                         "checkpoint": f"step-{steps:04d}.pt"})


def infer(checkpoint, panel_path, output, *, split="validation", threshold=0.15, device="mps", selection=None):
    import torch
    from torchvision.ops import batched_nms
    from torchvision.transforms.functional import pil_to_tensor
    torch.set_num_threads(4)
    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Panel changed")
    weight_hash = sha256(checkpoint)
    if split == "test":
        from .selection import read_selection
        chosen = read_selection(selection, panel)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights["config"]["manifest_sha256"] != panel["manifest_sha256"]:
        raise ValueError("Weights were trained against another split")
    model = build_model(pretrained=False).to(device).eval()
    model.load_state_dict(weights["model"])
    output = Path(output)
    config = {"variant": "supervised_detector", "model": weights["config"]["architecture"],
              "detector_sha256": weight_hash, "panel_sha256": panel["panel_sha256"], "split": split,
              "threshold": threshold, "crop_size": 768, "stride": 576, "class_aware_nms_iou": 0.4,
              "device": device, "rpn_score_threshold": 0.0,
              "detector_role": "proposals; requires VLM interpretation"}
    if split == "test":
        from .selection import detector_signature
        if detector_signature(config) != chosen["detector_signature"]:
            raise ValueError("Final detector weights/parameters differ from validation selection")
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Inference config changed")
    else:
        save_new(output / "config.json", config)
    for row in panel["panels"][split]:
        case = output / digest(row["id"])[:16]
        if (case / "predictions.json").exists():
            continue
        started = time.monotonic()
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Inference image changed")
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        boxes, scores, labels = [], [], []
        def origins(length):
            return sorted(set([*range(0, max(1, length - 768 + 1), 576), max(0, length - 768)]))
        with torch.inference_mode():
            for y in origins(image.height):
                for x in origins(image.width):
                    crop = image.crop((x, y, min(image.width, x + 768), min(image.height, y + 768)))
                    prediction = model([pil_to_tensor(crop).float().div_(255).to(device)])[0]
                    keep = prediction["scores"] >= threshold
                    b = prediction["boxes"][keep].cpu()
                    b[:, [0, 2]] += x
                    b[:, [1, 3]] += y
                    boxes.append(b)
                    scores.append(prediction["scores"][keep].cpu())
                    labels.append(prediction["labels"][keep].cpu())
        boxes, scores, labels = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        indices = batched_nms(boxes, scores, labels, 0.4)
        predictions = [{"id": f"detector-{int(i)}", "bbox": boxes[i].tolist(),
                        "label": SYMBOLS[int(labels[i]) - 1], "confidence": float(scores[i]),
                        "disposition": "review_proposal", "attributes": {"geometry_basis": "supervised_raster_proposal"}}
                       for i in indices]
        save_new(case / "predictions.json", {"id": row["id"], "image_sha256": row["image_sha256"],
                 "config_sha256": digest(config), "predictions": predictions, "errors": [],
                 "runtime_seconds": time.monotonic() - started, "reserved_or_charged_usd": 0,
                 "detector_sha256": weight_hash})
        print(f"{row['id']}: {len(predictions)} supervised proposals in {time.monotonic() - started:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("train")
    a.add_argument("--training", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--steps", type=int, default=1000)
    a.add_argument("--device", default="mps")
    a = sub.add_parser("infer")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--panel", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--split", default="validation", choices=["train", "validation", "test", "development_exposed"])
    a.add_argument("--threshold", type=float, default=0.15)
    a.add_argument("--device", default="mps")
    a.add_argument("--selection")
    args = p.parse_args()
    if args.command == "prepare":
        prepare_training(args.manifest, args.out)
    elif args.command == "train":
        train(args.training, args.out, steps=args.steps, device=args.device)
    else:
        infer(args.checkpoint, args.panel, args.out, split=args.split, threshold=args.threshold,
              device=args.device, selection=args.selection)


if __name__ == "__main__":
    main()
