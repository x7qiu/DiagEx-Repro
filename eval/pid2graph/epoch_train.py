"""Resumable, validation-only epoch training under a sealed study schedule."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

from .data import SYMBOLS, digest, save_new, sha256


def sealed(path, key):
    value = json.loads(Path(path).read_text())
    if digest({k: v for k, v in value.items() if k != key}) != value[key]:
        raise ValueError(f"Seal mismatch: {path}")
    return value


def epoch_order(samples, schedule, epoch):
    """Every base crop once, then the predeclared background/rare repeats."""
    spec = schedule["sampling"]
    if len(samples) != spec["base_crops"]:
        raise ValueError("Base crop count changed")
    rng = random.Random(f"{schedule['seed']}:epoch:{epoch}")
    order = list(range(len(samples)))
    backgrounds = [i for i, r in enumerate(samples) if not r["labels"]]
    if spec["extra_background_draws"]:
        if not backgrounds:
            raise ValueError("No background crops available")
        order.extend(rng.choices(backgrounds, k=spec["extra_background_draws"]))
    for label, item in sorted(spec["rare_classes"].items()):
        pool = [i for i, r in enumerate(samples) if int(label) in r["labels"]]
        if len(pool) != item["available_crops"]:
            raise ValueError(f"Rare class {label} coverage changed")
        if item["extra_draws"]:
            order.extend(rng.choices(pool, k=item["extra_draws"]))
    rng.shuffle(order)
    if len(order) != spec["epoch_presentations"]:
        raise ValueError("Epoch size differs from frozen schedule")
    return order


def learning_rate(step, total_steps, initial):
    if step <= 100:
        return initial * step / 100
    progress = min(1.0, (step - 100) / max(1, total_steps - 100))
    return 0.0001 + (initial - 0.0001) * (1 + math.cos(math.pi * progress)) / 2


def selection_state(history, schedule):
    """Epoch zero participates in selection and anchors the patience reference."""
    best = max(history, key=lambda r: (r["macro_drawing_f1"], -r["epoch"]))
    reference = history[0]["macro_drawing_f1"]
    stale = 0
    for row in history[1:]:
        if row["macro_drawing_f1"] >= reference + schedule["minimum_macro_f1_improvement"]:
            reference, stale = row["macro_drawing_f1"], 0
        else:
            stale += 1
    stop = history[-1]["epoch"] >= schedule["minimum_epochs"] and stale >= schedule["early_stopping_patience"]
    return {"best_epoch": best["epoch"], "best_macro_drawing_f1": best["macro_drawing_f1"],
            "patience_reference": reference, "stale_epochs": stale, "early_stop": stop}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def atomic_torch(path, value):
    import torch
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def train(training_path, schedule_path, manifest_path, output, *, max_new_steps=None, skip_epoch_zero=False):
    import torch
    import torchvision
    from PIL import Image, ImageEnhance
    from torchvision.transforms.functional import pil_to_tensor

    from .__main__ import score
    from .detector import build_model

    training_path, schedule_path, output = Path(training_path).resolve(), Path(schedule_path).resolve(), Path(output).resolve()
    data = sealed(training_path, "training_sha256")
    schedule = sealed(schedule_path, "schedule_sha256")
    manifest = sealed(manifest_path, "manifest_sha256")
    panel_path = schedule_path.parent / schedule["validation_panel"]
    panel = sealed(panel_path, "panel_sha256")
    if data["training_sha256"] != schedule["training_sha256"] or data["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Training inputs differ from frozen schedule/split")
    if panel["manifest_sha256"] != manifest["manifest_sha256"] or list(data["classes"]) != list(SYMBOLS):
        raise ValueError("Validation split/classes differ")
    lookup = {r["id"]: r for r in manifest["drawings"]}
    source_ids = {r["id"] for r in data["sources"]}
    for r in data["sources"]:
        if r["split"] != "train" or any(r[k] != lookup[r["id"]][k] for k in ("split", "group", "image_sha256", "graph_sha256")):
            raise ValueError("Training source provenance differs")
    for r in data["samples"]:
        if r["source_id"] not in source_ids or r["source_group"] != lookup[r["source_id"]]["group"]:
            raise ValueError("Crop not associated with a training source")
    if any(lookup[r["id"]]["split"] != "validation" for r in panel["panels"]["validation"]):
        raise ValueError("Non-validation drawing in validation panel")
    initial = Path(schedule["initial_checkpoint"]).resolve()
    if sha256(initial) != schedule["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    # Also validate the sampler before reserving an output directory or loading MPS.
    epoch_order(data["samples"], schedule, 1)
    if skip_epoch_zero and not max_new_steps:
        raise ValueError("Skipping epoch zero is restricted to bounded software smoke runs")
    if max_new_steps is not None and max_new_steps <= 0:
        raise ValueError("max_new_steps must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trainer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = schedule["device"]
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; use the authorized local execution environment")
        torch.set_num_threads(4)
        torch.manual_seed(schedule["seed"])
        rng = random.Random(schedule["seed"])
        code = {str(p.resolve()): sha256(p) for p in [Path(__file__), Path(__file__).with_name("detector.py"),
                Path(__file__).with_name("data.py"), Path(__file__).with_name("scoring.py"), Path(__file__).with_name("__main__.py")]}
        config = {"architecture": "fasterrcnn_mobilenet_v3_large_320_fpn", "classes": list(SYMBOLS), "min_size": 640,
                  "manifest_sha256": manifest["manifest_sha256"], "training_sha256": data["training_sha256"],
                  "schedule_sha256": schedule["schedule_sha256"], "panel_sha256": panel["panel_sha256"],
                  "initial_checkpoint_sha256": sha256(initial), "source_hashes": code,
                  "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                  "device": device, "software_smoke_only": skip_epoch_zero,
                  "optimizer_reset_at_initialization": True, "trainable_parameters": "all parameters of reconstructed FrozenBatchNorm architecture"}
        if (output / "config.json").exists():
            if json.loads((output / "config.json").read_text()) != config:
                raise ValueError("Training configuration/source changed on resume")
        else:
            save_new(output / "config.json", config)
        model = build_model(pretrained=False).to(device).train()
        opt = schedule["optimizer"]
        optimizer = torch.optim.SGD(model.parameters(), lr=opt["learning_rate"], momentum=opt["momentum"], weight_decay=opt["weight_decay"])
        state = {"epoch": 1, "cursor": 0, "step": 0, "presentations": 0, "history": [], "training_seconds": 0.0}
        latest = output / "latest.pt"
        if latest.exists():
            weights = torch.load(latest, map_location="cpu", weights_only=True)
            if weights["config"] != config:
                raise ValueError("Checkpoint configuration changed")
            model.load_state_dict(weights["model"])
            optimizer.load_state_dict(weights["optimizer"])
            state = weights["state"]
            rng.setstate(weights["python_rng"])
            torch.set_rng_state(weights["torch_rng"])
            if device == "mps":
                torch.mps.set_rng_state(weights["device_rng"])
            print(f"Resumed step {state['step']}, epoch {state['epoch']}, cursor {state['cursor']}", flush=True)
        else:
            weights = torch.load(initial, map_location="cpu", weights_only=True)
            model.load_state_dict(weights["model"])
        del weights
        stop_requested = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: stop_requested.append(signum))

        def persist(status):
            atomic_torch(latest, {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                 "config": config, "optimizer": optimizer.state_dict(), "state": state,
                                 "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                                 "device_rng": torch.mps.get_rng_state().cpu() if device == "mps" else None})
            atomic_json(output / "progress.json", {**state, "status": status, "pid": os.getpid(), "updated_at": time.time(),
                                                   "schedule_sha256": schedule["schedule_sha256"]})

        def validate(epoch, checkpoint):
            for path, expected in code.items():
                if sha256(path) != expected:
                    raise ValueError("Validation implementation changed during training")
            run = output / f"validation-epoch-{epoch:03d}"
            report_path = output / f"validation-epoch-{epoch:03d}.json"
            model.cpu()
            if device == "mps":
                torch.mps.empty_cache()
            subprocess.run([sys.executable, "-m", "eval.pid2graph.detector", "infer", "--checkpoint", str(checkpoint),
                            "--panel", str(panel_path), "--out", str(run), "--split", "validation", "--device", device], check=True)
            report = json.loads(report_path.read_text()) if report_path.exists() else score(manifest_path, panel_path, run, report_path, "validation")
            if not report["complete"] or report["config"]["detector_sha256"] != sha256(checkpoint):
                raise ValueError("Validation incomplete or checkpoint differs")
            metric = report["summary"]["macro_drawing_f1"]
            if not isinstance(metric, (float, int)) or not math.isfinite(metric):
                raise ValueError("Invalid validation metric")
            state["history"].append({"epoch": epoch, "macro_drawing_f1": metric, "checkpoint": str(checkpoint),
                                     "report": str(report_path), "report_sha256": sha256(report_path),
                                     "step": state["step"], "presentations": state["presentations"]})
            state["selection"] = selection_state(state["history"], schedule)
            model.to(device).train()
            print(json.dumps({"validation": state["history"][-1], "selection": state["selection"]}), flush=True)

        if not latest.exists():
            persist("initialized")
        if not state["history"] and not skip_epoch_zero:
            validate(0, initial)
            persist("epoch_zero_validated")
        # Discard losses beyond the last committed state if a previous process died.
        log_path = output / "losses.jsonl"
        if log_path.exists():
            committed = []
            for line in log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if row["step"] <= state["step"]:
                    committed.append(line)
            log_path.write_text("".join(line + "\n" for line in committed))
        invocation_start = state["step"]
        batch_size = schedule["batch_size"]
        total_steps = math.ceil(schedule["sampling"]["epoch_presentations"] / batch_size) * schedule["maximum_epochs"]
        with log_path.open("a") as log:
            while state["epoch"] <= schedule["maximum_epochs"]:
                if state.get("selection", {}).get("early_stop"):
                    break
                order = epoch_order(data["samples"], schedule, state["epoch"])
                while state["cursor"] < len(order):
                    if stop_requested or (max_new_steps is not None and state["step"] - invocation_start >= max_new_steps):
                        persist("paused")
                        return
                    started = time.monotonic()
                    batch = order[state["cursor"]:state["cursor"] + batch_size]
                    images, targets = [], []
                    for index in batch:
                        row = data["samples"][index]
                        path = training_path.parent / row["image"]
                        if sha256(path) != row["image_sha256"]:
                            raise ValueError("Training crop changed")
                        with Image.open(path) as source:
                            image = source.convert("RGB")
                        boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
                        if rng.random() < schedule["augmentation"]["horizontal_flip_probability"]:
                            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                            boxes[:, [0, 2]] = image.width - boxes[:, [2, 0]]
                        image = ImageEnhance.Contrast(image).enhance(rng.uniform(*schedule["augmentation"]["contrast_range"]))
                        images.append(pil_to_tensor(image).float().div_(255).to(device))
                        targets.append({"boxes": boxes.to(device), "labels": torch.tensor(row["labels"], dtype=torch.int64, device=device)})
                    rate = learning_rate(state["step"] + 1, total_steps, opt["learning_rate"])
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    optimizer.zero_grad(set_to_none=True)
                    losses = model(images, targets)
                    loss = sum(losses.values())
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    numbers = {k: float(v.detach().cpu()) for k, v in losses.items()}
                    state["step"] += 1
                    state["cursor"] += len(batch)
                    state["presentations"] += len(batch)
                    state["training_seconds"] += time.monotonic() - started
                    row = {"step": state["step"], "epoch": state["epoch"], "cursor": state["cursor"],
                           "presentations": state["presentations"], "training_seconds": state["training_seconds"],
                           "learning_rate": rate, "gradient_norm": float(norm.cpu()), "loss": sum(numbers.values()), **numbers}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if state["step"] % 20 == 0 or state["step"] == 1:
                        print(json.dumps(row), flush=True)
                    if state["step"] % schedule["checkpoint_every_steps"] == 0:
                        persist("training")
                persist("awaiting_epoch_validation")
                epoch = state["epoch"]
                checkpoint = output / f"epoch-{epoch:03d}.pt"
                if not checkpoint.exists():
                    atomic_torch(checkpoint, {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                             "config": config, "step": state["step"], "epoch": epoch})
                marker = checkpoint.with_suffix(".sha256.json")
                if marker.exists():
                    if json.loads(marker.read_text())["sha256"] != sha256(checkpoint):
                        raise ValueError("Epoch checkpoint changed")
                else:
                    save_new(marker, {"sha256": sha256(checkpoint), "step": state["step"], "epoch": epoch})
                if skip_epoch_zero:
                    raise ValueError("Software smoke reached a full epoch; use the study trainer")
                validate(epoch, checkpoint)
                state["epoch"], state["cursor"] = epoch + 1, 0
                persist("epoch_validated")
        persist("complete")
        atomic_json(output / "completed.json", {**state, "reason": "early_stopping" if state["selection"]["early_stop"] else "maximum_epochs",
                                               "schedule_sha256": schedule["schedule_sha256"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-steps", type=int)
    p.add_argument("--skip-epoch-zero", action="store_true", help="Bounded software smoke only; excluded from model selection")
    a = p.parse_args()
    train(a.training, a.schedule, a.manifest, a.out, max_new_steps=a.max_new_steps, skip_epoch_zero=a.skip_epoch_zero)


if __name__ == "__main__":
    main()
