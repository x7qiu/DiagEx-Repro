"""Completion-gated broader detector reference; no API or final-test inference."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import tempfile
import time
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json

from .__main__ import score
from .checkpoint_comparison import choose_checkpoint, save_identical
from .data import digest, sha256
from .epoch_train import sealed
from .guidance import load_proposals
from .reuse_detector_validation import seed
from .training_report import report as training_report

NAME = "detector-trained-broad-v1"
SOURCE_NAMES = ("broader_detector.py", "checkpoint_comparison.py", "training_report.py",
    "reuse_detector_validation.py", "data.py", "epoch_train.py", "detector.py",
    "__main__.py", "scoring.py", "guidance.py", "selection.py")


def prepare(study, manifest, python):
    study = Path(study)
    training = study / "detector-epochs-v1"
    if not (training / "completed.json").is_file():
        raise ValueError("Training is incomplete; broader detector inference must wait")
    policy = sealed(study / "validation-promotion-policy.json", "policy_sha256")
    pilot = sealed(study / "panel-pilot.json", "panel_sha256")
    broad = sealed(study / "panel-broad.json", "panel_sha256")
    manifest_data = sealed(manifest, "manifest_sha256")
    if (pilot["panel_sha256"] != policy["pilot_panel_sha256"]
            or broad["panel_sha256"] != policy["broad_panel_sha256"]
            or not pilot["manifest_sha256"] == broad["manifest_sha256"] == manifest_data["manifest_sha256"]
            or sha256(Path(__file__).with_name("selection.py")) != policy["selection_source_sha256"]):
        raise ValueError("Frozen validation panels, split or selection policy changed")
    with tempfile.TemporaryDirectory(dir=study) as temporary:
        completed = training_report(study / "training-data/training.json", study / "training-schedule-v1.json",
                                    training, Path(temporary) / "verified.json")
    chosen = choose_checkpoint(completed)
    save_identical(study / (NAME + "-training-completion.json"), completed)
    source = training / f"validation-epoch-{chosen['epoch']:03d}"
    _, guidance = load_proposals(source, pilot, "validation")
    validation = json.loads(Path(chosen["report"]).read_text())
    if (sha256(chosen["report"]) != chosen["report_sha256"]
            or guidance["detector_config"] != validation["config"]
            or sha256(chosen["checkpoint"]) != guidance["detector_config"]["detector_sha256"]):
        raise ValueError("Selected checkpoint or pilot proposal evidence changed")
    output = study / NAME
    config = {**guidance["detector_config"], "panel_sha256": broad["panel_sha256"]}
    command = [str(python), "-m", "eval.pid2graph.detector", "infer", "--checkpoint", chosen["checkpoint"],
        "--panel", str(study / "panel-broad.json"), "--out", str(output), "--split", "validation",
        "--device", config["device"], "--threshold", str(config["threshold"])]
    record = {"epoch": chosen["epoch"], "checkpoint": chosen["checkpoint"],
        "checkpoint_sha256": config["detector_sha256"], "source_proposals": str(source),
        "source_guidance": guidance, "config": config, "config_sha256": digest(config),
        "command": command, "policy_sha256": policy["policy_sha256"], "test_access_authorized": False,
        "scope": "Broader detector diagnostic for the same best post-pilot checkpoint used by both queued hybrids. Selection uses pilot validation only; no VLM checkpoint choice or final method selection."}
    record["registration_sha256"] = digest(record)
    save_identical(study / (NAME + "-registration.json"), record)
    return record


def verify_reused_inputs(study, record):
    study = Path(study)
    output, source = study / NAME, Path(record["source_proposals"])
    if json.loads((output / "config.json").read_text()) != record["config"]:
        raise ValueError("Broader detector configuration changed")
    panel = sealed(study / "panel-pilot.json", "panel_sha256")
    _, guidance = load_proposals(source, panel, "validation")
    if guidance != record["source_guidance"]:
        raise ValueError("Original pilot hints changed after registration")
    for row in panel["panels"]["validation"]:
        relative = Path(digest(row["id"])[:16]) / "predictions.json"
        original = json.loads((source / relative).read_text())
        copied = json.loads((output / relative).read_text())
        keys = {"config_sha256", "validation_reuse"}
        if ({k: v for k, v in original.items() if k not in keys}
                != {k: v for k, v in copied.items() if k not in keys}
                or copied["config_sha256"] != record["config_sha256"]
                or copied.get("validation_reuse", {}).get("source_sha256") != sha256(source / relative)):
            raise ValueError("Reused detector hints or provenance changed")


def execute(study, manifest, python, *, wait_for_training=False):
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    def status(stage, **extra):
        atomic_write_json(study / (NAME + "-status.json"), {"stage": stage, "at": time.time(), **extra})
        print(json.dumps({"stage": stage, **extra}), flush=True)
    with (study / (NAME + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            sources = {str(Path(__file__).with_name(name)): sha256(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
            launch = {"manifest": str(manifest), "python": str(python), "source_sha256": sources,
                      "scope": "Broader validation detector only; no paid API calls or final test"}
            save_identical(study / (NAME + "-launch.json"), launch)
            if wait_for_training:
                status("waiting_for_training")
                with (study / "detector-epochs-v1/trainer.lock").open("r") as trainer:
                    fcntl.flock(trainer, fcntl.LOCK_SH)
            if any(sha256(path) != expected for path, expected in sources.items()):
                raise ValueError("Queued detector implementation changed while waiting")
            record = prepare(study, manifest, python)
            output = study / NAME
            if not output.exists():
                seed(study / "panel-pilot.json", study / "panel-broad.json", record["source_proposals"],
                     output, record["checkpoint"], manifest)
            verify_reused_inputs(study, record)
            report_path = study / "detector-trained-broad-report.json"
            if not report_path.exists():
                status("running_broader_detector", epoch=record["epoch"])
                subprocess.run(record["command"], check=True)
                result = score(manifest, study / "panel-broad.json", output, report_path, "validation")
            else:
                result = json.loads(report_path.read_text())
            broad = sealed(study / "panel-broad.json", "panel_sha256")
            _, guidance = load_proposals(output, broad, "validation")
            if (not result["complete"] or result["stopped"]
                    or result["config_sha256"] != record["config_sha256"]
                    or result["config"] != record["config"]
                    or result["manifest_sha256"] != broad["manifest_sha256"]
                    or guidance["detector_config"] != record["config"]
                    or result["split"] != "validation" or result["panel_sha256"] != broad["panel_sha256"]
                    or [c["id"] for c in result["cases"]] != [r["id"] for r in broad["panels"]["validation"]]):
                raise ValueError("Broader detector validation remains incomplete or inconsistent")
            save_identical(study / (NAME + "-completion.json"), {
                "registration_sha256": record["registration_sha256"],
                "report_sha256": sha256(report_path), "guidance": guidance})
            status("broader_detector_complete", epoch=record["epoch"], report=str(report_path),
                   report_sha256=sha256(report_path), macro_drawing_f1=result["summary"]["macro_drawing_f1"],
                   remaining="Wait for VLM checkpoint comparisons and promotion; final selection/test remain gated")
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "manifest", "python"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--wait-for-training", action="store_true")
    args = parser.parse_args()
    execute(args.study, args.manifest, args.python, wait_for_training=args.wait_for_training)
