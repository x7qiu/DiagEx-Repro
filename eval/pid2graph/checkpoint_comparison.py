"""Completion-gated, resumable execution of the two registered weight comparisons.

Never opens test inputs or selects a final method. Run from the repository root.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json

from .__main__ import score
from .data import digest, save_new, sha256
from .epoch_train import sealed
from .guidance import load_proposals
from .training_report import report as training_report

DRIVERS = {"broad_raster_recognition": "raster_broad_drive",
           "explicit_proposal_decisions": "raster_review_drive"}


def choose_checkpoint(report):
    if not report["complete"]:
        raise ValueError("Training has not completed its frozen stopping rule")
    candidates = [row for row in report["validation"] if row["epoch"] >= 1]
    if not candidates or any(not math.isfinite(row["macro_drawing_f1"]) for row in candidates):
        raise ValueError("No valid post-pilot checkpoint comparison is available")
    return max(candidates, key=lambda row: (row["macro_drawing_f1"], -row["epoch"]))


def save_identical(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Cannot resume changed comparison artifact: {path}")
    else:
        save_new(path, value)


def prepare(study, output):
    study, output = Path(study), Path(output)
    training = study / "detector-epochs-v1"
    if not (training / "completed.json").is_file():
        raise ValueError("Training is incomplete; no comparison registration or paid inference is allowed")
    policy = sealed(study / "validation-promotion-policy.json", "policy_sha256")
    shortlist = sealed(study / "pilot-shortlist.json", "shortlist_sha256")
    registry_path = study / "controlled-hybrids.json"
    if (shortlist["policy_sha256"] != policy["policy_sha256"]
            or sha256(registry_path) != policy["hybrid_registry_sha256"]
            or sha256(Path(__file__).with_name("selection.py")) != policy["selection_source_sha256"]
            or policy["maximum_additional_pilot_checkpoint_runs"] != 2
            or len(shortlist["shortlisted_variants"]) != 2
            or set(shortlist["shortlisted_variants"]) != set(DRIVERS)
            or shortlist["shortlisted_variants"] != [r["variant"] for r in shortlist["ranked"][:2]]):
        raise ValueError("Frozen promotion policy or shortlist differs")
    output.mkdir(parents=True, exist_ok=True)
    exposure_path = output / "training-completion.json"
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        exposure = training_report(study / "training-data/training.json", study / "training-schedule-v1.json",
                                   training, Path(temporary) / "verified.json")
    save_identical(exposure_path, exposure)
    # Even on resume, validate the trainer's archived sources and the checkpoint
    # against its original validation measurement before authorizing paid work.
    config = json.loads((training / "config.json").read_text())
    for path, expected in config["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError("Frozen trainer or validation implementation changed")
    chosen = choose_checkpoint(exposure)
    if sha256(chosen["report"]) != chosen["report_sha256"]:
        raise ValueError("Selected checkpoint validation report changed")
    validation = json.loads(Path(chosen["report"]).read_text())
    checkpoint_hash = sha256(chosen["checkpoint"])
    if validation["config"]["detector_sha256"] != checkpoint_hash:
        raise ValueError("Selected checkpoint weights changed")
    proposal_run = training / f"validation-epoch-{chosen['epoch']:03d}"
    panel_path = study / "panel-pilot.json"
    panel = sealed(panel_path, "panel_sha256")
    if panel["panel_sha256"] != policy["pilot_panel_sha256"] or config["panel_sha256"] != panel["panel_sha256"]:
        raise ValueError("Pilot validation panel changed")
    _, guidance = load_proposals(proposal_run, panel, "validation")
    if guidance["detector_config"] != validation["config"]:
        raise ValueError("Replacement proposals differ from the selected checkpoint")
    registrations = {}
    for item in json.loads(registry_path.read_text())["registered"]:
        path = study / item["registration"]
        reg = json.loads(path.read_text())
        registrations[reg["runner_variant"]] = (path, reg)
    arms = []
    for ranked in shortlist["ranked"][:2]:
        variant = ranked["variant"]
        registration_path, registration = registrations[variant]
        if (sha256(registration_path) != ranked["registration_sha256"]
                or sha256(ranked["report"]) != ranked["report_sha256"]):
            raise ValueError("Original pilot evidence changed")
        for path, expected in registration["method_fingerprints"].items():
            if sha256(path) != expected:
                raise ValueError(f"Frozen hybrid source changed: {path}")
        original = json.loads((Path(ranked["run"]) / "config.json").read_text())
        if original != json.loads(Path(ranked["report"]).read_text())["config"]:
            raise ValueError("Original pilot configuration changed")
        replacement = {**copy.deepcopy(original), "guidance": guidance}
        destination = output / variant
        destination.mkdir(exist_ok=True)
        # The frozen runner checks this exact prewritten configuration before
        # sending a request; changing anything besides guidance aborts the run.
        save_identical(destination / "config.json", replacement)
        record = {"variant": variant, "original_registration_sha256": sha256(registration_path),
            "reference_report": ranked["report"], "reference_report_sha256": ranked["report_sha256"],
            "shortlist_sha256": shortlist["shortlist_sha256"], "policy_sha256": policy["policy_sha256"],
            "training_completion_sha256": sha256(exposure_path), "checkpoint": chosen["checkpoint"],
            "checkpoint_sha256": checkpoint_hash, "epoch": chosen["epoch"],
            "proposal_run": str(proposal_run), "config": replacement, "config_sha256": digest(replacement),
            "controlled_difference": "Detector weights and resulting hints only; frozen algorithm, inputs and provider policy retained",
            "test_access_authorized": False}
        record["registration_sha256"] = digest(record)
        save_identical(output / f"{variant}-registration.json", record)
        command = [sys.executable, "-m", "eval.pid2graph." + DRIVERS[variant],
            "--panel", str(panel_path), "--out", str(destination), "--ledger", str(study / "spending.json"),
            "--prices", str(study / "prices-v1.json"), "--split", "validation", "--variant", variant,
            "--model", original["model"], "--proposal-run", str(proposal_run), "--tiles-per-process", "4"]
        arms.append({"variant": variant, "run": str(destination), "command": command,
                     "config_sha256": digest(replacement), "registration_sha256": record["registration_sha256"]})
    plan = {"epoch": chosen["epoch"], "checkpoint": chosen["checkpoint"], "checkpoint_sha256": checkpoint_hash,
            "arms": arms, "test_access_authorized": False, "policy_sha256": policy["policy_sha256"]}
    save_identical(output / "plan.json", plan)
    return plan


def execute(study, output, manifest, *, wait_for_training=False):
    study, output = Path(study), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    def status(stage, **values):
        atomic_write_json(status_path, {"stage": stage, "at": time.time(), **values})
        print(json.dumps({"stage": stage, **values}), flush=True)
    with (output / "comparison.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if wait_for_training:
                status("waiting_for_training")
                # A blocking OS lock wakes on trainer exit without repeated
                # polling. Completed-state verification below distinguishes
                # success from an interrupted or failed training process.
                with (study / "detector-epochs-v1/trainer.lock").open("r") as trainer_lock:
                    fcntl.flock(trainer_lock, fcntl.LOCK_SH)
                    plan = prepare(study, output)
            else:
                plan = prepare(study, output)
            for arm in plan["arms"]:
                report_path = output / f"{arm['variant']}-report.json"
                if report_path.exists():
                    existing = json.loads(report_path.read_text())
                    if existing["config_sha256"] != arm["config_sha256"] or any(c["status"] == "missing" for c in existing["cases"]):
                        raise ValueError("Existing comparison report is incompatible or incomplete")
                    continue
                status("running_comparison", variant=arm["variant"], epoch=plan["epoch"])
                subprocess.run(arm["command"], check=True)
                ledger_path = output / f"{arm['variant']}-billing-snapshot.json"
                if not ledger_path.exists():
                    save_new(ledger_path, json.loads((study / "spending.json").read_text()))
                result = score(manifest, study / "panel-pilot.json", arm["run"], report_path, "validation", ledger_path)
                if any(c["status"] == "missing" for c in result["cases"]):
                    raise ValueError("Comparison still contains unattempted drawings")
                status("comparison_scored", variant=arm["variant"], macro_drawing_f1=result["summary"]["macro_drawing_f1"])
            status("comparisons_complete", epoch=plan["epoch"], remaining="Reconcile billing, analyze errors, select checkpoint per method and broaden validation; final test remains sealed")
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "out", "manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--wait-for-training", action="store_true")
    args = parser.parse_args()
    execute(args.study, args.out, args.manifest, wait_for_training=args.wait_for_training)
