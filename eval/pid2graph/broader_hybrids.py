"""Expand promoted hybrids to validation only, preserving paid pilot attempts.

Run after checkpoint_promotion. This controller uses the existing frozen drivers,
does not select a final method, and never opens test inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json

from .__main__ import score
from .checkpoint_comparison import DRIVERS, save_identical
from .checkpoint_promotion import choose_pair, verify_attempts
from .data import digest, save_new, sha256
from .epoch_train import sealed
from .guidance import load_proposals
from .reuse_validation import seed
from .selection import perception_signature


def prepare(study, promotion_path, manifest, output, proposal_runs):
    study, output = Path(study), Path(output)
    promotion = sealed(promotion_path, "promotion_sha256")
    policy = sealed(study / "validation-promotion-policy.json", "policy_sha256")
    panel = sealed(study / "panel-broad.json", "panel_sha256")
    choices = promotion["choices"]
    if (promotion["policy_sha256"] != policy["policy_sha256"]
            or promotion["broad_panel_sha256"] != panel["panel_sha256"]
            or policy["broad_panel_sha256"] != panel["panel_sha256"]
            or policy["maximum_broader_hybrids"] != 2
            or len(choices) != 2 or {c["variant"] for c in choices} != set(DRIVERS)
            or promotion["test_access_authorized"]):
        raise ValueError("Expected exactly the two promoted validation hybrids")
    # Validate both arms before creating any destinations or launching inference.
    arms = []
    for choice in choices:
        variant = choice["variant"]
        reg = sealed(choice["registration"], "registration_sha256")
        if (reg["registration_sha256"] != choice["registration_sha256"]
                or reg["runner_variant"] != variant or reg["policy_sha256"] != policy["policy_sha256"]
                or reg["source_run"] != choice["source_run"]
                or reg["selected_weight_source"] != choice["weight_source"]
                or reg["proposal_run"] != choice["source_proposals"]
                or reg["refreshed_report"] != choice["selected_report"]
                or sha256(reg["refreshed_report"]) != reg["refreshed_report_sha256"]
                or sha256(reg["source_report"]) != reg["source_report_sha256"]
                or not reg["method_fingerprints"]
                or any(sha256(p) != h for p, h in reg["method_fingerprints"].items())):
            raise ValueError("Promoted source registration or frozen method changed")
        reports = [json.loads(Path(p).read_text()) for p in choice["reports"]]
        selected = choose_pair(*reports)
        if (choice["weight_source"] != ("pilot", "trained")[selected]
                or choice["selected_report"] != choice["reports"][selected]):
            raise ValueError("Promoted checkpoint differs from frozen ranking")
        original = json.loads((Path(choice["source_run"]) / "config.json").read_text())
        if digest(original) != reg["source_config_sha256"] or original != reports[selected]["config"]:
            raise ValueError("Promoted inference configuration changed")
        proposals = Path(proposal_runs[choice["weight_source"]])
        _, guidance = load_proposals(proposals, panel, "validation")
        config = {**original, "panel_sha256": panel["panel_sha256"], "guidance": guidance}
        if perception_signature(config) != perception_signature(original):
            raise ValueError("Broader detector hints use different inference settings or weights")
        destination = output / variant
        command = [sys.executable, "-m", "eval.pid2graph." + DRIVERS[variant],
            "--panel", str(study / "panel-broad.json"), "--out", str(destination),
            "--ledger", str(study / "spending.json"), "--prices", str(study / "prices-v1.json"),
            "--split", "validation", "--variant", variant, "--model", config["model"],
            "--proposal-run", str(proposals), "--tiles-per-process", "4"]
        arms.append({"variant": variant, "run": str(destination), "command": command,
            "config_sha256": digest(config), "config": config, "choice": choice,
            "target_proposals": str(proposals)})
    plan = {"promotion_sha256": promotion["promotion_sha256"], "arms": arms,
            "panel_sha256": panel["panel_sha256"], "test_access_authorized": False,
            "controller_sources": {str(Path(__file__).with_name(name)): sha256(Path(__file__).with_name(name))
                for name in ("broader_hybrids.py", "reuse_validation.py", "checkpoint_promotion.py")}}
    save_identical(output / "plan.json", plan)
    for arm in arms:
        destination, choice = Path(arm["run"]), arm["choice"]
        if not destination.exists():
            seed(study / "panel-pilot.json", study / "panel-broad.json", choice["source_run"],
                 destination, choice["registration"], manifest,
                 source_proposals=choice["source_proposals"], target_proposals=arm["target_proposals"])
        reuse = json.loads((destination / "validation-reuse.json").read_text())
        if (json.loads((destination / "config.json").read_text()) != arm["config"]
                or reuse["registration_sha256"] != sha256(choice["registration"])
                or reuse["target_panel_sha256"] != panel["panel_sha256"]
                or reuse["source_config_sha256"] != digest(json.loads(
                    (Path(choice["source_run"]) / "config.json").read_text()))):
            raise ValueError("Existing validation destination differs from promoted source")
    return plan


def execute(study, promotion, manifest, output, proposal_runs, *, prepare_only=False):
    study, output = Path(study), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def status(stage, **fields):
        atomic_write_json(output / "status.json", {"stage": stage, "at": time.time(), **fields})
        print(json.dumps({"stage": stage, **fields}), flush=True)
    with (output / "broader-hybrids.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            plan = prepare(study, promotion, manifest, output, proposal_runs)
            if prepare_only:
                status("prepared", new_paid_requests=0)
                return plan
            panel = sealed(study / "panel-broad.json", "panel_sha256")
            for arm in plan["arms"]:
                report_path = output / f"{arm['variant']}-report.json"
                if not report_path.exists():
                    status("running_broader_validation", variant=arm["variant"])
                    subprocess.run(arm["command"], check=True)
                    ledger = json.loads((study / "spending.json").read_text())
                    attempts = verify_attempts(arm["run"], panel, ledger)
                    snapshot = output / f"{arm['variant']}-billing-snapshot.json"
                    # A prior snapshot can survive interruption before scoring;
                    # preserve it rather than silently replacing evidence.
                    if not snapshot.exists():
                        save_new(snapshot, ledger)
                    save_identical(output / f"{arm['variant']}-coverage.json",
                                   {k: v for k, v in attempts.items() if k != "billing"})
                    score(manifest, study / "panel-broad.json", arm["run"], report_path, "validation", snapshot)
                report = json.loads(report_path.read_text())
                if (report["config_sha256"] != arm["config_sha256"] or report["stopped"]
                        or [r["id"] for r in report["cases"]] != [r["id"] for r in panel["panels"]["validation"]]
                        or any(r["status"] not in {"complete", "partial"} for r in report["cases"])):
                    raise ValueError("Broader validation report is incompatible or unfinished")
                status("broader_hybrid_scored", variant=arm["variant"],
                       macro_drawing_f1=report["summary"]["macro_drawing_f1"])
            status("broader_hybrids_complete", test_access_authorized=False,
                   remaining="Reconcile costs, analyze broader results, include baseline in validation-only selection")
            return plan
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "promotion", "manifest", "out", "pilot-proposals", "trained-proposals"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    execute(args.study, args.promotion, args.manifest, args.out,
            {"pilot": args.pilot_proposals, "trained": args.trained_proposals}, prepare_only=args.prepare_only)
