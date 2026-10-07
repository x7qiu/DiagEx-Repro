"""Seal one validation-only final choice after all broader comparisons finish.

No inference is performed here. Test filenames in the already frozen panel are
metadata only; neither test images nor test GraphML are opened.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from diagex.llm.billing import summarize

from .__main__ import score
from .checkpoint_comparison import DRIVERS, save_identical
from .data import digest, save_new, sha256
from .epoch_train import sealed
from .selection import read_selection, seal


def verify_validation_attempts(run, panel, ledger):
    """Check actual request categories for baseline and promoted hybrids alike."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    category = "baseline" if config["variant"] == "baseline" else "experiments"
    requests = {r["id"]: r for r in ledger["requests"]}
    if (ledger["schema_version"] != 2 or len(requests) != len(ledger["requests"])
            or config["split"] != "validation" or config["panel_sha256"] != panel["panel_sha256"]
            or config["variant"] not in {"baseline", *DRIVERS} or (run / "stop-request.json").exists()):
        raise ValueError("Invalid validation run or ledger")
    ids, attempted, failed = [], 0, 0
    for row in panel["panels"]["validation"]:
        case = run / digest(row["id"])[:16]
        summary = json.loads((case / "predictions.json").read_text())
        tiles = [(p, json.loads(p.read_text())) for p in case.glob("p*-r*-c*.json")]
        if (summary["id"] != row["id"] or summary["image_sha256"] != row["image_sha256"]
                or sha256(row["image"]) != row["image_sha256"]
                or summary["config_sha256"] != digest(config) or not tiles or len(tiles) != summary["tiles"]
                or any(t["config_sha256"] != digest(config) or t["status"] not in {"complete", "failed"}
                       or p.name != t["tile_id"] + ".json" for p, t in tiles)):
            raise ValueError("Validation attempt coverage or source changed")
        predictions = [p for _, t in tiles for p in t["predictions"]]
        errors = {t["tile_id"]: t.get("error") for _, t in tiles if t["status"] == "failed"}
        identities = {i for _, t in tiles for i in t["ledger_request_ids"]}
        if (len({p["id"] for p in predictions}) != len(predictions)
                or sorted(predictions, key=lambda p: p["id"]) != sorted(summary["predictions"], key=lambda p: p["id"])
                or errors != {e["tile_id"]: e["error"] for e in summary["errors"]}
                or summary["completed_tiles"] != len(tiles) - len(errors)
                or identities != set(summary["ledger_request_ids"])
                or len(identities) != len(summary["ledger_request_ids"])):
            raise ValueError("Drawing summary differs from immutable tile attempts")
        if any(i not in requests or requests[i]["status"] == "pending"
               or requests[i]["category"] != category or requests[i]["model"] != config["model"] for i in identities):
            raise ValueError("Unknown, active or foreign validation request")
        ids.extend(summary["ledger_request_ids"])
        attempted += len(tiles)
        failed += len(errors)
    if len(ids) != len(set(ids)):
        raise ValueError("Request is attributed to multiple validation drawings")
    return {"attempted_tiles": attempted, "failed_tiles": failed, "billing": summarize(ledger, ids)}


def completed_report(path, panel, variant):
    report = json.loads(Path(path).read_text())
    if (report["split"] != "validation" or report["panel_sha256"] != panel["panel_sha256"]
            or report["manifest_sha256"] != panel["manifest_sha256"] or report["stopped"]
            or report["config"]["variant"] != variant
            or digest(report["config"]) != report["config_sha256"]
            or [c["id"] for c in report["cases"]] != [r["id"] for r in panel["panels"]["validation"]]
            or any(c["status"] not in {"complete", "partial"} for c in report["cases"])):
        raise ValueError("Required validation report is unfinished or incompatible")
    return report


def ready_inputs(study, hybrids, comparison):
    """Check prerequisite evidence before reading predictions or creating output."""
    study, hybrids, comparison = Path(study), Path(hybrids), Path(comparison)
    if json.loads((hybrids / "status.json").read_text())["stage"] != "broader_hybrids_complete":
        raise ValueError("Both broader hybrid comparisons must finish before final selection")
    training_path = comparison / "training-completion.json"
    training = json.loads(training_path.read_text())
    if (json.loads((comparison / "status.json").read_text())["stage"] != "comparisons_complete"
            or not training["complete"] or sha256(training["state_path"]) != training["state_sha256"]):
        raise ValueError("Training or checkpoint comparison evidence is incomplete or changed")
    for variant in DRIVERS:
        registration = sealed(comparison / f"{variant}-registration.json", "registration_sha256")
        if registration["training_completion_sha256"] != sha256(training_path):
            raise ValueError("Checkpoint comparisons are not bound to completed training")
    policy = sealed(study / "validation-promotion-policy.json", "policy_sha256")
    panel = sealed(study / "panel-broad.json", "panel_sha256")
    plan = json.loads((hybrids / "plan.json").read_text())
    if (policy["broad_panel_sha256"] != panel["panel_sha256"]
            or plan["panel_sha256"] != panel["panel_sha256"]
            or sha256(Path(__file__).with_name("selection.py")) != policy["selection_source_sha256"]
            or len(plan["arms"]) != 2 or {a["variant"] for a in plan["arms"]} != set(DRIVERS)):
        raise ValueError("Frozen broader policy, panel or candidate set changed")
    # The completed promotion already validates the full training trajectory and
    # both trained comparisons. Bind this gate to that evidence rather than
    # repeating training or opening any final inputs.
    promotion_paths = {Path(a["choice"]["registration"]).parent / "promotion.json" for a in plan["arms"]}
    if len(promotion_paths) != 1:
        raise ValueError("Broader candidates came from different promotions")
    promotion_path = promotion_paths.pop()
    promotion = sealed(promotion_path, "promotion_sha256")
    if (promotion["promotion_sha256"] != plan["promotion_sha256"]
            or promotion["policy_sha256"] != policy["policy_sha256"]
            or promotion["choices"] != [a["choice"] for a in plan["arms"]]):
        raise ValueError("Broader candidates differ from promoted choices")
    fingerprints = dict(plan["controller_sources"])
    inputs = [{"variant": "baseline", "run": str(study / "baseline-broad-v1"),
               "report": str(study / "baseline-broad-report.json")}]
    baseline_reuse = json.loads((study / "baseline-broad-v1/validation-reuse.json").read_text())
    baseline_registration = Path(baseline_reuse["registration"])
    if sha256(baseline_registration) != baseline_reuse["registration_sha256"]:
        raise ValueError("Baseline registration changed")
    registrations = [(baseline_registration, None)]
    for arm in plan["arms"]:
        choice = arm["choice"]
        registrations.append((Path(choice["registration"]), choice["registration_sha256"]))
        inputs.append({"variant": arm["variant"], "run": arm["run"],
                       "report": str(hybrids / f"{arm['variant']}-report.json")})
    for path, expected in registrations:
        reg = sealed(path, "registration_sha256") if expected else json.loads(path.read_text())
        if expected and reg["registration_sha256"] != expected:
            raise ValueError("Promoted method registration changed")
        if not reg["method_fingerprints"]:
            raise ValueError("Missing frozen inference sources")
        fingerprints.update(reg["method_fingerprints"])
    if any(sha256(path) != expected for path, expected in fingerprints.items()):
        raise ValueError("Frozen inference or orchestration source changed")
    for item in inputs:
        report = completed_report(item["report"], panel, item["variant"])
        config = json.loads((Path(item["run"]) / "config.json").read_text())
        if report["config"] != config:
            raise ValueError("Scored configuration differs from inference")
        if item["variant"] == "baseline":
            if digest(config) != baseline_reuse["target_config_sha256"]:
                raise ValueError("Baseline differs from its reused configuration")
        else:
            arm = next(a for a in plan["arms"] if a["variant"] == item["variant"])
            if digest(config) != arm["config_sha256"]:
                raise ValueError("Broader hybrid differs from promoted configuration")
    return panel, inputs, {"policy_sha256": policy["policy_sha256"],
        "promotion_sha256": promotion["promotion_sha256"], "broader_plan_sha256": sha256(hybrids / "plan.json"),
        "training_completion_sha256": sha256(training_path),
        "source_fingerprints": fingerprints}


def run(study, hybrids, comparison, manifest):
    study = Path(study)
    # A canonical location prevents this command from silently creating a second
    # candidate after test results exist. Resume returns the original decision.
    output = study / "final-selection-v1"
    panel, inputs, evidence = ready_inputs(study, hybrids, comparison)
    manifest_data = sealed(manifest, "manifest_sha256")
    if manifest_data["manifest_sha256"] != panel["manifest_sha256"]:
        raise ValueError("Split manifest changed")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "selection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {**evidence, "manifest_sha256": panel["manifest_sha256"],
                   "input_report_sha256": {r["report"]: sha256(r["report"]) for r in inputs}}
        save_identical(output / "inputs.json", binding)
        selection_path = output / "selection.json"
        if selection_path.exists():
            selected = read_selection(selection_path, panel)
            for row in selected["validation_reports"]:
                if sha256(row["path"]) != row["sha256"]:
                    raise ValueError("Sealed selection evidence changed")
            return selected
        snapshot = output / "billing-snapshot.json"
        ledger = json.loads((snapshot if snapshot.exists() else study / "spending.json").read_text())
        verified = {item["variant"]: verify_validation_attempts(item["run"], panel, ledger) for item in inputs}
        if not snapshot.exists():
            save_new(snapshot, ledger)
        refreshed, coverage = [], {}
        for item in inputs:
            attempts = verified[item["variant"]]
            original = json.loads(Path(item["report"]).read_text())
            destination = output / f"{item['variant']}-report.json"
            if destination.exists():
                fresh = completed_report(destination, panel, item["variant"])
            else:
                fresh = score(manifest, study / "panel-broad.json", item["run"], destination, "validation", snapshot)
            for key in ("config", "config_sha256", "summary", "per_collection", "runtime_seconds", "graph_eligible"):
                if fresh[key] != original[key]:
                    raise ValueError("Final-selection refresh changed evidence beyond billing")
            def cases(report):
                return [{k: v for k, v in c.items() if k != "reserved_or_charged_usd"}
                        for c in report["cases"]]
            if cases(fresh) != cases(original):
                raise ValueError("Final-selection refresh changed drawing evidence beyond billing")
            if fresh["billing"] != attempts["billing"]:
                raise ValueError("Final-selection billing attribution changed")
            coverage[item["variant"]] = {k: attempts[k] for k in ("attempted_tiles", "failed_tiles")}
            refreshed.append(destination)
        save_identical(output / "coverage.json", coverage)
        return seal(study / "panel-broad.json", refreshed, selection_path, prices_path=study / "prices-v1.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "hybrids", "comparison", "manifest"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    selected = run(args.study, args.hybrids, args.comparison, args.manifest)
    print(json.dumps({"selected": selected["variant"], "selection_sha256": selected["selection_sha256"],
                      "inference_performed": False}))
