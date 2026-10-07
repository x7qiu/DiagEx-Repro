"""Apply the frozen four-pilot promotion rule; never authorize final test inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagex.llm.billing import summarize

from .data import digest, save_new, sha256
from .epoch_train import sealed
from .scoring import aggregate
from .selection import ranking_key


def run(study, candidates, ledger_path, output):
    study = Path(study)
    policy = sealed(study / "validation-promotion-policy.json", "policy_sha256")
    registry_path = study / "controlled-hybrids.json"
    if (sha256(registry_path) != policy["hybrid_registry_sha256"]
            or sha256(Path(__file__).with_name("selection.py")) != policy["selection_source_sha256"]):
        raise ValueError("Registered methods or ranking implementation changed")
    registry = json.loads(registry_path.read_text())
    registrations = {}
    for item in registry["registered"]:
        path = study / item["registration"]
        reg = json.loads(path.read_text())
        registrations[reg["runner_variant"]] = (reg, path)
    if len(registrations) != 4 or len(candidates) != 4 or not registry["all_slots_registered"]:
        raise ValueError("All four registered methods are required")
    panel = sealed(study / "panel-pilot.json", "panel_sha256")
    if panel["panel_sha256"] != policy["pilot_panel_sha256"]:
        raise ValueError("Pilot panel differs from promotion policy")
    expected = {r["id"]: r for r in panel["panels"]["validation"]}
    ledger = json.loads(Path(ledger_path).read_text())
    results, seen = [], set()
    for report_path, run_path in candidates:
        report_path, run_path = Path(report_path), Path(run_path)
        report = json.loads(report_path.read_text())
        config = json.loads((run_path / "config.json").read_text())
        variant = config["variant"]
        if variant not in registrations or variant in seen:
            raise ValueError("Unexpected or duplicate pilot variant")
        seen.add(variant)
        reg, reg_path = registrations[variant]
        if (config != report["config"] or digest(config) != report["config_sha256"]
                or report["split"] != "validation" or config["split"] != "validation"
                or report["panel_sha256"] != panel["panel_sha256"]
                or report["manifest_sha256"] != panel["manifest_sha256"] or report.get("stopped")):
            raise ValueError("Report does not match completed pilot configuration")
        for key in ("panel_sha256", "model", "provider_tags", "guidance"):
            if config[key] != reg[key]:
                raise ValueError(f"Registration differs: {key}")
        for name, expected_hash in reg["method_fingerprints"].items():
            if sha256(name) != expected_hash:
                raise ValueError(f"Registered source changed: {name}")
        cases = {c["id"]: c for c in report["cases"]}
        if cases.keys() != expected.keys() or len(cases) != len(report["cases"]):
            raise ValueError("Pilot drawing coverage differs")
        if aggregate(report["cases"]) != report["summary"]:
            raise ValueError("Reported aggregate differs from drawing scores")
        identities, attempted, failures = [], 0, 0
        for identity, row in expected.items():
            case = cases[identity]
            directory = run_path / digest(identity)[:16]
            summary = json.loads((directory / "predictions.json").read_text())
            tiles = [json.loads(p.read_text()) for p in directory.glob("p*-r*-c*.json")]
            errors = sum(t["status"] == "failed" for t in tiles)
            if (summary["config_sha256"] != digest(config) or summary["image_sha256"] != row["image_sha256"]
                    or len(tiles) != summary["tiles"] or not tiles
                    or any(t["config_sha256"] != digest(config) or t["status"] not in {"complete", "failed"} for t in tiles)
                    or errors != len(summary["errors"])
                    or case["status"] != ("partial" if errors else "complete")):
                raise ValueError("Pilot has incomplete or inconsistent attempted coverage")
            tile_ids = {i for t in tiles for i in t["ledger_request_ids"]}
            if set(summary["ledger_request_ids"]) != tile_ids:
                raise ValueError("Drawing billing attribution differs from attempts")
            identities.extend(summary["ledger_request_ids"])
            attempted += len(tiles)
            failures += errors
        if len(identities) != len(set(identities)):
            raise ValueError("Request was attributed to multiple drawings")
        if summarize(ledger, identities) != report["billing"]:
            raise ValueError("Report billing is stale; rescore before ranking")
        if report["billing"]["active_reservations_usd"]:
            raise ValueError("A pilot still has active requests")
        results.append((report, {"variant": variant, "slot": reg["slot"], "report": str(report_path),
            "report_sha256": sha256(report_path), "run": str(run_path),
            "registration_sha256": sha256(reg_path), "summary": report["summary"],
            "billing": report["billing"], "runtime_seconds": report["runtime_seconds"],
            "attempted_tiles": attempted, "failed_tiles": failures}))
    ranked = [r for _, r in sorted(results, key=lambda pair: ranking_key(pair[0]))]
    result = {"policy_sha256": policy["policy_sha256"], "panel_sha256": panel["panel_sha256"],
              "billing_snapshot_sha256": sha256(ledger_path), "ranked": ranked,
              "shortlisted_variants": [r["variant"] for r in ranked[:2]],
              "stage": "Pilot algorithm shortlist only; not final model selection",
              "remaining": [policy["checkpoint_comparison"], policy["broader_validation"], policy["final_gate"]],
              "test_access_authorized": False}
    result["shortlist_sha256"] = digest(result)
    save_new(output, result)
    print(json.dumps({"shortlisted": result["shortlisted_variants"], "shortlist_sha256": result["shortlist_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "ledger", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--candidate", nargs=2, action="append", required=True, metavar=("REPORT", "RUN"))
    args = parser.parse_args()
    run(args.study, args.candidate, args.ledger, args.out)
