"""Choose each shortlisted hybrid's checkpoint for broader validation.

Offline only: validates completed attempts, refreshes billing from one snapshot,
and emits source registrations accepted by reuse_validation. No inference, final
selection, or test access is authorized here. Reconcile available bills first.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from diagex.llm.billing import summarize

from .__main__ import score
from .checkpoint_comparison import prepare as verify_comparison
from .data import digest, save_new, sha256
from .epoch_train import sealed
from .selection import ranking_key


def choose_pair(pilot, trained):
    """Preserve pilot-first order on an exact tie under the frozen ranking rule."""
    for report in (pilot, trained):
        if (report["split"] != "validation" or report.get("stopped")
                or any(c["status"] not in {"complete", "partial"} for c in report["cases"])
                or "billing" not in report or report["billing"]["active_reservations_usd"]):
            raise ValueError("Checkpoint ranking requires finished, accounted validation reports")
    if ({k: v for k, v in pilot["config"].items() if k != "guidance"}
            != {k: v for k, v in trained["config"].items() if k != "guidance"}
            or pilot["manifest_sha256"] != trained["manifest_sha256"]
            or pilot["panel_sha256"] != trained["panel_sha256"]
            or [c["id"] for c in pilot["cases"]] != [c["id"] for c in trained["cases"]]):
        raise ValueError("Checkpoint comparison changed inputs or method beyond detector guidance")
    return min(range(2), key=lambda i: ranking_key((pilot, trained)[i]))


def verify_attempts(run, panel, ledger):
    """Failures count as attempts; missing tiles and active calls cannot rank."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    config_hash = digest(config)
    if ((run / "stop-request.json").exists() or config["split"] != "validation"
            or config["panel_sha256"] != panel["panel_sha256"]):
        raise ValueError("Source is stopped or does not match the validation panel")
    requests = {r["id"]: r for r in ledger["requests"]}
    if ledger.get("schema_version") != 2 or len(requests) != len(ledger["requests"]):
        raise ValueError("Authoritative unique version-2 billing records are required")
    identities, attempted, failed = [], 0, 0
    expected = panel["panels"]["validation"]
    if not expected or len({r["id"] for r in expected}) != len(expected):
        raise ValueError("Validation panel is empty or duplicated")
    for row in expected:
        directory = run / digest(row["id"])[:16]
        path = directory / "predictions.json"
        if not path.is_file():
            raise ValueError("A validation drawing is still unattempted")
        drawing = json.loads(path.read_text())
        tiles = [(p, json.loads(p.read_text())) for p in directory.glob("p*-r*-c*.json")]
        if (drawing["id"] != row["id"] or drawing["image_sha256"] != row["image_sha256"]
                or sha256(row["image"]) != row["image_sha256"]
                or drawing["config_sha256"] != config_hash or not tiles
                or len(tiles) != drawing["tiles"]
                or any(t["config_sha256"] != config_hash or t["status"] not in {"complete", "failed"}
                       or p.name != t["tile_id"] + ".json" for p, t in tiles)):
            raise ValueError("Source image/configuration or tile coverage differs")
        tile_predictions = [p for _, t in tiles for p in t["predictions"]]
        if (len({p["id"] for p in tile_predictions}) != len(tile_predictions)
                or sorted(tile_predictions, key=lambda p: p["id"])
                != sorted(drawing["predictions"], key=lambda p: p["id"])):
            raise ValueError("Drawing predictions differ from immutable tile attempts")
        errors = {t["tile_id"]: t.get("error") for _, t in tiles if t["status"] == "failed"}
        if (errors != {e["tile_id"]: e["error"] for e in drawing["errors"]}
                or drawing["completed_tiles"] != len(tiles) - len(errors)):
            raise ValueError("Failed attempts or completion counts differ")
        ids = {i for _, t in tiles for i in t["ledger_request_ids"]}
        if ids != set(drawing["ledger_request_ids"]) or len(ids) != len(drawing["ledger_request_ids"]):
            raise ValueError("Drawing billing IDs differ from tile attempts")
        if any(i not in requests or requests[i]["status"] == "pending"
               or requests[i]["category"] != "experiments" or requests[i]["model"] != config["model"] for i in ids):
            raise ValueError("Unknown, active or foreign billing request in completed comparison")
        identities.extend(drawing["ledger_request_ids"])
        attempted += len(tiles)
        failed += len(errors)
    if len(identities) != len(set(identities)):
        raise ValueError("Request is attributed to multiple drawings")
    return {"attempted_tiles": attempted, "failed_tiles": failed,
            "billing": summarize(ledger, identities), "request_ids": sorted(identities)}


def run(study, comparison, manifest, ledger_path, output):
    study, comparison, output = Path(study), Path(comparison), Path(output)
    status_path = comparison / "status.json"
    if (not status_path.is_file()
            or json.loads(status_path.read_text())["stage"] != "comparisons_complete"):
        raise ValueError("Both trained-checkpoint comparisons must finish before promotion")
    if output.exists():
        raise FileExistsError("Choose a new offline promotion directory; records are immutable")
    plan = verify_comparison(study, comparison)
    policy = sealed(study / "validation-promotion-policy.json", "policy_sha256")
    shortlist = sealed(study / "pilot-shortlist.json", "shortlist_sha256")
    panel = sealed(study / "panel-pilot.json", "panel_sha256")
    broad = sealed(study / "panel-broad.json", "panel_sha256")
    if broad["panel_sha256"] != policy["broad_panel_sha256"] or policy["maximum_broader_hybrids"] != 2:
        raise ValueError("Broader validation policy changed")
    ledger = json.loads(Path(ledger_path).read_text())
    registrations = {}
    for entry in json.loads((study / "controlled-hybrids.json").read_text())["registered"]:
        path = study / entry["registration"]
        registration = json.loads(path.read_text())
        registrations[registration["runner_variant"]] = (path, registration)
    sources = []
    for original in shortlist["ranked"][:2]:
        variant = original["variant"]
        arm = next(a for a in plan["arms"] if a["variant"] == variant)
        trained_path = comparison / f"{variant}-report.json"
        trained = json.loads(trained_path.read_text())
        if trained["config_sha256"] != arm["config_sha256"]:
            raise ValueError("Trained comparison report differs from its registration")
        pair = [(Path(original["report"]), Path(original["run"])), (trained_path, Path(arm["run"]))]
        checked = [verify_attempts(directory, panel, ledger) for _, directory in pair]
        sources.append((variant, pair, checked))
    snapshot = output / "billing-snapshot.json"
    save_new(snapshot, ledger)
    choices = []
    for variant, pair, checked in sources:
        rescored, paths = [], []
        for label, (report_path, directory), attempts in zip(("pilot", "trained"), pair, checked, strict=True):
            path = output / f"{variant}-{label}-report.json"
            fresh = score(manifest, study / "panel-pilot.json", directory, path, "validation", snapshot)
            original = json.loads(report_path.read_text())
            for key in ("config", "config_sha256", "summary", "per_collection", "runtime_seconds", "graph_eligible"):
                if fresh[key] != original[key]:
                    raise ValueError("Rescoring changed predictions, metrics or runtime")
            if fresh["billing"] != attempts["billing"]:
                raise ValueError("Scoring billing attribution differs from verified attempts")
            rescored.append(fresh)
            paths.append(path)
        selected = choose_pair(*rescored)
        source_report, source_run = pair[selected]
        config = rescored[selected]["config"]
        original_path, original_registration = registrations[variant]
        record = copy.deepcopy(original_registration)
        proposal_run = (comparison / f"{variant}-registration.json")
        trained_registration = sealed(proposal_run, "registration_sha256")
        proposal_run = trained_registration["proposal_run"] if selected else original_registration["proposal_run"]
        record.update(guidance=config["guidance"], proposal_run=proposal_run,
            status="checkpoint_selected_for_broader_validation", test_access=False,
            original_registration_sha256=sha256(original_path), policy_sha256=policy["policy_sha256"],
            source_run=str(source_run), source_report=str(source_report),
            source_report_sha256=sha256(source_report), refreshed_report=str(paths[selected]),
            refreshed_report_sha256=sha256(paths[selected]), source_config_sha256=digest(config),
            selection_rule=policy["tie_policy"], selected_weight_source=("pilot", "trained")[selected])
        record["registration_sha256"] = digest(record)
        registration_path = output / f"{variant}-source-registration.json"
        save_new(registration_path, record)
        choices.append({"variant": variant, "weight_source": record["selected_weight_source"],
            "source_run": str(source_run), "source_proposals": str(proposal_run),
            "registration": str(registration_path), "registration_sha256": record["registration_sha256"],
            "reports": [str(p) for p in paths], "selected_report": str(paths[selected]),
            "summary": rescored[selected]["summary"], "billing": rescored[selected]["billing"],
            "attempted_tiles": checked[selected]["attempted_tiles"], "failed_tiles": checked[selected]["failed_tiles"]})
    result = {"policy_sha256": policy["policy_sha256"], "shortlist_sha256": shortlist["shortlist_sha256"],
        "billing_snapshot_sha256": sha256(snapshot), "pilot_panel_sha256": panel["panel_sha256"],
        "broad_panel_sha256": broad["panel_sha256"], "choices": choices,
        "scope": "Checkpoint selection for two shortlisted hybrids only. Baseline remains required. Generate/verify broader detector hints, then use validation reuse and existing frozen drivers. No final selection or test authorization.",
        "test_access_authorized": False}
    result["promotion_sha256"] = digest(result)
    save_new(output / "promotion.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "comparison", "manifest", "ledger", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args.study, args.comparison, args.manifest, args.ledger, args.out)
    print(json.dumps({"choices": [{"variant": r["variant"], "weight_source": r["weight_source"]}
                                 for r in result["choices"]], "test_access_authorized": False}))
