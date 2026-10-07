"""Reuse identical completed validation inputs without repeating paid inference.

Only validation-to-validation expansion is permitted. Original predictions,
failed attempts, usage and request IDs are preserved; panel/configuration seals
are rebound with an explicit per-file derivation record. No test data is read.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import uuid
from pathlib import Path

from .data import digest, save_new, sha256
from .epoch_train import sealed
from .guidance import load_proposals
from .selection import perception_signature


def seed(source_panel, target_panel, source_run, output, registration, manifest_path, *, target_proposals=None, source_proposals=None):
    source_panel = sealed(source_panel, "panel_sha256")
    target_panel = sealed(target_panel, "panel_sha256")
    manifest = sealed(manifest_path, "manifest_sha256")
    source_run, output, registration = Path(source_run), Path(output), Path(registration)
    if output.exists():
        raise FileExistsError("Validation reuse requires a new output directory")
    if (source_run / "stop-request.json").exists():
        raise ValueError("A stopped experimental arm is not eligible for reuse")
    if source_panel["manifest_sha256"] != target_panel["manifest_sha256"] or source_panel["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Drawing-group split manifest changed")
    config = json.loads((source_run / "config.json").read_text())
    if config["split"] != "validation" or config["panel_sha256"] != source_panel["panel_sha256"]:
        raise ValueError("Only matching validation runs may be expanded")
    if config["variant"] == "supervised_detector":
        raise ValueError("This helper reuses VLM records, not detector inference")
    reg = json.loads(registration.read_text())
    if reg["panel_sha256"] != source_panel["panel_sha256"] or reg["split"] != "validation":
        raise ValueError("Registration panel or split differs")
    if reg.get("runner_variant", reg.get("variant")) != config["variant"] or reg["model"] != config["model"]:
        raise ValueError("Registered method differs")
    if not reg.get("method_fingerprints") or any(sha256(path) != expected for path, expected in reg["method_fingerprints"].items()):
        raise ValueError("Registered inference implementation changed")
    src_rows = {r["id"]: r for r in source_panel["panels"]["validation"]}
    dst_rows = {r["id"]: r for r in target_panel["panels"]["validation"]}
    if not src_rows or len(src_rows) != len(source_panel["panels"]["validation"]) or len(dst_rows) != len(target_panel["panels"]["validation"]):
        raise ValueError("Empty or duplicate validation identities")
    if not src_rows.keys() <= dst_rows.keys() or any(row != dst_rows[i] for i, row in src_rows.items()):
        raise ValueError("Target must contain the exact original validation inputs")
    truth_rows = {r["id"]: r for r in manifest["drawings"]}
    for identity, row in dst_rows.items():
        source = truth_rows.get(identity)
        if not source or source["split"] != "validation" or source["image_sha256"] != row["image_sha256"] or source["size"] != row["size"]:
            raise ValueError("Target input is not the frozen validation drawing")
    new_config = copy.deepcopy(config)
    new_config["panel_sha256"] = target_panel["panel_sha256"]
    if "guidance" in config:
        if not source_proposals or not target_proposals:
            raise ValueError("Guided reuse needs original and expanded source-only proposal runs")
        old_predictions, old_guidance = load_proposals(source_proposals, source_panel, "validation")
        new_predictions, new_guidance = load_proposals(target_proposals, target_panel, "validation")
        if old_guidance != config["guidance"]:
            raise ValueError("Original proposal provenance differs")
        if any(old_predictions[i] != new_predictions[i] for i in src_rows):
            raise ValueError("Detector hints changed for a reused drawing")
        new_config["guidance"] = new_guidance
    elif source_proposals or target_proposals:
        raise ValueError("Unguided inference must not acquire detector hints during reuse")
    if perception_signature(config) != perception_signature(new_config):
        raise ValueError("Reuse would change the inference method")
    old_hash, new_hash = digest(config), digest(new_config)
    records, price_hashes = [], set()
    for identity, row in src_rows.items():
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Reused source image changed")
        key = digest(identity)[:16]
        case = source_run / key
        summary_path = case / "predictions.json"
        if not summary_path.exists():
            raise ValueError("Source drawing remains unfinished")
        summary = json.loads(summary_path.read_text())
        if summary["id"] != identity or summary["image_sha256"] != row["image_sha256"] or summary["config_sha256"] != old_hash:
            raise ValueError("Source drawing provenance differs")
        tiles = [(p, json.loads(p.read_text())) for p in sorted(case.glob("p*-r*-c*.json"))]
        if len(tiles) != summary["tiles"] or len({t["tile_id"] for _, t in tiles}) != len(tiles):
            raise ValueError("Source tile coverage incomplete or duplicated")
        if any(t["status"] not in {"complete", "failed"} for _, t in tiles) or summary["completed_tiles"] != sum(t["status"] == "complete" for _, t in tiles):
            raise ValueError("Tile attempts are unfinished or success counts differ")
        if any(t["config_sha256"] != old_hash or p.name != t["tile_id"] + ".json" for p, t in tiles):
            raise ValueError("Source tile configuration differs")
        # Tile filenames contain numeric row/column components, but lexicographic
        # order need not equal grid order. Compare unique prediction IDs instead.
        predictions = [p for _, t in tiles for p in t["predictions"]]
        if len({p["id"] for p in predictions}) != len(predictions):
            raise ValueError("Duplicate source prediction identities")
        if sorted(predictions, key=lambda p: p["id"]) != sorted(summary["predictions"], key=lambda p: p["id"]):
            raise ValueError("Drawing summary differs from its tile predictions")
        ids = {i for _, t in tiles for i in t["ledger_request_ids"]}
        if ids != set(summary["ledger_request_ids"]):
            raise ValueError("Drawing and tile billing request provenance differs")
        errors = {t["tile_id"]: t.get("error") for _, t in tiles if t["status"] == "failed"}
        if errors != {e["tile_id"]: e["error"] for e in summary["errors"]}:
            raise ValueError("Source failed attempts differ from drawing summary")
        for path, value in [*tiles, (summary_path, summary)]:
            copied = copy.deepcopy(value)
            copied["config_sha256"] = new_hash
            copied["validation_reuse"] = {"source_record": str(path.resolve()), "source_sha256": sha256(path),
                                         "source_config_sha256": old_hash, "new_paid_request": False}
            records.append((Path(key) / path.name, copied))
            if "price_snapshot_sha256" in value:
                price_hashes.add(value["price_snapshot_sha256"])
    for value in price_hashes:
        if sha256(source_run / "price-snapshots" / (value + ".json")) != value:
            raise ValueError("Source price snapshot changed")
    staging = output.with_name(output.name + ".preparing-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    save_new(staging / "config.json", new_config)
    for relative, value in records:
        save_new(staging / relative, value)
    for value in price_hashes:
        target = staging / "price-snapshots" / (value + ".json")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source_run / "price-snapshots" / target.name, target)
    if (source_run / "transport-resume.json").exists():
        shutil.copyfile(source_run / "transport-resume.json", staging / "transport-resume.json")
    provenance = {"source_panel_sha256": source_panel["panel_sha256"], "target_panel_sha256": target_panel["panel_sha256"],
                  "source_config_sha256": old_hash, "target_config_sha256": new_hash,
                  "registration": str(registration.resolve()), "registration_sha256": sha256(registration),
                  "reused_drawings": sorted(src_rows), "unattempted_drawings": sorted(dst_rows.keys() - src_rows.keys()),
                  "record_count": len(records), "new_paid_requests": 0,
                  "failures_preserved": True, "scope": "Validation expansion only. Predictions, attempted failures, recorded runtime, usage and request IDs are unchanged. The broader report includes reused costs for its inputs; study spending must count ledger request IDs only once."}
    save_new(staging / "validation-reuse.json", provenance)
    staging.rename(output)
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-panel", "target-panel", "source-run", "out", "registration", "manifest"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--source-proposals")
    p.add_argument("--target-proposals")
    a = p.parse_args()
    report = seed(a.source_panel, a.target_panel, a.source_run, a.out, a.registration, a.manifest,
                  source_proposals=a.source_proposals, target_proposals=a.target_proposals)
    print(json.dumps({"reused": len(report["reused_drawings"]), "still_unattempted": len(report["unattempted_drawings"]), "new_paid_requests": 0}))


if __name__ == "__main__":
    main()
