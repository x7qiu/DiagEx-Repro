"""Offline bootstrap for the second PID2Graph study; never reads new test truth."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data import SYMBOLS, digest, read_graph, save_new, sha256


def bootstrap(previous, output):
    previous, output = Path(previous), Path(output)
    if output.exists():
        raise FileExistsError("Use a new study directory")
    manifest = json.loads((previous / "manifest-v1.json").read_text())
    old_panel = json.loads((previous / "panel-v1.json").read_text())
    assert digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest["manifest_sha256"]
    assert digest({k: v for k, v in old_panel.items() if k != "panel_sha256"}) == old_panel["panel_sha256"]
    lookup = {r["id"]: r for r in manifest["drawings"]}
    # Exclude whole groups used by any existing test inference output, not only
    # the nominal original panel. Configuration discovery reads no test labels.
    exposed = {r["id"] for r in old_panel["panels"]["test"]}
    scanned_test_runs = []
    for path in previous.rglob("config.json"):
        config = json.loads(path.read_text())
        if config.get("split") != "test":
            continue
        scanned_test_runs.append(str(path))
        for summary in path.parent.glob("*/predictions.json"):
            exposed.add(json.loads(summary.read_text())["id"])
    groups = {lookup[identity]["group"] for identity in exposed}
    candidates = [r for r in manifest["drawings"] if r["split"] == "test" and r["group"] not in groups]
    selected = []
    for collection in sorted({r["collection"] for r in candidates}):
        rows = sorted((r for r in candidates if r["collection"] == collection),
                      key=lambda r: digest(["phase2-final-2026-09-19", r["id"]]))
        if len(rows) < 8:
            raise ValueError("Insufficient unused test drawings")
        selected.extend(rows[:8])
    assert len(selected) == 16

    def source(row):
        return {"id": row["id"], "collection": row["collection"], "size": row["size"],
                "image": str(Path(manifest["dataset_root"]) / row["image"]),
                "image_sha256": row["image_sha256"]}

    panels = {**old_panel["panels"], "test": [source(r) for r in selected]}
    for name, validation in (("panel-pilot.json", old_panel["panels"]["validation"]),
                             ("panel-broad.json", [source(r) for r in manifest["drawings"] if r["split"] == "validation"])):
        value = {"manifest_sha256": manifest["manifest_sha256"],
                 "panels": {**panels, "validation": validation},
                 "sampling": "Original group splits; phase2 final ID-hash order, eight per collection, excludes every previously used test group.",
                 "previous_test_ids_excluded": sorted(exposed)}
        value["panel_sha256"] = digest(value)
        save_new(output / name, value)
    save_new(output / "freeze.json", {"source_manifest_sha256": manifest["manifest_sha256"],
             "prior_test_runs_scanned": scanned_test_runs, "excluded_test_groups": sorted(groups),
             "remaining_test_candidates": len(candidates), "new_test_drawings": 16,
             "new_test_truth_read": False, "new_test_images_read": False})

    training = json.loads((previous / "detector-data-v1/training.json").read_text())
    cfg = json.loads((previous / "detector-v2/config.json").read_text())
    order = list(range(len(training["samples"])))
    random.Random(cfg["seed"]).shuffle(order)
    presentations = cfg["steps"] * cfg["batch_size"]
    assert presentations < len(order), "This reconstruction covers only the pilot's first epoch"
    presented = set(order[:presentations])
    by_source = defaultdict(list)
    for i, sample in enumerate(training["samples"]):
        by_source[sample["source_id"]].append((i, sample))
    totals, prepared, seen, full = (Counter() for _ in range(4))
    per_drawing = []
    for identity, samples in sorted(by_source.items()):
        row = lookup[identity]
        assert row["split"] == "train"
        graph_path = Path(manifest["dataset_root"]) / row["graph"]
        assert sha256(graph_path) == row["graph_sha256"]
        nodes = [n for n in read_graph(graph_path)["nodes"] if n["label"] in SYMBOLS]
        covered, actual, complete = set(), set(), set()
        for index, sample in samples:
            x, y = sample["origin"]
            w = min(training["crop_size"], row["size"][0])
            h = min(training["crop_size"], row["size"][1])
            for n in nodes:
                a, b, c, d = n["bbox"]
                intersection = max(0, min(c, x + w) - max(a, x)) * max(0, min(d, y + h) - max(b, y))
                area = (c - a) * (d - b)
                if intersection >= 0.5 * area and intersection > 4:
                    covered.add(n["id"])
                    if index in presented:
                        actual.add(n["id"])
                        if abs(intersection - area) < 1e-6:
                            complete.add(n["id"])
        for n in nodes:
            label = n["label"]
            totals[label] += 1
            prepared[label] += n["id"] in covered
            seen[label] += n["id"] in actual
            full[label] += n["id"] in complete
        per_drawing.append({"id": identity, "symbols": len(nodes), "prepared_coverage": len(covered),
                            "presented_coverage": len(actual), "presented_full_box_coverage": len(complete)})
    save_new(output / "pilot-training-coverage.json", {
        "training_sha256": training["training_sha256"], "presentations": presentations,
        "prepared_crops": len(order), "epochs": presentations / len(order),
        "definitions": {"represented": "At least 50% box area retained, matching original crop preparation.",
                        "fully_visible": "Entire reference box occurs in at least one presented crop."},
        "per_class": {c: {"total": totals[c], "prepared": prepared[c], "presented": seen[c],
                          "presented_full_box": full[c]} for c in SYMBOLS},
        "totals": {"symbols": sum(totals.values()), "prepared": sum(prepared.values()),
                   "presented": sum(seen.values()), "presented_full_box": sum(full.values())},
        "drawings": per_drawing, "validation_or_test_truth_read": False,
    })
    save_new(output / "checkpoint.json", {"state": "bootstrap_complete", "paid_calls": 0,
             "next": ["Fix and verify billing ledger before paid calls", "Prepare annotation-covering training crops",
                      "Benchmark local throughput; freeze epoch cap and early stopping before training"],
             "goal_complete": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default="output/pid2graph-development")
    parser.add_argument("--out", default="output/pid2graph-phase2")
    args = parser.parse_args()
    bootstrap(args.previous, args.out)
