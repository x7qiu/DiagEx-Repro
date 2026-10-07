"""Secondary VLM-affirmed symbol metrics; never an engineering approval metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import SYMBOLS, digest, read_graph, save_new, sha256
from .epoch_train import sealed
from .scoring import aggregate, score_drawing


def affirmed(prediction):
    attributes = prediction.get("attributes", {})
    # Existing perception's accepted detections include scanned fallbacks.
    # This is a recognition decision, not human approval of their graph use.
    if prediction["disposition"] == "graph_eligible":
        return True
    if attributes.get("raster_vlm_decision") == "symbol" and attributes.get("raster_proposal_id"):
        return True
    # Broad discoveries have no detector ID but pass the same strict symbol
    # box/class/evidence contract before entering the review inventory.
    return (attributes.get("geometry_basis") == "vlm_broad_raster_observation"
            and attributes.get("broad_category") in SYMBOLS
            and bool(attributes.get("raster_vlm_reason")))


def evaluate(manifest_path, panel_path, run_dir, primary_report, output):
    manifest = sealed(manifest_path, "manifest_sha256")
    panel = sealed(panel_path, "panel_sha256")
    primary = json.loads(Path(primary_report).read_text())
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    split = primary["split"]
    if primary["config_sha256"] != digest(config) or primary["config"] != config:
        raise ValueError("Primary report and run configuration differ")
    if primary["manifest_sha256"] != manifest["manifest_sha256"] or panel["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Split manifest differs")
    if primary["panel_sha256"] != panel["panel_sha256"] or config["split"] != split:
        raise ValueError("Panel or split differs")
    expected = {r["id"] for r in panel["panels"][split]}
    if {r["id"] for r in primary["cases"]} != expected or len(primary["cases"]) != len(expected):
        raise ValueError("Primary case coverage differs")
    if any(r["status"] == "missing" for r in primary["cases"]):
        raise ValueError("Primary attempted coverage is unfinished")
    # Check all inference records before reading annotation files.
    raw = {}
    for case in panel["panels"][split]:
        path = run_dir / digest(case["id"])[:16] / "predictions.json"
        record = json.loads(path.read_text())
        if record["config_sha256"] != digest(config) or record["image_sha256"] != case["image_sha256"]:
            raise ValueError("Prediction provenance differs")
        raw[case["id"]] = record
    lookup = {r["id"]: r for r in manifest["drawings"]}
    cases = []
    for case in panel["panels"][split]:
        row = lookup[case["id"]]
        if row["split"] != split:
            raise ValueError("Annotation split differs")
        path = Path(manifest["dataset_root"]) / row["graph"]
        if sha256(path) != row["graph_sha256"]:
            raise ValueError("Reference annotations changed")
        record = raw[case["id"]]
        selected = [p for p in record["predictions"] if affirmed(p)]
        result = score_drawing(read_graph(path), selected)
        result.update(id=case["id"], collection=case["collection"],
                      status="partial" if record["errors"] else "complete",
                      full_inventory_count=len(record["predictions"]), affirmed_inventory_count=len(selected))
        cases.append(result)
    report = {"analysis_type": "vlm_affirmed_symbol_inventory", "reporting_only_not_candidate": True,
              "primary_report": str(Path(primary_report).resolve()), "primary_report_sha256": sha256(primary_report),
              "source_config_sha256": digest(config), "panel_sha256": panel["panel_sha256"], "split": split,
              "predicate_source_sha256": sha256(__file__), "summary": aggregate(cases), "cases": cases,
              "per_collection": {c: aggregate([r for r in cases if r["collection"] == c])
                                 for c in sorted({r["collection"] for r in cases})},
              "runtime_seconds": primary["runtime_seconds"], "billing": primary.get("billing"),
              "scope": "Secondary reporting only. Full truth denominators and failed attempts retained. Affirmed means a VLM symbol observation accepted by the existing perception parser, an explicit positive raster decision, or a valid broad discovery. Detector proposals alone do not qualify. Geometry review, legend interpretation and engineering approval remain separate. These are the same API requests, not additional spending."}
    save_new(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "panel", "run", "primary-report", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    report = evaluate(a.manifest, a.panel, a.run, a.primary_report, a.out)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
