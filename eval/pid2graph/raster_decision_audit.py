"""Audit explicit raster decisions from source-only proposal and tile records."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import digest, save_new, sha256
from .epoch_train import sealed
from .guidance import load_proposals


def summarize_case(proposals, tiles, predictions):
    lookup = {p["id"]: p for p in proposals}
    requested = set()
    statuses = defaultdict(set)
    returned_counts = Counter()
    no_response_views = 0
    unknown_rows = 0
    omitted = 0
    for tile in tiles:
        ids = set(tile["requested_raster_proposal_ids"])
        if not ids <= lookup.keys():
            raise ValueError("Requested IDs do not belong to the source detector")
        requested.update(ids)
        omitted += tile["raster_guides_omitted_by_limit"]
        attempts = tile["raster_review_attempts"]
        results = [r for a in attempts for r in a["tool_results"]]
        if not results:
            no_response_views += len(ids)
        for result in results:
            unknown_rows += len(result.get("unrecognized_rows", []))
            for d in result.get("decisions", []):
                if d["proposal_id"] not in ids:
                    raise ValueError("Decision identity not present in the view")
                statuses[d["proposal_id"]].add(d["status"])
                returned_counts[d["status"]] += 1
    retained = {p["attributes"]["raster_proposal_id"] for p in predictions if p.get("attributes", {}).get("raster_vlm_decision") == "symbol"}
    if not retained <= lookup.keys():
        raise ValueError("Retained predictions have unknown source proposal IDs")
    recognized = {i for i, s in statuses.items() if "symbol" in s}
    if not retained <= recognized:
        raise ValueError("Retained prediction lacks an explicit symbol decision")
    rejected_only = {i for i, s in statuses.items() if s == {"reject"}}
    conflict = {i for i, s in statuses.items() if "symbol" in s and "reject" in s}
    unresolved = requested - recognized - rejected_only
    groups = {"never_requested": set(lookup) - requested, "recognized": recognized,
              "retained_in_output": retained, "recognized_without_retained_output": recognized - retained,
              "rejected_only": rejected_only, "unresolved_without_recognition": unresolved,
              "recognition_rejection_conflict": conflict}
    return {"detector_proposals": len(lookup), "unique_requested": len(requested),
            "counts": {k: len(v) for k, v in groups.items()}, "ids": {k: sorted(v) for k, v in groups.items()},
            "decision_rows_across_attempts_and_views": dict(returned_counts),
            "proposal_views_without_response": no_response_views, "unrecognized_output_rows": unknown_rows,
            "guide_views_omitted_by_limit": omitted,
            "per_detector_class": {label: {k: sum(lookup[i]["label"] == label for i in v) for k, v in groups.items()}
                                   for label in sorted({p["label"] for p in proposals})}}


def run(panel_path, proposal_run, run_dir, output, split):
    panel = sealed(panel_path, "panel_sha256")
    proposals, guidance = load_proposals(proposal_run, panel, split)
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    if config["panel_sha256"] != panel["panel_sha256"] or config["split"] != split or config["variant"] not in {"explicit_proposal_decisions", "proposal_source_crops", "broad_raster_recognition"}:
        raise ValueError("Raster-review run configuration differs")
    if config["guidance"] != guidance:
        raise ValueError("Proposal provenance differs from the inference run")
    cases = []
    for row in panel["panels"][split]:
        directory = run_dir / digest(row["id"])[:16]
        summary = json.loads((directory / "predictions.json").read_text())
        if summary["config_sha256"] != digest(config) or summary["image_sha256"] != row["image_sha256"]:
            raise ValueError("Drawing summary provenance differs")
        tiles = [json.loads(p.read_text()) for p in sorted(directory.glob("p*-r*-c*.json"))]
        if len(tiles) != summary["tiles"] or any(t["config_sha256"] != digest(config) for t in tiles):
            raise ValueError("Required tile coverage is unfinished or has changed")
        cases.append({"id": row["id"], "collection": row["collection"], "failed_tiles": len(summary["errors"]),
                      **summarize_case(proposals[row["id"]], tiles, summary["predictions"])})
    counts = Counter()
    for case in cases:
        counts.update(case["counts"])
    report = {"panel_sha256": panel["panel_sha256"], "split": split, "attempted_coverage_complete": True,
              "drawings": len(cases), "counts": dict(counts), "cases": cases,
              "run_config_sha256": sha256(run_dir / "config.json"),
              "interpretation": "Explicit decisions are VLM observations, not ground truth. IDs are scoped per drawing. Multiple views/attempts may disagree. Recognized-without-output can reflect geometry/ownership filtering and is not automatically a rejection. Confirmed raster outputs still require engineering review; discovery outputs without detector IDs are excluded here."}
    save_new(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "proposal-run", "run", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    a = p.parse_args()
    report = run(a.panel, a.proposal_run, a.run, a.out, a.split)
    print(json.dumps({"drawings": report["drawings"], "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
