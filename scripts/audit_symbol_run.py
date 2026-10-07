"""Read-only scoring of saved extractions against a frozen reviewed instance set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluate_symbol_perception import score
from score_symbol_reasoning import correct_family

from diagex.extractors.evidence_checkpoint import atomic_write_json


def evaluate(run, truth):
    bundle = json.loads((run / "detection.json").read_text())
    detections = bundle["detections"]
    geometry = score(truth, detections)
    by_id = {d["id"]: d for d in detections}
    matched = {tid: by_id[did] for did, tid in geometry["matching_iou50"].items()}
    results = [
        {
            "truth": t,
            "detection": matched.get(t["id"]),
            "localized": t["id"] in matched,
            "correct": t["id"] in matched and correct_family(t, matched[t["id"]]),
        }
        for t in truth
    ]
    categories = {
        c: {
            "total": sum(t["category"] == c for t in truth),
            "localized": sum(r["truth"]["category"] == c and r["localized"] for r in results),
            "correct": sum(r["truth"]["category"] == c and r["correct"] for r in results),
        }
        for c in sorted({t["category"] for t in truth})
    }
    counts = Counter(d.get("attributes", {}).get("symbol_candidate_id") for d in detections)
    candidates = {c["id"]: c for c in bundle["candidates"]}
    published = {cid for cid in counts if cid}
    selected = {r["candidate_id"] for r in bundle["reviews"] if r["status"] == "selected"}
    cost = json.loads((run / "cost.json").read_text())
    manifest = json.loads((run / "checkpoints/manifest.json").read_text())
    completed = set(manifest["completed"].get("perception", []))
    stale = []
    for path in (run / "checkpoints/perception").glob("*.json"):
        checkpoint = json.loads(path.read_text())
        if path.stem not in completed or checkpoint.get("symbol_perception_version") != manifest[
            "stage_versions"
        ].get("symbol_perception"):
            stale.append(path.name)
    stats = {
        "detections": len(detections),
        "native_candidates": len(candidates),
        "truth_count": len(truth),
        "localized": len(matched),
        "correct_family": sum(r["correct"] for r in results),
        "categories": categories,
        "reviews": dict(Counter(r["status"] for r in bundle["reviews"])),
        "native_reviews": dict(
            Counter(r["status"] for r in bundle["reviews"] if r.get("candidate_id") in candidates)
        ),
        "unanchored_review_proposals": sum(not r.get("candidate_id") for r in bundle["reviews"]),
        "duplicate_native_reviews": {
            cid: n
            for cid, n in Counter(r.get("candidate_id") for r in bundle["reviews"]).items()
            if cid in candidates and n > 1
        },
        "duplicate_native_detections": {cid: n for cid, n in counts.items() if cid and n > 1},
        "detection_review_disagreement": sorted(published ^ selected),
        "detections_with_changed_native_geometry": [
            d["id"]
            for d in detections
            if (cid := d.get("attributes", {}).get("symbol_candidate_id")) in candidates
            and d["bbox"] != candidates[cid]["bbox"]
        ],
        "manifest_status": manifest["status"],
        "completed_crops": len(completed),
        "stale_checkpoint_files": stale,
        "manifest_errors": manifest["errors"],
        "stage_versions": manifest["stage_versions"],
        "runtime_seconds": cost["wall_clock_s"],
        "transport_retries": cost["retries"],
        "output_tokens": cost["output_tokens"],
        "legend_entries": len(bundle["legend_pack"]["entries"]),
        "legend_row_status": dict(
            Counter(
                e.get("attributes", {}).get("row_status", "not_applicable")
                for e in bundle["legend_pack"]["entries"]
            )
        ),
        "legend_coverage": dict(
            Counter(c["status"] for c in bundle["legend_pack"].get("coverage", []))
        ),
    }
    return {
        "run": str(run),
        "source_sha256": bundle["source_sha256"],
        "stats": stats,
        "geometry": geometry,
        "native_geometry": score(truth, bundle["candidates"]),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = [
        dict(t, page_index=t["page"] - 1)
        for t in json.loads(args.truth.read_text())
        if t["category"] != "valve_callout"
    ]
    rows = [evaluate(run, truth) for run in args.runs]
    assert len({r["source_sha256"] for r in rows}) == 1, "Comparisons require the same drawing"
    atomic_write_json(
        args.output,
        {
            "review_file": str(args.truth),
            "review_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
            "scope": "Reviewed subset, not full-document precision or subtype/tag accuracy",
            "runs": rows,
        },
    )
    for row in rows:
        print(json.dumps({"run": row["run"], **row["stats"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
