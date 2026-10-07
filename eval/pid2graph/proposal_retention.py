"""Describe proposal-associated recall loss without inventing VLM decisions.

Inputs are independently scored detector and VLM reports on identical panels.
Truth IDs are used only here, after inference, and are scoped to each drawing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import save_new, sha256


def compare_case(detector, hybrid):
    if detector["id"] != hybrid["id"] or detector["collection"] != hybrid["collection"]:
        raise ValueError("Drawing identity/collection differs")
    if "missing" in (detector["status"], hybrid["status"]):
        raise ValueError("Required drawing has not finished attempted coverage")
    d = {m["truth_id"]: m for m in detector["matches"]}
    h = {m["truth_id"]: m for m in hybrid["matches"]}
    dl = {m["truth_id"]: m for m in detector["localization_matches"]}
    hl = {m["truth_id"]: m for m in hybrid["localization_matches"]}
    retained, lost = d.keys() & h.keys(), d.keys() - h.keys()
    class_error = lost & hl.keys()
    no_localization = lost - hl.keys()
    recovered = h.keys() - d.keys()
    corrected = {identity for identity in recovered & dl.keys()
                 if dl[identity]["truth_label"] != dl[identity]["prediction_label"]}
    counts = {"detector_true_positives": len(d), "hybrid_true_positives": len(h),
              "retained_detector_truth": len(retained), "lost_detector_truth": len(lost),
              "lost_but_still_localized": len(class_error), "lost_without_localization": len(no_localization),
              "recovered_beyond_detector": len(recovered), "corrected_detector_class_errors": len(corrected)}
    return {"id": detector["id"], "collection": detector["collection"],
            "detector_status": detector["status"], "hybrid_status": hybrid["status"], "counts": counts,
            "class_aware_retention": len(retained) / len(d) if d else None,
            "lost_truth_ids": sorted(lost), "lost_but_localized_truth_ids": sorted(class_error),
            "lost_without_localization_truth_ids": sorted(no_localization),
            "recovered_truth_ids": sorted(recovered), "corrected_class_truth_ids": sorted(corrected),
            "lost_by_class": dict(Counter(d[i]["truth_label"] for i in lost))}


def compare(detector, hybrid):
    for field in ("manifest_sha256", "panel_sha256", "split"):
        if detector[field] != hybrid[field]:
            raise ValueError(f"Reports disagree on {field}")
    if detector["config"]["variant"] != "supervised_detector":
        raise ValueError("Reference must be a detector report")
    d = {r["id"]: r for r in detector["cases"]}
    h = {r["id"]: r for r in hybrid["cases"]}
    if not d or d.keys() != h.keys() or len(d) != len(detector["cases"]) or len(h) != len(hybrid["cases"]):
        raise ValueError("Reports must cover the same unique drawings")
    cases = [compare_case(d[i], h[i]) for i in sorted(d)]

    def pool(rows):
        counts = Counter()
        for r in rows:
            counts.update(r["counts"])
        n = counts["detector_true_positives"]
        return {"drawings": len(rows), "counts": dict(counts),
                "class_aware_retention": counts["retained_detector_truth"] / n if n else None,
                "drawings_with_api_failures": sum(r["hybrid_status"] == "partial" for r in rows)}

    return {"panel_sha256": detector["panel_sha256"], "split": detector["split"],
            "compared_variant": hybrid["config"]["variant"],
            "attempted_coverage_complete": True, "summary": pool(cases), "cases": cases,
            "per_collection": {c: pool([r for r in cases if r["collection"] == c]) for c in sorted({r["collection"] for r in cases})},
            "interpretation": "Shared truth IDs measure observable retention after inference, not explicit proposal acceptance or rejection. Advisory prompts do not return per-proposal decisions. Failures remain in the denominator. Localization/class-aware matching are the frozen independent confidence-greedy IoU>=0.5 assignments; duplicate predictions can affect the assignments.",
            "detector_metrics": detector["summary"], "hybrid_metrics": hybrid["summary"],
            "hybrid_billing": hybrid.get("billing"), "hybrid_runtime_seconds": hybrid["runtime_seconds"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("detector", "hybrid", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = compare(json.loads(a.detector.read_text()), json.loads(a.hybrid.read_text()))
    result["inputs"] = {"detector": {"path": str(a.detector), "sha256": sha256(a.detector)},
                        "hybrid": {"path": str(a.hybrid), "sha256": sha256(a.hybrid)}}
    save_new(a.out, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
