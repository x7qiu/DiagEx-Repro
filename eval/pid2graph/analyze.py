"""Reproduce descriptive analysis from frozen score reports, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_report(path):
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def analyze(report, *, paired=None):
    cases = report["cases"]
    by_id = {c["id"]: c for c in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate report case IDs")
    confusions = Counter(
        (m["truth_label"], m["prediction_label"])
        for c in cases for m in c["localization_matches"]
        if m["truth_label"] != m["prediction_label"]
    )
    correct = sum(c["classification_given_localization"]["correct"] for c in cases)
    localized = sum(c["classification_given_localization"]["total"] for c in cases)
    result = {
        "split": report["split"],
        "summary": report["summary"],
        "per_collection": report["per_collection"],
        "case_statuses": dict(Counter(c["status"] for c in cases)),
        "classification_given_localization": {
            "correct": correct, "localized": localized,
            "accuracy": correct / localized if localized else None,
        },
        "confusions": [
            {"truth": t, "prediction": p, "count": n}
            for (t, p), n in sorted(confusions.items(), key=lambda item: (-item[1], item[0]))
        ],
        "runtime_seconds": report["runtime_seconds"],
        "reserved_or_charged_usd": report["reserved_or_charged_usd"],
        "limitations": [
            "Descriptive analysis of saved score reports; no inference or scoring changes.",
            "Conditional classification uses independent class-agnostic matches.",
            "Costs include unresolved reservations and are not actual invoices.",
            "Case coverage and failed tiles must be reported separately from accuracy.",
        ],
    }
    if "billing" in report:
        result["billing"] = report["billing"]
        result["limitations"][2] = (
            "billing separates actual billed usage, active reservations, and unresolved upper bounds; "
            "reserved_or_charged_usd is legacy budget exposure, never an actual bill."
        )
    if paired is not None:
        for key in ("manifest_sha256", "panel_sha256", "split"):
            if report[key] != paired[key]:
                raise ValueError(f"Paired reports differ in {key}")
        other = {c["id"]: c for c in paired["cases"]}
        if len(other) != len(paired["cases"]) or set(other) != set(by_id):
            raise ValueError("Paired reports must have identical unique case IDs")
        rows = []
        for c in cases:
            b = other[c["id"]]
            if c["symbols"]["actual"] != b["symbols"]["actual"]:
                raise ValueError("Paired reports have different truth denominators")
            rows.append({
                "id": c["id"],
                "drawing_f1_delta": (c["symbols"]["f1"] or 0) - (b["symbols"]["f1"] or 0),
                "tp_delta": c["symbols"]["tp"] - b["symbols"]["tp"],
                "fp_delta": c["symbols"]["fp"] - b["symbols"]["fp"],
            })
        result["paired_drawings"] = rows
        result["paired_direction"] = "report minus paired reference; descriptive, not significance testing"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--run", type=Path, help="Optional raw tile checkpoints for failure counts")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report, report_hash = read_report(args.report)
    paired, paired_hash = read_report(args.paired) if args.paired else (None, None)
    result = analyze(report, paired=paired)
    result.update(report_sha256=report_hash, paired_report_sha256=paired_hash)
    if args.run:
        tiles = [json.loads(p.read_text()) for p in sorted(args.run.glob("*/*-r*-c*.json"))]
        result["tiles"] = dict(Counter(t["status"] for t in tiles))
        result["failure_types"] = dict(Counter(
            t.get("error", "unknown").split(":", 1)[0]
            for t in tiles if t["status"] == "failed"
        ))
        result["raw_tile_checkpoint_directory"] = str(args.run)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
