"""Score a frozen reasoning experiment using the existing one-to-one geometry matcher."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_symbol_perception import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reasoning-ab-r5427"


def correct_family(truth, detection):
    category = truth["category"]
    kind = detection["kind"]
    attrs = detection.get("attributes", {})
    cls = str(attrs.get("equipment_class", "")).lower().strip().replace(" ", "_")
    if category == "instrument":
        return kind == "instrument"
    if category == "opc":
        return kind == "opc"
    if category == "motor":
        return kind == "equipment" and cls in {"motor", "electric_motor"}
    if category == "screw_compressor":
        return kind == "equipment" and cls in {
            "compressor",
            "screw_compressor",
            "dry_screw_compressor",
        }
    if category == "vessel":
        return kind == "equipment" and cls in {
            "vessel",
            "vertical_vessel",
            "horizontal_vessel",
            "tank",
            "air_receiver",
            "receiver",
        }
    if category == "valve":
        return kind == "equipment" and (bool(attrs.get("valve_type")) or cls == "valve")
    raise ValueError(category)


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    cases = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "cases").glob("*.json"))}
    truth = [t for c in cases.values() for t in c["truth"]]
    assert len({t["id"] for t in truth}) == len(truth)
    records = [json.loads(p.read_text()) for p in sorted((OUT / "responses").glob("*.json"))]
    groups = defaultdict(list)
    for r in records:
        groups[(r["condition"], r["repeat"])].append(r)
    output = {
        "truth_count": len(truth),
        "categories": dict(Counter(t["category"] for t in truth)),
        "runs": [],
    }
    for (condition, repeat), rs in sorted(groups.items()):
        expected_tiles = {
            tid
            for tid in cases
            if repeat < manifest.get("case_repeats", {}).get(tid, manifest["repeats"])
        }
        completed_tiles = {r["tile_id"] for r in rs if "response" in r or "error" in r}
        evaluated_truth = [t for tid in sorted(completed_tiles) for t in cases[tid]["truth"]]
        detections = [d for r in rs for d in r.get("detections", [])]
        bydet = {d["id"]: d for d in detections}
        geometry = score(evaluated_truth, detections)
        bytruth = {tid: bydet[did] for did, tid in geometry["matching_iou50"].items()}
        results = [
            {
                "truth_id": t["id"],
                "category": t["category"],
                "localized": t["id"] in bytruth,
                "correct": t["id"] in bytruth and correct_family(t, bytruth[t["id"]]),
                "detection": bytruth.get(t["id"]),
            }
            for t in evaluated_truth
        ]
        usage = Counter()
        for r in rs:
            for k, v in r.get("response", {}).get("usage", {}).items():
                if isinstance(v, (int, float)):
                    usage[k] += v
        output["runs"].append(
            {
                "condition": condition,
                "repeat": repeat,
                "calls": len(rs),
                "evaluated_truth_count": len(evaluated_truth),
                "complete_comparison": completed_tiles == expected_tiles,
                "completed": sum(r.get("http_status") == 200 for r in rs),
                "parse_errors": sum(bool(r.get("parse_error")) for r in rs),
                "transport_errors": sum(bool(r.get("error")) and "response" not in r for r in rs),
                "localized": sum(r["localized"] for r in results),
                "correct": sum(r["correct"] for r in results),
                "category_correct": dict(Counter(r["category"] for r in results if r["correct"])),
                "category_localized": dict(
                    Counter(r["category"] for r in results if r["localized"])
                ),
                "thinking_chars": sum(r.get("thinking_chars", 0) for r in rs),
                "reasoning_tokens": sum(
                    r.get("response", {})
                    .get("usage", {})
                    .get("output_tokens_details", {})
                    .get("thinking_tokens", 0)
                    for r in rs
                ),
                "latency_seconds": sum(r.get("wall_seconds", 0) for r in rs),
                "usage": dict(usage),
                "results": results,
                "geometry": geometry,
            }
        )
    output["pairs"] = []
    for repeat in range(2):
        off = next(
            (r for r in output["runs"] if r["condition"] == "off" and r["repeat"] == repeat), None
        )
        on = next(
            (r for r in output["runs"] if r["condition"] == "on" and r["repeat"] == repeat), None
        )
        if not off or not on or not off["complete_comparison"] or not on["complete_comparison"]:
            continue
        off_results = {r["truth_id"]: r for r in off["results"]}
        gained = [
            r["truth_id"]
            for r in on["results"]
            if r["correct"] and not off_results[r["truth_id"]]["correct"]
        ]
        lost = [
            r["truth_id"]
            for r in on["results"]
            if not r["correct"] and off_results[r["truth_id"]]["correct"]
        ]
        output["pairs"].append(
            {"repeat": repeat, "improved_with_reasoning": gained, "regressed_with_reasoning": lost}
        )
    (OUT / "scores.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("TRUTH", output["truth_count"], output["categories"])
    for r in output["runs"]:
        print({k: v for k, v in r.items() if k not in {"results", "geometry"}})
    print("PAIRS", output["pairs"])


if __name__ == "__main__":
    main()
