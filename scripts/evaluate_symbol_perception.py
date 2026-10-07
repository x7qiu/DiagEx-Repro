"""Offline native candidate coverage; no model calls and no source-run mutation.

This evaluates geometry proposals, NOT end-to-end detector accuracy. The optional
reviewed audit corpus and public fixture annotations can be incomplete and use
different body/actuator conventions. Unmatched proposals are not false positives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import fitz

from diagex.vision.evidence import PageEvidence, extract_page_evidence
from diagex.vision.models import BBox, DiagramPage
from diagex.vision.symbol_candidates import SYMBOL_PERCEPTION_VERSION, symbol_candidates

ROOT = Path(__file__).resolve().parents[1]


def score(truth: list[dict], candidates: list[dict]) -> dict:
    def overlap(t, c):
        return (
            BBox(**t["bbox"]).iou(BBox(**c["bbox"])) if t["page_index"] == c["page_index"] else 0.0
        )

    def matching(threshold):
        neighbors = {
            t["id"]: sorted(
                ((overlap(t, c), c["id"]) for c in candidates if overlap(t, c) >= threshold),
                reverse=True,
            )
            for t in truth
        }
        owners = {}

        def augment(tid, seen):
            for _, cid in neighbors[tid]:
                if cid in seen:
                    continue
                seen.add(cid)
                if cid not in owners or augment(owners[cid], seen):
                    owners[cid] = tid
                    return True
            return False

        for tid in sorted(neighbors, key=lambda tid: (len(neighbors[tid]), tid)):
            augment(tid, set())
        return owners

    matches = {str(t): matching(t) for t in [0.3, 0.5, 0.75]}
    return {
        "reviewed_instances": len(truth),
        "candidate_count": len(candidates),
        "matches": {t: len(rows) for t, rows in matches.items()},
        "categories": {
            cat: {
                "reviewed": sum(t["category"] == cat for t in truth),
                "matched_iou50": sum(
                    t["category"] == cat and t["id"] in matches["0.5"].values() for t in truth
                ),
            }
            for cat in sorted({t["category"] for t in truth})
        },
        "candidate_duplicates_iou50": sum(
            max(0, sum(overlap(t, c) >= 0.5 for c in candidates) - 1) for t in truth
        ),
        "candidates_overlapping_multiple_reviewed_instances_iou30": sum(
            sum(overlap(t, c) >= 0.3 for t in truth) > 1 for c in candidates
        ),
        "unmatched_instance_ids_iou50": [
            t["id"] for t in truth if t["id"] not in matches["0.5"].values()
        ],
        "matching_iou50": matches["0.5"],
        "warning": "Candidate geometry coverage only; not final detection accuracy or precision.",
    }


def fixture(stem: str) -> tuple[dict, list[dict]]:
    pdf = ROOT / "tests/p-ids-public" / f"{stem}.pdf"
    cs = []
    with fitz.open(pdf) as doc:
        for i, p in enumerate(doc):
            dpi = min(300.0, 6000 * 72 / max(p.rect.width, p.rect.height))
            page = DiagramPage(
                page_index=i,
                width=math.ceil(p.rect.width * dpi / 72),
                height=math.ceil(p.rect.height * dpi / 72),
                dpi=dpi,
                effective_dpi=dpi,
                is_scanned=False,
                source_ref=f"{stem}#page={i + 1}",
            )
            evidence = extract_page_evidence(page=page, source_path=pdf, pdf_page=p)
            cs.extend(c.model_dump(mode="json") for c in symbol_candidates(evidence))
    truth = []
    for line in (
        (ROOT / "eval/datasets" / stem / "annotations.truth.jsonl").read_text().splitlines()
    ):
        t = json.loads(line)
        if t["kind"] not in {"equipment", "instrument", "opc"}:
            continue
        truth.append(
            {
                "id": t["id"],
                "page_index": t["page_index"],
                "bbox": t["bbox_global"],
                "category": t["attributes"].get("equipment_class", t["kind"]),
            }
        )
    return score(truth, cs), cs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.run_dir) != bool(args.audit_dir):
        parser.error("--run-dir and --audit-dir must be supplied together")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"version": SYMBOL_PERCEPTION_VERSION, "fixtures": {}}
    for name in ["dexpi-reference", "tennessee1"]:
        metrics, cs = fixture(name)
        report["fixtures"][name] = metrics
        (args.out / f"{name}.candidates.json").write_text(json.dumps(cs, indent=2))
    if args.run_dir:
        truth = [
            dict(t, page_index=t["page"] - 1)
            for t in json.loads((args.audit_dir / "reviewed-instances.json").read_text())
            if t["category"] != "valve_callout"
        ]
        cs = []
        for f in sorted((args.run_dir / "evidence").glob("page-*.json")):
            page = PageEvidence.model_validate_json(f.read_text())
            cs.extend(c.model_dump(mode="json") for c in symbol_candidates(page))
        report["reviewed_run"] = score(truth, cs)
        report["reviewed_run"]["source_run"] = str(args.run_dir)
        (args.out / "run.candidates.json").write_text(json.dumps(cs, ensure_ascii=False, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
