"""Paired offline candidate-retention evaluation; never a claim of end-to-end accuracy.

python -m diagex.knowledge.evaluate manifest.yaml --output report.yaml
Manifest rows contain saved PageEvidence, DetectionRecords, and independently labelled
expected_connector_ids. Identical evidence/candidates are replayed in all conditions.
"""

import argparse
from pathlib import Path

import yaml

from diagex.vision.evidence import PageEvidence
from diagex.vision.fusion import _prepare_detections
from diagex.vision.symbol_interpretation import DetectionRecord

from .resolver import knowledge_snapshot


def evaluate(rows, profile, overrides=None):
    report = {
        "scope": "offline candidate retention; held-fixed detections, no model inference",
        "conditions": {},
    }
    for mode in ("off", "general", "profile"):
        snapshot = knowledge_snapshot(
            mode, profile if mode == "profile" else None, overrides if mode == "profile" else None
        )
        values = []
        for row in rows:
            page = PageEvidence.model_validate(row["page"])
            candidates = [DetectionRecord.model_validate(d) for d in row["detections"]]
            expected = set(row["expected_connector_ids"])
            accepted, rejected = _prepare_detections(
                candidates, {page.page_index: page}, knowledge=snapshot
            )
            predicted = {d.id for d in accepted if d.kind == "opc"}
            values.append(
                {
                    "id": row["id"],
                    "true_positive": len(predicted & expected),
                    "false_positive": len(predicted - expected),
                    "false_negative": len(expected - predicted),
                    "recovered_ids": sorted(predicted),
                    "findings": rejected,
                    "requires_review": sum(
                        bool(d.attributes.get("knowledge_exception")) for d in accepted
                    ),
                }
            )
        totals = {
            key: sum(v[key] for v in values)
            for key in ("true_positive", "false_positive", "false_negative", "requires_review")
        }
        report["conditions"][mode] = {
            "knowledge_identity": snapshot.get("identity"),
            "totals": totals,
            "drawings": values,
        }
    baseline = report["conditions"]["off"]["totals"]
    for result in report["conditions"].values():
        result["delta_true_positive"] = (
            result["totals"]["true_positive"] - baseline["true_positive"]
        )
        result["delta_false_positive"] = (
            result["totals"]["false_positive"] - baseline["false_positive"]
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = yaml.safe_load(args.manifest.read_text())
    result = evaluate(source["drawings"], source["profile"], source.get("overrides"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
