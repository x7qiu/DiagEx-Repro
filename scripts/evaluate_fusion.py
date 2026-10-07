"""Offline fusion audit; never calls a model or overwrites the source run.

Usage: python scripts/evaluate_fusion.py --run-dir runs/... --out tmp/fusion-audit
The public fixture check perturbs reviewed boxes to simulate repeated crop
observations. It measures instance matching, not end-to-end perception accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import fitz

from diagex.vision.evidence import PageEvidence, extract_page_evidence
from diagex.vision.fusion import fuse_objects
from diagex.vision.instance_matching import FUSION_VERSION, match_instances
from diagex.vision.legend_models import LegendPack
from diagex.vision.models import BBox, DiagramPage
from diagex.vision.perception import DetectionRecord

ROOT = Path(__file__).resolve().parents[1]


def fixture_audit(stem: str) -> dict:
    pdf = ROOT / "tests" / "p-ids-public" / f"{stem}.pdf"
    annotations = ROOT / "eval" / "datasets" / stem / "annotations.truth.jsonl"
    pages = {}
    with fitz.open(pdf) as doc:
        for index, pdf_page in enumerate(doc):
            dpi = min(300.0, 6000 * 72 / max(pdf_page.rect.width, pdf_page.rect.height))
            page = DiagramPage(
                page_index=index,
                width=math.ceil(pdf_page.rect.width * dpi / 72),
                height=math.ceil(pdf_page.rect.height * dpi / 72),
                dpi=dpi,
                effective_dpi=dpi,
                is_scanned=False,
                source_ref=f"{stem}#page={index + 1}",
            )
            pages[index] = extract_page_evidence(page=page, source_path=pdf, pdf_page=pdf_page)
    detections = []
    truth_for_detection = {}
    for row in annotations.read_text().splitlines():
        item = json.loads(row)
        if item["kind"] not in {"equipment", "instrument", "opc"}:
            continue
        box = BBox.model_validate(item["bbox_global"])
        page = pages[item["page_index"]]
        if box.x2 > page.width or box.y2 > page.height:
            raise ValueError(f"{stem}: annotation coordinates do not match native page frame")
        for copy in range(2):
            moved = box.model_copy(
                update={
                    "x": min(page.width - box.w, box.x + copy * max(1, round(box.w * 0.02))),
                    "y": min(page.height - box.h, box.y + copy * max(1, round(box.h * 0.02))),
                }
            )
            detection_id = f"{item['id']}:crop-{copy}"
            detections.append(
                DetectionRecord(
                    id=detection_id,
                    tile_id=f"crop-{copy}",
                    page_index=item["page_index"],
                    kind=item["kind"],
                    label=item.get("label", ""),
                    bbox=moved,
                    confidence="high",
                    attributes=item.get("attributes", {}),
                )
            )
            truth_for_detection[detection_id] = item["id"]
    clusters, conflicts = match_instances(detections, pages)
    outputs_for_truth = defaultdict(set)
    false_merges = 0
    for index, cluster in enumerate(clusters):
        truths = {truth_for_detection[d.id] for d in cluster.detections}
        false_merges += len(truths) > 1
        for truth in truths:
            outputs_for_truth[truth].add(index)
    return {
        "fixture": stem,
        "evaluation": "reviewed instance boxes plus 2% crop jitter",
        "truth_instance_count": len(set(truth_for_detection.values())),
        "observation_count": len(detections),
        "output_instance_count": len(clusters),
        "false_merge_count": false_merges,
        "retained_duplicate_count": sum(len(v) - 1 for v in outputs_for_truth.values()),
        "review_conflict_count": len(conflicts),
        "native_path_count": sum(len(p.paths) for p in pages.values()),
    }


def connectivity(graph: dict) -> dict:
    edges = graph["edges"]
    accepted = [e for e in edges if not e.get("attributes", {}).get("provisional_review_only")]
    attached = {n for e in accepted for n in (e["from_node"], e["to_node"])}
    return {
        "accepted_edges": len(accepted),
        "provisional_edges": len(edges) - len(accepted),
        "accepted_isolated_nodes": sum(n["id"] not in attached for n in graph["nodes"]),
    }


def audit_run(run: Path, out: Path) -> dict:
    pages = [
        PageEvidence.model_validate_json(p.read_text())
        for p in sorted((run / "evidence").glob("page-*.json"))
    ]
    detections = [
        DetectionRecord.model_validate(d)
        for p in sorted((run / "checkpoints/perception").glob("*.json"))
        for d in json.loads(p.read_text()).get("detections", [])
    ]
    if not pages or not detections:
        raise ValueError("run must contain native evidence and perception checkpoints")
    old = json.loads((run / "checkpoints/assembly/objects.json").read_text())["nodes"]
    fused = fuse_objects(
        source_name=json.loads((run / "graph.json").read_text())["source_path"],
        pages=pages,
        detections=detections,
        per_page_status={p.page_index: "ok" for p in pages},
        legend_pack=LegendPack.model_validate_json((run / "legend.json").read_text()),
    )
    new = [n.model_dump(mode="json") for n in fused.graph.nodes]
    vessel_checks = []
    for detection in detections:
        if detection.attributes.get("equipment_class") != "vessel":
            continue
        before = next((n for n in old if detection.id in n["source_annotation_ids"]), None)
        after = next((n for n in new if detection.id in n["source_annotation_ids"]), None)
        vessel_checks.append(
            {
                "detection_id": detection.id,
                "page": detection.page_index + 1,
                "label": detection.label,
                "observation_bbox": detection.bbox.model_dump(),
                "before_class": before["attributes"].get("equipment_class") if before else None,
                "before_valve_type": before["attributes"].get("valve_type") if before else None,
                "after_class": after["attributes"].get("equipment_class") if after else None,
                "after_bbox": after["bbox_global"] if after else None,
            }
        )
    (out / "objects.replayed.json").write_text(fused.model_dump_json(indent=2))
    return {
        "source_run": str(run),
        "raw_observation_count": len(detections),
        "before_fused_count": len(old),
        "after_fused_count": len(new),
        "fusion_conflicts": dict(Counter(c["type"] for c in fused.ambiguities)),
        "original_connectivity": connectivity(json.loads((run / "graph.json").read_text())),
        "vessel_observation_checks": vessel_checks,
        "limitation": "Fusion-only replay; downstream solver was not rerun. No new accepted-connectivity score.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.run_dir and out.is_relative_to(args.run_dir.resolve()):
        parser.error("output must be outside the original run")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "fusion_version": FUSION_VERSION,
        "fixtures": [fixture_audit(s) for s in ("dexpi-reference", "tennessee1")],
    }
    if args.run_dir:
        report["run"] = audit_run(args.run_dir.resolve(), out)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = [
        "# Fusion validation",
        "",
        f"Fusion version: {FUSION_VERSION}. No model calls.",
        "",
        "## Vector fixture checks",
        "",
        "Reviewed instance boxes were duplicated with 2% position jitter. These are fusion checks, "
        "not end-to-end extraction accuracy measurements.",
        "",
        "| Fixture | Instances | Observations | False merges | Retained duplicates |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["fixtures"]:
        lines.append(
            f"| {item['fixture']} | {item['truth_instance_count']} | "
            f"{item['observation_count']} | {item['false_merge_count']} | "
            f"{item['retained_duplicate_count']} |"
        )
    if "run" in report:
        result = report["run"]
        lines.extend(
            [
                "",
                "## Saved run replay",
                "",
                f"Source: `{result['source_run']}`",
                "",
                f"Replayed {result['raw_observation_count']} saved observations. "
                f"Fusion output changed from {result['before_fused_count']} to "
                f"{result['after_fused_count']} objects; an increased count alone is not an accuracy score.",
                "",
                "Tagged vessel observations previously merged into valve objects:",
                "",
                "| Page | Observation label | Before | After | Original observation box retained |",
                "|---|---|---|---|---|",
            ]
        )
        for item in result["vessel_observation_checks"]:
            if item["label"] and item["before_valve_type"]:
                lines.append(
                    f"| {item['page']} | {item['label']} | Valve ({item['before_valve_type']}) | "
                    f"{item['after_class']} | {item['observation_bbox'] == item['after_bbox']} |"
                )
        baseline = result["original_connectivity"]
        lines.extend(
            [
                "",
                f"Original graph: {baseline['accepted_edges']} accepted connections, "
                f"{baseline['provisional_edges']} provisional connections, and "
                f"{baseline['accepted_isolated_nodes']} objects isolated on accepted connections.",
                "",
                result["limitation"],
                "",
                "Unproven crop merges remain separate review candidates. DEXPI duplicate-ID errors "
                "are outside this change. The original run artifacts were not modified.",
            ]
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(out / "report.md"), "fixtures": report["fixtures"]}, indent=2))


if __name__ == "__main__":
    main()
