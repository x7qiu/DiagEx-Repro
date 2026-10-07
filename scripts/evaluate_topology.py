"""Offline native-vector topology replay. Never runs models or modifies a run.

Fixture metrics use reviewed graph endpoints with original native PDF paths.
They isolate line reconstruction from perception, and match undirected local
endpoint pairs; unmatched predictions are not automatically annotation errors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import fitz

from diagex.vision.evidence import PageEvidence, extract_page_evidence
from diagex.vision.fusion import FusionResult
from diagex.vision.models import DiagramPage, ReconciledGraph
from diagex.vision.topology import TOPOLOGY_VERSION, build_page_topology, route_evidence_failures
from diagex.vision.vector_geometry import symbol_contours, vector_arrows

ROOT = Path(__file__).resolve().parents[1]


def replay(pages, nodes, out):
    results, summary = [], []
    for page in pages:
        if page.role != "pid":
            continue
        local = [
            n
            for n in nodes
            if n.page_index == page.page_index and n.kind in {"equipment", "instrument", "opc"}
        ]
        result = build_page_topology(page=page, nodes=local)
        eligible = [e for e in result.edges if not route_evidence_failures(e, page)]
        summary.append(
            {
                "page": page.page_index + 1,
                "nodes": len(local),
                "structurally_supported_candidates": len(eligible),
                "provisional_candidates": len(result.edges) - len(eligible),
                "grouped_alternatives": sum(
                    max(0, len(e.attributes.get("route_alternatives", [])) - 1)
                    for e in result.edges
                ),
                "invalid_route_proposals": sum(
                    len(c.get("rejected_routes", [])) for c in result.ambiguities
                ),
                "native_contours": len(symbol_contours(page, local)),
                "native_arrows": len(vector_arrows(page, local)),
                "directed_candidates": sum(
                    e.attributes.get("flow_direction") == "forward" for e in eligible
                ),
            }
        )
        results.extend(result.edges)
        (out / f"page-{page.page_index + 1:04d}.json").write_text(result.model_dump_json(indent=2))
    return results, summary


def fixture(name, out):
    pdf = ROOT / "tests/p-ids-public" / f"{name}.pdf"
    graph = ReconciledGraph.model_validate_json(
        (ROOT / "eval/datasets" / name / "graph.truth.json").read_text()
    )
    pages = []
    with fitz.open(pdf) as doc:
        for index, native in enumerate(doc):
            dpi = min(300.0, 6000 * 72 / max(native.rect.width, native.rect.height))
            rendered = DiagramPage(
                page_index=index,
                width=math.ceil(native.rect.width * dpi / 72),
                height=math.ceil(native.rect.height * dpi / 72),
                dpi=dpi,
                effective_dpi=dpi,
                is_scanned=False,
                source_ref=f"{name}#page={index + 1}",
            )
            pages.append(extract_page_evidence(page=rendered, source_path=pdf, pdf_page=native))
    edges, summary = replay(pages, graph.nodes, out)
    expected = {frozenset((e.from_node, e.to_node)) for e in graph.edges if not e.cross_sheet}
    eligible = [e for e in edges if not e.attributes.get("provisional_review_only")]
    predicted = {frozenset((e.from_node, e.to_node)) for e in eligible}
    return {
        "fixture": name,
        "pages": summary,
        "reviewed_pairs": len(expected),
        "matched_pairs": len(expected & predicted),
        "missed_pairs": len(expected - predicted),
        "unmatched_predicted_pairs": len(predicted - expected),
        "missing_pairs": [sorted(p) for p in sorted(expected - predicted, key=lambda p: sorted(p))],
        "extra_pairs": [sorted(p) for p in sorted(predicted - expected, key=lambda p: sorted(p))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    objects = parser.add_mutually_exclusive_group()
    objects.add_argument(
        "--objects", type=Path, help="FusionResult JSON replayed from the same saved detections"
    )
    objects.add_argument(
        "--graph", type=Path, help="Use existing graph nodes, retaining contextual decisions"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.run_dir and out.is_relative_to(args.run_dir.resolve()):
        parser.error("output must be outside the source run")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "port_topology_version": TOPOLOGY_VERSION,
        "model_calls": 0,
        "scope": "structural topology only; not accepted semantic graph or end-to-end accuracy",
        "fixtures": [],
    }
    for name in ["dexpi-reference", "tennessee1"]:
        target = out / name
        target.mkdir(exist_ok=True)
        report["fixtures"].append(fixture(name, target))
    if args.run_dir:
        if not (args.objects or args.graph):
            parser.error("--run-dir requires --objects or --graph")
        target = out / "run"
        target.mkdir(exist_ok=True)
        pages = [
            PageEvidence.model_validate_json(p.read_text())
            for p in sorted((args.run_dir / "evidence").glob("page-*.json"))
        ]
        nodes = (
            ReconciledGraph.model_validate_json(args.graph.read_text()).nodes
            if args.graph
            else FusionResult.model_validate_json(args.objects.read_text()).graph.nodes
        )
        _, report["run_pages"] = replay(pages, nodes, target)
        report["source_run"] = str(args.run_dir.resolve())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
