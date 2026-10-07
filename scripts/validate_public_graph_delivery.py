"""Validate the graph stage on the repository's three full-graph fixtures.

Identity is supplied from fixture truth to isolate connection recovery. This is
not an end-to-end symbol benchmark or a claim of complete human ground truth.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from diagex.config import LLMConfig, RuntimeBudgets, ScanConfig, TilingConfig
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.llm.model_policy import FAST_MODEL
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import extract_page_evidence
from diagex.vision.loader import iter_pages, load
from diagex.vision.models import ReconciledGraph
from diagex.vision.page_graph import classify_page_line_evidence, solve_page_graph
from diagex.vision.topology import build_page_topology, route_evidence_failures

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/2401-delivery"
DEST = OUT / "public-graph-validation"


def run(name):
    if (DEST / f"{name}.json").exists():
        return
    source = ROOT / f"tests/p-ids-public/{name}.pdf"
    truth_path = ROOT / f"eval/datasets/{name}/graph.truth.json"
    truth = ReconciledGraph.model_validate_json(truth_path.read_text())
    page = next(iter_pages(load(source, tiling=TilingConfig(), scan_cfg=ScanConfig())))
    evidence = extract_page_evidence(page=page, source_path=source)
    evidence.role = "pid"
    frozen = dict(
        dataset=name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        truth_nodes=len(truth.nodes),
        truth_edges=len(truth.edges),
        model=FAST_MODEL,
        scope="Graph stage with supplied fixture nodes; no symbol perception. Existing fixture edges scored as provided, not independently certified complete.",
        limitations="Repository metadata calls these full annotations; source inspection shows omitted symbols, approximate/empty-space boxes and stale debug counts. No independent second-rater validation is claimed.",
        rendering=dict(
            width=page.width, height=page.height, dpi=page.dpi, is_scanned=page.is_scanned
        ),
    )
    atomic_write_json(DEST / f"{name}-input.json", frozen)
    cost = CostTracker()
    start = time.monotonic()
    cfg = replace(
        LLMConfig.from_env(),
        model=FAST_MODEL,
        vision_model=None,
        reasoning_model=None,
        escalation_model=None,
        transport="openrouter",
        production_open_weight=True,
        reasoning_mode="disabled",
        spending_category="graph",
        spending_ledger=str(OUT / "model-preflight/spending.json"),
        verified_prices=str(OUT / "model-preflight/verified-prices.json"),
    )
    result = dict(frozen)
    try:
        topology = build_page_topology(page=evidence, nodes=truth.nodes, raster_image=page.image)
        atomic_write_json(DEST / f"{name}-topology.json", topology.model_dump(mode="json"))
        visual = classify_page_line_evidence(
            client=LLMClient(cfg, budgets=RuntimeBudgets(retry_attempts=1)),
            cost_tracker=cost,
            reporter=NullReporter(),
            page=evidence,
            rendered_image=page.image,
            nodes=truth.nodes,
            topology=topology,
            step=1,
        )
        output = solve_page_graph(
            client=LLMClient(
                replace(cfg, reasoning_mode="enabled"), budgets=RuntimeBudgets(retry_attempts=1)
            ),
            cost_tracker=cost,
            reporter=NullReporter(),
            page=evidence,
            rendered_image=page.image,
            nodes=truth.nodes,
            all_nodes=truth.nodes,
            topology=topology,
            pages=[evidence],
            step=len(cost.steps) + 1,
            max_tokens=8000,
            visual_evidence=visual,
        )
        edges = [e for e in output.edges if not e.attributes.get("provisional_review_only")]

        def key(e):
            return (tuple(sorted((e.from_node, e.to_node))), e.line_type)

        expected = {key(e) for e in truth.edges}
        predicted = {key(e) for e in edges}
        failures = {
            e.id: route_evidence_failures(e, evidence)
            for e in edges
            if route_evidence_failures(e, evidence)
        }
        result.update(
            status="complete",
            metrics=dict(
                tp=len(expected & predicted),
                fp=len(predicted - expected),
                fn=len(expected - predicted),
            ),
            confirmed_edges=len(edges),
            provisional_edges=len(output.edges) - len(edges),
            source_proof_failures=failures,
            warnings=topology.warnings,
            output=output.model_dump(mode="json"),
            visual_evidence=visual.model_dump(mode="json"),
        )
        assert not failures, "Confirmed connection lacks source proof"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result.update(elapsed_s=time.monotonic() - start, usage=[asdict(s) for s in cost.steps])
    atomic_write_json(DEST / f"{name}.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"output", "visual_evidence", "usage"}},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, ["dexpi-reference", "tennessee1", "butane1"]))
