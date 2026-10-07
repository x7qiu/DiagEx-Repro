"""Paired graph-context test on a frozen, source-reviewed page-7 sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from diagex.config import LLMConfig, RuntimeBudgets
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.llm.model_policy import FAST_MODEL
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import ReconciledEdge, ReconciledNode
from diagex.vision.page_graph import PageLineEvidence, solve_page_graph
from diagex.vision.topology import TopologyResult

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/2401-delivery"
RUN = ROOT / "runs/2401/2026-09-09T00-58-04_qwen-qwen3.5-35b-a3b_r-8707"
DEST = OUT / "context-comparison"


def read(path):
    return json.loads(path.read_text())


def prepare():
    if (DEST / "sample.json").exists():
        raise ValueError("Frozen connection sample already exists")
    nodes = read(RUN / "checkpoints/assembly/objects.json")["nodes"]
    topology = read(RUN / "checkpoints/topology/page-0007.json")
    truth = {e["id"]: "process" for e in topology["edges"]}
    # Source image context-page7.png: all fourteen sampled routes follow visible
    # process pipes, including the two OPC routes with unsupported native ports.
    # Three isolated endpoint pairs are negative controls for invented shortcuts.
    local = [n for n in nodes if n["page_index"] == 6]
    by_label = {n["label"]: n for n in local if n["label"]}
    negative_pairs = [("PG 00501", "PT 00502"), ("PG 00501", "FT 00502"), ("PSV 002", "TE 00502")]
    for index, (left, right) in enumerate(negative_pairs):
        a, b = by_label[left], by_label[right]
        edge = ReconciledEdge(
            id=f"negative-{index}",
            from_node=a["id"],
            to_node=b["id"],
            line_type="process",
            confidence="low",
            attributes={
                "evaluation_negative_control": True,
                "route_evidence": {"status": "uncertain", "failures": ["no_drawn_direct_route"]},
            },
        )
        topology["edges"].append(edge.model_dump(mode="json"))
        truth[edge.id] = None
    sample = dict(
        page_index=6,
        nodes=nodes,
        topology=topology,
        truth=truth,
        visual_evidence=read(RUN / "checkpoints/line_evidence/page-0007.json"),
        pages=[read(p) for p in sorted((RUN / "evidence").glob("page-*.json"))],
        legend=read(OUT / "audit/reviewed-snapshot.json")["legend_pack"]["entries"],
        process_context=read(OUT / "process-context.json"),
        repeats=2,
        source_audit={
            "actor_type": "agent",
            "source_image": "evaluation/context-page7.png",
            "scope": "14 sampled direct route candidates and 3 explicitly unsupported shortcut controls; not whole-page connection recall",
            "image_sha256": hashlib.sha256(
                (OUT / "evaluation/context-page7.png").read_bytes()
            ).hexdigest(),
        },
        scoring="Exact sampled edge ID plus source-audited line type; provisional edges and hypotheses excluded. Model suggestions outside the frozen sample are reported separately, not counted as true positives.",
        model=FAST_MODEL,
        conditions=["without-context", "with-context"],
    )
    atomic_write_json(DEST / "sample.json", sample)
    print(json.dumps({"positive_routes": 14, "negative_controls": 3}), flush=True)


def run_case(condition, repeat):
    dest = DEST / f"{condition}-{repeat}.json"
    if dest.exists():
        return
    sample = read(DEST / "sample.json")
    nodes = [ReconciledNode.model_validate(n) for n in sample["nodes"]]
    pages = [PageEvidence.model_validate(p) for p in sample["pages"]]
    page = next(p for p in pages if p.page_index == 6)
    cfg = replace(
        LLMConfig.from_env(),
        transport="openrouter",
        model=FAST_MODEL,
        vision_model=None,
        reasoning_model=None,
        escalation_model=None,
        production_open_weight=True,
        reasoning_mode="enabled",
        spending_ledger=str(OUT / "model-preflight/spending.json"),
        verified_prices=str(OUT / "model-preflight/verified-prices.json"),
        spending_category="context",
    )
    cost = CostTracker()
    started = time.monotonic()
    result = dict(
        condition=condition,
        repeat=repeat,
        sample_sha256=hashlib.sha256((DEST / "sample.json").read_bytes()).hexdigest(),
    )
    try:
        output = solve_page_graph(
            client=LLMClient(cfg, budgets=RuntimeBudgets(retry_attempts=1)),
            cost_tracker=cost,
            reporter=NullReporter(),
            page=page,
            rendered_image=Image.open(OUT / "audit/page-07.png"),
            nodes=[n for n in nodes if n.page_index == 6],
            all_nodes=nodes,
            topology=TopologyResult.model_validate(sample["topology"]),
            pages=pages,
            step=1,
            max_tokens=8000,
            visual_evidence=PageLineEvidence.model_validate(sample["visual_evidence"]),
            legend_summary=sample["legend"],
            process_context=sample["process_context"] if condition == "with-context" else [],
        )
        predicted = {
            e.id: e.line_type
            for e in output.edges
            if not e.attributes.get("provisional_review_only")
        }
        tp = sum(
            sample["truth"].get(i) == typ and typ is not None
            for i, typ in predicted.items()
            if i in sample["truth"]
        )
        fp = sum(
            sample["truth"].get(i) != typ for i, typ in predicted.items() if i in sample["truth"]
        )
        fn = sum(typ is not None and predicted.get(i) != typ for i, typ in sample["truth"].items())
        result.update(
            status="complete",
            metrics={"tp": tp, "fp": fp, "fn": fn},
            output=output.model_dump(mode="json"),
            out_of_sample_confirmed=[i for i in predicted if i not in sample["truth"]],
        )
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    result.update(elapsed_s=time.monotonic() - started, usage=[asdict(s) for s in cost.steps])
    atomic_write_json(dest, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"output", "usage"}}, ensure_ascii=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(
                pool.map(
                    lambda job: run_case(*job),
                    [(c, r) for r in (1, 2) for c in ("without-context", "with-context")],
                )
            )
