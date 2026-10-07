"""Build the reviewed 2401 draft once, with the shared production budget."""

import json
from dataclasses import replace
from pathlib import Path

from diagex.config import Config, RuntimeBudgets
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.extractors.pid_evidence import run_pid_evidence_extract
from diagex.llm.model_policy import apply_production_profile
from diagex.vision.process_context import context_document

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/2401-delivery"


def main():
    target = OUT / "build-result.json"
    if target.exists():
        raise ValueError("Delivery build already recorded; inspect it before creating another run")
    snapshot = json.loads((OUT / "audit/reviewed-snapshot.json").read_text())
    provenance = json.loads((OUT / "audit/provenance.json").read_text())
    cfg = apply_production_profile(Config())
    cfg.llm = replace(
        cfg.llm,
        spending_ledger=str(OUT / "model-preflight/spending.json"),
        verified_prices=str(OUT / "model-preflight/verified-prices.json"),
        spending_category="graph",
        reasoning_mode="auto",
    )
    cfg.budgets = RuntimeBudgets(retry_attempts=2)
    cfg.runs_dir = ROOT / "runs"
    cfg.process_context = [
        context_document(
            "Source audit of pages 4–12 identifies compressor packages, interstage coolers, air buffer vessels, dryer/filter packages, instrument air and nitrogen generation/analysis equipment. Parallel A/B packages and repeated printed identifiers occur. The source has inconsistent package and vessel identifiers: preserve each printed string and resolve continuations from explicit drawing references, never silently renumber. Source annotations and legend pages 1–3 take precedence over this overview.",
            source=f"agent source audit: {provenance['source_sha256']}#pages=4-12",
            kind="process_overview",
        ),
        context_document(
            "Engineering context for hypotheses only: compressed-air systems commonly proceed through compression, cooling, buffering, drying and filtration before downstream consumers. Nitrogen generation packages may receive treated compressed air and have product analysis/control. A pressure or temperature measurement generally refers to its visibly connected process branch; a matching loop identifier can motivate inspection but does not prove a signal wire. A plausible process sequence must never establish a drawn edge. Require continuous source strokes with supported endpoint ports, or explicit compatible off-page references. Treat unmarked crossing lines as separate and controller wiring symbols as representations rather than additional physical valves.",
            source="Agent-authored generic engineering assumptions; not a source drawing fact or approved design rule",
            kind="engineering_rules",
        ),
    ]
    atomic_write_json(OUT / "process-context.json", cfg.process_context)
    result = run_pid_evidence_extract(
        diagram=Path(provenance["source"]),
        symbol_standard="isa-5.1",
        legend_path=None,
        legend_pages=[1, 2, 3],
        legend_region=None,
        no_legend=False,
        legend_key=None,
        effort="medium",
        config=cfg,
        persist=True,
        fresh=True,
        out_path=OUT / "2401.dexpi.json",
        confidence_report_path=OUT / "confidence.html",
        console=None,
        reviewed_inputs=snapshot,
    )
    atomic_write_json(
        target,
        dict(
            run_dir=str(result.run_dir),
            run_id=result.run_id,
            quality_status=result.quality_status,
            graph_nodes=len(result.graph.nodes),
            graph_edges=len(result.graph.edges),
            dexpi_stats=result.dexpi_stats,
            dexpi_issues=result.dexpi_issues,
            validation_issues=result.validation_issues,
            cost_summary=result.cost_summary,
            source_sha256=snapshot["source_sha256"],
            review_revision=snapshot["revision"],
            review_origin=snapshot["review_origin"],
        ),
    )
    print(target.read_text(), flush=True)


if __name__ == "__main__":
    main()
