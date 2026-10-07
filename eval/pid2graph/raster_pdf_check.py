"""Live PDF-to-symbol-review integration check; no accuracy or approval claim."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from diagex.config import Config, LLMConfig, PricingConfig
from diagex.extractors.pid_evidence import run_pid_evidence_extract
from diagex.review.detection import DetectionReviewStore
from diagex.vision.loader import iter_pages, load
from diagex.vision.pdf_raster_guidance import load_pdf_guidance
from diagex.vision.raster_pipeline import implementation_sha256
from diagex.vision.tiling import AspectAwareStrategy, tile

from .data import digest, save_new, sha256
from .transfer_accounting import transfer_billing


def run(source, proposals, output, ledger, prices):
    source, proposals, output = Path(source).resolve(), Path(proposals).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new check directory; a live check is never silently repeated")
    model = "deepseek/deepseek-v4.1-flash"
    price = json.loads(Path(prices).read_text())["models"][model]
    payload = json.loads(proposals.read_text())
    category = "raster_pdf_check_" + digest(str(output))[:16]
    cfg = Config(runs_dir=output / "runs", llm=LLMConfig(
        transport="openrouter", model=model, vision_model=model, reasoning_model=model,
        escalation_model=None, production_open_weight=False, reasoning_mode="disabled",
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        spending_ledger=str(Path(ledger).resolve()), verified_prices=str(Path(prices).resolve()),
        spending_category=category))
    cfg.pid.engine = "evidence-v2"
    cfg.tiling.max_page_dim_px = payload["render_settings"]["max_page_dim_px"]
    cfg.tiling.target_dpi = payload["render_settings"]["target_dpi"]
    cfg.symbol_perception.request_timeout_s = 120
    cfg.raster_proposals = load_pdf_guidance(proposals, source, cfg)
    cfg.raster_symbol_mode = "broad_review"
    cfg.pricing = PricingConfig(input_per_mtok=price["input_per_token"] * 1e6,
        output_per_mtok=price["output_per_token"] * 1e6,
        cache_read_per_mtok=price["input_per_token"] * 1e6,
        cache_write_per_mtok=price["input_per_token"] * 1e6)
    # This harness is intentionally bounded to one fully covered source page.
    if payload["page_count"] != 1 or [p["page_index"] for p in payload["pages"]] != [0]:
        raise ValueError("This check requires a one-page PDF with page-zero proposals")
    page = next(iter_pages(load(source, cfg.tiling, cfg.scan)))
    grid = tile(page, AspectAwareStrategy(cfg.tiling.max_tokens_per_tile,
                    cfg.tiling.overlap_frac, cfg.tiling.token_per_pixel))
    config = {"source": str(source), "source_sha256": sha256(source),
        "proposals": str(proposals), "proposals_sha256": sha256(proposals),
        "model": model, "provider_tags": price["provider_tags"], "billing_category": category,
        "prices_sha256": sha256(prices), "implementation_sha256": implementation_sha256(),
        "source_hashes": {p: sha256(p) for p in [
            __file__, "src/diagex/extractors/pid_evidence.py", "src/diagex/vision/pdf_raster_guidance.py"]},
        "detector": payload["detector"], "render_settings": payload["render_settings"],
        "rendered_size": [page.width, page.height], "expected_tiles": len(grid),
        "legend_policy": "No source legend supplied; resolve built-in ISA definitions before symbol perception",
        "scope": "Live software integration through the public evidence extraction entry point. No GraphML or held-out test access, no extraction accuracy or human/engineering approval claim."}
    save_new(output / "config.json", config)
    before = {r["id"] for r in json.loads(Path(ledger).read_text())["requests"]}
    started = time.monotonic()
    report = {"config_sha256": digest(config), "scope": config["scope"], "status": "failed"}
    try:
        result = run_pid_evidence_extract(diagram=source, symbol_standard="isa-5.1",
            legend_path=None, legend_pages=None, legend_region=None, no_legend=True,
            legend_key=None, effort="low", config=cfg, persist=True, fresh=True,
            out_path=None, confidence_report_path=None, console=None, stop_after="detection")
        run_dir = result.run_dir
        saved = json.loads((run_dir / "result.json").read_text())
        bundle = json.loads((run_dir / "detection.json").read_text())
        review = DetectionReviewStore(run_dir).read()
        observations = [r for r in bundle["reviews"] if r.get("object", {}).get("kind") == "raster_symbol"]
        diagnostics = list(run_dir.rglob("perception_diagnostics/*.json"))
        manifest = json.loads((run_dir / "checkpoints/manifest.json").read_text())
        completed = manifest.get("completed", {}).get("perception", [])
        checks = {"source_unchanged": sha256(source) == config["source_sha256"],
            "source_bound_review": bundle["source_sha256"] == config["source_sha256"],
            "stopped_at_symbol_review": result.workflow_stage == "detection" and not saved.get("stop_reason"),
            "all_tiles_checkpointed": len(completed) == len(grid),
            "broad_observations_persisted": bool(observations),
            "all_symbols_pending": all(s["status"] == "pending" for s in review["symbols"]),
            "no_unreviewed_graph_nodes": len(result.graph.nodes) == 0,
            "legend_prepared": result.legend_source == "no_legend"}
        report.update(status="passed" if all(checks.values()) else "incomplete", checks=checks,
            run_dir=str(run_dir), workflow_stage=result.workflow_stage, quality_status=result.quality_status,
            stop_reason=saved.get("stop_reason"), legend_source=result.legend_source,
            legend_entries=result.legend_entry_count, raw_detections=len(bundle["detections"]),
            broad_observations=len(observations), pending_symbols=len(review["symbols"]),
            suggested_kinds=dict(Counter(s["detection"]["kind"] for s in review["symbols"])),
            interpretation_decisions=dict(Counter(r.get("legend_interpretation", {}).get("decision", "absent") for r in observations)),
            completed_tiles=len(completed), diagnostic_artifacts=len(diagnostics),
            human_reviewed=False)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rows = [r for r in json.loads(Path(ledger).read_text())["requests"]
                if r["id"] not in before and r["category"] == category]
        ids = [r["id"] for r in rows]
        report.update(runtime_seconds=time.monotonic() - started, ledger_request_ids=ids,
                      **transfer_billing(ledger, ids))
        save_new(output / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "proposals", "out", "ledger", "prices"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.source, args.proposals, args.out, args.ledger, args.prices)
