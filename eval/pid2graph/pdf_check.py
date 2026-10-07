"""Qualitative native-PDF perception and geometry smoke on prepared source tiles."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

from diagex.config import Config, LLMConfig, PricingConfig
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.fusion import fuse_objects
from diagex.vision.models import DiagramPage, Tile
from diagex.vision.perception import DetectionRecord, perceive_tile
from diagex.vision.symbol_candidates import SymbolCandidate
from diagex.vision.tiling import AspectAwareStrategy, ownership_core, tile
from diagex.vision.topology import build_page_topology
from diagex.vision.views import ViewProvider

from .data import digest, save_new, sha256
from .transfer_accounting import transfer_billing


def run(prepared, output, ledger, prices, legend=None):
    prepared, output = Path(prepared), Path(output)
    manifest = json.loads((prepared / "prepared-pages.json").read_text())
    cfg = Config()
    cfg.symbol_perception.request_timeout_s = 120
    model = "deepseek/deepseek-v4.1-flash"
    price = json.loads(Path(prices).read_text())["models"][model]
    client = LLMClient(LLMConfig(transport="openrouter", model=model, reasoning_mode="disabled",
                      openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"), spending_ledger=str(Path(ledger).resolve()),
                      verified_prices=str(Path(prices).resolve()), spending_category="legend_transfer"))
    pricing = PricingConfig(input_per_mtok=price["input_per_token"] * 1e6,
                      output_per_mtok=price["output_per_token"] * 1e6,
                      cache_read_per_mtok=price["input_per_token"] * 1e6, cache_write_per_mtok=price["input_per_token"] * 1e6)
    config = {"model": model, "provider_tags": price["provider_tags"],
              "prepared_manifest_sha256": sha256(prepared / "prepared-pages.json"),
              "perception_sha256": sha256("src/diagex/vision/perception.py"),
              "legend_sha256": sha256(legend) if legend else None,
              "scope": "Two selected tiles per PDF; native perception, fusion and deterministic topology only. No full drawing or engineering-accuracy claim."}
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("PDF check configuration changed")
    else:
        save_new(output / "config.json", config)
    summaries = []
    for item in manifest["pages"]:
        source = prepared / item["id"]
        out = output / item["id"]
        if sha256(item["source_pdf"]) != item["source_pdf_sha256"] or sha256(source / "source.png") != item["image_sha256"]:
            raise ValueError("PDF source/render changed")
        page_info = json.loads((source / "page.json").read_text())
        with Image.open(source / "source.png") as image:
            page = DiagramPage(**page_info, image=image.convert("RGB"))
        evidence = PageEvidence.model_validate_json((source / "native-evidence.json").read_text())
        candidates = [SymbolCandidate.model_validate(c) for c in json.loads((source / "candidates.json").read_text())]
        path_ids = {p.id for p in evidence.paths}
        if any(not set(c.source_path_ids) <= path_ids for c in candidates):
            raise ValueError("Candidate lacks native source paths")
        selected = [Tile.model_validate(t) for t in json.loads((source / "selected-tiles.json").read_text())]
        grid = tile(page, AspectAwareStrategy(cfg.tiling.max_tokens_per_tile, cfg.tiling.overlap_frac, cfg.tiling.token_per_pixel))
        views = ViewProvider(page, grid)
        # A project legend is never transferred to an unrelated project.
        legend_entries = json.loads(Path(legend).read_text()) if legend and item["id"].startswith("2401-") else []
        detections = []
        for index, current in enumerate(selected):
            checkpoint = out / f"{current.id}.json"
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text())
            else:
                while client._rate_limit_not_before > time.monotonic():
                    time.sleep(min(30, client._rate_limit_not_before - time.monotonic()))
                started = time.monotonic()
                before = {r["id"] for r in json.loads(Path(ledger).read_text())["requests"]}
                cost = CostTracker(pricing=pricing)
                view, info = views.get_tile(current.id)
                result = {"config_sha256": digest(config), "tile_id": current.id, "detections": [], "status": "complete"}
                try:
                    outcome = perceive_tile(client=client, cost_tracker=cost, reporter=NullReporter(),
                        page=evidence, tile=current, view_image=view, view_info=info,
                        ownership_bbox=ownership_core(current, grid), legend_summary=legend_entries,
                        step=index + 1, candidates=candidates, policy=cfg.symbol_perception)
                    result.update(detections=[d.model_dump(mode="json") for d in outcome.detections],
                                  batch=outcome.batch.model_dump(mode="json"))
                except Exception as exc:
                    from diagex.llm.budget import BudgetExceeded
                    if isinstance(exc, BudgetExceeded):
                        raise
                    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                requests = [r for r in json.loads(Path(ledger).read_text())["requests"]
                            if r["id"] not in before and r["category"] == "legend_transfer"]
                result.update(runtime_seconds=time.monotonic() - started, usage=cost.summary(),
                              ledger_request_ids=[r["id"] for r in requests],
                              **transfer_billing(ledger, [r["id"] for r in requests]))
                save_new(checkpoint, result)
            detections.extend(DetectionRecord.model_validate(d) for d in result["detections"])
            print(item["id"], current.id, result["status"], len(result["detections"]), flush=True)
        fused = fuse_objects(source_name=item["source_pdf"], pages=[evidence], detections=detections,
                             per_page_status={page.page_index: "partial"})
        topology = build_page_topology(page=evidence, nodes=fused.graph.nodes)
        if not (out / "fusion.json").exists():
            save_new(out / "fusion.json", fused.model_dump(mode="json"))
            save_new(out / "topology.json", topology.model_dump(mode="json"))
        overlay = page.image.copy()
        draw = ImageDraw.Draw(overlay)
        for d in detections:
            draw.rectangle((d.bbox.x, d.bbox.y, d.bbox.x2, d.bbox.y2), outline="#168155", width=4)
        for current in selected:
            draw.rectangle((current.bbox.x, current.bbox.y, current.bbox.x2, current.bbox.y2), outline="#2166cc", width=4)
        overlay.thumbnail((1800, 1800))
        overlay.save(out / "perception-overlay.png")
        summaries.append({"id": item["id"], "selected_tiles": len(selected), "detections": len(detections),
                          "fused_nodes": len(fused.graph.nodes), "topology_edges": len(topology.edges),
                          "native_text_spans": len(evidence.text_spans), "native_paths": len(evidence.paths)})
    save_new(output / "summary.json", {"pages": summaries, "scope": config["scope"], "human_reviewed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "out", "ledger", "prices"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--legend")
    args = parser.parse_args()
    run(args.prepared, args.out, args.ledger, args.prices, args.legend)
