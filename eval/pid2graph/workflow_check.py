"""Exercise agent review and the real graph runner on a controlled PDF fixture.

This is a software integration check, not human review or extraction accuracy.
The fixture supplies known symbols so graph construction can be checked without
confounding the separate held-out symbol experiment. All model calls are metered.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fitz

from diagex.config import Config, LLMConfig, PricingConfig
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.extractors.pid_evidence import run_pid_evidence_extract
from diagex.review.detection import DetectionReviewStore, write_detection_bundle
from diagex.vision.evidence import extract_page_evidence
from diagex.vision.legend_models import LegendEntry, LegendPack
from diagex.vision.loader import iter_pages, load
from diagex.vision.models import BBox
from diagex.vision.perception import DetectionRecord

from .data import save_new, sha256
from .transfer_accounting import transfer_billing


def run(output, ledger, prices, model):
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("Use a new workflow-check directory; paid runs are never silently repeated")
    root.mkdir(parents=True)
    price = json.loads(Path(prices).read_text())["models"][model]
    cfg = Config(runs_dir=root / "runs", llm=LLMConfig(
        transport="openrouter", model=model, reasoning_mode="disabled",
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        spending_ledger=str(Path(ledger).resolve()), verified_prices=str(Path(prices).resolve()),
        spending_category="legend_transfer"))
    cfg.pid.engine = "evidence-v2"
    cfg.pricing = PricingConfig(input_per_mtok=price["input_per_token"] * 1e6,
        output_per_mtok=price["output_per_token"] * 1e6,
        cache_read_per_mtok=price["input_per_token"] * 1e6,
        cache_write_per_mtok=price["input_per_token"] * 1e6)
    source = root / "software-fixture.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=400, height=250)
        page.insert_text((20, 30), "LEGEND: rectangular vessel")
        page.draw_rect(fitz.Rect(30, 50, 70, 100))
        page = doc.new_page(width=400, height=250)
        page.insert_text((20, 25), "P&ID - SOFTWARE INTEGRATION FIXTURE")
        page.draw_rect(fitz.Rect(50, 80, 100, 180))
        page.draw_rect(fitz.Rect(280, 80, 330, 180))
        page.draw_line((100, 130), (280, 130))
        page.insert_text((55, 70), "V-001")
        page.insert_text((285, 70), "V-002")
        doc.save(source)
    origin = root / "review-input"
    pages = []
    for page in iter_pages(load(source, cfg.tiling, cfg.scan)):
        evidence = extract_page_evidence(page=page, source_path=source)
        evidence.role = "legend" if page.page_index == 0 else "pid"
        evidence.role_reason = "Controlled software fixture; page role fixed by its author"
        pages.append(evidence)
        atomic_write_json(origin / "evidence" / f"page-{page.page_index + 1:04d}.json",
                          evidence.model_dump(mode="json"))
    sx, sy = pages[1].width / 400, pages[1].height / 250
    detections = [DetectionRecord(id=f"fixture-{i}", page_index=1, tile_id="fixture",
        kind="equipment", label=f"V-00{i}", confidence="high",
        bbox=BBox(x=round(x * sx), y=round(80 * sy), w=round(50 * sx), h=round(100 * sy)),
        attributes={"equipment_class": "vessel"}) for i, x in ((1, 50), (2, 280))]
    pack = LegendPack(entries=[LegendEntry(label="rectangular vessel", symbol_class="vessel",
        kind="equipment", source="legend_extracted", source_page_index=0,
        source_bbox=BBox(x=round(30 * sx), y=round(50 * sy), w=round(40 * sx), h=round(50 * sy)))])
    write_detection_bundle(origin, source_hash=sha256(source), pages=pages, detections=detections,
        legend_pack=pack, per_page_status={0: "ok", 1: "ok"}, reviews=[], candidates=[])
    store = DetectionReviewStore(origin)

    def action(name, **kwargs):
        return store.apply({"action": name, "revision": store.read()["revision"],
            "rater": "Codex software fixture check", "actor_type": "agent",
            "evidence_refs": [str(source)], **kwargs})

    try:
        action("confirm_detected", ids=["fixture-1"])
    except ValueError as exc:
        assert "legend" in str(exc).lower()
    else:
        raise AssertionError("Symbols were accepted before legend review")
    action("confirm_legends", ids=["legend-0"])
    edited = detections[0].model_dump(mode="json")
    edited["label"] = "V-REVIEWED"
    action("save_symbol", id="fixture-1", detection=edited, status="confirmed")
    action("confirm_detected", ids=["fixture-2"])
    action("coverage", page_index=1, checked=True)
    snapshot = store.snapshot(store.read()["revision"])
    assert snapshot["review_origin"] == "agent"
    save_new(root / "snapshot.json", snapshot)
    before = {r["id"] for r in json.loads(Path(ledger).read_text())["requests"]}
    built = run_pid_evidence_extract(diagram=source, symbol_standard="isa-5.1", legend_path=None,
        legend_pages=None, legend_region=None, no_legend=False, legend_key=None, effort="low",
        config=cfg, persist=True, fresh=True, out_path=None, confidence_report_path=None,
        console=None, reviewed_inputs=snapshot)
    graph = built.graph.model_dump(mode="json")
    assert len(graph["nodes"]) == 2, "Reviewed instance count changed"
    assert not (origin / "graph.json").exists(), "Review inputs were overwritten"
    persisted = json.loads((built.run_dir / "reviewed.inputs.json").read_text())
    assert persisted == snapshot
    by_label = {n["label"]: n for n in graph["nodes"]}
    assert by_label["V-REVIEWED"]["bbox_global"] == edited["bbox"]
    assert by_label["V-REVIEWED"]["attributes"]["equipment_class"] == "vessel"
    requests = [r for r in json.loads(Path(ledger).read_text())["requests"]
                if r["id"] not in before and r["category"] == "legend_transfer" and r["model"] == model]
    report = {"scope": __doc__, "human_reviewed": False, "model": model,
        "source_sha256": sha256(source), "snapshot_sha256": sha256(root / "snapshot.json"),
        "run_dir": str(built.run_dir), "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
        "quality_status": built.quality_status, "legend_source": built.legend_source,
        "native_paths": sum(len(p.paths) for p in pages),
        "checks": ["legend prerequisite enforced", "agent provenance preserved", "review edit persisted",
                   "real graph runner completed", "reviewed instances and boxes preserved", "source run immutable"],
        "ledger_request_ids": [r["id"] for r in requests],
        **transfer_billing(ledger, [r["id"] for r in requests])}
    save_new(root / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "ledger", "prices"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model", default="qwen/qwen3.8-flash")
    args = parser.parse_args()
    run(args.out, args.ledger, args.prices, args.model)
