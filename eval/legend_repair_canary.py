"""Replay saved invalid first responses, then measure one live source-bound repair.

This isolates recovery of known contract failures, not fresh extraction accuracy.
Original source rows and responses are frozen before any paid call. Every live
request uses the shared cumulative spending guard and is saved for inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import fitz
from PIL import Image

from diagex.config import Config, LLMConfig
from diagex.extractors import legend_rows as classifier
from diagex.llm.billing import summarize
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import DiagramPage


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def baseline_geometry(study):
    name = "src/diagex/vision/legend_rows.py"
    path = study / "baseline-legend-geometry.py"
    with tarfile.open(study / "baseline-source.tar.gz") as archive:
        source = archive.extractfile(name).read()
    if path.exists() and path.read_bytes() != source:
        raise ValueError("Baseline module changed")
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("legend_canary_baseline_geometry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    study, run_dir, out = args.study.resolve(), args.run.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError("Use a new output directory; never automatically repeat interrupted calls")
    out.mkdir(parents=True)
    source_sha = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    checkpoint = json.loads((run_dir / "checkpoints/manifest.json").read_text())
    if source_sha != checkpoint["source_sha256"]:
        raise ValueError("PDF checksum differs from the original run")
    original = json.loads((run_dir / "legend.json").read_text())
    failed = {c["source_row_id"] for c in original["coverage"]
              if c.get("failure_kind") == "contract" and c.get("source_row_id")}
    entries = {e["source_row_id"]: e for e in original["entries"] if e.get("source_row_id") in failed}
    if len(entries) != len(failed):
        raise ValueError("Failure lacks a saved source observation")
    if not failed or len(failed) > classifier.BATCH_SIZE:
        raise ValueError("This bounded canary requires one nonempty batch of saved failures per page")
    geometry = baseline_geometry(study)
    cfg = Config()
    cfg.symbol_perception.transport_attempts = 1
    cfg.symbol_perception.reasoning_timeout_s = 90
    config = LLMConfig.from_env()
    config.transport = "openrouter"
    config.model = args.model
    config.reasoning_mode = "disabled"
    config.spending_ledger = str(study / "spending.json")
    config.verified_prices = str(study / "verified-prices.json")
    config.spending_category = "legend_contract_repair"
    config.production_open_weight = False
    client = LLMClient(config)
    state = json.loads((study / "spending.json").read_text())
    remaining = state["limit_usd"] - state["prior_spend"]["usd"] - summarize(state)["budget_exposure_usd"]
    price = client.spending.prices[args.model]
    if time.time() > price["valid_until"]:
        raise ValueError("Refresh the verified endpoint prices before running the canary")
    per_call_ceiling = (price["context_length"] * price["input_per_token"]
                        + 6000 * price["output_per_token"] + price.get("request_usd", 0))
    maximum_calls = len({e["source_page_index"] for e in entries.values()}) * args.repeats
    if remaining < 5 + maximum_calls * per_call_ceiling:
        raise ValueError("Preserve at least $5 for final validation")
    before = {r["id"] for r in state["requests"]}
    manifest = {"model": args.model, "source": str(args.pdf.resolve()), "source_sha256": source_sha,
                "historical_run": str(run_dir), "row_ids": sorted(failed), "repeats": args.repeats,
                "classifier_sha256": hashlib.sha256(Path(classifier.__file__).read_bytes()).hexdigest(),
                "baseline_geometry_sha256": hashlib.sha256((study / "baseline-legend-geometry.py").read_bytes()).hexdigest(),
                "maximum_live_calls": maximum_calls, "maximum_live_exposure_usd": maximum_calls * per_call_ceiling,
                "scope": "Known six-row contract-failure recovery; initial responses replayed. Not independent held-out accuracy.",
                "source_review": "Six source glyphs and printed labels visually checked before calls; actuator definitions show contextual valve bodies. No subtype may be inferred from those bodies."}
    save(out / "manifest.json", manifest)
    cost = CostTracker()
    original_create, original_rows = client.messages_create, classifier.native_legend_rows
    records = []
    try:
        with fitz.open(args.pdf) as doc:
            for repeat in range(args.repeats):
                for page_index in sorted({e["source_page_index"] for e in entries.values()}):
                    evidence_path = run_dir / "evidence" / f"page-{page_index + 1:04d}.json"
                    evidence = PageEvidence.model_validate_json(evidence_path.read_text())
                    rows = [r for r in geometry.native_legend_rows(evidence) if r.id in failed]
                    expected = {key for key, value in entries.items() if value["source_page_index"] == page_index}
                    if {row.id for row in rows} != expected:
                        raise ValueError("Frozen source rows do not match the saved evidence")
                    classifier.native_legend_rows = lambda *_a, _rows=rows, **_k: _rows
                    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(evidence.dpi / 72, evidence.dpi / 72), alpha=False)
                    if (pix.width, pix.height) != (evidence.width, evidence.height):
                        raise ValueError("Rendered page coordinates differ from saved evidence")
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    page = DiagramPage(page_index=page_index, image=image, width=evidence.width,
                                       height=evidence.height, dpi=evidence.dpi,
                                       effective_dpi=evidence.effective_dpi, is_scanned=False,
                                       source_ref=evidence.source_ref)
                    calls = 0
                    prefix = f"repeat-{repeat + 1}-page-{page_index + 1}"

                    def dispatch(_rows=rows, _prefix=prefix, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            raw = [d for row in _rows for d in json.loads(entries[row.id]["attributes"]["classification_evidence"])]
                            save(out / f"{_prefix}-replayed-first-response.json", raw)
                            return SimpleNamespace(content=[{"type": "tool_use", "name": "submit_legend_rows", "input": {"rows": raw}}], usage=None)
                        if calls > 2:
                            raise RuntimeError("Canary exceeded one repair per source row")
                        save(out / f"{_prefix}-request.json", kwargs)
                        started = time.monotonic()
                        try:
                            response = original_create(**kwargs)
                            save(out / f"{_prefix}-response.json", response.model_dump(mode="json"))
                            return response
                        finally:
                            save(out / f"{_prefix}-elapsed.json", {"elapsed_s": time.monotonic() - started})

                    client.messages_create = dispatch
                    coverage = []
                    result = classifier.classify_native_legend(page=page, evidence=evidence, region=None,
                                client=client, cost_tracker=cost, cfg=cfg, coverage=coverage)
                    save(out / f"{prefix}-entries.json", [e.model_dump(mode="json") for e in result])
                    save(out / f"{prefix}-coverage.json", [c.model_dump(mode="json") for c in coverage])
                    records.extend({"repeat": repeat + 1, "label": e.label, "row_id": e.source_row_id,
                                    "status": e.attributes.get("row_status"), "kind": e.kind,
                                    "symbol_class": e.symbol_class, "attributes": e.attributes}
                                   for e in result)
                    print(json.dumps({"case": prefix, "rows": len(rows), "accepted": sum(e.attributes.get("row_status") == "accept" for e in result), "complete": sum(c.status == "complete" for c in coverage)}, ensure_ascii=False), flush=True)
    finally:
        client.messages_create, classifier.native_legend_rows = original_create, original_rows
        state = json.loads((study / "spending.json").read_text())
        request_ids = [r["id"] for r in state["requests"] if r["id"] not in before]
        save(out / "summary.json", {"records": records, "billing": summarize(state, request_ids),
             "request_ids": request_ids, "completed_row_observations": len(records),
             "expected_row_observations": len(failed) * args.repeats, "baseline_contract_failures": len(failed),
             "scope": manifest["scope"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "run", "out", "pdf"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    run(parser.parse_args())
