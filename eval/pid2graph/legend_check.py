"""Fresh VLM legend check against printed definitions in a prepared PDF region."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from diagex.config import Config, LLMConfig, PricingConfig
from diagex.extractors.legend_rows import classify_native_legend
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import DiagramPage

from .data import save_new, sha256
from .transfer_accounting import transfer_billing

CAPTION_GUIDANCE = """\nSeparate a column heading from a caption by source layout and visible ink. A caption containing words such as 'general', 'symbol', or 'legend' can still define a generic physical glyph when the same row visibly pairs it with that glyph. Inspect that pairing before rejecting a row as a header. A generic glyph definition does not justify a specific subtype. Continue to reject headings without paired glyphs and retain uncertainty when the actual glyph is absent or clipped."""


def run(prepared, output, ledger, prices, model="deepseek/deepseek-v4.1-flash", caption_guidance=False):
    prepared, output = Path(prepared), Path(output)
    selection = json.loads((prepared / "legend-source-selection.json").read_text())
    if sha256(selection["source"]) != selection["source_sha256"]:
        raise ValueError("PDF source changed")
    evidence = PageEvidence.model_validate_json((prepared / "2401-page1-native-evidence.json").read_text())
    cfg = Config()
    cfg.symbol_perception.reasoning_timeout_s = 120
    price = json.loads(Path(prices).read_text())["models"][model]
    client = LLMClient(LLMConfig(transport="openrouter", model=model, reasoning_mode="disabled",
                      openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"), spending_ledger=str(Path(ledger).resolve()),
                      verified_prices=str(Path(prices).resolve()), spending_category="legend_transfer"))
    if caption_guidance:
        original_create = client.messages_create

        def create_with_caption_guidance(**kwargs):
            kwargs["system"] += CAPTION_GUIDANCE
            return original_create(**kwargs)

        client.messages_create = create_with_caption_guidance
    cost = CostTracker(pricing=PricingConfig(input_per_mtok=price["input_per_token"] * 1e6,
                      output_per_mtok=price["output_per_token"] * 1e6,
                      cache_read_per_mtok=price["input_per_token"] * 1e6, cache_write_per_mtok=price["input_per_token"] * 1e6))
    config = {"model": model, "source_sha256": selection["source_sha256"], "region": selection["region"],
              "selection_sha256": sha256(prepared / "legend-source-selection.json"),
              "provider_tags": price["provider_tags"], "policy": asdict(cfg.symbol_perception),
              "experimental_caption_guidance": CAPTION_GUIDANCE if caption_guidance else None,
              "legend_classifier_sha256": sha256("src/diagex/extractors/legend_rows.py")}
    save_new(output / "config.json", config)
    expected = [{"row_id": r["id"], "printed_label": r["label"],
                 "expected_valve": r["label"] not in {"盲板连接", "限流孔板"}} for r in selection["rows"]]
    save_new(output / "source-reference-checks.json", {
        "basis": "Printed definitions in the source valve/fitting legend column, visually inspected before fresh model inference",
        "rows": expected, "limits": "Broad source-consistency checks only; no claim of independent engineering qualification or general legend accuracy"})
    with Image.open(prepared / "2401-page1-native-render.png") as image:
        page = DiagramPage(page_index=evidence.page_index, image=image.convert("RGB"), width=evidence.width,
                           height=evidence.height, dpi=evidence.dpi, effective_dpi=evidence.effective_dpi,
                           is_scanned=False, source_ref=evidence.source_ref)
    before = {r["id"] for r in json.loads(Path(ledger).read_text())["requests"]}
    started = time.monotonic()
    coverage = []
    region = selection["region"]
    entries = classify_native_legend(page=page, evidence=evidence,
                region=(region["x"], region["y"], region["w"], region["h"]), client=client,
                cost_tracker=cost, cfg=cfg, coverage=coverage)
    records = [e.model_dump(mode="json") for e in entries]
    save_new(output / "legend-entries.json", records)
    lookup = {e["source_row_id"]: e for e in records}
    checks = []
    for reference in expected:
        entry = lookup.get(reference["row_id"])
        checks.append({**reference, "emitted": entry is not None,
                       "actual_kind": entry["kind"] if entry else None,
                       "label_preserved": entry is not None and entry["label"] == reference["printed_label"],
                       "broad_kind_consistent": entry is not None and ((entry["kind"] == "valve") == reference["expected_valve"]),
                       "attributes": {k: v for k, v in entry["attributes"].items() if k in
                                      {"row_status", "valve_type", "equipment_class", "symbol_role"}} if entry else {}})
    requests = [r for r in json.loads(Path(ledger).read_text())["requests"]
                if r["id"] not in before and r["category"] == "legend_transfer" and r["model"] == model]
    result = {"checks": checks, "coverage": [c.model_dump(mode="json") for c in coverage],
              "runtime_seconds": time.monotonic() - started, "usage": cost.summary(),
              "ledger_request_ids": [r["id"] for r in requests],
              **transfer_billing(ledger, [r["id"] for r in requests]),
              "limits": "Fresh classification on one previously exposed PDF column. Printed labels supply broad reference checks; detailed engineering semantics remain unqualified. GraphML was not used."}
    save_new(output / "report.json", result)
    print(json.dumps({"rows": len(checks), "emitted": len(records),
                      "source_consistent": sum(c["label_preserved"] and c["broad_kind_consistent"] for c in checks),
                      "billing": result.get("billing"),
                      "reserved_or_charged_usd": result["reserved_or_charged_usd"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "out", "ledger", "prices"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--caption-guidance", action="store_true")
    args = parser.parse_args()
    run(args.prepared, args.out, args.ledger, args.prices, caption_guidance=args.caption_guidance)
