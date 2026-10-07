"""Bounded source-only public legend extraction; scoring never enters inference.

Example (run is billable):
  python -m eval.public_legend_canary check --input PATH/inference-input.json
  python -m eval.public_legend_canary run --input PATH/inference-input.json --out NEW_DIR
  python -m eval.public_legend_canary score --run NEW_DIR --annotations PATH/annotations.json
"""
from __future__ import annotations

import argparse
import base64
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from diagex.config import Config, LLMConfig
from diagex.extractors.pid_legend import resolve_evidence_legend
from diagex.llm.billing import summarize
from diagex.llm.budget import BudgetExceeded
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.loader import load as load_source
from eval.public_symbol_study import DEFAULT_STUDY, ROOT, BoundedLedger, box_iou, load, save, sha

SOURCE_DIR = "public-legend-development"
PUBLIC_URL = "https://blog.projectmaterials.com/epc-projects/engineering/pid-symbols-list/"
SCOPE = (
    "Six public illustrations in a constructed development legend, already present in the catalog. "
    "Knowledge disabled. Tests extraction contracts and source-caption association; "
    "not independent held-out accuracy or topology."
)


def checked_input(path, study=DEFAULT_STUDY):
    """Validate the frozen public artifacts, but never read scoring annotations."""
    root = (Path(study) / SOURCE_DIR).resolve()
    path = Path(path).resolve()
    if path.parent != root:
        raise ValueError("Input must be inside the prepared public legend directory")
    raw, freeze = load(path), load(root / "freeze.json")
    if not freeze.get("constructed") or not freeze.get("freeze_before_model_predictions"):
        raise ValueError("Public constructed source must be frozen before predictions")
    fields = {
        "source_pdf": ("source_sha256", ".pdf"),
        "image": ("image_sha256", ".png"),
        "page_evidence": ("page_evidence_sha256", ".json"),
    }
    for key, (checksum, suffix) in fields.items():
        artifact = Path(raw[key]).resolve()
        if artifact.parent != root or artifact.suffix != suffix:
            raise ValueError(f"Source artifact escapes the public source directory: {key}")
        if sha(artifact) != raw[checksum] or raw[checksum] != freeze["artifacts"].get(artifact.name):
            raise ValueError(f"Frozen public source checksum changed: {key}")
    if sha(path) != freeze["artifacts"].get(path.name):
        raise ValueError("Frozen inference input changed")
    provenance_file = root / "provenance.json"
    if sha(provenance_file) != freeze["artifacts"].get("provenance.json"):
        raise ValueError("Frozen public provenance changed")
    provenance = load(provenance_file)
    if provenance.get("url") != PUBLIC_URL or len(provenance.get("assets", [])) != 6:
        raise ValueError("Public six-asset provenance is missing")
    for asset in provenance["assets"]:
        file = Path(asset["asset_file"]).resolve()
        if not file.is_relative_to(root / "assets") or sha(file) != asset["asset_sha256"]:
            raise ValueError("Public source asset path or checksum changed")
        if (not asset["reference_id"].startswith("projectmaterials.pid-symbols.")
                or urlparse(asset["asset_url"]).hostname != "blog.projectmaterials.com"
                or asset["source_url"].split("#")[0] != PUBLIC_URL):
            raise ValueError("Nonpublic source entered the legend experiment")
    if (raw.get("constructed") is not True or raw.get("knowledge") != "off"
            or raw.get("symbol_standard") != "none" or raw.get("legend_pages") != [0]
            or raw.get("page_index") != 0):
        raise ValueError("Canary requires one constructed legend page with knowledge off")
    with Image.open(raw["image"]) as image:
        if list(image.size) != raw["image_pixel_size"]:
            raise ValueError("Source render dimensions changed")
    evidence = PageEvidence.model_validate_json(Path(raw["page_evidence"]).read_text())
    if ([evidence.width, evidence.height] != raw["image_pixel_size"]
            or evidence.page_index != 0 or evidence.dpi != raw["render_dpi"]):
        raise ValueError("Native evidence and frozen render have different coordinates")
    # No arbitrary prose, source reference records, expected labels/classes or
    # annotation fields cross this boundary into the extraction call.
    return {key: raw[key] for key in (
        "panel_id", "source_pdf", "source_sha256", "page_index", "legend_pages",
        "render_dpi", "image", "image_sha256", "image_pixel_size",
        "page_evidence", "page_evidence_sha256",
    )}


def invoke(source, client, out, tile_image_budget=None):
    cfg = Config(llm=client.config)
    cfg.knowledge = {}
    cfg.tiling.target_dpi = source["render_dpi"]
    if tile_image_budget is not None:
        if tile_image_budget <= 0:
            raise ValueError("Tile image budget must be positive")
        cfg.tiling.max_tokens_per_tile = tile_image_budget
    cfg.symbol_perception.transport_attempts = cfg.budgets.retry_attempts = 1
    cfg.symbol_perception.request_timeout_s = cfg.symbol_perception.reasoning_timeout_s = 120
    evidence = PageEvidence.model_validate_json(Path(source["page_evidence"]).read_text())
    diagram = load_source(source["source_pdf"], tiling=cfg.tiling, scan_cfg=cfg.scan)
    cost = CostTracker()
    resolved = resolve_evidence_legend(
        source=diagram, pages=[evidence], symbol_standard="none", cfg=cfg,
        client=client, cost_tracker=cost, legend_pages=source["legend_pages"],
        runs_dir_for_stem=Path(out), reporter=NullReporter(), fresh=True,
    )
    pack = resolved.resolution.pack
    return {
        "entries": [entry.model_dump(mode="json") for entry in pack.entries],
        "coverage": [item.model_dump(mode="json") for item in pack.coverage],
        "coverage_complete": bool(pack.coverage) and all(item.status == "complete" for item in pack.coverage),
        "notes": pack.notes,
        "source": resolved.resolution.source,
        "used_builtin_only": resolved.used_builtin_only,
        "effective_page_indices": resolved.effective_page_indices,
        "token_usage": [asdict(step) for step in cost.steps],
        "estimated_cost_not_billing": cost.total_usd(),
    }


def recorded_client(client, out, max_calls):
    """Stop before dispatch; ledger also caps underlying billable reservations."""
    original, calls = client.messages_create, []

    def dispatch(**kwargs):
        if len(calls) >= max_calls:
            raise BudgetExceeded("Public legend maximum call count reached before dispatch")
        kwargs["max_attempts"] = 1
        kwargs["time_budget_s"] = min(kwargs.get("time_budget_s") or 120, 120)
        index = len(calls) + 1
        save(out / f"request-{index:02d}.json", {k: v for k, v in kwargs.items() if not callable(v)})
        record = {"call": index, "started_at": time.time()}
        calls.append(record)
        try:
            response = original(**kwargs)
            raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else vars(response)
            save(out / f"response-{index:02d}.json", raw)
            record["status"] = "completed"
            return response
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            record["elapsed_s"] = time.time() - record["started_at"]
            save(out / "calls.json", calls)

    client.messages_create = dispatch
    return calls


def run(args):
    source = checked_input(args.input, args.study)
    if args.out.exists():
        raise ValueError("Use a new output directory; an existing run may contain billable attempts")
    if not 0 < args.max_usd <= 2 or not 1 <= args.max_calls <= 12:
        raise ValueError("Canary is limited to at most $2 and 12 calls")
    tile_image_budget = getattr(args, "tile_image_budget", None)
    if tile_image_budget is not None and tile_image_budget <= 0:
        raise ValueError("Tile image budget must be positive")
    cfg = LLMConfig.from_env()
    cfg.transport = "openrouter"
    cfg.model = cfg.vision_model = cfg.reasoning_model = args.model
    cfg.reasoning_mode, cfg.production_open_weight = "disabled", False
    cfg.spending_ledger = str(args.study / "spending.json")
    cfg.verified_prices = str(args.study / "verified-prices.json")
    cfg.spending_category = "public_legend_" + uuid.uuid4().hex
    if load(cfg.spending_ledger).get("schema_version") != 2:
        raise ValueError("Use the cumulative schema-version-2 spending ledger")
    client = LLMClient(cfg)
    client.spending = BoundedLedger(
        cfg.spending_ledger, cfg.verified_prices, cfg.spending_category,
        run_cap=args.max_usd, calls_cap=args.max_calls, final_reserve=5,
    )
    save(args.out / "manifest.json", {
        "input": source, "input_manifest_sha256": sha(args.input),
        "provenance_sha256": sha(args.input.parent / "provenance.json"),
        "scope": SCOPE, "knowledge": "off", "symbol_standard": "none",
        "annotations_read": False, "model": args.model, "max_calls": args.max_calls,
        "max_usd": args.max_usd, "final_reserve_usd": 5,
        "spending_category": cfg.spending_category, "harness_sha256": sha(__file__),
        "verified_prices_sha256": sha(cfg.verified_prices),
        "tile_image_budget": tile_image_budget if tile_image_budget is not None else Config().tiling.max_tokens_per_tile,
        "tile_image_budget_overridden": tile_image_budget is not None,
        "production_code": {str(p.relative_to(ROOT)): sha(p) for directory in (
            "src/diagex/extractors", "src/diagex/agent", "src/diagex/vision", "src/diagex/llm/prompts"
        ) for p in (ROOT / directory).glob("*.py")},
        "deployment_difference": f"Maximum {args.max_calls} calls total, ${args.max_usd} exposure, one transport attempt and 120-second request limit. Any explicit tile image budget override is recorded separately; default tiling is preserved when omitted.",
    })
    calls = recorded_client(client, args.out, args.max_calls)
    started, result = time.monotonic(), {"status": "failed", "entries": [], "coverage": []}
    try:
        result = {**invoke(source, client, args.out, tile_image_budget), "status": "completed"}
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        state = load(cfg.spending_ledger)
        ids = [r["id"] for r in state["requests"] if r["category"] == cfg.spending_category]
        result.update(elapsed_s=time.monotonic()-started, request_ids=ids, billing=summarize(state, ids))
        save(args.out / "result.json", result)
        save(args.out / "calls.json", calls)
        for i, entry in enumerate(result.get("entries", [])):
            if entry.get("image_b64"):
                (args.out / f"entry-{i:02d}.png").write_bytes(base64.b64decode(entry["image_b64"]))
    print({"status": result["status"], "entries": len(result["entries"]), "billing": result["billing"]})
    return result


def area(box):
    return box["w"] * box["h"]


def intersection(a, b):
    return max(0, min(a["x"]+a["w"], b["x"]+b["w"])-max(a["x"], b["x"])) * max(
        0, min(a["y"]+a["h"], b["y"]+b["h"])-max(a["y"], b["y"]))


def normalized(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def metrics(entries, annotations):
    """Caption matching and geometry diagnostics, only after inference finishes."""
    rows = annotations["rows"]
    details, consumed = [], set()
    for row in rows:
        indices = [i for i, e in enumerate(entries) if normalized(e["label"]) == normalized(row["caption"])]
        candidates = [i for i in indices if entries[i].get("source_bbox")]
        index = max(candidates, key=lambda i: box_iou(entries[i]["source_bbox"], row["glyph_bbox"]), default=None)
        detail = {"row_id": row["id"], "caption": row["caption"], "caption_found": bool(indices),
                  "candidate_indices": indices, "selected_index": index}
        consumed.update(indices)
        if index is not None:
            entry, glyph = entries[index], row["glyph_bbox"]
            box = entry["source_bbox"]
            attrs = entry.get("attributes", {})
            kind = entry["kind"]
            if kind == "equipment" and (attrs.get("equipment_class") == "valve" or entry["symbol_class"].endswith("valve")):
                kind = "valve"
            overlap = intersection(box, glyph)
            other = [r["id"] for r in rows if r["id"] != row["id"] and intersection(box, r["glyph_bbox"]) > 0]
            caption_ink = sum(intersection(box, r["caption_bbox"]) for r in rows)
            detail.update(
                kind=kind, symbol_class=entry["symbol_class"], symbol_role=attrs.get("symbol_role"),
                expected_broad_class=row.get("expected_broad_class"),
                broad_class_exact=(entry["symbol_class"] == row.get("expected_broad_class"))
                if row.get("expected_broad_class") else None,
                kind_correct=kind == row["expected_kind"],
                role_correct=(row.get("symbol_role") is None or attrs.get("symbol_role") == row["symbol_role"] or entry["symbol_class"] == row["symbol_role"]),
                crop_iou=box_iou(box, glyph), glyph_coverage=overlap / max(1, area(glyph)),
                crop_to_glyph_area=area(box) / max(1, area(glyph)),
                overlapping_other_rows=other, caption_overlap_area=caption_ink,
                caption_glyph_pairing_supported=overlap/max(1, area(glyph)) >= .8 and not other and not caption_ink,
            )
        details.append(detail)
    return {"expected_rows": len(rows), "entries": len(entries),
            "captions_found": sum(d["caption_found"] for d in details),
            "caption_glyph_pairs_supported": sum(d.get("caption_glyph_pairing_supported", False) for d in details),
            "unknown_caption_indices": [i for i in range(len(entries)) if i not in consumed],
            "duplicate_caption_count": sum(max(0, len(d["candidate_indices"])-1) for d in details),
            "rows": details,
            "scope": "Caption pairing uses >=80% source-glyph coverage without another row's glyph or caption overlap. IoU/area reported separately because thin line exemplars need crop padding. Not extraction-wide accuracy."}


def score(args):
    manifest, result, annotations = load(args.run / "manifest.json"), load(args.run / "result.json"), load(args.annotations)
    source = manifest["input"]
    if (annotations["source_pdf_sha256"] != source["source_sha256"]
            or annotations["panel_id"] != source["panel_id"]
            or annotations["image_pixel_size"] != source["image_pixel_size"]):
        raise ValueError("Scoring annotations do not belong to this frozen source")
    report = {**metrics(result.get("entries", []), annotations), "run_status": result["status"],
              "annotations_sha256": sha(args.annotations), "result_sha256": sha(args.run / "result.json"),
              "annotator": annotations["annotator"], "billing": result["billing"],
              "coverage": result.get("coverage", [])}
    save(args.run / "score.json", report)
    print({k: report[k] for k in ("expected_rows", "entries", "captions_found", "caption_glyph_pairs_supported")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--study", type=Path, default=DEFAULT_STUDY)
        if name == "run":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
            p.add_argument("--max-calls", type=int, default=8)
            p.add_argument("--max-usd", type=float, default=2)
            p.add_argument("--tile-image-budget", type=int, default=None,
                           help="Explicit test-only image-token budget per tile; omitted preserves production defaults")
    p = sub.add_parser("score")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        print(checked_input(args.input, args.study))
    elif args.command == "run":
        run(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
