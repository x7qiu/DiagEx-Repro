"""Frozen-baseline derivative for explicit raster review; no GraphML at inference.

Separate study runner keeps ongoing baseline/advisory code immutable. Shared
perception, source loading, tiling and geometry projection remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from diagex.config import Config, LLMConfig, PricingConfig, SymbolPerceptionConfig
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import extract_page_evidence
from diagex.vision.loader import iter_pages, load
from diagex.vision.raster_broad import BroadRasterClient as LLMClient
from diagex.vision.raster_broad import perceive_broad_raster as perceive_tile
from diagex.vision.tiling import AspectAwareStrategy, ownership_core, tile
from diagex.vision.views import ViewProvider

from .connection_accounting import configure as configure_connection_accounting
from .connection_accounting import record_unsent_final_attempt
from .data import digest, save_new, sha256
from .deadline import wall_clock_limit


def label(kind, attributes):
    if kind == "raster_symbol":
        return attributes["broad_category"]
    if kind == "instrument":
        return "instrumentation"
    if kind == "opc":
        return "inlet/outlet"
    cls = str(attributes.get("equipment_class") or "").lower().replace(" ", "_")
    if attributes.get("valve_type") or cls == "valve" or cls.endswith("_valve"):
        return "valve"
    if cls in {"pump", "compressor"}:
        return "pump"
    if cls in {"tank", "vessel"}:
        return "tank"
    return "general" if kind == "equipment" else "unknown"


def record(identity, kind, attributes, box, confidence, sx, sy, disposition):
    return {"id": identity, "label": label(kind, attributes),
            "bbox": [box["x"] * sx, box["y"] * sy, (box["x"] + box["w"]) * sx, (box["y"] + box["h"]) * sy],
            "confidence": {"high": 0.9, "medium": 0.6, "low": 0.3}.get(confidence, 0.3),
            "disposition": disposition, "attributes": attributes}


def _owned_requests(client, ledger, previous_requests, category, model):
    return [r for r in json.loads(Path(ledger).read_text())["requests"]
            if r["id"] not in previous_requests and r["id"] in client._evaluation_owned_request_ids
            and r["category"] == category and r["model"] == model]


def prepare(manifest_path, output, *, per_collection=8):
    """Freeze source-only panels before inference; deterministic ordering ignores labels."""
    manifest = json.loads(Path(manifest_path).read_text())
    check = dict(manifest)
    seal = check.pop("manifest_sha256")
    if digest(check) != seal:
        raise ValueError("Manifest content changed")
    panels = {}
    for split in ("train", "validation", "test", "development_exposed"):
        rows = [r for r in manifest["drawings"] if r["split"] == split]
        selected = []
        for collection in sorted({r["collection"] for r in rows}):
            candidates = sorted((r for r in rows if r["collection"] == collection),
                                key=lambda r: digest(["panel-v1", r["id"]]))
            count = 2 if split in {"train", "development_exposed"} else per_collection
            selected.extend(candidates[:count])
        panels[split] = [{"id": r["id"], "image": str(Path(manifest["dataset_root"]) / r["image"]),
                          "image_sha256": r["image_sha256"], "size": r["size"],
                          "collection": r["collection"]} for r in selected]
    value = {"manifest_sha256": seal, "panels": panels,
             "sampling": "ID-hash order, up to eight drawings per synthetic collection; no annotation-dependent selection"}
    value["panel_sha256"] = digest(value)
    save_new(output, value)
    return value


def run(panel_path, output, ledger, prices, *, split="validation", model="deepseek/deepseek-v4.1-flash",
        variant="broad_raster_recognition", selection=None, max_drawings=None, request_timeout=120.0, proposal_run=None,
        max_new_tiles=None):
    if variant != "broad_raster_recognition":
        raise ValueError("Unknown perception variant")
    panel = json.loads(Path(panel_path).read_text())
    check = dict(panel)
    if digest({k: v for k, v in check.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Source panel content changed")
    if split == "test":
        from .selection import read_selection
        chosen = read_selection(selection, panel)
        if chosen["inference_variant"] != variant or chosen["model"] != model:
            raise ValueError("Selection does not match final inference")
    proposals, guidance_config = {}, None
    if variant == "broad_raster_recognition":
        if not proposal_run:
            raise ValueError("Supervised guidance requires source-only detector results")
        from .guidance import load_proposals
        proposals, guidance_config = load_proposals(proposal_run, panel, split)
    config = Config()
    policy = SymbolPerceptionConfig()
    # Keep a common, explicit provider-latency allowance across compared methods.
    # The application's 45-second default is retained; its timeout smoke is archived.
    policy.request_timeout_s = request_timeout
    price_snapshot = Path(prices).read_bytes()
    price_snapshot_hash = hashlib.sha256(price_snapshot).hexdigest()
    price = json.loads(price_snapshot)["models"][model]
    if split == "test" and "final_price_limits" in chosen:
        from .selection import price_signature
        if price_signature(price) != chosen["final_price_limits"]:
            raise ValueError("Final provider price limits differ from the sealed selection")
    pricing = PricingConfig(input_per_mtok=price["input_per_token"] * 1e6,
                            output_per_mtok=price["output_per_token"] * 1e6,
                            cache_read_per_mtok=price["input_per_token"] * 1e6,
                            cache_write_per_mtok=price["input_per_token"] * 1e6)
    category = "final" if split == "test" else "baseline" if variant == "baseline" else "experiments"
    cfg = LLMConfig(transport="openrouter", model=model, openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
                    reasoning_mode="disabled", spending_ledger=str(Path(ledger).resolve()),
                    verified_prices=str(Path(prices).resolve()), spending_category=category)
    if not cfg.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is unavailable")
    client = LLMClient(cfg)
    configure_connection_accounting(client)
    output = Path(output)
    config_record = {"panel_sha256": panel["panel_sha256"], "split": split, "model": model, "variant": variant,
                     "provider_tags": price["provider_tags"],
                     "operational_adjustment": "Benchmark uses an explicit request timeout; application defaults are unchanged",
                     "runner_source_sha256": sha256(__file__),
                     "tiling": asdict(config.tiling), "scan": asdict(config.scan), "perception": asdict(policy),
                     "source_fingerprints": {name: sha256(Path(__file__).resolve().parents[2] / "src/diagex" / name)
                                             for name in ("vision/perception.py", "vision/loader.py", "vision/tiling.py", "llm/client.py", "vision/raster_broad.py")}}
    config_hash = digest(config_record)
    if guidance_config is not None:
        config_record["guidance"] = guidance_config
        config_hash = digest(config_record)
    if split == "test":
        from .selection import perception_signature
        if perception_signature(config_record) != chosen["inference_signature"]:
            raise ValueError("Final perception settings/source differ from validation selection")
    cfg_path = output / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != config_record:
            raise ValueError("Cannot resume with changed configuration or source code")
    else:
        save_new(cfg_path, config_record)
    # Prices are operational routing/accounting inputs, retained separately from
    # the frozen visual method. Record the exact snapshot for every new tile.
    price_archive = output / "price-snapshots" / f"{price_snapshot_hash}.json"
    price_archive.parent.mkdir(parents=True, exist_ok=True)
    if price_archive.exists():
        if price_archive.read_bytes() != price_snapshot:
            raise ValueError("Archived price snapshot changed")
    else:
        with price_archive.open("xb") as stream:
            stream.write(price_snapshot)
    transport_path = output / "transport-resume.json"
    if transport_path.exists():
        state = json.loads(transport_path.read_text())
        client._rate_limit_not_before = time.monotonic() + max(0, state["not_before_unix"] - time.time())
        client._rate_limit_cooldown_s = state["cooldown_seconds"]
        client._rate_limit_successes = state["successes"]
    new_tiles = 0
    rows = panel["panels"][split]
    if max_drawings is not None:
        rows = rows[:max_drawings]
    for row in rows:
        case_dir = output / digest(row["id"])[:16]
        case_dir.mkdir(parents=True, exist_ok=True)
        summary_path = case_dir / "predictions.json"
        if summary_path.exists():
            continue
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError(f"Source checksum changed: {row['id']}")
        preparation_started = time.monotonic()
        page = next(iter_pages(load(row["image"], config.tiling, config.scan)))
        evidence = extract_page_evidence(page=page, source_path=Path(row["image"]))
        grid = tile(page, AspectAwareStrategy(config.tiling.max_tokens_per_tile, config.tiling.overlap_frac, config.tiling.token_per_pixel))
        views = ViewProvider(page, grid)
        sx, sy = row["size"][0] / page.width, row["size"][1] / page.height
        preparation_seconds = time.monotonic() - preparation_started
        predictions, errors = [], []
        consecutive_errors = 0
        invocation_errors = 0
        for index, current in enumerate(grid):
            checkpoint = case_dir / f"{current.id}.json"
            cached = checkpoint.exists()
            if cached:
                result = json.loads(checkpoint.read_text())
                if result["config_sha256"] != config_hash:
                    raise ValueError("Checkpoint configuration mismatch")
            else:
                if max_new_tiles is not None and new_tiles >= max_new_tiles:
                    return
                if (output / "stop-request.json").exists():
                    raise RuntimeError("Run stopped between tiles; see stop-request.json")
                new_tiles += 1
                started = time.monotonic()
                # A provider cooldown belongs between attempts, not inside the
                # next tile's inference deadline. Do not mark untouched tiles
                # failed merely because a preceding request was rate-limited.
                while client._rate_limit_not_before > time.monotonic():
                    time.sleep(min(30, client._rate_limit_not_before - time.monotonic()))
                cost = CostTracker(pricing=pricing)
                previous_requests = {r["id"] for r in json.loads(Path(ledger).read_text())["requests"]}
                result = {"config_sha256": config_hash, "tile_id": current.id,
                          "predictions": [], "status": "complete",
                          "price_snapshot_sha256": price_snapshot_hash}
                budget_stop = None
                client.raster_review_attempts = []
                client.raster_proposal_ids = []
                client.raster_guides_omitted = 0
                try:
                    view, info = views.get_tile(current.id)
                    page_context = None
                    if guidance_config is not None:
                        from .guidance import view_guidance
                        page_context = view_guidance(proposals[row["id"]], info.page_bbox, (sx, sy))
                        client.begin_raster_view(page_context)
                    # Allow both documented format passes plus cleanup time.
                    # Bound silent SSE keepalives that never yield an SDK event.
                    with wall_clock_limit(2 * request_timeout + 30):
                        outcome = perceive_tile(client=client, cost_tracker=cost, reporter=NullReporter(),
                            page=evidence, tile=current, view_image=view, view_info=info,
                            ownership_bbox=ownership_core(current, grid), legend_summary=[], step=index + 1,
                            candidates=[], policy=policy, page_context=page_context)
                    result["outcome"] = outcome.batch.model_dump(mode="json")
                    for d in outcome.detections:
                        result["predictions"].append(record(d.id, d.kind, d.attributes, d.bbox.model_dump(),
                                                            d.confidence, sx, sy, "graph_eligible"))
                    for j, review in enumerate(outcome.batch.candidate_reviews):
                        obj = review.get("object")
                        if obj and not review.get("candidate_id") and review.get("bbox"):
                            attributes = {**obj.get("attributes", {}), **{k: obj.get(k) for k in
                                          ("equipment_class", "valve_type", "actuation")}}
                            result["predictions"].append(record(f"{current.id}:proposal:{j}", obj["kind"], attributes,
                                review["bbox"], obj.get("confidence", "medium"), sx, sy, "review_proposal"))
                except Exception as exc:
                    from diagex.llm.budget import BudgetExceeded
                    if isinstance(exc, BudgetExceeded):
                        if not _owned_requests(client, ledger, previous_requests, category, model):
                            raise  # No attempt was reserved: this tile is still untouched.
                        # A prior format/retry attempt already consumed a reservation.
                        # Preserve that attempted failure before stopping, so resume
                        # cannot silently repeat a paid tile or lose its cost record.
                        budget_stop = exc
                    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    unsent = record_unsent_final_attempt(client, exc, previous_requests)
                    if unsent is not None:
                        result["non_dispatch_evidence"] = unsent
                    if any(term in result["error"].lower() for term in ("connection", "ssl", "peer closed", "incomplete chunked", "wall-clock deadline")):
                        # A broken pooled TLS connection must not poison later
                        # independent tiles. Keep the failure and reservation,
                        # plus the outer client's provider cooldown state.
                        client._client.close()
                        client._client = LLMClient._build_client(cfg)
                        configure_connection_accounting(client)
                        # An outage affects new connections too. Give the
                        # existing local proxy/upstream route time to recover
                        # before dispatching the next independent paid tile.
                        client._rate_limit_not_before = max(
                            client._rate_limit_not_before, time.monotonic() + 30)
                        result["transport_connection_reset_after_error"] = True
                        result["transport_backoff_seconds"] = 30
                result["raster_review_attempts"] = client.raster_review_attempts
                result["requested_raster_proposal_ids"] = client.raster_proposal_ids
                result["raster_guides_omitted_by_limit"] = client.raster_guides_omitted
                result.update(runtime_seconds=time.monotonic() - started, usage=cost.summary())
                requests = _owned_requests(client, ledger, previous_requests, category, model)
                result["ledger_request_ids"] = [r["id"] for r in requests]
                result["reserved_or_charged_usd"] = sum(r["charged_usd"] for r in requests)
                atomic_write_json(checkpoint, result)
                atomic_write_json(transport_path, {
                    "not_before_unix": time.time() + max(0, client._rate_limit_not_before - time.monotonic()),
                    "cooldown_seconds": client._rate_limit_cooldown_s,
                    "successes": client._rate_limit_successes,
                    "note": "Resume preserves provider cooldown; completed and failed tile records are immutable."})
                if budget_stop is not None:
                    raise budget_stop
            predictions.extend(result["predictions"])
            if result["status"] != "complete":
                errors.append({"tile_id": current.id, "error": result.get("error")})
                if not cached:
                    consecutive_errors += 1
                    invocation_errors += 1
            elif not cached:
                consecutive_errors = 0
            if not cached:
                print(f"{row['id']} {index + 1}/{len(grid)}: {result['status']}; {len(result['predictions'])} proposals", flush=True)
            # Resume continues untouched tiles after a transport outage. Cached
            # failures remain failures in the result and scoring denominator;
            # only fresh attempts control this invocation's circuit breaker.
            if consecutive_errors >= policy.consecutive_failure_limit or invocation_errors >= policy.total_failure_limit:
                raise RuntimeError("Perception failure limit reached; inspect checkpoints before continuing")
        tile_results = [json.loads((case_dir / f"{t.id}.json").read_text()) for t in grid]
        detector_seconds = guidance_config["detector_runtime_seconds"][row["id"]] if guidance_config else 0
        save_new(summary_path, {"id": row["id"], "image_sha256": row["image_sha256"],
                               "config_sha256": config_hash, "predictions": predictions, "errors": errors,
                               "tiles": len(grid), "completed_tiles": len(grid) - len(errors),
                               "ledger_request_ids": sorted({i for r in tile_results for i in r["ledger_request_ids"]}),
                               "runtime_seconds": detector_seconds + preparation_seconds + sum(r["runtime_seconds"] for r in tile_results),
                               "detector_seconds": detector_seconds,
                               "preparation_seconds": preparation_seconds,
                               "runtime_scope": "page preparation plus recorded tile processing/cooldown durations; excludes original source checksum read and interrupted invocation overhead",
                               "reserved_or_charged_usd": sum(r["reserved_or_charged_usd"] for r in tile_results),
                               "usage_upper_usd": sum(r["usage"]["total_usd"] for r in tile_results)})


def overlay(image_path, predictions_path, output):
    from PIL import ImageDraw
    rows = json.loads(Path(predictions_path).read_text())["predictions"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in rows:
        color = "#008655" if row["disposition"] == "graph_eligible" else "#e77700"
        draw.rectangle(row["bbox"], outline=color, width=3)
        draw.text(row["bbox"][:2], row["label"], fill=color, stroke_width=1, stroke_fill="white")
    image.save(output)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "out", "ledger", "prices", "proposal-run"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    p.add_argument("--variant", choices=["broad_raster_recognition"], default="broad_raster_recognition")
    p.add_argument("--selection")
    p.add_argument("--request-timeout", type=float, default=120)
    p.add_argument("--max-new-tiles", type=int)
    a = p.parse_args()
    run(a.panel, a.out, a.ledger, a.prices, split=a.split, model=a.model, variant=a.variant,
        selection=a.selection, request_timeout=a.request_timeout, proposal_run=a.proposal_run,
        max_new_tiles=a.max_new_tiles)


if __name__ == "__main__":
    main()
