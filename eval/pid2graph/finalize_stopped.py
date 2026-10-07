"""Retain attempted tiles of a deliberately stopped arm in its scored inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagex.config import ScanConfig, TilingConfig
from diagex.vision.loader import iter_pages, load
from diagex.vision.tiling import AspectAwareStrategy, tile

from .data import digest, save_new, sha256


def finalize(panel_path, run_dir):
    root = Path(run_dir)
    stop = root / "stop-request.json"
    if not stop.exists():
        raise ValueError("Only an explicitly stopped run can be finalized")
    panel = json.loads(Path(panel_path).read_text())
    config = json.loads((root / "config.json").read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != config["panel_sha256"]:
        raise ValueError("Panel changed")
    tiling, scan = TilingConfig(**config["tiling"]), ScanConfig(**config["scan"])
    for row in panel["panels"][config["split"]]:
        case = root / digest(row["id"])[:16]
        paths = sorted(case.glob("*-r*-c*.json"))
        if not paths or (case / "predictions.json").exists():
            continue  # Entirely unattempted drawings must remain missing.
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Source changed")
        page = next(iter_pages(load(row["image"], tiling, scan)))
        grid = tile(page, AspectAwareStrategy(tiling.max_tokens_per_tile, tiling.overlap_frac, tiling.token_per_pixel))
        results = [json.loads(p.read_text()) for p in paths]
        if any(r["config_sha256"] != digest(config) for r in results):
            raise ValueError("Checkpoint configuration changed")
        present = {r["tile_id"] for r in results}
        if not present <= {t.id for t in grid}:
            raise ValueError("Unexpected tile checkpoints")
        unattempted = [t.id for t in grid if t.id not in present]
        errors = [{"tile_id": r["tile_id"], "error": r.get("error")} for r in results if r["status"] != "complete"]
        errors += [{"tile_id": t, "error": "Unattempted: experiment stopped", "attempted": False} for t in unattempted]
        save_new(case / "predictions.json", {
            "id": row["id"], "image_sha256": row["image_sha256"], "config_sha256": digest(config),
            "predictions": [p for r in results for p in r["predictions"]], "errors": errors,
            "tiles": len(grid), "completed_tiles": sum(r["status"] == "complete" for r in results),
            "attempted_tiles": len(results), "unattempted_tiles": unattempted,
            "stopped": True, "stop_request_sha256": sha256(stop),
            "ledger_request_ids": sorted({i for r in results for i in r["ledger_request_ids"]}),
            "runtime_seconds": sum(r["runtime_seconds"] for r in results),
            "runtime_scope": "Stored tile durations only; original page preparation unavailable for interrupted case",
            "reserved_or_charged_usd": sum(r["reserved_or_charged_usd"] for r in results)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    finalize(args.panel, args.run)
