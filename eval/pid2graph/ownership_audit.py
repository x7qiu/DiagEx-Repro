"""Replay geometry only to explain recognized raster proposals missing from output.

Validation-only analysis. No model calls, GraphML reads or prediction changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagex.config import ScanConfig, TilingConfig
from diagex.vision.loader import iter_pages, load
from diagex.vision.perception import (
    PerceivedObject,
    _bbox_center_is_owned,
    _project_normalized_bbox,
)
from diagex.vision.tiling import AspectAwareStrategy, ownership_core, tile
from diagex.vision.views import ViewProvider

from .data import digest, save_new, sha256
from .epoch_train import sealed


def run(panel_path, run_path, audit_path, output):
    panel = sealed(panel_path, "panel_sha256")
    root = Path(run_path)
    config = json.loads((root / "config.json").read_text())
    audit = json.loads(Path(audit_path).read_text())
    if (config["split"] != "validation" or config["panel_sha256"] != panel["panel_sha256"]
            or config["variant"] not in {"explicit_proposal_decisions", "proposal_source_crops"}
            or audit["run_config_sha256"] != sha256(root / "config.json")
            or not audit["attempted_coverage_complete"]):
        raise ValueError("Requires completed matching validation raster-review records")
    sources = {}
    for name in ("vision/perception.py", "vision/loader.py", "vision/tiling.py"):
        actual = sha256(Path(__file__).resolve().parents[2] / "src/diagex" / name)
        if actual != config["source_fingerprints"][name]:
            raise ValueError("Geometry implementation changed since inference")
        sources[name] = actual
    sources["vision/views.py"] = sha256(Path(__file__).resolve().parents[2] / "src/diagex/vision/views.py")
    tc, sc = TilingConfig(**config["tiling"]), ScanConfig(**config["scan"])
    lookup = {r["id"]: r for r in audit["cases"]}
    if set(lookup) != {r["id"] for r in panel["panels"]["validation"]}:
        raise ValueError("Audit drawing coverage differs")
    cases, checked_objects, checked_tiles = [], 0, 0
    for row in panel["panels"]["validation"]:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Validation image changed")
        lost = set(lookup[row["id"]]["ids"]["recognized_without_retained_output"])
        observations = {i: [] for i in sorted(lost)}
        page = next(iter_pages(load(row["image"], tc, sc)))
        grid = tile(page, AspectAwareStrategy(tc.max_tokens_per_tile, tc.overlap_frac, tc.token_per_pixel))
        views = ViewProvider(page, grid)
        for current in grid:
            path = root / digest(row["id"])[:16] / (current.id + ".json")
            raw = json.loads(path.read_text())
            if raw["config_sha256"] != digest(config):
                raise ValueError("Tile configuration changed")
            checked_tiles += 1
            _, info = views.get_tile(current.id)
            core = ownership_core(current, grid)
            owned = set()
            for value in raw.get("outcome", {}).get("objects", []):
                identity = value.get("attributes", {}).get("raster_proposal_id")
                if not identity:
                    continue
                obj = PerceivedObject.model_validate(value)
                if obj.candidate_id or obj.bbox is None:
                    raise ValueError("Raster observation unexpectedly claims native identity or lacks geometry")
                box = _project_normalized_bbox(obj.bbox, page=page, view_info=info)
                is_owned = _bbox_center_is_owned(box, core, page)
                checked_objects += 1
                if is_owned:
                    owned.add(identity)
                if identity in lost:
                    observations[identity].append({"tile": current.id, "tile_sha256": sha256(path),
                        "bbox": box.model_dump(), "ownership_core": core.model_dump(), "owned": is_owned})
            retained = {p["attributes"]["raster_proposal_id"] for p in raw["predictions"]
                        if p.get("attributes", {}).get("raster_vlm_decision") == "symbol"}
            if owned != retained:
                raise ValueError(f"Ownership replay does not explain saved output: {row['id']} {current.id}")
        if any(not rows or any(r["owned"] for r in rows) for rows in observations.values()):
            raise ValueError("At least one recognized-without-output proposal has another explanation")
        cases.append({"id": row["id"], "collection": row["collection"],
                      "recognized_without_output": len(lost), "unowned_observations": observations})
        print(row["id"], len(lost), "missing recognized proposals explained by ownership", flush=True)
    report = {"panel_sha256": panel["panel_sha256"], "split": "validation", "cases": cases,
              "drawings": len(cases), "checked_tiles": checked_tiles, "checked_raster_objects": checked_objects,
              "recognized_without_output": sum(c["recognized_without_output"] for c in cases),
              "owned_object_ids_equal_saved_prediction_ids_on_every_tile": True,
              "audit_sha256": sha256(audit_path), "run_config_sha256": sha256(root / "config.json"),
              "source_sha256": sources, "api_calls": 0,
              "interpretation": "All missing recognized proposals occur only as objects whose projected centers are outside their observing tile's ownership core. This identifies the deterministic filtering mechanism, not why the VLM omitted recognition in the owning view. Removing ownership may add duplicate/partial detections and is not evaluated or proposed as a new variant here. Validation images only; no GraphML, new inference, or altered predictions."}
    save_new(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "run", "audit", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.panel, args.run, args.audit, args.out)
