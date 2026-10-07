"""Check reusable VLM hint payloads against frozen validation inference, without API calls."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from diagex.config import Config
from diagex.vision.loader import iter_pages, load
from diagex.vision.raster_guidance import implementation_sha256, view_guidance
from diagex.vision.tiling import AspectAwareStrategy, tile
from diagex.vision.views import ViewProvider

from .data import digest, save_new, sha256
from .guidance import load_proposals
from .guidance import view_guidance as benchmark_guidance


def check(panel_path, proposal_run, output):
    started = time.monotonic()
    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Panel changed")
    proposals, reference = load_proposals(proposal_run, panel, "validation")
    cfg = Config()
    strategy = AspectAwareStrategy(cfg.tiling.max_tokens_per_tile, cfg.tiling.overlap_frac, cfg.tiling.token_per_pixel)
    cases = []
    for row in panel["panels"]["validation"]:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Source image changed")
        page = next(iter_pages(load(row["image"], cfg.tiling, cfg.scan)))
        grid = tile(page, strategy)
        views = ViewProvider(page, grid)
        scale = (row["size"][0] / page.width, row["size"][1] / page.height)
        checks = []
        for current in grid:
            _, info = views.get_tile(current.id)
            expected = benchmark_guidance(proposals[row["id"]], info.page_bbox, scale)
            actual = view_guidance(proposals[row["id"]], info.page_bbox, scale)
            if expected != actual:
                raise ValueError(f"Guidance mismatch: {row['id']} {current.id}")
            checks.append({"tile_id": current.id, "exact_parity": True, "payload_sha256": digest(actual)})
        cases.append({"id": row["id"], "image_sha256": row["image_sha256"], "tiles": checks})
        print(f"{row['id']}: {len(checks)} hint payloads exactly equal", flush=True)
    report = {"panel_sha256": panel["panel_sha256"], "split": "validation", "cases": cases,
              "tiles": sum(len(c["tiles"]) for c in cases), "all_equal": True,
              "reference_guidance_sha256": reference["guidance_source_sha256"],
              "reusable_guidance_sha256": implementation_sha256(),
              "api_calls": 0, "ground_truth_read": False, "test_images_read": False,
              "runtime_seconds": time.monotonic() - started}
    save_new(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--proposal-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    check(args.panel, args.proposal_run, args.out)
