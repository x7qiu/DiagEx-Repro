"""Check production ink filtering against frozen validation outputs, without truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from diagex.config import ScanConfig, TilingConfig
from diagex.vision.loader import iter_pages, load
from diagex.vision.raster_ink import filter_observations, implementation_sha256, ink_guard

from .data import digest, save_new, sha256
from .postprocess import ink_guard as frozen_ink_guard


def verify(panel_path, run, output):
    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Panel changed")
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    if config["split"] != "validation" or config["panel_sha256"] != panel["panel_sha256"]:
        raise ValueError("Parity uses only matching validation outputs")
    cases = []
    for row in panel["panels"]["validation"]:
        path = run / digest(row["id"])[:16] / "predictions.json"
        saved = json.loads(path.read_text())
        if sha256(row["image"]) != row["image_sha256"] or saved["image_sha256"] != row["image_sha256"]:
            raise ValueError("Source changed")
        predictions = saved["predictions"]
        with Image.open(row["image"]) as image:
            expected = frozen_ink_guard(image, predictions)
            if ink_guard(image, predictions) != expected:
                raise AssertionError("Production pixel filter differs from frozen evaluation")
        page = next(iter_pages(load(row["image"], TilingConfig(**config["tiling"]), ScanConfig(**config["scan"]))))
        sx, sy = row["size"][0] / page.width, row["size"][1] / page.height
        reviews = []
        for index, prediction in enumerate(predictions):
            a, b, c, d = prediction["bbox"]
            box = {"x": round(a / sx), "y": round(b / sy),
                   "w": round((c - a) / sx), "h": round((d - b) / sy)}
            reviews.append({"page_index": 0, "bbox": box, "object": {"kind": "equipment"},
                            "parity_index": index})
        _, kept, audit = filter_observations(
            row["image"], [SimpleNamespace(page_index=0, width=page.width, height=page.height)], [], reviews)
        # Compare by index: provider-generated IDs need not be globally unique.
        with Image.open(row["image"]) as image:
            indexed, _ = frozen_ink_guard(image, [{**p, "id": str(i)} for i, p in enumerate(predictions)])
        if [r["parity_index"] for r in kept] != [int(r["id"]) for r in indexed]:
            raise AssertionError("Original-pixel production adapter changed the retained inventory")
        cases.append({"id": row["id"], "predictions": len(predictions), "kept": len(kept),
                      "rejected": len(audit["rejected"]), "exact_pixel_filter": True,
                      "exact_adapter_inventory": True, "source_predictions_sha256": sha256(path)})
    report = {"panel_sha256": panel["panel_sha256"], "implementation_sha256": implementation_sha256(),
              "split": "validation", "truth_read": False, "api_calls": 0, "cases": cases,
              "predictions": sum(c["predictions"] for c in cases),
              "rejected": sum(c["rejected"] for c in cases), "all_exact": True}
    save_new(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    verify(args.panel, args.run, args.out)
