"""Freeze selected source crops and reviewed instances without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import fitz
from PIL import Image

from diagex.config import Config
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import DiagramPage
from diagex.vision.tiling import AspectAwareStrategy, ownership_core, tile
from diagex.vision.views import ViewProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--tiles", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads((args.source_run / "detection.json").read_text())
    workbench = json.loads((args.source_run / "workbench.json").read_text())
    source = Path(workbench["source_path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == bundle["source_sha256"]
    truth = [
        dict(t, page_index=t["page"] - 1)
        for t in json.loads(args.truth.read_text())
        if t["category"] != "valve_callout"
    ]
    cfg = Config()
    strategy = AspectAwareStrategy(
        max_tokens_per_tile=cfg.tiling.max_tokens_per_tile,
        overlap_frac=cfg.tiling.overlap_frac,
        token_per_pixel=cfg.tiling.token_per_pixel,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    with fitz.open(source) as pdf:
        for tid in args.tiles:
            index = int(tid.split("-")[0][1:])
            evidence = PageEvidence.model_validate_json(
                (args.source_run / "evidence" / f"page-{index + 1:04d}.json").read_text()
            )
            pix = pdf[index].get_pixmap(
                matrix=fitz.Matrix(evidence.dpi / 72, evidence.dpi / 72), alpha=False
            )
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            assert image.size == (evidence.width, evidence.height)
            page = DiagramPage(
                **{
                    k: getattr(evidence, k)
                    for k in (
                        "page_index",
                        "source_ref",
                        "width",
                        "height",
                        "dpi",
                        "effective_dpi",
                        "is_scanned",
                    )
                },
                image=image,
            )
            tiles = tile(page, strategy)
            current = next(t for t in tiles if t.id == tid)
            core = ownership_core(current, tiles)
            view, info = ViewProvider(page, tiles).get_tile(tid)

            def owned(row, index=index, core=core):
                b = row["bbox"]
                return (
                    row["page_index"] == index
                    and core.x <= b["x"] + b["w"] / 2 < core.x2
                    and core.y <= b["y"] + b["h"] / 2 < core.y2
                )

            case = {
                "source_sha256": bundle["source_sha256"],
                "tile_id": tid,
                "page_index": index,
                "tile": current.model_dump(mode="json", exclude={"image"}),
                "core": core.model_dump(),
                "view_info": {**asdict(info), "page_bbox": info.page_bbox.model_dump()},
                "candidates": [c for c in bundle["candidates"] if owned(c)],
                "truth": [t for t in truth if owned(t)],
            }
            atomic_write_json(args.output / f"{tid}.json", case)
            view.save(args.output / f"{tid}.png")
            print(
                f"{tid}: {len(case['candidates'])} candidates, {len(case['truth'])} reviewed instances"
            )


if __name__ == "__main__":
    main()
