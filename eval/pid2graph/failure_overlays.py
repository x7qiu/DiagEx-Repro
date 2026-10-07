"""Scoring-side failure figures. Annotated images must never enter inference."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .data import SYMBOLS, digest, read_graph, save_new, sha256


def render(report_path, manifest_path, run_dir, output, maximum=3):
    report = json.loads(Path(report_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed")
    if report["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Scoring report belongs to another manifest")
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError("Failure figures require a new output directory")
    if digest(json.loads((run_dir / "config.json").read_text())) != report["config_sha256"]:
        raise ValueError("Run configuration differs from report")
    output.mkdir(parents=True)
    lookup = {r["id"]: r for r in manifest["drawings"]}
    cases = sorted((r for r in report["cases"] if r["status"] != "missing"),
                   key=lambda r: (r["symbols"]["f1"] or 0, r["id"]))[:maximum]
    figures = []
    for case in cases:
        row = lookup[case["id"]]
        source = Path(manifest["dataset_root"]) / row["image"]
        graph_path = Path(manifest["dataset_root"]) / row["graph"]
        if sha256(source) != row["image_sha256"] or sha256(graph_path) != row["graph_sha256"]:
            raise ValueError("Scored source changed")
        truth = [n for n in read_graph(graph_path)["nodes"] if n["label"] in SYMBOLS]
        predictions = json.loads((run_dir / digest(row["id"])[:16] / "predictions.json").read_text())["predictions"]
        fp_ids, missed_ids = set(case["false_positive_ids"]), set(case["missed_ids"])
        with Image.open(source) as im:
            original = im.convert("RGB")
        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        for t in truth:
            if t["id"] in missed_ids:
                draw.rectangle(t["bbox"], outline="#2474d2", width=4)
        for p in predictions:
            draw.rectangle(p["bbox"], outline="#d83d40" if p["id"] in fp_ids else "#148750", width=3)
        overlay.thumbnail((2000, 2000))
        key = digest(row["id"])[:16]
        overlay.save(output / f"{key}-page.png")
        examples = [("FP", p) for p in sorted(predictions, key=lambda r: (-r["confidence"], r["id"])) if p["id"] in fp_ids][:2]
        examples += [("MISS", t) for t in truth if t["id"] in missed_ids][:2]
        details = Image.new("RGB", (512 * max(1, len(examples)), 552), "white")
        for i, (kind, target) in enumerate(examples):
            a, b, c, d = target["bbox"]
            side = min(max(original.size), max(256, math.ceil(max(c - a, d - b) * 1.6)))
            x = max(0, min(original.width - side, round((a + c - side) / 2)))
            y = max(0, min(original.height - side, round((b + d - side) / 2)))
            crop = original.crop((x, y, min(original.width, x + side), min(original.height, y + side)))
            cd = ImageDraw.Draw(crop)
            for p in predictions:
                aa, bb, cc, dd = p["bbox"]
                if cc > x and dd > y and aa < x + side and bb < y + side:
                    cd.rectangle((aa - x, bb - y, cc - x, dd - y), outline="#d83d40" if p["id"] in fp_ids else "#148750", width=2)
            for t in truth:
                aa, bb, cc, dd = t["bbox"]
                if t["id"] in missed_ids and cc > x and dd > y and aa < x + side and bb < y + side:
                    cd.rectangle((aa - x, bb - y, cc - x, dd - y), outline="#2474d2", width=2)
            crop.thumbnail((512, 512))
            details.paste(crop, (i * 512, 40))
            ImageDraw.Draw(details).text((i * 512 + 8, 10), f"{kind}: {target['label']} ({target['id']})", fill="black")
        details.save(output / f"{key}-details.png")
        figures.append({"id": row["id"], "metrics": case["symbols"], "status": case["status"],
                        "overview": f"{key}-page.png", "details": f"{key}-details.png",
                        "examples": [{"type": k, "id": r["id"], "label": r["label"]} for k, r in examples]})
    save_new(output / "index.json", {"report_sha256": sha256(report_path), "figures": figures,
             "colors": {"green": "matched prediction", "red": "false positive", "blue": "missed truth"},
             "prohibition": "Scoring-only annotated artifacts. Never provide these images or reference GraphML to validation/test inference."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "manifest", "run", "out"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    render(args.report, args.manifest, args.run, args.out)
