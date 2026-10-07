"""Training-only crops with complete annotation coverage and traceable negatives."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from .data import SYMBOLS, digest, read_graph, save_new, sha256


def crop_box(box, size, minimum=768):
    a, b, c, d = box
    side = max(minimum, math.ceil(max(c - a, d - b) * 1.25 / 64) * 64)
    x = max(0, min(max(0, size[0] - side), round((a + c - side) / 2)))
    y = max(0, min(max(0, size[1] - side), round((b + d - side) / 2)))
    return [x, y, x + side, y + side]


def contained(box, crop):
    return all((box[0] >= crop[0], box[1] >= crop[1], box[2] <= crop[2], box[3] <= crop[3]))


def overlap(box, crop):
    return max(0, min(box[2], crop[2]) - max(box[0], crop[0])) * max(0, min(box[3], crop[3]) - max(box[1], crop[1]))


def drawing_plan(nodes, size):
    physical = [n for n in nodes if n["label"] in SYMBOLS]
    counts = Counter(n["label"] for n in physical)
    # Rare classes first, deterministic tie breaking. A crop can cover neighbors,
    # but small glyphs downscaled below eight pixels still get their own crop.
    ordered = sorted(physical, key=lambda n: (counts[n["label"]], n["label"], n["id"]))
    covered, crops = set(), []
    for anchor in ordered:
        if anchor["id"] in covered:
            continue
        crop = crop_box(anchor["bbox"], size)
        scale = 768 / (crop[2] - crop[0])
        visible = [n["id"] for n in physical if contained(n["bbox"], crop) and (
            scale == 1 or min(n["bbox"][2] - n["bbox"][0], n["bbox"][3] - n["bbox"][1]) * scale >= 8)]
        # An unusually large, thin anchor may remain below eight output pixels.
        # Retain it with explicit resolution metadata rather than dropping it.
        visible = sorted(set(visible) | {anchor["id"]})
        covered.update(visible)
        crops.append({"bbox": crop, "anchor": anchor["id"], "covered_ids": visible, "kind": "symbol"})
    helpers = sorted((n for n in nodes if n["label"] not in SYMBOLS), key=lambda n: digest(n["id"]))
    for node in helpers:
        crop = crop_box(node["bbox"], size)
        if crop[2] - crop[0] == 768 and not any(overlap(n["bbox"], crop) > 0 for n in physical):
            crops.append({"bbox": crop, "anchor": node["id"], "covered_ids": [], "kind": "background"})
            break
    assert covered == {n["id"] for n in physical}
    return physical, crops


def plan(manifest_path, output):
    manifest = json.loads(Path(manifest_path).read_text())
    assert digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest["manifest_sha256"]
    rows, counts = [], Counter()
    for row in manifest["drawings"]:
        if row["split"] != "train":
            continue
        path = Path(manifest["dataset_root"]) / row["graph"]
        assert sha256(path) == row["graph_sha256"]
        physical, crops = drawing_plan(read_graph(path)["nodes"], row["size"])
        counts.update(n["label"] for n in physical)
        rows.append({"source": row, "physical": physical, "crops": crops})
    value = {"manifest_sha256": manifest["manifest_sha256"], "dataset_root": manifest["dataset_root"],
             "source_sha256": sha256(__file__), "output_size": 768, "drawings": rows,
             "classes": list(SYMBOLS), "symbol_counts": dict(counts),
             "crop_count": sum(len(r["crops"]) for r in rows),
             "background_crops": sum(c["kind"] == "background" for r in rows for c in r["crops"]),
             "coverage": "Every training physical symbol has a fully contained anchor/neighbor crop. Resolution exceptions retained explicitly."}
    value["plan_sha256"] = digest(value)
    save_new(output, value)
    return value


def render(plan_path, output):
    value = json.loads(Path(plan_path).read_text())
    assert digest({k: v for k, v in value.items() if k != "plan_sha256"}) == value["plan_sha256"]
    assert value["source_sha256"] == sha256(__file__), "Crop code changed after planning"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples, sources = [], []
    for index, row in enumerate(value["drawings"]):
        source = row["source"]
        assert source["split"] == "train"
        root = output / digest(source["id"])[:16]
        root.mkdir(exist_ok=True)
        done = root / "samples.json"
        if done.exists():
            record = json.loads(done.read_text())
            assert record["plan_sha256"] == value["plan_sha256"]
            for sample in record["samples"]:
                assert sha256(output / sample["image"]) == sample["image_sha256"]
            samples.extend(record["samples"])
        else:
            image_path = Path(value["dataset_root"]) / source["image"]
            assert sha256(image_path) == source["image_sha256"]
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            records = []
            for number, crop in enumerate(row["crops"]):
                a, b, c, d = crop["bbox"]
                scale = 768 / (c - a)
                patch = Image.new("RGB", (c - a, d - b), "white")
                patch.paste(image.crop((a, b, min(c, image.width), min(d, image.height))), (0, 0))
                patch = patch.resize((768, 768), Image.Resampling.LANCZOS)
                path = root / f"crop-{number:04d}.png"
                patch.save(path)
                boxes, labels, identities = [], [], []
                for n in row["physical"]:
                    x0, y0, x1, y1 = n["bbox"]
                    area = overlap(n["bbox"], crop["bbox"])
                    if area > 4 and area >= 0.5 * (x1 - x0) * (y1 - y0):
                        boxes.append([(max(x0, a) - a) * scale, (max(y0, b) - b) * scale,
                                      (min(x1, c) - a) * scale, (min(y1, d) - b) * scale])
                        labels.append(SYMBOLS.index(n["label"]) + 1)
                        identities.append(n["id"])
                assert set(crop["covered_ids"]) <= set(identities)
                records.append({"image": str(path.relative_to(output)), "image_sha256": sha256(path),
                                "source_id": source["id"], "source_group": source["group"],
                                "origin": [a, b], "source_crop_size": c - a, "source_to_crop_scale": scale,
                                "boxes": boxes, "labels": labels, "reference_ids": identities,
                                "fully_covered_ids": crop["covered_ids"], "kind": crop["kind"],
                                "minimum_box_side_px": min((min(box[2] - box[0], box[3] - box[1]) for box in boxes), default=None)})
            save_new(done, {"plan_sha256": value["plan_sha256"], "samples": records})
            samples.extend(records)
        sources.append({k: source[k] for k in ("id", "group", "split", "image_sha256", "graph_sha256")})
        if (index + 1) % 50 == 0:
            print(f"Rendered {index + 1}/{len(value['drawings'])} drawings; {len(samples)} crops", flush=True)
    result = {"manifest_sha256": value["manifest_sha256"], "plan_sha256": value["plan_sha256"],
              "classes": list(SYMBOLS), "sources": sources, "samples": samples,
              "crop_size": 768, "symbol_counts": value["symbol_counts"]}
    result["training_sha256"] = digest(result)
    save_new(output / "training.json", result)
    print(f"Completed {len(samples)} crops with complete training-symbol coverage", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "render"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = plan(args.source, args.out) if args.command == "plan" else render(args.source, args.out)
    if result is not None:
        print(json.dumps({k: result[k] for k in ("crop_count", "background_crops", "symbol_counts")}))
