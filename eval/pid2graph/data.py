"""Read-only dataset inventory and drawing-group splits. Never supplies truth to inference."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SYMBOLS = ("general", "valve", "pump", "tank", "instrumentation", "arrow", "inlet/outlet")
HELPERS = ("connector", "crossing", "border", "background")
NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
VERSION = "pid2graph-symbols/1.0"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def read_graph(path):
    raw = Path(path).read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError(f"DTD/entity declarations are unsupported: {path}")
    root = ET.fromstring(raw)
    keys = {k.attrib["id"]: k.attrib["attr.name"] for k in root.findall("g:key", NS)}
    defaults = {k.attrib["attr.name"]: k.find("g:default", NS).text
                for k in root.findall("g:key", NS) if k.find("g:default", NS) is not None}
    def attrs(element):
        return {**defaults, **{keys[v.attrib["key"]]: v.text for v in element.findall("g:data", NS)}}
    graph = root.find("g:graph", NS)
    if graph is None:
        raise ValueError(f"Missing graph: {path}")
    nodes = []
    for node in graph.findall("g:node", NS):
        a = attrs(node)
        box = [float(a[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
        if not all(math.isfinite(v) for v in box):
            raise ValueError(f"Invalid box {node.attrib['id']}: {path}")
        nodes.append({"id": node.attrib["id"], "label": a["label"], "bbox": box})
    ids = {n["id"] for n in nodes}
    if len(ids) != len(nodes):
        raise ValueError(f"Duplicate node IDs: {path}")
    edges = []
    for edge in graph.findall("g:edge", NS):
        a = attrs(edge)
        if edge.attrib["source"] not in ids or edge.attrib["target"] not in ids:
            raise ValueError(f"Dangling edge: {path}")
        edges.append({"source": edge.attrib["source"], "target": edge.attrib["target"],
                      "label": a.get("edge_label", "unknown")})
    return {"directed": graph.attrib.get("edgedefault") == "directed", "nodes": nodes, "edges": edges}


def paired_image(graph):
    matches = [graph.with_suffix(ext) for ext in IMAGE_EXTS if graph.with_suffix(ext).exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one paired image: {graph}")
    return matches[0]


def image_signature(path):
    with Image.open(path) as image:
        size = list(image.size)
        gray = image.convert("L")
        thumb = gray.resize((64, 64), Image.Resampling.LANCZOS)
        small = list(gray.resize((17, 16), Image.Resampling.LANCZOS).getdata())
        dhash = 0
        for y in range(16):
            for x in range(16):
                dhash = (dhash << 1) | (small[y * 17 + x] > small[y * 17 + x + 1])
        return size, hashlib.sha256(thumb.tobytes()).hexdigest(), f"{dhash:064x}"


def inventory(root, out):
    root, out = Path(root).resolve(), Path(out)
    if out.exists():
        raise FileExistsError("Use a new output directory; audits are immutable")
    complete, patch_rows, errors = [], [], []
    by_parent = {}
    for graph in sorted((root / "Complete").rglob("*.graphml")):
        image = paired_image(graph)
        data = read_graph(graph)
        size, thumb, dhash = image_signature(image)
        counts = Counter(n["label"] for n in data["nodes"])
        row = {"id": f"{graph.parent.name}/{graph.stem}", "collection": graph.parent.name,
               "image": str(image.relative_to(root)), "graph": str(graph.relative_to(root)),
               "image_sha256": sha256(image), "graph_sha256": sha256(graph), "size": size,
               "thumbnail_sha256": thumb, "dhash": dhash, "node_counts": dict(counts),
               "edge_counts": dict(Counter(e["label"] for e in data["edges"])),
               "directed": data["directed"], "patch_count": 0}
        # Re-encoded/resized copies with the same normalized annotation layout stay together.
        row["layout_sha256"] = digest(sorted((n["label"], *[round(v / size[i % 2], 4)
            for i, v in enumerate(n["bbox"])]) for n in data["nodes"]))
        invalid = [n["id"] for n in data["nodes"] if n["label"] in SYMBOLS and (
            n["bbox"][0] < -1 or n["bbox"][1] < -1 or n["bbox"][2] > size[0] + 1
            or n["bbox"][3] > size[1] + 1 or n["bbox"][2] <= n["bbox"][0]
            or n["bbox"][3] <= n["bbox"][1])]
        row["invalid_symbol_ids"] = invalid
        unknown = sorted(set(counts) - set(SYMBOLS) - set(HELPERS))
        if invalid or unknown:
            errors.append({"id": row["id"], "invalid_symbol_ids": invalid, "unknown_labels": unknown})
        complete.append(row)
        by_parent[row["id"]] = row
        if len(complete) % 100 == 0:
            print(f"Audited {len(complete)} complete drawings", flush=True)
    for graph in sorted((root / "Patched").rglob("*.graphml")):
        parent = f"{graph.parent.parent.name}/{graph.parent.name}"
        if parent not in by_parent:
            raise ValueError(f"Orphan patch {graph}")
        image = paired_image(graph)
        data = read_graph(graph)
        by_parent[parent]["patch_count"] += 1
        patch_rows.append({"parent_id": parent, "image": str(image.relative_to(root)),
                           "graph": str(graph.relative_to(root)), "graph_sha256": sha256(graph),
                           "node_count": len(data["nodes"]), "edge_count": len(data["edges"])})
        if len(patch_rows) % 5000 == 0:
            print(f"Audited {len(patch_rows)} patch graphs", flush=True)
    # Validate mapping files as numeric arrays, never unpickle dataset content.
    import numpy as np
    mappings = []
    for path in sorted((root / "Patched").rglob("*.npy")):
        try:
            arr = np.load(path, allow_pickle=False)
            mappings.append({"path": str(path.relative_to(root)), "shape": list(arr.shape),
                             "dtype": str(arr.dtype), "sha256": sha256(path)})
        except ValueError as exc:
            mappings.append({"path": str(path.relative_to(root)), "error": str(exc)})
    result = {"version": VERSION, "root": str(root), "complete": complete, "patches": patch_rows,
              "mappings": mappings, "errors": errors,
              "patch_image_hashes": "Not read: patches are assigned by parent; experiments use complete images only."}
    result["inventory_sha256"] = digest(result)
    save_new(out / "inventory.json", result)
    return result


def freeze(audit_path, out, seed="pid2graph-2026-09-14-v1"):
    audit = json.loads(Path(audit_path).read_text())
    if audit["errors"]:
        raise ValueError("Resolve or explicitly version annotation defects before freezing")
    rows = audit["complete"]
    parent = {r["id"]: r["id"] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def join(a, b):
        a, b = sorted((find(a), find(b)))
        parent[b] = a
    matches = []
    for i, a in enumerate(rows):
        for b in rows[:i]:
            reason = None
            if a["collection"] == b["collection"] == "PID2Graph OPEN100":
                reason = "same previously exposed industrial project"
            elif any(a[k] == b[k] for k in ("image_sha256", "thumbnail_sha256", "layout_sha256")):
                reason = "identical image, thumbnail, or normalized annotation layout"
            elif abs(a["size"][0] / a["size"][1] - b["size"][0] / b["size"][1]) < 0.02 and (
                int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count() <= 6:
                reason = "conservative near-duplicate image grouping"
            if reason:
                join(a["id"], b["id"])
                matches.append({"a": a["id"], "b": b["id"], "reason": reason})
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row)
    assignments = []
    for gid, members in sorted(groups.items()):
        bucket = int(digest([seed, gid])[:8], 16) % 100
        split = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        if any(r["collection"] == "PID2Graph OPEN100" for r in members):
            split = "development_exposed"
        for row in members:
            assignments.append({**row, "group": gid, "split": split})
    contract = {
        "version": VERSION, "coordinates": "original_image_pixels_xyxy", "iou_threshold": 0.5,
        "classes": SYMBOLS, "non_symbol_annotations": HELPERS,
        "prediction_threshold": 0.0, "matching": "confidence-ranked greedy, one-to-one per class",
        "localization_matching": "confidence-ranked greedy ignoring class; wrong classes reported separately",
        "confidence_order": {"high": 0.9, "medium": 0.6, "low": 0.3},
        "unmatched_predictions": "all count as false positives; helper/background annotations do not erase predictions",
        "missing_outputs": "score as empty predictions and separately report failed/incomplete runs",
        "primary_selection": "macro drawing symbol F1 on validation; report recall, precision, per-class and per-collection metrics",
        "ties": "lower measured API cost then lower runtime",
        "test_policy": "select candidate before final evaluation; no tuning on final scores",
        "legend_scope": "No legend or detailed semantics ground truth; separate source-based regression checks only",
        "primary_inventory": "symbol-review inventory: graph-eligible detections plus quarantined VLM symbol proposals",
        "secondary_inventory": "graph-eligible detections only; never promote proposals for scoring",
        "prediction_label_mapping": {
            "instrument": "instrumentation", "opc": "inlet/outlet", "equipment:valve": "valve",
            "equipment:pump": "pump", "equipment:compressor": "pump", "equipment:tank": "tank",
            "equipment:vessel": "tank", "equipment:other": "general"},
    }
    manifest = {"version": VERSION, "seed": seed, "inventory_sha256": audit["inventory_sha256"],
                "dataset_root": audit["root"], "contract": contract, "drawings": assignments,
                "duplicate_links": matches,
                "limits": ["750 synthetic drawings; shared generator vocabularies across splits.",
                           "OPEN100 is a previously exposed single project and excluded from held-out claims.",
                           "No unseen-industrial-project qualification; undetected redraw relationships remain possible.",
                           "Every patch inherits its complete drawing's split; patches are not independent samples."]}
    manifest["manifest_sha256"] = digest(manifest)
    save_new(out, manifest)
    return manifest
