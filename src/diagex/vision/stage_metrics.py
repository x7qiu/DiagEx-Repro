"""Small module-level evaluators; references must use the same page coordinate frame.

These metrics do not substitute for the frozen PID2Graph study scorer. Relations
require fixed upstream IDs; OCR compares spatially matched text regions, not a
concatenated page reading order. Reference labels must be supplied explicitly.
"""

from __future__ import annotations

import math


def _counts(tp, predicted, expected):
    precision = tp / predicted if predicted else 0.0
    recall = tp / expected if expected else 0.0
    return {
        "tp": tp,
        "fp": predicted - tp,
        "fn": expected - tp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _iou(a, b):
    intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    )
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union else 0.0


def _box(row):
    b = row.get("bbox", row.get("bbox_global"))
    if isinstance(b, dict):
        values = [b[k] for k in ("x", "y", "w", "h")]
    else:  # Raster detector output uses xmin, ymin, xmax, ymax.
        values = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
    if not all(math.isfinite(v) for v in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError("Invalid scoring box")
    return values


def _matching(reference, predicted, score):
    # Maximum-cardinality bipartite matching; each region can count only once.
    neighbors = [
        sorted(((j, score(a, b)) for j, b in enumerate(predicted)), key=lambda x: (-x[1], x[0]))
        for a in reference
    ]
    owners = {}

    def visit(i, seen):
        for j, value in neighbors[i]:
            if value <= 0 or j in seen:
                continue
            seen.add(j)
            if j not in owners or visit(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(reference)):
        visit(i, set())
    return sorted((i, j) for j, i in owners.items())


def _edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def score_boxes(reference, predicted, *, iou=0.5, labels=False):
    def quality(a, b):
        overlap = _iou(_box(a), _box(b))
        same_label = a.get("category", a.get("label", a.get("shape", a.get("kind")))) == b.get(
            "category", b.get("label", b.get("shape", b.get("kind")))
        )
        return overlap if overlap >= iou and (not labels or same_label) else 0

    pairs = _matching(reference, predicted, quality)
    return _counts(len(pairs), len(predicted), len(reference)), pairs


def score_stage(stage, reference, prediction, *, iou=0.5, line_tolerance=3.0):
    if not 0 < iou <= 1 or line_tolerance < 0:
        raise ValueError("Invalid IoU or line tolerance")
    if stage in {"symbol_detection", "symbol_interpretation"}:
        if stage == "symbol_detection":
            actual = (
                prediction.get("raster_proposals")
                if prediction.get("detector") is not None
                else prediction.get("native_candidates", [])
            )
        else:
            actual = [{**d, "category": d["kind"]} for d in prediction.get("detections", [])]
            if not actual and prediction.get("batch", {}).get("candidate_reviews"):
                actual = [
                    {"bbox": r["bbox"], "category": r["object"]["attributes"]["broad_category"]}
                    for r in prediction["batch"]["candidate_reviews"]
                    if r.get("object", {}).get("kind") == "raster_symbol"
                ]
        expected = reference["symbols"]
        return {
            "localization": score_boxes(expected, actual, iou=iou)[0],
            "classification_and_localization": score_boxes(expected, actual, iou=iou, labels=True)[
                0
            ],
        }
    if stage == "text_detection":
        expected, actual = reference["text_spans"], prediction["text_spans"]
        locations, pairs = score_boxes(expected, actual, iou=iou)
        errors = sum(_edit_distance(expected[i]["text"], actual[j]["text"]) for i, j in pairs)
        errors += sum(
            len(r["text"]) for i, r in enumerate(expected) if i not in {a for a, _ in pairs}
        )
        errors += sum(
            len(r["text"]) for j, r in enumerate(actual) if j not in {b for _, b in pairs}
        )
        chars = sum(len(r["text"]) for r in expected)
        return {
            "localization": locations,
            "exact_transcriptions": sum(expected[i]["text"] == actual[j]["text"] for i, j in pairs),
            "spatial_character_errors": errors,
            "reference_characters": chars,
            "spatial_character_error_rate": errors / chars if chars else None,
        }
    if stage == "line_detection":
        expected, actual = reference["paths"], prediction["paths"]

        def quality(a, b):
            x, y = a["points"], b["points"]
            if len(x) != len(y) or not x:
                return 0
            distance = min(
                max(math.dist(p, q) for p, q in zip(x, z, strict=True)) for z in (y, y[::-1])
            )
            return 1 / (1 + distance) if distance <= line_tolerance else 0

        return {
            "polyline_geometry": _counts(
                len(_matching(expected, actual, quality)), len(actual), len(expected)
            ),
            "tolerance_pixels": line_tolerance,
        }
    if stage == "text_assignment":
        # Explicit one-to-many bindings; unknown/unresolved scope is not an assignment.
        actual = {
            (sid, n["id"])
            for n in prediction["nodes"]
            for sid in n.get("attributes", {}).get("source_text_ids", [])
        }
        actual.update(
            (sid, target)
            for b in prediction.get("text_bindings", [])
            if b.get("status") == "supported"
            for sid in b.get("source_text_ids", [])
            for target in b.get("assembly_ids", []) + b.get("node_ids", [])
        )
        expected = {(b["text_id"], b["target_id"]) for b in reference["bindings"]}
        return {"assignment": _counts(len(actual & expected), len(actual), len(expected))}
    if stage == "connection_inference":

        def edges(rows):
            return {(r["from_node"], r["to_node"], r.get("line_type")) for r in rows}

        expected, actual = edges(reference["edges"]), edges(prediction["edges"])
        return {"directed_typed_edges": _counts(len(actual & expected), len(actual), len(expected))}
    raise ValueError("No accuracy metric for visual line assessments; use connection references")
