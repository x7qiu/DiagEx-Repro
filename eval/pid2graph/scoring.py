"""Immutable detection scoring, independent of the extraction implementation."""
from __future__ import annotations

import math
from collections import Counter

from .data import SYMBOLS


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def validate_prediction(row):
    b = row["bbox"]
    if len(b) != 4 or not all(math.isfinite(v) for v in b) or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("Prediction bbox must be finite positive-area xyxy")
    if not math.isfinite(row["confidence"]) or not 0 <= row["confidence"] <= 1:
        raise ValueError("Prediction confidence must be finite and in [0,1]")


def matches(truth, predictions, *, class_aware=True, threshold=0.5):
    used = set()
    found = []
    # Stable tie-breaking must not depend on truth classes or overlap.
    order = sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], str(predictions[i]["id"])))
    for pi in order:
        p = predictions[pi]
        eligible = [(iou(t["bbox"], p["bbox"]), str(t["id"]), ti) for ti, t in enumerate(truth)
                    if ti not in used and (not class_aware or t["label"] == p["label"])]
        eligible = [x for x in eligible if x[0] >= threshold]
        if eligible:
            overlap, _, ti = sorted(eligible, key=lambda x: (-x[0], x[1]))[0]
            used.add(ti)
            found.append({"truth_id": truth[ti]["id"], "prediction_id": p["id"], "iou": overlap,
                          "truth_label": truth[ti]["label"], "prediction_label": p["label"]})
    return found


def metrics(tp, predicted, actual):
    precision = tp / predicted if predicted else None
    recall = tp / actual if actual else None
    return {"tp": tp, "fp": predicted - tp, "fn": actual - tp,
            "predicted": predicted, "actual": actual, "precision": precision, "recall": recall,
            "f1": 2 * tp / (predicted + actual) if predicted + actual else None}


def score_drawing(graph, predictions):
    for row in predictions:
        validate_prediction(row)
    if len({p["id"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    truth = [n for n in graph["nodes"] if n["label"] in SYMBOLS]
    classified = matches(truth, predictions)
    localized = matches(truth, predictions, class_aware=False)
    correct_ids = {m["truth_id"] for m in classified}
    prediction_ids = {m["prediction_id"] for m in classified}
    return {
        "symbols": metrics(len(classified), len(predictions), len(truth)),
        "localization": metrics(len(localized), len(predictions), len(truth)),
        "classification_given_localization": {
            "correct": sum(m["truth_label"] == m["prediction_label"] for m in localized),
            "total": len(localized)},
        "per_class": {c: metrics(sum(m["truth_label"] == c for m in classified),
                                 sum(p["label"] == c for p in predictions),
                                 sum(t["label"] == c for t in truth)) for c in SYMBOLS},
        "unknown_prediction_labels": dict(Counter(p["label"] for p in predictions if p["label"] not in SYMBOLS)),
        "missed_ids": [t["id"] for t in truth if t["id"] not in correct_ids],
        "false_positive_ids": [p["id"] for p in predictions if p["id"] not in prediction_ids],
        "matches": classified, "localization_matches": localized,
    }


def aggregate(results):
    def pooled(rows):
        return metrics(sum(r["tp"] for r in rows), sum(r["predicted"] for r in rows), sum(r["actual"] for r in rows))
    return {"drawings": len(results),
            "symbols": pooled([r["symbols"] for r in results]),
            "localization": pooled([r["localization"] for r in results]),
            "macro_drawing_f1": (sum(r["symbols"]["f1"] or 0 for r in results) / len(results)) if results else None,
            "per_class": {c: pooled([r["per_class"][c] for r in results]) for c in SYMBOLS}}
