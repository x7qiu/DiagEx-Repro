"""Seal validation selection and enforce unchanged final-test inference settings."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from .data import digest, save_new, sha256


def detector_signature(config):
    return {k: v for k, v in config.items() if k not in {"split", "panel_sha256"}}


def perception_signature(config):
    result = {k: copy.deepcopy(v) for k, v in config.items() if k not in {"split", "panel_sha256"}}
    if "guidance" in result:
        guidance = result["guidance"]
        for key in ("proposal_files_sha256", "detector_runtime_seconds"):
            guidance.pop(key, None)
        guidance["detector_config"] = detector_signature(guidance["detector_config"])
    return result


def postprocess_signature(config):
    return {k: v for k, v in config.items() if k not in {"split", "panel_sha256", "base_config_sha256"}}


def price_signature(price):
    return {**{key: price[key] for key in
               ("context_length", "input_per_token", "output_per_token", "provider_tags")},
            "request_usd": price.get("request_usd", 0.0)}


def ranking_key(report):
    """Accuracy first; reservations must never masquerade as billed cost."""
    metric = report["summary"]["macro_drawing_f1"]
    runtime = report["runtime_seconds"]
    if not isinstance(metric, (int, float)) or not math.isfinite(metric) or not 0 <= metric <= 1:
        raise ValueError("Invalid selection metric")
    if not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime < 0:
        raise ValueError("Invalid selection runtime")
    billing = report.get("billing")
    if billing is None:
        # Preserve the historical phase-1 rule for its archived report format.
        cost = report["reserved_or_charged_usd"]
    else:
        actual = billing["actual_billed_usd"]
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < 0:
            raise ValueError("Invalid actual billed cost")
        cost = actual if billing["actual_billing_complete"] else math.inf
    return (-metric, cost, runtime, report["config"]["variant"])


def read_selection(path, panel):
    if not path:
        raise ValueError("Final evaluation requires a sealed selection record")
    selection = json.loads(Path(path).read_text())
    if digest({k: v for k, v in selection.items() if k != "selection_sha256"}) != selection.get("selection_sha256"):
        raise ValueError("Selection seal is invalid")
    if selection["panel_sha256"] != panel["panel_sha256"]:
        raise ValueError("Selection belongs to another panel")
    return selection


def seal(panel_path, report_paths, output, *, prices_path=None):
    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Panel changed")
    expected = {r["id"] for r in panel["panels"]["validation"]}
    reports = []
    for path in report_paths:
        report = json.loads(Path(path).read_text())
        if report.get("stopped"):
            raise ValueError("A stopped experimental arm is not eligible for selection")
        if report["split"] != "validation" or report["panel_sha256"] != panel["panel_sha256"] or report["manifest_sha256"] != panel["manifest_sha256"]:
            raise ValueError("Only matching validation reports can select a candidate")
        if {r["id"] for r in report["cases"]} != expected or len(report["cases"]) != len(expected):
            raise ValueError("Validation case coverage changed")
        if any(r["status"] == "missing" for r in report["cases"]):
            raise ValueError("Validation drawings remain unattempted")
        if any(r["status"] not in {"complete", "partial"} for r in report["cases"]):
            raise ValueError("Unknown validation case status")
        if digest(report["config"]) != report["config_sha256"]:
            raise ValueError("Report configuration changed")
        if report["config"]["variant"] == "supervised_detector":
            raise ValueError("Detector-only diagnostics are not a VLM pipeline candidate")
        reports.append((report, {"path": str(Path(path).resolve()), "sha256": sha256(path)}))
    if not reports:
        raise ValueError("No validation reports supplied")
    baseline = next((r for r, _ in reports if r["config"]["variant"] == "baseline"), None)
    if baseline is None:
        raise ValueError("Baseline must be included in selection")
    accounted = [r.get("billing") is not None for r, _ in reports]
    if any(accounted) and not all(accounted):
        raise ValueError("Cannot mix legacy exposure reports with actual billing reports")
    # All failed tiles remain in their drawing denominators. A partial drawing
    # with recorded failures is allowed; an unattempted drawing is not.
    winner, _ = min(reports, key=lambda pair: ranking_key(pair[0]))
    config = winner["config"]
    postprocess = config["variant"] in {"ink_guard", "nms", "ink_guard_nms"}
    if postprocess and config["base_config_sha256"] != baseline["config_sha256"]:
        raise ValueError("Postprocessing result does not derive from the included baseline")
    inference = baseline["config"] if postprocess else config
    selected = {"panel_sha256": panel["panel_sha256"], "manifest_sha256": panel["manifest_sha256"],
                "variant": config["variant"], "model": config["model"],
                "inference_variant": inference["variant"], "inference_signature": perception_signature(inference),
                "postprocess_signature": postprocess_signature(config) if postprocess else None,
                "detector_signature": detector_signature(inference["guidance"]["detector_config"]) if "guidance" in inference else None,
                "validation_reports": [r for _, r in reports], "selected_validation_summary": winner["summary"],
                "rule": ("Maximum macro drawing F1 at frozen IoU; exact ties prefer fully accounted actual billed cost, then runtime then name. Unknown actual totals rank after known totals; reservations and unresolved upper bounds are not prices."
                         if all(accounted) else "Maximum macro drawing F1 at frozen IoU; ties by legacy exposure then runtime then name"),
                "failed_tiles": "Retained in full-drawing recall denominators; no test data used in selection"}
    if prices_path is not None:
        price = json.loads(Path(prices_path).read_text())["models"][config["model"]]
        selected["final_price_limits"] = price_signature(price)
        selected["price_snapshot_sha256"] = sha256(prices_path)
    selected["selection_sha256"] = digest(selected)
    save_new(output, selected)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prices", required=True, help="Pin provider price filters before final inference")
    args = parser.parse_args()
    result = seal(args.panel, args.report, args.out, prices_path=args.prices)
    print(json.dumps({"selected": result["variant"], "selection_sha256": result["selection_sha256"]}))
