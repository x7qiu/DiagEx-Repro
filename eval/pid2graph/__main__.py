"""Commands for frozen PID2Graph development. Run from the DiagEx project directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import digest, freeze, inventory, read_graph, save_new, sha256
from .scoring import aggregate, score_drawing


def score(manifest_path, panel_path, run_dir, output, split, ledger_path=None):
    manifest = json.loads(Path(manifest_path).read_text())
    panel = json.loads(Path(panel_path).read_text())
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Manifest changed")
    if digest({k: v for k, v in panel.items() if k != "panel_sha256"}) != panel["panel_sha256"]:
        raise ValueError("Panel changed")
    if panel["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Panel and manifest differ")
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    ledger_path = Path(ledger_path) if ledger_path else run_dir.parent / "spending.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    ledger_rows = {r["id"]: r for r in ledger["requests"]} if ledger else {}
    category = "final" if split == "test" else "baseline" if config["variant"] in {"baseline", "ink_guard", "nms", "ink_guard_nms"} else "experiments"
    if config["panel_sha256"] != panel["panel_sha256"] or config["split"] != split:
        raise ValueError("Run configuration does not match evaluation panel")
    lookup = {r["id"]: r for r in manifest["drawings"]}
    scores = []
    billed_ids = set()
    for case in panel["panels"][split]:
        row = lookup[case["id"]]
        path = Path(manifest["dataset_root"]) / row["graph"]
        if sha256(path) != row["graph_sha256"]:
            raise ValueError("Truth changed after freeze")
        graph = read_graph(path)
        prediction_path = run_dir / digest(row["id"])[:16] / "predictions.json"
        result = json.loads(prediction_path.read_text()) if prediction_path.exists() else None
        if result and (result["config_sha256"] != digest(config) or result["image_sha256"] != row["image_sha256"]):
            raise ValueError("Prediction configuration/source mismatch")
        predictions = result["predictions"] if result else []
        current = score_drawing(graph, predictions)
        charged = result["reserved_or_charged_usd"] if result else None
        if result and ledger and config["variant"] != "supervised_detector":
            request_ids = result.get("ledger_request_ids")
            if request_ids is None:
                checkpoints = [json.loads(p.read_text()) for p in prediction_path.parent.glob("*.json") if p.name != "predictions.json"]
                request_ids = [identity for r in checkpoints for identity in r.get("ledger_request_ids", [])]
            if not request_ids and predictions:
                raise ValueError("Paid prediction provenance lacks ledger request IDs")
            if any(identity not in ledger_rows for identity in request_ids):
                raise ValueError("Prediction refers to unknown ledger requests")
            # Older runner invocations captured the shared ledger's time window.
            # Category filtering prevents concurrent legend/experiment requests
            # from being attributed to this run. No reservations are released.
            charged = sum(ledger_rows[i]["charged_usd"] for i in set(request_ids)
                          if ledger_rows[i]["category"] == category and ledger_rows[i]["model"] == config["model"])
            billed_ids.update(i for i in request_ids
                              if ledger_rows[i]["category"] == category and ledger_rows[i]["model"] == config["model"])
        current.update(id=row["id"], collection=row["collection"],
                       status="missing" if result is None else "partial" if result["errors"] else "complete",
                       runtime_seconds=result["runtime_seconds"] if result else None,
                       reserved_or_charged_usd=charged,
                       graph_eligible=score_drawing(graph, [p for p in predictions if p["disposition"] == "graph_eligible"]))
        scores.append(current)
    report = {"manifest_sha256": manifest["manifest_sha256"], "panel_sha256": panel["panel_sha256"],
              "config": config, "config_sha256": digest(config), "split": split,
              "complete": all(r["status"] == "complete" for r in scores),
              "summary": aggregate(scores), "cases": scores,
              "per_collection": {c: aggregate([r for r in scores if r["collection"] == c])
                                 for c in sorted({r["collection"] for r in scores})},
              "graph_eligible": aggregate([r["graph_eligible"] for r in scores]),
              "runtime_seconds": sum(r["runtime_seconds"] or 0 for r in scores),
              "reserved_or_charged_usd": sum(r["reserved_or_charged_usd"] or 0 for r in scores),
              "stopped": (run_dir / "stop-request.json").exists(),
              "stop_request_sha256": sha256(run_dir / "stop-request.json") if (run_dir / "stop-request.json").exists() else None,
              "accounting": {"ledger_sha256": sha256(ledger_path), "category": category,
                             "method": "Unique recorded request IDs filtered by run category/model; reservations retained"} if ledger else None,
              "limitations": manifest["limits"]}
    if ledger and ledger.get("schema_version") == 2:
        from diagex.llm.billing import summarize
        report["billing"] = summarize(ledger, billed_ids)
        report["accounting"]["method"] = "Authoritative billed cost is separate from active reservations and unresolved upper bounds; legacy reserved_or_charged_usd is budget exposure only"
        report["accounting"]["prior_study_spend"] = ledger["prior_spend"]
    save_new(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--root", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("freeze")
    a.add_argument("--audit", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--per-collection", type=int, default=8)
    a = sub.add_parser("run")
    a.add_argument("--panel", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--ledger", required=True)
    a.add_argument("--prices", required=True)
    a.add_argument("--split", choices=["train", "validation", "test", "development_exposed"], default="validation")
    a.add_argument("--variant", default="baseline")
    a.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    a.add_argument("--selection")
    a.add_argument("--max-drawings", type=int)
    a.add_argument("--request-timeout", type=float, default=120)
    a.add_argument("--proposal-run")
    a.add_argument("--max-new-tiles", type=int)
    a = sub.add_parser("score")
    a.add_argument("--manifest", required=True)
    a.add_argument("--panel", required=True)
    a.add_argument("--run", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--ledger")
    args = p.parse_args()
    if args.command == "audit":
        r = inventory(args.root, args.out)
        print(json.dumps({"complete": len(r["complete"]), "patches": len(r["patches"]), "errors": r["errors"]}))
    elif args.command == "freeze":
        r = freeze(args.audit, args.out)
        from collections import Counter
        print(json.dumps(dict(Counter(r["split"] for r in r["drawings"]))))
    elif args.command == "prepare":
        from .runner import prepare
        r = prepare(args.manifest, args.out, per_collection=args.per_collection)
        print(json.dumps({k: len(v) for k, v in r["panels"].items()}))
    elif args.command == "run":
        from .runner import run
        run(args.panel, args.out, args.ledger, args.prices, split=args.split, model=args.model,
            variant=args.variant, selection=args.selection, max_drawings=args.max_drawings,
            request_timeout=args.request_timeout, proposal_run=args.proposal_run, max_new_tiles=args.max_new_tiles)
    else:
        r = score(args.manifest, args.panel, args.run, args.out, args.split, args.ledger)
        print(json.dumps({"complete": r["complete"], "summary": r["summary"]}, indent=2))


if __name__ == "__main__":
    main()
