"""Bounded canary through the production SDK, streaming client and symbol stage.

Requires explicit --run. Never resumes or repeats an uncertain dispatch. Frozen
case JSON/PNG pairs use the format produced by benchmark_symbol_reasoning.py.
The optional final probe seeds uncertain decisions locally to exercise the real
targeted-reasoning request even when all normal first passes are confident.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from evaluate_symbol_perception import score
from PIL import Image
from score_symbol_reasoning import correct_family

from diagex.config import Config
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, Tile
from diagex.vision.perception import _request_diagnostic, perceive_tile
from diagex.vision.symbol_candidates import SYMBOL_PERCEPTION_VERSION, SymbolCandidate
from diagex.vision.views import ViewInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Authorize bounded live inference")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--cases", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-requests", type=int, default=6)
    parser.add_argument("--reasoning-probe", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Live validation requires --run")
    if not 1 <= args.max_requests <= 8:
        parser.error("Canaries allow between one and eight HTTP dispatches")
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against replaying an interrupted paid check.
    with (args.output / "started.json").open("x") as f:
        json.dump(
            {
                "started_at": time.time(),
                "version": SYMBOL_PERCEPTION_VERSION,
                "request_limit": args.max_requests,
            },
            f,
        )

    workbench = json.loads((args.source_run / "workbench.json").read_text())
    bundle = json.loads((args.source_run / "detection.json").read_text())
    cfg = Config()
    cfg.llm = replace(
        cfg.llm,
        transport=workbench["settings"]["provider"],
        model=workbench["settings"]["vision_model"],
        openrouter_base_url=workbench["settings"]["base_url"],
        reasoning_mode="disabled",
    )
    client = LLMClient(cfg.llm, budgets=cfg.budgets)
    sent = []

    def before_dispatch(request):
        if len(sent) >= args.max_requests:
            raise RuntimeError("Canary HTTP request budget exhausted; no further dispatch")
        payload = json.loads(request.content)
        record = {
            "number": len(sent) + 1,
            "started_at": time.time(),
            "url": str(request.url),
            "request": _request_diagnostic(payload),
        }
        atomic_write_json(args.output / f"dispatch-{len(sent) + 1:02d}.json", record)
        sent.append(record)

    # Observe the actual serialized SDK request, including stream=True, without
    # changing routing, retries, auth, request shape or the production transport.
    client._client._client.event_hooks["request"].append(before_dispatch)
    results = []

    def evaluate(case_path, *, probe=False):
        case = json.loads(case_path.read_text())
        p = PageEvidence.model_validate_json(
            (args.source_run / "evidence" / f"page-{case['page_index'] + 1:04d}.json").read_text()
        )
        candidates = [SymbolCandidate.model_validate(c) for c in case["candidates"]]
        current_ids = {c["id"]: c for c in bundle["candidates"]}
        assert all(current_ids[c.id] == c.model_dump(mode="json") for c in candidates)
        vi = {**case["view_info"], "page_bbox": BBox(**case["view_info"]["page_bbox"])}
        events = []
        name = case["tile_id"] + ("--reasoning-probe" if probe else "")

        def save_event(event):
            events.append(event)
            atomic_write_json(args.output / f"{name}.events.json", events)

        class SeedUncertainty:
            first = True

            def messages_create(self, **kwargs):
                if self.first:
                    self.first = False
                    return SimpleNamespace(
                        usage=None,
                        content=[
                            {
                                "type": "tool_use",
                                "name": "submit_pid_objects",
                                "input": {
                                    "candidate_results": [
                                        {
                                            "candidate_id": c.id,
                                            "decision": "uncertain",
                                            "reason": "Seeded uncertainty for the production reasoning-transport canary",
                                        }
                                        for c in candidates
                                    ]
                                },
                            }
                        ],
                    )
                return client.messages_create(**kwargs)

        started = time.monotonic()
        outcome = perceive_tile(
            client=SeedUncertainty() if probe else client,
            cost_tracker=CostTracker(),
            reporter=NullReporter(),
            page=p,
            tile=Tile.model_validate(case["tile"]),
            view_image=Image.open(case_path.with_suffix(".png")).convert("RGB"),
            view_info=ViewInfo(**vi),
            ownership_bbox=BBox(**case["core"]),
            legend_summary=[
                e for e in bundle["legend_pack"]["entries"]
                if e.get("source") == "customer_override"
                or e.get("attributes", {}).get("row_status") not in {"uncertain", "reject"}
            ],
            step=1,
            page_context={"coverage": "deterministic fixed grid"},
            candidates=candidates,
            reasoning_mode="enabled",
            policy=cfg.symbol_perception,
            on_diagnostic=save_event,
        )
        detections = [d.model_dump(mode="json") for d in outcome.detections]
        counts = Counter(d["attributes"].get("symbol_candidate_id") for d in detections)
        assert all(n == 1 for cid, n in counts.items() if cid), "Duplicate native candidate detection"
        selected = {r["candidate_id"] for r in outcome.batch.candidate_reviews if r["status"] == "selected"}
        assert set(counts) == selected, "Published detections and review states disagree"
        geom = score(case["truth"], detections)
        bydet = {d["id"]: d for d in detections}
        matched = {t: bydet[d] for d, t in geom["matching_iou50"].items()}
        result = {
            "name": name,
            "seeded_reasoning_probe": probe,
            "elapsed_s": time.monotonic() - started,
            "contract_failed": outcome.contract_failed,
            "request_errors": [e["error"] for e in events if e["phase"] == "request_error"],
            "attempts": outcome.attempts,
            "candidates": len(candidates),
            "truth": len(case["truth"]),
            "localized": len(matched),
            "correct_family": sum(
                t["id"] in matched and correct_family(t, matched[t["id"]]) for t in case["truth"]
            ),
            "detections": detections,
            "reviews": outcome.batch.candidate_reviews,
            "rejected_objects": outcome.batch.rejected_objects,
        }
        atomic_write_json(args.output / f"{name}.json", result)
        results.append(result)
        atomic_write_json(
            args.output / "summary.json", {"requests_sent": len(sent), "results": results}
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in {"detections", "reviews", "rejected_objects"}
                }
            ),
            flush=True,
        )
        if outcome.contract_failed:
            raise RuntimeError(
                "Canary failed the candidate contract; stopping before more paid calls"
            )
        if result["request_errors"]:
            raise RuntimeError("Canary request failed; stopping before further paid calls")

    try:
        for case_path in args.cases:
            if len(sent) >= args.max_requests:
                break
            evaluate(case_path)
        if args.reasoning_probe and len(sent) < args.max_requests:
            evaluate(args.cases[0], probe=True)
    finally:
        client._client.close()
        atomic_write_json(
            args.output / "summary.json", {"requests_sent": len(sent), "results": results}
        )


if __name__ == "__main__":
    main()
