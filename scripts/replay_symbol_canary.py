"""Replay saved production responses through current symbol postprocessing offline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from evaluate_symbol_perception import score
from PIL import Image
from score_symbol_reasoning import correct_family

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, Tile
from diagex.vision.perception import perceive_tile
from diagex.vision.symbol_candidates import SymbolCandidate
from diagex.vision.views import ViewInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads((args.source_run / "detection.json").read_text())
    results = []
    for path in args.cases:
        case = json.loads(path.read_text())
        events = json.loads((args.responses / f"{case['tile_id']}.events.json").read_text())
        responses = [e["response"] for e in events if e["phase"] == "response"]

        class Replay:
            calls = 0

            def messages_create(self, responses=responses, **kwargs):
                response = responses[self.calls]
                self.calls += 1
                return SimpleNamespace(
                    content=response["content"], usage=None, stop_reason=response.get("stop_reason")
                )

        replay = Replay()
        page = PageEvidence.model_validate_json(
            (args.source_run / "evidence" / f"page-{case['page_index'] + 1:04d}.json").read_text()
        )
        info = ViewInfo(
            **{**case["view_info"], "page_bbox": BBox(**case["view_info"]["page_bbox"])}
        )
        outcome = perceive_tile(
            client=replay,
            cost_tracker=CostTracker(),
            reporter=NullReporter(),
            page=page,
            tile=Tile.model_validate(case["tile"]),
            view_image=Image.open(path.with_suffix(".png")).convert("RGB"),
            view_info=info,
            ownership_bbox=BBox(**case["core"]),
            legend_summary=bundle["legend_pack"]["entries"],
            step=1,
            candidates=[SymbolCandidate.model_validate(c) for c in case["candidates"]],
            reasoning_mode="enabled",
        )
        assert replay.calls == len(responses), "Current routing differs from the saved experiment"
        detections = [d.model_dump(mode="json") for d in outcome.detections]
        counts = Counter(d["attributes"].get("symbol_candidate_id") for d in detections)
        assert all(n == 1 for cid, n in counts.items() if cid), (
            "Duplicate native candidate detection"
        )
        selected = {
            r["candidate_id"] for r in outcome.batch.candidate_reviews if r["status"] == "selected"
        }
        assert set(counts) == selected, "Published detections and review states disagree"
        geometry = score(case["truth"], detections)
        by_id = {d["id"]: d for d in detections}
        matched = {tid: by_id[did] for did, tid in geometry["matching_iou50"].items()}
        result = {
            "tile_id": case["tile_id"],
            "truth": len(case["truth"]),
            "localized": len(matched),
            "correct_family": sum(
                t["id"] in matched and correct_family(t, matched[t["id"]]) for t in case["truth"]
            ),
            "detections": detections,
            "reviews": outcome.batch.candidate_reviews,
            "contract_failed": outcome.contract_failed,
        }
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"detections", "reviews"}}))
    atomic_write_json(
        args.output, {"mode": "offline replay of saved live responses", "results": results}
    )


if __name__ == "__main__":
    main()
