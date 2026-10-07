"""Bounded production-client validation of explicitly selected native legend rows.

Only row discovery is restricted to the declared sample. The production source
row renderer, prompt, parser, client, streaming transport and routing are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image

from diagex.config import Config
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.extractors.legend_rows import classify_native_legend
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.vision.evidence import PageEvidence
from diagex.vision.legend_models import LegendPack
from diagex.vision.legend_rows import native_legend_rows
from diagex.vision.models import DiagramPage
from diagex.vision.perception import _request_diagnostic, _response_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--row-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=4)
    parser.add_argument("--prior-pack", type=Path)
    args = parser.parse_args()
    if not args.run or not 1 <= args.max_requests <= 8 or len(args.row_ids) > 24:
        parser.error("Requires --run, at most 24 source rows and 1–8 HTTP dispatches")
    wb = json.loads((args.source_run / "workbench.json").read_text())
    source = Path(wb["source_path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == wb["source_sha256"]
    cfg = Config()
    cfg.llm = replace(
        cfg.llm,
        transport=wb["settings"]["provider"],
        model=wb["settings"]["vision_model"],
        openrouter_base_url=wb["settings"]["base_url"],
        reasoning_mode="disabled",
    )
    client = LLMClient(cfg.llm, budgets=cfg.budgets)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "started.json").open("x") as f:
        json.dump(
            {
                "row_ids": args.row_ids,
                "source_sha256": wb["source_sha256"],
                "request_limit": args.max_requests,
                "started_at": time.time(),
            },
            f,
        )
    dispatches, responses, entries, coverage = [], [], [], []

    def before_dispatch(request):
        if len(dispatches) >= args.max_requests:
            raise RuntimeError("Legend canary HTTP request budget exhausted")
        dispatches.append(
            {
                "number": len(dispatches) + 1,
                "started_at": time.time(),
                "request": _request_diagnostic(json.loads(request.content)),
            }
        )
        atomic_write_json(args.output / "dispatches.json", dispatches)

    client._client._client.event_hooks["request"].append(before_dispatch)

    class RecordingClient:
        def messages_create(self, **kwargs):
            response = client.messages_create(**kwargs)
            responses.append(_response_diagnostic(response))
            atomic_write_json(args.output / "responses.json", responses)
            return response

    requested = set(args.row_ids)
    found = set()
    tracker = CostTracker()
    prior = LegendPack.model_validate_json(args.prior_pack.read_text()) if args.prior_pack else None
    with fitz.open(source) as pdf:
        for path in sorted((args.source_run / "evidence").glob("page-*.json")):
            evidence = PageEvidence.model_validate_json(path.read_text())
            if evidence.role != "legend":
                continue
            rows = [r for r in native_legend_rows(evidence) if r.id in requested]
            if not rows:
                continue
            found.update(r.id for r in rows)
            pix = pdf[evidence.page_index].get_pixmap(
                matrix=fitz.Matrix(evidence.dpi / 72, evidence.dpi / 72), alpha=False
            )
            rendered = DiagramPage(
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
                image=Image.frombytes("RGB", (pix.width, pix.height), pix.samples),
            )
            with patch("diagex.extractors.legend_rows.native_legend_rows", return_value=rows):
                entries.extend(
                    classify_native_legend(
                        page=rendered,
                        evidence=evidence,
                        region=None,
                        client=RecordingClient(),
                        cost_tracker=tracker,
                        cfg=cfg,
                        coverage=coverage,
                        prior=prior,
                    )
                )
            atomic_write_json(
                args.output / "entries.json", [e.model_dump(mode="json") for e in entries]
            )
            atomic_write_json(
                args.output / "coverage.json", [c.model_dump(mode="json") for c in coverage]
            )
            if any(c.status != "complete" for c in coverage):
                raise RuntimeError(
                    "Legend canary has an unresolved row; stopping before more calls"
                )
    assert found == requested, f"Unknown source row IDs: {requested - found}"
    atomic_write_json(
        args.output / "summary.json",
        {
            "requests_sent": len(dispatches),
            "rows": len(entries),
            "responses": len(responses),
            "all_coverage_complete": all(c.status == "complete" for c in coverage),
        },
    )
    print(json.dumps({"requests_sent": len(dispatches), "rows": len(entries)}))


if __name__ == "__main__":
    main()
