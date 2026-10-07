"""Run a bounded complete legend/symbol extraction with production configuration.

Creates ordinary extraction artifacts and a separate dispatch audit. Requires
--run and an exclusive audit directory; never automatically resumes this script.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from diagex.config import Config
from diagex.extractors import pid_evidence
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.client import LLMClient
from diagex.vision.perception import _request_diagnostic, _response_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=360)
    args = parser.parse_args()
    if not args.run or not 1 <= args.max_requests <= 400:
        parser.error("Requires --run and 1–400 HTTP dispatches")
    wb = json.loads((args.source_run / "workbench.json").read_text())
    cfg = Config()
    cfg.llm = replace(
        cfg.llm,
        transport=wb["settings"]["provider"],
        model=wb["settings"]["vision_model"],
        vision_model=wb["settings"]["vision_model"],
        reasoning_model=wb["settings"]["reasoning_model"],
        openrouter_base_url=wb["settings"]["base_url"],
        reasoning_mode="enabled",
    )
    cfg.pid = replace(cfg.pid, engine="evidence-v2")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    created = datetime.now(UTC).isoformat()
    with (args.output / "started.json").open("x") as f:
        json.dump(
            {
                "created_at": created,
                "source_run": str(args.source_run),
                "max_requests": args.max_requests,
            },
            f,
        )
    dispatches = 0

    def before_dispatch(request):
        nonlocal dispatches
        if dispatches >= args.max_requests or time.monotonic() - started > 3600:
            raise RuntimeError("Full validation request/time budget exhausted")
        dispatches += 1
        atomic_write_json(
            args.output / f"dispatch-{dispatches:03d}.json",
            {
                "number": dispatches,
                "request_bytes": len(request.content),
                "started_at": time.time(),
                "request": _request_diagnostic(json.loads(request.content)),
            },
        )

    class TracedClient(LLMClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._client._client.event_hooks["request"].append(before_dispatch)

        def messages_create(self, **kwargs):
            response = super().messages_create(**kwargs)
            atomic_write_json(
                args.output / f"response-{dispatches:03d}.json", _response_diagnostic(response)
            )
            return response

    with (args.output / "progress.log").open("w") as log:
        with patch.object(pid_evidence, "LLMClient", TracedClient):
            result = pid_evidence.run_pid_evidence_extract(
                diagram=Path(wb["source_path"]),
                symbol_standard="isa-5.1",
                legend_path=None,
                legend_pages=None,
                legend_region=None,
                no_legend=False,
                legend_key=None,
                effort=wb["settings"]["effort"],
                config=cfg,
                persist=True,
                fresh=False,
                out_path=None,
                confidence_report_path=None,
                console=Console(file=log, force_terminal=False, color_system=None, width=160),
                stop_after="detection",
            )
    finished = datetime.now(UTC).isoformat()
    if result.run_dir:
        atomic_write_json(
            result.run_dir / "workbench.json",
            {
                **wb,
                "created_at": created,
                "finished_at": finished,
                "source_linked_at": finished,
                "job_id": "validation-" + args.output.name,
                "settings": {
                    **wb["settings"],
                    "fresh": False,
                    "stop_after": "detection",
                    "reasoning_mode": "enabled",
                },
            },
        )
    summary = {
        "run_dir": str(result.run_dir),
        "requests_sent": dispatches,
        "elapsed_s": time.monotonic() - started,
        "finished_at": finished,
    }
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
