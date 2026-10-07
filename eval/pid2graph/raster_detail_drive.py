"""Run bounded inference invocations, preserving the frozen panel and cooldowns."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--variant", default="proposal_source_crops")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--proposal-run")
    parser.add_argument("--selection")
    parser.add_argument("--tiles-per-process", type=int, default=4)
    args = parser.parse_args()
    if args.tiles_per_process < 1:
        raise ValueError("Each invocation must process at least one new tile")
    panel = json.loads(Path(args.panel).read_text())
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    keys = [digest(r["id"])[:16] for r in panel["panels"][args.split]]
    command = [sys.executable, "-m", "eval.pid2graph.raster_detail_runner", "--panel", args.panel,
               "--out", args.out, "--ledger", args.ledger, "--prices", args.prices,
               "--split", args.split, "--variant", args.variant, "--model", args.model,
               "--request-timeout", "120", "--max-new-tiles", str(args.tiles_per_process)]
    for key in ("proposal_run", "selection"):
        if getattr(args, key):
            command.extend(["--" + key.replace("_", "-"), getattr(args, key)])
    with (root / "invocations.jsonl").open("a") as log:
        while not all((root / key / "predictions.json").exists() for key in keys):
            before = len(list(root.glob("*/*-r*-c*.json")))
            started = time.time()
            result = subprocess.run(command, check=False)
            after = len(list(root.glob("*/*-r*-c*.json")))
            completed = sum((root / key / "predictions.json").exists() for key in keys)
            log.write(json.dumps({"started_at": started, "elapsed_seconds": time.time() - started,
                                 "returncode": result.returncode, "new_tile_records": after - before,
                                 "drawings_with_summary": completed}) + "\n")
            log.flush()
            if result.returncode:
                raise SystemExit(result.returncode)
            if after == before and completed != len(keys):
                raise RuntimeError("Inference made no progress; inspect existing records")
        print(f"Finished attempted coverage for {len(keys)} drawings; failures remain in scoring.", flush=True)


if __name__ == "__main__":
    main()
