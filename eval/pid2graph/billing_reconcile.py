"""Initialize phase-2 accounting or reconcile it using read-only OpenRouter APIs."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

import diagex.config  # noqa: F401 - Loads the existing environment configuration.
from diagex.llm.billing import AccountedSpendingLedger, summarize

from .data import save_new


def reconcile(ledger, client):
    state = json.loads(ledger.path.read_text())
    for row in state["requests"]:
        # A generation ID can arrive while a stream is still active. Leave
        # its reservation intact until the owning attempt actually finishes.
        if row["status"] != "unresolved" or row["actual_billed_usd"] is not None or len(row["generation_ids"]) != 1:
            continue
        generation = row["generation_ids"][0]
        error, http_status = None, None
        try:
            response = client.get("/generation", params={"id": generation})
            status = http_status = response.status_code
            if status == 200:
                ledger.reconcile_generation(row["id"], response.json()["data"])
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            status = type(exc).__name__
            error = str(exc)
        with ledger._locked() as latest:
            record = next(r for r in latest["requests"] if r["id"] == row["id"])
            record.setdefault("lookup_history", []).append({"at": time.time(), "generation_id": generation,
                                                           "status": status, "http_status": http_status, "error": error})
        # A 404 or transport failure is not proof of zero cost. Nothing above
        # settles a request unless matching metadata contains an explicit cost.
    response = client.get("/key")
    response.raise_for_status()
    data = response.json()["data"]
    with ledger._locked() as state:
        state.setdefault("key_usage_snapshots", []).append({
            "at": time.time(), "scope": "Key-wide; not allocation to individual study calls",
            "data": {k: data.get(k) for k in ("usage", "usage_daily", "usage_weekly", "usage_monthly")},
        })
        return summarize(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "reconcile", "summary"))
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--prices", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        save_new(args.ledger, {"schema_version": 2, "limit_usd": 60,
                 "prior_spend": {"usd": 0.64, "source": "User-reported first study total on 2026-09-19", "request_level_reconciled": False},
                 "authorization": "Original $60 cumulative limit across both phases; no cloud GPU rental",
                 "requests": [], "created_at": time.time()})
    elif args.command == "reconcile":
        if not args.prices:
            parser.error("--prices is required for reconciliation")
        ledger = AccountedSpendingLedger(args.ledger, args.prices, "reconciliation")
        with httpx.Client(base_url="https://openrouter.ai/api/v1", timeout=20, follow_redirects=False,
                          headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}) as client:
            reconcile(ledger, client)
    state = json.loads(args.ledger.read_text())
    print(json.dumps({"prior_spend": state["prior_spend"], "current_phase": summarize(state)}, indent=2))
