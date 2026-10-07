"""Snapshot billing for PDF/legend integration checks without treating exposure as cost."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from diagex.llm.billing import summarize


def transfer_billing(ledger, request_ids):
    raw = Path(ledger).read_bytes()
    state = json.loads(raw)
    identities = list(request_ids)
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate transfer request IDs")
    lookup = {r["id"]: r for r in state["requests"]}
    if len(lookup) != len(state["requests"]) or not set(identities) <= lookup.keys():
        raise ValueError("Transfer request IDs do not match the ledger")
    rows = [lookup[i] for i in identities]
    accounting = {"ledger_sha256": hashlib.sha256(raw).hexdigest(),
                  "request_ids": identities,
                  "scope": "Snapshot of these requests only; later reconciliation may update unknown charges."}
    if state.get("schema_version") != 2:
        return {"accounting": {**accounting, "method": "Legacy exposure only; actual billing unavailable"},
                "reserved_or_charged_usd": sum(r["charged_usd"] for r in rows)}
    billing = summarize(state, identities)
    billing.update(active_request_count=sum(r["status"] == "pending" for r in rows),
                   finished_unresolved_request_count=sum(r["status"] == "unresolved" for r in rows))
    return {"accounting": {**accounting, "method": "Actual bills, active reservations and unresolved upper bounds are separate"},
            "billing": billing, "reserved_or_charged_usd": billing["budget_exposure_usd"]}
