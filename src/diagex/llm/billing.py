"""Version-2 billing: distinguish actual cost from conservative budget exposure.

The legacy ledger remains readable and unchanged. Unresolved work is never
declared free. Generation IDs permit later authoritative reconciliation.
"""
from __future__ import annotations

import math
import time
import uuid

from .budget import BudgetExceeded, SpendingLedger


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _money(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def summarize(state, request_ids=None):
    rows = state["requests"]
    if request_ids is not None:
        allowed = set(request_ids)
        rows = [r for r in rows if r["id"] in allowed]
    return {
        "actual_billed_usd": sum(r["actual_billed_usd"] or 0 for r in rows),
        "billed_request_count": sum(r["actual_billed_usd"] is not None for r in rows),
        "active_reservations_usd": sum(r["exposure_usd"] for r in rows if r["status"] == "pending"),
        "unresolved_upper_usd": sum(r["exposure_usd"] for r in rows if r["status"] == "unresolved"),
        "unresolved_request_count": sum(r["actual_billed_usd"] is None for r in rows),
        "budget_exposure_usd": sum(r["exposure_usd"] for r in rows),
        "actual_billing_complete": all(r["actual_billed_usd"] is not None for r in rows),
    }


class AccountedSpendingLedger(SpendingLedger):
    """A fresh schema-version-2 ledger, never an implicit legacy migration."""

    schema_version = 2

    def reserve(self, model, max_tokens):
        price = self.prices.get(model)
        if price is None or time.time() > float(price["valid_until"]):
            raise BudgetExceeded("Verified model pricing unavailable or expired")
        values = [price[k] for k in ("context_length", "input_per_token", "output_per_token")]
        values.append(price.get("request_usd", 0))
        if any(_money(v) is None for v in values) or not 1 <= max_tokens <= values[0]:
            raise BudgetExceeded("Invalid price ceiling or token limit")
        # A maximum exposure is still needed before dispatch. It is temporary
        # risk accounting, never a bill or an estimate of actual token use.
        amount = values[0] * values[1] + max_tokens * values[2] + values[3]
        with self._locked() as state:
            if state.get("schema_version") != 2:
                raise ValueError("Version-2 billing requires a new version-2 ledger")
            prior = state["prior_spend"]["usd"]
            if _money(prior) is None or state.get("halted"):
                raise BudgetExceeded("Billing state requires reconciliation")
            exposure = summarize(state)["budget_exposure_usd"]
            if prior + exposure + amount > state["limit_usd"]:
                raise BudgetExceeded("Budget exposure exceeds cap; reconcile unresolved billing before more calls")
            identity = uuid.uuid4().hex
            state["requests"].append({
                "id": identity, "model": model, "category": self.category,
                "started_at": time.time(), "status": "pending", "price": price,
                "reserved_usd": amount, "exposure_usd": amount,
                "actual_billed_usd": None, "billing_source": None,
                # Legacy callers use this only as reserved-or-charged exposure.
                "charged_usd": amount, "generation_ids": [],
            })
        return identity

    def observe_stream_event(self, identity, event):
        message = _get(event, "message")
        generation = _get(message, "id")
        usage = _get(event, "usage") or _get(message, "usage")
        cost = _money(_get(usage, "cost"))
        if not generation and cost is None:
            return
        with self._locked() as state:
            row = next(r for r in state["requests"] if r["id"] == identity)
            if generation and generation not in row["generation_ids"]:
                row["generation_ids"].append(generation)
            if cost is not None:
                row["stream_reported_cost_usd"] = cost
                row["stream_cost_event"] = _get(event, "type")

    def observe_error(self, identity, error):
        body = _get(error, "body", {})
        ids = set()

        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"id", "request_id", "generation_id"} and isinstance(item, str) and item.startswith("gen-"):
                        ids.add(item)
                    elif isinstance(item, (dict, list)):
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(body)
        with self._locked() as state:
            row = next(r for r in state["requests"] if r["id"] == identity)
            row["generation_ids"] = sorted(set(row["generation_ids"]) | ids)
            row["error_type"] = type(error).__name__
            row["http_status"] = _get(error, "status_code")

    @staticmethod
    def _record_cost(state, row, amount, source):
        if amount > row["reserved_usd"] + 1e-8:
            state["halted"] = "Observed charge exceeds reserved maximum; verify prices before continuing"
        row.update(actual_billed_usd=amount, exposure_usd=amount, charged_usd=amount,
                   status="billed", billing_source=source, reconciled_at=time.time())

    def settle(self, identity, response):
        usage = _get(response, "usage")
        with self._locked() as state:
            row = next(r for r in state["requests"] if r["id"] == identity)
            generation = _get(response, "id")
            if generation and generation not in row["generation_ids"]:
                row["generation_ids"].append(generation)
            row["elapsed_s"] = time.time() - row["started_at"]
            row["response_completed"] = True
            amount = _money(_get(usage, "cost"))
            if amount is None and row.get("stream_cost_event") == "message_delta":
                amount = row.get("stream_reported_cost_usd")
            if amount is not None:
                self._record_cost(state, row, amount, "OpenRouter completed response usage.cost")
                return
            # Token arithmetic bounds unresolved cost; it does not establish
            # what OpenRouter actually billed, including caching discounts.
            fields = ("input_tokens", "output_tokens")
            if usage is not None and all(_get(usage, key) is not None for key in fields):
                tokens = [_get(usage, key, 0) or 0 for key in
                          ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")]
                if all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in tokens):
                    price = row["price"]
                    ceiling = sum(tokens[:3]) * price["input_per_token"] + tokens[3] * price["output_per_token"] + price.get("request_usd", 0)
                    row.update(input_tokens=sum(tokens[:3]), output_tokens=tokens[3], token_price_upper_usd=ceiling)
                    row["exposure_usd"] = row["charged_usd"] = ceiling
                    if ceiling > row["reserved_usd"] + 1e-8:
                        state["halted"] = "Token usage exceeds reserved maximum"
            row["status"] = "unresolved"

    def finish_attempt(self, identity):
        with self._locked() as state:
            row = next(r for r in state["requests"] if r["id"] == identity)
            if row["status"] == "pending":
                row.update(status="unresolved", elapsed_s=time.time() - row["started_at"])

    def reconcile_generation(self, identity, metadata):
        amount = _money(metadata.get("total_cost"))
        if amount is None:
            raise ValueError("Generation metadata lacks a valid total_cost")
        with self._locked() as state:
            row = next(r for r in state["requests"] if r["id"] == identity)
            if row["status"] == "pending":
                raise ValueError("Cannot reconcile an active attempt")
            if len(row["generation_ids"]) != 1:
                raise ValueError("Ambiguous generation identity; keep unresolved exposure")
            if metadata.get("id") not in row["generation_ids"]:
                raise ValueError("Generation identity/model mismatch")
            model_identity = {"requested_model": row["model"], "billed_model": metadata.get("model"), "match": "exact"}
            if metadata.get("model") != row["model"]:
                # OpenRouter can bill an immutable model ID for an alias.
                # Accept only the exact provider/model pair documented by the
                # endpoint snapshot stored with this reservation, never a
                # guessed date suffix or today's possibly changed alias.
                price = row["price"]
                endpoint = next((e for e in price.get("endpoint_snapshots", [])
                                 if e.get("model_id") == row["model"]
                                 and e.get("provider_name") == metadata.get("provider_name")
                                 and e.get("tag") in price["provider_tags"]
                                 and e.get("name") == f"{metadata.get('provider_name')} | {metadata.get('model')}"), None)
                if (not endpoint or not isinstance(metadata.get("model"), str)
                        or not price.get("verified_at", float("inf")) <= row["started_at"] <= price["valid_until"]):
                    raise ValueError("Generation identity/model mismatch")
                model_identity.update(match="reservation_endpoint_snapshot", endpoint={
                    k: endpoint[k] for k in ("model_id", "provider_name", "tag", "name")},
                    snapshot_verified_at=price["verified_at"])
            if any(metadata["id"] in r["generation_ids"] for r in state["requests"] if r["id"] != identity):
                raise ValueError("Generation ID belongs to multiple reservations")
            row["generation_metadata"] = {k: metadata.get(k) for k in
                                          ("id", "model", "total_cost", "created_at", "provider_name", "cancelled", "finish_reason")}
            row["generation_model_identity"] = model_identity
            self._record_cost(state, row, amount, "OpenRouter GET /generation total_cost")
