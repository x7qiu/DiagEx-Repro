"""Durable, shared pre-dispatch spending reservations.

A failed/abandoned request retains its entire reservation: a broken stream is
not evidence that the provider did no billable work. Prices are a caller-supplied
verified ceiling, never the legacy global pricing defaults.
"""
from __future__ import annotations

import fcntl
import json
import math
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json


class BudgetExceeded(RuntimeError):
    pass


class SpendingLedger:
    def __init__(self, path, prices, category):
        self.path = Path(path)
        self.prices = json.loads(Path(prices).read_text())["models"]
        self.category = category
        if not self.path.exists():
            raise ValueError("Initialize the spending ledger with an explicit limit before inference")

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = json.loads(self.path.read_text())
                yield state
                atomic_write_json(self.path, state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, model, max_tokens):
        price = self.prices.get(model)
        if price is None:
            raise BudgetExceeded(f"No verified price ceiling for {model}")
        if time.time() > float(price["valid_until"]):
            raise BudgetExceeded("Price verification expired; refresh before more inference")
        values = [price[k] for k in ("context_length", "input_per_token", "output_per_token")]
        values.append(price.get("request_usd", 0))
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in values) or values[0] <= 0:
            raise BudgetExceeded("Invalid model price ceiling")
        if max_tokens < 1 or max_tokens > price["context_length"]:
            raise BudgetExceeded("Requested output exceeds verified model context")
        # Full context bound includes images, tools and reasoning. No tokenizer
        # or image-size heuristic can under-reserve the submitted input.
        amount = price["context_length"] * price["input_per_token"] + max_tokens * price["output_per_token"] + price.get("request_usd", 0)
        with self._locked() as state:
            used = sum(r["charged_usd"] for r in state["requests"])
            category_used = sum(r["charged_usd"] for r in state["requests"] if r["category"] == self.category)
            if self.category not in state["category_limits"]:
                raise BudgetExceeded(f"Unknown spending category {self.category}")
            if used + amount > state["limit_usd"] or category_used + amount > state["category_limits"][self.category]:
                raise BudgetExceeded(f"Spending cap would be exceeded by {model}; request not dispatched")
            request_id = uuid.uuid4().hex
            state["requests"].append({"id": request_id, "model": model, "category": self.category,
                                      "reserved_usd": amount, "charged_usd": amount,
                                      "status": "reserved", "started_at": time.time()})
        return request_id

    def settle(self, request_id, response):
        usage = getattr(response, "usage", None)
        with self._locked() as state:
            row = next(r for r in state["requests"] if r["id"] == request_id)
            row["elapsed_s"] = time.time() - row["started_at"]
            if usage is None or any(getattr(usage, k, None) is None for k in ("input_tokens", "output_tokens")):
                row["status"] = "usage_unavailable"
                return
            price = self.prices[row["model"]]
            input_tokens = sum(int(getattr(usage, key, 0) or 0) for key in
                               ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            ceiling = input_tokens * price["input_per_token"] + output_tokens * price["output_per_token"] + price.get("request_usd", 0)
            if min(input_tokens, output_tokens) < 0 or ceiling > row["reserved_usd"] + 1e-8:
                row["status"] = "price_bound_violation"
                # Fail closed for subsequent calls; preserve the reported upper
                # cost instead of disguising a broken price contract.
                row["charged_usd"] = max(ceiling, row["reserved_usd"])
                state["limit_usd"] = 0
            else:
                row.update(status="complete", charged_usd=ceiling,
                           input_tokens=input_tokens, output_tokens=output_tokens)
