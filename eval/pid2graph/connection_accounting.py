"""Record proof that the final attempt never established an API connection.

Never infer zero cost from an error message, HTTP status, missing generation
record, or interrupted response. This applies prospectively to typed connection
setup errors only, with redirects and hidden SDK retries disabled.
"""
from __future__ import annotations

import time

import anthropic
import httpx


def configure(client):
    sdk = client._client
    if sdk.max_retries != 0 or not isinstance(sdk._client, httpx.Client):
        raise ValueError("Evaluation requires an explicit SDK retry policy and HTTPX client")
    sdk._client.follow_redirects = False
    if not hasattr(client, "_evaluation_owned_request_ids"):
        client._evaluation_owned_request_ids = []
        reserve = client.spending.reserve

        def tracked_reserve(model, max_tokens):
            identity = reserve(model, max_tokens)
            client._evaluation_owned_request_ids.append(identity)
            return identity

        client.spending.reserve = tracked_reserve


def record_unsent_final_attempt(client, exc, previous_requests):
    sdk = client._client
    if sdk.max_retries != 0 or sdk._client.follow_redirects:
        return None
    if not isinstance(exc, anthropic.APIConnectionError):
        return None
    cause = exc.__cause__
    if not isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout)):
        return None
    owned = [i for i in getattr(client, "_evaluation_owned_request_ids", []) if i not in previous_requests]
    if not owned:
        return None
    expected = str(sdk.base_url).rstrip("/") + "/v1/messages"
    # Reject a different endpoint, including an error following a redirect.
    if str(exc.request.url) != expected or exc.request.method != "POST":
        return None
    evidence = {
        "exception_type": type(exc).__name__, "cause_type": type(cause).__name__,
        "endpoint": expected, "sdk_max_retries": 0, "follow_redirects": False,
        "interpretation": "HTTPX failed to establish the connection for this final attempt; no inference request body was sent.",
        "reference": "https://www.python-httpx.org/exceptions/",
        "recorded_at": time.time(),
    }
    with client.spending._locked() as state:
        rows = [r for r in state["requests"] if r["id"] == owned[-1]
                and r["category"] == client.config.spending_category
                and r["model"] == client.config.model]
        if not rows:
            return None
        # Exact per-client reservation identity excludes concurrent runners.
        # Earlier retries may have reached the provider; keep them reserved.
        last = rows[0]
        version2 = state.get("schema_version") == 2
        if last["status"] not in ({"pending", "unresolved"} if version2 else {"reserved"}):
            return None
        evidence["request_id"] = last["id"]
        last.update(status="not_dispatched", charged_usd=0.0,
                    non_dispatch_evidence=evidence,
                    elapsed_s=time.time() - last["started_at"])
        if version2:
            last.update(actual_billed_usd=0.0, exposure_usd=0.0,
                        billing_source="Verified connection setup failure; request not dispatched")
    return evidence
