import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from diagex.llm.budget import SpendingLedger
from eval.pid2graph.connection_accounting import configure, record_unsent_final_attempt
from eval.pid2graph.data import save_new


def fixture(tmp_path):
    ledger_path, price_path = tmp_path / "spending.json", tmp_path / "prices.json"
    rows = [{"id": identity, "category": "baseline", "model": "m", "reserved_usd": 1,
             "charged_usd": 1, "status": "reserved", "started_at": index}
            for index, identity in enumerate(["historical", "earlier_attempt", "last_attempt"])]
    save_new(ledger_path, {"limit_usd": 60, "category_limits": {"baseline": 25}, "requests": rows})
    save_new(price_path, {"models": {}})
    client = SimpleNamespace(config=SimpleNamespace(spending_category="baseline", model="m"),
        _evaluation_owned_request_ids=["historical", "earlier_attempt", "last_attempt"],
        spending=SpendingLedger(ledger_path, price_path, "baseline"),
        _client=SimpleNamespace(max_retries=0, base_url="https://openrouter.ai/api/",
                               _client=SimpleNamespace(follow_redirects=False)))
    return client, ledger_path


def error(cause_type, url="https://openrouter.ai/api/v1/messages"):
    request = httpx.Request("POST", url)
    result = anthropic.APIConnectionError(request=request)
    result.__cause__ = cause_type("connection test", request=request)
    return result


@pytest.mark.parametrize("cause", [httpx.ConnectError, httpx.ConnectTimeout])
def test_only_proven_unsent_last_attempt_is_zeroed(tmp_path, cause):
    client, path = fixture(tmp_path)
    proof = record_unsent_final_attempt(client, error(cause), {"historical"})
    state = json.loads(path.read_text())
    assert [r["charged_usd"] for r in state["requests"]] == [1, 1, 0]
    assert state["limit_usd"] == 60 and state["category_limits"] == {"baseline": 25}
    assert state["requests"][-1]["reserved_usd"] == 1
    assert state["requests"][-1]["non_dispatch_evidence"] == proof
    assert record_unsent_final_attempt(client, error(cause), {"historical"}) is None


@pytest.mark.parametrize("cause", [httpx.ReadError, httpx.ReadTimeout, httpx.WriteError,
                                   httpx.RemoteProtocolError, httpx.ProxyError])
def test_uncertain_failures_keep_full_reservations(tmp_path, cause):
    client, path = fixture(tmp_path)
    before = path.read_bytes()
    assert record_unsent_final_attempt(client, error(cause), {"historical"}) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("unsafe", ["redirects", "sdk_retries", "other_endpoint", "untyped"])
def test_ambiguous_dispatch_history_cannot_release_a_reservation(tmp_path, unsafe):
    client, path = fixture(tmp_path)
    exc = error(httpx.ConnectError)
    if unsafe == "redirects":
        client._client._client.follow_redirects = True
    elif unsafe == "sdk_retries":
        client._client.max_retries = 1
    elif unsafe == "other_endpoint":
        exc = error(httpx.ConnectError, "https://example.com/v1/messages")
    else:
        exc = RuntimeError("ConnectError")
    before = path.read_bytes()
    assert record_unsent_final_attempt(client, exc, {"historical"}) is None
    assert path.read_bytes() == before


def test_concurrent_same_category_model_request_is_never_released(tmp_path):
    client, path = fixture(tmp_path)
    state = json.loads(path.read_text())
    state["requests"].append({**state["requests"][-1], "id": "other-client", "started_at": 999})
    path.write_text(json.dumps(state))
    proof = record_unsent_final_attempt(client, error(httpx.ConnectError), {"historical"})
    assert proof["request_id"] == "last_attempt"
    assert json.loads(path.read_text())["requests"][-1]["charged_usd"] == 1


def test_reservation_tracking_survives_transport_reconfiguration():
    with httpx.Client(follow_redirects=True) as transport:
        client = SimpleNamespace(_client=SimpleNamespace(max_retries=0, _client=transport),
                                 spending=SimpleNamespace(reserve=lambda model, max_tokens: "own-id"))
        configure(client)
        configure(client)
        assert client.spending.reserve("m", 10) == "own-id"
        assert client._evaluation_owned_request_ids == ["own-id"]
        assert transport.follow_redirects is False
