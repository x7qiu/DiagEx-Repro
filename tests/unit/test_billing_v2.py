import json
import time
from types import SimpleNamespace

import httpx
import pytest

from diagex.config import LLMConfig
from diagex.llm.billing import AccountedSpendingLedger, summarize
from diagex.llm.budget import BudgetExceeded
from diagex.llm.client import LLMClient
from eval.pid2graph.billing_reconcile import reconcile


@pytest.fixture
def billing(tmp_path):
    path, prices = tmp_path / "ledger.json", tmp_path / "prices.json"
    path.write_text(json.dumps({"schema_version": 2, "limit_usd": 1.2,
                               "prior_spend": {"usd": 0.64}, "requests": []}))
    prices.write_text(json.dumps({"models": {"test": {"context_length": 100,
                      "input_per_token": 0.004, "output_per_token": 0.001,
                      "valid_until": time.time() + 60, "provider_tags": ["test"]}}}))
    return AccountedSpendingLedger(path, prices, "baseline"), prices


def read(ledger):
    return json.loads(ledger.path.read_text())


def test_actual_pending_and_unresolved_are_distinct_and_all_enforce_cap(billing):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    s = summarize(read(ledger))
    assert s["actual_billed_usd"] == 0 and s["active_reservations_usd"] == 0.5
    assert not s["actual_billing_complete"]
    with pytest.raises(BudgetExceeded):
        ledger.reserve("test", 100)
    ledger.finish_attempt(identity)
    s = summarize(read(ledger))
    assert s["actual_billed_usd"] == s["active_reservations_usd"] == 0
    assert s["unresolved_upper_usd"] == 0.5
    with pytest.raises(BudgetExceeded):
        ledger.reserve("test", 100)
    ledger.settle(identity, {"id": "gen-one", "usage": {"cost": 0.01}})
    assert summarize(read(ledger))["actual_billed_usd"] == 0.01
    assert ledger.reserve("test", 100)


def test_token_arithmetic_is_not_reported_as_actual_billing(billing):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.settle(identity, SimpleNamespace(id="gen-one", usage=SimpleNamespace(input_tokens=2, output_tokens=2)))
    row = read(ledger)["requests"][0]
    assert row["actual_billed_usd"] is None and row["exposure_usd"] == 0.01
    ledger.reconcile_generation(identity, {"id": "gen-one", "model": "test", "total_cost": 0.003})
    assert summarize(read(ledger))["actual_billed_usd"] == 0.003
    assert summarize(read(ledger))["actual_billing_complete"]


def test_partial_stream_zero_is_not_a_final_bill(billing):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.observe_stream_event(identity, {"type": "message_start", "message": {"id": "gen-partial", "usage": {"cost": 0}}})
    ledger.finish_attempt(identity)
    assert read(ledger)["requests"][0]["actual_billed_usd"] is None
    ledger.reconcile_generation(identity, {"id": "gen-partial", "model": "test", "total_cost": 0})
    assert summarize(read(ledger))["budget_exposure_usd"] == 0


@pytest.mark.parametrize("patch", [{"id": "gen-other"}, {"model": "other"}, {"total_cost": -1}, {"total_cost": float("nan")}])
def test_bad_reconciliation_cannot_release_exposure(billing, patch):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.observe_stream_event(identity, {"type": "message_start", "message": {"id": "gen-one"}})
    ledger.finish_attempt(identity)
    with pytest.raises(ValueError):
        ledger.reconcile_generation(identity, {"id": "gen-one", "model": "test", "total_cost": 0, **patch})
    assert summarize(read(ledger))["budget_exposure_usd"] == 0.5


@pytest.mark.parametrize("change", [{}, {"model": "test-wrong-date"}, {"provider_name": "other"}, {"id": "gen-other"}])
def test_dated_model_requires_exact_reservation_endpoint_evidence(billing, change):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.observe_stream_event(identity, {"type": "message_start", "message": {"id": "gen-one"}})
    ledger.finish_attempt(identity)
    with ledger._locked() as state:
        row = state["requests"][0]
        row["price"]["verified_at"] = row["started_at"] - 1
        row["price"]["endpoint_snapshots"] = [{"model_id": "test", "provider_name": "Provider",
                                                "tag": "test", "name": "Provider | test-20260101"}]
    metadata = {"id": "gen-one", "model": "test-20260101", "provider_name": "Provider", "total_cost": 0.003, **change}
    if change:
        with pytest.raises(ValueError, match="identity/model"):
            ledger.reconcile_generation(identity, metadata)
        assert summarize(read(ledger))["unresolved_upper_usd"] == 0.5
    else:
        ledger.reconcile_generation(identity, metadata)
        row = read(ledger)["requests"][0]
        assert row["actual_billed_usd"] == 0.003
        assert row["generation_metadata"]["model"] == "test-20260101"
        assert row["generation_model_identity"]["match"] == "reservation_endpoint_snapshot"


def test_active_attempt_cannot_be_reconciled_even_with_matching_metadata(billing):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.observe_stream_event(identity, {"type": "message_start", "message": {"id": "gen-one"}})
    with pytest.raises(ValueError, match="active"):
        ledger.reconcile_generation(identity, {"id": "gen-one", "model": "test", "total_cost": 0})
    assert summarize(read(ledger))["active_reservations_usd"] == 0.5


def test_missing_generation_stays_unresolved_after_key_snapshot(billing):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.observe_stream_event(identity, {"type": "message_start", "message": {"id": "gen-missing"}})
    ledger.finish_attempt(identity)

    def handle(request):
        if request.url.path.endswith("/generation"):
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"data": {"usage": 10, "usage_daily": 0}})

    with httpx.Client(base_url="https://test.invalid/api/v1", transport=httpx.MockTransport(handle)) as client:
        result = reconcile(ledger, client)
    assert result["unresolved_upper_usd"] == 0.5
    assert not result["actual_billing_complete"]
    assert read(ledger)["requests"][0]["lookup_history"][-1]["status"] == 404


def test_reconciliation_waits_for_active_stream_to_finish(billing):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.observe_stream_event(identity, {"type": "message_start", "message": {"id": "gen-active"}})
    lookups = []

    def handle(request):
        lookups.append(request.url.path)
        if request.url.path.endswith("/generation"):
            return httpx.Response(200, json={"data": {"id": "gen-active", "model": "test", "total_cost": 0.003}})
        return httpx.Response(200, json={"data": {"usage": 10}})

    with httpx.Client(base_url="https://test.invalid/api/v1", transport=httpx.MockTransport(handle)) as client:
        active = reconcile(ledger, client)
        assert active["active_reservations_usd"] == 0.5 and active["actual_billed_usd"] == 0
        assert lookups == ["/api/v1/key"]
        ledger.finish_attempt(identity)
        finished = reconcile(ledger, client)
    assert finished["actual_billing_complete"] and finished["actual_billed_usd"] == 0.003
    assert finished["active_reservations_usd"] == finished["unresolved_upper_usd"] == 0


@pytest.mark.parametrize("broken", [False, True])
def test_real_client_tracks_stream_ids_without_callbacks_or_deadlines(billing, monkeypatch, broken):
    ledger, prices = billing
    config = LLMConfig(transport="openrouter", model="test", openrouter_api_key="test",
                       spending_ledger=str(ledger.path), verified_prices=str(prices))
    client = LLMClient(config)

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield {"type": "message_start", "message": {"id": "gen-stream", "usage": {"cost": 0}}}
            if broken:
                raise httpx.ReadError("interrupted")
            yield {"type": "message_delta", "usage": {"cost": 0.007}}

        def get_final_message(self):
            return SimpleNamespace(id="gen-stream", usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    monkeypatch.setattr(client._client.messages, "stream", lambda **kwargs: Stream())
    if broken:
        with pytest.raises(httpx.ReadError):
            client.messages_create(system="test", messages=[], max_tokens=100, max_attempts=1)
    else:
        client.messages_create(system="test", messages=[], max_tokens=100, max_attempts=1)
    row = read(ledger)["requests"][0]
    assert row["generation_ids"] == ["gen-stream"]
    assert row["status"] == ("unresolved" if broken else "billed")
    assert row["actual_billed_usd"] == (None if broken else 0.007)
    client._client.close()


def test_unexpected_charge_halts_new_dispatch(billing):
    ledger, _ = billing
    identity = ledger.reserve("test", 100)
    ledger.settle(identity, {"usage": {"cost": 0.6}})
    with pytest.raises(BudgetExceeded, match="reconciliation"):
        ledger.reserve("test", 10)
