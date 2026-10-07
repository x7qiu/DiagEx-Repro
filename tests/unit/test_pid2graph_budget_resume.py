import json
from types import SimpleNamespace

import pytest
from PIL import Image

from diagex.llm.budget import BudgetExceeded
from eval.pid2graph import runner
from eval.pid2graph.data import digest, save_new, sha256


@pytest.mark.parametrize("prior_attempt", [False, True])
def test_budget_stop_preserves_attempted_tile_but_leaves_untouched_tile_resumable(
    tmp_path, monkeypatch, prior_attempt,
):
    image = tmp_path / "drawing.png"
    Image.new("RGB", (128, 128), "white").save(image)
    row = {"id": "drawing", "image": str(image), "image_sha256": sha256(image), "size": [128, 128]}
    panel = {"panels": {"validation": [row]}}
    panel["panel_sha256"] = digest(panel)
    save_new(tmp_path / "panel.json", panel)
    save_new(tmp_path / "prices.json", {"models": {"m": {
        "input_per_token": 0.001, "output_per_token": 0.001, "provider_tags": [],
    }}})
    ledger = tmp_path / "ledger.json"
    save_new(ledger, {"requests": []})
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setattr(runner, "configure_connection_accounting", lambda client: None)

    class Client:
        def __init__(self, config):
            self._evaluation_owned_request_ids = []
            self._rate_limit_not_before = 0
            self._rate_limit_cooldown_s = 0
            self._rate_limit_successes = 0
            self._client = SimpleNamespace(max_retries=0, _client=SimpleNamespace(follow_redirects=False))

    monkeypatch.setattr(runner, "LLMClient", Client)
    calls = []

    def fail(**kwargs):
        calls.append(1)
        if prior_attempt:
            state = json.loads(ledger.read_text())
            state["requests"].append({"id": "attempt", "category": "baseline", "model": "m", "charged_usd": 0.3})
            ledger.write_text(json.dumps(state))
            kwargs["client"]._evaluation_owned_request_ids.append("attempt")
        raise BudgetExceeded("Spending cap would be exceeded; request not dispatched")

    monkeypatch.setattr(runner, "perceive_tile", fail)
    args = (tmp_path / "panel.json", tmp_path / "run", ledger, tmp_path / "prices.json")
    with pytest.raises(BudgetExceeded):
        runner.run(*args, model="m")
    case = tmp_path / "run" / digest("drawing")[:16]
    tiles = list(case.glob("*.json"))
    assert len(tiles) == int(prior_attempt)
    if prior_attempt:
        checkpoint = tiles[0].read_bytes()
        result = json.loads(checkpoint)
        assert result["status"] == "failed" and result["predictions"] == []
        assert result["ledger_request_ids"] == ["attempt"]
        assert result["reserved_or_charged_usd"] == 0.3
        assert result["price_snapshot_sha256"] == sha256(tmp_path / "prices.json")
        assert (tmp_path / "run/price-snapshots" / (result["price_snapshot_sha256"] + ".json")).read_bytes() == (tmp_path / "prices.json").read_bytes()
        runner.run(*args, model="m")
        assert calls == [1]  # Resume must not attempt this paid tile again.
        assert tiles[0].read_bytes() == checkpoint
        summary = json.loads((case / "predictions.json").read_text())
        assert summary["completed_tiles"] == 0 and len(summary["errors"]) == 1
        assert summary["reserved_or_charged_usd"] == 0.3
    else:
        with pytest.raises(BudgetExceeded):
            runner.run(*args, model="m")
        assert calls == [1, 1]  # No previous attempt: the tile is eligible to resume.
