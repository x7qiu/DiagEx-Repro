import copy

import pytest

from eval.pid2graph.data import digest, save_new, sha256
from eval.pid2graph.selection import (
    perception_signature,
    price_signature,
    ranking_key,
    read_selection,
    seal,
)


def fixture_report(panel, variant, f1, status="complete"):
    config = {"panel_sha256": panel["panel_sha256"], "split": "validation", "variant": variant, "model": "m"}
    return {"split": "validation", "panel_sha256": panel["panel_sha256"], "manifest_sha256": "manifest",
            "config": config, "config_sha256": digest(config), "cases": [{"id": "drawing", "status": status}],
            "summary": {"macro_drawing_f1": f1}, "reserved_or_charged_usd": 1, "runtime_seconds": 2}


def test_selection_requires_complete_panel_and_includes_failed_tile_denominators(tmp_path):
    panel = {"manifest_sha256": "manifest", "panels": {"validation": [{"id": "drawing"}]}}
    panel["panel_sha256"] = digest(panel)
    save_new(tmp_path / "panel.json", panel)
    base = fixture_report(panel, "baseline", 0.4, status="partial")
    candidate = fixture_report(panel, "supervised_guidance", 0.5, status="missing")
    save_new(tmp_path / "baseline.json", base)
    save_new(tmp_path / "missing.json", candidate)
    with pytest.raises(ValueError, match="unattempted"):
        seal(tmp_path / "panel.json", [tmp_path / "baseline.json", tmp_path / "missing.json"], tmp_path / "bad.json")
    candidate["cases"][0]["status"] = "complete"
    save_new(tmp_path / "candidate.json", candidate)
    result = seal(tmp_path / "panel.json", [tmp_path / "baseline.json", tmp_path / "candidate.json"], tmp_path / "selection.json")
    assert result["variant"] == "supervised_guidance"
    assert read_selection(tmp_path / "selection.json", panel) == result
    result["model"] = "changed"
    save_new(tmp_path / "tampered.json", result)
    with pytest.raises(ValueError, match="seal"):
        read_selection(tmp_path / "tampered.json", panel)
    candidate["stopped"] = True
    save_new(tmp_path / "stopped.json", candidate)
    with pytest.raises(ValueError, match="stopped"):
        seal(tmp_path / "panel.json", [tmp_path / "baseline.json", tmp_path / "stopped.json"], tmp_path / "bad-stopped.json")


def test_final_signature_allows_new_images_but_not_threshold_or_weights():
    validation = {"split": "validation", "model": "m", "guidance": {
        "proposal_files_sha256": {"validation": "x"}, "detector_runtime_seconds": {"validation": 5},
        "detector_config": {"split": "validation", "threshold": 0.15, "detector_sha256": "weights"}}}
    final = copy.deepcopy(validation)
    final["split"] = "test"
    final["guidance"]["proposal_files_sha256"] = {"test": "y"}
    final["guidance"]["detector_runtime_seconds"] = {"test": 9}
    final["guidance"]["detector_config"]["split"] = "test"
    assert perception_signature(final) == perception_signature(validation)
    final["guidance"]["detector_config"]["threshold"] = 0.1
    assert perception_signature(final) != perception_signature(validation)
    final["guidance"]["detector_config"].update(threshold=0.15, detector_sha256="other weights")
    assert perception_signature(final) != perception_signature(validation)


def test_selection_pins_price_routing_limits_but_allows_fresh_identical_verification(tmp_path, monkeypatch):
    from eval.pid2graph import runner

    panel = {"manifest_sha256": "manifest", "panels": {"validation": [{"id": "drawing"}], "test": []}}
    panel["panel_sha256"] = digest(panel)
    save_new(tmp_path / "panel.json", panel)
    save_new(tmp_path / "baseline.json", fixture_report(panel, "baseline", 0.4))
    price = {"input_per_token": 2.2e-7, "output_per_token": 6.6e-7,
             "context_length": 1048576, "provider_tags": ["provider"], "valid_until": 10}
    save_new(tmp_path / "prices.json", {"models": {"m": price}})
    selected = seal(tmp_path / "panel.json", [tmp_path / "baseline.json"], tmp_path / "selection.json",
                    prices_path=tmp_path / "prices.json")
    assert selected["price_snapshot_sha256"] == sha256(tmp_path / "prices.json")
    assert selected["final_price_limits"] == price_signature(price)
    assert price_signature({**price, "valid_until": 20}) == selected["final_price_limits"]
    assert price_signature({**price, "input_per_token": 3e-7}) != selected["final_price_limits"]
    assert price_signature({**price, "provider_tags": ["other"]}) != selected["final_price_limits"]
    save_new(tmp_path / "changed-prices.json", {"models": {"m": {**price, "input_per_token": 3e-7}}})
    monkeypatch.setattr(runner, "LLMClient", lambda cfg: pytest.fail("Price mismatch must fail before API client creation"))
    with pytest.raises(ValueError, match="price limits differ"):
        runner.run(tmp_path / "panel.json", tmp_path / "final", tmp_path / "ledger.json",
                   tmp_path / "changed-prices.json", split="test", model="m", selection=tmp_path / "selection.json")


def test_actual_billing_breaks_ties_without_using_reservations(tmp_path):
    panel = {"manifest_sha256": "manifest", "panels": {"validation": [{"id": "drawing"}]}}
    panel["panel_sha256"] = digest(panel)
    save_new(tmp_path / "panel.json", panel)
    base = fixture_report(panel, "baseline", 0.5)
    guided = fixture_report(panel, "supervised_guidance", 0.5)
    base.update(reserved_or_charged_usd=58, billing={"actual_billed_usd": 0.1, "actual_billing_complete": True})
    guided.update(reserved_or_charged_usd=1, billing={"actual_billed_usd": 0.2, "actual_billing_complete": True})
    save_new(tmp_path / "base.json", base)
    save_new(tmp_path / "guided.json", guided)
    selected = seal(tmp_path / "panel.json", [tmp_path / "base.json", tmp_path / "guided.json"], tmp_path / "chosen.json")
    assert selected["variant"] == "baseline" and "actual billed cost" in selected["rule"]
    base["billing"] = {"actual_billed_usd": 0, "actual_billing_complete": False}
    assert ranking_key(guided) < ranking_key(base)  # Unknown is not free.
    base["summary"]["macro_drawing_f1"] = 0.6
    assert ranking_key(base) < ranking_key(guided)  # Accuracy still comes first.


def test_selection_rejects_mixed_cost_formats_and_invalid_scores(tmp_path):
    panel = {"manifest_sha256": "manifest", "panels": {"validation": [{"id": "drawing"}]}}
    panel["panel_sha256"] = digest(panel)
    save_new(tmp_path / "panel.json", panel)
    base = fixture_report(panel, "baseline", 0.5)
    guided = fixture_report(panel, "supervised_guidance", 0.6)
    guided["billing"] = {"actual_billed_usd": 0.1, "actual_billing_complete": True}
    save_new(tmp_path / "base.json", base)
    save_new(tmp_path / "guided.json", guided)
    with pytest.raises(ValueError, match="mix legacy"):
        seal(tmp_path / "panel.json", [tmp_path / "base.json", tmp_path / "guided.json"], tmp_path / "chosen.json")
    base["summary"]["macro_drawing_f1"] = float("nan")
    with pytest.raises(ValueError, match="metric"):
        ranking_key(base)
