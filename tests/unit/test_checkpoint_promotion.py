import json

import pytest

from eval.pid2graph import checkpoint_promotion as promotion
from eval.pid2graph.__main__ import score
from eval.pid2graph.data import digest, sha256
from eval.pid2graph.guidance import load_proposals
from eval.pid2graph.reuse_validation import seed


def write(path, value, seal=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if seal:
        value[seal] = digest(value)
    path.write_text(json.dumps(value))
    return value


def report(metric=.8, actual=.01, known=True):
    return {"split": "validation", "manifest_sha256": "manifest", "panel_sha256": "panel",
        "config": {"variant": "broad_raster_recognition", "split": "validation", "model": "m", "guidance": {}},
        "cases": [{"id": "drawing", "status": "partial"}], "summary": {"macro_drawing_f1": metric},
        "runtime_seconds": 10, "billing": {"actual_billed_usd": actual,
            "active_reservations_usd": 0, "actual_billing_complete": known}}


def test_checkpoint_choice_uses_frozen_accuracy_cost_and_pilot_first_ties():
    assert promotion.choose_pair(report(.8, .1), report(.79, .001)) == 0
    assert promotion.choose_pair(report(.79), report(.8)) == 1
    assert promotion.choose_pair(report(known=False), report(actual=.02)) == 1
    assert promotion.choose_pair(report(), report()) == 0


@pytest.mark.parametrize("change", ["model", "test", "missing", "active", "unaccounted", "different_drawing", "stopped"])
def test_unfinished_or_uncontrolled_pair_cannot_rank(change):
    a, b = report(), report()
    if change == "model":
        b["config"]["model"] = "other"
    elif change == "test":
        b["split"] = "test"
    elif change == "missing":
        b["cases"][0]["status"] = "missing"
    elif change == "active":
        b["billing"]["active_reservations_usd"] = .2
    elif change == "unaccounted":
        del b["billing"]
    elif change == "different_drawing":
        b["cases"][0]["id"] = "different"
    else:
        b["stopped"] = True
    with pytest.raises(ValueError):
        promotion.choose_pair(a, b)


def test_incomplete_comparisons_cannot_create_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(promotion, "verify_comparison", lambda *a: pytest.fail("Incomplete comparisons must not be prepared again"))
    with pytest.raises(ValueError, match="must finish"):
        promotion.run(tmp_path, tmp_path / "comparison", "manifest", "ledger", tmp_path / "promotion")
    assert not (tmp_path / "promotion").exists()


@pytest.fixture
def study(tmp_path, monkeypatch):
    root, comparison = tmp_path / "study", tmp_path / "comparison"
    root.mkdir()
    image = root / "software-source.png"
    image.write_bytes(b"software fixture source; never passed to inference")
    graph = root / "software-truth.graphml"
    graph.write_text('''<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <key id="l" attr.name="label"/><key id="x1" attr.name="xmin"/>
      <key id="y1" attr.name="ymin"/><key id="x2" attr.name="xmax"/><key id="y2" attr.name="ymax"/>
      <graph edgedefault="undirected"><node id="truth">
      <data key="l">valve</data><data key="x1">0</data><data key="y1">0</data>
      <data key="x2">10</data><data key="y2">10</data></node></graph></graphml>''')
    rows = [{"id": name, "image": str(image), "image_sha256": sha256(image), "collection": "fixture", "size": [20, 20]}
            for name in ["first", "second"]]
    manifest = write(root / "manifest.json", {"dataset_root": str(root), "limits": ["software fixture"],
        "drawings": [{**r, "graph": graph.name, "graph_sha256": sha256(graph), "split": "validation"} for r in rows]}, "manifest_sha256")
    panels = {}
    for name, selected in [("pilot", rows[:1]), ("broad", rows)]:
        panels[name] = write(root / f"panel-{name}.json", {"manifest_sha256": manifest["manifest_sha256"],
            "panels": {"validation": selected, "test": []}}, "panel_sha256")
    proposal_runs, guidances = {}, {}
    for weight in ("pilot", "trained"):
        for size in ("pilot", "broad"):
            directory = root / f"proposals-{weight}-{size}"
            config = {"variant": "supervised_detector", "split": "validation",
                      "panel_sha256": panels[size]["panel_sha256"], "detector_sha256": weight}
            write(directory / "config.json", config)
            for row in panels[size]["panels"]["validation"]:
                write(directory / digest(row["id"])[:16] / "predictions.json", {
                    "id": row["id"], "image_sha256": row["image_sha256"], "config_sha256": digest(config),
                    "predictions": [], "runtime_seconds": 1})
            proposal_runs[weight, size] = directory
            guidances[weight, size] = load_proposals(directory, panels[size], "validation")[1]
    ledger = {"schema_version": 2, "prior_spend": {"usd": 0}, "requests": []}
    registry, ranked, arms, configs, runs = [], [], [], {}, {}
    method = root / "software-method.py"
    method.write_text("# software fixture")
    for variant in ("broad_raster_recognition", "explicit_proposal_decisions"):
        for weight in ("pilot", "trained"):
            directory = (root if weight == "pilot" else comparison) / variant
            config = {"variant": variant, "model": "model", "split": "validation",
                      "panel_sha256": panels["pilot"]["panel_sha256"], "guidance": guidances[weight, "pilot"]}
            configs[variant, weight], runs[variant, weight] = config, directory
            write(directory / "config.json", config)
            identity = variant + "-" + weight
            ledger["requests"].append({"id": identity, "status": "billed", "model": "model", "category": "experiments",
                "actual_billed_usd": .02, "charged_usd": .02, "exposure_usd": .02})
            best = (variant == "broad_raster_recognition") == (weight == "trained")
            predictions = [{"id": "p", "label": "valve", "bbox": [0, 0, 10, 10], "confidence": .9, "disposition": "review_proposal"}]
            if not best:
                predictions.append({"id": "fp", "label": "tank", "bbox": [10, 10, 20, 20], "confidence": .9, "disposition": "review_proposal"})
            case = directory / digest("first")[:16]
            write(case / "p0-r0-c0.json", {"tile_id": "p0-r0-c0", "status": "complete", "predictions": predictions,
                "config_sha256": digest(config), "ledger_request_ids": [identity]})
            write(case / "predictions.json", {"id": "first", "image_sha256": rows[0]["image_sha256"],
                "config_sha256": digest(config), "tiles": 1, "completed_tiles": 1, "predictions": predictions,
                "ledger_request_ids": [identity], "errors": [], "runtime_seconds": 10, "reserved_or_charged_usd": .02})
        registration = {"runner_variant": variant, "model": "model", "split": "validation",
            "panel_sha256": panels["pilot"]["panel_sha256"], "method_fingerprints": {str(method): sha256(method)},
            "guidance": guidances["pilot", "pilot"], "proposal_run": str(proposal_runs["pilot", "pilot"])}
        path = root / (variant + "-registration.json")
        write(path, registration)
        registry.append({"registration": path.name})
        trained_config = configs[variant, "trained"]
        arms.append({"variant": variant, "run": str(runs[variant, "trained"]), "config_sha256": digest(trained_config)})
        write(comparison / f"{variant}-registration.json", {"proposal_run": str(proposal_runs["trained", "pilot"])}, "registration_sha256")
    ledger_path = root / "ledger.json"
    write(ledger_path, ledger)
    for variant in ("broad_raster_recognition", "explicit_proposal_decisions"):
        for weight in ("pilot", "trained"):
            path = (root if weight == "pilot" else comparison) / f"{variant}-report.json"
            score(root / "manifest.json", root / "panel-pilot.json", runs[variant, weight], path, "validation", ledger_path)
            if weight == "pilot":
                ranked.append({"variant": variant, "report": str(path), "run": str(runs[variant, weight])})
    # Emulate later reconciliation without changing any prediction or metric.
    for row in ledger["requests"]:
        row.update(actual_billed_usd=.01, charged_usd=.01, exposure_usd=.01)
    write(ledger_path, ledger)
    write(root / "controlled-hybrids.json", {"registered": registry})
    write(root / "pilot-shortlist.json", {"ranked": ranked}, "shortlist_sha256")
    write(root / "validation-promotion-policy.json", {"broad_panel_sha256": panels["broad"]["panel_sha256"],
        "maximum_broader_hybrids": 2, "tie_policy": "Use frozen ranking key; pilot first on exact ties"}, "policy_sha256")
    write(comparison / "status.json", {"stage": "comparisons_complete"})
    monkeypatch.setattr(promotion, "verify_comparison", lambda *a: {"arms": arms})
    return root, comparison, ledger_path, panels, runs, proposal_runs


def test_real_scoring_selects_both_weight_outcomes_and_registration_supports_reuse(study, tmp_path):
    root, comparison, ledger, panels, runs, proposal_runs = study
    output = tmp_path / "promotion"
    result = promotion.run(root, comparison, root / "manifest.json", ledger, output)
    assert result["test_access_authorized"] is False
    assert [c["weight_source"] for c in result["choices"]] == ["trained", "pilot"]
    assert all(c["billing"]["actual_billed_usd"] == .01 for c in result["choices"])
    for choice in result["choices"]:
        target = tmp_path / (choice["variant"] + "-expanded")
        seeded = seed(root / "panel-pilot.json", root / "panel-broad.json", choice["source_run"],
            target, choice["registration"], root / "manifest.json", source_proposals=choice["source_proposals"],
            target_proposals=proposal_runs[choice["weight_source"], "broad"])
        assert seeded["new_paid_requests"] == 0 and seeded["unattempted_drawings"] == ["second"]
        reused = json.loads((target / digest("first")[:16] / "p0-r0-c0.json").read_text())
        assert reused["ledger_request_ids"] == [choice["variant"] + "-" + choice["weight_source"]]
    with pytest.raises(FileExistsError):
        promotion.run(root, comparison, root / "manifest.json", ledger, output)


@pytest.mark.parametrize("change", ["active", "foreign_model", "missing_tile", "missing_drawing", "source_changed", "mismatched_predictions"])
def test_invalid_attempt_provenance_blocks_promotion(study, tmp_path, change):
    root, comparison, ledger_path, panels, runs, _ = study
    run = runs["broad_raster_recognition", "trained"]
    case = run / digest("first")[:16]
    ledger = json.loads(ledger_path.read_text())
    if change in {"active", "foreign_model"}:
        row = next(r for r in ledger["requests"] if r["id"] == "broad_raster_recognition-trained")
        row.update({"status": "pending"} if change == "active" else {"model": "other"})
    elif change == "missing_tile":
        (case / "p0-r0-c0.json").unlink()
    elif change == "missing_drawing":
        (case / "predictions.json").unlink()
    elif change == "source_changed":
        (root / "software-source.png").write_bytes(b"changed")
    else:
        value = json.loads((case / "predictions.json").read_text())
        value["predictions"] = []
        write(case / "predictions.json", value)
    with pytest.raises(ValueError):
        promotion.verify_attempts(run, panels["pilot"], ledger)


def test_failed_finished_attempt_is_retained_with_unknown_cost(study):
    _, _, ledger_path, panels, runs, _ = study
    run = runs["broad_raster_recognition", "trained"]
    case = run / digest("first")[:16]
    tile = json.loads((case / "p0-r0-c0.json").read_text())
    tile.update(status="failed", error="provider timeout", predictions=[])
    write(case / "p0-r0-c0.json", tile)
    drawing = json.loads((case / "predictions.json").read_text())
    drawing.update(completed_tiles=0, predictions=[], errors=[{"tile_id": tile["tile_id"], "error": tile["error"]}])
    write(case / "predictions.json", drawing)
    ledger = json.loads(ledger_path.read_text())
    row = next(r for r in ledger["requests"] if r["id"] == "broad_raster_recognition-trained")
    row.update(status="unresolved", actual_billed_usd=None, charged_usd=.2, exposure_usd=.2)
    result = promotion.verify_attempts(run, panels["pilot"], ledger)
    assert result["failed_tiles"] == result["attempted_tiles"] == 1
    assert result["billing"]["actual_billing_complete"] is False
    assert result["billing"]["unresolved_upper_usd"] == .2


def test_broader_controller_reuses_pilot_preserves_failure_and_resumes_without_calls(study, tmp_path, monkeypatch):
    from pathlib import Path

    from eval.pid2graph import broader_hybrids as controller

    root, comparison, ledger, panels, _, proposals = study
    promoted = tmp_path / "promoted"
    promotion.run(root, comparison, root / "manifest.json", ledger, promoted)
    write(root / "spending.json", json.loads(ledger.read_text()))
    output = tmp_path / "expanded"
    kwargs = (root, promoted / "promotion.json", root / "manifest.json", output,
              {weight: proposals[weight, "broad"] for weight in ("pilot", "trained")})
    calls = []

    def fake_driver(command, check):
        assert check is True
        assert command[command.index("--split") + 1] == "validation"
        run = Path(command[command.index("--out") + 1])
        config = json.loads((run / "config.json").read_text())
        variant = config["variant"]
        calls.append(variant)
        # The reused drawing must already be present before any new request.
        prior = json.loads((run / digest("first")[:16] / "predictions.json").read_text())
        assert prior["validation_reuse"]["new_paid_request"] is False
        case = run / digest("second")[:16]
        request_id = "new-" + variant
        write(case / "p0-r0-c0.json", {"tile_id": "p0-r0-c0", "status": "failed", "error": "timeout",
            "config_sha256": digest(config), "predictions": [], "ledger_request_ids": [request_id]})
        write(case / "predictions.json", {"id": "second", "image_sha256": prior["image_sha256"],
            "config_sha256": digest(config), "tiles": 1, "completed_tiles": 0, "predictions": [],
            "ledger_request_ids": [request_id], "errors": [{"tile_id": "p0-r0-c0", "error": "timeout"}],
            "runtime_seconds": 10, "reserved_or_charged_usd": .2})
        current = json.loads((root / "spending.json").read_text())
        current["requests"].append({"id": request_id, "model": "model", "category": "experiments",
            "status": "unresolved", "actual_billed_usd": None, "exposure_usd": .2, "charged_usd": .2})
        write(root / "spending.json", current)

    monkeypatch.setattr(controller.subprocess, "run", fake_driver)
    controller.execute(*kwargs, prepare_only=True)
    assert calls == []
    controller.execute(*kwargs)
    assert calls == list(controller.DRIVERS)
    controller.execute(*kwargs)
    assert calls == list(controller.DRIVERS)
    for variant in calls:
        report = json.loads((output / f"{variant}-report.json").read_text())
        coverage = json.loads((output / f"{variant}-coverage.json").read_text())
        assert report["panel_sha256"] == panels["broad"]["panel_sha256"]
        assert report["cases"][1]["status"] == "partial"
        assert report["billing"]["actual_billed_usd"] == .01
        assert report["billing"]["unresolved_upper_usd"] == .2
        assert coverage["attempted_tiles"] == 2 and coverage["failed_tiles"] == 1
    assert json.loads((output / "status.json").read_text())["stage"] == "broader_hybrids_complete"


@pytest.mark.parametrize("change", ["extra_variant", "source_code", "wrong_weights", "target_config"])
def test_broader_controller_refuses_incompatible_promotion_before_paid_calls(study, tmp_path, monkeypatch, change):
    from eval.pid2graph import broader_hybrids as controller

    root, comparison, ledger, _, _, proposals = study
    promoted, output = tmp_path / "promoted", tmp_path / "expanded"
    result = promotion.run(root, comparison, root / "manifest.json", ledger, promoted)
    mapping = {weight: proposals[weight, "broad"] for weight in ("pilot", "trained")}
    args = (root, promoted / "promotion.json", root / "manifest.json", output, mapping)
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k: pytest.fail("No paid inference allowed"))
    if change == "extra_variant":
        result["choices"].append(result["choices"][0])
        del result["promotion_sha256"]
        write(promoted / "promotion.json", result, "promotion_sha256")
    elif change == "source_code":
        (root / "software-method.py").write_text("# changed source")
    elif change == "wrong_weights":
        mapping["trained"] = proposals["pilot", "broad"]
    else:
        controller.execute(*args, prepare_only=True)
        config_path = output / "broad_raster_recognition/config.json"
        config = json.loads(config_path.read_text())
        config["model"] = "another-model"
        write(config_path, config)
    with pytest.raises(ValueError):
        controller.execute(*args)


def test_broader_controller_does_not_run_without_completed_promotion(study, tmp_path, monkeypatch):
    from eval.pid2graph import broader_hybrids as controller

    root, _, _, _, _, proposals = study
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **k: pytest.fail("No paid inference allowed"))
    with pytest.raises(FileNotFoundError):
        controller.execute(root, tmp_path / "missing-promotion.json", root / "manifest.json", tmp_path / "expanded",
                           {weight: proposals[weight, "broad"] for weight in ("pilot", "trained")})


@pytest.fixture
def finished_broader(study, tmp_path, request):
    from pathlib import Path

    from eval.pid2graph import broader_hybrids, selection

    root, comparison, ledger_path, panels, _, proposals = study
    winner = getattr(request, "param", "broad_raster_recognition")
    policy_path = root / "validation-promotion-policy.json"
    policy = json.loads(policy_path.read_text())
    del policy["policy_sha256"]
    policy["selection_source_sha256"] = sha256(selection.__file__)
    write(policy_path, policy, "policy_sha256")
    state = root / "completed-training.json"
    write(state, {"reason": "maximum_epochs"})
    training = comparison / "training-completion.json"
    write(training, {"complete": True, "state_path": str(state), "state_sha256": sha256(state)})
    for variant in broader_hybrids.DRIVERS:
        path = comparison / f"{variant}-registration.json"
        reg = json.loads(path.read_text())
        del reg["registration_sha256"]
        reg["training_completion_sha256"] = sha256(training)
        write(path, reg, "registration_sha256")
    promoted, hybrids = tmp_path / "promoted", tmp_path / "hybrids"
    promotion.run(root, comparison, root / "manifest.json", ledger_path, promoted)
    plan = broader_hybrids.prepare(root, promoted / "promotion.json", root / "manifest.json", hybrids,
        {weight: proposals[weight, "broad"] for weight in ("pilot", "trained")})
    ledger = json.loads(ledger_path.read_text())
    baseline_config = {"variant": "baseline", "model": "model", "split": "validation",
                       "panel_sha256": panels["broad"]["panel_sha256"]}
    baseline = root / "baseline-broad-v1"
    write(baseline / "config.json", baseline_config)
    reg = root / "baseline-registration.json"
    write(reg, {"method_fingerprints": {str(root / "software-method.py"): sha256(root / "software-method.py")}})
    write(baseline / "validation-reuse.json", {"registration": str(reg), "registration_sha256": sha256(reg),
        "target_config_sha256": digest(baseline_config)})
    configs = [("baseline", baseline, baseline_config)] + [
        (a["variant"], Path(a["run"]), a["config"]) for a in plan["arms"]]
    for variant, directory, config in configs:
        for row in panels["broad"]["panels"]["validation"]:
            if variant != "baseline" and row["id"] == "first":
                continue  # Preserve the actually reused pilot record and cost.
            identity = variant + "-new-" + row["id"]
            succeeds = variant == winner
            predictions = [{"id": "p", "label": "valve", "bbox": [0, 0, 10, 10],
                            "confidence": .9, "disposition": "review_proposal"}] if succeeds else []
            errors = [] if succeeds else [{"tile_id": "p0-r0-c0", "error": "timeout"}]
            case = directory / digest(row["id"])[:16]
            write(case / "p0-r0-c0.json", {"tile_id": "p0-r0-c0", "status": "complete" if succeeds else "failed",
                "error": None if succeeds else "timeout", "config_sha256": digest(config),
                "predictions": predictions, "ledger_request_ids": [identity]})
            write(case / "predictions.json", {"id": row["id"], "image_sha256": row["image_sha256"],
                "config_sha256": digest(config), "tiles": 1, "completed_tiles": int(succeeds),
                "predictions": predictions, "ledger_request_ids": [identity], "errors": errors,
                "runtime_seconds": 10, "reserved_or_charged_usd": .02 if succeeds else .2})
            ledger["requests"].append({"id": identity, "category": "baseline" if variant == "baseline" else "experiments", "model": "model",
                "status": "billed" if succeeds else "unresolved", "actual_billed_usd": .01 if succeeds else None,
                "charged_usd": .01 if succeeds else .2, "exposure_usd": .01 if succeeds else .2})
    write(root / "spending.json", ledger)
    for variant, directory, _ in configs:
        report = root / "baseline-broad-report.json" if variant == "baseline" else hybrids / f"{variant}-report.json"
        score(root / "manifest.json", root / "panel-broad.json", directory, report, "validation", root / "spending.json")
    write(root / "prices-v1.json", {"models": {"model": {"context_length": 1000, "input_per_token": .001,
        "output_per_token": .002, "provider_tags": ["fixture"]}}})
    write(hybrids / "status.json", {"stage": "broader_hybrids_complete"})
    return root, hybrids, comparison, winner


@pytest.mark.parametrize("finished_broader", ["baseline", "broad_raster_recognition", "explicit_proposal_decisions"], indirect=True)
def test_final_selection_uses_full_validation_and_can_select_each_candidate(finished_broader, monkeypatch):
    from eval.pid2graph import final_selection

    root, hybrids, comparison, winner = finished_broader
    ledger = json.loads((root / "spending.json").read_text())
    for row in ledger["requests"]:
        if row["actual_billed_usd"] is not None:
            row.update(actual_billed_usd=.005, charged_usd=.005, exposure_usd=.005)
    write(root / "spending.json", ledger)
    scoring = final_selection.score
    splits = []
    def checked_score(*args):
        splits.append(args[4])
        assert args[4] == "validation"
        return scoring(*args)
    monkeypatch.setattr(final_selection, "score", checked_score)
    selected = final_selection.run(root, hybrids, comparison, root / "manifest.json")
    assert selected["variant"] == winner
    assert splits == ["validation"] * 3
    assert len(selected["validation_reports"]) == 3
    chosen_report = json.loads((root / "final-selection-v1" / (winner + "-report.json")).read_text())
    assert chosen_report["billing"]["actual_billed_usd"] == .01
    # A changed later ledger must not cause re-ranking or rewrite the decision.
    write(root / "spending.json", {"changed_after_selection": True})
    assert final_selection.run(root, hybrids, comparison, root / "manifest.json") == selected
    assert len(splits) == 3
    coverage = json.loads((root / "final-selection-v1/coverage.json").read_text())
    assert coverage[winner]["attempted_tiles"] == 2
    assert coverage[winner]["failed_tiles"] == 0
    assert sum(v["failed_tiles"] for v in coverage.values()) >= 2


@pytest.mark.parametrize("change", ["running", "missing_report", "missing_tile", "active_request", "foreign_category", "changed_prediction", "changed_source", "changed_training", "missing_baseline"])
def test_final_selection_rejects_incomplete_or_changed_evidence(finished_broader, change):
    from eval.pid2graph import final_selection

    root, hybrids, comparison, _ = finished_broader
    if change == "running":
        write(hybrids / "status.json", {"stage": "running_broader_validation"})
    elif change == "missing_report":
        (hybrids / "explicit_proposal_decisions-report.json").unlink()
    elif change == "missing_tile":
        (root / "baseline-broad-v1" / digest("second")[:16] / "p0-r0-c0.json").unlink()
    elif change == "active_request":
        path = root / "spending.json"
        ledger = json.loads(path.read_text())
        ledger["requests"][-1]["status"] = "pending"
        write(path, ledger)
    elif change == "foreign_category":
        path = root / "spending.json"
        ledger = json.loads(path.read_text())
        ledger["requests"][-1]["category"] = "baseline"
        write(path, ledger)
    elif change == "changed_prediction":
        path = hybrids / "broad_raster_recognition" / digest("second")[:16] / "predictions.json"
        drawing = json.loads(path.read_text())
        drawing["predictions"] = []
        write(path, drawing)
    elif change == "changed_source":
        (root / "software-method.py").write_text("# changed source")
    elif change == "changed_training":
        write(root / "completed-training.json", {"reason": "interrupted"})
    else:
        (root / "baseline-broad-report.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        final_selection.run(root, hybrids, comparison, root / "manifest.json")
    assert not (root / "final-selection-v1/selection.json").exists()


def test_final_selection_refuses_changed_report_after_sealing(finished_broader):
    from eval.pid2graph import final_selection

    root, hybrids, comparison, _ = finished_broader
    final_selection.run(root, hybrids, comparison, root / "manifest.json")
    path = hybrids / "broad_raster_recognition-report.json"
    report = json.loads(path.read_text())
    report["summary"]["macro_drawing_f1"] = .01
    write(path, report)
    with pytest.raises(ValueError, match="Cannot resume changed"):
        final_selection.run(root, hybrids, comparison, root / "manifest.json")
