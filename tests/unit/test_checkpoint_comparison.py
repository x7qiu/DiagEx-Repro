import copy
import json
import subprocess
from pathlib import Path

import pytest

from eval.pid2graph import checkpoint_comparison as comparison
from eval.pid2graph.data import digest, sha256


def write(path, value, seal=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if seal:
        value[seal] = digest(value)
    path.write_text(json.dumps(value))
    return value


def test_choose_uses_post_pilot_only_and_earliest_exact_tie():
    rows = [{"epoch": 0, "macro_drawing_f1": .99}, {"epoch": 1, "macro_drawing_f1": .8},
            {"epoch": 2, "macro_drawing_f1": .8}, {"epoch": 3, "macro_drawing_f1": .7}]
    assert comparison.choose_checkpoint({"complete": True, "validation": rows})["epoch"] == 1
    with pytest.raises(ValueError, match="not completed"):
        comparison.choose_checkpoint({"complete": False, "validation": rows})
    rows[2]["macro_drawing_f1"] = float("nan")
    with pytest.raises(ValueError, match="valid post-pilot"):
        comparison.choose_checkpoint({"complete": True, "validation": rows})


def test_incomplete_training_cannot_register_or_launch_paid_inference(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(comparison.subprocess, "run", lambda *a, **kw: launched.append(a))
    with pytest.raises(ValueError, match="Training is incomplete"):
        comparison.execute(tmp_path / "study", tmp_path / "compare", "manifest")
    assert not launched and not (tmp_path / "compare/plan.json").exists()
    assert json.loads((tmp_path / "compare/status.json").read_text())["stage"] == "failed"


@pytest.fixture
def study(tmp_path, monkeypatch):
    root, training = tmp_path / "study", tmp_path / "study/detector-epochs-v1"
    training.mkdir(parents=True)
    write(training / "completed.json", {"software_fixture": True})
    panel = write(root / "panel-pilot.json", {"manifest_sha256": "manifest", "panels": {"validation": [
        {"id": "validation-drawing", "image_sha256": "image"}]}}, "panel_sha256")
    weights = training / "epoch-001.pt"
    weights.write_bytes(b"software fixture; not real weights")
    detector = {"variant": "supervised_detector", "split": "validation", "panel_sha256": panel["panel_sha256"],
                "detector_sha256": sha256(weights)}
    proposals = training / "validation-epoch-001"
    write(proposals / "config.json", detector)
    write(proposals / digest("validation-drawing")[:16] / "predictions.json", {
        "image_sha256": "image", "config_sha256": digest(detector), "predictions": [], "runtime_seconds": 1})
    validated = training / "validation-epoch-001.json"
    write(validated, {"config": detector})
    write(training / "config.json", {"source_hashes": {}, "panel_sha256": panel["panel_sha256"]})
    exposure = {"complete": True, "validation": [{"epoch": 1, "macro_drawing_f1": .8,
        "checkpoint": str(weights), "report": str(validated), "report_sha256": sha256(validated)}]}
    monkeypatch.setattr(comparison, "training_report", lambda *a: copy.deepcopy(exposure))
    registrations, ranked = [], []
    method = tmp_path / "frozen-method.py"
    method.write_text("# fixture source\n")
    for variant in comparison.DRIVERS:
        config = {"variant": variant, "model": "test/model", "split": "validation", "panel_sha256": panel["panel_sha256"],
                  "guidance": {"old": True}, "provider_tags": ["provider"], "method": "frozen"}
        run = root / f"pilot-{variant}"
        write(run / "config.json", config)
        report = root / f"{variant}-report.json"
        write(report, {"config": config})
        reg = root / f"{variant}-registration.json"
        write(reg, {"runner_variant": variant, "method_fingerprints": {str(method): sha256(method)}})
        registrations.append({"registration": reg.name})
        ranked.append({"variant": variant, "run": str(run), "report": str(report), "report_sha256": sha256(report),
                       "registration_sha256": sha256(reg)})
    registry = root / "controlled-hybrids.json"
    write(registry, {"registered": registrations})
    policy = write(root / "validation-promotion-policy.json", {
        "hybrid_registry_sha256": sha256(registry), "selection_source_sha256": sha256(Path(comparison.__file__).with_name("selection.py")),
        "maximum_additional_pilot_checkpoint_runs": 2, "pilot_panel_sha256": panel["panel_sha256"]}, "policy_sha256")
    write(root / "pilot-shortlist.json", {"policy_sha256": policy["policy_sha256"], "shortlisted_variants": list(comparison.DRIVERS),
                                        "ranked": ranked}, "shortlist_sha256")
    return root, tmp_path / "comparisons", method


def test_replacement_changes_only_guidance_and_registers_exact_config_before_launch(study):
    root, output, _ = study
    plan = comparison.prepare(root, output)
    assert len(plan["arms"]) == 2 and plan["epoch"] == 1 and plan["test_access_authorized"] is False
    for arm in plan["arms"]:
        config = json.loads((Path(arm["run"]) / "config.json").read_text())
        original = json.loads((root / f"pilot-{arm['variant']}/config.json").read_text())
        assert {k: v for k, v in config.items() if k != "guidance"} == {k: v for k, v in original.items() if k != "guidance"}
        assert config["guidance"] != original["guidance"]
        assert digest(config) == arm["config_sha256"]
        assert arm["command"][arm["command"].index("--split") + 1] == "validation"
        assert (output / f"{arm['variant']}-registration.json").exists()
    assert comparison.prepare(root, output) == plan


def test_changed_frozen_method_prevents_registration(study):
    root, output, method = study
    method.write_text("# changed method\n")
    with pytest.raises(ValueError, match="Frozen hybrid source changed"):
        comparison.prepare(root, output)
    assert not (output / "plan.json").exists()


def test_changed_replacement_config_cannot_resume(study):
    root, output, _ = study
    plan = comparison.prepare(root, output)
    path = Path(plan["arms"][0]["run"]) / "config.json"
    config = json.loads(path.read_text())
    config["model"] = "different/model"
    write(path, config)
    with pytest.raises(ValueError, match="Cannot resume changed"):
        comparison.prepare(root, output)


def test_failed_comparison_stops_without_scoring_or_launching_second_arm(study, monkeypatch):
    root, output, _ = study
    launched = []
    def fail(command, **kw):
        launched.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(comparison.subprocess, "run", fail)
    monkeypatch.setattr(comparison, "score", lambda *a: pytest.fail("Incomplete run must not be scored as complete"))
    with pytest.raises(subprocess.CalledProcessError):
        comparison.execute(root, output, "manifest")
    assert len(launched) == 1
    assert json.loads((output / "status.json").read_text())["stage"] == "failed"
