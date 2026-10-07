from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from typer.testing import CliRunner

from diagex.cli import app
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence, TextEvidence
from diagex.vision.line_detection import detect_lines
from diagex.vision.models import BBox, ReconciledNode
from diagex.vision.stage_contracts import StageRequest
from diagex.vision.stages import ModelRuntime, file_hash, run_stage
from diagex.vision.text_assignment import assign_text
from diagex.vision.topology import build_page_topology


@pytest.fixture
def page():
    return PageEvidence(
        page_index=0,
        source_ref="drawing#page=1",
        width=200,
        height=200,
        dpi=72,
        effective_dpi=72,
        is_scanned=False,
        text_spans=[TextEvidence(id="t1", text="PT-101", bbox=BBox(x=45, y=45, w=30, h=12))],
    )


@pytest.fixture
def node():
    return ReconciledNode(
        id="s1",
        page_index=0,
        kind="instrument",
        label="unlabelled",
        bbox_global=BBox(x=40, y=40, w=50, h=50),
        confidence="high",
    )


def request(stage, backend, inputs, **kw):
    return StageRequest(stage=stage, backend=backend, inputs=inputs, **kw)


def test_assignment_does_not_mutate_upstream_and_retains_source_ids(page, node):
    original = node.model_dump()
    result = assign_text(nodes=[node], pages=[page])
    assert node.model_dump() == original
    assert result.nodes[0].label == "PT-101"
    assert result.nodes[0].attributes["source_text_ids"] == ["t1"]
    assert result.assigned_count == 1


def test_prepared_lines_replay_without_running_line_detector(page, monkeypatch):
    lines = detect_lines(page=page)
    expected = build_page_topology(page=page, nodes=[])
    monkeypatch.setattr(
        "diagex.vision.topology.extract_raster_paths",
        lambda **kw: pytest.fail("Line detector must not run for saved lines"),
    )
    actual = build_page_topology(page=page, nodes=[], detected_lines=lines)
    assert actual == expected
    with pytest.raises(ValueError, match="different page"):
        build_page_topology(
            page=page.model_copy(update={"page_index": 1}), nodes=[], detected_lines=lines
        )


def test_captured_assignment_replays_production_output(page, node, tmp_path):
    from diagex.vision.fusion import _assign_text_stage
    from diagex.vision.stage_capture import capture_stage

    def record(req, output):
        capture_stage(
            tmp_path, "text_assignment", "geometry", "objects", req["inputs"], output=output
        )

    result = _assign_text_stage(
        nodes=[node], pages=[page], legend_pack=None, recorder=record
    )
    path = tmp_path / "module_inputs/text_assignment/objects.json"
    req = StageRequest.model_validate_json(path.read_text())
    replay, reused = run_stage(req, tmp_path / "replay")
    assert not reused
    assert replay["output"] == result.model_dump(mode="json")
    assert replay["output"] == json.loads(path.with_suffix(".output.json").read_text())


def test_saved_line_detection_feeds_independent_connection_stage(page, node, tmp_path):
    from diagex.vision.stage_capture import capture_stage

    capture_stage(
        tmp_path,
        "line_detection",
        "geometry",
        "page-0001",
        {"page": page.model_dump(mode="json")},
        image=Image.new("RGB", (200, 200), "white"),
    )
    path = tmp_path / "module_inputs/line_detection/page-0001.json"
    req = StageRequest.model_validate_json(path.read_text())
    lines, _ = run_stage(req, tmp_path / "replay", base_dir=path.parent)
    connection = request(
        "connection_inference",
        "geometry",
        {
            "page": page.model_dump(mode="json"),
            "nodes": [node.model_dump(mode="json")],
            "lines": lines["output"],
        },
    )
    graph, _ = run_stage(connection, tmp_path / "replay")
    expected = build_page_topology(
        page=page,
        nodes=[node],
        detected_lines=detect_lines(page=page, image=Image.new("RGB", (200, 200), "white")),
    )
    assert graph["output"] == expected.model_dump(mode="json")


def test_raster_text_assignment_preserves_origin(page, node):
    page.text_spans[0].origin = "raster_vlm"
    result = assign_text(nodes=[node], pages=[page])
    assigned = result.nodes[0]
    assert assigned.label == "PT-101"
    assert assigned.attributes["assigned_text_origin"] == "raster_vlm"
    assert "native_tag_assignment" not in assigned.attributes
    assert "native_text_agreement" not in assigned.attributes["system_confidence_evidence"]


def test_stage_cache_reuses_and_invalidates_inputs_and_implementation(
    page, node, tmp_path, monkeypatch
):
    from diagex.vision import stages

    req = request(
        "text_assignment",
        "geometry",
        {"pages": [page.model_dump(mode="json")], "nodes": [node.model_dump(mode="json")]},
    )
    first, reused = run_stage(req, tmp_path)
    assert not reused and first["output"]["nodes"][0]["label"] == "PT-101"
    compute = stages._compute
    monkeypatch.setattr(stages, "_compute", lambda *a: pytest.fail("Cached stage was rerun"))
    assert run_stage(req, tmp_path) == (first, True)
    monkeypatch.setattr(stages, "_compute", compute)
    changed = req.model_copy(deep=True)
    changed.inputs["pages"][0]["text_spans"][0]["text"] = "PT-102"
    second, reused = run_stage(changed, tmp_path)
    assert not reused and second["identity_sha256"] != first["identity_sha256"]
    assert second["output"]["nodes"][0]["label"] == "PT-102"
    monkeypatch.setattr(stages, "implementation_signature", lambda *a: {"revision": "different"})
    third, reused = run_stage(req, tmp_path)
    assert not reused and third["identity_sha256"] != first["identity_sha256"]


def test_tampered_output_is_not_reused(page, tmp_path):
    req = request("text_detection", "native", {"page": page.model_dump(mode="json")})
    result, _ = run_stage(req, tmp_path)
    path = tmp_path / req.stage / result["identity_sha256"] / "result.json"
    corrupt = copy.deepcopy(result)
    corrupt["output"]["text_spans"] = []
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="corrupt"):
        run_stage(req, tmp_path)


def test_asset_hash_rejected_before_model_call(page, tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (200, 200), "white").save(path)
    req = request(
        "text_detection",
        "vlm",
        {"page": page.model_dump(mode="json")},
        assets={"image": {"path": str(path), "sha256": "0" * 64}},
        model="m",
        transport="openrouter",
    )
    with pytest.raises(ValueError, match="checksum"):
        run_stage(req, tmp_path / "runs")
    req.assets["image"].sha256 = file_hash(path)
    with pytest.raises(ValueError, match="runtime"):
        run_stage(req, tmp_path / "runs")


def test_vlm_text_backend_is_separate_accounted_and_replayable(page, tmp_path):
    from anthropic.types import Message

    class Client:
        config = SimpleNamespace(model="test-model", transport="test")
        calls = 0

        def messages_create(self, **kwargs):
            self.calls += 1
            assert kwargs["tool_choice"]["name"] == "submit_text"
            return Message(
                id="message-1",
                model="test-model",
                type="message",
                role="assistant",
                content=[
                    {
                        "type": "tool_use",
                        "id": "tool1",
                        "name": "submit_text",
                        "input": {
                            "words": [
                                {
                                    "text": "PT-101",
                                    "x": 0.25,
                                    "y": 0.25,
                                    "width": 0.25,
                                    "height": 0.1,
                                }
                            ]
                        },
                    }
                ],
                stop_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 20},
            )

    path = tmp_path / "page.png"
    Image.new("RGB", (200, 200), "white").save(path)
    req = request(
        "text_detection",
        "vlm",
        {"page": page.model_dump(mode="json")},
        assets={"image": {"path": str(path), "sha256": file_hash(path)}},
        model="test-model",
        transport="test",
    )
    client = Client()
    cost = CostTracker()
    runtime = ModelRuntime(client, cost, NullReporter())
    result, reused = run_stage(req, tmp_path / "runs", runtime=runtime)
    assert not reused and client.calls == 1 and len(cost.steps) == 1
    span = result["output"]["text_spans"][0]
    assert span["origin"] == "raster_vlm" and span["bbox"] == {"x": 50, "y": 50, "w": 50, "h": 20}
    assert run_stage(req, tmp_path / "runs", runtime=runtime)[1]
    assert client.calls == 1
    client.config.model = "wrong-model"
    with pytest.raises(ValueError, match="differ"):
        run_stage(req, tmp_path / "runs", runtime=runtime)


def test_cv_detector_receives_only_pixels_and_preserves_raw_proposals(page):
    from diagex.vision.symbol_detection import detect_symbols

    class Detector:
        signature = {"weights": "fixture"}

        def predict(self, image):
            assert image.size == (page.width, page.height)
            return [{"id": "cv1", "bbox": [10, 10, 20, 20], "label": "valve", "confidence": 0.8}]

    result = detect_symbols(page=page, image=Image.new("RGB", (200, 200)), detector=Detector())
    assert result.raster_proposals[0]["label"] == "valve"
    assert result.status == "proposals_require_interpretation"


def test_cli_offline_replay_and_explicit_live_gate(page, tmp_path):
    path = tmp_path / "request.json"
    path.write_text(
        request(
            "text_detection", "native", {"page": page.model_dump(mode="json")}
        ).model_dump_json()
    )
    args = ["stage-run", str(path), "--out", str(tmp_path / "artifacts")]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert '"reused": false' in first.output
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0 and '"reused": true' in second.output
    data = json.loads(path.read_text())
    data["backend"] = "vlm"
    path.write_text(json.dumps(data))
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2 and "--live" in result.output


def test_legacy_imports_are_same_implementation():
    from diagex.vision import connection_inference, page_graph, perception, symbol_interpretation

    assert perception is symbol_interpretation
    assert page_graph is connection_inference


def test_cv_contract_import_does_not_load_torch_or_llm():
    import os
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import diagex.vision.symbol_detection; assert 'torch' not in sys.modules; assert 'diagex.llm.client' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
