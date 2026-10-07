"""Export production-stage inputs without changing inference or approving results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.vision.stage_contracts import StageRequest
from diagex.vision.stages import file_hash


def capture_stage(
    run_dir, stage, backend, item, inputs, *, image=None, model=None, transport=None, output=None
):
    if run_dir is None:
        return
    root = Path(run_dir) / "module_inputs"
    assets = {}
    if image is not None:
        pixels = hashlib.sha256(
            image.mode.encode() + str(image.size).encode() + image.tobytes()
        ).hexdigest()
        path = root / "images" / (pixels + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            image.save(path)
        assets["image"] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    snapshot_path = Path(run_dir) / "knowledge.snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else {}
    request = StageRequest(
        knowledge=snapshot,
        stage=stage, backend=backend, inputs=inputs, assets=assets, model=model, transport=transport
    )
    # Validate stage inputs when recording, not only when replaying them.
    from diagex.vision.stages import STAGES

    STAGES[stage][0].model_validate(inputs)
    path = root / stage / (item + ".json")
    atomic_write_json(path, request.model_dump(mode="json"))
    if output is not None:
        if snapshot:
            from diagex.knowledge.resolver import resolve
            task = {"symbol_detection": "symbol_interpretation", "text_detection": "text_assignment", "line_detection": "line_interpretation", "connection_inference": "connections"}.get(stage, stage)
            output = {**output, "knowledge_trace": resolve(snapshot, task, inputs.get("page", {}).get("page_index", 0), "connector arrow tag line")}
        atomic_write_json(path.with_suffix(".output.json"), output)
