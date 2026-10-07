"""Run one extraction responsibility with immutable, content-addressed artifacts.

No model client is constructed here. Paid backends require an explicit runtime;
CV/native stages neither import Torch until needed nor invoke a VLM.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.vision.stage_contracts import (
    AssignmentInput,
    ConnectionInput,
    PageInput,
    StageRequest,
    SymbolInput,
    SymbolInterpretationInput,
    TextInput,
)

STAGES = {
    "symbol_detection": (
        SymbolInput,
        ("native", "cv"),
        ("symbol_detection", "symbol_candidates", "vector_geometry", "raster_detector"),
    ),
    "text_detection": (
        TextInput,
        ("native", "vlm"),
        ("text_detection", "text_recognition_vlm", "encode"),
    ),
    "line_detection": (PageInput, ("geometry",), ("line_detection",)),
    "text_assignment": (
        AssignmentInput,
        ("geometry",),
        ("text_assignment", "native_hierarchy", "native_text", "vector_geometry", "reconcile"),
    ),
    "symbol_interpretation": (
        SymbolInterpretationInput,
        ("baseline", "broad_review"),
        (
            "symbol_interpretation",
            "raster_pipeline",
            "raster_broad",
            "raster_semantics",
            "reference_evidence",
            "raster_review",
            "raster_guidance",
            "legend_context",
            "views",
            "encode",
            "symbol_candidates",
            "vector_geometry",
        ),
    ),
    "connection_inference": (
        ConnectionInput,
        ("geometry", "vlm"),
        (
            "topology",
            "connection_inference",
            "native_hierarchy",
            "native_text",
            "legend_context",
            "line_detection",
            "vector_geometry",
            "process_context",
            "encode",
        ),
    ),
    "line_interpretation": (
        ConnectionInput,
        ("vlm",),
        (
            "connection_inference",
            "legend_context",
            "topology",
            "line_detection",
            "vector_geometry",
            "encode",
        ),
    ),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def implementation_signature(stage, backend):
    root = Path(__file__).parent
    files = {
        root / (name + ".py")
        for name in (
            *STAGES[stage][2],
            "stages",
            "stage_contracts",
            "evidence",
            "models",
            "legend_models",
        )
    }
    files.add(root.parent / "config.py")
    files.add(root.parent / "llm/prompts/output_language.py")
    files.update((root.parent / "knowledge").glob("*.py"))
    if is_live(stage, backend):
        files.update(root.parent / "llm" / name for name in ("client.py", "billing.py", "cost.py"))
    sources = {str(p.relative_to(root.parent)): file_hash(p) for p in sorted(files)}
    packages = ["pydantic", "Pillow", "PyMuPDF"]
    if stage == "symbol_detection" and backend == "cv":
        packages += ["torch", "torchvision"]
    if stage == "line_detection":
        packages += ["opencv-python-headless", "numpy"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"sources": sources, "packages": versions}


def is_live(stage, backend):
    return stage == "symbol_interpretation" or backend == "vlm"


@dataclass
class ModelRuntime:
    client: Any
    cost_tracker: Any
    reporter: Any

    def signature(self):
        config = self.client.config
        # Never serialize credentials or an endpoint URL that could contain one.
        result = {
            name: getattr(config, name, None)
            for name in ("model", "transport", "reasoning_mode", "production_open_weight")
        }
        for name in ("azure_endpoint", "azure_deployment", "azure_api_version"):
            value = getattr(config, name, None)
            if value:
                result[name + "_sha256"] = digest(value)
        return result


def _page_image(page, image):
    if image is not None and image.size != (page.width, page.height):
        raise ValueError("Asset pixels differ from the declared page coordinate frame")
    return image


def _diagram(page, image):
    from diagex.vision.models import DiagramPage

    return DiagramPage(
        page_index=page.page_index,
        width=page.width,
        height=page.height,
        dpi=page.dpi,
        effective_dpi=page.effective_dpi,
        is_scanned=page.is_scanned,
        source_ref=page.source_ref,
        image=image,
        rotation_deg=page.rotation_deg,
    )


def _compute(request, data, assets, runtime):
    image = None
    if "image" in assets:
        with Image.open(assets["image"]) as im:
            image = im.convert("RGB")
        _page_image(data.page, image)
    stage, backend = request.stage, request.backend
    if request.knowledge:
        from diagex.knowledge.resolver import resolve
        task = {"connection_inference": "connections"}.get(stage, stage)
        context = resolve(request.knowledge, task, getattr(getattr(data, "page", None), "page_index", 0), "connector arrow tag line")
        if stage == "symbol_interpretation":
            from diagex.knowledge.resolver import resolve_symbol_context
            context = resolve_symbol_context(request.knowledge, data.page, data.candidates,
                                             data.legend_entries, data.ownership_bbox)
            data.page_context["knowledge"] = context
        elif stage == "text_assignment":
            data.knowledge = request.knowledge
        elif stage in {"line_interpretation", "connection_inference"}:
            data.knowledge_context = context
    if stage == "symbol_detection":
        from diagex.vision.symbol_detection import detect_symbols

        detector = None
        if backend == "cv":
            from diagex.vision.raster_detector import RasterDetector

            detector = RasterDetector(
                assets["checkpoint"],
                device=data.device,
                expected_sha256=request.assets["checkpoint"].sha256,
            )
        return detect_symbols(page=data.page, image=image, detector=detector).model_dump(
            mode="json"
        )
    if stage == "text_detection":
        from diagex.vision.text_detection import detect_text, extract_pdf_text

        if "source_pdf" in assets:
            import fitz

            with fitz.open(assets["source_pdf"]) as doc:
                data.page.text_spans = extract_pdf_text(
                    doc[data.page.page_index], _diagram(data.page, image)
                )
        recognizer = None
        if backend == "vlm":
            from diagex.vision.text_recognition_vlm import VLMTextRecognizer

            recognizer = VLMTextRecognizer(
                client=runtime.client, cost_tracker=runtime.cost_tracker, max_tokens=data.max_tokens
            )
        return detect_text(page=data.page, image=image, recognizer=recognizer).model_dump(
            mode="json"
        )
    if stage == "line_detection":
        from diagex.vision.line_detection import detect_lines

        return detect_lines(page=data.page, image=image).model_dump(mode="json")
    if stage == "text_assignment":
        from diagex.vision.text_assignment import assign_text

        return assign_text(**{k: getattr(data, k) for k in type(data).model_fields}).model_dump(
            mode="json"
        )
    if stage == "symbol_interpretation":
        from diagex.config import SymbolPerceptionConfig
        from diagex.vision.symbol_interpretation import perceive_tile
        from diagex.vision.views import ViewProvider

        policy = SymbolPerceptionConfig(**data.policy)
        if policy.workflow != "fixed":
            raise ValueError("Independent symbol replay currently supports fixed views")
        provider = ViewProvider(_diagram(data.page, image), [data.tile])
        view, info = provider.get_tile(data.tile.id)
        perceive = perceive_tile
        if backend == "broad_review":
            from diagex.vision.raster_pipeline import perceive_broad_with_semantics

            perceive = perceive_broad_with_semantics
        outcome = perceive(
            client=runtime.client,
            cost_tracker=runtime.cost_tracker,
            reporter=runtime.reporter,
            page=data.page,
            tile=data.tile,
            view_image=view,
            view_info=info,
            ownership_bbox=data.ownership_bbox,
            legend_summary=data.legend_entries,
            step=len(runtime.cost_tracker.steps) + 1,
            page_context=data.page_context,
            candidates=data.candidates,
            policy=policy,
            region_provider=provider.get_region,
            reasoning_mode=getattr(runtime.client.config, "reasoning_mode", "auto"),
        )
        return {
            "detections": [d.model_dump(mode="json") for d in outcome.detections],
            "batch": outcome.batch.model_dump(mode="json"),
            "attempts": outcome.attempts,
            "contract_failed": outcome.contract_failed,
            "candidates": [c.model_dump(mode="json") for c in outcome.candidates],
            "diagnostics": outcome.recovery_diagnostics,
        }
    from diagex.vision.topology import LegendLineProfile, TopologyResult, build_page_topology

    if backend == "geometry":
        return build_page_topology(
            page=data.page,
            nodes=data.nodes,
            detected_lines=data.lines,
            legend_line_profile=LegendLineProfile.model_validate(data.legend_line_profile)
            if data.legend_line_profile is not None
            else None,
        ).model_dump(mode="json")
    from diagex.vision.connection_inference import (
        PageLineEvidence,
        classify_page_line_evidence,
        solve_page_graph,
    )

    if data.topology is None:
        raise ValueError("Model-based connections require saved geometry topology")
    kwargs = dict(
        knowledge_context=data.knowledge_context,
        client=runtime.client,
        cost_tracker=runtime.cost_tracker,
        reporter=runtime.reporter,
        page=data.page,
        rendered_image=image,
        nodes=data.nodes,
        topology=TopologyResult.model_validate(data.topology),
        step=len(runtime.cost_tracker.steps) + 1,
        max_tokens=data.max_tokens,
    )
    if stage == "line_interpretation":
        return classify_page_line_evidence(**kwargs).model_dump(mode="json")
    return solve_page_graph(
        **kwargs,
        all_nodes=data.all_nodes or data.nodes,
        pages=data.pages or [data.page],
        legend_summary=data.legend_entries,
        visual_evidence=PageLineEvidence.model_validate(data.visual_evidence)
        if data.visual_evidence
        else None,
        include_visual_context=False,
    ).model_dump(mode="json")


def run_stage(request: StageRequest, output: Path, *, base_dir=None, runtime=None):
    """Validate inputs before any call; replay only successful matching artifacts."""
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    request = request.model_copy(deep=True)
    model, backends, _ = STAGES[request.stage]
    if request.backend not in backends:
        raise ValueError(f"Backend must be one of {backends}")
    data = model.model_validate(request.inputs)
    assets = {}
    allowed_assets = {
        "symbol_detection": {"image", "checkpoint"},
        "text_detection": {"image", "source_pdf"},
        "line_detection": {"image"},
        "text_assignment": set(),
        "symbol_interpretation": {"image"},
        "connection_inference": {"image"},
        "line_interpretation": {"image"},
    }
    if set(request.assets) - allowed_assets[request.stage]:
        raise ValueError("Unexpected asset for this stage")
    for name, asset in request.assets.items():
        path = (Path(base_dir) / asset.path).resolve()
        if file_hash(path) != asset.sha256:
            raise ValueError(f"Asset checksum mismatch: {name}")
        assets[name] = path
        asset.path = str(path)
    live = is_live(request.stage, request.backend)
    if request.backend == "cv" and not {"image", "checkpoint"} <= set(assets):
        raise ValueError("CV detection requires image and checkpoint assets")
    if live and "image" not in assets:
        raise ValueError("Model-based stages require a rendered image asset")
    if live and not (request.model and request.transport):
        raise ValueError("Model-based stages require explicit model and transport")
    if live and runtime is None:
        raise ValueError("A model runtime is required; the CLI needs --live")
    runtime_signature = runtime.signature() if live else None
    if live and any(runtime_signature[k] != getattr(request, k) for k in ("model", "transport")):
        raise ValueError("Runtime model/transport differ from the request")
    identity = {
        "request": request.model_dump(mode="json"),
        "implementation": implementation_signature(request.stage, request.backend),
        "model_runtime": runtime_signature,
    }
    key = digest(identity)
    folder = Path(output) / request.stage / key
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result_path = folder / "result.json"
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if record.get("identity_sha256") != key or digest(record["output"]) != record.get(
                "output_sha256"
            ):
                raise ValueError("Stored stage artifact is corrupt")
            return record, True
        atomic_write_json(folder / "request.json", identity["request"])
        atomic_write_json(folder / "identity.json", identity)
        try:
            usage_start = len(runtime.cost_tracker.steps) if live else 0
            result = _compute(request, data, assets, runtime)
            if getattr(data, "knowledge_context", None):
                result["knowledge"] = data.knowledge_context
            if getattr(data, "page_context", {}).get("knowledge"):
                result["knowledge"] = data.page_context["knowledge"]
            if request.knowledge:
                from diagex.knowledge.resolver import resolve
                result["knowledge_trace"] = result.get("knowledge") or resolve(request.knowledge, {"symbol_detection": "symbol_interpretation", "text_detection": "text_assignment", "line_detection": "line_interpretation", "connection_inference": "connections"}.get(request.stage, request.stage), getattr(getattr(data, "page", None), "page_index", 0), "connector arrow tag line")
            # Detect a source edited while inference was running; never cache it as valid.
            if any(file_hash(assets[n]) != a.sha256 for n, a in request.assets.items()):
                raise ValueError("An input asset changed during execution")
            record = {
                "schema_version": "1.0",
                "stage": request.stage,
                "backend": request.backend,
                "identity_sha256": key,
                "output_sha256": digest(result),
                "output": result,
                "status": "partial" if result.get("contract_failed") else "complete",
            }
            if live:
                from diagex.llm.cost import CostTracker

                usage = CostTracker(pricing=runtime.cost_tracker.pricing)
                usage.steps = copy.deepcopy(runtime.cost_tracker.steps[usage_start:])
                record["usage_estimate"] = usage.summary()
                record["accounting"] = (
                    "Per-stage token/cost estimate only; actual bills and reservations remain in the configured spending ledger."
                )
            atomic_write_json(result_path, record)
            return record, False
        except Exception as exc:
            atomic_write_json(
                folder / "failure.json", {"error": type(exc).__name__, "message": str(exc)}
            )
            raise
