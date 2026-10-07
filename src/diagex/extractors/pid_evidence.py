"""Evidence-first P&ID extraction engine (opt-in ``evidence-v2``)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import fitz

from diagex.config import EFFORT_PROFILES, Config, EffortLevel
from diagex.extractors.evidence_checkpoint import (
    CheckpointStore,
    atomic_write_json,
    atomic_write_text,
    find_resumable_run_with_report,
)
from diagex.extractors.pid import (
    PidExtractionResult,
    _new_run_id,
    _safe_model_name,
    _safe_stem,
    _timestamp,
    _write_confidence_report,
    _write_run_artefacts,
)
from diagex.llm.client import LLMClient, is_non_retryable_api_error
from diagex.llm.cost import CostTracker
from diagex.llm.diagnostics import failure_summary, safe_error
from diagex.vision.connection_inference import (
    PageGraphResult,
    PageLineEvidence,
    classify_page_line_evidence,
    solve_page_graph,
)
from diagex.vision.contextual import (
    ContextualPageResult,
    apply_contextual_results,
    find_contextual_candidates,
    resolve_contextual_page,
)
from diagex.vision.evidence import (
    PageEvidence,
    extract_page_evidence,
    sha256_file,
)
from diagex.vision.fusion import assemble_graph, fuse_objects
from diagex.vision.instance_matching import FUSION_DEPENDENT_STAGES, FUSION_VERSION
from diagex.vision.legend_coverage import LEGEND_GATE_VERSION, legend_coverage_findings
from diagex.vision.legend_models import LegendPack, SymbolStandard
from diagex.vision.line_detection import LINE_DETECTION_VERSION, LineDetectionResult, detect_lines
from diagex.vision.native_text import build_native_text_inventory
from diagex.vision.quality import QualityReport, add_dexpi_results, assess_quality
from diagex.vision.stage_capture import capture_stage
from diagex.vision.symbol_candidates import (
    PERCEPTION_DEPENDENT_STAGES,
    SYMBOL_PERCEPTION_VERSION,
)
from diagex.vision.symbol_detection import detect_symbols
from diagex.vision.symbol_interpretation import DetectionRecord, PerceptionRunGuard, perceive_tile
from diagex.vision.text_assignment import TEXT_ASSIGNMENT_VERSION
from diagex.vision.tiling import AspectAwareStrategy, ownership_core, tile
from diagex.vision.topology import (
    TOPOLOGY_DEPENDENT_STAGES,
    TOPOLOGY_VERSION,
    LegendLineProfile,
    TopologyResult,
    build_page_topology,
    learn_legend_line_profile,
)
from diagex.vision.views import ViewProvider

if TYPE_CHECKING:
    from rich.console import Console


PAGE_GRAPH_PIPELINE_VERSION = "2.1.0"


def run_pid_evidence_extract(
    *,
    diagram: Path,
    symbol_standard: SymbolStandard,
    legend_path: Path | None,
    legend_pages: list[int] | None,
    legend_region: tuple[int, int, int, int, int] | None,
    no_legend: bool,
    legend_key: str | None,
    effort: EffortLevel,
    config: Config,
    persist: bool,
    fresh: bool = False,
    out_path: Path | None,
    confidence_report_path: Path | None,
    console: Console | None,
    stop_after: str = "graph",
) -> PidExtractionResult:
    from rich.console import Console as RichConsole

    from diagex.extractors.pid_legend import (
        LEGEND_EXTRACTOR_VERSION,
        load_run_builtin_pack,
        resolve_evidence_legend,
    )
    from diagex.ui.progress import make_reporter
    from diagex.vision.loader import load

    if stop_after not in {"graph", "detection"}:
        raise ValueError("stop_after must be graph or detection")
    cfg = config
    stem = _safe_stem(diagram)
    started = time.perf_counter()
    source_hash = sha256_file(diagram)
    if cfg.raster_symbol_mode not in {"baseline", "broad_review"}:
        raise ValueError("Unsupported raster symbol mode")
    if cfg.raster_symbol_mode == "broad_review" and cfg.raster_proposals is None:
        raise ValueError("Broad review requires source-bound raster proposals")
    if cfg.raster_ink_filter:
        from diagex.vision.raster_ink import validate_source
        validate_source(diagram)
        if cfg.scan.deskew or cfg.symbol_perception.workflow != "fixed":
            raise ValueError("Raster ink filtering requires fixed inspection with deskew disabled")
    if cfg.raster_proposals is not None:
        if diagram.suffix.lower() == ".pdf":
            from diagex.vision.pdf_raster_guidance import validate_pdf_guidance
            validate_pdf_guidance(cfg.raster_proposals, diagram, cfg, source_sha256=source_hash)
        else:
            from diagex.vision.raster_guidance import validate_guidance
            validate_guidance(cfg.raster_proposals, diagram, source_sha256=source_hash)
        if cfg.scan.deskew or cfg.symbol_perception.workflow != "fixed":
            raise ValueError("Experimental raster guidance requires fixed inspection with deskew disabled")
    vision_model = cfg.llm.vision_model or cfg.llm.model
    reasoning_model = cfg.llm.reasoning_model or cfg.llm.model
    config_hash = _configuration_hash(
        cfg=cfg,
        symbol_standard=symbol_standard,
        vision_model=vision_model,
        reasoning_model=reasoning_model,
        legend_path=legend_path,
        legend_pages=legend_pages,
        legend_region=legend_region,
        no_legend=no_legend,
        legend_key=legend_key,
        effort=effort,
    )


    run_dir: Path | None = None
    run_id = _new_run_id()
    store: CheckpointStore | None = None
    resumed = False
    resume_report: dict[str, Any] = {"reason": "persistence disabled"}
    initial_completed: dict[str, int] = {}
    invalidated_completed: dict[str, int] = {}
    runs_root = cfg.runs_dir / stem
    if persist:
        runs_root.mkdir(parents=True, exist_ok=True)
        if fresh:
            resume_report = {
                "reason": "fresh run requested; checkpoint discovery skipped",
                "considered_runs": 0,
                "fresh_requested": True,
            }
        else:
            store, resume_report = find_resumable_run_with_report(
                runs_root=runs_root,
                source_sha256=source_hash,
                config_sha256=config_hash,
                required_stage_versions={
                    "legend_extraction": LEGEND_EXTRACTOR_VERSION,
                    "symbol_perception": SYMBOL_PERCEPTION_VERSION,
                    "object_fusion": FUSION_VERSION,
                    "port_topology": TOPOLOGY_VERSION,
                },
            )
        if store is not None and (
            store.manifest.status == "complete" or (store.run_dir / "detection.json").exists()
        ):
            source_store = store
            run_dir, run_id = _prepare_v2_run_dir(cfg, stem, vision_model)
            store = CheckpointStore.create(
                run_dir=run_dir,
                source_sha256=source_hash,
                config_sha256=config_hash,
                run_id=run_id,
            )
            store.seed_raw_evidence_from(
                source_store,
                include_contextual=(
                    source_store.manifest.stage_versions.get("object_fusion") == FUSION_VERSION
                    and source_store.manifest.stage_versions.get("page_graph_pipeline")
                    == PAGE_GRAPH_PIPELINE_VERSION
                ),
            )
            resume_report["source_run_dir"] = str(source_store.run_dir)
            resume_report["reason"] = "completed run evidence copied for derived-stage upgrade"
            resumed = True
        elif store is not None:
            run_dir = store.run_dir
            run_id = store.manifest.run_id
            resumed = True
        else:
            run_dir, run_id = _prepare_v2_run_dir(cfg, stem, vision_model)
            store = CheckpointStore.create(
                run_dir=run_dir,
                source_sha256=source_hash,
                config_sha256=config_hash,
                run_id=run_id,
            )

    if cfg.knowledge and run_dir is not None:
        atomic_write_json(run_dir / "knowledge.snapshot.json", cfg.knowledge)
    prior_cost = _load_prior_cost(run_dir) if resumed and run_dir is not None else {}
    if run_dir is not None and cfg.raster_proposals is not None:
        from diagex.vision.raster_guidance import implementation_sha256
        atomic_write_json(run_dir / "raster-proposals.json", {
            **cfg.raster_proposals, "guidance_implementation_sha256": implementation_sha256(),
        })
    if store is not None:
        initial_completed = {
            stage: len(items) for stage, items in sorted(store.manifest.completed.items())
        }
        store.ensure_stage_version(
            "legend_extraction",
            LEGEND_EXTRACTOR_VERSION,
            invalidate=PERCEPTION_DEPENDENT_STAGES,
        )
        store.ensure_stage_version(
            "symbol_perception",
            SYMBOL_PERCEPTION_VERSION,
            invalidate=PERCEPTION_DEPENDENT_STAGES,
        )
        store.ensure_stage_version(
            "page_graph_pipeline",
            PAGE_GRAPH_PIPELINE_VERSION,
            invalidate=(
                "contextual",
                "topology",
                "line_evidence",
                "page_graph",
                "assembly",
                "export",
            ),
        )
        store.ensure_stage_version(
            "text_assignment", TEXT_ASSIGNMENT_VERSION,
            invalidate=("text_assignment", "contextual", "topology", "line_evidence", "page_graph", "assembly", "export"),
        )
        store.ensure_stage_version(
            "object_fusion",
            FUSION_VERSION,
            invalidate=FUSION_DEPENDENT_STAGES,
        )
        store.ensure_stage_version(
            "line_detection", LINE_DETECTION_VERSION,
            invalidate=("line_detection", "topology", "line_evidence", "page_graph", "assembly", "export"),
        )
        store.ensure_stage_version(
            "port_topology",
            TOPOLOGY_VERSION,
            invalidate=TOPOLOGY_DEPENDENT_STAGES,
        )
        remaining = {stage: len(items) for stage, items in sorted(store.manifest.completed.items())}
        invalidated_completed = {
            stage: count - remaining.get(stage, 0)
            for stage, count in initial_completed.items()
            if count > remaining.get(stage, 0)
        }
    source = load(diagram, tiling=cfg.tiling, scan_cfg=cfg.scan)
    progress_console = console or RichConsole()
    _print_reuse_start(
        progress_console,
        resumed=resumed,
        run_dir=run_dir,
        reason=str(resume_report.get("reason") or ""),
        available={stage: len(items) for stage, items in sorted(store.manifest.completed.items())}
        if store is not None
        else {},
        invalidated=invalidated_completed,
    )
    if run_dir is not None:
        atomic_write_json(
            run_dir / "reuse.report.json",
            {
                "mode": "resumed" if resumed else "new",
                "fresh_requested": fresh,
                "run_dir": str(run_dir),
                "decision": resume_report,
                "initial_completed_by_stage": initial_completed,
                "available_after_invalidation_by_stage": {
                    stage: len(items) for stage, items in sorted(store.manifest.completed.items())
                }
                if store is not None
                else {},
                "invalidated_by_stage": invalidated_completed,
                "status": "running",
            },
        )
    reporter_factory = lambda: make_reporter(progress_console, effort=effort)  # noqa: E731

    cost = CostTracker(pricing=cfg.pricing)
    vision_cfg = replace(
        cfg.llm,
        model=vision_model,
        reasoning_mode="disabled",
    )
    reasoning_mode = cfg.llm.reasoning_mode
    if reasoning_mode == "auto":
        reasoning_mode = "enabled"
    reasoning_cfg = replace(
        cfg.llm,
        model=reasoning_model,
        reasoning_mode=reasoning_mode,
    )
    vision_llm = LLMClient(vision_cfg, budgets=cfg.budgets)
    reasoning_llm = LLMClient(reasoning_cfg, budgets=cfg.budgets)
    escalation_llm = LLMClient(replace(cfg.llm, model=cfg.llm.escalation_model, reasoning_mode="enabled"), budgets=cfg.budgets) if cfg.llm.escalation_model else None
    vision_llm.reset_retry_counter()
    reasoning_llm.reset_retry_counter()


    # 1. Inspect pages and persist immutable native evidence.
    evidence_pages, page_states = _inspect_pages(
        source=source,
        diagram=diagram,
        store=store,
        run_dir=run_dir,
        reporter=reporter_factory(),
    )
    if cfg.raster_proposals is not None and cfg.raster_proposals.get("format") == "pdf_page_proposals_v1":
        from diagex.vision.pdf_raster_guidance import route_guided_pages

        routing = route_guided_pages(evidence_pages, cfg.raster_proposals, legend_pages=legend_pages)
        if routing and run_dir is not None:
            atomic_write_json(run_dir / "pdf-proposal-routing.json", {
                "source_sha256": source_hash, "changes": routing,
                "scope": "Fail-open routing for explicitly guided scans; native text and geometry unchanged",
            })
            changed_pages = {row["page_index"] for row in routing}
            for page in evidence_pages:
                if page.page_index in changed_pages:
                    atomic_write_json(run_dir / "evidence" / f"page-{page.page_index + 1:04d}.json",
                                      page.model_dump(mode="json"))
    # 2. Resolve only explicitly selected or deterministically classified legend pages.
    detected_legends = [page.page_index for page in evidence_pages if page.role == "legend"]

    legend_pack: LegendPack
    legend_source = ""
    legend_reporter = reporter_factory()
    try:
        with legend_reporter:
            legend_reporter.on_phase_start(
                name="v2 legend resolution", total_items=len(detected_legends) or 1
            )
            routed = resolve_evidence_legend(
                source=source,
                pages=evidence_pages,
                symbol_standard=symbol_standard,
                cfg=cfg,
                client=vision_llm,
                cost_tracker=cost,
                legend_path=legend_path,
                legend_pages=legend_pages,
                legend_region=legend_region,
                no_legend=no_legend,
                legend_key=legend_key,
                runs_dir_for_stem=runs_root if persist else None,
                reporter=legend_reporter,
                fresh=fresh,
            )
            resolution = routed.resolution
            legend_pack = resolution.pack
            legend_source = resolution.source
            legend_reporter.on_phase_end(
                detail=f"{len(legend_pack.entries)} entries · source={legend_source}"
            )
    except Exception as exc:  # noqa: BLE001 - built-in pack is a safe fallback
        if is_non_retryable_api_error(exc):
            raise
        legend_pack = load_run_builtin_pack(symbol_standard, cfg)
        legend_source = f"fallback_builtin(error={exc!r})"
    _checkpoint_cost(run_dir, prior_cost, cost)

    legend_failure = _legend_prerequisite_error(legend_pack, legend_source)
    if run_dir is not None:
        atomic_write_json(run_dir / "legend.warnings.json", {
            "gate_version": LEGEND_GATE_VERSION,
            "findings": legend_coverage_findings(legend_pack),
        })
    if store is not None:
        store.manifest.errors = [e for e in store.manifest.errors if e.stage != "legend"]
        if legend_failure:
            store.mark_error("legend", "coverage", legend_failure)
        else:
            store.save()

    # Persist uncertain definitions as observations, not interpretation rules.
    interpretation_entries = [
        entry
        for entry in legend_pack.entries
        if entry.source == "customer_override"
        or entry.attributes.get("row_status") not in {"uncertain", "reject"}
    ]
    interpretation_pack = legend_pack.model_copy(update={"entries": interpretation_entries})
    legend_summary = [
        {
            "label": entry.label,
            "kind": entry.kind,
            "symbol_class": entry.symbol_class,
            "description": entry.description or "",
            "attributes": dict(entry.attributes),
        }
        for entry in interpretation_entries
    ]
    legend_visual_entries = [entry.model_dump(mode="json") for entry in interpretation_entries]
    if store is not None:
        # A repaired partial legend changes the evidence supplied to symbols
        # even when source/model/implementation versions remain identical.
        content_hash = hashlib.sha256(
            json.dumps(legend_visual_entries, sort_keys=True).encode()
        ).hexdigest()
        store.ensure_stage_version(
            "legend_content", content_hash, invalidate=PERCEPTION_DEPENDENT_STAGES
        )
    fallback_legend_line_images = [
        {
            "label": entry.label,
            "description": entry.description or "",
            "image_b64": entry.image_b64,
        }
        for entry in interpretation_entries
        if entry.kind == "line" and entry.image_b64
    ]
    native_legend_line_images = _native_line_legend_images(
        source=source,
        pages=evidence_pages,
        legend_pack=interpretation_pack,
    )
    native_labels = {_legend_text_key(entry["label"]) for entry in native_legend_line_images}
    legend_line_images = [
        *native_legend_line_images,
        *(
            entry
            for entry in fallback_legend_line_images
            if _legend_text_key(entry["label"]) not in native_labels
        ),
    ]

    # 3. Deterministic crop coverage and raw instance classification.
    perception_stop_reason = None
    detections, per_page_status, perception_call_counts, perception_stop_reason = (
        _run_perception(
            source=source,
            pages=evidence_pages,
            cfg=cfg,
            client=vision_llm,
            cost=cost,
            reporter=reporter_factory(),
            store=store,
            legend_summary=legend_visual_entries,
            run_dir=run_dir,
            prior_cost=prior_cost,
            prerequisite_error=legend_failure,
            escalation_client=escalation_llm,
        )
    )

    for region_coverage in legend_pack.coverage:
        if region_coverage.status != "complete":
            per_page_status[region_coverage.page_index] = "partial"

    if run_dir is not None:
        from diagex.extractors.detection_artifacts import write_detection_bundle

        audit_path = run_dir / "perception.review.json"
        reviews = (
            json.loads(audit_path.read_text()).get("reviews", []) if audit_path.exists() else []
        )
        write_detection_bundle(
            run_dir,
            source_hash=source_hash,
            pages=evidence_pages,
            detections=detections,
            legend_pack=legend_pack,
            per_page_status=per_page_status,
            candidates=[
                c.model_dump(mode="json")
                for p in evidence_pages
                if p.role == "pid"
                for c in detect_symbols(page=p).native_candidates
            ],
            reviews=reviews,
        )
        atomic_write_json(run_dir / "legend.json", legend_pack.model_dump(mode="json"))
    if cfg.knowledge and run_dir is not None:
        from diagex.knowledge.resolver import resolve
        atomic_write_json(run_dir / "knowledge.snapshot.json", cfg.knowledge)
        atomic_write_json(run_dir / "knowledge.stages.json", {
            task: [resolve(cfg.knowledge, task, page.page_index, "connector arrow " + " ".join(s.text for s in page.text_spans), legend_summary) for page in evidence_pages]
            for task in ("symbol_interpretation", "text_assignment", "line_interpretation", "connections", "review")
        })

    if stop_after == "detection" or perception_stop_reason:
        from diagex.vision.models import ReconciledGraph

        current_cost = cost.summary()
        current_cost["tool_call_counts"] = perception_call_counts
        current_cost["n_tool_calls"] = sum(perception_call_counts.values())
        current_cost["wall_clock_s"] = round(time.perf_counter() - started, 3)
        current_cost["retries"] = vision_llm.retries_total
        cost_summary = _merge_cost_summaries(prior_cost, current_cost)
        quality_status = (
            "partial" if any(v != "ok" for v in per_page_status.values()) else "needs_review"
        )
        if run_dir is not None:
            atomic_write_json(run_dir / "cost.json", cost_summary)
            atomic_write_json(
                run_dir / "result.json",
                {
                    "run_id": run_id,
                    "engine": "evidence-v2",
                    "model": vision_model,
                    "workflow_stage": "detection",
                    "quality_status": quality_status,
                    "detection_count": len(detections),
                    "legend_entry_count": len(legend_pack.entries),
                    "stop_reason": perception_stop_reason,
                },
            )
            if store is not None:
                store.manifest.pause_reason = perception_stop_reason
                store.set_status(
                    "paused" if perception_stop_reason else "partial"
                    if store.manifest.errors
                    or perception_stop_reason
                    or quality_status == "partial"
                    else "complete"
                )
            atomic_write_json(
                run_dir / "reuse.report.json",
                {
                    "mode": "resumed" if resumed else "new",
                    "decision": resume_report,
                    "workflow_stage": "detection",
                    "status": quality_status,
                },
            )
        return PidExtractionResult(
            diagram_stem=stem,
            effort=effort,
            model=vision_model,
            graph=ReconciledGraph(source_path=diagram.name),
            dexpi_json_path=None,
            dexpi_stats={"detection_count": len(detections)},
            cost_summary=cost_summary,
            legend_source=legend_source,
            legend_entry_count=len(legend_pack.entries),
            run_dir=run_dir,
            run_id=run_id,
            engine="evidence-v2",
            quality_status=quality_status,
            workflow_stage="detection",
        )

    # 4. Fuse overlapping object observations exactly once.
    def record_text_assignment(request, result):
        capture_stage(run_dir, "text_assignment", "geometry", "objects", request["inputs"], output=result)
        if store is not None:
            store.write_json_artifact("text_assignment", "request", request)
            store.write_json_artifact("text_assignment", "output", result)

    object_fusion = fuse_objects(
        knowledge=cfg.knowledge,
        on_text_assignment=record_text_assignment,
        source_name=diagram.name,
        pages=evidence_pages,
        detections=detections,
        per_page_status=per_page_status,
        legend_pack=interpretation_pack,
    )
    if store is not None:
        store.write_json_artifact(
            "assembly",
            "objects",
            object_fusion.graph.model_dump(
                mode="json", include={"source_path", "nodes", "assemblies", "text_bindings"}
            ),
        )

    # 5. Resolve only ambiguous small glyphs attached to valve bodies, using
    # project-specific legend crops and page context before topology can snap
    # symbol strokes as process connections.
    contextual_results, contextual_calls = _run_contextual_resolution(
        source=source,
        pages=evidence_pages,
        nodes=object_fusion.graph.nodes,
        legend_entries=legend_visual_entries,
        client=vision_llm,
        cost=cost,
        reporter=reporter_factory(),
        store=store,
        per_page_status=per_page_status,
        run_dir=run_dir,
        prior_cost=prior_cost,
    )
    if contextual_results:
        object_fusion = apply_contextual_results(object_fusion, contextual_results)
    if store is not None:
        store.write_json_artifact(
            "assembly",
            "contextual_objects",
            object_fusion.graph.model_dump(
                mode="json", include={"source_path", "nodes", "assemblies", "text_bindings"}
            ),
        )

    # 6. Build deterministic topology candidates from PDF/raster line evidence.
    # Native inspection survives legend upgrades. A learned line profile must
    # instead follow the exact definitions used by this run/review snapshot.
    profile_item = (
        "legend_line_profile-"
        + hashlib.sha256(interpretation_pack.model_dump_json().encode("utf-8")).hexdigest()[:16]
    )
    if store is not None and store.is_done("inspection", profile_item):
        legend_line_profile = LegendLineProfile.model_validate(
            store.read_json_artifact("inspection", profile_item)
        )
    else:
        legend_line_profile = learn_legend_line_profile(
            pages=evidence_pages,
            legend_pack=interpretation_pack,
        )
    if store is not None and not store.is_done("inspection", profile_item):
        store.write_json_artifact(
            "inspection",
            profile_item,
            legend_line_profile.model_dump(mode="json"),
        )
    topology_results = _run_topology(
        source=source,
        pages=evidence_pages,
        nodes=object_fusion.graph.nodes,
        legend_line_profile=legend_line_profile,
        reporter=reporter_factory(),
        store=store,
        per_page_status=per_page_status,
    )
    # 7. Collect bounded, non-thinking visual facts for recovered line candidates.
    line_evidence, line_evidence_calls = _run_line_evidence(
        knowledge=cfg.knowledge,
        source=source,
        pages=evidence_pages,
        topology=topology_results,
        nodes=object_fusion.graph.nodes,
        client=vision_llm,
        cost=cost,
        reporter=reporter_factory(),
        store=store,
        per_page_status=per_page_status,
        run_dir=run_dir,
        prior_cost=prior_cost,
        legend_line_images=legend_line_images,
    )
    # 8. Resolve only the remaining semantic relationships from structured evidence.
    page_graph_results, page_graph_calls = _run_page_graphs(
        knowledge=cfg.knowledge,
        source=source,
        pages=evidence_pages,
        topology=topology_results,
        nodes=object_fusion.graph.nodes,
        client=reasoning_llm,
        cost=cost,
        reporter=reporter_factory(),
        store=store,
        per_page_status=per_page_status,
        run_dir=run_dir,
        prior_cost=prior_cost,
        effort=effort,
        legend_summary=legend_summary,
        legend_line_images=legend_line_images,
        line_evidence=line_evidence,
        include_visual_context=False,
        process_context=cfg.process_context,
        inspection_client=vision_llm,
        escalation_client=escalation_llm,
    )
    fusion = assemble_graph(
        objects=object_fusion,
        pages=evidence_pages,
        topology=topology_results,
        page_graph_results=page_graph_results,
        per_page_status=per_page_status,
    )
    graph = fusion.graph
    if run_dir is not None:
        atomic_write_json(run_dir / "hypotheses.json", {
            "status": "review_only", "hypotheses": [h for result in page_graph_results for h in result.hypotheses],
            "process_context": cfg.process_context,
        })
    _checkpoint_cost(run_dir, prior_cost, cost)

    native_text_inventory = build_native_text_inventory(
        pages=evidence_pages,
        graph=graph,
        legend_pack=interpretation_pack,
    )
    quality = assess_quality(
        graph=graph,
        pages=evidence_pages,
        topology=topology_results,
        native_text_inventory=native_text_inventory,
    )

    # 9. Existing deterministic DEXPI build, semantic validation, JSON and XML.
    dexpi_stats: dict[str, Any] = {}
    dexpi_issues: list[str] = []
    validation_issues: list[dict[str, Any]] = []
    dexpi_model = None
    dexpi_json_path: Path | None = None
    dexpi_xml_path: Path | None = None
    export_reporter = reporter_factory()
    with export_reporter:
        export_reporter.on_phase_start(name="v2 validate and export", total_items=3)
        try:
            from diagex.extractors.dexpi_builder import build_dexpi, serialize_model, validate_model

            export_reporter.on_phase_item_start(item=1, total_items=3, label="build DEXPI")
            build = build_dexpi(graph)
            dexpi_model = build.model
            dexpi_stats = dict(build.stats)
            dexpi_issues = list(build.issues)
            export_reporter.on_phase_item_end(
                detail=f"{dexpi_stats.get('segment_count', 0)} segments"
            )

            export_reporter.on_phase_item_start(item=2, total_items=3, label="semantic validation")
            validation_issues = list(validate_model(dexpi_model))
            export_reporter.on_phase_item_end(
                detail=f"{len(validation_issues)} issue(s)",
                is_error=bool(validation_issues),
            )

            export_reporter.on_phase_item_start(item=3, total_items=3, label="write JSON and XML")
            if persist:
                if out_path is not None:
                    output = Path(out_path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    serial_stem = output.name[:-5] if output.name.endswith(".json") else output.name
                    dexpi_json_path = serialize_model(dexpi_model, output.parent, serial_stem)
                elif run_dir is not None:
                    dexpi_json_path = serialize_model(dexpi_model, run_dir, "pid.dexpi")
                if run_dir is not None:
                    try:
                        from diagex.dexpi import validate as dexpi_validate
                        from diagex.dexpi.xml_io import dump as dump_xml
                        from diagex.dexpi.xml_io import load as load_xml

                        dexpi_xml_path = dump_xml(dexpi_model, run_dir / "pid.dexpi.xml")
                        # JSON validation alone cannot catch duplicate XML IDs,
                        # unresolved References, or envelope/schema failures.
                        # Parse the exact written artifact and report only
                        # error-severity findings as completion blockers.
                        parsed_xml_model = load_xml(dexpi_xml_path)
                        for issue in dexpi_validate.semantic_validate(parsed_xml_model):
                            record = {
                                "path": issue.path or "<xml-semantic>",
                                "msg": f"{issue.rule_id}: {issue.message}",
                            }
                            if issue.severity == "error":
                                validation_issues.append(record)
                            else:
                                dexpi_issues.append(
                                    f"DEXPI XML warning at {record['path']}: {record['msg']}"
                                )
                        try:
                            xsd_issues = dexpi_validate.xsd_validate(dexpi_xml_path)
                        except dexpi_validate.XmlschemaUnavailableError:
                            dexpi_issues.append(
                                "DEXPI XML XSD validation skipped: install the 'validate' extra"
                            )
                        else:
                            validation_issues.extend(
                                {
                                    "path": issue.path or "<xml-xsd>",
                                    "msg": f"{issue.rule_id}: {issue.message}",
                                }
                                for issue in xsd_issues
                                if issue.severity == "error"
                            )
                    except Exception as exc:  # noqa: BLE001 - JSON remains usable
                        validation_issues.append({"path": "<xml-roundtrip>", "msg": repr(exc)})
                        dexpi_issues.append(f"DEXPI XML serialise/validate failed: {exc!r}")
            export_reporter.on_phase_item_end(
                detail=(
                    f"json={dexpi_json_path.name if dexpi_json_path else 'none'}, "
                    f"xml={dexpi_xml_path.name if dexpi_xml_path else 'none'}"
                )
            )
        except Exception as exc:  # noqa: BLE001
            dexpi_issues.append(f"dexpi build failed: {exc!r}")
            export_reporter.on_phase_item_end(detail=repr(exc), is_error=True)
        export_reporter.on_phase_end(detail="complete")

    quality = add_dexpi_results(
        quality,
        stats=dexpi_stats,
        build_issues=dexpi_issues,
        validation_issues=validation_issues,
    )

    current_cost = cost.summary()
    current_cost["retries"] = vision_llm.retries_total + reasoning_llm.retries_total
    current_cost["tool_call_counts"] = {
        **perception_call_counts,
        "submit_contextual_symbol_corrections": contextual_calls,
        "submit_line_evidence": line_evidence_calls,
        "submit_page_graph": page_graph_calls,
    }
    current_cost["n_tool_calls"] = (
        sum(perception_call_counts.values())
        + contextual_calls
        + line_evidence_calls
        + page_graph_calls
    )
    current_cost["wall_clock_s"] = round(time.perf_counter() - started, 3)
    cost_summary = _merge_cost_summaries(prior_cost, current_cost)

    if run_dir is not None:
        atomic_write_text(
            run_dir / "evidence" / "native-text-inventory.json",
            native_text_inventory.model_dump_json(indent=2),
        )
        _write_run_artefacts(
            run_dir=run_dir,
            runs_root=run_dir.parent,
            run_id=run_id,
            stem=stem,
            effort=effort,
            model=f"vision={vision_model}; reasoning={reasoning_model}",
            graph=graph,
            cost_summary=cost_summary,
            dexpi_stats=dexpi_stats,
            dexpi_issues=dexpi_issues,
            validation_issues=validation_issues,
            dexpi_json_path=dexpi_json_path,
            legend_pack=legend_pack,
            legend_source_tag=legend_source,
            states=page_states,
            per_page_status=per_page_status,
            confidence_report_path=confidence_report_path,
            engine="evidence-v2",
        )
        atomic_write_text(
            run_dir / "quality.report.json",
            quality.model_dump_json(indent=2),
        )
        _augment_public_manifests(
            run_dir=run_dir,
            pages=evidence_pages,
            quality=quality,
            dexpi_xml_path=dexpi_xml_path,
            resumed=resumed,
        )
        if store is not None:
            store.write_json_artifact(
                "export",
                "result",
                {
                    "dexpi_json_path": str(dexpi_json_path) if dexpi_json_path else None,
                    "dexpi_xml_path": str(dexpi_xml_path) if dexpi_xml_path else None,
                    "build_issues": dexpi_issues,
                    "validation_issues": validation_issues,
                },
            )
        atomic_write_text(
            run_dir / "evidence" / "document.json",
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "source_name": diagram.name,
                    "source_sha256": source_hash,
                    "pages": [
                        {
                            "page_index": page.page_index,
                            "role": page.role,
                            "role_confidence": page.role_confidence,
                            "role_reason": page.role_reason,
                            "text_count": len(page.text_spans),
                            "path_count": len(page.paths),
                        }
                        for page in evidence_pages
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        if store is not None:
            store.write_json_artifact("assembly", "graph", graph.model_dump(mode="json"))
            if store.manifest.errors:
                store.set_status("partial")
            else:
                store.set_status("complete")
            reuse_summary = store.reuse_summary()
            atomic_write_json(
                run_dir / "reuse.report.json",
                {
                    "mode": "resumed" if resumed else "new",
                    "fresh_requested": fresh,
                    "run_dir": str(run_dir),
                    "decision": resume_report,
                    "initial_completed_by_stage": initial_completed,
                    "invalidated_by_stage": invalidated_completed,
                    **reuse_summary,
                    "legend_source": legend_source,
                    "status": store.manifest.status,
                },
            )
            _print_reuse_end(progress_console, summary=reuse_summary)
    elif confidence_report_path is not None:
        _write_confidence_report(
            Path(confidence_report_path),
            stem=stem,
            run_id=run_id,
            effort=effort,
            cost_summary=cost_summary,
            legend_source_tag=legend_source,
            legend_entry_count=len(legend_pack.entries),
            dexpi_stats=dexpi_stats,
            dexpi_issues=dexpi_issues,
            validation_issues=validation_issues,
            graph=graph,
            per_page_status=per_page_status,
        )

    return PidExtractionResult(
        diagram_stem=stem,
        effort=effort,
        model=f"vision={vision_model}; reasoning={reasoning_model}",
        graph=graph,
        dexpi_json_path=dexpi_json_path,
        dexpi_stats=dexpi_stats,
        dexpi_issues=dexpi_issues,
        validation_issues=validation_issues,
        legend_source=legend_source,
        legend_entry_count=len(legend_pack.entries),
        cost_summary=cost_summary,
        run_dir=run_dir,
        run_id=run_id,
        engine="evidence-v2",
        quality_status=quality.status,
    )


def _inspect_pages(
    *,
    source: Any,
    diagram: Path,
    store: CheckpointStore | None,
    run_dir: Path | None,
    reporter: Any,
) -> tuple[list[PageEvidence], list[Any]]:
    from diagex.vision.loader import iter_pages

    evidence: list[PageEvidence] = []
    states: list[Any] = []
    pdf_doc = fitz.open(diagram) if diagram.suffix.lower() == ".pdf" else None
    total = int(source.metadata.get("page_count", 0) or 0)
    try:
        with reporter:
            reporter.on_phase_start(name="v2 page inspection", total_items=total)
            for page in iter_pages(source):
                item = f"page-{page.page_index + 1:04d}"
                reporter.on_phase_item_start(
                    item=page.page_index + 1,
                    total_items=total,
                    label=f"inspect page {page.page_index + 1}",
                )
                public_path = (
                    run_dir / "evidence" / f"page-{page.page_index + 1:04d}.json"
                    if run_dir is not None
                    else None
                )
                if (
                    store is not None
                    and store.is_done("inspection", item)
                    and public_path is not None
                    and public_path.is_file()
                ):
                    page_evidence = PageEvidence.model_validate_json(
                        public_path.read_text(encoding="utf-8")
                    )
                    if (
                        pdf_doc is not None
                        and pdf_doc[page.page_index].rotation
                        and page_evidence.native_coordinate_frame != "rendered_page"
                    ):
                        page_evidence = extract_page_evidence(
                            page=page, source_path=diagram, pdf_page=pdf_doc[page.page_index]
                        )
                        atomic_write_text(public_path, page_evidence.model_dump_json(indent=2))
                    else:
                        store.record_reuse("inspection", item)
                else:
                    page_evidence = extract_page_evidence(
                        page=page,
                        source_path=diagram,
                        pdf_page=pdf_doc[page.page_index] if pdf_doc is not None else None,
                    )
                    if public_path is not None:
                        atomic_write_text(public_path, page_evidence.model_dump_json(indent=2))
                    if store is not None:
                        store.mark_done("inspection", item)
                capture_stage(run_dir, "text_detection", "native", item,
                    {"page": page_evidence.model_dump(mode="json")},
                    output={"page_index": page_evidence.page_index,
                            "text_spans": [t.model_dump(mode="json") for t in page_evidence.text_spans],
                            "backend": "native_pdf", "warnings": []})
                evidence.append(page_evidence)
                states.append(
                    SimpleNamespace(
                        page=page.model_copy(update={"image": None}),
                        transcript=[],
                    )
                )
                reporter.on_phase_item_end(
                    detail=(
                        f"role={page_evidence.role} ({page_evidence.role_confidence}), "
                        f"text={len(page_evidence.text_spans)}, paths={len(page_evidence.paths)}"
                    )
                )
            reporter.on_phase_end(detail=f"{len(evidence)} pages classified")
    finally:
        if pdf_doc is not None:
            pdf_doc.close()
    return evidence, states


def _legend_prerequisite_error(pack, source):
    failed = [c for c in legend_coverage_findings(pack) if c["blocks_symbol_extraction"]]
    if failed:
        return f"Symbol extraction paused: {len(failed)} source legend rows/regions were not successfully inspected. Completed legend rows are saved; retry the unresolved rows before symbol extraction."
    if source.startswith("fallback_builtin(error="):
        return "Symbol extraction paused because source legend extraction failed; built-in definitions do not replace the missing source review."
    return None


def _run_perception(
    *,
    source: Any,
    pages: list[PageEvidence],
    cfg: Config,
    client: LLMClient,
    cost: CostTracker,
    reporter: Any,
    store: CheckpointStore | None,
    legend_summary: list[dict[str, Any]],
    run_dir: Path | None,
    prior_cost: dict[str, Any],
    prerequisite_error: str | None = None,
    escalation_client: LLMClient | None = None,
) -> tuple[list[DetectionRecord], dict[int, str], dict[str, int], str | None]:
    from diagex.vision.loader import iter_pages

    evidence_by_index = {page.page_index: page for page in pages}
    pid_pages = [page for page in pages if page.role == "pid"]
    strategy = AspectAwareStrategy(
        max_tokens_per_tile=cfg.tiling.max_tokens_per_tile,
        overlap_frac=cfg.tiling.overlap_frac,
        token_per_pixel=cfg.tiling.token_per_pixel,
    )
    detections: list[DetectionRecord] = []
    statuses = {page.page_index: "ok" for page in pages}
    page_error_counts: dict[int, int] = {page.page_index: 0 for page in pid_pages}
    page_view_counts: dict[int, int] = {page.page_index: 0 for page in pid_pages}
    call_counts = {"submit_pid_objects": 0}
    perception_reviews: list[dict[str, Any]] = []
    request_failures: list[dict[str, Any]] = []
    guard = PerceptionRunGuard(cfg.symbol_perception)
    deadline = time.monotonic() + cfg.symbol_perception.run_timeout_s
    stop_reason: str | None = prerequisite_error
    if run_dir is not None:
        (run_dir / "perception.stop.json").unlink(missing_ok=True)
        atomic_write_json(run_dir / "request.errors.json", {
            "schema_version": "1.0.0", "scope": "current_execution", "failures": [],
        })
    if store is not None:
        store.manifest.errors = [
            e
            for e in store.manifest.errors
            if not (e.stage == "perception" and e.item == "run_guard")
        ]
        store.save()

    def next_step() -> int:
        # Parsing can fail after the provider has returned billable usage.  In
        # that case CostTracker already contains the attempt even though no
        # structured tool call was accepted, so use recorded steps rather than
        # successful-call counters to avoid duplicate step identifiers.
        return max((row.step for row in cost.steps), default=0) + 1

    def record_model_attempt(tool_name="submit_pid_objects") -> None:
        call_counts[tool_name] = call_counts.get(tool_name, 0) + 1

    planned_total = sum(
        len(tile(rendered_page, strategy))
        for rendered_page in iter_pages(source)
        if evidence_by_index[rendered_page.page_index].role == "pid"
    )

    progress_item = 0
    with reporter:
        reporter.on_phase_start(name="v2 fixed-crop object perception", total_items=planned_total)
        for rendered_page in iter_pages(source):
            if stop_reason:
                break
            evidence = evidence_by_index[rendered_page.page_index]
            if evidence.role != "pid":
                continue
            fixed_tiles = tile(rendered_page, strategy)
            detection = detect_symbols(page=evidence)
            candidates = detection.native_candidates
            capture_stage(run_dir, "symbol_detection", "native", f"page-{evidence.page_index + 1:04d}",
                {"page": evidence.model_dump(mode="json")}, output=detection.model_dump(mode="json"))
            if store is not None:
                store.write_json_artifact("symbol_detection", f"page-{evidence.page_index + 1:04d}",
                    {"input": evidence.model_dump(mode="json"), "output": detection.model_dump(mode="json")})
            page_view_counts[evidence.page_index] = len(fixed_tiles)
            provider = ViewProvider(rendered_page, fixed_tiles)
            raster_hints = None
            raster_scale = (1, 1)
            if cfg.raster_proposals is not None:
                if cfg.raster_proposals.get("format") == "pdf_page_proposals_v1":
                    from diagex.vision.pdf_raster_guidance import page_proposals
                    raster_hints = page_proposals(cfg.raster_proposals, rendered_page)
                else:
                    width, height = cfg.raster_proposals["size"]
                    raster_scale = (width / rendered_page.width, height / rendered_page.height)
                    raster_hints = cfg.raster_proposals["predictions"]
            for current_tile in fixed_tiles:
                if time.monotonic() >= deadline:
                    stop_reason = "Symbol extraction stopped early at its configured time limit; completed crops are saved."
                    break
                progress_item += 1
                item = f"p{rendered_page.page_index + 1:04d}__{current_tile.id}"
                reporter.on_phase_item_start(
                    item=progress_item,
                    total_items=planned_total,
                    label=(f"page {rendered_page.page_index + 1} · {current_tile.id}"),
                )
                failed: bool | None = None
                diagnostic_events: list[dict[str, Any]] = []

                def record_diagnostic(
                    event: dict[str, Any],
                    events: list[dict[str, Any]] = diagnostic_events,
                    artifact_item: str = item,
                    tile_id: str = current_tile.id,
                ) -> None:
                    events.append(event)
                    if store is not None:
                        atomic_write_json(
                            store.artifact_path("perception_diagnostics", artifact_item),
                            {"tile_id": tile_id, "events": events},
                        )

                try:
                    if store is not None and store.is_done("perception", item):
                        saved = store.read_json_artifact("perception", item)
                        failed = (
                            saved.get("contract_failed") if saved.get("native_candidates") else None
                        )
                        tile_detections = [
                            DetectionRecord.model_validate(value)
                            for value in saved.get("detections", [])
                        ]
                        detail = f"checkpoint · {len(tile_detections)} objects"
                        perception_reviews.extend(saved.get("candidate_reviews", []))
                        rejected_count = len(saved.get("rejected_objects", []))
                        if rejected_count:
                            page_error_counts[evidence.page_index] += 1
                            detail += f", {rejected_count} malformed object(s) skipped"
                    else:
                        view_image, view_info = provider.get_tile(current_tile.id)
                        core = ownership_core(current_tile, fixed_tiles)
                        page_context = {"coverage": "deterministic fixed grid"}
                        if cfg.knowledge:
                            from diagex.knowledge.resolver import resolve_symbol_context
                            page_context["knowledge"] = resolve_symbol_context(
                                cfg.knowledge, evidence, candidates, legend_summary, core
                            )
                        if raster_hints is not None:
                            from diagex.vision.raster_guidance import view_guidance
                            page_context.update(view_guidance(
                                raster_hints, view_info.page_bbox, raster_scale,
                            ))
                        tile_perception = perceive_tile
                        if cfg.raster_symbol_mode == "broad_review" and raster_hints is not None:
                            from diagex.vision.raster_pipeline import perceive_broad_with_semantics
                            tile_perception = perceive_broad_with_semantics
                        capture_stage(run_dir, "symbol_interpretation",
                            "broad_review" if cfg.raster_symbol_mode == "broad_review" and raster_hints is not None else "baseline",
                            item, {"page": evidence.model_dump(mode="json"),
                                   "tile": current_tile.model_dump(mode="json", exclude={"image"}),
                                   "ownership_bbox": core.model_dump(mode="json"),
                                   "candidates": [c.model_dump(mode="json") for c in candidates],
                                   "legend_entries": legend_summary, "page_context": page_context,
                                   "policy": asdict(cfg.symbol_perception)},
                            image=rendered_page.image, model=cfg.llm.vision_model or cfg.llm.model,
                            transport=cfg.llm.transport)
                        outcome = tile_perception(
                            client=client,
                            cost_tracker=cost,
                            reporter=reporter,
                            page=evidence,
                            tile=current_tile,
                            view_image=view_image,
                            view_info=view_info,
                            ownership_bbox=core,
                            legend_summary=legend_summary,
                            step=next_step(),
                            page_context=page_context,
                            overview_image=provider.get_overview()[0] if cfg.symbol_perception.workflow == "adaptive" else None,
                            region_provider=provider.get_region,
                            escalation_client=escalation_client,
                            on_attempt=record_model_attempt,
                            candidates=candidates,
                            reasoning_mode=cfg.llm.reasoning_mode,
                            policy=cfg.symbol_perception,
                            deadline=deadline,
                            on_diagnostic=record_diagnostic,
                        )
                        failed = outcome.contract_failed if outcome.candidates else None
                        tile_detections = outcome.detections
                        batch = outcome.batch
                        for review in batch.candidate_reviews:
                            review.update(page_index=evidence.page_index, tile_id=current_tile.id)
                        perception_reviews.extend(batch.candidate_reviews)
                        if store is not None:
                            store.invalidate(
                                "contextual",
                                "topology",
                                "line_evidence",
                                "page_graph",
                                "assembly",
                                "export",
                            )
                            store.write_json_artifact(
                                "perception",
                                item,
                                {
                                    "detections": [
                                        value.model_dump(mode="json") for value in tile_detections
                                    ],
                                    "knowledge": page_context.get("knowledge", {}),
                                    "observations": batch.observations,
                                    "uncertainties": batch.uncertainties,
                                    "rejected_objects": batch.rejected_objects,
                                    "symbol_perception_version": SYMBOL_PERCEPTION_VERSION,
                                    "native_candidates": [
                                        c.model_dump(mode="json") for c in outcome.candidates
                                    ],
                                    "candidate_reviews": batch.candidate_reviews,
                                    "ownership_core": core.model_dump(mode="json"),
                                    "reason": "deterministic fixed grid",
                                    "model_attempts": outcome.attempts,
                                    "format_recovery": outcome.recovery_diagnostics,
                                    "contract_failed": outcome.contract_failed,
                                },
                            )
                        _checkpoint_cost(run_dir, prior_cost, cost)
                        detail = f"{len(tile_detections)} objects"
                        if outcome.attempts > 1:
                            detail += f" · {outcome.attempts} attempts"
                        if batch.rejected_objects:
                            page_error_counts[evidence.page_index] += 1
                            detail += f", {len(batch.rejected_objects)} malformed object(s) skipped"
                        if any(r.get("status") == "unreviewed" for r in batch.candidate_reviews):
                            page_error_counts[evidence.page_index] += 1
                    detections.extend(tile_detections)
                    reporter.on_phase_item_end(detail=detail)
                except Exception as exc:  # noqa: BLE001 - preserve other tiles and resume later
                    failed = True
                    page_error_counts[evidence.page_index] += 1
                    failure = {**failure_summary(diagnostic_events, exc),
                               "page_index": evidence.page_index, "tile_id": current_tile.id,
                               "timestamp": time.time()}
                    request_failures.append(failure)
                    if run_dir is not None:
                        atomic_write_json(run_dir / "request.errors.json", {
                            "schema_version": "1.0.0", "scope": "current_execution",
                            "failures": request_failures,
                        })
                    from diagex.vision.symbol_interpretation import _bbox_center_is_owned
                    failed_core = ownership_core(current_tile, fixed_tiles)
                    perception_reviews.extend({
                        "candidate_id": c.id, "page_index": evidence.page_index,
                        "bbox": c.bbox.model_dump(mode="json"), "source_path_ids": c.source_path_ids,
                        "tile_id": current_tile.id, "status": "unreviewed",
                        "reason": failure["label"], "request_failure": failure,
                    } for c in candidates if _bbox_center_is_owned(c.bbox, failed_core, evidence))
                    if store is not None:
                        atomic_write_json(store.artifact_path("perception_errors", item), {
                            **failure, "invalid_responses": getattr(exc, "diagnostics", []),
                        })
                        store.mark_error("perception", item, safe_error(exc))
                    _checkpoint_cost(run_dir, prior_cost, cost)
                    reporter.on_phase_item_end(detail=f"{failure['label']} · {failure['error']}", is_error=True)
                    if is_non_retryable_api_error(exc):
                        raise
                stop_reason = guard.observe(failed)
                if stop_reason:
                    break
            if stop_reason:
                break
        reporter.on_phase_end(
            detail=(
                stop_reason
                or f"{len(detections)} raw detections; {sum(call_counts.values())} fixed-crop calls"
            )
        )

    if stop_reason:
        if store is not None:
            store.mark_error("perception", "run_guard", stop_reason)
        if run_dir is not None:
            atomic_write_json(
                run_dir / "perception.stop.json",
                {
                    "reason": stop_reason,
                    "processed_crops": progress_item,
                    "planned_crops": planned_total,
                },
            )
        assessed = {r.get("candidate_id") for r in perception_reviews}
        for page in pid_pages:
            statuses[page.page_index] = "partial"
            perception_reviews.extend(
                {
                    "candidate_id": c.id,
                    "page_index": page.page_index,
                    "bbox": c.bbox.model_dump(mode="json"),
                    "source_path_ids": c.source_path_ids,
                    "status": "unreviewed",
                    "reason": "运行已停止，此候选尚未处理。",
                    "processing_status": "not_processed",
                    "run_stop_reason": stop_reason,
                }
                for c in detect_symbols(page=page).native_candidates
                if c.id not in assessed
            )

    if cfg.raster_ink_filter:
        from diagex.vision.raster_ink import filter_observations
        detections, perception_reviews, ink_audit = filter_observations(
            source.path, pages, detections, perception_reviews,
        )
        if run_dir is not None:
            atomic_write_json(run_dir / "raster-ink-filter.json", ink_audit)

    for page in pid_pages:
        errors = page_error_counts.get(page.page_index, 0)
        inspected = page_view_counts.get(page.page_index, 0)
        page_detections = [value for value in detections if value.page_index == page.page_index]
        if errors:
            statuses[page.page_index] = (
                "error" if not page_detections and errors >= inspected else "partial"
            )

    if run_dir is not None:
        atomic_write_json(
            run_dir / "perception.review.json",
            {
                "version": SYMBOL_PERCEPTION_VERSION,
                "reviews": perception_reviews,
                "note": "Unreviewed candidates and unanchored proposals are not confirmed detections.",
            },
        )
    return detections, statuses, call_counts, stop_reason


def _native_line_legend_images(
    *,
    source: Any,
    pages: list[PageEvidence],
    legend_pack: LegendPack,
) -> list[dict[str, Any]]:
    """Recover complete line-style rows from vector legend pages.

    Model-produced legend boxes are deliberately tight around the glyph. For
    line samples that can clip the route to a single dash. Positioned native
    text lets us recrop the whole row deterministically from a vector PDF.
    """
    from diagex.vision.loader import iter_pages

    line_entries = [entry for entry in legend_pack.entries if entry.kind == "line"]
    if not line_entries:
        return []
    evidence_by_index = {page.page_index: page for page in pages}
    results: list[dict[str, Any]] = []
    found: set[str] = set()
    for rendered_page in iter_pages(source):
        evidence = evidence_by_index.get(rendered_page.page_index)
        if (
            evidence is None
            or evidence.role != "legend"
            or evidence.is_scanned
            or rendered_page.image is None
        ):
            continue
        for entry in line_entries:
            label_key = _legend_text_key(entry.label)
            if not label_key or label_key in found:
                continue
            span = next(
                (
                    candidate
                    for candidate in evidence.text_spans
                    if _legend_span_matches(label_key, _legend_text_key(candidate.text))
                ),
                None,
            )
            if span is None:
                continue
            height = max(1, span.bbox.h)
            bounds = (
                max(0, span.bbox.x - max(320, height * 10)),
                max(0, span.bbox.y - max(24, height)),
                min(rendered_page.image.width, span.bbox.x2 + max(100, height * 3)),
                min(rendered_page.image.height, span.bbox.y2 + max(24, height)),
            )
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            crop = rendered_page.image.crop(bounds).convert("RGB")
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG", optimize=True)
            results.append(
                {
                    "label": entry.label,
                    "description": entry.description or "",
                    "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "source": "native_vector_legend_row",
                    "page": evidence.page_index + 1,
                }
            )
            found.add(label_key)
    return results


def _legend_text_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def _legend_span_matches(label_key: str, span_key: str) -> bool:
    if not span_key:
        return False
    if label_key == span_key:
        return True
    # Native PDF text may split a long wrapped legend label across two spans.
    return len(span_key) >= 4 and (label_key.startswith(span_key) or span_key.startswith(label_key))


def _run_contextual_resolution(
    *,
    source: Any,
    pages: list[PageEvidence],
    nodes: list[Any],
    legend_entries: list[dict[str, Any]],
    client: LLMClient,
    cost: CostTracker,
    reporter: Any,
    store: CheckpointStore | None,
    per_page_status: dict[int, str],
    run_dir: Path | None,
    prior_cost: dict[str, Any],
) -> tuple[list[ContextualPageResult], int]:
    """Run one project-legend comparison for each page that needs it."""
    from diagex.vision.loader import iter_pages

    nodes_by_page: dict[int, list[Any]] = {}
    for node in nodes:
        nodes_by_page.setdefault(node.page_index, []).append(node)
    eligible = {
        page.page_index
        for page in pages
        if page.role == "pid"
        and find_contextual_candidates(nodes, page_index=page.page_index)
        and any(entry.get("image_b64") and entry.get("kind") != "line" for entry in legend_entries)
    }
    if not eligible:
        return [], 0

    results: list[ContextualPageResult] = []
    calls = 0

    def next_step() -> int:
        return max((row.step for row in cost.steps), default=0) + 1

    def record_attempt() -> None:
        nonlocal calls
        calls += 1

    with reporter:
        reporter.on_phase_start(name="v2 contextual symbol resolution", total_items=len(eligible))
        item_number = 0
        for rendered_page in iter_pages(source):
            if rendered_page.page_index not in eligible:
                continue
            item_number += 1
            item = f"page-{rendered_page.page_index + 1:04d}"
            reporter.on_phase_item_start(
                item=item_number,
                total_items=len(eligible),
                label=f"page {rendered_page.page_index + 1} · project legend comparison",
            )
            try:
                if store is not None and store.is_done("contextual", item):
                    result = ContextualPageResult.model_validate(
                        store.read_json_artifact("contextual", item)
                    )
                    detail = f"checkpoint · {len(result.resolutions)} correction(s)"
                else:
                    if rendered_page.image is None:
                        raise ValueError("rendered page image unavailable")
                    result = resolve_contextual_page(
                        client=client,
                        cost_tracker=cost,
                        reporter=reporter,
                        page_index=rendered_page.page_index,
                        rendered_image=rendered_page.image,
                        nodes=nodes_by_page.get(rendered_page.page_index, []),
                        legend_entries=legend_entries,
                        step=next_step(),
                        on_attempt=record_attempt,
                    )
                    if store is not None:
                        store.invalidate(
                            "topology", "line_evidence", "page_graph", "assembly", "export"
                        )
                        store.write_json_artifact(
                            "contextual", item, result.model_dump(mode="json")
                        )
                    _checkpoint_cost(run_dir, prior_cost, cost)
                    detail = (
                        f"corrections={len(result.resolutions)}, "
                        f"uncertainties={sum(1 for value in result.conflicts if value.get('status') == 'unresolved')}"
                    )
                    if result.attempts > 1:
                        detail += f" · recovered after {result.attempts} attempts"
                results.append(result)
                reporter.on_phase_item_end(detail=detail)
            except Exception as exc:  # noqa: BLE001 - preserve fused objects
                if per_page_status.get(rendered_page.page_index) == "ok":
                    per_page_status[rendered_page.page_index] = "partial"
                if store is not None:
                    store.mark_error("contextual", item, repr(exc))
                _checkpoint_cost(run_dir, prior_cost, cost)
                reporter.on_phase_item_end(detail=repr(exc), is_error=True)
                if is_non_retryable_api_error(exc):
                    raise
        reporter.on_phase_end(
            detail=f"{sum(len(result.resolutions) for result in results)} applied correction(s); {calls} call(s)"
        )
    return results, calls


def _run_topology(
    *,
    source: Any,
    pages: list[PageEvidence],
    nodes: list[Any],
    legend_line_profile: LegendLineProfile,
    reporter: Any,
    store: CheckpointStore | None,
    per_page_status: dict[int, str],
) -> list[TopologyResult]:
    from diagex.vision.loader import iter_pages

    page_by_index = {page.page_index: page for page in pages}
    nodes_by_page: dict[int, list[Any]] = {}
    for node in nodes:
        nodes_by_page.setdefault(node.page_index, []).append(node)
    results: list[TopologyResult] = []
    with reporter:
        reporter.on_phase_start(name="v2 deterministic topology", total_items=len(pages))
        for rendered_page in iter_pages(source):
            evidence = page_by_index[rendered_page.page_index]
            item = f"page-{rendered_page.page_index + 1:04d}"
            reporter.on_phase_item_start(
                item=rendered_page.page_index + 1,
                total_items=len(pages),
                label=f"page {rendered_page.page_index + 1} · {evidence.role}",
            )
            if evidence.role != "pid":
                result = TopologyResult(page_index=rendered_page.page_index)
            elif store is not None and store.is_done("topology", item):
                result = TopologyResult.model_validate(store.read_json_artifact("topology", item))
            else:
                if store is not None and store.is_done("line_detection", item):
                    detected = LineDetectionResult.model_validate(store.read_json_artifact("line_detection", item)["output"])
                else:
                    detected = detect_lines(page=evidence, image=rendered_page.image)
                    if store is not None:
                        store.write_json_artifact("line_detection", item,
                            {"input": evidence.model_dump(mode="json"), "output": detected.model_dump(mode="json")})
                capture_stage(store.run_dir if store else None, "line_detection", "geometry", item,
                    {"page": evidence.model_dump(mode="json")}, image=rendered_page.image,
                    output=detected.model_dump(mode="json"))
                capture_stage(store.run_dir if store else None, "connection_inference", "geometry", item,
                    {"page": evidence.model_dump(mode="json"),
                     "nodes": [n.model_dump(mode="json") for n in nodes_by_page.get(rendered_page.page_index, [])],
                     "lines": detected.model_dump(mode="json"),
                     "legend_line_profile": legend_line_profile.model_dump(mode="json")})
                result = build_page_topology(
                    detected_lines=detected,
                    page=evidence,
                    nodes=nodes_by_page.get(rendered_page.page_index, []),
                    raster_image=rendered_page.image,
                    legend_line_profile=legend_line_profile,
                )
                if store is not None:
                    store.write_json_artifact("topology", item, result.model_dump(mode="json"))
            if (
                evidence.role == "pid"
                and result.warnings
                and len(nodes_by_page.get(rendered_page.page_index, [])) > 1
            ):
                if per_page_status.get(rendered_page.page_index) == "ok":
                    per_page_status[rendered_page.page_index] = "partial"
            results.append(result)
            reporter.on_phase_item_end(
                detail=(
                    f"edges={len(result.edges)}, used_paths={len(result.used_path_ids)}, "
                    f"styled_runs={len(result.line_style_evidence)}, "
                    f"ambiguities={len(result.ambiguities)}"
                )
            )
        reporter.on_phase_end(detail=f"{sum(len(result.edges) for result in results)} edges")
    return results


def _run_line_evidence(
    *,
    knowledge: dict | None = None,
    source: Any,
    pages: list[PageEvidence],
    topology: list[TopologyResult],
    nodes: list[Any],
    client: LLMClient,
    cost: CostTracker,
    reporter: Any,
    store: CheckpointStore | None,
    per_page_status: dict[int, str],
    run_dir: Path | None,
    prior_cost: dict[str, Any],
    legend_line_images: list[dict[str, Any]],
) -> tuple[list[PageLineEvidence], int]:
    """Run one bounded non-thinking visual line pass per non-empty P&ID page."""
    from diagex.vision.loader import iter_pages

    page_by_index = {page.page_index: page for page in pages}
    topology_by_page = {result.page_index: result for result in topology}
    nodes_by_page: dict[int, list[Any]] = {}
    for node in nodes:
        nodes_by_page.setdefault(node.page_index, []).append(node)
    eligible = [
        page
        for page in pages
        if page.role == "pid"
        and topology_by_page.get(page.page_index, TopologyResult(page_index=page.page_index)).edges
    ]
    results: list[PageLineEvidence] = []
    calls = 0

    def next_step() -> int:
        return max((row.step for row in cost.steps), default=0) + 1

    def record_attempt() -> None:
        nonlocal calls
        calls += 1

    with reporter:
        reporter.on_phase_start(
            name="v2 non-thinking visual line evidence", total_items=len(eligible)
        )
        progress_item = 0
        for rendered_page in iter_pages(source):
            evidence = page_by_index[rendered_page.page_index]
            page_topology = topology_by_page.get(
                rendered_page.page_index,
                TopologyResult(page_index=rendered_page.page_index),
            )
            if evidence.role != "pid" or not page_topology.edges:
                continue
            progress_item += 1
            item = f"page-{rendered_page.page_index + 1:04d}"
            reporter.on_phase_item_start(
                item=progress_item,
                total_items=len(eligible),
                label=f"page {rendered_page.page_index + 1} · visible line facts",
            )
            try:
                if store is not None and store.is_done("line_evidence", item):
                    result = PageLineEvidence.model_validate(
                        store.read_json_artifact("line_evidence", item)
                    )
                    detail = f"checkpoint · {len(result.assessments)} candidates"
                else:
                    if rendered_page.image is None:
                        raise ValueError("rendered page image unavailable")
                    from diagex.knowledge.resolver import resolve
                    stage_context = resolve(knowledge, "line_interpretation", evidence.page_index, "connector arrow line connection")
                    result = classify_page_line_evidence(
                        knowledge_context=stage_context,
                        client=client,
                        cost_tracker=cost,
                        reporter=reporter,
                        page=evidence,
                        rendered_image=rendered_page.image,
                        nodes=nodes_by_page.get(rendered_page.page_index, []),
                        topology=page_topology,
                        step=next_step(),
                        legend_line_images=legend_line_images,
                        max_tokens=max(2500, min(6000, 800 + len(page_topology.edges) * 90)),
                        on_attempt=record_attempt,
                    )
                    result.knowledge = stage_context
                    if store is not None:
                        store.invalidate("page_graph", "assembly", "export")
                        store.write_json_artifact(
                            "line_evidence", item, result.model_dump(mode="json")
                        )
                    _checkpoint_cost(run_dir, prior_cost, cost)
                    detail = (
                        f"assessed={len(result.assessments)}, diagnostics={len(result.diagnostics)}"
                    )
                    if result.format_recovery:
                        detail += " · recovered after 2 attempts"
                results.append(result)
                reporter.on_phase_item_end(detail=detail)
            except Exception as exc:  # noqa: BLE001 - preserve topology and continue cautiously
                if per_page_status.get(evidence.page_index) == "ok":
                    per_page_status[evidence.page_index] = "partial"
                if store is not None:
                    diagnostics = getattr(exc, "diagnostics", None)
                    if diagnostics:
                        atomic_write_json(
                            store.artifact_path("line_evidence_errors", item),
                            {
                                "error": str(exc),
                                "attempts": getattr(exc, "attempts", 1),
                                "invalid_responses": diagnostics,
                            },
                        )
                    store.mark_error("line_evidence", item, repr(exc))
                _checkpoint_cost(run_dir, prior_cost, cost)
                reporter.on_phase_item_end(detail=repr(exc), is_error=True)
                if is_non_retryable_api_error(exc):
                    raise
        reporter.on_phase_end(
            detail=f"{len(results)}/{len(eligible)} pages assessed; {calls} model calls"
        )
    return results, calls


def _run_page_graphs(
    *,
    knowledge: dict | None = None,
    source: Any,
    pages: list[PageEvidence],
    topology: list[TopologyResult],
    nodes: list[Any],
    client: LLMClient,
    cost: CostTracker,
    reporter: Any,
    store: CheckpointStore | None,
    per_page_status: dict[int, str],
    run_dir: Path | None,
    prior_cost: dict[str, Any],
    effort: EffortLevel,
    legend_summary: list[dict[str, Any]],
    legend_line_images: list[dict[str, Any]],
    line_evidence: list[PageLineEvidence],
    include_visual_context: bool,
    process_context: list[dict[str, Any]] | None = None,
    inspection_client: LLMClient | None = None,
    escalation_client: LLMClient | None = None,
) -> tuple[list[PageGraphResult], int]:
    """Run one relationship solve for each non-empty P&ID page."""
    from diagex.vision.loader import iter_pages

    page_by_index = {page.page_index: page for page in pages}
    topology_by_page = {result.page_index: result for result in topology}
    line_evidence_by_page = {result.page_index: result for result in line_evidence}
    nodes_by_page: dict[int, list[Any]] = {}
    for node in nodes:
        nodes_by_page.setdefault(node.page_index, []).append(node)
    eligible = [page for page in pages if page.role == "pid" and nodes_by_page.get(page.page_index)]
    results: list[PageGraphResult] = []
    calls = 0

    def next_step() -> int:
        return max((row.step for row in cost.steps), default=0) + 1

    def record_page_graph_attempt() -> None:
        nonlocal calls
        calls += 1

    with reporter:
        reporter.on_phase_start(name="v2 page relationship solving", total_items=len(eligible))
        progress_item = 0
        for rendered_page in iter_pages(source):
            evidence = page_by_index[rendered_page.page_index]
            local_nodes = nodes_by_page.get(rendered_page.page_index, [])
            if evidence.role != "pid" or not local_nodes:
                continue
            progress_item += 1
            item = f"page-{rendered_page.page_index + 1:04d}"
            reporter.on_phase_item_start(
                item=progress_item,
                total_items=len(eligible),
                label=f"page {rendered_page.page_index + 1} · typed relationships",
            )
            try:
                if store is not None and store.is_done("page_graph", item):
                    result = PageGraphResult.model_validate(
                        store.read_json_artifact("page_graph", item)
                    )
                    detail = f"checkpoint · {len(result.edges)} edges"
                else:
                    if rendered_page.image is None:
                        raise ValueError("rendered page image unavailable")
                    from diagex.knowledge.resolver import resolve
                    stage_context = resolve(knowledge, "connections", evidence.page_index, "connector arrow line connection")
                    result = solve_page_graph(
                        knowledge_context=stage_context,
                        client=client,
                        cost_tracker=cost,
                        reporter=reporter,
                        page=evidence,
                        rendered_image=rendered_page.image,
                        nodes=local_nodes,
                        all_nodes=nodes,
                        topology=topology_by_page.get(
                            rendered_page.page_index,
                            TopologyResult(page_index=rendered_page.page_index),
                        ),
                        pages=pages,
                        step=next_step(),
                        output_effort=EFFORT_PROFILES[effort].api_effort,
                        max_tokens=max(6000, EFFORT_PROFILES[effort].max_output_tokens),
                        legend_summary=legend_summary,
                        legend_line_images=legend_line_images,
                        visual_evidence=line_evidence_by_page.get(rendered_page.page_index),
                        on_attempt=record_page_graph_attempt,
                        process_context=process_context,
                        inspection_client=inspection_client,
                        escalation_client=escalation_client,
                        include_visual_context=include_visual_context,
                    )
                    result.knowledge = stage_context
                    if store is not None:
                        store.invalidate("assembly", "export")
                        store.write_json_artifact(
                            "page_graph", item, result.model_dump(mode="json")
                        )
                    _checkpoint_cost(run_dir, prior_cost, cost)
                    detail = (
                        f"edges={len(result.edges)}, conflicts={len(result.conflicts)}, "
                        f"rejected_output={len(result.diagnostics)}"
                    )
                    if result.format_recovery:
                        detail += " · recovered after 2 attempts"
                results.append(result)
                reporter.on_phase_item_end(detail=detail)
            except Exception as exc:  # noqa: BLE001 - preserve objects/topology
                if per_page_status.get(evidence.page_index) == "ok":
                    per_page_status[evidence.page_index] = "partial"
                if store is not None:
                    diagnostics = getattr(exc, "diagnostics", None)
                    if diagnostics:
                        atomic_write_json(
                            store.artifact_path("page_graph_errors", item),
                            {
                                "error": str(exc),
                                "attempts": getattr(exc, "attempts", 1),
                                "invalid_responses": diagnostics,
                            },
                        )
                    store.mark_error("page_graph", item, repr(exc))
                _checkpoint_cost(run_dir, prior_cost, cost)
                reporter.on_phase_item_end(detail=repr(exc), is_error=True)
                if is_non_retryable_api_error(exc):
                    raise
        reporter.on_phase_end(
            detail=f"{len(results)}/{len(eligible)} pages solved; {calls} model calls"
        )
    return results, calls


def _prepare_v2_run_dir(cfg: Config, stem: str, vision_model: str) -> tuple[Path, str]:
    runs_root = cfg.runs_dir / stem
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        run_id = _new_run_id()
        run_dir = runs_root / f"{_timestamp()}_{_safe_model_name(vision_model)}_{run_id}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Could not allocate a unique run directory after 16 attempts")
    (run_dir / "evidence").mkdir(exist_ok=True)
    return run_dir, run_id


def _stage_count_text(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{stage}={count}" for stage, count in sorted(values.items()))


def _print_reuse_start(
    console: Any,
    *,
    resumed: bool,
    run_dir: Path | None,
    reason: str,
    available: dict[str, int],
    invalidated: dict[str, int],
) -> None:
    if run_dir is None:
        console.print("[diagex.cache] persistence disabled; no checkpoint artifacts will be reused")
        return
    mode = "RESUME" if resumed else "NEW RUN"
    console.print(f"[diagex.cache] {mode} · {run_dir}")
    console.print(f"[diagex.cache] decision: {reason}")
    console.print(f"[diagex.cache] reusable checkpoint items: {_stage_count_text(available)}")
    if invalidated:
        console.print(
            "[diagex.cache] invalidated by pipeline version change: "
            + _stage_count_text(invalidated)
        )


def _print_reuse_end(console: Any, *, summary: dict[str, Any]) -> None:
    console.print(
        "[diagex.cache] final checkpoint reuse: "
        f"{summary['reused_total']} reused vs {summary['computed_total']} computed "
        f"({summary['reuse_percent']:.1f}% reused)"
    )
    console.print(
        "[diagex.cache] reused by stage: " + _stage_count_text(summary["reused_by_stage"])
    )
    console.print(
        "[diagex.cache] computed by stage: " + _stage_count_text(summary["computed_by_stage"])
    )
    console.print("[diagex.cache] details saved to reuse.report.json")


def _configuration_hash(
    *,
    cfg: Config,
    symbol_standard: str,
    vision_model: str,
    reasoning_model: str,
    legend_path: Path | None,
    legend_pages: list[int] | None,
    legend_region: tuple[int, int, int, int, int] | None,
    no_legend: bool,
    legend_key: str | None,
    effort: EffortLevel,
) -> str:
    legend_hash = sha256_file(legend_path) if legend_path is not None else None
    payload = {
        "engine": "evidence-v2",
        "engine_schema": "3.7.0",
        "transport": cfg.llm.transport,
        "vision_model": vision_model,
        "reasoning_model": reasoning_model,
        "reasoning_mode": cfg.llm.reasoning_mode,
        "escalation_model": cfg.llm.escalation_model,
        "production_open_weight": cfg.llm.production_open_weight,
        "process_context": cfg.process_context,
        "symbol_perception": asdict(cfg.symbol_perception),
        "effort": effort,
        "tiling": asdict(cfg.tiling),
        "scan": asdict(cfg.scan),
        "symbol_standard": symbol_standard,
        "legend": {
            "path_hash": legend_hash,
            "pages": legend_pages,
            "region": legend_region,
            "disabled": no_legend,
            "key": legend_key,
        },
        # Preserve the historical raw-evidence cache key. Derived behavior is
        # versioned in manifest.stage_versions; changing this field would also
        # discard compatible native evidence and perception before invalidation.
        "page_graph_pipeline": "1.9.0",
    }
    if cfg.knowledge:
        payload["knowledge"] = cfg.knowledge
    if cfg.raster_ink_filter:
        from diagex.vision.raster_ink import implementation_sha256 as ink_implementation_sha256
        payload["raster_ink_filter"] = {"implementation_sha256": ink_implementation_sha256()}
    if cfg.raster_proposals is not None:
        from diagex.vision.raster_guidance import implementation_sha256
        payload["raster_proposals"] = cfg.raster_proposals
        payload["raster_guidance_implementation_sha256"] = implementation_sha256()
        if cfg.raster_proposals.get("format") == "pdf_page_proposals_v1":
            from diagex.vision.pdf_raster_guidance import (
                implementation_sha256 as pdf_guidance_sha256,
            )
            payload["pdf_raster_guidance_implementation_sha256"] = pdf_guidance_sha256()
    if cfg.raster_symbol_mode != "baseline":
        from diagex.vision.raster_pipeline import implementation_sha256 as raster_pipeline_sha256
        payload["raster_symbol_mode"] = cfg.raster_symbol_mode
        payload["raster_pipeline_implementation_sha256"] = raster_pipeline_sha256()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_prior_cost(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    path = run_dir / "cost.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _merge_cost_summaries(prior: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    if not prior:
        return dict(current)
    additive = {
        "total_usd",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "image_tokens",
        "retries",
        "n_tool_calls",
        "wall_clock_s",
    }
    merged = dict(current)
    for key in additive:
        merged[key] = prior.get(key, 0) + current.get(key, 0)
    merged["steps"] = [*(prior.get("steps") or []), *(current.get("steps") or [])]
    tool_counts: dict[str, int] = {}
    for summary in (prior, current):
        for key, value in (summary.get("tool_call_counts") or {}).items():
            tool_counts[key] = tool_counts.get(key, 0) + int(value or 0)
    merged["tool_call_counts"] = tool_counts
    return merged


def _checkpoint_cost(
    run_dir: Path | None,
    prior: dict[str, Any],
    current: CostTracker,
) -> None:
    if run_dir is None:
        return
    atomic_write_text(
        run_dir / "cost.json",
        json.dumps(_merge_cost_summaries(prior, current.summary()), indent=2, sort_keys=True),
    )


def _augment_public_manifests(
    *,
    run_dir: Path,
    pages: list[PageEvidence],
    quality: QualityReport,
    dexpi_xml_path: Path | None,
    resumed: bool,
) -> None:
    pages_path = run_dir / "pages.json"
    try:
        pages_value = json.loads(pages_path.read_text(encoding="utf-8"))
        evidence_by_index = {page.page_index: page for page in pages}
        for row in pages_value.get("pages", []):
            evidence = evidence_by_index.get(int(row.get("page_index", -1)))
            if evidence is None:
                continue
            row.update(
                {
                    "role": evidence.role,
                    "role_confidence": evidence.role_confidence,
                    "role_reason": evidence.role_reason,
                    "fail_open": evidence.fail_open,
                    "native_text_count": len(evidence.text_spans),
                    "native_path_count": len(evidence.paths),
                }
            )
        atomic_write_text(
            pages_path,
            json.dumps(pages_value, indent=2, ensure_ascii=False),
        )
    except (OSError, ValueError, TypeError):
        pass

    result_path = run_dir / "result.json"
    try:
        result_value = json.loads(result_path.read_text(encoding="utf-8"))
        result_value.update(
            {
                "quality_status": quality.status,
                "quality_report_path": str(run_dir / "quality.report.json"),
                "dexpi_xml_path": str(dexpi_xml_path) if dexpi_xml_path else None,
                "resumed": resumed,
                "reuse_report_path": str(run_dir / "reuse.report.json"),
            }
        )
        atomic_write_text(
            result_path,
            json.dumps(result_value, indent=2, ensure_ascii=False),
        )
    except (OSError, ValueError, TypeError):
        pass
