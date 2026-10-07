"""Phase 2 P&ID extractor — spec §7.2.

Orchestrates: load → legend resolve → per-page agent tile-walk → reconcile →
DexpiBuilder → serialise → write artefacts.

The pipeline order matches spec §7.2 (1-7). Perception is the LLM (via the
ReAct runtime); reconciliation, graph→DEXPI mapping, validation, and all
artefact writing are deterministic Python.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diagex.config import Config, EffortLevel, PidConfig, PidEngine, load_config
from diagex.llm.cost import (
    format_elapsed,
    format_tokens_millions,
    total_tokens_from_summary,
)
from diagex.vision.legend_models import LegendPack, SymbolStandard
from diagex.vision.models import ReconciledGraph

if TYPE_CHECKING:  # optional rich console for progress display
    from rich.console import Console


from diagex.extractors.pid_legend import resolve_legend

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PidExtractionResult:
    diagram_stem: str
    effort: EffortLevel
    model: str
    graph: ReconciledGraph
    dexpi_json_path: Path | None
    dexpi_stats: dict = field(default_factory=dict)
    dexpi_issues: list[str] = field(default_factory=list)
    validation_issues: list[dict] = field(default_factory=list)
    legend_source: str = ""
    legend_entry_count: int = 0
    cost_summary: dict = field(default_factory=dict)
    run_dir: Path | None = None
    run_id: str = ""
    engine: str = "legacy"
    quality_status: str = ""
    workflow_stage: str = "graph"

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append(f"P&ID: {self.diagram_stem}")
        lines.append(f"effort: {self.effort}   model: {self.model}   engine: {self.engine}")
        s = self.dexpi_stats or {}
        lines.append(
            "stats: "
            f"equipment={s.get('equipment_count', 0)}, "
            f"valves={s.get('valve_count', 0)}, "
            f"instruments={s.get('instrument_count', 0)}, "
            f"segments={s.get('segment_count', 0)}, "
            f"opcs={s.get('opc_count', 0)}, "
            f"unclassified={s.get('unclassified_count', 0)}, "
            f"dropped_edges={s.get('dropped_edges', 0)}"
        )
        lines.append(f"legend: source={self.legend_source} entries={self.legend_entry_count}")
        if self.validation_issues:
            lines.append(f"validation: {len(self.validation_issues)} issue(s)")
        if self.dexpi_issues:
            lines.append(f"build issues: {len(self.dexpi_issues)}")
        if self.quality_status:
            lines.append(f"quality: {self.quality_status}")
        partial_pages = sorted(
            page + 1 for page, status in self.graph.per_page_status.items() if status == "partial"
        )
        if partial_pages:
            lines.append("partial pages: " + ", ".join(str(page) for page in partial_pages))
        lines.append(
            "tokens: "
            f"{format_tokens_millions(total_tokens_from_summary(self.cost_summary))} "
            f"({format_tokens_millions(self.cost_summary.get('input_tokens', 0))} in / "
            f"{format_tokens_millions(self.cost_summary.get('output_tokens', 0))} out)   "
            f"elapsed: {format_elapsed(self.cost_summary.get('wall_clock_s', 0.0))}"
        )
        if self.dexpi_json_path is not None:
            lines.append(f"dexpi: {self.dexpi_json_path}")
        if self.run_dir is not None:
            lines.append(f"run: {self.run_dir}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "diagram_stem": self.diagram_stem,
                "effort": self.effort,
                "model": self.model,
                "engine": self.engine,
                "quality_status": self.quality_status or None,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "dexpi_json_path": str(self.dexpi_json_path) if self.dexpi_json_path else None,
                "dexpi_stats": self.dexpi_stats,
                "dexpi_issues": self.dexpi_issues,
                "validation_issues": self.validation_issues,
                "legend": {"source": self.legend_source, "entry_count": self.legend_entry_count},
                "cost": self.cost_summary,
                "graph": json.loads(self.graph.model_dump_json()),
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Run-folder helpers (mirrored from query.py to keep that module untouched)
# ---------------------------------------------------------------------------


_STEM_SAFE = re.compile(r"[^a-z0-9._-]+")


def _safe_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = _STEM_SAFE.sub("-", stem).strip("-")
    return stem or "diagram"


def _safe_model_name(model: str) -> str:
    value = _STEM_SAFE.sub("-", model.lower()).strip("-")
    return value[:80] or "model"


def _new_run_id() -> str:
    return "r-" + secrets.token_hex(2)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _prepare_run_dir(cfg: Config, stem: str) -> tuple[Path, str]:
    runs_root = cfg.runs_dir / stem
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()
    run_dir = runs_root / f"{_timestamp()}_{_safe_model_name(cfg.llm.model)}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tiles").mkdir(exist_ok=True)
    return run_dir, run_id


def _append_index(runs_root: Path, run_id: str, action: str, snippet: str) -> None:
    idx = runs_root / "index.md"
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    snippet_one_line = snippet.replace("\n", " ").strip()
    if len(snippet_one_line) > 100:
        snippet_one_line = snippet_one_line[:97] + "..."
    with idx.open("a", encoding="utf-8") as f:
        f.write(f"{ts}  {run_id}  {action!r} → {snippet_one_line}\n")


def _coerce_status(s: str) -> str:
    if s in ("ok", "partial", "cost_exhausted", "error"):
        return s
    return "error"


def _page_step_limit(*, tile_count: int, cfg: PidConfig, explicit_max_steps: int | None) -> int:
    if explicit_max_steps is not None:
        return max(1, int(explicit_max_steps))
    dynamic = max(0, int(tile_count)) + max(0, int(cfg.page_step_buffer))
    return min(
        max(1, int(cfg.page_max_steps)),
        max(max(1, int(cfg.page_min_steps)), dynamic),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pid_extract(
    *,
    diagram: Path,
    symbol_standard: SymbolStandard = "isa-5.1",
    legend_path: Path | None = None,
    legend_pages: list[int] | None = None,
    legend_region: tuple[int, int, int, int, int] | None = None,
    no_legend: bool = False,
    legend_key: str | None = None,
    effort: EffortLevel = "medium",
    max_steps: int | None = None,
    engine: PidEngine | None = None,
    config: Config | None = None,
    persist: bool = True,
    fresh: bool = False,
    out_path: Path | None = None,
    confidence_report_path: Path | None = None,
    console: Console | None = None,
    stop_after: str = "graph",
    reviewed_inputs: dict | None = None,
) -> PidExtractionResult:
    cfg = config or load_config()
    selected_engine = engine or cfg.pid.engine
    if stop_after not in {"graph", "detection"}:
        raise ValueError("stop_after must be graph or detection")
    if selected_engine != "evidence-v2" and (stop_after != "graph" or reviewed_inputs is not None):
        raise ValueError("Staged detection review requires evidence-v2")
    if selected_engine == "evidence-v2":
        from diagex.extractors.pid_evidence import run_pid_evidence_extract

        return run_pid_evidence_extract(
            diagram=diagram,
            symbol_standard=symbol_standard,
            legend_path=legend_path,
            legend_pages=legend_pages,
            legend_region=legend_region,
            no_legend=no_legend,
            legend_key=legend_key,
            effort=effort,
            config=cfg,
            persist=persist,
            fresh=fresh,
            stop_after=stop_after,
            reviewed_inputs=reviewed_inputs,
            out_path=out_path,
            confidence_report_path=confidence_report_path,
            console=console,
        )

    # Defer heavy imports so `--help` paths in the CLI don't pay for them.
    from rich.console import Console as _RichConsole

    from diagex.agent.runtime import ReactRuntime, RunConfig
    from diagex.agent.state import AgentState, aggregate_tool_call_counts
    from diagex.agent.tools import build_phase2_tools
    from diagex.llm.client import LLMClient, is_non_retryable_api_error
    from diagex.llm.cost import CostTracker
    from diagex.llm.prompts.phase2_pid import build_pid_system_prompt
    from diagex.ui.progress import make_reporter
    from diagex.vision.loader import iter_pages, load
    from diagex.vision.reconcile import reconcile
    from diagex.vision.tiling import AspectAwareStrategy, tile
    from diagex.vision.views import ViewProvider

    stem = _safe_stem(diagram)
    _t_run_start = time.perf_counter()

    run_dir: Path | None = None
    run_id: str = _new_run_id()
    runs_root: Path | None = None
    if persist:
        # Ensure the per-diagram runs root exists ahead of legend resolution so
        # `runs/<stem>/legend.cache.json` has somewhere to live (spec §7.2.2).
        runs_root = cfg.runs_dir / stem
        runs_root.mkdir(parents=True, exist_ok=True)
        run_dir, run_id = _prepare_run_dir(cfg, stem)

    # --- 1. Load + tile substrate ----------------------------------------------
    source = load(diagram, tiling=cfg.tiling, scan_cfg=cfg.scan)

    progress_console = console or _RichConsole()
    reporter_factory = lambda: make_reporter(progress_console, effort=effort)  # noqa: E731

    # --- 2. Shared infrastructure ----------------------------------------------
    cost = CostTracker(pricing=cfg.pricing)
    llm = LLMClient(cfg.llm, budgets=cfg.budgets)
    llm.reset_retry_counter()

    # --- 3. Legend resolution (best-effort) ------------------------------------
    legend_source_tag = ""
    legend_entry_count = 0
    legend_pack: LegendPack | None = None
    legend_budget = None
    legend_reporter = reporter_factory()
    try:
        with legend_reporter:
            legend_reporter.on_phase_start(
                name="legend resolution",
                total_items=source.metadata.get("page_count"),
            )
            resolution = resolve_legend(  # type: ignore[misc]
                source=source,
                symbol_standard=symbol_standard,
                cfg=cfg,
                client=llm,
                cost_tracker=cost,
                legend_path=legend_path,
                legend_pages=legend_pages,
                legend_region=legend_region,
                no_legend=no_legend,
                legend_key=legend_key,
                runs_dir_for_stem=runs_root,
                reporter=legend_reporter,
            )
            legend_reporter.on_phase_end(detail=str(getattr(resolution, "source", "resolved")))
        legend_pack = getattr(resolution, "pack", None)
        legend_budget = getattr(resolution, "budget", None)
        legend_source_tag = str(getattr(resolution, "source", "") or "")
        legend_entry_count = len(legend_pack.entries) if legend_pack else 0
    except Exception as exc:  # noqa: BLE001 - legend is best-effort; fall back
        if is_non_retryable_api_error(exc):
            raise
        legend_source_tag = f"fallback_builtin(error={exc!r})"
        # Tiny synthetic pack so downstream code has an object to serialise.
        legend_pack = LegendPack(
            schema_version="0.1.0",
            source_hash="",
            source_ref=f"fallback://{stem}",
            standard=symbol_standard,
            entries=[],
            notes=f"legend resolution failed: {exc!r}",
        )

    few_shot = list(getattr(legend_budget, "few_shot", []) or [])
    lookup_only = list(getattr(legend_budget, "lookup_only", []) or [])
    has_lookup = len(lookup_only) > 0

    # --- 4. Build Phase 2 system prompt (cached, reused across pages) ----------
    system_blocks = build_pid_system_prompt(
        symbol_standard=symbol_standard,
        few_shot=few_shot,
        has_lookup_tool=has_lookup,
        effort_max_steps=cfg.pid.page_max_steps,
    )

    # --- 5. Runtime construction -----------------------------------------------
    runtime = ReactRuntime(
        client=llm,
        system_blocks=system_blocks,
        cost_tracker=cost,
        budgets=cfg.budgets,
    )
    runtime.tools_override = build_phase2_tools(with_lookup=has_lookup)

    # --- 6. Per-page agent tile-walk -------------------------------------------
    all_annotations: list = []
    per_page_states: list = []
    per_page_status: dict[int, str] = {}
    # Retained for the optional LLM-arbitration pass (§5.5). Skipped when
    # arbitration is disabled so memory stays bounded on 200-sheet docs.
    page_images: dict[int, Any] = {}
    # Page handles retained for the edge-resolve sub-loop (which needs the
    # full DiagramPage, not just the PIL image, to construct view providers).
    pages_by_index: dict[int, Any] = {}

    for page in iter_pages(source):
        tiles = tile(
            page,
            AspectAwareStrategy(
                max_tokens_per_tile=cfg.tiling.max_tokens_per_tile,
                overlap_frac=cfg.tiling.overlap_frac,
                token_per_pixel=cfg.tiling.token_per_pixel,
            ),
        )
        vp = ViewProvider(page, tiles)
        page_max_steps = _page_step_limit(
            tile_count=len(tiles),
            cfg=cfg.pid,
            explicit_max_steps=max_steps,
        )
        runtime.system_blocks = build_pid_system_prompt(
            symbol_standard=symbol_standard,
            few_shot=few_shot,
            has_lookup_tool=has_lookup,
            effort_max_steps=page_max_steps,
            expected_tile_count=len(tiles),
        )
        state = AgentState(
            question="Extract this P&ID to structured DEXPI form.",
            page=page,
        )
        # Overflow legend entries back the lookup_symbol tool.
        if has_lookup and legend_pack is not None:
            state.lookup_legend = LegendPack(
                schema_version=legend_pack.schema_version,
                source_hash=legend_pack.source_hash,
                source_ref=legend_pack.source_ref,
                standard=legend_pack.standard,
                entries=lookup_only,
                notes=legend_pack.notes,
            )

        reporter = reporter_factory()
        runtime.reporter = reporter
        try:
            with reporter:
                runtime.run(
                    state=state,
                    view_provider=vp,
                    run_cfg=RunConfig(
                        effort=effort,
                        max_steps=page_max_steps,
                        require_tile_coverage=True,
                        minimum_tile_coverage=cfg.pid.page_min_tile_coverage,
                        no_progress_step_limit=cfg.pid.page_no_progress_steps,
                    ),
                )
            per_page_status[page.page_index] = (
                "partial" if state.completion_status == "partial" else "ok"
            )
        except Exception as exc:  # noqa: BLE001
            state.push_transcript("error", {"text": f"page {page.page_index}: {exc!r}"})
            if is_non_retryable_api_error(exc):
                raise
            per_page_status[page.page_index] = "error"

        all_annotations.extend(state.annotations.all())
        per_page_states.append(state)
        if cfg.pid.arbitrate_conflicts and page.image is not None:
            page_images[page.page_index] = page.image
        if cfg.pid.resolve_dangling_edges and page.image is not None:
            pages_by_index[page.page_index] = page

        if run_dir is not None:
            _write_page_artefacts(run_dir, page, tiles, state)

    # --- 7. Reconcile ----------------------------------------------------------
    graph = reconcile(all_annotations, source_path=source.path.name)
    graph.per_page_status = {k: _coerce_status(v) for k, v in per_page_status.items()}  # type: ignore[misc]

    # --- 7a. Adaptive zoom for dangling / unsnapped edges ----------------------
    if cfg.pid.resolve_dangling_edges and pages_by_index:
        try:
            from diagex.vision.edge_resolve import (
                EdgeResolveConfig,
                resolve_dangling_edges,
            )

            new_annotations, _edge_resolve_records = resolve_dangling_edges(
                graph=graph,
                pages_by_index=pages_by_index,
                runtime=runtime,
                config=EdgeResolveConfig(
                    max_edge_resolves_per_page=cfg.pid.max_edge_resolves_per_page,
                    focus_window_px=cfg.pid.edge_resolve_focus_window_px,
                    max_steps_per_edge=cfg.pid.edge_resolve_max_steps_per_edge,
                ),
                log_path=(run_dir / "edge_resolve.jsonl") if run_dir is not None else None,
            )
            if new_annotations:
                all_annotations.extend(new_annotations)
                graph = reconcile(all_annotations, source_path=source.path.name)
                graph.per_page_status = {  # type: ignore[misc]
                    k: _coerce_status(v) for k, v in per_page_status.items()
                }
        except Exception as exc:  # noqa: BLE001 - edge-resolve is best-effort
            if is_non_retryable_api_error(exc):
                raise
            graph.conflicts.append({"type": "edge_resolve_error", "detail": repr(exc)})

    # --- 7b. LLM-arbitrated reconciliation (spec §5.5) -------------------------
    arbitration_records: list = []
    arb_log_path = (run_dir / "arbitration.jsonl") if run_dir is not None else None
    arb_cfg = None
    if (cfg.pid.arbitrate_conflicts and graph.conflicts and page_images) or (
        cfg.pid.arbitrate_low_confidence and page_images
    ):
        from diagex.vision.arbitrate import ArbitrationConfig

        arb_cfg = ArbitrationConfig(
            max_arbitrations_per_run=cfg.pid.max_arbitrations_per_run,
            low_conf_max=cfg.pid.arb_low_conf_max,
            low_conf_crop_pad_px=cfg.pid.arb_low_conf_crop_pad_px,
            low_conf_crop_max_dim=cfg.pid.arb_low_conf_crop_max_dim,
        )

    if cfg.pid.arbitrate_conflicts and graph.conflicts and page_images:
        try:
            from diagex.vision.arbitrate import arbitrate_conflicts

            arbitration_records = arbitrate_conflicts(
                graph=graph,
                page_images=page_images,
                client=llm,
                cost_tracker=cost,
                config=arb_cfg,
                log_path=arb_log_path,
            )
        except Exception as exc:  # noqa: BLE001 - arbitration is best-effort
            if is_non_retryable_api_error(exc):
                raise
            graph.conflicts.append({"type": "arbitration_error", "detail": repr(exc)})

    # --- 7c. Low-confidence second pass ----------------------------------------
    # Re-asks the model on every medium/low-confidence equipment/instrument
    # node. Stabilises the "wobbly tail" that otherwise drives run-to-run
    # entity-count variance on accessory-heavy drawings (see plan
    # /home/abb/.claude/plans/flesh-out-3-arbitration-optimized-orbit.md).
    if cfg.pid.arbitrate_low_confidence and page_images:
        try:
            from diagex.vision.arbitrate import arbitrate_low_confidence

            low_conf_records = arbitrate_low_confidence(
                graph=graph,
                page_images=page_images,
                client=llm,
                cost_tracker=cost,
                config=arb_cfg,
                log_path=arb_log_path,
            )
            arbitration_records.extend(low_conf_records)
        except Exception as exc:  # noqa: BLE001 - arbitration is best-effort
            if is_non_retryable_api_error(exc):
                raise
            graph.conflicts.append(
                {"type": "arbitration_low_confidence_error", "detail": repr(exc)}
            )

    retries = int(llm.retries_total)
    tool_call_counts = aggregate_tool_call_counts(per_page_states)

    cost_summary = cost.summary()
    cost_summary["retries"] = retries
    cost_summary["tool_call_counts"] = tool_call_counts
    cost_summary["n_tool_calls"] = sum(tool_call_counts.values())

    # --- 8. DEXPI build (graceful) ---------------------------------------------
    dexpi_stats: dict = {}
    dexpi_issues: list[str] = []
    validation_issues: list[dict] = []
    dexpi_model = None
    try:
        from diagex.extractors.dexpi_builder import (
            build_dexpi,
            serialize_model,
            validate_model,
        )

        build_result = build_dexpi(graph)
        dexpi_model = build_result.model
        dexpi_stats = dict(build_result.stats)
        dexpi_issues = list(build_result.issues)
    except Exception as exc:  # noqa: BLE001
        dexpi_issues.append(f"dexpi build failed: {exc!r}")

    # --- 9. Validate (graceful) ------------------------------------------------
    if dexpi_model is not None:
        try:
            from diagex.extractors.dexpi_builder import validate_model

            validation_issues = list(validate_model(dexpi_model))
        except Exception as exc:  # noqa: BLE001
            validation_issues = [{"path": "<validate>", "msg": repr(exc)}]

    # --- 10. Serialize ---------------------------------------------------------
    dexpi_json_path: Path | None = None
    if dexpi_model is not None:
        try:
            from diagex.extractors.dexpi_builder import serialize_model

            if out_path is not None and persist:
                out = Path(out_path)
                parent = out.parent if str(out.parent) else Path(".")
                parent.mkdir(parents=True, exist_ok=True)
                # Stem may or may not end in `.dexpi`; serialize_model appends `.json`.
                stem_for_serializer = out.name
                if stem_for_serializer.endswith(".json"):
                    stem_for_serializer = stem_for_serializer[: -len(".json")]
                dexpi_json_path = serialize_model(dexpi_model, parent, stem_for_serializer)
            elif persist and run_dir is not None:
                dexpi_json_path = serialize_model(dexpi_model, run_dir, "pid.dexpi")
            # else: persist=False and no out_path -> skip serialisation.
        except Exception as exc:  # noqa: BLE001
            dexpi_issues.append(f"dexpi serialise failed: {exc!r}")
            dexpi_json_path = None

    # Include loading, legend resolution, page extraction, reconciliation,
    # DEXPI construction, validation, and serialisation in the console total.
    cost_summary["wall_clock_s"] = round(time.perf_counter() - _t_run_start, 3)

    # --- 11. Artefacts ---------------------------------------------------------
    if run_dir is not None:
        _write_run_artefacts(
            run_dir=run_dir,
            runs_root=run_dir.parent,
            run_id=run_id,
            stem=stem,
            effort=effort,
            model=cfg.llm.model,
            graph=graph,
            cost_summary=cost_summary,
            dexpi_stats=dexpi_stats,
            dexpi_issues=dexpi_issues,
            validation_issues=validation_issues,
            dexpi_json_path=dexpi_json_path,
            legend_pack=legend_pack,
            legend_source_tag=legend_source_tag,
            states=per_page_states,
            per_page_status=per_page_status,
            confidence_report_path=confidence_report_path,
        )
    elif confidence_report_path is not None:
        # User asked for a report even without persistence.
        _write_confidence_report(
            Path(confidence_report_path),
            stem=stem,
            run_id=run_id,
            effort=effort,
            cost_summary=cost_summary,
            legend_source_tag=legend_source_tag,
            legend_entry_count=legend_entry_count,
            dexpi_stats=dexpi_stats,
            dexpi_issues=dexpi_issues,
            validation_issues=validation_issues,
            graph=graph,
            per_page_status=per_page_status,
        )

    return PidExtractionResult(
        diagram_stem=stem,
        effort=effort,
        model=cfg.llm.model,
        graph=graph,
        dexpi_json_path=dexpi_json_path,
        dexpi_stats=dexpi_stats,
        dexpi_issues=dexpi_issues,
        validation_issues=validation_issues,
        legend_source=legend_source_tag,
        legend_entry_count=legend_entry_count,
        cost_summary=cost_summary,
        run_dir=run_dir,
        run_id=run_id,
        engine="legacy",
    )


# ---------------------------------------------------------------------------
# Artefact writers
# ---------------------------------------------------------------------------


def _write_page_artefacts(run_dir: Path, page, tiles, state) -> None:
    fetched_ids = [tid for tid, n in state.tile_fetch_counts.items() if n > 0]
    region_fetches = getattr(state, "region_fetches", []) or []
    if not fetched_ids and not region_fetches:
        return
    by_id = {t.id: t for t in tiles}
    for tid in fetched_ids:
        t = by_id.get(tid)
        if t is None or t.image is None:
            continue
        t.image.save(run_dir / "tiles" / f"p{page.page_index:02d}_{tid}.png")
    for idx, region in enumerate(region_fetches):
        if region.image is None:
            continue
        name = (
            f"p{page.page_index:02d}_region-{idx:02d}"
            f"_x{region.x}-y{region.y}-w{region.w}-h{region.h}.png"
        )
        region.image.save(run_dir / "tiles" / name)


def _write_run_artefacts(
    *,
    run_dir: Path,
    runs_root: Path,
    run_id: str,
    stem: str,
    effort: str,
    model: str,
    graph: ReconciledGraph,
    cost_summary: dict,
    dexpi_stats: dict,
    dexpi_issues: list[str],
    validation_issues: list[dict],
    dexpi_json_path: Path | None,
    legend_pack: LegendPack | None,
    legend_source_tag: str,
    states: list,
    per_page_status: dict[int, str],
    confidence_report_path: Path | None,
    engine: str = "legacy",
) -> None:
    # graph.json — full reconciled graph.
    (run_dir / "graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")

    # pages.json — the exact coordinate frame used by graph.json.  The human
    # review workbench uses this to reproduce source-aligned page renderings
    # even if rendering defaults change after the extraction run.
    page_records: list[dict] = []
    seen_pages: set[int] = set()
    for state in states:
        page = state.page
        if page.page_index in seen_pages:
            continue
        seen_pages.add(page.page_index)
        page_records.append(
            {
                "page_index": page.page_index,
                "width": page.width,
                "height": page.height,
                "dpi": page.dpi,
                "effective_dpi": page.effective_dpi,
                "is_scanned": page.is_scanned,
                "rotation_deg": page.rotation_deg,
                "source_ref": page.source_ref,
            }
        )
    (run_dir / "pages.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "scan": {
                    "deskew": bool(getattr(load_config().scan, "deskew", False)),
                    "contrast": bool(getattr(load_config().scan, "contrast", True)),
                    "despeckle": bool(getattr(load_config().scan, "despeckle", True)),
                },
                "pages": sorted(page_records, key=lambda item: item["page_index"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # pid.svg — self-contained visualisation of the extracted DEXPI model.
    render_metadata = {
        "run_id": run_id,
        "model": model,
        "engine": engine,
        "effort": effort,
        "total_tokens": total_tokens_from_summary(cost_summary),
        "timestamp": run_dir.name.split("_", 1)[0],
    }
    try:
        from diagex.extractors.dexpi_svg import write_svg as _write_dexpi_svg

        _write_dexpi_svg(
            graph,
            run_dir / "pid.svg",
            title=f"{stem} · {run_id}",
            metadata=render_metadata,
        )
    except Exception as exc:
        dexpi_issues.append(f"SVG render failed: {exc}")

    # pid.drawio — same graph rendered with diagrams.net "Process Engineering"
    # stencils (mxgraph.pid*). Editable in diagrams.net for review / cleanup.
    try:
        from diagex.extractors.dexpi_drawio import write_drawio as _write_drawio

        _write_drawio(
            graph,
            run_dir / "pid.drawio",
            title=f"{stem} · {run_id}",
            metadata=render_metadata,
        )
    except Exception as exc:
        dexpi_issues.append(f"drawio render failed: {exc}")

    # debug_report.md — scan-friendly textual dump for side-by-side PDF inspection.
    try:
        from diagex.extractors.debug_report import write_debug_report as _write_debug

        _write_debug(
            graph,
            run_dir / "debug_report.md",
            title=f"{stem} · {run_id}",
            metadata=render_metadata,
        )
    except Exception as exc:
        dexpi_issues.append(f"debug report render failed: {exc}")

    # cost.json — same shape as Phase 1.
    (run_dir / "cost.json").write_text(
        json.dumps(cost_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # transcript.jsonl — one line per step across all pages.
    with (run_dir / "transcript.jsonl").open("w", encoding="utf-8") as f:
        for st in states:
            for ts in st.transcript:
                f.write(
                    json.dumps(
                        {
                            "page_index": st.page.page_index,
                            "step": ts.step,
                            "kind": ts.kind,
                            "payload": ts.payload,
                        }
                    )
                    + "\n"
                )

    # legend.json — exact pack the run used (audit).
    if legend_pack is not None:
        (run_dir / "legend.json").write_text(
            legend_pack.model_dump_json(indent=2), encoding="utf-8"
        )

    # result.json — compact summary.
    result_obj = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "diagram_stem": stem,
        "effort": effort,
        "model": model,
        "engine": engine,
        "stats": dexpi_stats,
        "dexpi_issues": dexpi_issues,
        "validation_issues": validation_issues,
        "cost": cost_summary,
        "wall_clock_s": float(cost_summary.get("wall_clock_s", 0.0)),
        "retries": int(cost_summary.get("retries", 0)),
        "tool_call_counts": dict(cost_summary.get("tool_call_counts", {})),
        "legend": {
            "source": legend_source_tag,
            "entry_count": len(legend_pack.entries) if legend_pack else 0,
        },
        "dexpi_json_path": str(dexpi_json_path) if dexpi_json_path else None,
        "per_page_status": {str(k): v for k, v in per_page_status.items()},
    }
    (run_dir / "result.json").write_text(json.dumps(result_obj, indent=2), encoding="utf-8")

    # confidence_report.html — printable, self-contained digest.
    report_target = (
        Path(confidence_report_path)
        if confidence_report_path
        else (run_dir / "confidence_report.html")
    )
    _write_confidence_report(
        report_target,
        stem=stem,
        run_id=run_id,
        effort=effort,
        cost_summary=cost_summary,
        legend_source_tag=legend_source_tag,
        legend_entry_count=len(legend_pack.entries) if legend_pack else 0,
        dexpi_stats=dexpi_stats,
        dexpi_issues=dexpi_issues,
        validation_issues=validation_issues,
        graph=graph,
        per_page_status=per_page_status,
    )

    # index.md line.
    snippet = (
        f"dexpi={dexpi_json_path.name if dexpi_json_path else 'none'} "
        f"equipment={dexpi_stats.get('equipment_count', 0)} "
        f"valves={dexpi_stats.get('valve_count', 0)} "
        f"instruments={dexpi_stats.get('instrument_count', 0)}"
    )
    _append_index(runs_root, run_id, f"extract-pid {stem}", snippet)


# ---------------------------------------------------------------------------
# Confidence report (HTML)
# ---------------------------------------------------------------------------


_REPORT_CSS = """
* { box-sizing: border-box; }
body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       margin: 24px; color: #222; line-height: 1.45; }
h1, h2, h3 { font-family: ui-sans-serif, system-ui, sans-serif; }
h1 { margin: 0 0 6px 0; font-size: 22px; }
h2 { margin: 22px 0 8px 0; font-size: 16px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { margin: 12px 0 4px 0; font-size: 14px; }
.banner { background: #f4f6f8; padding: 10px 12px; border-left: 4px solid #2266cc;
          margin-bottom: 14px; }
.banner .kv { display: inline-block; margin-right: 18px; }
.banner .k { color: #556; font-weight: 600; }
table { border-collapse: collapse; width: auto; margin: 6px 0 12px 0; }
th, td { border: 1px solid #bbb; padding: 4px 10px; text-align: left; font-size: 13px;
         vertical-align: top; }
th { background: #eef; }
td.num { text-align: right; }
ul { margin: 4px 0 10px 20px; }
.empty { color: #888; font-style: italic; }
.section-count { color: #556; font-weight: normal; font-size: 13px; }
pre { background: #f8f8f8; padding: 6px; border: 1px solid #ddd;
      white-space: pre-wrap; word-break: break-all; font-size: 12px; margin: 4px 0; }
"""


def _h(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _write_confidence_report(
    path: Path,
    *,
    stem: str,
    run_id: str,
    effort: str,
    cost_summary: dict,
    legend_source_tag: str,
    legend_entry_count: int,
    dexpi_stats: dict,
    dexpi_issues: list[str],
    validation_issues: list[dict],
    graph: ReconciledGraph,
    per_page_status: dict[int, str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>diagex P&ID report — {_h(stem)}</title>")
    parts.append(f"<style>{_REPORT_CSS}</style></head><body>")

    # Banner
    parts.append("<div class='banner'>")
    parts.append(f"<h1>P&amp;ID extraction — {_h(stem)}</h1>")
    parts.append(
        f"<div><span class='kv'><span class='k'>run:</span> {_h(run_id)}</span>"
        f"<span class='kv'><span class='k'>effort:</span> {_h(effort)}</span>"
        f"<span class='kv'><span class='k'>tokens:</span> "
        f"{format_tokens_millions(total_tokens_from_summary(cost_summary))}</span>"
        f"<span class='kv'><span class='k'>elapsed:</span> "
        f"{format_elapsed(cost_summary.get('wall_clock_s', 0.0))}</span>"
        f"<span class='kv'><span class='k'>legend:</span> {_h(legend_source_tag)} "
        f"({legend_entry_count})</span></div>"
    )
    parts.append("</div>")

    # Stats
    parts.append("<h2>Stats</h2>")
    parts.append("<table><tr><th>metric</th><th>value</th></tr>")
    for key in (
        "equipment_count",
        "valve_count",
        "instrument_count",
        "segment_count",
        "opc_count",
        "unclassified_count",
        "dropped_edges",
    ):
        parts.append(
            f"<tr><td>{_h(key)}</td><td class='num'>{_h(dexpi_stats.get(key, 0))}</td></tr>"
        )
    parts.append("</table>")

    # Validation issues
    parts.append(
        f"<h2>Validation issues <span class='section-count'>({len(validation_issues)})</span></h2>"
    )
    if not validation_issues:
        parts.append("<p class='empty'>none</p>")
    else:
        parts.append("<ul>")
        for iss in validation_issues:
            parts.append(
                f"<li><code>{_h(iss.get('path', ''))}</code>: {_h(iss.get('msg', ''))}</li>"
            )
        parts.append("</ul>")

    # DexpiBuilder issues (spec §7.2 step 7 — "offending annotations")
    parts.append(
        f"<h2>DexpiBuilder issues <span class='section-count'>({len(dexpi_issues)})</span></h2>"
    )
    if not dexpi_issues:
        parts.append("<p class='empty'>none</p>")
    else:
        parts.append("<ul>")
        for msg in dexpi_issues:
            parts.append(f"<li>{_h(msg)}</li>")
        parts.append("</ul>")

    # Reconciliation conflicts grouped by type.
    conflicts = list(getattr(graph, "conflicts", []) or [])
    parts.append(
        f"<h2>Reconciliation conflicts <span class='section-count'>({len(conflicts)})</span></h2>"
    )
    if not conflicts:
        parts.append("<p class='empty'>none</p>")
    else:
        by_type: dict[str, list[dict]] = {}
        for c in conflicts:
            t = str(c.get("type", "other"))
            by_type.setdefault(t, []).append(c)
        for t in (
            "iou_grey_zone",
            "ocr_flip_candidate",
            "unstitched_line_endpoint",
            "ambiguous_opc",
            "other",
        ):
            rows = by_type.get(t)
            if not rows:
                continue
            parts.append(f"<h3>{_h(t)} <span class='section-count'>({len(rows)})</span></h3>")
            parts.append("<ul>")
            for c in rows[:10]:
                arb = c.get("arbitration") if isinstance(c, dict) else None
                arb_badge = ""
                if isinstance(arb, dict):
                    verdict = arb.get("verdict") or arb.get("status") or "?"
                    arb_badge = f" <strong>[arbitration: {_h(verdict)}]</strong>"
                parts.append(f"<li>{arb_badge}<pre>{_h(json.dumps(c, sort_keys=True))}</pre></li>")
            if len(rows) > 10:
                parts.append(f"<li class='empty'>… {len(rows) - 10} more omitted</li>")
            parts.append("</ul>")
        # Any unrecognised types.
        unknown = {
            k: v
            for k, v in by_type.items()
            if k
            not in {
                "iou_grey_zone",
                "ocr_flip_candidate",
                "unstitched_line_endpoint",
                "ambiguous_opc",
                "other",
            }
        }
        for t, rows in unknown.items():
            parts.append(f"<h3>{_h(t)} <span class='section-count'>({len(rows)})</span></h3>")
            parts.append("<ul>")
            for c in rows[:10]:
                parts.append(f"<li><pre>{_h(json.dumps(c, sort_keys=True))}</pre></li>")
            parts.append("</ul>")

    # Per-page status table.
    parts.append("<h2>Per-page status</h2>")
    if not per_page_status:
        parts.append("<p class='empty'>no pages processed</p>")
    else:
        parts.append("<table><tr><th>page</th><th>status</th></tr>")
        for p in sorted(per_page_status.keys()):
            parts.append(
                f"<tr><td class='num'>{_h(p + 1)}</td><td>{_h(per_page_status[p])}</td></tr>"
            )
        parts.append("</table>")

    # Top 20 low-confidence annotations, grouped by kind (nodes only; edges have no label).
    low_nodes = [n for n in getattr(graph, "nodes", []) if n.confidence == "low"]
    low_nodes = sorted(
        low_nodes,
        key=lambda n: (n.kind, n.page_index, n.label),
    )
    parts.append(
        f"<h2>Low-confidence annotations <span class='section-count'>"
        f"(showing up to 20 of {len(low_nodes)})</span></h2>"
    )
    if not low_nodes:
        parts.append("<p class='empty'>none</p>")
    else:
        shown = low_nodes[:20]
        by_kind: dict[str, list] = {}
        for n in shown:
            by_kind.setdefault(n.kind, []).append(n)
        for kind in sorted(by_kind.keys()):
            rows = by_kind[kind]
            parts.append(f"<h3>{_h(kind)} <span class='section-count'>({len(rows)})</span></h3>")
            parts.append("<table><tr><th>page</th><th>label</th><th>bbox</th></tr>")
            for n in rows:
                bb = n.bbox_global
                bb_str = f"({bb.x},{bb.y}) {bb.w}x{bb.h}"
                parts.append(
                    f"<tr><td class='num'>{_h(n.page_index + 1)}</td>"
                    f"<td>{_h(n.label)}</td><td>{_h(bb_str)}</td></tr>"
                )
            parts.append("</table>")

    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")
