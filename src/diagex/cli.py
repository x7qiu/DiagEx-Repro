"""diagex CLI.

Phase 1 ships the `query` command (spec §7.1); Phase 2 ships `extract-pid`
(spec §7.2). Phase 3 commands land later.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from diagex.config import EFFORT_PROFILES, load_config

app = typer.Typer(
    name="diagex",
    help="LLM-vision extraction pipeline for industrial schematics.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _parse_pages(spec: str, total: int) -> list[int]:
    """`--page 3`, `--page 3-7`, `--page all` → 0-based page indices."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total))
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo) - 1, int(hi)))
    return [int(spec) - 1]


@app.command()
def query(
    diagram: Path = typer.Argument(..., exists=True, readable=True, help="Path to a PDF or image."),
    question: str | None = typer.Option(None, "--question", "-q", help="Natural-language question."),
    question_file: Path | None = typer.Option(
        None, "--question-file", exists=True, readable=True,
        help="File containing the question (mutually exclusive with --question).",
    ),
    page: str = typer.Option("1", "--page", help="Page spec: N, N-M, or 'all'. 1-based."),
    effort: str = typer.Option("high", "--effort", help="Reasoning depth: low|medium|high|xhigh."),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Override default runs/ directory."),
    fmt: str = typer.Option("both", "--format", help="Stdout format: text|json|both."),
    no_persist: bool = typer.Option(False, "--no-persist", help="Skip on-disk artefact writes."),
) -> None:
    """Ask a natural-language question of a single diagram (Phase 1).

    A live status line and event log stream to stdout when run in a terminal;
    when stdout is piped the same events are emitted as plain timestamped lines.
    """
    if (question is None) == (question_file is None):
        console.print("[red]Provide exactly one of --question or --question-file.[/red]")
        raise typer.Exit(2)
    if effort not in EFFORT_PROFILES:
        console.print(f"[red]Unknown --effort '{effort}'. Pick one of {list(EFFORT_PROFILES)}.[/red]")
        raise typer.Exit(2)
    if fmt not in {"text", "json", "both"}:
        console.print(f"[red]Unknown --format '{fmt}'. Pick text|json|both.[/red]")
        raise typer.Exit(2)

    q_text = question if question is not None else question_file.read_text(encoding="utf-8").strip()
    cfg = load_config()
    if out_dir is not None:
        cfg.runs_dir = out_dir

    # Deferred import: heavy deps (pymupdf, anthropic) shouldn't load for `--help`.
    from diagex.extractors.query import run_query

    try:
        result = run_query(
            diagram=diagram,
            question=q_text,
            page_spec=page,
            effort=effort,   # type: ignore[arg-type]
            config=cfg,
            persist=not no_persist,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]query failed:[/red] {exc}")
        raise typer.Exit(1)

    if fmt in {"text", "both"}:
        console.print(result.to_text())
    if fmt in {"json", "both"}:
        console.print_json(result.to_json())

    sys.exit(0)


_LEGEND_REGION_RE = re.compile(
    r"^p(?P<page>\d+)\s*:\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*,\s*(?P<w>\d+)\s*,\s*(?P<h>\d+)\s*$"
)

_VALID_SYMBOL_STANDARDS = {"isa-5.1", "iso-10628", "sama", "none"}


def _parse_legend_pages(spec: str) -> list[int]:
    """`--legend-pages 1,2,5` → 0-based indices. Accepts single pages only; no ranges (§7.2.2)."""
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok) - 1)
    return out


def _parse_legend_region(spec: str) -> tuple[int, int, int, int, int]:
    """`--legend-region "p1:100,200,300,400"` → (page_idx, x, y, w, h) with page_idx 0-based."""
    m = _LEGEND_REGION_RE.match(spec)
    if not m:
        raise typer.BadParameter(
            "expected format 'p<page>:x,y,w,h' (1-based page, pixel coords)"
        )
    return (
        int(m["page"]) - 1,
        int(m["x"]),
        int(m["y"]),
        int(m["w"]),
        int(m["h"]),
    )


@app.command("inspect-pid-evidence")
def inspect_pid_evidence(
    diagram: Path = typer.Argument(
        ..., exists=True, readable=True, help="P&ID PDF or image to inspect."
    ),
    symbol_standard: str = typer.Option(
        "isa-5.1",
        "--symbol-standard",
        help="Built-in symbol library: isa-5.1 | iso-10628 | sama | none.",
    ),
    legend_key: str | None = typer.Option(
        None,
        "--legend-key",
        help="Optional shared cache key for this project's automatically detected legend.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Override the runs root; a new inspection run directory is created beneath it.",
    ),
) -> None:
    """Inspect native PDF text, tags, and automatically detected legend pages only."""
    if symbol_standard not in _VALID_SYMBOL_STANDARDS:
        console.print(
            f"[red]Unknown --symbol-standard '{symbol_standard}'. "
            f"Pick one of {sorted(_VALID_SYMBOL_STANDARDS)}.[/red]"
        )
        raise typer.Exit(2)

    cfg = load_config()
    if out_dir is not None:
        cfg.runs_dir = out_dir

    from diagex.extractors.pid_inspect import run_pid_evidence_inspection

    try:
        result = run_pid_evidence_inspection(
            diagram=diagram,
            symbol_standard=symbol_standard,  # type: ignore[arg-type]
            legend_key=legend_key,
            config=cfg,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]inspect-pid-evidence failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(result.to_text())
    sys.exit(0)


@app.command("extract-pid")
def extract_pid(
    diagram: Path = typer.Argument(..., exists=True, readable=True, help="P&ID PDF or image."),
    symbol_standard: str = typer.Option(
        "isa-5.1", "--symbol-standard",
        help="Built-in symbol library: isa-5.1 | iso-10628 | sama | none.",
    ),
    legend: Path | None = typer.Option(
        None, "--legend", exists=True, readable=True,
        help="Legend source file (PDF/image). Mutually exclusive with --legend-pages / --legend-region / --no-legend.",
    ),
    legend_pages: str | None = typer.Option(
        None, "--legend-pages",
        help="Comma-separated 1-based page numbers of the input that contain the legend.",
    ),
    legend_region: str | None = typer.Option(
        None, "--legend-region",
        help="Legend region on the input, formatted 'p<n>:x,y,w,h' in page pixels.",
    ),
    no_legend: bool = typer.Option(
        False, "--no-legend",
        help="Skip legend detection; use the built-in library only.",
    ),
    legend_key: str | None = typer.Option(
        None, "--legend-key",
        help="Shared legend-cache key (e.g. 'acme-2024') across a customer's drawing series.",
    ),
    process_overview: Path | None = typer.Option(None, "--process-overview", exists=True, dir_okay=False, help="Source-attributed process overview text file (Evidence v2)."),
    engineering_rules: Path | None = typer.Option(None, "--engineering-rules", exists=True, dir_okay=False, help="Engineering context for review-only connection hypotheses (Evidence v2)."),
    production_open_weight: bool = typer.Option(False, "--production-open-weight", help="Use the verified fast/strong open-weight production profile."),
    adaptive_inspection: bool = typer.Option(False, "--adaptive-inspection", help="Opt into the experimental bounded visual reinspection workflow."),
    raster_proposals: Path | None = typer.Option(  # noqa: B008 - Typer declares CLI options in defaults.
        None, "--raster-proposals", exists=True, dir_okay=False, readable=True,
        help="Experimental source-bound image or PDF-page detector proposals; guides VLM inspection (Evidence v2).",
    ),
    raster_ink_filter: bool = typer.Option(
        False, "--raster-ink-filter",
        help="Filter nearly blank raster symbol boxes before review; retain an audit (Evidence v2).",
    ),
    raster_symbol_mode: str = typer.Option(  # noqa: B008 - Typer declares CLI options in defaults.
        "baseline", "--raster-symbol-mode",
        help="Experimental symbol route: baseline or broad_review (requires raster proposals; uses legend-supported model classifications).",
    ),
    effort: str = typer.Option(
        "medium", "--effort",
        help="Reasoning depth: low|medium|high|xhigh (independent of page step limit).",
    ),
    engine: str | None = typer.Option(
        None,
        "--engine",
        help="Extraction engine: legacy | evidence-v2 (default: DIAGEX_PID_ENGINE or legacy).",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        min=1,
        max=200,
        help=(
            "Override the dynamic per-page step ceiling. By default it is based "
            "on tile count and clamped to 20-60."
        ),
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Path for the DEXPI JSON output (default: run_dir/pid.dexpi.json).",
    ),
    confidence_report: Path | None = typer.Option(
        None, "--confidence-report",
        help="Path for the HTML confidence report (default: run_dir/confidence_report.html).",
    ),
    out_dir: Path | None = typer.Option(None, "--out-dir", help="Override default runs/ directory."),
    no_persist: bool = typer.Option(False, "--no-persist", help="Skip on-disk artefact writes."),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help=(
            "Start a new run directory instead of resuming a matching incomplete "
            "evidence-v2 checkpoint. Shared legend caches remain available."
        ),
    ),
    no_arbitrate_low_confidence: bool = typer.Option(
        False, "--no-arbitrate-low-confidence",
        help="Disable the low-confidence second-pass arbitration "
             "(on by default; see cfg.pid.arbitrate_low_confidence).",
    ),
) -> None:
    """Extract a full P&ID into DEXPI 2.0 JSON (Phase 2, spec §7.2)."""
    # Legend-flag mutual exclusivity (§7.2.2 resolution order).
    explicit_flags = [
        ("--legend", legend is not None),
        ("--legend-pages", legend_pages is not None),
        ("--legend-region", legend_region is not None),
        ("--no-legend", no_legend),
    ]
    used = [name for name, on in explicit_flags if on]
    if len(used) > 1:
        console.print(f"[red]Legend flags are mutually exclusive — got: {', '.join(used)}.[/red]")
        raise typer.Exit(2)

    if symbol_standard not in _VALID_SYMBOL_STANDARDS:
        console.print(
            f"[red]Unknown --symbol-standard '{symbol_standard}'. "
            f"Pick one of {sorted(_VALID_SYMBOL_STANDARDS)}.[/red]"
        )
        raise typer.Exit(2)
    if effort not in EFFORT_PROFILES:
        console.print(f"[red]Unknown --effort '{effort}'. Pick one of {list(EFFORT_PROFILES)}.[/red]")
        raise typer.Exit(2)

    parsed_pages = _parse_legend_pages(legend_pages) if legend_pages else None
    parsed_region = _parse_legend_region(legend_region) if legend_region else None

    cfg = load_config(production_open_weight=True) if production_open_weight else load_config()
    if production_open_weight:
        from diagex.llm.model_policy import apply_production_profile
        apply_production_profile(cfg, workflow="adaptive" if adaptive_inspection else "fixed")
    elif adaptive_inspection:
        cfg.symbol_perception.workflow = "adaptive"
    if engine is not None:
        normalised_engine = engine.strip().lower().replace("_", "-")
        if normalised_engine not in {"legacy", "evidence-v2"}:
            console.print(
                f"[red]Unknown --engine '{engine}'. Pick legacy or evidence-v2.[/red]"
            )
            raise typer.Exit(2)
        cfg.pid.engine = normalised_engine  # type: ignore[assignment]
    if production_open_weight and cfg.pid.engine != "evidence-v2":
        raise typer.BadParameter("The open-weight production profile requires evidence-v2")
    if out_dir is not None:
        cfg.runs_dir = out_dir
    if no_arbitrate_low_confidence:
        cfg.pid.arbitrate_low_confidence = False
    from diagex.vision.process_context import context_file
    cfg.process_context = [context_file(path, kind=kind) for path, kind in
                           ((process_overview, "process_overview"), (engineering_rules, "engineering_rules")) if path]
    if cfg.process_context and cfg.pid.engine != "evidence-v2":
        raise typer.BadParameter("Process context requires --engine evidence-v2")
    if raster_ink_filter:
        if cfg.pid.engine != "evidence-v2":
            raise typer.BadParameter("Raster ink filtering requires --engine evidence-v2")
        cfg.raster_ink_filter = True
    if raster_proposals is not None:
        if cfg.pid.engine != "evidence-v2":
            raise typer.BadParameter("Raster proposal guidance requires --engine evidence-v2")
        from diagex.vision.raster_guidance import load_guidance
        try:
            if diagram.suffix.lower() == ".pdf":
                from diagex.vision.pdf_raster_guidance import load_pdf_guidance
                cfg.raster_proposals = load_pdf_guidance(raster_proposals, diagram, cfg)
            else:
                cfg.raster_proposals = load_guidance(raster_proposals, diagram)
        except (ValueError, OSError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--raster-proposals") from exc
    if raster_symbol_mode not in {"baseline", "broad_review"}:
        raise typer.BadParameter("Choose baseline or broad_review", param_hint="--raster-symbol-mode")
    if raster_symbol_mode == "broad_review" and cfg.raster_proposals is None:
        raise typer.BadParameter("broad_review requires --raster-proposals", param_hint="--raster-symbol-mode")
    cfg.raster_symbol_mode = raster_symbol_mode

    # Deferred import: heavy deps shouldn't load for `--help`.
    from diagex.extractors.pid import run_pid_extract

    try:
        result = run_pid_extract(
            diagram=diagram,
            symbol_standard=symbol_standard,   # type: ignore[arg-type]
            legend_path=legend,
            legend_pages=parsed_pages,
            legend_region=parsed_region,
            no_legend=no_legend,
            legend_key=legend_key,
            effort=effort,                     # type: ignore[arg-type]
            max_steps=max_steps,
            engine=cfg.pid.engine,
            config=cfg,
            persist=not no_persist,
            fresh=fresh,
            out_path=out,
            confidence_report_path=confidence_report,
            console=console,
        )
    except Exception as exc:
        console.print(f"[red]extract-pid failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(result.to_text())
    sys.exit(0)


@app.command("render-dexpi")
def render_dexpi(
    graph_json: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Path to a graph.json emitted by `diagex extract-pid` (or a run_dir).",
    ),
    out: Path | None = typer.Option(
        None, "--out",
        help="Output SVG path (default: sibling of graph.json with .svg suffix).",
    ),
    title: str | None = typer.Option(
        None, "--title", help="Title shown in the SVG header (default: source_path)."
    ),
    ortho: bool = typer.Option(
        True, "--ortho/--no-ortho",
        help="Manhattan-route inferred edges (default on). Bootstrap polylines "
             "are never modified.",
    ),
) -> None:
    """Render a ReconciledGraph JSON to an SVG for visual inspection (spec §7.2)."""
    # Accept either a graph.json file or a run-dir that contains one.
    target = graph_json
    if target.is_dir():
        candidate = target / "graph.json"
        if not candidate.exists():
            console.print(f"[red]{target} has no graph.json.[/red]")
            raise typer.Exit(2)
        target = candidate

    from diagex.extractors.dexpi_svg import render_graph_json_to_svg

    out_path = out if out is not None else target.with_suffix(".svg")
    try:
        svg = render_graph_json_to_svg(target, title=title or "", orthogonal_inferred=ortho)
    except Exception as exc:
        console.print(f"[red]render failed:[/red] {exc}")
        raise typer.Exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    console.print(f"wrote {out_path}")


@app.command("render-drawio")
def render_drawio(
    graph_json: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Path to a graph.json emitted by `diagex extract-pid` (or a run_dir).",
    ),
    out: Path | None = typer.Option(
        None, "--out",
        help="Output .drawio path (default: sibling of graph.json with .drawio suffix).",
    ),
    title: str | None = typer.Option(
        None, "--title", help="Title shown in the title block (default: source_path)."
    ),
    ortho: bool = typer.Option(
        True, "--ortho/--no-ortho",
        help="Manhattan-route inferred edges (default on). Bootstrap polylines "
             "are never modified.",
    ),
) -> None:
    """Render a ReconciledGraph JSON to an editable draw.io file using the
    Process Engineering stencils from diagrams.net (mxgraph.pid* shapes)."""
    target = graph_json
    if target.is_dir():
        candidate = target / "graph.json"
        if not candidate.exists():
            console.print(f"[red]{target} has no graph.json.[/red]")
            raise typer.Exit(2)
        target = candidate

    from diagex.extractors.dexpi_drawio import render_graph_json_to_drawio

    out_path = out if out is not None else target.with_suffix(".drawio")
    try:
        xml = render_graph_json_to_drawio(target, title=title or "", orthogonal_inferred=ortho)
    except Exception as exc:
        console.print(f"[red]render failed:[/red] {exc}")
        raise typer.Exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")
    console.print(f"wrote {out_path}")


@app.command("dexpi-validate")
def dexpi_validate(
    target: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Path to a DEXPI XML file (.xml) or a diagex JSON model (.json / .dexpi.json).",
    ),
    rules: str | None = typer.Option(
        None, "--rules",
        help="Comma-separated list of rule IDs to enable (e.g. DEX0001,DEX0003). "
             "Default: all rules.",
    ),
    skip_xsd: bool = typer.Option(
        False, "--skip-xsd",
        help="Skip the XSD envelope check even when the input is XML.",
    ),
) -> None:
    """v2 Phase F: validate a DEXPI 2.0 model against the XSD + semantic rule pack.

    Exits with code 1 if any error-severity issue is found, 0 otherwise.
    Warnings don't fail the run; pipe through `grep '\\[error\\]'` to gate.
    """
    from diagex.dexpi import json_io as _json_io
    from diagex.dexpi import validate as _validate
    from diagex.dexpi import xml_io as _xml_io

    rule_filter = None
    if rules:
        rule_filter = {r.strip() for r in rules.split(",") if r.strip()}

    is_xml = target.suffix.lower() == ".xml"
    issues: list = []

    if is_xml and not skip_xsd:
        try:
            issues.extend(_validate.xsd_validate(target))
        except _validate.XmlschemaUnavailableError as exc:
            console.print(f"[yellow]xsd skipped:[/yellow] {exc}")

    try:
        if is_xml:
            model = _xml_io.load(target)
        else:
            model = _json_io.load(target)
    except Exception as exc:
        console.print(f"[red]parse failed:[/red] {exc}")
        raise typer.Exit(2)

    issues.extend(_validate.semantic_validate(model, rule_filter=rule_filter))

    for issue in issues:
        marker = "[red]" if issue.severity == "error" else "[yellow]"
        console.print(f"{marker}{issue}[/]")
    n_err = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warning")
    console.print(f"summary: {n_err} error(s), {n_warn} warning(s)")
    if _validate.has_errors(issues):
        raise typer.Exit(1)


@app.command("dexpi-render")
def dexpi_render(
    model_json: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Path to a diagex JSON model (e.g. pid.dexpi.json).",
    ),
    out: Path | None = typer.Option(
        None, "--out",
        help="Output XML path (default: sibling of input with .xml suffix).",
    ),
) -> None:
    """v2 Phase F: re-render a diagex JSON model as DEXPI 2.0 XML.

    Useful when you have a JSON model in ``runs/`` and want to hand the XML
    form to an external DEXPI consumer without re-running the extractor.
    """
    from diagex.dexpi import json_io as _json_io
    from diagex.dexpi import xml_io as _xml_io

    try:
        model = _json_io.load(model_json)
    except Exception as exc:
        console.print(f"[red]load failed:[/red] {exc}")
        raise typer.Exit(2)

    out_path = out if out is not None else model_json.with_suffix(".xml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _xml_io.dump(model, out_path)
    console.print(f"wrote {out_path}")


@app.command("render-debug")
def render_debug(
    graph_json: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Path to a graph.json emitted by `diagex extract-pid` (or a run_dir).",
    ),
    out: Path | None = typer.Option(
        None, "--out",
        help="Output Markdown path (default: sibling of graph.json with .debug.md suffix).",
    ),
    title: str | None = typer.Option(
        None, "--title", help="Title shown in the report header (default: source_path)."
    ),
    coords: bool = typer.Option(
        False, "--coords", help="Include bbox centre coordinates alongside zone labels.",
    ),
) -> None:
    """Render a ReconciledGraph JSON as a condensed text debug report (spec §7.2)."""
    target = graph_json
    if target.is_dir():
        candidate = target / "graph.json"
        if not candidate.exists():
            console.print(f"[red]{target} has no graph.json.[/red]")
            raise typer.Exit(2)
        target = candidate

    from diagex.extractors.debug_report import render_graph_json_to_debug

    out_path = out if out is not None else target.with_suffix(".debug.md")
    try:
        text = render_graph_json_to_debug(
            target, title=title or "", show_coords=coords
        )
    except Exception as exc:
        console.print(f"[red]render failed:[/red] {exc}")
        raise typer.Exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    console.print(f"wrote {out_path}")


@app.command()
def version() -> None:
    """Print the installed diagex version."""
    from diagex import __version__
    console.print(__version__)


@app.command("web")
def web_workbench(
    host: str = typer.Option("127.0.0.1", "--host", help="Local server bind address."),
    port: int = typer.Option(
        8765,
        "--port",
        min=0,
        max=65535,
        help="Local server port; 0 selects a free port.",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Do not open the browser automatically.",
    ),
) -> None:
    """Configure and run P&ID extraction in a local browser."""
    from diagex.web.server import Workbench, serve_workbench

    cfg = load_config()
    workbench = Workbench(cfg)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[yellow]warning:[/yellow] the workbench has no authentication or TLS; "
            "API keys entered in a remotely accessed page would travel unencrypted"
        )
    console.print(
        f"runs: {Path(cfg.runs_dir).expanduser().resolve()}   "
        "credentials: memory only   (Ctrl+C stops safely)"
    )
    serve_workbench(
        workbench,
        host=host,
        port=port,
        open_browser=not no_open,
        on_ready=lambda url: console.print(f"workbench: [cyan]{url}[/cyan]"),
    )




# ---------------------------------------------------------------------------
# `diagex gt …` — ground-truth dataset tooling (spec §9.1)
# ---------------------------------------------------------------------------

gt_app = typer.Typer(
    help="Ground-truth dataset tools.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(gt_app, name="gt")


@gt_app.command("lint")
def gt_lint(
    path: Path = typer.Argument(
        Path("eval/datasets"),
        help="Fixture dir, dataset root, or a single ground-truth/graph file.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat missing recommended files and warnings as errors.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress info-level messages.",
    ),
) -> None:
    """Validate a ground-truth dataset per spec §9.1.

    Accepts a dataset root (walks manifest + every fixture dir), a single
    fixture dir, or a specific file (graph.json / queries.truth.yaml /
    annotations.truth.jsonl / meta.yaml / *.manifest.yaml). Exits 0 if clean,
    1 if any errors (or warnings under --strict).
    """
    from diagex import gt_lint as linter

    if not path.exists():
        console.print(f"[red]path not found:[/red] {path}")
        raise typer.Exit(2)

    report = linter.lint(path, strict=strict)

    if not report.issues:
        console.print(f"[green]✓[/green] {len(report.files_seen)} file(s) clean")
        raise typer.Exit(0)

    colors = {"error": "red", "warn": "yellow", "info": "cyan"}
    for issue in report.issues:
        if quiet and issue.level == "info":
            continue
        color = colors.get(issue.level, "white")
        console.print(f"[{color}]{issue.level.upper():5}[/{color}]  {issue.path}: {issue.message}")

    console.print(
        f"\n{len(report.files_seen)} file(s) checked — "
        f"[red]{report.error_count} error(s)[/red], "
        f"[yellow]{report.warn_count} warning(s)[/yellow]"
    )
    raise typer.Exit(0 if report.ok(strict=strict) else 1)


@gt_app.command("edit")
def gt_edit(
    fixture_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Fixture directory under eval/datasets/ (must contain graph.bootstrap.json).",
    ),
    resume: bool | None = typer.Option(
        None, "--resume/--no-resume",
        help="Continue from graph.truth.history.json if it exists (default: prompt).",
    ),
    render_dpi: int = typer.Option(150, "--render-dpi", help="DPI for per-node crop previews."),
    crop_pad: int = typer.Option(80, "--crop-pad", help="Pixel padding around each bbox in the crop."),
    skip_confirmed: bool = typer.Option(
        False, "--skip-confirmed",
        help="Auto-keep nodes the extractor flagged arbitration=confirmed.",
    ),
    rater: str = typer.Option("", "--rater", help="Name of the rater (recorded in history)."),
) -> None:
    """Walk graph.bootstrap.json entity-by-entity and author graph.truth.json.

    Two passes:
      1. nodes   — keep / revise / drop / skip / add-missing / undo / quit
      2. edges   — keep / revise / drop / skip / add-missing / undo / quit

    If `graph.bootstrap_v2.json` exists (i.e. a VIA correction pass already ran
    via `scripts/via_to_bootstrap_v2.py`), it is preferred over the original
    bootstrap. Crash-safe — every action writes through to
    graph.truth.history.json, and `--resume` replays from there.
    """
    from diagex.gt_edit import run_edit

    try:
        summary = run_edit(
            fixture_dir,
            render_dpi=render_dpi,
            crop_pad_px=crop_pad,
            skip_confirmed=skip_confirmed,
            rater=rater,
            resume=resume,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]gt edit failed:[/red] {exc}")
        raise typer.Exit(1)

    if summary.get("finalised"):
        console.print(
            f"[green]✓[/green] wrote graph.truth.json   "
            f"({summary['final_nodes']} nodes — "
            f"kept {summary['kept']}, revised {summary['revised']}, "
            f"dropped {summary['dropped']}, added {summary['added']}; "
            f"retention {summary['retention_pct']}%)"
        )
        console.print(
            f"  edges: {summary['final_edges']} "
            f"(kept {summary['edge_kept']}, revised {summary['edge_revised']}, "
            f"dropped {summary['edge_dropped']}, added {summary['edge_added']}; "
            f"retention {summary['edge_retention_pct']}%)"
        )
    else:
        console.print("[yellow]checkpoint saved — run `diagex gt edit <dir> --resume` to continue[/yellow]")


@gt_app.command("add-edge")
def gt_add_edge(
    fixture_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Fixture directory under eval/datasets/ (must contain graph.truth.json).",
    ),
    rater: str = typer.Option("", "--rater", help="Name of the rater (recorded in history)."),
) -> None:
    """Append edges to a finalised graph.truth.json.

    Tight `[a]dd / [q]uit-and-save` loop. Use after `gt edit` finalised a
    session and you spotted a few connections you missed. Each add appends an
    `edge_add` action to graph.truth.history.json (timestamped) and updates
    summary.edge_added / summary.final_edges.
    """
    from diagex.gt_edit import run_add_edges

    try:
        result = run_add_edges(fixture_dir, rater=rater)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]gt add-edge failed:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"[green]✓[/green] added {result['added_this_session']} edge(s) "
        f"this session; truth now has {result['final_edges']} edges "
        f"(total session edge_add actions: {result['edge_added_total']})."
    )


@app.command("stage-run")
def stage_run(
    request_file: Annotated[Path, typer.Argument(exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", help="Content-addressed stage artifacts.")] = Path(
        "output/stages"
    ),
    live: bool = typer.Option(
        False, "--live", help="Allow this explicit request to call its configured model."
    ),
) -> None:
    """Run one extraction module from a saved, versioned request."""
    import json
    from dataclasses import replace

    from diagex.vision.stage_contracts import StageRequest
    from diagex.vision.stages import ModelRuntime, is_live, run_stage

    try:
        request = StageRequest.model_validate_json(request_file.read_text())
        runtime = None
        if is_live(request.stage, request.backend):
            if not live:
                raise ValueError("This stage calls a model; pass --live explicitly")
            from diagex.llm.client import LLMClient
            from diagex.llm.cost import CostTracker
            from diagex.ui.progress import NullReporter

            cfg = load_config()
            if not request.model or not request.transport:
                raise ValueError("Specify the model and transport in the request")
            # Preserve the existing credential, budget and metering configuration.
            llm = replace(cfg.llm, model=request.model, transport=request.transport)
            if llm.transport == "openrouter" and not (llm.spending_ledger and llm.verified_prices):
                raise ValueError(
                    "OpenRouter stage calls require DIAGEX_SPENDING_LEDGER and DIAGEX_VERIFIED_PRICES"
                )
            runtime = ModelRuntime(LLMClient(llm), CostTracker(pricing=cfg.pricing), NullReporter())
        result, reused = run_stage(request, out, base_dir=request_file.parent, runtime=runtime)
        console.print(
            json.dumps(
                {
                    "stage": request.stage,
                    "reused": reused,
                    "status": result["status"],
                    "artifact": str(
                        (out / request.stage / result["identity_sha256"] / "result.json").resolve()
                    ),
                }
            )
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        console.print(f"[red]stage-run failed:[/red] {exc}")
        raise typer.Exit(2) from exc


@app.command("stage-list")
def stage_list() -> None:
    """List independently runnable responsibilities and their backends."""
    from diagex.vision.stages import STAGES

    for name, (_, backends, _) in STAGES.items():
        console.print(f"{name}: {', '.join(backends)}")


@app.command("stage-score")
def stage_score(
    reference: Annotated[Path, typer.Argument(exists=True, readable=True)],
    result: Annotated[Path, typer.Argument(exists=True, readable=True)],
    iou: float = typer.Option(0.5, "--iou"),
    line_tolerance: float = typer.Option(3.0, "--line-tolerance"),
) -> None:
    """Score a stage result against explicit, stage-specific reference labels."""
    import json

    from diagex.vision.stage_metrics import score_stage
    from diagex.vision.stages import digest

    try:
        record = json.loads(result.read_text())
        if digest(record["output"]) != record["output_sha256"]:
            raise ValueError("Result checksum mismatch")
        metrics = score_stage(
            record["stage"],
            json.loads(reference.read_text()),
            record["output"],
            iou=iou,
            line_tolerance=line_tolerance,
        )
        console.print_json(data=metrics)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        console.print(f"[red]stage-score failed:[/red] {exc}")
        raise typer.Exit(2) from exc


if __name__ == "__main__":
    app()
