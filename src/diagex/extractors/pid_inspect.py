"""Bounded native-text, tag, and automatically detected legend inspection."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diagex.config import Config
from diagex.extractors.evidence_checkpoint import atomic_write_json, atomic_write_text
from diagex.extractors.pid import _new_run_id, _safe_model_name, _safe_stem, _timestamp
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker, format_elapsed, format_tokens_millions
from diagex.vision.legend_models import LegendEntry, SymbolStandard
from diagex.vision.models import ReconciledGraph
from diagex.vision.native_text import NativeTextInventory, build_native_text_inventory

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class PidEvidenceInspectionResult:
    run_dir: Path
    page_count: int
    page_roles: dict[int, str]
    legend_pages: list[int]
    legend_source: str
    legend_entry_count: int
    learned_entries: list[dict[str, Any]]
    inventory: NativeTextInventory
    cost_summary: dict[str, Any]
    elapsed_s: float
    abbreviation_summary: dict[str, Any]

    def to_text(self) -> str:
        learned_images = sum(bool(entry.get("has_image")) for entry in self.learned_entries)
        kind_counts: dict[str, int] = {}
        for entry in self.learned_entries:
            kind = str(entry.get("kind") or "other")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        kind_summary = ", ".join(
            f"{kind}={count}" for kind, count in sorted(kind_counts.items())
        ) or "none"
        preview_labels = [str(entry["label"]) for entry in self.learned_entries[:12]]
        preview = ", ".join(preview_labels)
        if len(self.learned_entries) > len(preview_labels):
            preview += f", +{len(self.learned_entries) - len(preview_labels)} more"
        preview = preview or "none"
        pages = ", ".join(str(index + 1) for index in self.legend_pages) or "none"
        summary = self.inventory.summary
        return "\n".join(
            (
                "P&ID text / tag / legend inspection",
                f"pages: {self.page_count}   auto-detected legend pages: {pages}",
                (
                    f"legend: source={self.legend_source}   "
                    f"learned={len(self.learned_entries)} ({learned_images} with images)   "
                    f"merged={self.legend_entry_count}"
                ),
                f"learned kinds: {kind_summary}",
                f"learned labels: {preview}",
                (
                    f"native tags: {summary.get('reviewable_tag_count', 0)} candidates   "
                    f"typed={_typed_tag_count(self.inventory)}   "
                    f"unknown={_unknown_tag_count(self.inventory)}"
                ),
                (
                    f"tokens: {format_tokens_millions(self.cost_summary.get('total_tokens', 0))}   "
                    f"elapsed: {format_elapsed(self.elapsed_s)}"
                ),
                f"learned legend report: {self.run_dir / 'legend.learned.md'}",
                f"full legend: {self.run_dir / 'legend.json'}",
                (
                    f"abbreviation tables: {self.abbreviation_summary.get('row_count', 0)} rows   "
                    f"review: {self.run_dir / 'legend.review.html'}"
                ),
                f"tag inventory: {self.run_dir / 'evidence' / 'native-text-inventory.json'}",
                f"run: {self.run_dir}",
            )
        )


def run_pid_evidence_inspection(
    *,
    diagram: Path,
    symbol_standard: SymbolStandard,
    legend_key: str | None,
    config: Config,
    console: Console | None,
) -> PidEvidenceInspectionResult:
    """Inspect text and legends, then stop before engineering-object perception."""
    from rich.console import Console as RichConsole

    from diagex.extractors.pid_evidence import _inspect_pages
    from diagex.extractors.pid_legend import resolve_evidence_legend
    from diagex.ui.progress import make_reporter
    from diagex.vision.loader import load

    started = time.perf_counter()
    cfg = config
    vision_model = cfg.llm.vision_model or cfg.llm.model
    run_id = _new_run_id()
    stem = _safe_stem(diagram)
    runs_root = cfg.runs_dir / stem
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / (
        f"{_timestamp()}_{_safe_model_name(vision_model)}_evidence-inspection_{run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "evidence").mkdir()

    reporter_console = console or RichConsole()
    source = load(diagram, tiling=cfg.tiling, scan_cfg=cfg.scan)
    pages, _ = _inspect_pages(
        source=source,
        diagram=diagram,
        store=None,
        run_dir=run_dir,
        reporter=make_reporter(reporter_console, effort="medium"),
    )
    legend_pages = [page.page_index for page in pages if page.role == "legend"]

    cost = CostTracker(pricing=cfg.pricing)
    legend_cfg = replace(cfg.llm, model=vision_model, reasoning_mode="disabled")
    client = LLMClient(legend_cfg, budgets=cfg.budgets)
    reporter = make_reporter(reporter_console, effort="medium")
    with reporter:
        reporter.on_phase_start(name="legend-only extraction", total_items=len(legend_pages) or 1)
        routed = resolve_evidence_legend(
            source=source,
            pages=pages,
            symbol_standard=symbol_standard,
            cfg=cfg,
            client=client,
            cost_tracker=cost,
            legend_key=legend_key,
            runs_dir_for_stem=runs_root,
            reporter=reporter,
        )
        resolution = routed.resolution
        reporter.on_phase_end(detail=f"{len(resolution.pack.entries)} merged entries")
    legend_pages = routed.detected_page_indices
    legend_pack = resolution.pack
    legend_source = f"evidence_v2_router:{resolution.source}"
    abbreviation_inventory = routed.abbreviation_inventory

    empty_graph = ReconciledGraph(
        source_path=diagram.name,
        per_page_status={page.page_index: "ok" for page in pages},
    )
    inventory = build_native_text_inventory(
        pages=pages,
        graph=empty_graph,
        legend_pack=legend_pack,
    )
    learned_entries = _write_learned_entries(
        run_dir,
        [entry for entry in legend_pack.entries if entry.source != "built_in"],
    )
    learned_entries.sort(key=lambda entry: (entry["kind"], entry["label"].casefold()))

    page_roles = {page.page_index: page.role for page in pages}
    cost_summary = cost.summary()
    cost_summary["retries"] = client.retries_total
    source_hash = _sha256_file(diagram)

    abbreviation_json = run_dir / "legend.abbreviations.json"
    atomic_write_json(abbreviation_json, {
        "schema_version": "1.0.0", "source_sha256": source_hash,
        "summary": abbreviation_inventory.summary, "sections": abbreviation_inventory.sections,
        "rows": [row.model_dump(mode="json") for row in abbreviation_inventory.rows],
    })
    elapsed_s = round(time.perf_counter() - started, 3)

    atomic_write_text(run_dir / "legend.json", legend_pack.model_dump_json(indent=2))
    atomic_write_json(
        run_dir / "legend.learned.json",
        {
            "schema_version": "1.0.0",
            "source_name": diagram.name,
            "source_sha256": source_hash,
            "auto_detected_legend_pages": [index + 1 for index in legend_pages],
            "legend_source": legend_source,
            "learned_entry_count": len(learned_entries),
            "merged_entry_count": len(legend_pack.entries),
            "abbreviation_inventory": abbreviation_inventory.summary,
            "entries": learned_entries,
        },
    )
    atomic_write_text(
        run_dir / "legend.learned.md",
        _legend_markdown(
            diagram=diagram,
            legend_pages=legend_pages,
            legend_source=legend_source,
            learned_entries=learned_entries,
            merged_count=len(legend_pack.entries),
        ),
    )
    inventory_payload = inventory.model_dump(mode="json")
    inventory_payload["inspection_only"] = True
    inventory_payload["note"] = (
        "No object perception was run; tag statuses are unresolved by design. "
        "Use expected_kind and tag_semantics to inspect deterministic classification."
    )
    atomic_write_json(run_dir / "evidence" / "native-text-inventory.json", inventory_payload)
    atomic_write_json(
        run_dir / "pages.json",
        {
            "schema_version": "1.0.0",
            "source_name": diagram.name,
            "source_sha256": source_hash,
            "pages": [
                {
                    "page_index": page.page_index,
                    "page_number": page.page_index + 1,
                    "role": page.role,
                    "role_confidence": page.role_confidence,
                    "role_reason": page.role_reason,
                    "text_count": len(page.text_spans),
                    "path_count": len(page.paths),
                }
                for page in pages
            ],
        },
    )
    atomic_write_json(run_dir / "cost.json", {**cost_summary, "wall_clock_s": elapsed_s})
    atomic_write_json(
        run_dir / "result.json",
        {
            "schema_version": "1.0.0",
            "command": "inspect-pid-evidence",
            "source_name": diagram.name,
            "source_sha256": source_hash,
            "page_count": len(pages),
            "page_roles": {str(index): role for index, role in page_roles.items()},
            "auto_detected_legend_pages": [index + 1 for index in legend_pages],
            "legend_source": legend_source,
            "learned_entry_count": len(learned_entries),
            "merged_entry_count": len(legend_pack.entries),
            "abbreviation_inventory": abbreviation_inventory.summary,
            "abbreviation_json": abbreviation_json.name,
            "native_text_inventory": inventory.summary,
            "model": vision_model,
            "cost": cost_summary,
            "wall_clock_s": elapsed_s,
        },
    )

    return PidEvidenceInspectionResult(
        run_dir=run_dir,
        page_count=len(pages),
        page_roles=page_roles,
        legend_pages=legend_pages,
        legend_source=legend_source,
        legend_entry_count=len(legend_pack.entries),
        learned_entries=learned_entries,
        inventory=inventory,
        cost_summary=cost_summary,
        elapsed_s=elapsed_s,
        abbreviation_summary=abbreviation_inventory.summary,
    )


def _write_learned_entries(
    run_dir: Path, entries: list[LegendEntry]
) -> list[dict[str, Any]]:
    from diagex.extractors.pid_legend import validate_legend_image_bytes

    image_dir = run_dir / "legend-images"
    learned: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        image_bytes: bytes | None = None
        image_error: str | None = None
        try:
            image_bytes = entry.image_bytes()
        except Exception as exc:  # malformed model output remains visible in the report
            image_error = repr(exc)
        if image_bytes and (rejection := validate_legend_image_bytes(image_bytes)):
            image_error = rejection
            image_bytes = None
        image_file: str | None = None
        if image_bytes:
            image_dir.mkdir(exist_ok=True)
            image_name = f"entry-{index:04d}-{hashlib.sha256(image_bytes).hexdigest()[:10]}.png"
            (image_dir / image_name).write_bytes(image_bytes)
            image_file = f"legend-images/{image_name}"
        learned.append(_learned_entry(entry, image_bytes, image_file, image_error))
    return learned


def _learned_entry(
    entry: LegendEntry,
    image_bytes: bytes | None,
    image_file: str | None,
    image_error: str | None,
) -> dict[str, Any]:
    return {
        "label": entry.label,
        "description": entry.description,
        "kind": entry.kind,
        "symbol_class": entry.symbol_class,
        "attributes": dict(entry.attributes),
        "source": entry.source,
        "has_image": image_bytes is not None,
        "image_file": image_file,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest() if image_bytes else None,
        "image_error": image_error,
        "source_page": entry.source_page_index,
        "source_bbox": entry.source_bbox.model_dump() if entry.source_bbox else None,
        "source_label_bbox": (
            entry.source_label_bbox.model_dump() if entry.source_label_bbox else None
        ),
        "source_view": entry.source_view,
        "crop_method": entry.crop_method,
        "crop_quality": entry.crop_quality,
    }


def _legend_markdown(
    *,
    diagram: Path,
    legend_pages: list[int],
    legend_source: str,
    learned_entries: list[dict[str, Any]],
    merged_count: int,
) -> str:
    pages = ", ".join(str(index + 1) for index in legend_pages) or "None"
    lines = [
        "# Learned P&ID legend",
        "",
        f"- Source: `{diagram.name}`",
        f"- Automatically detected legend pages: {pages}",
        f"- Resolution source: `{legend_source}`",
        f"- Project entries learned: {len(learned_entries)}",
        f"- Entries after built-in merge: {merged_count}",
        "",
    ]
    if not learned_entries:
        lines.append("No project-specific legend entries were extracted.")
        lines.append("")
        return "\n".join(lines)
    lines.extend(
        (
            "| Label | Kind | Symbol class | Crop quality | Source | Attributes | Image saved | Description |",
            "|---|---|---|---|---|---|---:|---|",
        )
    )
    for entry in learned_entries:
        attrs = json.dumps(entry["attributes"], ensure_ascii=False, sort_keys=True)
        lines.append(
            "| "
            + " | ".join(
                (
                    _md(entry["label"]),
                    _md(entry["kind"]),
                    _md(entry["symbol_class"]),
                    _md(entry.get("crop_quality") or "not applicable"),
                    _md(
                        (
                            f"page {entry['source_page'] + 1}, "
                            f"{entry.get('crop_method') or 'no crop'}"
                        )
                        if entry.get("source_page") is not None
                        else "built-in/customer"
                    ),
                    _md(attrs),
                    (
                        f"![symbol]({entry['image_file']})"
                        if entry.get("image_file")
                        else "decode error"
                        if entry.get("image_error")
                        else "no"
                    ),
                    _md(entry.get("description") or ""),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _typed_tag_count(inventory: NativeTextInventory) -> int:
    return sum(item.blocking and item.expected_kind is not None for item in inventory.items)


def _unknown_tag_count(inventory: NativeTextInventory) -> int:
    return sum(item.blocking and item.expected_kind is None for item in inventory.items)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
