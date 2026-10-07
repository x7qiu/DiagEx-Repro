"""Legend resolution pipeline (spec §7.2.2).

Given a diagram + CLI flags, produce a `LegendResolution`: a merged
`LegendPack` (extracted union built-in, extracted wins on label collisions)
split into a cached few-shot block and a `lookup_symbol` overflow.

Resolution order (first matching rule wins):
    1. --legend <path>
    2. --legend-pages X,Y
    3. --legend-region "p<n>:x,y,w,h"
    4. --no-legend
    5. default: auto-detect via overview-only yes/no classifier per page.

Caching: extracted packs are persisted under either
    runs/<stem>/legend.cache.json   (per-diagram)         OR
    legends/<key>.json              (shared, --legend-key <key>)
The cache requires both `source_hash` (sha256 of the input bytes) and an
extractor fingerprint covering the schema, model, and crop contract. A
mismatch in either forces re-extraction.

The extracted pack is cached *before* merge with built-ins so the built-in
library can evolve without invalidating every customer cache.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat
from pydantic import ValidationError

from diagex.agent.runtime import ReactRuntime, RunConfig
from diagex.agent.state import AgentState
from diagex.config import EFFORT_PROFILES, Config, PidConfig
from diagex.extractors.evidence_checkpoint import atomic_write_text
from diagex.llm.client import LLMClient, is_non_retryable_api_error
from diagex.llm.cost import CostTracker
from diagex.llm.prompts.phase2_legend import build_legend_system_prompt
from diagex.ui.progress import NullReporter, ProgressReporter
from diagex.vision.encode import encode_image_block
from diagex.vision.evidence import PageEvidence, PathEvidence, TextEvidence
from diagex.vision.legend_models import (
    LEGEND_CACHE_SCHEMA_VERSION,
    LegendBudget,
    LegendEntry,
    LegendPack,
    LegendRegionCoverage,
    SymbolStandard,
    merge_source_row,
)
from diagex.vision.legend_tables import (
    AbbreviationInventory,
    extract_abbreviation_tables,
)
from diagex.vision.loader import iter_pages
from diagex.vision.models import BBox, DiagramPage, DiagramSource, Tile
from diagex.vision.tiling import AspectAwareStrategy
from diagex.vision.views import ViewProvider

LEGEND_EXTRACTOR_VERSION = "3.2.0"

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class LegendResolution:
    """Outcome of `resolve_legend`: merged pack + budget split + provenance."""

    pack: LegendPack  # merged (extracted ∪ built-in)
    budget: LegendBudget  # pack split into few_shot + lookup_only
    source: str  # "explicit_file" | "explicit_pages" | "explicit_region"
    #  | "auto_detected" | "no_legend" | "cache_hit" | "built_in_only"
    cache_path: Path | None  # where the *extracted* pack was persisted


@dataclass
class EvidenceLegendResolution:
    """Shared evidence-v2 page routing plus the ordinary legend result."""

    resolution: LegendResolution
    detected_page_indices: list[int]
    effective_page_indices: list[int] | None
    used_builtin_only: bool
    abbreviation_inventory: AbbreviationInventory


def resolve_evidence_legend(
    *,
    source: DiagramSource,
    pages: list[PageEvidence],
    symbol_standard: SymbolStandard,
    cfg: Config,
    client: LLMClient,
    cost_tracker: CostTracker,
    legend_path: Path | None = None,
    legend_pages: list[int] | None = None,
    legend_region: tuple[int, int, int, int, int] | None = None,
    no_legend: bool = False,
    legend_key: str | None = None,
    runs_dir_for_stem: Path | None = None,
    reporter: ProgressReporter | None = None,
    fresh: bool = False,
) -> EvidenceLegendResolution:
    """Resolve a legend using the exact page-routing policy shared by evidence-v2.

    If the caller did not provide an explicit selector, deterministic page
    inspection supplies all pages classified as legends.  If none were found,
    evidence-v2 deliberately uses the built-in pack rather than spending model
    calls on unrelated pages.
    """

    detected = [page.page_index for page in pages if page.role == "legend"]
    effective_pages = legend_pages
    effective_no_legend = no_legend
    if legend_path is None and legend_pages is None and legend_region is None and not no_legend:
        if detected:
            effective_pages = detected
        else:
            effective_no_legend = True

    resolution = resolve_legend(
        source=source,
        symbol_standard=symbol_standard,
        cfg=cfg,
        client=client,
        cost_tracker=cost_tracker,
        legend_path=legend_path,
        legend_pages=effective_pages,
        legend_region=legend_region,
        no_legend=effective_no_legend,
        legend_key=legend_key,
        runs_dir_for_stem=runs_dir_for_stem,
        reporter=reporter,
        evidence_pages=pages,
        fresh=fresh,
    )
    # Native vector text is the source of truth for abbreviation tables.  Add
    # the reconstructed rows after ordinary legend resolution (including a
    # cache hit) so inspection and full extraction cannot diverge.  These
    # entries intentionally win over model and built-in definitions.
    abbreviation_inventory = extract_abbreviation_tables(pages)
    if abbreviation_inventory.rows:
        abbreviation_pack = abbreviation_inventory.to_legend_pack(base=resolution.pack)
        merged = abbreviation_pack.merge(resolution.pack)
        resolution = LegendResolution(
            pack=merged,
            budget=apply_budget(merged, cfg.pid),
            source=resolution.source,
            cache_path=resolution.cache_path,
        )
    return EvidenceLegendResolution(
        resolution=resolution,
        detected_page_indices=detected,
        effective_page_indices=effective_pages,
        used_builtin_only=effective_no_legend and not abbreviation_inventory.rows,
        abbreviation_inventory=abbreviation_inventory,
    )


# ---------------------------------------------------------------------------
# Built-in loader
# ---------------------------------------------------------------------------


def load_builtin_pack(standard: SymbolStandard) -> LegendPack:
    """Read src/diagex/assets/symbols/{standard}.json into a LegendPack.

    Returns an empty pack when `standard == "none"` so callers can merge
    unconditionally. Missing files raise FileNotFoundError — this is a
    packaging bug, not a runtime condition.
    """
    if standard == "none":
        return LegendPack(standard="none", source_ref="built-in:none", entries=[])

    try:
        files = resources.files("diagex.assets.symbols")
        raw = (files / f"{standard}.json").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(f"built-in symbol library missing: {standard}") from exc

    data = json.loads(raw)
    return LegendPack.model_validate(data)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_source_hash(
    *,
    path: Path | None = None,
    page_bytes: list[bytes] | None = None,
    region_bytes: bytes | None = None,
) -> str:
    """sha256 hex over the legend input bytes. Exactly one kwarg must be set."""
    provided = [x is not None for x in (path, page_bytes, region_bytes)]
    if sum(provided) != 1:
        raise ValueError(
            "compute_source_hash: pass exactly one of path / page_bytes / region_bytes"
        )

    h = hashlib.sha256()
    if path is not None:
        # Stream the file so a 200-page PDF doesn't land in RAM.
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    elif page_bytes is not None:
        # Deterministic: hash in the order passed, with a 32-bit length prefix
        # per chunk so `[b"a", b"bc"]` and `[b"ab", b"c"]` hash differently.
        for b in page_bytes:
            h.update(len(b).to_bytes(4, "big"))
            h.update(b)
    else:
        assert region_bytes is not None
        h.update(region_bytes)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Budget split
# ---------------------------------------------------------------------------


def apply_budget(pack: LegendPack, cfg_pid: PidConfig) -> LegendBudget:
    """Split `pack.entries` into few_shot + lookup_only per §7.2.2 prioritisation.

    Priority (entries earlier in the list are admitted first):
      1. source in {"legend_extracted", "customer_override"} — customer-specific wins.
      2. For project entries, image-bearing symbols before text-only rows.
      3. standard == pack.standard (the requested symbol_standard).
      4. kind in {"instrument", "valve", "equipment"} before {"line", "connector", "other"}.
      5. Remaining entries in file order (stable, deterministic).

    Token estimate: ~40 tokens for label+description text; +300 when image_b64
    is set. v0 built-ins are text-only so the 40-token figure dominates.
    Admission stops on the first boundary hit: token budget OR max entry count.
    """
    few_shot_token_cap = cfg_pid.legend_few_shot_tokens
    max_entries = cfg_pid.legend_few_shot_max_entries
    target_standard = pack.standard

    kind_rank = {"instrument": 0, "valve": 0, "equipment": 0, "line": 1, "connector": 1, "other": 2}

    # Decorate with (priority, original_index) then sort — stable on ties.
    decorated: list[tuple[tuple[int, int, int, int], int, LegendEntry]] = []
    for i, e in enumerate(pack.entries):
        p1 = 0 if e.source in ("legend_extracted", "customer_override") else 1
        # Native abbreviation rows are valuable to deterministic tag
        # normalisation and lookup, but should never crowd project symbol
        # thumbnails out of the visual few-shot block.
        p_visual = (
            0 if e.image_b64 else 2 if e.attributes.get("legend_kind") == "abbreviation" else 1
        )
        p2 = 0 if (e.standard == target_standard) else 1
        p3 = kind_rank.get(e.kind, 2)
        decorated.append(((p1, p_visual, p2, p3), i, e))

    # Sorting on (priority_tuple, original_index) is stable and deterministic.
    decorated.sort(key=lambda t: (t[0], t[1]))

    few_shot: list[LegendEntry] = []
    lookup_only: list[LegendEntry] = []
    used = 0
    for _, _, e in decorated:
        cost = _entry_token_estimate(e)
        # Admit while both caps hold; otherwise demote.
        if len(few_shot) < max_entries and (used + cost) <= few_shot_token_cap:
            few_shot.append(e)
            used += cost
        else:
            lookup_only.append(e)

    return LegendBudget(
        few_shot=few_shot,
        lookup_only=lookup_only,
        budget_tokens=few_shot_token_cap,
        used_tokens=used,
    )


def _entry_token_estimate(e: LegendEntry) -> int:
    return 40 + (300 if e.image_b64 else 0)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path_for(
    *,
    legend_key: str | None,
    cfg_pid: PidConfig,
    runs_dir_for_stem: Path | None,
) -> Path | None:
    """Resolve the cache file path.

    Shared (`legend_key` set) takes priority over per-diagram (`runs_dir_for_stem`).
    Returns None if neither is provided — caller then runs uncached.
    """
    if legend_key:
        return Path(cfg_pid.legends_dir) / f"{legend_key}.json"
    if runs_dir_for_stem:
        return Path(runs_dir_for_stem) / "legend.cache.json"
    return None


def _legend_extractor_fingerprint(cfg: Config, client: LLMClient) -> str:
    """Fingerprint every input that can change cached legend semantics."""
    payload = {
        "version": LEGEND_EXTRACTOR_VERSION,
        "model": client.config.model,
        "reasoning_mode": client.config.reasoning_mode,
        "thumbnail_max_dim": cfg.pid.legend_thumb_max_dim,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached(
    path: Path | None,
    expected_hash: str,
    expected_fingerprint: str,
    *,
    allow_partial: bool = False,
) -> LegendPack | None:
    """Return a cache only when its source and extraction contract match."""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pack = LegendPack.model_validate(data)
    except (json.JSONDecodeError, ValidationError, OSError):
        # Corrupt or schema-shifted cache -> treat as miss; will be overwritten.
        return None
    if not allow_partial:
        from diagex.extractors.legend_rows import suspicious_legend_label
        if any(e.attributes.get("row_status") == "accept" and suspicious_legend_label(e.label) for e in pack.entries):
            return None
    if (
        pack.schema_version != LEGEND_CACHE_SCHEMA_VERSION
        or pack.source_hash != expected_hash
        or pack.extractor_fingerprint != expected_fingerprint
        or (
            not allow_partial
            and any(
                region.status != "complete"
                or (
                    region.source_row_id
                    and region.entry_count == 0
                    and region.verification_passes < 2
                )
                for region in pack.coverage
            )
        )
    ):
        return None
    return pack


def _persist(path: Path | None, pack: LegendPack) -> Path | None:
    """Write `pack` to `path`. Creates parent dirs. Returns the path or None."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    # model_dump_json handles Path / Literal / nested models cleanly.
    atomic_write_text(path, pack.model_dump_json(indent=2))
    return path


# ---------------------------------------------------------------------------
# Image / page helpers
# ---------------------------------------------------------------------------


def _page_png_bytes(page: DiagramPage) -> bytes:
    """Serialise a page image to PNG bytes for hashing (deterministic)."""
    if page.image is None:
        return b""
    buf = io.BytesIO()
    page.image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _iter_legend_pages(source: DiagramSource, indices: list[int]) -> list[DiagramPage]:
    """Materialise only the requested page indices (0-based).

    Streams pages and keeps a reference to each one we want — callers
    subsequently run the extraction agent over the kept pages.
    """
    wanted = set(indices)
    kept: list[DiagramPage] = []
    for page in iter_pages(source):
        if page.page_index in wanted:
            kept.append(page)
        if len(kept) == len(wanted):
            break
    return kept


def _crop_region(page: DiagramPage, x: int, y: int, w: int, h: int) -> Image.Image:
    """Clip a region to the page bounds and return the crop."""
    if page.image is None:
        raise ValueError(f"Page {page.page_index} has no image; cannot crop region.")
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(page.width, int(x) + int(w))
    y1 = min(page.height, int(y) + int(h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty region: ({x},{y},{w},{h}) on page {page.width}x{page.height}")
    return page.image.crop((x0, y0, x1, y1))


def _thumbnail_b64(img: Image.Image, max_dim: int) -> str:
    """Downsize to `max_dim` long side, encode PNG -> base64."""
    thumb = img.copy()
    thumb.thumbnail((max_dim, max_dim), Image.BILINEAR)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def validate_legend_image_bytes(data: bytes) -> str | None:
    """Return a rejection reason for blank/corrupt legend thumbnails."""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            gray = opened.convert("L")
            if gray.width < 2 or gray.height < 2:
                return "image_too_small"
            extrema = ImageStat.Stat(gray).extrema[0]
            if extrema[0] >= 245:
                return "blank_image"
            histogram = gray.histogram()
            pixel_count = gray.width * gray.height
            dark_fraction = sum(histogram[:235]) / pixel_count
            if dark_fraction > 0.97:
                return "solid_dark_image"
            # Permit sparse line-style legend samples, but require several
            # genuinely dark pixels so antialiasing noise is not accepted.
            if sum(histogram[:220]) < 3:
                return "blank_image"
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        return f"invalid_image:{exc!r}"
    return None


def _image_rejection_reason(image: Image.Image) -> str | None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return validate_legend_image_bytes(buf.getvalue())


def _normalise_legend_text(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _intersection_area(a: BBox, b: BBox) -> int:
    return max(0, min(a.x2, b.x2) - max(a.x, b.x)) * max(0, min(a.y2, b.y2) - max(a.y, b.y))


def _bbox_distance(a: BBox, b: BBox) -> tuple[int, int]:
    dx = max(0, a.x - b.x2, b.x - a.x2)
    dy = max(0, a.y - b.y2, b.y - a.y2)
    return dx, dy


def _matching_label_span(label: str, evidence: PageEvidence) -> TextEvidence | None:
    target = _normalise_legend_text(label)
    if not target:
        return None
    exact = [span for span in evidence.text_spans if _normalise_legend_text(span.text) == target]
    if not exact:
        return None

    # Duplicate labels can occur in notes. Prefer the occurrence with nearby
    # compact vector geometry, which is the likely legend row.
    def score(span: TextEvidence) -> tuple[int, int, int]:
        height = max(1, span.bbox.h)
        nearby = sum(
            1
            for path in evidence.paths
            if _bbox_distance(span.bbox, path.bbox)[0] <= height * 14
            and _bbox_distance(span.bbox, path.bbox)[1] <= height * 3
            and path.bbox.w <= height * 14
            and path.bbox.h <= height * 10
        )
        return (-nearby, span.bbox.y, span.bbox.x)

    return min(exact, key=score)


def _cluster_path_boxes(paths: list[PathEvidence], gap: int) -> list[list[BBox]]:
    clusters: list[list[BBox]] = []
    for path in paths:
        box = path.bbox
        touching: list[int] = []
        for index, cluster in enumerate(clusters):
            if any(
                _bbox_distance(box, existing)[0] <= gap and _bbox_distance(box, existing)[1] <= gap
                for existing in cluster
            ):
                touching.append(index)
        if not touching:
            clusters.append([box])
            continue
        first = touching[0]
        clusters[first].append(box)
        for index in reversed(touching[1:]):
            clusters[first].extend(clusters.pop(index))
    return clusters


def _union_boxes(boxes: list[BBox]) -> BBox:
    x0 = min(box.x for box in boxes)
    y0 = min(box.y for box in boxes)
    x1 = max(box.x2 for box in boxes)
    y1 = max(box.y2 for box in boxes)
    return BBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))


def _native_symbol_bbox(label: TextEvidence, evidence: PageEvidence) -> BBox | None:
    """Find compact vector geometry immediately beside a native legend label."""
    height = max(8, label.bbox.h)
    row_top = label.bbox.y - 3 * height
    row_bottom = label.bbox.y2 + 3 * height
    candidates = [
        path
        for path in evidence.paths
        if path.bbox.y2 >= row_top
        and path.bbox.y <= row_bottom
        and path.bbox.w <= 14 * height
        and path.bbox.h <= 10 * height
        and (
            0 <= label.bbox.x - path.bbox.x2 <= 14 * height
            or 0 <= path.bbox.x - label.bbox.x2 <= 14 * height
        )
    ]
    if not candidates:
        return None

    clusters = _cluster_path_boxes(candidates, gap=max(3, height // 2))
    ranked: list[tuple[float, BBox]] = []
    for cluster in clusters:
        box = _union_boxes(cluster)
        dx, dy = _bbox_distance(box, label.bbox)
        if _intersection_area(box, label.bbox):
            continue
        centre_delta = abs((box.y + box.h / 2) - (label.bbox.y + label.bbox.h / 2))
        tiny_penalty = 5.0 if box.w < height // 3 and box.h < height // 3 else 0.0
        # Nearby, vertically aligned, multi-primitive clusters are most likely
        # the symbol cell rather than a table border or an unrelated glyph.
        score = dx / height + centre_delta / height + dy / height
        score += tiny_penalty - min(len(cluster), 12) * 0.12
        ranked.append((score, box))
    if not ranked:
        return None
    box = min(ranked, key=lambda item: item[0])[1]
    padding = max(3, height // 3)
    return BBox(
        x=max(0, box.x - padding),
        y=max(0, box.y - padding),
        w=min(evidence.width, box.x2 + padding) - max(0, box.x - padding),
        h=min(evidence.height, box.y2 + padding) - max(0, box.y - padding),
    )


def _model_crop_is_suspicious(
    bbox: BBox,
    label_span: TextEvidence | None,
    evidence: PageEvidence | None,
) -> str | None:
    if evidence is None:
        return None
    if label_span is not None:
        height = max(8, label_span.bbox.h)
        dx, dy = _bbox_distance(bbox, label_span.bbox)
        if dx > 20 * height or dy > 4 * height:
            return "bbox_far_from_label"
    for span in evidence.text_spans:
        if _intersection_area(bbox, span.bbox) == 0:
            continue
        text = span.text.strip()
        if any("\u4e00" <= character <= "\u9fff" for character in text) or len(text) > 2:
            return "bbox_overlaps_printed_text"
    return None


def _select_legend_crop(
    *,
    page: DiagramPage,
    evidence: PageEvidence | None,
    label: str,
    model_bbox: BBox,
) -> tuple[Image.Image | None, BBox | None, BBox | None, str | None, str]:
    """Select a trustworthy model crop or recover one from native PDF evidence."""
    if page.image is None:
        return None, None, None, None, "unavailable"
    label_span = _matching_label_span(label, evidence) if evidence is not None else None
    model_reason = _model_crop_is_suspicious(model_bbox, label_span, evidence)

    # On a vector PDF, the exact native label plus adjacent drawing paths is
    # better evidence than a model-predicted rectangle.  Prefer reconstructing
    # that symbol cell even when the model crop happens to pass a coarse blank
    # check; this avoids accepted but clipped strokes and table borders.
    if evidence is not None and label_span is not None and evidence.paths:
        recovered_bbox = _native_symbol_bbox(label_span, evidence)
        if recovered_bbox is not None:
            recovered = page.image.crop(
                (recovered_bbox.x, recovered_bbox.y, recovered_bbox.x2, recovered_bbox.y2)
            )
            if _image_rejection_reason(recovered) is None:
                return (
                    recovered,
                    recovered_bbox,
                    label_span.bbox,
                    "native_text_paths",
                    "recovered",
                )
    if (
        model_bbox.w > 0
        and model_bbox.h > 0
        and model_bbox.x < page.width
        and model_bbox.y < page.height
        and model_bbox.x2 > 0
        and model_bbox.y2 > 0
    ):
        model_crop = page.image.crop(
            (
                max(0, model_bbox.x),
                max(0, model_bbox.y),
                min(page.width, model_bbox.x2),
                min(page.height, model_bbox.y2),
            )
        )
        image_reason = _image_rejection_reason(model_crop)
        if model_reason is None and image_reason is None:
            return (
                model_crop,
                model_bbox,
                label_span.bbox if label_span else None,
                "model_bbox",
                "accepted",
            )

    rejected = "rejected_text_overlap" if model_reason else "rejected_blank"
    return None, model_bbox, label_span.bbox if label_span else None, None, rejected


# ---------------------------------------------------------------------------
# Auto-detect: "is this page a legend?"
# ---------------------------------------------------------------------------


_AUTO_DETECT_MIN_DIM = 200  # pages smaller than this in either axis are skipped


def _image_to_content_block(img: Image.Image, max_dim: int = 1200) -> dict[str, Any]:
    """Downsampled image packaged as a size-safe Anthropic image content-block."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return encode_image_block(img)


def _auto_detect_page(
    *,
    page: DiagramPage,
    client: LLMClient,
    cost_tracker: CostTracker,
) -> tuple[bool, tuple[int, int, int, int] | None]:
    """One-shot yes/no classifier over the page overview.

    Returns `(is_legend, bbox_or_None)` where bbox is in page-pixel coords.
    `bbox` is None when the model answered 'full' or omitted a box.
    """
    if page.image is None:
        return False, None
    if min(page.width, page.height) < _AUTO_DETECT_MIN_DIM:
        return False, None

    # System prompt: stable short instruction. Per-page framing is trivial
    # text so we don't bother with cache_control here.
    system_blocks = [
        {
            "type": "text",
            "text": (
                "You are a binary classifier over engineering-drawing pages. "
                "Answer the user's yes/no question about whether the supplied "
                "page is a symbol or abbreviation legend. Never add commentary."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_text = (
        f"Page size: {page.width}x{page.height} px. Is this page (or a corner "
        f"of it) a symbol or abbreviation legend for an engineering drawing? "
        f"Reply with exactly one of:\n"
        f"  Line 1: 'yes' or 'no'.\n"
        f"  Line 2 (only if yes): 'full' if the whole page is legend, else "
        f"'x,y,w,h' in page pixels bounding the legend region."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                _image_to_content_block(page.image, max_dim=1200),
            ],
        }
    ]

    resp = client.messages_create(
        system=system_blocks,
        messages=messages,
        tools=None,
        max_tokens=128,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
    )

    cost_tracker.record(resp, step=0, page_index=page.page_index)
    text = _extract_text(resp).strip()
    return _parse_auto_detect(text, page_w=page.width, page_h=page.height)


def _extract_text(resp: Any) -> str:
    """Concatenate text blocks from an Anthropic Messages response."""
    content = getattr(resp, "content", None) or []
    out: list[str] = []
    for b in content:
        t = getattr(b, "type", None) if not isinstance(b, dict) else b.get("type")
        if t == "text":
            out.append(getattr(b, "text", "") if not isinstance(b, dict) else (b.get("text") or ""))
    return "\n".join(out)


def _parse_auto_detect(
    text: str, *, page_w: int, page_h: int
) -> tuple[bool, tuple[int, int, int, int] | None]:
    """Parse the yes/no + bbox response.

    Robust to stray whitespace, trailing punctuation, and case; bails to
    'no' on anything ambiguous.
    """
    lines = [ln.strip().strip(".").lower() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, None
    first = lines[0]
    if first.startswith("no"):
        return False, None
    if not first.startswith("yes"):
        return False, None
    # yes; look for a bbox on any subsequent line.
    for ln in lines[1:]:
        if ln == "full":
            return True, None
        parts = [p.strip() for p in ln.replace(";", ",").split(",")]
        if len(parts) == 4:
            try:
                x, y, w, h = (int(float(p)) for p in parts)
            except ValueError:
                continue
            # Clip to page and drop nonsense boxes.
            x = max(0, min(page_w, x))
            y = max(0, min(page_h, y))
            w = max(0, min(page_w - x, w))
            h = max(0, min(page_h - y, h))
            if w > 0 and h > 0:
                return True, (x, y, w, h)
    # yes without a parseable bbox — treat as full-page legend.
    return True, None


# ---------------------------------------------------------------------------
# Extraction agent pass
# ---------------------------------------------------------------------------


def _native_page_evidence(page: DiagramPage, source_path: Path) -> PageEvidence:
    from diagex.vision.evidence import extract_page_evidence

    return extract_page_evidence(page=page, source_path=source_path)


def _extract_from_page(
    *,
    page: DiagramPage,
    region: tuple[int, int, int, int] | None,
    client: LLMClient,
    cost_tracker: CostTracker,
    cfg: Config,
    page_evidence: PageEvidence | None = None,
    coverage: list[LegendRegionCoverage] | None = None,
    prior: LegendPack | None = None,
    run_state: Any = None,
) -> list[LegendEntry]:
    """Classify whole native rows; retain detail-region navigation for raster input.

    Uncertain rows survive with partial coverage. Built-in fallback entries never
    count as inspected source rows.
    """
    x, y, w, h = region or (0, 0, page.width, page.height)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(page.width, x + w), min(page.height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty legend extraction region")
    if (
        page_evidence is not None
        and not page_evidence.is_scanned
        and page_evidence.paths
        and page_evidence.text_spans
    ):
        from diagex.extractors.legend_rows import classify_native_legend

        return classify_native_legend(
            page=page,
            evidence=page_evidence,
            region=(x0, y0, x1 - x0, y1 - y0),
            client=client,
            cost_tracker=cost_tracker,
            cfg=cfg,
            coverage=coverage,
            prior=prior,
            run_state=run_state,
        )
    area = page.model_copy(update={"width": x1 - x0, "height": y1 - y0})
    strategy = AspectAwareStrategy(
        max_tokens_per_tile=cfg.tiling.max_tokens_per_tile,
        overlap_frac=cfg.tiling.overlap_frac,
        token_per_pixel=cfg.tiling.token_per_pixel,
    )
    regions = strategy.plan(area)
    profile = EFFORT_PROFILES["medium"]
    entries: list[LegendEntry] = []
    for index, (rx, ry, rw, rh) in enumerate(regions):
        origin = (x0 + rx, y0 + ry)
        box = BBox(x=origin[0], y=origin[1], w=rw, h=rh)
        work_page = page.model_copy(
            update={
                "image": _crop_region(page, box.x, box.y, box.w, box.h),
                "width": rw,
                "height": rh,
                "source_ref": f"{page.source_ref}#legend-region-{index}",
            }
        )
        # A single mandatory full-detail tile makes coverage measurable. The
        # next region is scheduled even if this agent finishes early or fails.
        detail = Tile(
            id=f"legend-{index}", page_index=page.page_index, bbox=BBox(x=0, y=0, w=rw, h=rh)
        )
        vp = ViewProvider(work_page, [detail])
        state = AgentState(
            question=(
                "Extract every legend row in this detail region, including every "
                "column and the bottom rows. Inspect the detail tile before finishing. "
                "Keep clipped/unclear rows explicit; do not invent labels or classes."
            ),
            page=work_page,
        )
        runtime = ReactRuntime(
            client=client,
            system_blocks=build_legend_system_prompt(effort_max_steps=profile.max_steps),
            cost_tracker=cost_tracker,
            budgets=cfg.budgets,
        )
        try:
            runtime.run(
                state=state,
                view_provider=vp,
                run_cfg=RunConfig(
                    effort="medium",
                    require_tile_coverage=True,
                    no_progress_step_limit=3,
                ),
            )
        except Exception as exc:
            if is_non_retryable_api_error(exc):
                raise
            state.completion_status = "partial"
            state.completion_reason = f"extraction failed: {exc}"
        # Scope native label recovery to this region, especially for repeated
        # labels in separate columns. Geometry stays in original page pixels.
        scoped = page_evidence
        if scoped is not None:
            scoped = scoped.model_copy(
                update={
                    "text_spans": [t for t in scoped.text_spans if _intersection_area(t.bbox, box)],
                    "paths": [p for p in scoped.paths if _intersection_area(p.bbox, box)],
                }
            )
        region_entries = []
        for annotation in state.annotations.all():
            entry = _annotation_to_entry(
                annotation=annotation,
                page=page,
                origin=origin,
                cfg_pid=cfg.pid,
                page_evidence=scoped,
            )
            if entry is not None:
                region_entries.append(entry)
        entries.extend(region_entries)
        if coverage is not None:
            complete = state.completion_status == "complete" and bool(state.tile_fetch_counts)
            coverage.append(
                LegendRegionCoverage(
                    page_index=page.page_index,
                    bbox=box,
                    status="complete" if complete else "partial",
                    failure_kind=None if complete else "not_inspected",
                    entry_count=len(region_entries),
                    reason=state.completion_reason or "Detail inspection not completed",
                )
            )
    return entries


def _annotation_to_entry(
    *,
    annotation: Any,
    page: DiagramPage,
    origin: tuple[int, int],
    cfg_pid: PidConfig,
    page_evidence: PageEvidence | None = None,
) -> LegendEntry | None:
    """Convert a legend-pass Annotation into a LegendEntry.

    Rules:
      - Require a non-empty label. Illegible entries are skipped.
      - `legend_symbol_class` from attributes becomes `symbol_class`; fall
        back to a heuristic from `kind` when missing.
      - Crop the symbol thumbnail from the page image at bbox_global shifted
        back into the original page's coordinate system (via `origin`).
    """
    label = (annotation.label or "").strip()
    if not label:
        return None

    attrs = dict(annotation.attributes or {})
    symbol_class = str(attrs.pop("legend_symbol_class", "") or "").strip()
    if not symbol_class:
        symbol_class = _default_symbol_class_for_kind(annotation.kind)
    description = str(attrs.pop("legend_description", "") or "").strip() or None
    legend_kind = str(attrs.pop("legend_kind", "") or "").strip()
    standard_val = str(attrs.pop("legend_standard", "") or "").strip() or None
    standard: SymbolStandard | None
    if standard_val in ("isa-5.1", "iso-10628", "sama", "none"):
        standard = standard_val  # type: ignore[assignment]
    else:
        standard = None

    # Reproject bbox back to the original page if the extraction ran over a
    # cropped sub-region; origin is (0,0) otherwise so this is a no-op.
    bbox = annotation.bbox_global
    page_bbox = BBox(
        x=bbox.x + origin[0],
        y=bbox.y + origin[1],
        w=bbox.w,
        h=bbox.h,
    )
    image_b64: str | None = None
    source_bbox: BBox | None = page_bbox
    source_label_bbox: BBox | None = None
    crop_method: str | None = None
    crop_quality = "unavailable"
    if legend_kind == "abbreviation":
        crop_quality = "omitted_abbreviation"
    else:
        try:
            crop, source_bbox, source_label_bbox, crop_method, crop_quality = _select_legend_crop(
                page=page,
                evidence=page_evidence,
                label=label,
                model_bbox=page_bbox,
            )
            if crop is not None:
                image_b64 = _thumbnail_b64(crop, cfg_pid.legend_thumb_max_dim)
        except Exception:
            image_b64 = None
            crop_quality = "unavailable"

    # Normalise the kind into the LegendEntry Literal set. The Annotation's
    # kind set is broader (includes text / note / opc); fold those into
    # "other" rather than guessing.
    entry_kind_map = {
        "equipment": "equipment",
        "instrument": "instrument",
        "line": "line",
        "connection": "connector",
    }
    entry_kind = entry_kind_map.get(annotation.kind, "other")

    if legend_kind:
        attrs.setdefault("legend_kind", legend_kind)

    try:
        return LegendEntry(
            label=label,
            description=description,
            symbol_class=symbol_class or "unclassified_equipment",
            kind=entry_kind,  # type: ignore[arg-type]
            standard=standard,
            image_b64=image_b64,
            attributes={k: str(v) for k, v in attrs.items()},
            source="legend_extracted",
            source_page_index=page.page_index,
            source_bbox=source_bbox,
            source_label_bbox=source_label_bbox,
            source_view=annotation.source_view,
            crop_method=crop_method,
            crop_quality=crop_quality,
        )
    except ValidationError:
        return None


def _default_symbol_class_for_kind(kind: str) -> str:
    """Fallback when the agent forgot `legend_symbol_class`."""
    return {
        "instrument": "unclassified_instrument",
        "equipment": "unclassified_equipment",
        "line": "line",
        "connection": "connector",
    }.get(kind, "unclassified_equipment")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def resolve_legend(
    *,
    source: DiagramSource,
    symbol_standard: SymbolStandard,
    cfg: Config,
    client: LLMClient,
    cost_tracker: CostTracker,
    legend_path: Path | None = None,
    legend_pages: list[int] | None = None,
    legend_region: tuple[int, int, int, int, int] | None = None,
    no_legend: bool = False,
    legend_key: str | None = None,
    runs_dir_for_stem: Path | None = None,
    reporter: ProgressReporter | None = None,
    evidence_pages: list[PageEvidence] | None = None,
    fresh: bool = False,
) -> LegendResolution:
    """Resolve, extract, cache, merge, and budget-split the legend.

    See module docstring for the resolution order. Returns a LegendResolution
    the Phase 2 orchestrator feeds into `build_pid_system_prompt` + the
    lookup_symbol tool registration.
    """
    # Input-selector mutual exclusion. `legend_key` and `runs_dir_for_stem`
    # are cache modifiers; they don't count.
    selectors = [
        legend_path is not None,
        legend_pages is not None,
        legend_region is not None,
        no_legend,
    ]
    if sum(selectors) > 1:
        raise ValueError(
            "resolve_legend: --legend / --legend-pages / --legend-region / "
            "--no-legend are mutually exclusive."
        )

    builtin = load_builtin_pack(symbol_standard)
    progress = reporter or NullReporter()
    cache_path = _cache_path_for(
        legend_key=legend_key,
        cfg_pid=cfg.pid,
        runs_dir_for_stem=runs_dir_for_stem,
    )
    evidence_by_page = {page.page_index: page for page in (evidence_pages or [])}
    extractor_fingerprint = _legend_extractor_fingerprint(cfg, client)

    # --- rule 4: --no-legend -------------------------------------------------
    if no_legend:
        merged = LegendPack(
            standard=symbol_standard,
            source_ref=f"{source.path.stem}#no-legend",
            notes="--no-legend; built-in only" if builtin.entries else "--no-legend; empty",
        ).merge(builtin)
        return LegendResolution(
            pack=merged,
            budget=apply_budget(merged, cfg.pid),
            source="no_legend",
            cache_path=None,
        )

    # --- rule 1: explicit --legend <path> -----------------------------------
    if legend_path is not None:
        return _resolve_from_explicit_path(
            source=source,
            legend_path=legend_path,
            symbol_standard=symbol_standard,
            builtin=builtin,
            cfg=cfg,
            client=client,
            cost_tracker=cost_tracker,
            cache_path=cache_path,
            extractor_fingerprint=extractor_fingerprint,
            fresh=fresh,
        )

    # --- rule 2: --legend-pages ---------------------------------------------
    if legend_pages is not None:
        return _resolve_from_pages(
            source=source,
            page_indices=legend_pages,
            symbol_standard=symbol_standard,
            builtin=builtin,
            cfg=cfg,
            client=client,
            cost_tracker=cost_tracker,
            cache_path=cache_path,
            evidence_by_page=evidence_by_page,
            extractor_fingerprint=extractor_fingerprint,
            fresh=fresh,
        )

    # --- rule 3: --legend-region --------------------------------------------
    if legend_region is not None:
        return _resolve_from_region(
            source=source,
            region=legend_region,
            symbol_standard=symbol_standard,
            builtin=builtin,
            cfg=cfg,
            client=client,
            cost_tracker=cost_tracker,
            cache_path=cache_path,
            evidence_by_page=evidence_by_page,
            extractor_fingerprint=extractor_fingerprint,
            fresh=fresh,
        )

    # --- rule 5 (default): auto-detect --------------------------------------
    return _resolve_auto(
        source=source,
        symbol_standard=symbol_standard,
        builtin=builtin,
        cfg=cfg,
        client=client,
        cost_tracker=cost_tracker,
        cache_path=cache_path,
        reporter=progress,
        evidence_by_page=evidence_by_page,
        extractor_fingerprint=extractor_fingerprint,
            fresh=fresh,
    )


# ---------------------------------------------------------------------------
# Per-rule helpers
# ---------------------------------------------------------------------------


def _finalise(
    *,
    extracted: LegendPack,
    builtin: LegendPack,
    symbol_standard: SymbolStandard,
    cfg_pid: PidConfig,
    source_label: str,
    cache_path: Path | None,
) -> LegendResolution:
    """Persist extracted, merge with built-in (extracted wins), budget-split."""
    persisted = _persist(cache_path, extracted)
    merged = extracted.merge(builtin)
    # Merge does not overwrite `standard`; force it to the caller-requested
    # value so downstream consumers always see a concrete standard.
    merged = LegendPack(
        schema_version=merged.schema_version,
        source_hash=merged.source_hash,
        extractor_fingerprint=merged.extractor_fingerprint,
        source_ref=merged.source_ref,
        standard=symbol_standard,
        entries=merged.entries,
        coverage=merged.coverage,
        notes=merged.notes,
    )
    return LegendResolution(
        pack=merged,
        budget=apply_budget(merged, cfg_pid),
        source=source_label,
        cache_path=persisted,
    )


def _resolve_from_explicit_path(
    *,
    source: DiagramSource,
    legend_path: Path,
    symbol_standard: SymbolStandard,
    builtin: LegendPack,
    cfg: Config,
    client: LLMClient,
    cost_tracker: CostTracker,
    cache_path: Path | None,
    extractor_fingerprint: str,
    fresh: bool = False,
) -> LegendResolution:
    if not legend_path.exists():
        raise FileNotFoundError(f"--legend file not found: {legend_path}")
    src_hash = compute_source_hash(path=legend_path)

    cached = None if fresh else _load_cached(cache_path, src_hash, extractor_fingerprint)
    if cached is not None:
        return _finalise(
            extracted=cached,
            builtin=builtin,
            symbol_standard=symbol_standard,
            cfg_pid=cfg.pid,
            source_label="cache_hit",
            cache_path=cache_path,
        )

    # Extract: load the legend file as its own DiagramSource; walk every page.
    from diagex.extractors.legend_rows import LegendRunState
    from diagex.vision.loader import load as _load  # local import; avoids cycle

    prior = None if fresh else _load_cached(cache_path, src_hash, extractor_fingerprint, allow_partial=True)
    run_state = LegendRunState()
    legend_source = _load(legend_path, tiling=cfg.tiling, scan_cfg=cfg.scan)
    all_entries: list[LegendEntry] = []
    coverage: list[LegendRegionCoverage] = []
    notes = ""
    from diagex.vision.evidence import extract_page_evidence

    for page in iter_pages(legend_source):
        try:
            entries = _extract_from_page(
                page=page,
                region=None,
                client=client,
                cost_tracker=cost_tracker,
                cfg=cfg,
                page_evidence=extract_page_evidence(page=page, source_path=legend_source.path),
                coverage=coverage,
                prior=prior,
                run_state=run_state,
            )
        except Exception as exc:
            if is_non_retryable_api_error(exc):
                raise
            notes = f"extraction failed on page {page.page_index}: {exc}".strip()
            coverage.append(
                LegendRegionCoverage(
                    page_index=page.page_index,
                    bbox=BBox(x=0, y=0, w=page.width, h=page.height),
                    status="partial",
                    failure_kind="not_inspected",
                    reason=notes,
                )
            )
            continue
        all_entries.extend(entries)

    extracted = LegendPack(
        source_hash=src_hash,
        extractor_fingerprint=extractor_fingerprint,
        source_ref=f"file:{legend_path.name}",
        standard=symbol_standard,
        entries=_dedupe(all_entries),
        coverage=coverage,
        notes=notes,
    )
    if not extracted.entries and not notes:
        extracted.notes = "no legend entries extracted"
    return _finalise(
        extracted=extracted,
        builtin=builtin,
        symbol_standard=symbol_standard,
        cfg_pid=cfg.pid,
        source_label="explicit_file",
        cache_path=cache_path,
    )


def _resolve_from_pages(
    *,
    source: DiagramSource,
    page_indices: list[int],
    symbol_standard: SymbolStandard,
    builtin: LegendPack,
    cfg: Config,
    client: LLMClient,
    cost_tracker: CostTracker,
    cache_path: Path | None,
    evidence_by_page: dict[int, PageEvidence],
    extractor_fingerprint: str,
    fresh: bool = False,
) -> LegendResolution:
    pages = _iter_legend_pages(source, page_indices)
    if not pages:
        raise ValueError(f"--legend-pages {page_indices} not found in {source.path.name}")

    # Hash over concatenated page PNGs (stable: same page list -> same hash).
    page_pngs = [_page_png_bytes(p) for p in pages]
    src_hash = compute_source_hash(page_bytes=page_pngs)

    cached = None if fresh else _load_cached(cache_path, src_hash, extractor_fingerprint)
    if cached is not None:
        return _finalise(
            extracted=cached,
            builtin=builtin,
            symbol_standard=symbol_standard,
            cfg_pid=cfg.pid,
            source_label="cache_hit",
            cache_path=cache_path,
        )

    # Matching partial caches are evidence for completed source rows, never a
    # completed legend. Retry only the unresolved rows under a shared guard.
    from diagex.extractors.legend_rows import LegendRunState

    prior = None if fresh else _load_cached(cache_path, src_hash, extractor_fingerprint, allow_partial=True)
    run_state = LegendRunState()
    all_entries: list[LegendEntry] = []
    coverage: list[LegendRegionCoverage] = []
    notes = ""
    for page in pages:
        try:
            entries = _extract_from_page(
                page=page,
                region=None,
                client=client,
                cost_tracker=cost_tracker,
                cfg=cfg,
                page_evidence=evidence_by_page.get(page.page_index)
                or _native_page_evidence(page, source.path),
                coverage=coverage,
                prior=prior,
                run_state=run_state,
            )
        except Exception as exc:
            if is_non_retryable_api_error(exc):
                raise
            notes = f"extraction failed on page {page.page_index}: {exc}".strip()
            coverage.append(
                LegendRegionCoverage(
                    page_index=page.page_index,
                    bbox=BBox(x=0, y=0, w=page.width, h=page.height),
                    status="partial",
                    failure_kind="not_inspected",
                    reason=notes,
                )
            )
            continue
        all_entries.extend(entries)

    extracted = LegendPack(
        source_hash=src_hash,
        extractor_fingerprint=extractor_fingerprint,
        source_ref=f"{source.path.stem}#pages={','.join(str(i) for i in page_indices)}",
        standard=symbol_standard,
        entries=_dedupe(all_entries),
        coverage=coverage,
        notes=notes,
    )
    return _finalise(
        extracted=extracted,
        builtin=builtin,
        symbol_standard=symbol_standard,
        cfg_pid=cfg.pid,
        source_label="explicit_pages",
        cache_path=cache_path,
    )


def _resolve_from_region(
    *,
    source: DiagramSource,
    region: tuple[int, int, int, int, int],
    symbol_standard: SymbolStandard,
    builtin: LegendPack,
    cfg: Config,
    client: LLMClient,
    cost_tracker: CostTracker,
    cache_path: Path | None,
    evidence_by_page: dict[int, PageEvidence],
    extractor_fingerprint: str,
    fresh: bool = False,
) -> LegendResolution:
    page_idx, x, y, w, h = region
    pages = _iter_legend_pages(source, [page_idx])
    if not pages:
        raise ValueError(f"--legend-region page {page_idx} not found in {source.path.name}")
    page = pages[0]

    crop = _crop_region(page, x, y, w, h)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    src_hash = compute_source_hash(region_bytes=buf.getvalue())

    cached = None if fresh else _load_cached(cache_path, src_hash, extractor_fingerprint)
    if cached is not None:
        return _finalise(
            extracted=cached,
            builtin=builtin,
            symbol_standard=symbol_standard,
            cfg_pid=cfg.pid,
            source_label="cache_hit",
            cache_path=cache_path,
        )

    coverage: list[LegendRegionCoverage] = []
    try:
        entries = _extract_from_page(
            page=page,
            region=(x, y, w, h),
            client=client,
            cost_tracker=cost_tracker,
            cfg=cfg,
            page_evidence=evidence_by_page.get(page.page_index)
            or _native_page_evidence(page, source.path),
            coverage=coverage,
            prior=None if fresh else _load_cached(cache_path, src_hash, extractor_fingerprint, allow_partial=True),
        )
        notes = ""
    except Exception as exc:
        if is_non_retryable_api_error(exc):
            raise
        entries = []
        notes = f"extraction failed: {exc}"
        coverage.append(
            LegendRegionCoverage(
                page_index=page.page_index,
                bbox=BBox(x=x, y=y, w=w, h=h),
                status="partial",
                failure_kind="not_inspected",
                reason=notes,
            )
        )

    extracted = LegendPack(
        source_hash=src_hash,
        extractor_fingerprint=extractor_fingerprint,
        source_ref=f"{source.path.stem}#region=p{page_idx}:{x},{y},{w},{h}",
        standard=symbol_standard,
        entries=_dedupe(entries),
        coverage=coverage,
        notes=notes,
    )
    return _finalise(
        extracted=extracted,
        builtin=builtin,
        symbol_standard=symbol_standard,
        cfg_pid=cfg.pid,
        source_label="explicit_region",
        cache_path=cache_path,
    )


def _resolve_auto(
    *,
    source: DiagramSource,
    symbol_standard: SymbolStandard,
    builtin: LegendPack,
    cfg: Config,
    client: LLMClient,
    cost_tracker: CostTracker,
    cache_path: Path | None,
    reporter: ProgressReporter,
    evidence_by_page: dict[int, PageEvidence],
    extractor_fingerprint: str,
    fresh: bool = False,
) -> LegendResolution:
    """Default path: classify each page, extract detected legends, merge."""
    detected: list[tuple[DiagramPage, tuple[int, int, int, int] | None]] = []
    detected_page_bytes: list[bytes] = []
    raw_total = source.metadata.get("page_count")
    total_pages = raw_total if isinstance(raw_total, int) else None
    reporter.on_phase_start(name="legend scan", total_items=total_pages)
    processed_pages = 0
    for page in iter_pages(source):
        processed_pages += 1
        reporter.on_phase_item_start(
            item=processed_pages,
            total_items=total_pages,
            label=f"classifying page {page.page_index + 1}",
        )
        try:
            is_legend, bbox = _auto_detect_page(
                page=page,
                client=client,
                cost_tracker=cost_tracker,
            )
        except Exception as exc:  # preserve best-effort legend detection
            reporter.on_phase_item_end(detail=str(exc), is_error=True)
            if is_non_retryable_api_error(exc):
                raise
            continue
        reporter.on_phase_item_end(detail="legend detected" if is_legend else "not a legend")
        if is_legend:
            detected.append((page, bbox))
            detected_page_bytes.append(_page_png_bytes(page))

    reporter.on_phase_end(detail=f"{len(detected)} legend page(s) detected")

    if not detected:
        merged = LegendPack(
            standard=symbol_standard,
            source_ref=f"{source.path.stem}#no-legend-detected",
            notes="auto-detect found no legend page",
        ).merge(builtin)
        return LegendResolution(
            pack=merged,
            budget=apply_budget(merged, cfg.pid),
            source="built_in_only",
            cache_path=None,
        )

    # Cache key includes the detected page bytes AND each bbox (or 'full')
    # so changing either forces re-extraction.
    h = hashlib.sha256()
    for (_, bbox), b in zip(detected, detected_page_bytes, strict=True):
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)
        marker = b"full" if bbox is None else f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}".encode()
        h.update(len(marker).to_bytes(2, "big"))
        h.update(marker)
    src_hash = h.hexdigest()

    cached = None if fresh else _load_cached(cache_path, src_hash, extractor_fingerprint)
    if cached is not None:
        return _finalise(
            extracted=cached,
            builtin=builtin,
            symbol_standard=symbol_standard,
            cfg_pid=cfg.pid,
            source_label="cache_hit",
            cache_path=cache_path,
        )

    all_entries: list[LegendEntry] = []
    coverage: list[LegendRegionCoverage] = []
    notes_parts: list[str] = []
    reporter.on_phase_start(name="legend extraction", total_items=len(detected))
    for item, (page, bbox) in enumerate(detected, start=1):
        reporter.on_phase_item_start(
            item=item,
            total_items=len(detected),
            label=f"extracting page {page.page_index + 1}",
        )
        try:
            entries = _extract_from_page(
                page=page,
                region=bbox,
                client=client,
                cost_tracker=cost_tracker,
                cfg=cfg,
                page_evidence=evidence_by_page.get(page.page_index)
                or _native_page_evidence(page, source.path),
                coverage=coverage,
            )
        except Exception as exc:
            notes_parts.append(f"page {page.page_index}: {exc}")
            reporter.on_phase_item_end(detail=str(exc), is_error=True)
            if is_non_retryable_api_error(exc):
                raise
            continue
        all_entries.extend(entries)
        reporter.on_phase_item_end(detail=f"{len(entries)} entries")

    reporter.on_phase_end(detail=f"{len(all_entries)} entries extracted")

    extracted = LegendPack(
        source_hash=src_hash,
        extractor_fingerprint=extractor_fingerprint,
        source_ref=f"{source.path.stem}#auto-detected",
        standard=symbol_standard,
        entries=_dedupe(all_entries),
        coverage=coverage,
        notes="; ".join(notes_parts),
    )
    return _finalise(
        extracted=extracted,
        builtin=builtin,
        symbol_standard=symbol_standard,
        cfg_pid=cfg.pid,
        source_label="auto_detected",
        cache_path=cache_path,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _dedupe(entries: list[LegendEntry]) -> list[LegendEntry]:
    """Consolidate source rows, retain repeated instances, and sanitize image evidence."""
    seen: set[str] = set()
    out: list[LegendEntry] = []
    for e in entries:
        key = e.source_row_id or e.label.strip().lower()
        if key in seen and e.source_row_id:
            index = next(i for i, row in enumerate(out) if row.source_row_id == e.source_row_id)
            out[index] = merge_source_row(out[index], e)
            continue
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(e)

    sanitized: list[LegendEntry] = []
    hashes: dict[str, list[int]] = {}
    for entry in out:
        updated = entry
        issue: str | None = None
        if entry.attributes.get("legend_kind") == "abbreviation" and entry.image_b64:
            issue = "abbreviation_has_no_symbol_image"
            updated = entry.model_copy(
                update={"image_b64": None, "crop_quality": "omitted_abbreviation"}
            )
        elif entry.image_b64:
            try:
                raw = entry.image_bytes()
                issue = validate_legend_image_bytes(raw or b"")
            except Exception as exc:  # noqa: BLE001 - malformed model/cache image
                issue = f"invalid_image:{exc!r}"
            if issue:
                attrs = dict(entry.attributes)
                attrs["legend_image_issue"] = issue
                updated = entry.model_copy(
                    update={
                        "image_b64": None,
                        "attributes": attrs,
                        "crop_quality": "rejected_blank",
                    }
                )
        sanitized.append(updated)
        if updated.image_b64 and not updated.source_row_id:
            digest = hashlib.sha256(updated.image_bytes() or b"").hexdigest()
            hashes.setdefault(digest, []).append(len(sanitized) - 1)

    # Identical pixels assigned to different labels usually indicate a stale
    # coordinate frame or whitespace crop.  Omit the visual evidence for all
    # members instead of arbitrarily trusting the first label.
    for indices in hashes.values():
        labels = {sanitized[index].label.casefold() for index in indices}
        if len(indices) < 2 or len(labels) < 2:
            continue
        for index in indices:
            entry = sanitized[index]
            attrs = dict(entry.attributes)
            attrs["legend_image_issue"] = "duplicate_pixels_for_different_labels"
            sanitized[index] = entry.model_copy(
                update={
                    "image_b64": None,
                    "attributes": attrs,
                    "crop_quality": "rejected_duplicate",
                }
            )
    return sanitized
