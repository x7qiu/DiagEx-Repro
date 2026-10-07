"""Immutable source evidence for the evidence-first P&ID engine.

The legacy runtime presents raster views to an agent and lets the agent carry
coordinate provenance.  Evidence-v2 instead captures positioned PDF text and
vector primitives before any model call, normalises them into the rendered
page-pixel frame, and assigns stable source IDs in Python.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import fitz
from pydantic import BaseModel, Field

from diagex.vision.models import BBox, DiagramPage

PageRole = Literal["pid", "legend", "cover", "notes", "other"]
EvidenceConfidence = Literal["high", "medium", "low"]
EvidenceOrigin = Literal["pdf_text", "pdf_vector", "raster_cv"]
VisualLineStyle = Literal["solid", "dashed", "dotted", "dash_dot", "unknown"]
LineStyleSource = Literal[
    "pdf_path",
    "pdf_dash_metadata",
    "vector_fragment_pattern",
    "raster_pattern",
    "unknown",
]

EVIDENCE_SCHEMA_VERSION = "1.2.0"


def stable_evidence_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16]}"


class TextEvidence(BaseModel):
    id: str
    text: str
    bbox: BBox
    origin: Literal["pdf_text", "raster_ocr", "raster_vlm"] = "pdf_text"
    block_index: int | None = None
    line_index: int | None = None
    word_index: int | None = None


class PathEvidence(BaseModel):
    id: str
    page_index: int
    points: list[tuple[int, int]]
    bbox: BBox
    origin: Literal["pdf_vector", "raster_cv"]
    primitive: Literal["line", "rect", "quad", "curve", "raster_line"]
    closed: bool = False
    stroke_width: float = 1.0
    stroke_color: list[float] | None = None
    fill_color: list[float] | None = None
    dashes: str | None = None
    # Visual appearance is deliberately separate from engineering meaning.
    # A dashed line may be electric, pneumatic, capillary, or project-specific;
    # the legend/page solver maps this evidence to ``LineType`` later.
    visual_style: VisualLineStyle = "unknown"
    style_confidence: float | None = None
    style_source: LineStyleSource = "unknown"
    # Derived vector runs retain the immutable PDF primitives that support the
    # inferred style. Ordinary source paths leave this empty.
    source_path_ids: list[str] = Field(default_factory=list)
    source_drawing_index: int | None = None


class PageEvidence(BaseModel):
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    page_index: int
    source_ref: str
    width: int
    height: int
    dpi: float
    effective_dpi: float
    is_scanned: bool
    rotation_deg: float = 0.0
    native_coordinate_frame: Literal["legacy", "rendered_page"] = "legacy"
    role: PageRole = "pid"
    role_confidence: EvidenceConfidence = "low"
    role_reason: str = "uncertain page; processed as P&ID"
    fail_open: bool = True
    text_spans: list[TextEvidence] = Field(default_factory=list)
    paths: list[PathEvidence] = Field(default_factory=list)
    raster_available: bool = True
    warnings: list[str] = Field(default_factory=list)


class DocumentEvidence(BaseModel):
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    source_name: str
    source_sha256: str
    pages: list[PageEvidence] = Field(default_factory=list)


_LEGEND_TERMS = (
    "legend",
    "symbol legend",
    "instrument legend",
    "图例",
    "符号说明",
    "仪表符号",
    "管道符号",
)
_COVER_TERMS = (
    "cover sheet",
    "drawing index",
    "图纸目录",
    "封面",
    "目录",
)
_NOTES_TERMS = (
    "general notes",
    "design notes",
    "说明：",
    "说明:",
    "设计说明",
    "注：",
    "注:",
)
_PID_TERMS = (
    "p&id",
    "piping and instrumentation",
    "管道及仪表流程图",
    "工艺管道及仪表流程图",
)
_TAG_RE = re.compile(
    r"\b(?:[A-Z]{1,5}[- ]?\d{2,6}[A-Z]?|DN\s*\d+|\d{1,4}[-][A-Z]{1,5}[-]\d{2,6})\b",
    re.IGNORECASE,
)


def classify_page(
    *,
    page_index: int,
    text_spans: list[TextEvidence],
    paths: list[PathEvidence],
) -> tuple[PageRole, EvidenceConfidence, str, bool]:
    """Conservative deterministic page routing.

    Strong textual signals can route non-P&ID pages away from perception.  Any
    weak or contradictory page fails open as a P&ID so the router cannot hide
    engineering content.
    """
    text = " ".join(span.text for span in text_spans).casefold()

    def term(values: tuple[str, ...]) -> str | None:
        return next((value for value in values if value.casefold() in text), None)

    if hit := term(_LEGEND_TERMS):
        return "legend", "high", f"legend marker {hit!r}", False
    if hit := term(_PID_TERMS):
        return "pid", "high", f"P&ID title marker {hit!r}", False
    if hit := term(_COVER_TERMS):
        return "cover", "high", f"cover/index marker {hit!r}", False

    tag_count = sum(1 for span in text_spans if _TAG_RE.search(span.text))
    line_count = sum(1 for path in paths if path.primitive == "line")
    if tag_count >= 8 and line_count >= 30:
        return "pid", "medium", "engineering tags and vector-line density", False

    if (hit := term(_NOTES_TERMS)) and tag_count < 4 and line_count < 80:
        return "notes", "medium", f"notes marker {hit!r} with low diagram density", False

    if page_index == 0 and len(text_spans) < 25 and line_count < 30:
        return "cover", "medium", "sparse first page", False

    if not text_spans and not paths:
        return "pid", "low", "no native evidence; raster perception required", True

    return "pid", "low", "uncertain page; fail-open P&ID routing", True


def extract_page_evidence(
    *,
    page: DiagramPage,
    source_path: Path,
    pdf_page: fitz.Page | None = None,
) -> PageEvidence:
    """Extract native evidence for one rendered page.

    ``pdf_page`` may be supplied by the document-level orchestrator to avoid
    reopening a large PDF for each page.  Image inputs simply return an empty
    native layer and are routed through the raster fallback.
    """
    warnings: list[str] = []
    text_spans: list[TextEvidence] = []
    paths: list[PathEvidence] = []

    owned_doc: fitz.Document | None = None
    try:
        if pdf_page is None and source_path.suffix.lower() == ".pdf":
            owned_doc = fitz.open(source_path)
            pdf_page = owned_doc[page.page_index]

        if pdf_page is not None:
            text_spans = _extract_text(pdf_page, page)
            paths = _extract_paths(pdf_page, page)
        else:
            warnings.append("native PDF evidence unavailable; raster fallback required")
    except Exception as exc:  # noqa: BLE001 - evidence degrades to raster
        warnings.append(f"native evidence extraction failed: {exc!r}")
        text_spans = []
        paths = []
    finally:
        if owned_doc is not None:
            owned_doc.close()

    role, role_confidence, reason, fail_open = classify_page(
        page_index=page.page_index,
        text_spans=text_spans,
        paths=paths,
    )
    return PageEvidence(
        page_index=page.page_index,
        source_ref=page.source_ref,
        width=page.width,
        height=page.height,
        dpi=page.dpi,
        effective_dpi=page.effective_dpi,
        is_scanned=page.is_scanned,
        rotation_deg=page.rotation_deg,
        native_coordinate_frame="rendered_page",
        role=role,
        role_confidence=role_confidence,
        role_reason=reason,
        fail_open=fail_open,
        text_spans=text_spans,
        paths=paths,
        raster_available=page.image is not None,
        warnings=warnings,
    )


def _scale_factors(pdf_page: fitz.Page, page: DiagramPage) -> tuple[float, float]:
    rect = pdf_page.rect
    return (
        page.width / max(float(rect.width), 1e-9),
        page.height / max(float(rect.height), 1e-9),
    )


def _bbox_from_floats(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    sx: float,
    sy: float,
) -> BBox:
    left = int(round(min(x0, x1) * sx))
    top = int(round(min(y0, y1) * sy))
    right = int(round(max(x0, x1) * sx))
    bottom = int(round(max(y0, y1) * sy))
    return BBox(x=left, y=top, w=max(1, right - left), h=max(1, bottom - top))




def _point_xy(value: Any) -> tuple[float, float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise TypeError(f"unsupported PDF point {value!r}")


def _scaled_points(values: list[Any], *, sx: float, sy: float) -> list[tuple[int, int]]:
    return [
        (int(round(_point_xy(value)[0] * sx)), int(round(_point_xy(value)[1] * sy)))
        for value in values
    ]


def _path_bbox(points: list[tuple[int, int]]) -> BBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return BBox(
        x=min(xs),
        y=min(ys),
        w=max(1, max(xs) - min(xs)),
        h=max(1, max(ys) - min(ys)),
    )


def _extract_paths(pdf_page: fitz.Page, page: DiagramPage) -> list[PathEvidence]:
    sx, sy = _scale_factors(pdf_page, page)
    out: list[PathEvidence] = []
    for drawing_index, drawing in enumerate(pdf_page.get_drawings() or []):
        width = float(drawing.get("width", 1.0) or 1.0) * ((sx + sy) / 2.0)
        color = drawing.get("color")
        stroke_color = [float(value) for value in color] if color is not None else None
        dashes = str(drawing.get("dashes") or "").strip() or None
        visual_style, style_confidence, style_source = classify_pdf_dash_pattern(
            dashes,
            stroke_width=width,
        )
        for item_index, item in enumerate(drawing.get("items") or []):
            if not item:
                continue
            operator = str(item[0])
            primitive: Literal["line", "rect", "quad", "curve"]
            closed = False
            raw_points: list[Any]
            if operator == "l" and len(item) >= 3:
                primitive = "line"
                raw_points = [item[1], item[2]]
            elif operator == "re" and len(item) >= 2:
                primitive = "rect"
                rect = item[1]
                raw_points = [
                    (rect.x0, rect.y0),
                    (rect.x1, rect.y0),
                    (rect.x1, rect.y1),
                    (rect.x0, rect.y1),
                    (rect.x0, rect.y0),
                ]
                closed = True
            elif operator == "qu" and len(item) >= 2:
                primitive = "quad"
                quad = item[1]
                raw_points = [quad.ul, quad.ur, quad.lr, quad.ll, quad.ul]
                closed = True
            elif operator == "c" and len(item) >= 5:
                primitive = "curve"
                raw_points = [item[1], item[2], item[3], item[4]]
            else:
                continue

            # get_drawings/get_text return unrotated PDF coordinates, whereas
            # the rendered page and detector boxes include the PDF rotation.
            rotated_points = [
                fitz.Point(*_point_xy(value)) * pdf_page.rotation_matrix for value in raw_points
            ]
            points = _scaled_points(rotated_points, sx=sx, sy=sy)
            if len(set(points)) < 2:
                continue
            bbox = _path_bbox(points)
            out.append(
                PathEvidence(
                    id=stable_evidence_id(
                        "vec",
                        page.page_index,
                        drawing_index,
                        item_index,
                        primitive,
                        points,
                    ),
                    page_index=page.page_index,
                    points=points,
                    bbox=bbox,
                    origin="pdf_vector",
                    primitive=primitive,
                    closed=closed,
                    stroke_width=max(0.1, width),
                    stroke_color=stroke_color,
                    fill_color=list(drawing["fill"]) if drawing.get("fill") is not None else None,
                    dashes=dashes,
                    visual_style=visual_style,
                    style_confidence=style_confidence,
                    style_source=style_source,
                    source_drawing_index=drawing_index,
                )
            )
    return out


def classify_pdf_dash_pattern(
    dashes: str | None,
    *,
    stroke_width: float = 1.0,
) -> tuple[VisualLineStyle, float, LineStyleSource]:
    """Classify an explicit PDF stroke pattern without assigning semantics.

    PyMuPDF exposes dash arrays as strings such as ``"[3 2] 0"``. Some CAD
    exports instead draw every visible dash as a separate solid primitive;
    those are intentionally left for topology's fragment-pattern detector.
    """
    text = (dashes or "").strip().lower()
    if not text or text in {"[] 0", "[]0", "none", "solid"}:
        return "solid", 0.99, "pdf_path"

    match = re.search(r"\[([^]]*)\]", text)
    if match is None:
        return "unknown", 0.35, "pdf_dash_metadata"
    values = [
        abs(float(value))
        for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", match.group(1))
        if float(value) != 0
    ]
    if not values:
        return "solid", 0.99, "pdf_path"

    on_lengths = values[::2]
    scale = max(0.1, float(stroke_width))
    if len(on_lengths) >= 2 and max(on_lengths) >= max(3.0 * min(on_lengths), 4.0 * scale):
        return "dash_dot", 0.96, "pdf_dash_metadata"
    if max(on_lengths) <= 2.5 * scale:
        return "dotted", 0.93, "pdf_dash_metadata"
    return "dashed", 0.97, "pdf_dash_metadata"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_spans_intersecting(page: PageEvidence, bbox: BBox) -> list[TextEvidence]:
    return [
        span
        for span in page.text_spans
        if span.bbox.x < bbox.x2
        and span.bbox.x2 > bbox.x
        and span.bbox.y < bbox.y2
        and span.bbox.y2 > bbox.y
    ]


def _extract_text(pdf_page: fitz.Page, page: DiagramPage) -> list[TextEvidence]:
    # Lazy import keeps source record types independent of the recognition backend.
    from diagex.vision.text_detection import extract_pdf_text
    return extract_pdf_text(pdf_page, page)
