"""Stateless, source-bound visual perception for evidence-v2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from diagex.config import SymbolPerceptionConfig
from diagex.llm.client import LLMClient, is_malformed_tool_json_error, is_non_retryable_api_error
from diagex.llm.cost import CostTracker
from diagex.llm.diagnostics import request_summary, safe_error
from diagex.llm.prompts.output_language import CHINESE_EXPLANATIONS
from diagex.ui.progress import ProgressReporter
from diagex.vision.encode import encode_image_block
from diagex.vision.evidence import PageEvidence, stable_evidence_id, text_spans_intersecting
from diagex.vision.legend_context import select_legend_context
from diagex.vision.models import BBox, Confidence, Tile
from diagex.vision.reference_evidence import KnowledgeCitation, knowledge_matches
from diagex.vision.symbol_candidates import SymbolCandidate, symbol_candidates
from diagex.vision.views import ViewInfo

# Wire decisions encode rejection evidence directly, so a symbol row cannot
# acquire a contradictory rejection basis merely by filling an optional field.
_REJECTION_DECISIONS = {
    "reject_text_only": "text_only",
    "reject_line": "pipe_or_signal_line",
    "reject_annotation": "border_or_annotation",
    "reject_no_glyph": "no_distinct_glyph",
}
_VALVE_CLASSES = {
    "solenoid_valve": (None, "solenoid"),
    "motor_operated_valve": (None, "electric_motor"),
    "pneumatic_valve": (None, "pneumatic"),
    "hydraulic_valve": (None, "hydraulic"),
    "manual_valve": (None, "manual"),
    **{f"{kind}_valve": (kind, None) for kind in (
        "gate", "globe", "ball", "butterfly", "check", "needle", "plug",
        "control", "angle", "diaphragm", "three_way", "safety", "relief",
        "isolation", "shutdown", "regulating",
    )},
}


class NormalizedBBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

    @model_validator(mode="after")
    def _within_image(self) -> NormalizedBBox:
        values = (self.x, self.y, self.w, self.h)
        if not all(value == value and abs(value) != float("inf") for value in values):
            raise ValueError("normalised bbox values must be finite")
        if self.x < 0 or self.y < 0 or self.w <= 0 or self.h <= 0:
            raise ValueError("normalised bbox must have non-negative origin and positive size")
        if self.x > 1 or self.y > 1 or self.x + self.w > 1.001 or self.y + self.h > 1.001:
            raise ValueError("normalised bbox must fit inside [0,1] image coordinates")
        return self


class PerceivedObject(BaseModel):
    kind: Literal["equipment", "instrument", "opc"]
    candidate_id: str | None = None
    bbox: NormalizedBBox | None = None
    confidence: Confidence = "medium"
    printed_tag: str | None = None
    canonical_tag: str | None = None
    equipment_class: str | None = None
    valve_type: str | None = None
    actuation: (
        Literal[
            "manual",
            "solenoid",
            "electric_motor",
            "pneumatic",
            "hydraulic",
            "spring",
            "other",
        ]
        | None
    ) = None
    instrument_function: str | None = None
    measured_variable: (
        Literal["pressure", "temperature", "flow", "level", "analysis", "other"] | None
    ) = None
    loop_number: str | None = None
    opc_direction: Literal["in", "out"] | None = None
    service: str | None = None
    drawing_ref: str | None = None
    line_id: str | None = None
    structural_description: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_text_ids: list[str] = Field(default_factory=list)
    knowledge_reference_id: str | None = None
    knowledge_variant_id: str | None = None
    knowledge_evidence: str | None = None
    knowledge_citations: list[KnowledgeCitation] = Field(default_factory=list)
    legend_entry_ids: list[str] = Field(default_factory=list)
    legend_evidence: str | None = None
    recognition_evidence: str | None = None

    @model_validator(mode="after")
    def _has_geometry_reference(self) -> PerceivedObject:
        if not self.candidate_id and self.bbox is None:
            raise ValueError("object requires candidate_id or an explicit fallback bbox")
        return self

    @model_validator(mode="after")
    def _normalise_valve_class(self) -> PerceivedObject:
        if self.kind != "equipment" or not self.equipment_class:
            return self
        key = re.sub(r"[\s-]+", "_", self.equipment_class.lower())
        if key in _VALVE_CLASSES:
            subtype, actuation = _VALVE_CLASSES[key]
            self.attributes.setdefault("original_equipment_class", self.equipment_class)
            self.equipment_class = "valve"
            self.valve_type = self.valve_type or subtype
            self.actuation = self.actuation or actuation
        return self

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_object_shape(cls, value: Any) -> Any:
        """Read old checkpoints/provider output without advertising free-form fields."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        attributes = dict(data.get("attributes") or {})
        if data.get("candidate_id") and data.get("bbox") is not None:
            # Coordinates supplied alongside an opaque native reference are
            # redundant, even if malformed. Never let them move that instance.
            attributes["ignored_model_bbox"] = data.pop("bbox")
        legacy_label = data.pop("label", None)
        legacy_raw = data.pop("raw_text", None)
        data.setdefault("printed_tag", legacy_raw or legacy_label)
        data.setdefault("canonical_tag", legacy_label or legacy_raw)
        aliases = {
            "equipment_class": "equipment_class",
            "valve_type": "valve_type",
            "actuation": "actuation",
            "instrument_function": "instrument_function",
            "measured_variable": "measured_variable",
            "loop_number": "loop_number",
            "direction": "opc_direction",
            "service": "service",
            "drawing_ref": "drawing_ref",
            "target_sheet": "drawing_ref",
            "line_id": "line_id",
            "structural_description": "structural_description",
        }
        for source, target in aliases.items():
            if data.get(target) in (None, "") and attributes.get(source) not in (None, ""):
                data[target] = attributes[source]
        data["attributes"] = attributes
        return data

    @field_validator(
        "printed_tag",
        "canonical_tag",
        "equipment_class",
        "valve_type",
        "actuation",
        "instrument_function",
        "loop_number",
        "service",
        "drawing_ref",
        "line_id",
        "structural_description",
        mode="before",
    )
    @classmethod
    def _normalise_optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        text = " ".join(str(value).split()).strip()
        return text or None

    @property
    def label(self) -> str:
        return self.canonical_tag or self.printed_tag or ""

    @property
    def raw_text(self) -> str | None:
        return self.printed_tag

    def graph_attributes(self) -> dict[str, Any]:
        values = dict(self.attributes)
        typed = {
            "equipment_class": self.equipment_class,
            "valve_type": self.valve_type,
            "actuation": self.actuation,
            "instrument_function": self.instrument_function,
            "measured_variable": self.measured_variable,
            "loop_number": self.loop_number,
            "direction": self.opc_direction,
            "service": self.service,
            "drawing_ref": self.drawing_ref,
            "line_id": self.line_id,
            "structural_description": self.structural_description,
        }
        values.update({key: value for key, value in typed.items() if value not in (None, "")})
        if self.canonical_tag:
            values["canonical_tag"] = self.canonical_tag
        return values


class CandidateDisposition(BaseModel):
    candidate_id: str
    decision: Literal["reject", "uncertain"]
    reason: str = Field(min_length=1)
    non_symbol_basis: str | None = None


class CandidateResult(BaseModel):
    """One wire outcome for a native instance; coordinates remain owned by Python."""

    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    decision: Literal["symbol", "non_symbol", "uncertain"]
    symbol: dict[str, Any] | None = None
    reason: str | None = Field(default=None, min_length=1)
    non_symbol_basis: (
        Literal["text_only", "pipe_or_signal_line", "border_or_annotation", "no_distinct_glyph"]
        | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def _flat_classification(cls, value: Any) -> Any:
        """Read flat live rows and the nested v2.0 format without mixing them."""
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if value.get("decision") in _REJECTION_DECISIONS:
            if "non_symbol_basis" in value:
                raise ValueError("rejection decision already specifies its evidence basis")
            value["non_symbol_basis"] = _REJECTION_DECISIONS[value["decision"]]
            value["decision"] = "non_symbol"
        fields = set(PerceivedObject.model_fields) - {"candidate_id", "bbox", "attributes"}
        classification = {k: v for k, v in value.items() if k in fields}
        if not classification:
            return value
        if "symbol" in value:
            raise ValueError("flat and nested classifications cannot be mixed")
        return {**{k: v for k, v in value.items() if k not in fields}, "symbol": classification}

    @model_validator(mode="after")
    def _one_outcome(self) -> CandidateResult:
        if self.decision == "symbol":
            if self.symbol is None or self.non_symbol_basis is not None:
                raise ValueError("symbol outcome requires classification, not a rejection basis")
            allowed = set(PerceivedObject.model_fields) - {"candidate_id", "bbox", "attributes"}
            if set(self.symbol) - allowed:
                raise ValueError("classification must not override identity or native geometry")
            PerceivedObject.model_validate({**self.symbol, "candidate_id": self.candidate_id})
        else:
            if self.symbol is not None or not self.reason or not self.reason.strip():
                raise ValueError("non-symbol/uncertain outcome requires a reason and no symbol")
            if self.decision == "non_symbol" and self.non_symbol_basis is None:
                raise ValueError(
                    "non-symbol outcome requires evidence about the candidate's own ink"
                )
            if self.decision == "uncertain" and self.non_symbol_basis is not None:
                raise ValueError("uncertain outcome must not assert a non-symbol basis")
        return self


class PerceptionBatch(BaseModel):
    """Normalized checkpoint representation; live wire output uses CandidateResult."""

    objects: list[PerceivedObject] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    rejected_objects: list[dict[str, Any]] = Field(default_factory=list)
    candidate_decisions: list[CandidateDisposition] = Field(default_factory=list)
    candidate_reviews: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("observations", "uncertainties", mode="before")
    @classmethod
    def _normalise_diagnostic_text(cls, value: Any) -> Any:
        """Accept provider-specific string encoding for diagnostic fields.

        Some Anthropic-compatible OpenRouter models serialise optional
        ``list[str]`` tool parameters as one string.  These fields do not
        affect graph construction, so retaining that string as one diagnostic
        item is safer than rejecting otherwise valid object detections.  The
        engineering objects and their geometry remain strictly validated.
        """
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    text = item["text"].strip()
                else:
                    text = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if text:
                    out.append(text)
            return out
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return [text]
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
        if isinstance(decoded, str):
            decoded = decoded.strip()
            return [decoded] if decoded else []
        return [text]


class DetectionRecord(BaseModel):
    id: str
    page_index: int
    tile_id: str
    kind: Literal["equipment", "instrument", "opc"]
    label: str
    bbox: BBox
    confidence: Confidence
    raw_text: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_text_ids: list[str] = Field(default_factory=list)


class PerceptionResponseFormatError(ValueError):
    """A successful model response did not contain usable structured output."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 1,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.diagnostics = list(diagnostics or [])


@dataclass
class PerceptionOutcome:
    detections: list[DetectionRecord]
    batch: PerceptionBatch
    attempts: int = 1
    recovery_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[SymbolCandidate] = field(default_factory=list)
    contract_failed: bool = False


@dataclass
class PerceptionRunGuard:
    """Stop systemic contract/transport failures without counting ambiguity as failure."""

    policy: SymbolPerceptionConfig
    consecutive_failures: int = 0
    total_failures: int = 0

    def observe(self, failed: bool | None) -> str | None:
        # Empty candidate crops must not reset a failing native-candidate stream.
        if failed is None:
            return None
        self.consecutive_failures = self.consecutive_failures + 1 if failed else 0
        self.total_failures += int(failed)
        if (
            self.consecutive_failures >= self.policy.consecutive_failure_limit
            or self.total_failures >= self.policy.total_failure_limit
        ):
            return (
                "Symbol extraction stopped early after repeated invalid responses or request failures "
                f"({self.consecutive_failures} consecutive, {self.total_failures} total). "
                "Completed crops are saved; inspect perception diagnostics before retrying."
            )
        return None


def _contract_problem_count(batch: PerceptionBatch) -> int:
    return sum(
        r.get("status") == "unreviewed"
        or r.get("reason", "").startswith("Invalid candidate result:")
        or r.get("reason", "").lower() == "conflicting candidate decisions"
        for r in batch.candidate_reviews
        if r.get("candidate_id")
    )


def _contract_failed(batch: PerceptionBatch, candidates: list[SymbolCandidate]) -> bool:
    return bool(candidates) and _contract_problem_count(batch) >= max(1, len(candidates) / 2)


_SYSTEM_PROMPT = """\
Detect physical symbol instances in one fixed P&ID view. This is RAW SYMBOL
DETECTION. Return exactly one submit_pid_objects call. Do not decide which
symbols deserve independent graph nodes or compose an equipment assembly.

For EVERY native_symbol_candidates ID, return exactly one candidate_results row:
- decision="symbol": include kind and classification fields directly in the row;
  no nested symbol object and no rejection basis.
- decision="reject_text_only", "reject_line", "reject_annotation", or
  "reject_no_glyph": include a reason describing that evidence in its OWN ink,
  with no classification fields. Reject only when the candidate has no physical symbol.
- decision="uncertain": include a short reason, without classification fields.
Example: {"candidate_id":"candidate-example","decision":"symbol",
"kind":"equipment","equipment_class":"motor","confidence":"high"}.
Copy the real candidate_id exactly from the table. Never return a decision alone.
Omit unused optional fields; do not use null or empty strings.
Never list the same ID twice. A symbol row must not also reject that instance.
All listed IDs are already assigned to this crop by Python. Assess every one;
do not recheck ownership coordinates or reject it as outside the core. Python
retains the complete native box, even when the view clips part of its outline.
Never replace the candidate with its tag or a surrounding/nearby component.

Retain every visible physical component symbol, including small isolation valves,
untagged valves, attached M motor glyphs, and identifiable actuator glyphs.
Being attached, inside a package, lacking a tag, or having no known graph port
is NOT a reason to reject a symbol. Assembly grouping and valve/actuator
composition happen later. Classify each candidate's own ink, not the enclosing
vessel or compressor. A P/function circle inside a vessel is not the vessel;
M in a drive glyph means motor when the local symbol/legend supports that role.
Do not invent additional candidates for inseparable decorative strokes.

Classification fields:
- equipment: equipment bodies, motors, valve bodies and actuator symbols.
  For a distinct actuator glyph, equipment_class="actuator"; for a motor,
  equipment_class="motor". Unknown equipment subtype may be omitted.
  A valve BODY is equipment_class="valve"; record its actuator as actuation,
  e.g. "solenoid". Never label the valve body "actuator" merely because an
  actuator is attached above it. A separate actuator candidate remains actuator.
- instrument: one visible instrument/function symbol.
- opc: a visible off-page continuation glyph, even if the destination is unknown.
  Service text or an unresolved linking question alone does not establish an OPC.
Use the project legend and local text as evidence for subtype and actuation.
Candidate shapes are geometric hints, not engineering types: a capsule can be a
compressor or vessel; a circle can be a motor or instrument. Omit unsupported
attributes and lower confidence instead of inventing a specific subtype.
Compare internal strokes and attached drive symbols with the supplied equipment
definitions before calling a capsule a vessel. A P/function circle is an
instrument unless its OWN ink and the legend establish a different component.
Use a directly adjacent descriptive component label to distinguish similar
equipment outlines. Do not override explicit local component text with an
approximate legend resemblance. local_text_hints are nearby source words, not
proven labels: verify their placement in the original image. A package title
does not classify every component inside the package.

Distinguish an enclosed identity callout from an independent instrument symbol.
Use the source glyph, its leader attachment, repeated local drawing convention,
and any explicit legend. A circle containing a tag is not sufficient evidence
of an instrument or positioner. If the enclosure only identifies a separate
visible valve/equipment body, treat that enclosure and leader as annotation;
detect the physical body with its own tight box, excluding the callout/leader,
and attach the visible tag only when the association is supported. For a native
callout candidate use reject_annotation with the source evidence. Real instrument
or function bubbles remain symbols even when connected by process/signal lines.
Neither a connecting stroke nor particular tag letters alone justify rejection
or reclassification. Preserve uncertainty when the role or association is unclear.

Before submitting, sweep the ORIGINAL unmarked image independently of the C boxes.
Inspect each inline change of geometry and distinguish adjacent glyphs separately;
candidate coverage is not completeness. Open/single-diagonal symbols may have no
candidate. Do not infer an object from reference availability or pipeline text.
Pipeline identifiers (for example size-service-number-specification codes) label
lines, not an untagged valve body. Preserve them in nearby text; do not copy them
into printed_tag or canonical_tag without explicit source evidence of a device tag.
Only for a visible symbol ABSENT from the supplied candidates, use proposals:
an object with a tight bbox normalised to THIS IMAGE and no candidate_id.
Python checks its ownership and retains unsupported geometry for review.
Never duplicate a listed candidate through proposals or copy a nearby ID.
On scanned pages without candidates, put visible symbols in proposals.

printed_tag preserves the exact visible identifier. canonical_tag is a
conservative normalisation; omit either when uncertain. Nearby native words
are evidence, not entities. A package title belongs to the package, not its
nearest component. Pipe labels, line numbers, notes, borders, dimensions and
leader arrows alone are not physical symbols.
Images labelled Legend reference are definitions, never objects in this view.
Knowledge references are also definitions, never drawing evidence. Explicit
drawing legends override them. Supplied references are a partial selection, not
an inventory or whitelist of symbols allowed in this view. Inspect source ink
for all physical symbols independently of which references were supplied.
Retain a visible symbol that has no reference match; use its source evidence,
omit unsupported subtype/citations, and preserve uncertainty where needed.
Absence from the supplied references is not a reason to omit a symbol or force
it into the nearest available reference class.
Placeholder text such as (#), (##), and (*)
describes a text slot; never copy it into a drawing tag. Optionally report
knowledge_reference_id and knowledge_variant_id from supplied references,
with knowledge_evidence describing the actual source glyph and text supporting
the match. Omit these fields when uncertain; a match never creates an object.
For multiple knowledge references, use knowledge_citations, each with reference_id,
optional variant_id and drawing_evidence. Cite all references actually used.
When using a supplied legend definition, report its legend_entry_ids and
legend_evidence describing the matching source glyph/text. Cite only entries
actually used, not every supplied entry. Both legend and knowledge references
may be cited when both contributed. Otherwise leave their IDs empty and explain
the visible glyph/local text supporting recognition in recognition_evidence.
Use their complete printed definitions. Installation conventions alone do not
establish an instrument's variable/function. The first image is source ink;
blue C markers in the second image are guides, not source symbols.

Return {"candidate_results":[],"proposals":[]} only if there are no supplied
candidates and no visible symbols to propose.
"""


def _tool_input_schema() -> dict[str, Any]:
    # Avoid conditional/union schemas at the provider boundary. Some production
    # responses to v2.0's oneOf rows retained ONLY the branch's decision field.
    # Optional values are omitted (not null); exclusivity stays in CandidateResult.
    full_schema = PerceivedObject.model_json_schema()
    def simple(schema: dict[str, Any]) -> dict[str, Any]:
        if "$ref" in schema:
            return simple(full_schema["$defs"][schema["$ref"].rsplit("/", 1)[-1]])
        if "anyOf" in schema:
            return simple(next(s for s in schema["anyOf"] if s.get("type") != "null"))
        return {
            k: simple(v) if isinstance(v, dict) else v
            for k, v in schema.items()
            if k not in {"default", "title"}
        }

    properties = {
        k: simple(v)
        for k, v in full_schema["properties"].items()
        if k not in {"candidate_id", "bbox", "attributes"}
    }
    bbox_schema = {
        "type": "object",
        "properties": {k: {"type": "number"} for k in ("x", "y", "w", "h")},
        "required": ["x", "y", "w", "h"],
        "additionalProperties": False,
    }
    result_schema = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "minLength": 1},
            "decision": {
                "type": "string",
                "enum": ["symbol", "uncertain", *_REJECTION_DECISIONS],
                "description": "One outcome: classify a symbol, retain uncertainty, or state the specific non-symbol evidence.",
            },
            **properties,
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["candidate_id", "decision"],
        "additionalProperties": False,
    }
    proposal_schema = {
        "type": "object",
        "properties": {**properties, "bbox": bbox_schema},
        "required": ["kind", "bbox"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "candidate_results": {
                "type": "array",
                "items": result_schema,
                "description": "Exactly one outcome for each supplied native ID.",
            },
            "proposals": {
                "type": "array",
                "items": proposal_schema,
                "description": "New visible symbols without a supplied candidate; geometry needs review.",
            },
        },
        "required": ["candidate_results"],
        "additionalProperties": False,
    }


_SUBMIT_TOOL = {
    "name": "submit_pid_objects",
    "description": "Submit one raw symbol decision per native candidate and any unanchored proposals.",
    "input_schema": _tool_input_schema(),
}

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _local_text_hints(candidate, spans, view_info):
    """Expose nearby printed labels without promoting proximity to identity."""
    box = candidate.bbox
    limit = max(24, min(box.w, box.h) * 1.5)
    ranked = []
    for span in spans:
        if span.id in candidate.source_text_ids:
            continue
        dx = max(0, box.x - span.bbox.x2, span.bbox.x - box.x2)
        dy = max(0, box.y - span.bbox.y2, span.bbox.y - box.y2)
        distance = math.hypot(dx, dy)
        if distance <= limit:
            ranked.append((distance, span.id, span))
    return [
        {"evidence_id": span.id, "text": span.text,
         "bbox_normalized": _page_bbox_to_normalized(span.bbox, view_info)}
        for _, _, span in sorted(ranked)[:3]
    ]


def _candidate_detail_blocks(candidates, image, view_info, *, tight=False):
    """Enlarge source ink for the bounded recheck; IDs still own all geometry."""
    blocks = []
    for candidate in candidates[:4]:
        b = _page_bbox_to_normalized(candidate.bbox, view_info)
        x, y, w, h = b["x"] * image.width, b["y"] * image.height, b["w"] * image.width, b["h"] * image.height
        padding = max(2, min(w, h) * .15) if tight else max(30, min(w, h))
        crop = image.crop((max(0, int(x-padding)), max(0, int(y-padding)), min(image.width, math.ceil(x+w+padding)), min(image.height, math.ceil(y+h+padding))))
        scale = min(4, 768 / max(crop.size))
        crop = crop.resize((max(1, round(crop.width*scale)), max(1, round(crop.height*scale))), Image.Resampling.LANCZOS)
        blocks += [{"type": "text", "text": f"Enlarged source detail for {candidate.id} ({candidate.shape}). The target is the central native glyph; surrounding ink is context. This image does not introduce any new candidate or coordinates."}, encode_image_block(crop)]
    return blocks


def _perceive_tile(
    *,
    client: LLMClient,
    cost_tracker: CostTracker,
    reporter: ProgressReporter,
    page: PageEvidence,
    tile: Tile,
    view_image: Any,
    view_info: ViewInfo,
    ownership_bbox: BBox,
    legend_summary: list[dict[str, Any]] | None,
    step: int,
    page_context: dict[str, Any] | None = None,
    on_attempt: Callable[[], None] | None = None,
    candidates: list[SymbolCandidate] | None = None,
    reasoning_mode: Literal["auto", "enabled", "disabled"] = "auto",
    policy: SymbolPerceptionConfig | None = None,
    deadline: float | None = None,
    on_diagnostic: Callable[[dict[str, Any]], None] | None = None,
    overview_image: Any = None,
    region_provider: Callable[..., Any] | None = None,
    escalation_client: LLMClient | None = None,
) -> PerceptionOutcome:
    # First establish identity cheaply. Reasoning is reserved for genuinely
    # ambiguous candidate decisions, never for retrying a broken wire format.
    policy = policy or SymbolPerceptionConfig()
    adaptive = policy.workflow == "adaptive"
    mode = "disabled"
    page_candidates = symbol_candidates(page) if candidates is None else candidates
    owned_candidates = [
        c for c in page_candidates if _bbox_center_is_owned(c.bbox, ownership_bbox, page)
    ]
    # Show source ink separately so a guide cannot obscure a small symbol or
    # be mistaken for its outline. Both images use exactly the same frame.
    marked_image = view_image.copy()
    painter = ImageDraw.Draw(marked_image)
    markers = {}
    for number, candidate in enumerate(owned_candidates, 1):
        markers[candidate.id] = f"C{number}"
        box = _page_bbox_to_normalized(candidate.bbox, view_info)
        x, y = box["x"] * marked_image.width, box["y"] * marked_image.height
        x2, y2 = x + box["w"] * marked_image.width, y + box["h"] * marked_image.height
        painter.rectangle((x, y, x2, y2), outline="#1769dd", width=1)
        painter.text(
            (x, max(0, y - 12)),
            markers[candidate.id],
            fill="#1769dd",
            stroke_width=1,
            stroke_fill="white",
        )
    nearby = text_spans_intersecting(page, tile.bbox)
    text_payload = [
        {
            "evidence_id": span.id,
            "text": span.text,
            "bbox_normalized": _page_bbox_to_normalized(span.bbox, view_info),
        }
        for span in nearby[:250]
    ]
    legend_entries, legend_images, candidate_legend_refs = select_legend_context(
        list(legend_summary or []), owned_candidates, [span.text for span in nearby]
    )
    page_context = dict(page_context or {})
    if page_context.get("knowledge"):
        page_context["knowledge"] = _bounded_knowledge_context(
            page_context["knowledge"], legend_entries
        )
    prompt = {
        "page_index": page.page_index,
        "page_role": page.role,
        "source_tile": tile.id,
        "nearby_native_text": text_payload,
        "legend_entries": legend_entries,
        "page_overview_context": dict(page_context or {}),
        "native_symbol_candidates": [
            {
                "candidate_id": c.id,
                "marker": markers[c.id],
                "legend_entry_ids": candidate_legend_refs.get(c.id, []),
                "shape": c.shape,
                "bbox_normalized": _page_bbox_to_normalized(c.bbox, view_info),
                "partially_visible": not (
                    c.bbox.x >= view_info.page_bbox.x
                    and c.bbox.y >= view_info.page_bbox.y
                    and c.bbox.x2 <= view_info.page_bbox.x2
                    and c.bbox.y2 <= view_info.page_bbox.y2
                ),
                "native_text": c.text,
                "local_text_hints": _local_text_hints(c, nearby, view_info),
                "source_text_ids": c.source_text_ids,
            }
            for c in owned_candidates
        ],
    }
    prompt_json = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    invalid_responses: list[dict[str, Any]] = []
    initial: PerceptionOutcome | None = None
    retry_ids: set[str] = set()
    for attempt in range(1, 3):
        call_budget = (
            policy.reasoning_timeout_s if mode == "enabled" else policy.request_timeout_s
        )
        if deadline is not None:
            call_budget = min(call_budget, deadline - time.monotonic())
        if call_budget <= 0:
            if initial is not None:
                return initial
            raise TimeoutError("symbol extraction time budget exhausted")
        recovery_instruction = ""
        if attempt == 2:
            recovery_instruction = (
                "\nYour previous response could not be parsed. Do not explain or narrate. "
                "Call submit_pid_objects exactly once, using candidate_results and proposals, with one outcome per supplied ID."
            )
            if initial is not None:
                recovery_instruction = (
                    "\nResolve ONLY the candidate IDs in the table below. The first pass "
                    "skipped these or returned contradictory outcomes. Reinspect the source "
                    "ink and give exactly one candidate_results row for each ID. All other "
                    "detections are already retained; do not repeat them. If the drawing "
                    "remains ambiguous, mark uncertain."
                )
        messages = [
            {
                "role": "user",
                "content": [
                    encode_image_block(view_image),
                    {
                        "type": "text",
                        "text": (
                            "Inspect this high-resolution detail view and submit the structured "
                            "object list."
                            " The first image is original source ink. The second is the same"
                            " view with candidate guides; use the first to verify actual outlines."
                            " Blue C markers and boxes are generated candidate guides, not source ink."
                            f"{recovery_instruction}\n{prompt_json}"
                        ),
                    },
                    *([encode_image_block(marked_image)] if owned_candidates else []),
                    *(_candidate_detail_blocks([c for c in owned_candidates if c.id in retry_ids], view_image, view_info) if attempt == 2 else _candidate_detail_blocks([c for c in owned_candidates if c.shape == "open_inline_valve"], view_image, view_info, tight=True)),
                    *legend_images,
                    *_knowledge_image_blocks((page_context or {}).get("knowledge", {})),
                    *([{"type": "text", "text": "Whole-page overview for context only. Report coordinates in the first detail image, never this overview."}, encode_image_block(overview_image)] if adaptive and overview_image is not None else []),
                    *(_wider_source_blocks([c for c in owned_candidates if c.id in retry_ids], region_provider) if adaptive and attempt == 2 and region_provider is not None else []),
                ],
            }
        ]
        request = {
            "system": _SYSTEM_PROMPT + CHINESE_EXPLANATIONS,
            "messages": messages,
            "tools": [_SUBMIT_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_pid_objects"},
            "max_tokens": (
                policy.reasoning_tokens if mode == "enabled" else policy.first_pass_tokens
            ),
            "thinking": {"type": "adaptive" if mode == "enabled" else "disabled"},
            "reasoning_mode_override": mode,
            "output_config": {"effort": "low"},
            "time_budget_s": call_budget,
            "max_attempts": policy.transport_attempts,
        }
        if on_diagnostic is not None:
            on_diagnostic({"attempt": attempt, "phase": "request", "request": _request_diagnostic(request), "summary": request_summary(request)})
        if on_attempt is not None:
            on_attempt()
        try:
            active_client = escalation_client if adaptive and attempt == 2 and initial is not None and escalation_client is not None else client
            response = active_client.messages_create(
                **request,
                on_stream_delta=lambda kind, text: reporter.on_stream_delta(kind=kind, text=text),
                on_transport_event=(
                    lambda event, number=attempt: on_diagnostic({"attempt": number, "phase": "transport", "transport": event})
                ) if on_diagnostic is not None else None,
            )
        except Exception as exc:
            if on_diagnostic is not None:
                on_diagnostic({"attempt": attempt, "phase": "request_error", "error": safe_error(exc), "error_type": type(exc).__name__})
            if initial is not None and not is_non_retryable_api_error(exc):
                invalid_responses.append(
                    {
                        "attempt": attempt,
                        "phase": "candidate_decision_recovery",
                        "error": str(exc),
                    }
                )
                initial.attempts = attempt
                return initial
            if not is_malformed_tool_json_error(exc):
                raise
            invalid_responses.append(
                {
                    "attempt": attempt,
                    "error": str(exc),
                    "phase": "provider_tool_json_decode",
                    "response": None,
                }
            )
            if attempt == 2:
                raise PerceptionResponseFormatError(
                    "perception tool arguments were malformed after one recovery attempt",
                    attempts=attempt,
                    diagnostics=invalid_responses,
                ) from exc
            continue
        if on_diagnostic is not None:
            on_diagnostic({"attempt": attempt, "phase": "response", "response": _response_diagnostic(response)})
        _report_response_content(response, reporter)
        cost_tracker.record(
            response,
            step=step + attempt - 1,
            tile_id=tile.id,
            page_index=page.page_index,
        )
        reporter.on_token_update(total_tokens=cost_tracker.total_tokens())
        try:
            batch = parse_perception_response(response, page=page, view_info=view_info)
        except PerceptionResponseFormatError as exc:
            invalid_responses.append(
                {
                    "attempt": attempt,
                    "error": str(exc),
                    "response": _response_diagnostic(response),
                }
            )
            if mode == "enabled" and getattr(response, "stop_reason", None) == "max_tokens":
                # Reasoning is already the second and final call. Retain the
                # first-pass candidates for review; never start a third call.
                invalid_responses[-1]["reasoning_budget_exhausted"] = True
            if attempt == 2:
                if initial is not None:
                    initial.attempts = attempt
                    return initial
                raise PerceptionResponseFormatError(
                    "perception response was unstructured after one recovery attempt",
                    attempts=attempt,
                    diagnostics=invalid_responses,
                ) from exc
            continue

        knowledge_context = (page_context or {}).get("knowledge", {})
        _bind_knowledge(batch, knowledge_context)
        _bind_legend(batch, legend_entries)
        detections = project_batch(
            batch=batch,
            page=page,
            tile=tile,
            view_info=view_info,
            nearby_text_ids={span.id for span in nearby},
            ownership_bbox=ownership_bbox,
            candidates=(
                [c for c in owned_candidates if not retry_ids or c.id in retry_ids]
                if not page.is_scanned
                else None
            ),
        )
        # These are supplied references, not claims that the model proved a
        # particular legend match. Preserve them for offline review/replay.
        image_ids = [
            entry["legend_entry_id"] for entry in legend_entries if entry.get("has_reference_image")
        ]
        for detection in detections:
            cid = detection.attributes.get("symbol_candidate_id")
            refs = candidate_legend_refs.get(
                cid, [entry["legend_entry_id"] for entry in legend_entries]
            )
            if refs:
                detection.attributes["supplied_legend_entry_ids"] = refs
                detection.attributes["supplied_legend_image_ids"] = [
                    rid for rid in image_ids if rid in refs
                ]
        for review in batch.candidate_reviews:
            if knowledge_context:
                from diagex.knowledge.library import supplied_trace
                review["supplied_knowledge"] = supplied_trace(knowledge_context)
            refs = candidate_legend_refs.get(review.get("candidate_id"), [])
            if refs:
                review["supplied_legend_entry_ids"] = refs
        outcome = PerceptionOutcome(
            detections=detections,
            batch=batch,
            attempts=attempt,
            recovery_diagnostics=invalid_responses,
            candidates=owned_candidates,
            contract_failed=_contract_failed(batch, owned_candidates),
        )
        if initial is not None:
            return _merge_candidate_retry(initial, outcome, retry_ids)
        if attempt == 1 and not page.is_scanned:
            # Rounded equipment outlines repeatedly received confident but
            # wrong vessel/cooler/compressor classes in the reviewed runs.
            # Their geometry cannot settle the family. Verify these few bodies
            # explicitly instead of relying on model confidence as a gate.
            body_ids = {
                c.id for c in owned_candidates if c.shape == "capsule_body"
            } if reasoning_mode == "enabled" or adaptive else set()
            body_ids &= {
                d.attributes.get("symbol_candidate_id") for d in detections
                if d.kind == "equipment"
            }
            semantic_ids = _accepted_reinspection_ids(owned_candidates, detections) if adaptive else set()
            retry_ids = {
                r["candidate_id"]
                for r in batch.candidate_reviews
                if r.get("candidate_id")
                and (
                    r["status"] == "unreviewed"
                    or r.get("reason") == "conflicting candidate decisions"
                    or r.get("reason") == "Conflicting candidate decisions"
                    or r.get("reason", "").startswith("Invalid candidate result:")
                    or (
                        (reasoning_mode == "enabled" or adaptive)
                        and r["status"] in {"uncertain", "reject"}
                    )
                )
            } | body_ids | semantic_ids
            if retry_ids:
                initial = outcome
                # Valid ambiguity or a geometry/classification disagreement
                # warrants reasoning. Missing IDs/fields need short repair.
                if (reasoning_mode == "enabled" or adaptive) and not _contract_problem_count(batch):
                    mode = "enabled"
                prompt["native_symbol_candidates"] = [
                    c for c in prompt["native_symbol_candidates"] if c["candidate_id"] in retry_ids
                ]
                reviews_by_id = {r.get("candidate_id"): r for r in batch.candidate_reviews}
                for candidate_prompt in prompt["native_symbol_candidates"]:
                    cid = candidate_prompt["candidate_id"]
                    candidate_prompt["previous_issue"] = (
                        "Rounded equipment body needs family verification even if the first pass was confident. Use internal strokes and directly adjacent component labels; distinguish compressor stages from vessels and cooling equipment."
                        if cid in body_ids else "Independent accepted-detection check: compare the printed letters, nearby tag and project legend with the proposed class. A P instrument bubble is not a motor merely because both use circles. Preserve uncertainty when evidence disagrees." if cid in semantic_ids else reviews_by_id[cid]["reason"]
                    )
                prompt_json = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
                invalid_responses.append(
                    {
                        "phase": "candidate_decision_recovery",
                        "candidate_ids": sorted(retry_ids),
                        "reasoning_mode": mode,
                    }
                )
                continue
        return outcome

    raise AssertionError("perception recovery loop exited unexpectedly")


def _accepted_reinspection_ids(candidates, detections):
    """Text/class contradictions plus a reproducible 10% accepted sample."""
    import hashlib
    by_id = {c.id: c for c in candidates}
    result = set()
    for d in detections:
        cid = d.attributes.get("symbol_candidate_id")
        c = by_id.get(cid)
        if c is None:
            continue
        letters = {str(t).strip().upper() for t in c.text}
        contradiction = (d.attributes.get("equipment_class") == "motor" and c.shape == "round_symbol"
                         and bool(letters & {"P", "PI", "PT", "T", "TI", "L", "LI", "F", "FI"}) and "M" not in letters)
        if contradiction or int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16) % 10 == 0:
            result.add(cid)
    return result


def _wider_source_blocks(candidates, provider):
    blocks = []
    for c in candidates[:4]:
        b = c.bbox
        pad = max(100, b.w, b.h)
        image, info = provider(b.x-pad, b.y-pad, b.w+2*pad, b.h+2*pad)
        blocks.extend([{"type": "text", "text": f"Additional source region around {c.id}; page rectangle {info.page_bbox.model_dump()}. This is newly retrieved neighboring context, not new candidate geometry."}, encode_image_block(image)])
    return blocks


def perceive_tile(**kwargs) -> PerceptionOutcome:
    """Fixed perception or bounded adaptive perception plus discovery validation."""
    outcome = _perceive_tile(**kwargs)
    policy = kwargs.get("policy") or SymbolPerceptionConfig()
    if kwargs.get("region_provider") is not None:
        _validate_discoveries(outcome, kwargs, policy)
    if policy.workflow == "adaptive":
        _quarantine_text_contradictions(outcome)
    return outcome


def _quarantine_text_contradictions(outcome):
    """A failed reinspection cannot make a known text/class conflict acceptable."""
    by_id = {c.id: c for c in outcome.candidates}
    blocked = set()
    for detection in outcome.detections:
        cid = detection.attributes.get("symbol_candidate_id")
        candidate = by_id.get(cid)
        if candidate is None or detection.attributes.get("equipment_class") != "motor":
            continue
        letters = {str(t).strip().upper() for t in candidate.text}
        if "M" not in letters and letters & {"P", "PI", "PT", "T", "TI", "L", "LI", "F", "FI"}:
            blocked.add(cid)
    if not blocked:
        return
    outcome.detections = [d for d in outcome.detections if d.attributes.get("symbol_candidate_id") not in blocked]
    outcome.batch.objects = [o for o in outcome.batch.objects if o.candidate_id not in blocked]
    for review in outcome.batch.candidate_reviews:
        if review.get("candidate_id") in blocked:
            review.update(status="uncertain", reason="Printed instrument letters contradict motor classification after bounded inspection")
    outcome.batch.uncertainties.append("Unresolved text/class contradictions were withheld from graph inputs")


def _boxes_duplicate(a, b):
    intersection = max(0, min(a.x2,b.x2)-max(a.x,b.x)) * max(0,min(a.y2,b.y2)-max(a.y,b.y))
    return intersection / max(1, min(a.w*a.h,b.w*b.h)) > 0.5


def _validate_discoveries(outcome, kwargs, policy):
    """One extra source-inspection request, at most four unanchored proposals.

    Native candidates are preferred, but eligibility no longer depends solely
    on the handcrafted shape vocabulary. No proposal is promoted on engineering
    plausibility, a tag alone, or overlap with an existing glyph.
    """
    page = kwargs["page"]
    proposals = []
    content = []
    all_candidates = kwargs.get("candidates")
    if all_candidates is None:
        all_candidates = symbol_candidates(page)
    occupied = [c.bbox for c in all_candidates] + [d.bbox for d in outcome.detections]
    for review in outcome.batch.candidate_reviews:
        if review.get("candidate_id") or not review.get("object") or not review.get("bbox"):
            continue
        box = BBox.model_validate(review["bbox"])
        if any(_boxes_duplicate(box, b) for b in occupied):
            review["validation"] = "withheld: overlaps an existing native or detected glyph"
            continue
        paths = [p.id for p in page.paths if _boxes_duplicate(box, p.bbox)]
        if not page.is_scanned and not paths:
            review["validation"] = "withheld: no source vector strokes in proposed bounds"
            continue
        if len(proposals) >= 4:
            review["validation"] = "withheld: bounded discovery queue exhausted"
            continue
        index = len(proposals)
        pad = max(60, box.w, box.h)
        image, info = kwargs["region_provider"](box.x-pad,box.y-pad,box.w+2*pad,box.h+2*pad)
        proposals.append((review, box, paths))
        content.extend([{"type":"text", "text":json.dumps({"proposal_id":str(index), "proposed_object":review["object"], "target_page_bbox":box.model_dump(), "context_page_bbox":info.page_bbox.model_dump()})}, encode_image_block(image)])
    if not proposals:
        return
    time_budget = policy.reasoning_timeout_s
    if kwargs.get("deadline") is not None:
        time_budget = min(time_budget, kwargs["deadline"]-time.monotonic())
    if time_budget <= 0:
        return
    tool = {"name":"validate_discoveries", "description":"Verify the actual drawn glyph of each proposal.", "input_schema":{
        "type":"object", "required":["decisions"], "properties":{"decisions":{"type":"array","items":{
            "type":"object", "required":["proposal_id","decision","kind","visible_strokes","reason"],
            "properties":{"proposal_id":{"type":"string"}, "decision":{"enum":["accept","reject","uncertain"]},
                          "kind":{"enum":["equipment","instrument","opc"]}, "visible_strokes":{"type":"array","items":{"type":"string"}}, "reason":{"type":"string"}},
        }}}}}
    try:
        if kwargs.get("on_attempt"):
            kwargs["on_attempt"]()
        client = kwargs.get("escalation_client") or kwargs["client"]
        response = client.messages_create(system="Independently inspect source ink for proposed P&ID symbols. Accept only a complete, distinct graphical symbol of the proposed kind inside the target box. Text alone, pipeline strokes, title blocks, borders, or a component already represented by a larger symbol do not qualify. Describe the visible strokes. Engineering plausibility is not evidence. Preserve uncertainty. Return exactly one validation per proposal_id." + CHINESE_EXPLANATIONS, messages=[{"role":"user","content":content}], tools=[tool], tool_choice={"type":"tool","name":tool["name"]}, max_tokens=3000, thinking={"type":"disabled"}, reasoning_mode_override="disabled", time_budget_s=time_budget, max_attempts=policy.transport_attempts)
        kwargs["cost_tracker"].record(response, step=len(kwargs["cost_tracker"].steps)+1, page_index=page.page_index, tile_id="discovery_validation")
        decisions = {}
        for block in response.content:
            b = block if isinstance(block, dict) else block.model_dump()
            if b.get("type") == "tool_use" and b.get("name") == tool["name"]:
                for decision in b.get("input",{}).get("decisions",[]):
                    decisions.setdefault(decision.get("proposal_id"),[]).append(decision)
        outcome.attempts += 1
        for index, (review, box, paths) in enumerate(proposals):
            rows = decisions.get(str(index),[])
            review["validation"] = rows
            if len(rows) != 1:
                continue
            row = rows[0]
            strokes = row.get("visible_strokes")
            if row.get("decision") != "accept" or row.get("kind") != review["object"].get("kind") or not isinstance(strokes, list) or not strokes or any(not isinstance(s, str) or not s.strip() for s in strokes) or not isinstance(row.get("reason"), str) or not row["reason"].strip():
                continue
            if any(_boxes_duplicate(box,d.bbox) for d in outcome.detections):
                continue
            batch = PerceptionBatch(objects=[PerceivedObject.model_validate(review["object"])])
            detections = project_batch(batch=batch, page=page, tile=kwargs["tile"], view_info=kwargs["view_info"], nearby_text_ids={t.id for t in page.text_spans}, ownership_bbox=kwargs["ownership_bbox"], candidates=None)
            for detection in detections:
                detection.attributes.update(geometry_basis="validated_visual_discovery", source_path_ids=paths, discovery_validation=row)
                outcome.detections.append(detection)
                review.update(status="selected", promoted_detection_id=detection.id, reason="Independent source inspection validated an unanchored discovery")
    except Exception as exc:
        # Preserve the discovery queue when validation is interrupted.
        outcome.recovery_diagnostics.append({"phase":"discovery_validation", "error":str(exc)})
        if is_non_retryable_api_error(exc):
            raise


def _merge_candidate_retry(
    initial: PerceptionOutcome, retry: PerceptionOutcome, ids: set[str]
) -> PerceptionOutcome:
    """Replace only unresolved outcomes, retaining earlier evidence for review."""
    new_reviews = {r.get("candidate_id"): r for r in retry.batch.candidate_reviews}
    resolved = {
        r.get("candidate_id")
        for r in retry.batch.candidate_reviews
        if r["status"] in {"selected", "reject"}
    }
    replaced = {
        cid for cid, review in new_reviews.items()
        if cid in ids and review["status"] != "unreviewed"
    }
    reviews = []
    for old in initial.batch.candidate_reviews:
        cid = old.get("candidate_id")
        if cid in ids:
            new = new_reviews.get(cid)
            current = dict(new if new and new["status"] != "unreviewed" else old)
            current["previous_review"] = old
            current["retry_attempted"] = True
            reviews.append(current)
        else:
            reviews.append(old)
    reviews.extend(r for r in retry.batch.candidate_reviews if not r.get("candidate_id"))
    initial.batch.candidate_reviews = reviews
    initial.batch.rejected_objects = [
        r
        for r in initial.batch.rejected_objects
        if (r.get("object") or {}).get("candidate_id", r.get("candidate_id")) not in resolved
    ] + retry.batch.rejected_objects
    initial.batch.objects = [
        o for o in initial.batch.objects if o.candidate_id not in replaced
    ] + [o for o in retry.batch.objects if o.candidate_id in replaced or not o.candidate_id]
    initial.batch.candidate_decisions = [
        d for d in initial.batch.candidate_decisions if d.candidate_id not in replaced
    ] + [d for d in retry.batch.candidate_decisions if d.candidate_id in replaced]
    initial.batch.observations.extend(retry.batch.observations)
    initial.batch.uncertainties.extend(retry.batch.uncertainties)
    initial.detections = [
        d for d in initial.detections
        if d.attributes.get("symbol_candidate_id") not in replaced
    ] + retry.detections
    initial.attempts = retry.attempts
    initial.contract_failed = _contract_failed(initial.batch, initial.candidates)
    return initial


def parse_perception_response(
    response: Any,
    *,
    page: PageEvidence | None = None,
    view_info: ViewInfo | None = None,
) -> PerceptionBatch:
    text_parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "tool_use":
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if name == "submit_pid_objects":
                value = (
                    block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
                )
                return _validate_perception_payload(
                    _normalise_perception_payload(value or {}, page=page, view_info=view_info)
                )
        elif block_type == "text":
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if value:
                text_parts.append(str(value))

    raw = "\n".join(text_parts).strip()
    if not raw:
        raise PerceptionResponseFormatError(
            "perception response contained neither tool input nor JSON text"
        )
    raw = _FENCE_RE.sub("", raw).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise PerceptionResponseFormatError(
                "perception response did not contain a JSON object"
            ) from None
        try:
            value = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PerceptionResponseFormatError(
                f"perception response contained invalid JSON: {exc.msg}"
            ) from None
    return _validate_perception_payload(
        _normalise_perception_payload(value, page=page, view_info=view_info)
    )


def _response_diagnostic(response: Any) -> dict[str, Any]:
    """Return a JSON-safe response snapshot without echoing request image data."""
    content: list[Any] = []
    for block in getattr(response, "content", None) or []:
        if isinstance(block, dict):
            content.append(_json_safe(block))
        elif hasattr(block, "model_dump"):
            content.append(_json_safe(block.model_dump(mode="json")))
        else:
            content.append(
                _json_safe(
                    {
                        key: getattr(block, key)
                        for key in (
                            "type",
                            "text",
                            "thinking",
                            "reasoning",
                            "name",
                            "input",
                        )
                        if getattr(block, key, None) is not None
                    }
                )
            )
    for block in content:
        if isinstance(block, dict):
            for field in ("thinking", "reasoning"):
                if isinstance(block.get(field), str):
                    # Retain structured outputs exactly; reasoning text can be
                    # enormous and is not a recoverable symbol record.
                    value = block.pop(field)
                    block[field + "_chars"] = len(value)
                    block[field + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
    return _json_safe(
        {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "content": content,
            "usage": getattr(response, "usage", None),
        }
    )


def _request_diagnostic(request: dict[str, Any]) -> dict[str, Any]:
    """Persist the logical request and image fingerprints, never credentials."""
    def compact(value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("type") == "base64" and isinstance(value.get("data"), str):
                return {
                    "type": "base64",
                    "media_type": value.get("media_type"),
                    "data_sha256": hashlib.sha256(value["data"].encode()).hexdigest(),
                }
            return {k: compact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value

    return compact(request)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    return repr(value)


def _validate_perception_payload(value: Any) -> PerceptionBatch:
    """Validate objects independently so one malformed sibling is recoverable."""
    if not isinstance(value, dict):
        return PerceptionBatch.model_validate(value)
    raw_objects = value.get("objects", [])
    if not isinstance(raw_objects, list):
        return PerceptionBatch.model_validate(value)

    accepted: list[PerceivedObject] = []
    rejected: list[dict[str, Any]] = list(value.get("rejected_objects", []))
    for index, item in enumerate(raw_objects):
        try:
            accepted.append(PerceivedObject.model_validate(item))
        except Exception as exc:  # noqa: BLE001 - retain valid siblings from one call
            rejected.append({"index": index, "object": item, "error": str(exc)})
    if (
        raw_objects
        and not accepted
        and not value.get("_single_outcome_contract")
    ):
        raise ValueError(
            f"all {len(raw_objects)} perception object(s) were invalid: " + rejected[0]["error"]
        )

    diagnostics = dict(value)
    diagnostics["objects"] = accepted
    diagnostics["rejected_objects"] = rejected
    return PerceptionBatch.model_validate(diagnostics)


def _expand_candidate_results(value: dict[str, Any]) -> dict[str, Any]:
    """Adapt the single-outcome wire format to the compatible checkpoint model.

    Invalid siblings and duplicate IDs remain uncertain; no last-write-wins or
    automatic acceptance of conflicting outcomes. Old checkpoints use the same
    internal object/disposition representation, but it is no longer advertised.
    """
    if set(value) - {"candidate_results", "proposals"}:
        raise PerceptionResponseFormatError("candidate_results cannot be mixed with legacy fields")
    rows = value["candidate_results"]
    proposals = value.get("proposals", [])
    if not isinstance(rows, list) or not isinstance(proposals, list):
        raise PerceptionResponseFormatError("candidate_results and proposals must be arrays")
    by_id: dict[str, list[Any]] = {}
    rejected = []
    for index, row in enumerate(rows):
        cid = row.get("candidate_id") if isinstance(row, dict) else None
        if not isinstance(cid, str) or not cid.strip():
            rejected.append({"index": index, "object": row, "error": "missing candidate ID"})
        else:
            by_id.setdefault(cid, []).append(row)
    objects = []
    decisions = []
    for cid, items in by_id.items():
        try:
            if len(items) != 1:
                raise ValueError("duplicate outcome for the same native ID")
            result = CandidateResult.model_validate(items[0])
            if result.decision == "symbol":
                objects.append({**result.symbol, "candidate_id": cid})
            else:
                decisions.append(
                    {
                        "candidate_id": cid,
                        "decision": "reject" if result.decision == "non_symbol" else "uncertain",
                        "reason": result.reason,
                        "non_symbol_basis": result.non_symbol_basis,
                    }
                )
        except ValueError as exc:
            reason = f"Invalid candidate result: {exc}"
            decisions.append({"candidate_id": cid, "decision": "uncertain", "reason": reason})
            rejected.append({"candidate_id": cid, "outcomes": items, "error": reason})
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict) or "candidate_id" in proposal:
            rejected.append(
                {
                    "proposal_index": index,
                    "object": proposal,
                    "error": "proposal must not claim a native candidate ID",
                }
            )
        else:
            objects.append(proposal)
    return {
        "objects": objects,
        "candidate_decisions": decisions,
        "rejected_objects": rejected,
        "_single_outcome_contract": True,
    }


def _normalise_perception_payload(
    value: Any,
    *,
    page: PageEvidence | None,
    view_info: ViewInfo | None,
) -> Any:
    if not isinstance(value, dict):
        return value
    payload = _expand_candidate_results(value) if "candidate_results" in value else dict(value)
    # Legacy object/disposition payloads remain readable for saved results.
    objects = payload.get("objects", [])
    if isinstance(objects, str):
        decoded = _decode_json_array(objects)
        if decoded is not None:
            objects = decoded
            payload["objects"] = objects
    if not isinstance(objects, list) or page is None or view_info is None:
        return payload

    normalised_objects: list[Any] = []
    for item in objects:
        if (
            not isinstance(item, dict)
            or item.get("candidate_id")
            or not isinstance(item.get("bbox"), dict)
        ):
            normalised_objects.append(item)
            continue
        obj = dict(item)
        bbox, recovery = _recover_bbox_payload(obj["bbox"], page=page, view_info=view_info)
        obj["bbox"] = bbox
        if recovery != "normalized":
            attributes = dict(obj.get("attributes") or {})
            attributes.setdefault("coordinate_recovery", recovery)
            obj["attributes"] = attributes
        normalised_objects.append(obj)
    payload["objects"] = normalised_objects
    return payload


def _decode_json_array(value: str) -> list[Any] | None:
    text = _FENCE_RE.sub("", value.strip()).strip()
    candidates = [text]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, list):
            return decoded
    return None


def _recover_bbox_payload(
    value: dict[str, Any],
    *,
    page: PageEvidence,
    view_info: ViewInfo,
) -> tuple[dict[str, float], str]:
    try:
        x, y, w, h = (float(value[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return value, "normalized"
    if not all(math.isfinite(part) for part in (x, y, w, h)) or w <= 0 or h <= 0:
        return value, "normalized"

    clipped = _clip_small_overflow(x=x, y=y, w=w, h=h, width=1.0, height=1.0)
    if clipped is not None:
        # Reconstructing width as (x + w) - x can differ by floating-point
        # roundoff even with no clipping. Report only a material correction.
        recovery = (
            "normalized"
            if all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(clipped, (x, y, w, h), strict=True)
            )
            else "normalized_clipped"
        )
        return _bbox_dict(clipped), recovery

    page_candidate = _clip_small_overflow(
        x=x, y=y, w=w, h=h, width=float(page.width), height=float(page.height)
    )
    if page_candidate is not None and not _mostly_inside(
        page_candidate, _bbox_tuple(view_info.page_bbox)
    ):
        page_candidate = None

    view_candidate = _clip_small_overflow(
        x=x,
        y=y,
        w=w,
        h=h,
        width=float(view_info.view_size[0]),
        height=float(view_info.view_size[1]),
    )
    local_as_page = (
        _local_pixels_to_page(view_candidate, view_info=view_info)
        if view_candidate is not None
        else None
    )
    if local_as_page is not None and not _mostly_inside(
        local_as_page, _bbox_tuple(view_info.page_bbox)
    ):
        view_candidate = None
        local_as_page = None

    if page_candidate is not None and view_candidate is not None:
        if local_as_page is None or not _boxes_nearly_equal(page_candidate, local_as_page):
            raise ValueError(
                "bbox coordinates are ambiguous between page-global and tile-local pixels"
            )
        return _page_pixels_to_normalized(page_candidate, view_info), "page_pixels"
    if page_candidate is not None:
        return _page_pixels_to_normalized(page_candidate, view_info), "page_pixels"
    if view_candidate is not None:
        return _view_pixels_to_normalized(view_candidate, view_info), "tile_pixels"
    return value, "normalized"


def _clip_small_overflow(
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float] | None:
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(width, x + w)
    bottom = min(height, y + h)
    if right <= left or bottom <= top:
        return None
    retained = ((right - left) * (bottom - top)) / (w * h)
    tolerance_x = max(width * 0.05, 1e-9)
    tolerance_y = max(height * 0.05, 1e-9)
    if (
        retained < 0.8
        or x < -tolerance_x
        or y < -tolerance_y
        or x + w > width + tolerance_x
        or y + h > height + tolerance_y
    ):
        return None
    return left, top, right - left, bottom - top


def _mostly_inside(
    candidate: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> bool:
    x, y, w, h = candidate
    cx, cy, cw, ch = container
    left = max(x, cx)
    top = max(y, cy)
    right = min(x + w, cx + cw)
    bottom = min(y + h, cy + ch)
    if right <= left or bottom <= top:
        return False
    return ((right - left) * (bottom - top)) / (w * h) >= 0.8


def _bbox_tuple(bbox: BBox) -> tuple[float, float, float, float]:
    return float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)


def _local_pixels_to_page(
    bbox: tuple[float, float, float, float], *, view_info: ViewInfo
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    sx = view_info.scale_x or 1.0
    sy = view_info.scale_y or 1.0
    return (
        view_info.origin[0] + x / sx,
        view_info.origin[1] + y / sy,
        w / sx,
        h / sy,
    )


def _page_pixels_to_normalized(
    bbox: tuple[float, float, float, float], view_info: ViewInfo
) -> dict[str, float]:
    x, y, w, h = bbox
    sx = view_info.scale_x or 1.0
    sy = view_info.scale_y or 1.0
    return _bbox_dict(
        (
            (x - view_info.origin[0]) * sx / view_info.view_size[0],
            (y - view_info.origin[1]) * sy / view_info.view_size[1],
            w * sx / view_info.view_size[0],
            h * sy / view_info.view_size[1],
        )
    )


def _page_bbox_to_normalized(bbox: BBox, view_info: ViewInfo) -> dict[str, float]:
    """Express page-space evidence in the model image's only coordinate frame."""
    raw = _page_pixels_to_normalized(_bbox_tuple(bbox), view_info)
    left = max(0.0, min(1.0, raw["x"]))
    top = max(0.0, min(1.0, raw["y"]))
    right = max(left, min(1.0, raw["x"] + raw["w"]))
    bottom = max(top, min(1.0, raw["y"] + raw["h"]))
    return {"x": left, "y": top, "w": right - left, "h": bottom - top}


def _view_pixels_to_normalized(
    bbox: tuple[float, float, float, float], view_info: ViewInfo
) -> dict[str, float]:
    x, y, w, h = bbox
    return _bbox_dict(
        (
            x / view_info.view_size[0],
            y / view_info.view_size[1],
            w / view_info.view_size[0],
            h / view_info.view_size[1],
        )
    )


def _boxes_nearly_equal(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return all(abs(a - b) <= 1.0 for a, b in zip(left, right, strict=True))


def _bbox_dict(value: tuple[float, float, float, float]) -> dict[str, float]:
    return dict(zip(("x", "y", "w", "h"), value, strict=True))


def _report_response_content(response: Any, reporter: ProgressReporter) -> None:
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "thinking":
            value = (
                block.get("thinking")
                if isinstance(block, dict)
                else getattr(block, "thinking", None)
            )
            if value:
                reporter.on_thinking(text=str(value))
        elif block_type == "text":
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if value:
                reporter.on_text(text=str(value))


def project_batch(
    *,
    batch: PerceptionBatch,
    page: PageEvidence,
    tile: Tile,
    view_info: ViewInfo,
    nearby_text_ids: set[str],
    ownership_bbox: BBox | None = None,
    candidates: list[SymbolCandidate] | None = None,
) -> list[DetectionRecord]:
    out: list[DetectionRecord] = []
    by_id = {c.id: c for c in candidates or []}
    reviews = {
        c.id: {
            "candidate_id": c.id,
            "bbox": c.bbox.model_dump(),
            "source_path_ids": c.source_path_ids,
            "status": "unreviewed",
            "reason": "Model did not select or assess this native candidate",
        }
        for c in candidates or []
    }
    decisions: dict[str, list[CandidateDisposition]] = {}
    for decision in batch.candidate_decisions:
        decisions.setdefault(decision.candidate_id, []).append(decision)
        if decision.candidate_id not in by_id:
            batch.rejected_objects.append(
                {
                    "candidate_id": decision.candidate_id,
                    "error": "unknown or unowned candidate decision",
                }
            )
    counts: dict[str, int] = {}
    signatures: dict[str, set[str]] = {}
    for obj in batch.objects:
        if obj.candidate_id:
            counts[obj.candidate_id] = counts.get(obj.candidate_id, 0) + 1
            attrs = obj.graph_attributes()
            for key in ("ignored_model_bbox", "structural_description"):
                attrs.pop(key, None)
            signature = json.dumps([obj.kind, obj.label, obj.raw_text, attrs], sort_keys=True)
            signatures.setdefault(obj.candidate_id, set()).add(signature)
    for cid, rows in decisions.items():
        if cid in reviews:
            reviews[cid].update(status=rows[0].decision, reason=rows[0].reason)
            if rows[0].non_symbol_basis is not None:
                reviews[cid]["non_symbol_basis"] = rows[0].non_symbol_basis
    for cid in reviews:
        if (
            len(signatures.get(cid, set())) > 1
            or (counts.get(cid) and cid in decisions)
            or len({d.decision for d in decisions.get(cid, [])}) > 1
        ):
            reviews[cid].update(status="uncertain", reason="Conflicting candidate decisions")
        if counts.get(cid, 0) > 1 or (counts.get(cid) and cid in decisions):
            reviews[cid]["observation_count"] = counts[cid]
            reviews[cid]["observations"] = [
                obj.model_dump(mode="json") for obj in batch.objects if obj.candidate_id == cid
            ]
        if cid in decisions:
            reviews[cid]["decisions"] = [d.model_dump() for d in decisions[cid]]
    emitted: set[str] = set()
    for index, obj in enumerate(batch.objects):
        candidate = by_id.get(obj.candidate_id or "")
        if obj.candidate_id:
            error = None
            if candidate is None:
                error = "unknown or unowned native candidate ID"
            elif len(signatures[obj.candidate_id]) > 1 or obj.candidate_id in decisions:
                error = "conflicting candidate decisions"
            elif not _candidate_kind_compatible(candidate, obj):
                error = "classification contradicts native glyph geometry"
            if error:
                batch.rejected_objects.append(
                    {"index": index, "object": obj.model_dump(mode="json"), "error": error}
                )
                if candidate:
                    reviews[candidate.id].update(status="uncertain", reason=error)
                continue
            if obj.candidate_id in emitted:
                continue
        if candidate:
            bbox = candidate.bbox.model_copy()
            reviews[candidate.id].update(status="selected", reason="Model classified native glyph")
            emitted.add(candidate.id)
            if counts[candidate.id] > 1:
                reviews[candidate.id]["reason"] = "Equivalent repeated classifications consolidated"
        else:
            assert obj.bbox is not None
            bbox = _project_normalized_bbox(obj.bbox, page=page, view_info=view_info)
        if ownership_bbox is not None and not _bbox_center_is_owned(bbox, ownership_bbox, page):
            continue
        source_text_ids = sorted(set(obj.source_text_ids).intersection(nearby_text_ids))
        label = " ".join(obj.label.split()).strip()
        raw_text = " ".join((obj.raw_text or "").split()).strip() or None
        attributes = obj.graph_attributes()
        attributes["source_tile"] = tile.id
        attributes["model_confidence"] = obj.confidence
        if candidate:
            attributes["symbol_candidate_id"] = candidate.id
            attributes["geometry_basis"] = "native_symbol_candidate"
            attributes["source_path_ids"] = candidate.source_path_ids
            attributes["candidate_shape"] = candidate.shape
            if counts[candidate.id] > 1:
                attributes["equivalent_observation_count"] = counts[candidate.id]
        elif candidates is not None:
            batch.candidate_reviews.append(
                {
                    "status": "uncertain",
                    "object_index": index,
                    "bbox": bbox.model_dump(),
                    "object": obj.model_dump(mode="json"),
                    "reason": "Model proposal has no selected native glyph",
                }
            )
            # Retain proposals as review evidence, not graph endpoints. In
            # particular a guessed OPC must not certify a cross-sheet edge.
            continue
        if source_text_ids:
            attributes["source_text_ids"] = source_text_ids
        detection_id = stable_evidence_id(
            "det",
            page.page_index,
            tile.id,
            index,
            obj.kind,
            label,
            bbox.model_dump_json(),
        )
        out.append(
            DetectionRecord(
                id=detection_id,
                page_index=page.page_index,
                tile_id=tile.id,
                kind=obj.kind,
                label=label,
                bbox=bbox,
                confidence=obj.confidence,
                raw_text=raw_text,
                attributes=attributes,
                source_text_ids=source_text_ids,
            )
        )
    batch.candidate_reviews.extend(reviews[cid] for cid in sorted(reviews))
    return out


def _candidate_kind_compatible(candidate: SymbolCandidate, obj: PerceivedObject) -> bool:
    if candidate.shape == "continuation_arrow":
        return obj.kind == "opc"
    if candidate.shape in {"valve_body", "open_inline_valve"}:
        return obj.kind == "equipment" and (
            obj.equipment_class in (None, "valve", "unclassified") or bool(obj.valve_type)
        )
    if candidate.shape == "capsule_body":
        return (
            obj.kind in {"equipment", "instrument"}
            and not obj.valve_type
            and obj.equipment_class != "valve"
        )
    if candidate.shape == "round_symbol" and obj.kind == "opc":
        # Round outlines can enclose actual connector arrows. Require a source
        # description of the internal arrow AND its line/continuation context.
        evidence = (obj.recognition_evidence or "").casefold()
        arrow = any(t in evidence for t in ("arrow", "箭头"))
        context = any(t in evidence for t in ("line", "continuation", "管线", "管道", "连接", "延续"))
        return arrow and context
    return obj.kind in {"instrument", "equipment"}


def _bbox_center_is_owned(bbox: BBox, core: BBox, page: PageEvidence) -> bool:
    center_x = bbox.x + bbox.w / 2
    center_y = bbox.y + bbox.h / 2
    right_owned = center_x < core.x2 or core.x2 >= page.width
    bottom_owned = center_y < core.y2 or core.y2 >= page.height
    return center_x >= core.x and center_y >= core.y and right_owned and bottom_owned


def _project_normalized_bbox(
    bbox: NormalizedBBox,
    *,
    page: PageEvidence,
    view_info: ViewInfo,
) -> BBox:
    local_x0 = bbox.x * view_info.view_size[0]
    local_y0 = bbox.y * view_info.view_size[1]
    local_x1 = (bbox.x + bbox.w) * view_info.view_size[0]
    local_y1 = (bbox.y + bbox.h) * view_info.view_size[1]
    sx = view_info.scale_x or 1.0
    sy = view_info.scale_y or 1.0
    x0 = view_info.origin[0] + local_x0 / sx
    y0 = view_info.origin[1] + local_y0 / sy
    x1 = view_info.origin[0] + local_x1 / sx
    y1 = view_info.origin[1] + local_y1 / sy
    left = max(0, min(page.width - 1, int(round(min(x0, x1)))))
    top = max(0, min(page.height - 1, int(round(min(y0, y1)))))
    right = max(left + 1, min(page.width, int(round(max(x0, x1)))))
    bottom = max(top + 1, min(page.height, int(round(max(y0, y1)))))
    return BBox(x=left, y=top, w=right - left, h=bottom - top)


def _bounded_knowledge_context(context, legend_entries):
    """Bind model-facing drawing definitions to the selected legend context.

    Resolver snapshots may retain a full legend pack for provenance. Serializing
    that pack here would bypass local selection and repeat base64 images and
    rejected model diagnostics as prompt text. The separately labelled image
    blocks are the sole visual-reference channel. Do not mutate the snapshot.
    """
    if not context:
        return {}
    from diagex.knowledge.resolver import digest

    bounded = {
        **context,
        "drawing_definitions": legend_entries,
        "drawing_definitions_scope": "Selected local legend definitions; explicit drawing definitions take precedence over general references.",
        "retrieval_identity": context.get("retrieval_identity", context.get("identity")),
    }
    bounded.pop("identity", None)
    bounded["identity"] = digest(bounded)
    return bounded


def _knowledge_image_blocks(context):
    from diagex.knowledge.library import model_assets
    from diagex.knowledge.resolver import reference_images
    images = list(reference_images(context))
    if not images:
        return []
    blocks = []
    for asset, image in zip(model_assets(context), images, strict=True):
        blocks.extend([{"type": "text", "text": (
            "Knowledge reference illustration only, NOT this drawing. Do not extract its symbols or annotations. "
            f"Reference {asset['reference_id']} v{asset['reference_version']}; asset {asset['asset_id']}; "
            f"variants: {', '.join(asset['variant_ids']) or 'not specified'}."
        )}, encode_image_block(image)])
    return blocks


def _bind_knowledge(batch, context):
    """Attach source-bound attributions without rejecting uncertain objects."""
    from diagex.knowledge.library import supplied_trace
    for obj in batch.objects:
        # Free-form legacy attributes cannot bypass reference validation.
        for key in ("knowledge_match", "knowledge_matches", "supplied_knowledge", "knowledge_match_error", "knowledge_match_errors"):
            obj.attributes.pop(key, None)
        matches, errors = knowledge_matches(context, obj)
        if context:
            obj.attributes["supplied_knowledge"] = supplied_trace(context)
        if matches:
            obj.attributes["knowledge_match"] = matches[0]  # Historical readers.
            obj.attributes["knowledge_matches"] = matches
        if errors:
            obj.attributes["knowledge_match_error"] = errors[0]["reason"]
            obj.attributes["knowledge_match_errors"] = errors
        markers = {s["example_marker"] for e in context.get("references", []) for s in e.get("text_slots", [])}
        for tag_field in ("printed_tag", "canonical_tag", "drawing_ref", "line_id"):
            if getattr(obj, tag_field) in markers:
                obj.attributes.setdefault("discarded_reference_placeholders", {})[tag_field] = getattr(obj, tag_field)
                setattr(obj, tag_field, None)
        # Legacy provider/checkpoint shapes retain aliases in attributes even
        # after promoting them into typed fields. Clear those copies too, or
        # graph_attributes() would resurrect the discarded placeholder.
        for alias in ("printed_tag", "raw_text", "canonical_tag", "label", "drawing_ref", "target_sheet", "line_id"):
            value = obj.attributes.get(alias)
            if isinstance(value, str) and value in markers:
                obj.attributes.setdefault("discarded_reference_placeholders", {})[alias] = value
                obj.attributes.pop(alias)


def _bind_legend(batch, entries):
    from diagex.vision.reference_evidence import legend_matches
    for obj in batch.objects:
        for key in ("legend_matches", "legend_match_errors", "legend_entry_ids", "recognition_evidence", "recognition_method"):
            obj.attributes.pop(key, None)
        matches, errors = legend_matches(obj.legend_entry_ids, obj.legend_evidence, entries)
        obj.attributes["recognition_method"] = "vlm"
        if obj.recognition_evidence:
            obj.attributes["recognition_evidence"] = obj.recognition_evidence
        if matches:
            obj.attributes["legend_matches"] = matches
            obj.attributes["legend_entry_ids"] = [m["legend_entry_id"] for m in matches]
        if errors:
            obj.attributes["legend_match_errors"] = errors
