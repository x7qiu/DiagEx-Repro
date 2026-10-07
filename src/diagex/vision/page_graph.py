"""One bounded, page-level relationship solve for evidence-v2.

Object identity and geometry are owned by deterministic perception/fusion.
This module gives the reasoning model only short references to those objects
and to line candidates recovered from the PDF.  Provider output is validated
before it can affect the reconciled graph.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from collections.abc import Callable
from typing import Any, Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from diagex.llm.client import (
    LLMClient,
    is_malformed_tool_json_error,
    is_reasoning_required_error,
    model_requires_reasoning,
)
from diagex.llm.cost import CostTracker
from diagex.ui.progress import ProgressReporter
from diagex.vision.encode import encode_image_block
from diagex.vision.evidence import PageEvidence, VisualLineStyle
from diagex.vision.legend_context import select_graph_legend_context
from diagex.vision.models import (
    BBox,
    Confidence,
    LineType,
    ReconciledEdge,
    ReconciledNode,
)
from diagex.vision.topology import TopologyResult, build_page_topology, route_evidence_failures

_SIGNAL_TYPES = {
    "signal_electric",
    "signal_pneumatic",
    "instrument_capillary",
    "electrical_power",
}
_DRAWING_REF_RE = re.compile(r"\bDW\d{2}[- ]?\d{3,5}\b", re.IGNORECASE)
_LINE_LEGEND_SIGNAL_TERMS = (
    "signal",
    "electric",
    "pneumatic",
    "hydraulic",
    "capillary",
    "software",
    "data link",
    "电信号",
    "气压",
    "液压",
    "毛细",
    "软件",
    "数据链",
)
_LINE_LEGEND_PROCESS_TERMS = (
    "process line",
    "main process",
    "secondary process",
    "工艺管线",
)
_LINE_LEGEND_TOPOLOGY_TERMS = (
    "boundary",
    "crossing",
    "junction",
    "connection",
    "界限",
    "分界",
    "交叉",
    "连接",
)


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_ref: str
    decision: Literal["keep", "retype", "reject", "uncertain"]
    line_type: LineType | None = None
    reverse: bool = False
    confidence: Confidence = "medium"
    evidence: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, value: Any) -> Any:
        return _coerce_string_list(value)

    @model_validator(mode="after")
    def _retype_has_type(self) -> CandidateDecision:
        if self.decision == "retype" and self.line_type is None:
            raise ValueError("retype requires line_type")
        return self


class ProposedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_ref: str
    to_ref: str
    line_type: LineType
    cross_sheet: bool = False
    confidence: Confidence = "medium"
    evidence: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class OpcUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_ref: str
    direction: Literal["in", "out"] | None = None
    service: str | None = None
    drawing_ref: str | None = None
    line_id: str | None = None
    raw_printed_text: str | None = None
    evidence: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, value: Any) -> Any:
        return _coerce_string_list(value)

    @field_validator("service", "drawing_ref", "line_id", "raw_printed_text", mode="before")
    @classmethod
    def _clean_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split()).strip()
        return text or None


class PageGraphUncertainty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_refs: list[str] = Field(default_factory=list, max_length=12)
    candidate_refs: list[str] = Field(default_factory=list, max_length=12)
    reason: str

    @field_validator("node_refs", "candidate_refs", mode="before")
    @classmethod
    def _coerce_refs(cls, value: Any) -> Any:
        values = _coerce_string_list(value)
        return values[:12] if isinstance(values, list) else values


class ConnectionHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_ref: str
    to_ref: str
    line_type: LineType = "process"
    rationale: str
    question: str = "Does the drawing contain a continuous route between these endpoints?"
    context_sources: list[str] = Field(default_factory=list, max_length=8)


class PageGraphSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_decisions: list[CandidateDecision] = Field(default_factory=list, max_length=256)
    new_relations: list[ProposedRelation] = Field(default_factory=list, max_length=256)
    opc_updates: list[OpcUpdate] = Field(default_factory=list, max_length=64)
    uncertainties: list[PageGraphUncertainty] = Field(default_factory=list, max_length=128)
    hypotheses: list[ConnectionHypothesis] = Field(default_factory=list, max_length=64)

    @field_validator(
        "candidate_decisions",
        "new_relations",
        "opc_updates",
        "uncertainties",
        "hypotheses",
        mode="before",
    )
    @classmethod
    def _coerce_provider_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return value
        return decoded


class VisualCandidateAssessment(BaseModel):
    """Visible facts about one deterministic line candidate.

    This deliberately excludes graph decisions.  The non-thinking vision call
    reports only what can be seen; deterministic code and the text reasoner
    decide what that evidence means for the engineering graph.
    """

    # Some OpenAI-compatible providers add a short explanatory field beside
    # the requested schema. It is harmless evidence, not a format failure;
    # ignore it while retaining strict validation of every field we consume.
    model_config = ConfigDict(extra="ignore")

    candidate_ref: str
    route_visible: Literal["yes", "no", "uncertain"] = "uncertain"
    endpoint_alignment: Literal["both", "one", "neither", "uncertain"] = "uncertain"
    observed_style: VisualLineStyle = "unknown"
    arrow_direction: Literal["forward", "reverse", "none", "uncertain"] = "uncertain"
    legend_class: LineType | None = None
    confidence: Confidence = "medium"
    evidence: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, value: Any) -> Any:
        return _coerce_string_list(value)


class PageLineEvidenceSubmission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessments: list[VisualCandidateAssessment] = Field(default_factory=list, max_length=256)

    @field_validator("assessments", mode="before")
    @classmethod
    def _coerce_provider_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


class PageLineEvidence(BaseModel):
    page_index: int
    assessments: dict[str, VisualCandidateAssessment] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)
    format_recovery: list[dict[str, Any]] = Field(default_factory=list)


class PageGraphResponseFormatError(ValueError):
    """Raised after the bounded page-graph format recovery is exhausted."""

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


class PageGraphResult(BaseModel):
    page_index: int
    submission: dict[str, Any] = Field(default_factory=dict)
    edges: list[ReconciledEdge] = Field(default_factory=list)
    opc_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    accepted_candidate_ids: list[str] = Field(default_factory=list)
    provisional_candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    format_recovery: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)


_TOOL = {
    "name": "submit_page_graph",
    "description": "Classify enumerated P&ID relationships for one page.",
    "input_schema": PageGraphSubmission.model_json_schema(),
}

_LINE_EVIDENCE_TOOL = {
    "name": "submit_line_evidence",
    "description": "Report visible evidence for enumerated line candidates on one P&ID page.",
    "input_schema": PageLineEvidenceSubmission.model_json_schema(),
}

_LINE_EVIDENCE_SYSTEM = """\
Inspect the supplied P&ID page and report only visible facts for the enumerated
line candidates. This is a perception task, not graph reasoning.

For every candidate_ref:
- route_visible: whether the marked route can actually be followed in the image.
- endpoint_alignment: whether it visibly touches both, one, or neither marked endpoint.
- observed_style: solid, dashed, dotted, dash_dot, or unknown.
- arrow_direction: forward follows supplied from_ref to to_ref; reverse is the opposite.
- legend_class: set only when the supplied project legend visibly supports the mapping.
- evidence: at most four short literal observations.

Do not infer a connection from matching tags, loop numbers, engineering
expectations, or proximity. Do not invent nodes or relationships. Use uncertain
when the marks are too small, cross another line, disappear under a symbol, or
are not visible in the supplied images. Return exactly one submit_line_evidence
tool call and no prose.
"""

_PAGE_GRAPH_REASONING_BATCH_SIZE = 8

_REASONING_SYSTEM = """\
Resolve a bounded group of remaining semantic relationships for one P&ID page
from structured evidence. Visual perception and deterministic topology are
already complete: never invent, delete, rename, or move nodes. Use only supplied
node_ref and candidate_ref values.

Produce a compact engineering decision memo for the subsequent serializer. Use
one short line per candidate in this form:
candidate_ref | keep/retype/reject/uncertain | line type or - | forward/reverse |
confidence | literal evidence
Then add only evidence-supported NEW, OPC, or UNCERTAIN lines. Do not restate the
node table, narrate the page, or repeatedly reconsider a candidate. This pass
must reason, but it must remain concise; it does not call tools or emit the final
schema.

Candidate decisions:
- keep: the recovered line and its current type are visibly correct.
- retype: the endpoints are correct but line_type must change.
- reject: the recovered path is a border, symbol stroke, crossing artefact, or
  connects the wrong engineering objects.
- uncertain: the images do not support a safe choice.
Set reverse=true only when visible arrows or control semantics prove that the
candidate's supplied from_ref/to_ref direction is backwards.

The candidate table includes a non-thinking visual assessment. Treat it as
evidence, not as permission to invent topology. Visual style and engineering
meaning are separate. Map a styled connection to signal_electric,
signal_pneumatic, instrument_capillary, electrical_power, process, or other only
when the project legend and endpoint roles support that meaning. A shared loop
number alone does not prove a drawn edge. Use uncertain when the project
convention or endpoint function is unavailable.

Endpoint rules:
- process normally joins process equipment, valves, and OPCs. An instrument
  endpoint needs explicit inline/tap evidence.
- electric signal normally has an instrument/controller at one end and an
  instrument or identified actuator/control valve at the other.
- pneumatic signal normally joins a pneumatic controller/transmitter and an
  identified pneumatic actuator/control valve.
- capillary/impulse connections require a sensing instrument and a process tap
  or process object.
- electrical power requires an identified motor or electrical load.
Generic equipment-to-equipment signal proposals are unsupported.

new_relations may connect only enumerated nodes and require direct visible line
or explicit off-page continuation evidence. Cross-sheet relations require OPC
nodes on different pages, compatible in/out directions, and printed drawing,
line, or service evidence. Put unsupported possibilities in uncertainties.

OPC updates must copy literal printed evidence; omit fields that are not shown.
Keep evidence short and literal. This reasoning pass receives no page image. Use
only the supplied structured visual and native-PDF evidence and mark unsupported
choices uncertain.
"""

_SERIALIZATION_SYSTEM = """\
Convert an engineering decision memo into exactly one submit_page_graph tool
call. Do not repeat the engineering analysis and do not add decisions that are
absent from the memo. Use the accompanying source payload only to copy exact
node_ref and candidate_ref values and to resolve harmless spelling differences.

Use empty arrays when the memo contains no candidate decisions, new relations,
OPC updates, or uncertainties. Never invent nodes or candidate references. Keep
evidence entries short and literal. Return no prose outside the tool call.
"""


def classify_page_line_evidence(
    *,
    client: LLMClient,
    cost_tracker: CostTracker,
    reporter: ProgressReporter,
    page: PageEvidence,
    rendered_image: Image.Image,
    nodes: list[ReconciledNode],
    topology: TopologyResult,
    step: int,
    legend_line_images: list[dict[str, Any]] | None = None,
    max_tokens: int = 6000,
    on_attempt: Callable[[], None] | None = None,
    inspection_attempts: int = 2,
) -> PageLineEvidence:
    """Collect bounded, non-thinking visual facts for topology candidates."""

    node_refs, _ = _node_reference_table(page.page_index, nodes, nodes)
    edge_refs, _ = _edge_reference_table(topology.edges)
    if not edge_refs:
        return PageLineEvidence(page_index=page.page_index)
    payload = {
        "page": page.page_index + 1,
        "page_size": [page.width, page.height],
        "nodes": [_node_payload(ref, node) for ref, node in node_refs.items()],
        "candidate_edges": [_edge_payload(ref, edge, node_refs) for ref, edge in edge_refs.items()],
    }
    overview = _annotated_overview(rendered_image, node_refs, edge_refs)
    montage = _relationship_montage(rendered_image, node_refs, edge_refs, limit=24)
    legend_montage = _legend_line_montage(legend_line_images or [])
    image_blocks = [encode_image_block(overview)]
    if montage is not None:
        image_blocks.append(encode_image_block(montage))
    if legend_montage is not None:
        image_blocks.append(encode_image_block(legend_montage))

    invalid_responses: list[dict[str, Any]] = []
    for attempt in range(1, inspection_attempts + 1):
        recovery = ""
        if attempt == 2:
            recovery = (
                "The previous response was invalid. Call submit_line_evidence exactly once; "
                "return one compact assessment per candidate_ref and no prose.\n"
            )
        if on_attempt is not None:
            on_attempt()
        try:
            response = client.messages_create(
                system=_LINE_EVIDENCE_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            *image_blocks,
                            {
                                "type": "text",
                                "text": recovery
                                + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                            },
                        ],
                    }
                ],
                tools=[_LINE_EVIDENCE_TOOL],
                tool_choice={"type": "tool", "name": "submit_line_evidence"},
                max_tokens=max_tokens,
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
                on_stream_delta=lambda kind, text: reporter.on_stream_delta(kind=kind, text=text),
            )
        except ValueError as exc:
            if not is_malformed_tool_json_error(exc):
                raise
            invalid_responses.append(
                {
                    "attempt": attempt,
                    "phase": "provider_tool_json_decode",
                    "error": str(exc),
                    "response": None,
                }
            )
            if attempt == inspection_attempts:
                raise PageGraphResponseFormatError(
                    "line-evidence tool arguments were malformed after one recovery attempt",
                    attempts=attempt,
                    diagnostics=invalid_responses,
                ) from exc
            continue

        _report_response_content(response, reporter)
        cost_tracker.record(
            response,
            step=step + attempt - 1,
            tile_id="evidence_line_visual",
            page_index=page.page_index,
        )
        reporter.on_token_update(total_tokens=cost_tracker.total_tokens())
        try:
            submission = parse_line_evidence_response(response)
        except ValueError as exc:
            invalid_responses.append(
                {
                    "attempt": attempt,
                    "phase": "response_parse",
                    "error": str(exc),
                    "response": _response_diagnostic(response),
                }
            )
            if attempt == inspection_attempts:
                raise PageGraphResponseFormatError(
                    "line-evidence response was unstructured after one recovery attempt",
                    attempts=attempt,
                    diagnostics=invalid_responses,
                ) from exc
            continue

        assessments: dict[str, VisualCandidateAssessment] = {}
        diagnostics: list[str] = []
        for assessment in submission.assessments:
            if assessment.candidate_ref not in edge_refs:
                diagnostics.append(f"unknown candidate_ref {assessment.candidate_ref!r}")
                continue
            if assessment.candidate_ref in assessments:
                diagnostics.append(f"duplicate assessment for {assessment.candidate_ref!r}")
                continue
            assessments[assessment.candidate_ref] = assessment
        for ref in edge_refs:
            if ref not in assessments:
                diagnostics.append(f"missing visual assessment for {ref!r}")
        return PageLineEvidence(
            page_index=page.page_index,
            assessments=assessments,
            diagnostics=diagnostics,
            format_recovery=invalid_responses,
        )

    raise AssertionError("line-evidence recovery loop exited unexpectedly")


def parse_line_evidence_response(response: Any) -> PageLineEvidenceSubmission:
    texts: list[str] = []
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "tool_use":
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if name == "submit_line_evidence":
                value = (
                    block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
                )
                return PageLineEvidenceSubmission.model_validate(value or {})
        elif block_type == "text":
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if value:
                texts.append(str(value))
    raw = "\n".join(texts).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("line-evidence response did not contain structured JSON")
    return PageLineEvidenceSubmission.model_validate(json.loads(raw[start : end + 1]))


def solve_page_graph(
    *,
    client: LLMClient,
    cost_tracker: CostTracker,
    reporter: ProgressReporter,
    page: PageEvidence,
    rendered_image: Image.Image,
    nodes: list[ReconciledNode],
    all_nodes: list[ReconciledNode],
    topology: TopologyResult,
    pages: list[PageEvidence],
    step: int,
    output_effort: str = "medium",
    max_tokens: int = 16000,
    legend_summary: list[dict[str, Any]] | None = None,
    legend_line_images: list[dict[str, Any]] | None = None,
    visual_evidence: PageLineEvidence | None = None,
    on_attempt: Callable[[], None] | None = None,
    include_visual_context: bool = True,
    process_context: list[dict[str, Any]] | None = None,
    inspection_client: LLMClient | None = None,
    escalation_client: LLMClient | None = None,
) -> PageGraphResult:
    """Reason in bounded batches, then serialize each memo without thinking.

    Thinking and forced tool use are deliberately separate requests. Several
    open-weight reasoning models either reject that combination or spend the
    entire output budget in a reasoning block without emitting tool arguments.
    The serializer receives the provider-returned reasoning instead of solving
    the engineering task again from scratch.
    """

    node_refs, nodes_by_ref = _node_reference_table(page.page_index, nodes, all_nodes)
    edge_refs, edges_by_ref = _edge_reference_table(topology.edges)
    visual_by_ref = visual_evidence.assessments if visual_evidence is not None else {}
    semantic_refs = {
        ref: edge
        for ref, edge in edge_refs.items()
        if _candidate_requires_semantic_reasoning(
            edge,
            node_refs=node_refs,
            visual=visual_by_ref.get(ref),
        )
    }

    # The page relationship solver is intentionally text-only. The preceding
    # line-evidence stage has already reduced the overview/montage images to
    # structured visible facts, which lets DIAGEX_REASONING_MODEL be a local
    # text-only checkpoint independent of DIAGEX_VISION_MODEL.
    _ = rendered_image, legend_line_images, include_visual_context
    semantic_items = list(semantic_refs.items())
    batches = [
        semantic_items[index : index + _PAGE_GRAPH_REASONING_BATCH_SIZE]
        for index in range(0, len(semantic_items), _PAGE_GRAPH_REASONING_BATCH_SIZE)
    ]
    if not batches:
        # Still give an OPC-only or isolated-node page one bounded opportunity
        # to propose an explicitly evidenced relation.
        batches = [[]]

    invalid_responses: list[dict[str, Any]] = []
    reasoning_trace: list[dict[str, Any]] = []
    submissions: list[PageGraphSubmission] = []
    recorded_responses = 0
    reasoning_enabled = _page_graph_reasoning_enabled(client)

    def record_response(response: Any, *, phase: str) -> None:
        nonlocal recorded_responses
        _report_response_content(response, reporter)
        cost_tracker.record(
            response,
            step=step + recorded_responses,
            tile_id=f"evidence_page_graph_{phase}",
            page_index=page.page_index,
        )
        recorded_responses += 1
        reporter.on_token_update(total_tokens=cost_tracker.total_tokens())

    for batch_index, batch in enumerate(batches, start=1):
        batch_refs = [ref for ref, _edge in batch]
        allow_document_relations = batch_index == 1
        payload = {
            "page": page.page_index + 1,
            "process_context": list(process_context or []) if allow_document_relations else [],
            "hypothesis_instruction": "Process context and engineering rules can suggest hypotheses, never prove a drawn connection. Put plausible but unverified connections in hypotheses with context_sources and a concrete inspection question. Use only enumerated node refs. Do not keep or retype a candidate merely because a process rule predicts it.",
            "batch": {"index": batch_index, "count": len(batches)},
            "scope": {
                "candidate_refs": batch_refs,
                "allow_new_relations_and_opc_updates": allow_document_relations,
            },
            "page_size": [page.width, page.height],
            "nodes": [_node_payload(ref, node) for ref, node in node_refs.items()],
            "candidate_edges": [
                _edge_payload(ref, edge, node_refs, visual=visual_by_ref.get(ref))
                for ref, edge in batch
            ],
            "document_sheets": _sheet_table(pages) if allow_document_relations else [],
            "legend_entries": select_graph_legend_context(list(legend_summary or []), [node.label for node in node_refs.values()]),
            "other_page_opcs": (
                [
                    _node_payload(ref, node)
                    for ref, node in nodes_by_ref.items()
                    if node.page_index != page.page_index
                ]
                if allow_document_relations
                else []
            ),
            "visual_diagnostics": list(
                visual_evidence.diagnostics if visual_evidence else []
            )[:24],
        }

        reasoning_memo = ""
        if reasoning_enabled:
            if on_attempt is not None:
                on_attempt()
            reasoning_tokens = min(max_tokens, max(3000, 1200 + len(batch) * 600))
            reasoning_response = client.messages_create(
                system=_REASONING_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            }
                        ],
                    }
                ],
                max_tokens=reasoning_tokens,
                thinking=_thinking_payload(client),
                output_config={"effort": output_effort},
                on_stream_delta=lambda kind, text: reporter.on_stream_delta(
                    kind=kind, text=text
                ),
            )
            record_response(reasoning_response, phase="reasoning")
            reasoning_memo = _page_graph_reasoning_memo(reasoning_response)
            reasoning_trace.append(
                {
                    "batch": batch_index,
                    "candidate_refs": batch_refs,
                    "response": _response_diagnostic(reasoning_response),
                    "memo": reasoning_memo,
                }
            )

        serializer_input = {
            "source_payload": payload,
            "engineering_decision_memo": reasoning_memo,
            "instruction": (
                "The reasoning pass was disabled. Apply the supplied endpoint and visual "
                "evidence conservatively and serialize only supported decisions."
                if not reasoning_enabled
                else "Serialize the memo without repeating or extending its reasoning."
            ),
        }
        serializer_requires_reasoning = model_requires_reasoning(
            getattr(getattr(client, "config", None), "model", None)
        )
        for format_attempt in range(1, 3):
            if on_attempt is not None:
                on_attempt()
            try:
                while True:
                    serializer_reasoning_mode = (
                        "enabled" if serializer_requires_reasoning else "disabled"
                    )
                    try:
                        response = client.messages_create(
                            system=_SERIALIZATION_SYSTEM,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": json.dumps(
                                                serializer_input,
                                                ensure_ascii=False,
                                                separators=(",", ":"),
                                            ),
                                        }
                                    ],
                                }
                            ],
                            tools=[_TOOL],
                            tool_choice={"type": "tool", "name": "submit_page_graph"},
                            max_tokens=min(max_tokens, max(1800, 500 + len(batch) * 220)),
                            thinking=(
                                {"type": "adaptive", "display": "summarized"}
                                if serializer_requires_reasoning
                                else {"type": "disabled"}
                            ),
                            reasoning_mode_override=serializer_reasoning_mode,
                            output_config={"effort": "low"},
                            on_stream_delta=lambda kind, text: reporter.on_stream_delta(
                                kind=kind, text=text
                            ),
                        )
                        break
                    except Exception as exc:
                        if serializer_requires_reasoning or not is_reasoning_required_error(exc):
                            raise
                        # Provider aliases and newly released models may not yet be
                        # present in the capability registry. Retry the same model
                        # once with low-effort reasoning instead of introducing a
                        # second serializer model.
                        serializer_requires_reasoning = True
                        invalid_responses.append(
                            {
                                "batch": batch_index,
                                "attempt": format_attempt,
                                "phase": "reasoning_capability_recovery",
                                "error": str(exc),
                                "response": None,
                            }
                        )
                        if on_attempt is not None:
                            on_attempt()
            except ValueError as exc:
                if not is_malformed_tool_json_error(exc):
                    raise
                invalid_responses.append(
                    {
                        "batch": batch_index,
                        "attempt": format_attempt,
                        "phase": "provider_tool_json_decode",
                        "error": str(exc),
                        "response": None,
                    }
                )
                if format_attempt == 2:
                    raise PageGraphResponseFormatError(
                        "page-graph serializer emitted malformed tool arguments twice",
                        attempts=recorded_responses + format_attempt,
                        diagnostics=invalid_responses,
                    ) from exc
                serializer_input["instruction"] = (
                    "The previous serialization was malformed. Copy the memo into exactly "
                    "one submit_page_graph call using valid JSON and no prose."
                )
                continue

            record_response(response, phase="serialization")
            try:
                submission = parse_page_graph_response(response)
            except ValueError as exc:
                invalid_responses.append(
                    {
                        "batch": batch_index,
                        "attempt": format_attempt,
                        "phase": "response_parse",
                        "error": str(exc),
                        "response": _response_diagnostic(response),
                    }
                )
                if format_attempt == 2:
                    raise PageGraphResponseFormatError(
                        "page-graph serializer was unstructured twice",
                        attempts=recorded_responses,
                        diagnostics=invalid_responses,
                    ) from exc
                serializer_input["instruction"] = (
                    "The previous response was not a tool call. Return exactly one "
                    "submit_page_graph call with valid JSON and no prose."
                )
                continue
            submissions.append(submission)
            break

    submission = _merge_page_graph_submissions(submissions)
    result = validate_page_graph_submission(
        page=page,
        pages=pages,
        submission=submission,
        nodes_by_ref=nodes_by_ref,
        edges_by_ref=edges_by_ref,
        local_node_refs=node_refs,
        visual_evidence=visual_evidence,
    )
    result.format_recovery = invalid_responses
    result.reasoning_trace = reasoning_trace
    for hypothesis in result.hypotheses:
        hypothesis["context_documents"] = [{k: d[k] for k in ("kind", "source", "sha256") if k in d} for d in process_context or []]
    if inspection_client is not None and result.hypotheses:
        _inspect_proposed_routes(
            result=result, page=page, pages=pages, nodes=nodes, rendered_image=rendered_image,
            client=inspection_client, escalation_client=escalation_client, cost_tracker=cost_tracker,
            reporter=reporter, legend_line_images=legend_line_images, on_attempt=on_attempt,
        )
    return result


def _inspect_proposed_routes(*, result, page, pages, nodes, rendered_image, client,
                             escalation_client, cost_tracker, reporter, legend_line_images,
                             on_attempt):
    """Re-query all source strokes for at most four proposed endpoint pairs.

    No artificial connection or relaxed port/crossing tolerance is introduced.
    Reinspection can select a source route omitted by the global spanning tree.
    Both source geometry and a separate visual assessment must pass acceptance.
    """
    local_ids={n.id for n in nodes}
    proposals=[h for h in result.hypotheses if h['from_node'] in local_ids and h['to_node'] in local_ids][:4]
    pairs={frozenset((h['from_node'],h['to_node'])) for h in proposals}
    if not pairs:
        return
    recovered=build_page_topology(page=page,nodes=nodes,raster_image=rendered_image,requested_pairs=pairs)
    recovered.edges=[e for e in recovered.edges if not route_evidence_failures(e,page)]
    # One candidate per pair, deterministic shortest valid source route.
    unique={}
    for edge in sorted(recovered.edges,key=lambda e:(len(e.polyline_global),e.id)):
        unique.setdefault(frozenset((edge.from_node,edge.to_node)),edge)
    recovered.edges=list(unique.values())
    for h in proposals:
        edge=unique.get(frozenset((h['from_node'],h['to_node'])))
        h['targeted_inspection']={'native_paths_queried':len(page.paths),'source_route_found':edge is not None,'attempts':0}
    if not recovered.edges:
        return
    clients=[client]+([escalation_client] if escalation_client is not None else [])
    refs,by_ref=_node_reference_table(page.page_index,nodes,nodes)
    for attempt,active in enumerate(clients,1):
        edge_refs,edges_by_ref=_edge_reference_table(recovered.edges)
        try:
            facts=classify_page_line_evidence(client=active,cost_tracker=cost_tracker,reporter=reporter,
                page=page,rendered_image=rendered_image,nodes=nodes,topology=recovered,
                step=max((s.step for s in cost_tracker.steps),default=0)+1,
                legend_line_images=legend_line_images,on_attempt=on_attempt,inspection_attempts=1)
            checked=validate_page_graph_submission(page=page,pages=pages,
                submission=PageGraphSubmission(),nodes_by_ref=by_ref,edges_by_ref=edges_by_ref,
                local_node_refs=refs,visual_evidence=facts)
        except Exception as exc:
            from diagex.llm.client import is_non_retryable_api_error
            if is_non_retryable_api_error(exc):
                raise
            for h in proposals:
                h['targeted_inspection'].update(attempts=attempt,error=str(exc)[:500])
            continue
        accepted=[e for e in checked.edges if not e.attributes.get('provisional_review_only')]
        for h in proposals:
            h['targeted_inspection'].update(attempts=attempt,visual_evidence=facts.model_dump(mode='json'))
        for edge in accepted:
            pair=frozenset((edge.from_node,edge.to_node))
            edge.attributes['targeted_source_inspection']=True
            if not any(frozenset((e.from_node,e.to_node))==pair and e.line_type==edge.line_type for e in result.edges):
                result.edges.append(edge)
            for h in proposals:
                if frozenset((h['from_node'],h['to_node']))==pair:
                    h.update(status='source_route_recovered',resolved_edge_id=edge.id)
        accepted_ids={e.id for e in accepted}
        recovered.edges=[e for e in recovered.edges if e.id not in accepted_ids]
        if not recovered.edges:
            break


def parse_page_graph_response(response: Any) -> PageGraphSubmission:
    texts: list[str] = []
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "tool_use":
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if name == "submit_page_graph":
                value = (
                    block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
                )
                return PageGraphSubmission.model_validate(value or {})
        elif block_type == "text":
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if value:
                texts.append(str(value))
    raw = "\n".join(texts).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("page-graph response did not contain structured JSON")
    return PageGraphSubmission.model_validate(json.loads(raw[start : end + 1]))


def _page_graph_reasoning_memo(response: Any, *, max_chars: int = 64_000) -> str:
    """Extract provider-visible reasoning/text for the serialization request."""

    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        value: Any = None
        if block_type in {"thinking", "reasoning"}:
            for field in ("thinking", "reasoning", "text"):
                value = block.get(field) if isinstance(block, dict) else getattr(block, field, None)
                if value:
                    break
        elif block_type == "text":
            value = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        elif block_type == "tool_use":
            value = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
            if value is not None:
                value = json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
        if value:
            parts.append(str(value).strip())

    memo = "\n".join(part for part in parts if part).strip()
    if not memo:
        return "No usable decision memo was returned; serialize an empty conservative submission."
    if len(memo) <= max_chars:
        return memo
    return memo[:max_chars].rstrip() + "\n[decision memo truncated by DiagEx]"


def _merge_page_graph_submissions(
    submissions: list[PageGraphSubmission],
) -> PageGraphSubmission:
    """Merge deterministic, non-overlapping batch submissions once per page."""

    decisions: dict[str, CandidateDecision] = {}
    relations: dict[tuple[str, str, str, bool], ProposedRelation] = {}
    opc_updates: dict[str, OpcUpdate] = {}
    uncertainties: dict[tuple[tuple[str, ...], tuple[str, ...], str], PageGraphUncertainty] = {}
    hypotheses = {}
    for submission in submissions:
        for hypothesis in submission.hypotheses:
            hypotheses.setdefault((hypothesis.from_ref, hypothesis.to_ref, hypothesis.line_type), hypothesis)
        for decision in submission.candidate_decisions:
            decisions.setdefault(decision.candidate_ref, decision)
        for relation in submission.new_relations:
            key = (
                relation.from_ref,
                relation.to_ref,
                relation.line_type,
                relation.cross_sheet,
            )
            relations.setdefault(key, relation)
        for update in submission.opc_updates:
            opc_updates.setdefault(update.node_ref, update)
        for uncertainty in submission.uncertainties:
            key = (
                tuple(uncertainty.node_refs),
                tuple(uncertainty.candidate_refs),
                uncertainty.reason,
            )
            uncertainties.setdefault(key, uncertainty)
    return PageGraphSubmission(
        candidate_decisions=list(decisions.values()),
        new_relations=list(relations.values()),
        opc_updates=list(opc_updates.values()),
        uncertainties=list(uncertainties.values()),
        hypotheses=list(hypotheses.values())[:64],
    )


def _response_diagnostic(response: Any) -> dict[str, Any]:
    """Return a JSON-safe response snapshot without request image data."""

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
    return _json_safe(
        {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "content": content,
        }
    )


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


def _provisional_review_edge(
    edge: ReconciledEdge,
    endpoint_nodes: list[ReconciledNode | None],
    *,
    visual: VisualCandidateAssessment | None,
    decision: CandidateDecision | None,
    conflict_type: str,
    reason: str,
    proposed_line_type: str | None = None,
) -> ReconciledEdge:
    """Retain uncertain deterministic topology without treating it as verified.

    Review-only edges remain visible and editable in ``graph.json``.  The DEXPI
    builder deliberately omits them until a reviewer approves or modifies the
    edge, so preserving evidence cannot silently assert unsupported process
    connectivity in the exchange artifact.
    """

    provisional = edge.model_copy(deep=True)
    provisional.attributes.update(
        {
            "provisional_review_only": True,
            "requires_human_review": True,
            "review_conflict_type": conflict_type,
            "review_reason": reason,
            "original_line_type": edge.line_type,
            "page_graph_decision": decision.decision if decision is not None else "missing",
            "page_graph_evidence": decision.evidence if decision is not None else [],
        }
    )
    if proposed_line_type is not None:
        provisional.attributes["proposed_line_type"] = proposed_line_type
    provisional = _score_edge_evidence(
        provisional,
        endpoint_nodes,
        visual=visual,
        decision=decision,
    )
    return provisional


def validate_page_graph_submission(
    *,
    page: PageEvidence,
    pages: list[PageEvidence],
    submission: PageGraphSubmission,
    nodes_by_ref: dict[str, ReconciledNode],
    edges_by_ref: dict[str, ReconciledEdge],
    local_node_refs: dict[str, ReconciledNode],
    visual_evidence: PageLineEvidence | None = None,
) -> PageGraphResult:
    result = PageGraphResult(
        page_index=page.page_index,
        submission=submission.model_dump(mode="json"),
    )
    effective_nodes_by_ref = {ref: node.model_copy(deep=True) for ref, node in nodes_by_ref.items()}
    for hypothesis in submission.hypotheses:
        source, target = nodes_by_ref.get(hypothesis.from_ref), nodes_by_ref.get(hypothesis.to_ref)
        if source is None or target is None or source.id == target.id or page.page_index not in {source.page_index, target.page_index}:
            result.diagnostics.append("hypothesis has invalid or out-of-scope endpoints")
            continue
        result.hypotheses.append({
            "id": "hypothesis-" + _stable_relation_id(source.id, target.id, hypothesis.line_type, source.page_index != target.page_index),
            "page_index": page.page_index, "from_node": source.id, "to_node": target.id,
            "line_type": hypothesis.line_type, "basis": "engineering_hypothesis", "status": "unresolved",
            "rationale": hypothesis.rationale, "question": hypothesis.question,
            "context_sources": hypothesis.context_sources,
        })
    local_nodes_by_id = {node.id: node for node in local_node_refs.values()}
    visual_by_ref = visual_evidence.assessments if visual_evidence is not None else {}
    decisions: dict[str, CandidateDecision] = {}
    for decision in submission.candidate_decisions:
        edge = edges_by_ref.get(decision.candidate_ref)
        if edge is None:
            result.diagnostics.append(f"unknown candidate_ref {decision.candidate_ref!r}")
            continue
        if decision.candidate_ref in decisions:
            result.diagnostics.append(f"duplicate decision for {decision.candidate_ref!r}")
            continue
        decisions[decision.candidate_ref] = decision

    for ref, edge in edges_by_ref.items():
        endpoint_nodes = [
            local_nodes_by_id.get(node_id) for node_id in (edge.from_node, edge.to_node)
        ]
        visual = visual_by_ref.get(ref)
        structural_failures = route_evidence_failures(edge, page)
        explicit_decision = decisions.get(ref)
        if structural_failures and (explicit_decision is None or explicit_decision.decision != "reject"):
            reason = "; ".join(structural_failures)
            result.conflicts.append({
                "type": "unsupported_vector_route", "page_index": page.page_index,
                "edge_id": edge.id, "candidate_ref": ref,
                "node_ids": [edge.from_node, edge.to_node], "status": "unresolved",
                "reason": reason, "source_path_ids": edge.source_evidence_ids,
            })
            result.edges.append(_provisional_review_edge(
                edge, endpoint_nodes, visual=visual, decision=explicit_decision,
                conflict_type="unsupported_vector_route", reason=reason,
                proposed_line_type=explicit_decision.line_type if explicit_decision else None,
            ))
            result.provisional_candidate_ids.append(edge.id)
            continue
        requires_review = any(
            node and node.kind in {"instrument", "opc"} for node in endpoint_nodes
        )
        requires_review = (
            requires_review
            or edge.line_type != "process"
            or str(edge.attributes.get("visual_style") or "unknown")
            in {"dashed", "dotted", "dash_dot"}
        )
        decision = decisions.get(ref)
        edge = _orient_supported_route(edge, visual)
        endpoint_nodes = [local_nodes_by_id.get(node_id) for node_id in (edge.from_node, edge.to_node)]
        if decision is None and visual is not None:
            decision = _automatic_visual_decision(edge, endpoint_nodes, visual)
        if decision is None and not requires_review:
            if visual is not None and (
                visual.route_visible == "no" or visual.endpoint_alignment == "neither"
            ):
                result.conflicts.append(
                    {
                        "type": "visual_topology_disagreement",
                        "page_index": page.page_index,
                        "edge_id": edge.id,
                        "candidate_ref": ref,
                        "node_ids": [node.id for node in endpoint_nodes if node is not None],
                        "status": "unresolved",
                        "reason": "deterministic route and non-thinking visual evidence disagree",
                    }
                )
                result.edges.append(
                    _provisional_review_edge(
                        edge,
                        endpoint_nodes,
                        visual=visual,
                        decision=None,
                        conflict_type="visual_topology_disagreement",
                        reason="deterministic route and non-thinking visual evidence disagree",
                    )
                )
                result.provisional_candidate_ids.append(edge.id)
                continue
            result.edges.append(
                _score_edge_evidence(edge, endpoint_nodes, visual=visual, decision=None)
            )
            result.accepted_candidate_ids.append(edge.id)
            continue
        if decision is None or decision.decision == "uncertain":
            result.conflicts.append(
                {
                    "type": "page_graph_uncertain_candidate",
                    "page_index": page.page_index,
                    "edge_id": edge.id,
                    "candidate_ref": ref,
                    "status": "unresolved",
                    "reason": "page solver did not make a supported relationship decision",
                }
            )
            result.edges.append(
                _provisional_review_edge(
                    edge,
                    endpoint_nodes,
                    visual=visual,
                    decision=decision,
                    conflict_type="page_graph_uncertain_candidate",
                    reason="page solver did not make a supported relationship decision",
                    proposed_line_type=decision.line_type if decision is not None else None,
                )
            )
            result.provisional_candidate_ids.append(edge.id)
            continue
        if decision.decision == "reject":
            result.rejected_candidate_ids.append(edge.id)
            continue
        # Some providers use ``keep`` to mean "keep this candidate" while
        # still returning the engineering classification they selected.  Do
        # not silently discard that classification when deterministic
        # topology could only initialise the edge as ``other``.  Implicit
        # retyping remains conservative: it needs independent visual and
        # endpoint-semantic support before it can enter the graph.
        line_type = decision.line_type or edge.line_type
        implicit_retype = (
            decision.decision == "keep"
            and decision.line_type is not None
            and decision.line_type != edge.line_type
        )
        if implicit_retype and not _implicit_retype_supported(
            edge,
            endpoint_nodes,
            proposed_line_type=line_type,
            visual=visual,
        ):
            reason = (
                "page solver supplied a different line type with a keep decision, "
                "but independent visual and endpoint evidence did not corroborate it"
            )
            result.conflicts.append(
                {
                    "type": "endpoint_role_uncertain",
                    "page_index": page.page_index,
                    "edge_id": edge.id,
                    "candidate_ref": ref,
                    "node_ids": [node.id for node in endpoint_nodes if node is not None],
                    "proposed_line_type": line_type,
                    "status": "unresolved",
                    "endpoint_roles": [_endpoint_role(node) for node in endpoint_nodes],
                    "reason": reason,
                }
            )
            result.edges.append(
                _provisional_review_edge(
                    edge,
                    endpoint_nodes,
                    visual=visual,
                    decision=decision,
                    conflict_type="endpoint_role_uncertain",
                    reason=reason,
                    proposed_line_type=line_type,
                )
            )
            result.provisional_candidate_ids.append(edge.id)
            continue
        compatibility = _endpoint_compatibility(line_type, endpoint_nodes)
        if compatibility != "valid":
            conflict_type = (
                "unsupported_endpoint_combination"
                if compatibility == "invalid"
                else "endpoint_role_uncertain"
            )
            result.conflicts.append(
                {
                    "type": conflict_type,
                    "page_index": page.page_index,
                    "edge_id": edge.id,
                    "candidate_ref": ref,
                    "node_ids": [node.id for node in endpoint_nodes if node is not None],
                    "proposed_line_type": line_type,
                    "status": "unresolved",
                    "endpoint_roles": [_endpoint_role(node) for node in endpoint_nodes],
                    "reason": (
                        "line type is incompatible with the identified endpoint roles"
                        if compatibility == "invalid"
                        else "endpoint roles do not deterministically prove this line type; edge withheld"
                    ),
                }
            )
            result.edges.append(
                _provisional_review_edge(
                    edge,
                    endpoint_nodes,
                    visual=visual,
                    decision=decision,
                    conflict_type=conflict_type,
                    reason=(
                        "line type is incompatible with the identified endpoint roles"
                        if compatibility == "invalid"
                        else "endpoint roles do not deterministically prove this line type"
                    ),
                    proposed_line_type=line_type,
                )
            )
            result.provisional_candidate_ids.append(edge.id)
            continue
        if visual is not None and (
            visual.route_visible == "no" or visual.endpoint_alignment == "neither"
        ):
            result.conflicts.append(
                {
                    "type": "visual_topology_disagreement",
                    "page_index": page.page_index,
                    "edge_id": edge.id,
                    "candidate_ref": ref,
                    "node_ids": [node.id for node in endpoint_nodes if node is not None],
                    "status": "unresolved",
                    "reason": "deterministic route and non-thinking visual evidence disagree",
                }
            )
            result.edges.append(
                _provisional_review_edge(
                    edge,
                    endpoint_nodes,
                    visual=visual,
                    decision=decision,
                    conflict_type="visual_topology_disagreement",
                    reason="deterministic route and non-thinking visual evidence disagree",
                    proposed_line_type=line_type,
                )
            )
            result.provisional_candidate_ids.append(edge.id)
            continue
        update_values: dict[str, Any] = {"line_type": line_type}
        updated = edge.model_copy(deep=True, update=update_values)
        updated.attributes.update(
            {
                "page_graph_decision": decision.decision,
                "page_graph_evidence": decision.evidence,
            }
        )
        if implicit_retype:
            updated.attributes["line_type_source"] = "page_graph_keep_classification"
        updated = _score_edge_evidence(
            updated,
            endpoint_nodes,
            visual=visual,
            decision=decision,
        )
        result.edges.append(updated)
        result.accepted_candidate_ids.append(edge.id)

    for update in submission.opc_updates:
        node = local_node_refs.get(update.node_ref)
        if node is None or node.kind != "opc":
            result.diagnostics.append(f"OPC update targets unknown/non-OPC ref {update.node_ref!r}")
            continue
        if not update.evidence:
            result.diagnostics.append(f"OPC update {update.node_ref!r} has no printed evidence")
            continue
        if not _opc_update_supported(update, node, page):
            result.conflicts.append(
                {
                    "type": "unsupported_opc_update",
                    "page_index": page.page_index,
                    "node_id": node.id,
                    "status": "unresolved",
                    "reason": "proposed OPC fields were not corroborated by nearby native or object text",
                }
            )
            continue
        values = {
            "direction": update.direction,
            "service": update.service,
            "drawing_ref": update.drawing_ref,
            "line_id": update.line_id,
            "raw_printed_text": update.raw_printed_text,
            "page_graph_evidence": update.evidence,
        }
        accepted_update = {
            key: value for key, value in values.items() if value not in (None, "", [])
        }
        result.opc_updates[node.id] = accepted_update
        effective_nodes_by_ref[update.node_ref].attributes.update(accepted_update)

    seen_pairs = {
        _edge_key(edge.from_node, edge.to_node, edge.line_type, edge.cross_sheet)
        for edge in result.edges
    }
    for relation in submission.new_relations:
        source = effective_nodes_by_ref.get(relation.from_ref)
        target = effective_nodes_by_ref.get(relation.to_ref)
        if source is None or target is None:
            result.diagnostics.append(
                f"relation uses unknown node ref {relation.from_ref!r}->{relation.to_ref!r}"
            )
            continue
        if source.id == target.id:
            result.diagnostics.append(
                f"relation {relation.from_ref}->{relation.to_ref} is a self-loop"
            )
            continue
        if source.page_index != page.page_index and target.page_index != page.page_index:
            result.diagnostics.append("relation does not involve the page being solved")
            continue
        cross_sheet = source.page_index != target.page_index
        if cross_sheet != relation.cross_sheet:
            result.diagnostics.append("relation cross_sheet flag disagrees with node pages")
            continue
        if cross_sheet and not _cross_sheet_supported(source, target, relation, pages):
            result.conflicts.append(
                {
                    "type": "unsupported_cross_sheet_proposal",
                    "page_index": page.page_index,
                    "node_ids": [source.id, target.id],
                    "status": "unresolved",
                    "reason": "proposal lacks compatible direction and explicit printed continuation evidence",
                }
            )
            continue
        if not cross_sheet and relation.line_type == "process":
            result.diagnostics.append("new process relations require deterministic topology")
            result.hypotheses.append({
                "id": "hypothesis-" + _stable_relation_id(source.id, target.id, relation.line_type, False),
                "page_index": page.page_index, "from_node": source.id, "to_node": target.id,
                "line_type": relation.line_type, "basis": "unverified_visual_proposal", "status": "unresolved",
                "rationale": "; ".join(relation.evidence),
                "question": "Can a continuous source route be traced between these endpoints?",
            })
            continue
        compatibility = _endpoint_compatibility(relation.line_type, [source, target])
        if compatibility != "valid":
            result.conflicts.append(
                {
                    "type": (
                        "unsupported_endpoint_combination"
                        if compatibility == "invalid"
                        else "endpoint_role_uncertain"
                    ),
                    "page_index": page.page_index,
                    "node_ids": [source.id, target.id],
                    "proposed_line_type": relation.line_type,
                    "status": "unresolved",
                    "endpoint_roles": [_endpoint_role(source), _endpoint_role(target)],
                    "reason": (
                        "new relation is incompatible with the identified endpoint roles"
                        if compatibility == "invalid"
                        else "endpoint roles do not deterministically prove this new relation; edge withheld"
                    ),
                }
            )
            continue
        key = _edge_key(source.id, target.id, relation.line_type, cross_sheet)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        edge_id = _stable_relation_id(source.id, target.id, relation.line_type, cross_sheet)
        proposed_edge = ReconciledEdge(
            id=edge_id,
            from_node=source.id,
            to_node=target.id,
            line_type=relation.line_type,
            cross_sheet=cross_sheet,
            confidence=relation.confidence,
            source_evidence_ids=sorted(
                set(source.source_evidence_ids + target.source_evidence_ids)
            ),
            system_confidence={"high": 0.82, "medium": 0.64, "low": 0.42}[relation.confidence],
            system_confidence_level=relation.confidence,
            attributes={
                "topology_source": "page_graph_reasoning",
                "page_graph_evidence": relation.evidence,
                "geometry_missing": not cross_sheet,
            },
        )
        proposed_edge = _score_edge_evidence(
            proposed_edge,
            [source, target],
            visual=None,
            decision=None,
        )
        if not cross_sheet:
            proposed_edge = _provisional_review_edge(
                proposed_edge, [source, target], visual=None, decision=None,
                conflict_type="relationship_geometry_missing",
                reason="model-supported relationship has no deterministic source polyline",
            )
            result.provisional_candidate_ids.append(proposed_edge.id)
        result.edges.append(proposed_edge)
        if not cross_sheet:
            result.conflicts.append(
                {
                    "type": "relationship_geometry_missing",
                    "page_index": page.page_index,
                    "edge_id": edge_id,
                    "node_ids": [source.id, target.id],
                    "status": "unresolved",
                    "reason": "model-supported relationship has no deterministic source polyline",
                }
            )

    for uncertainty in submission.uncertainties:
        unknown_nodes = [ref for ref in uncertainty.node_refs if ref not in nodes_by_ref]
        unknown_edges = [ref for ref in uncertainty.candidate_refs if ref not in edges_by_ref]
        if unknown_nodes or unknown_edges:
            result.diagnostics.append("uncertainty contains unknown references")
            continue
        result.conflicts.append(
            {
                "type": "page_graph_uncertainty",
                "page_index": page.page_index,
                "node_ids": [nodes_by_ref[ref].id for ref in uncertainty.node_refs],
                "edge_ids": [edges_by_ref[ref].id for ref in uncertainty.candidate_refs],
                "status": "unresolved",
                "reason": uncertainty.reason,
            }
        )
    return result


def _orient_supported_route(
    edge: ReconciledEdge, visual: VisualCandidateAssessment | None,
) -> ReconciledEdge:
    """Endpoint ordering is not flow evidence. Native arrows take precedence."""
    updated = edge.model_copy(deep=True)
    if updated.attributes.get("direction_source") == "native_vector_arrow":
        return updated
    updated.attributes["flow_direction"] = "unknown"
    updated.attributes["direction_source"] = "unknown"
    if visual is None or visual.confidence != "high" or visual.arrow_direction not in {"forward", "reverse"}:
        return updated
    updated.attributes["flow_direction"] = "forward"
    updated.attributes["direction_source"] = "visual_arrow"
    if visual.arrow_direction == "reverse":
        updated.from_node, updated.to_node = updated.to_node, updated.from_node
        updated.polyline_global.reverse()
        if "endpoint_labels" in updated.attributes:
            updated.attributes["endpoint_labels"].reverse()
        proof = updated.attributes.get("route_evidence")
        if proof:
            proof["ports"].reverse()
    return updated


def _candidate_requires_semantic_reasoning(
    edge: ReconciledEdge,
    *,
    node_refs: dict[str, ReconciledNode],
    visual: VisualCandidateAssessment | None,
) -> bool:
    nodes_by_id = {node.id: node for node in node_refs.values()}
    endpoints = [nodes_by_id.get(edge.from_node), nodes_by_id.get(edge.to_node)]
    if visual is not None:
        automatic = _automatic_visual_decision(edge, endpoints, visual)
        if automatic is not None:
            return False
    return (
        any(node is None or node.kind in {"instrument", "opc"} for node in endpoints)
        or edge.line_type != "process"
        or str(edge.attributes.get("visual_style") or "unknown") in {"dashed", "dotted", "dash_dot"}
    )


def _automatic_visual_decision(
    edge: ReconciledEdge,
    endpoints: list[ReconciledNode | None],
    visual: VisualCandidateAssessment,
) -> CandidateDecision | None:
    """Return only decisions supported by strong visual and structural evidence."""

    if visual.route_visible != "yes" or visual.endpoint_alignment != "both":
        return None
    if visual.confidence != "high":
        return None
    proposed_type = visual.legend_class
    if proposed_type is None:
        if (
            visual.observed_style == "solid"
            and edge.line_type == "process"
            and all(node is not None and node.kind == "equipment" for node in endpoints)
        ):
            proposed_type = "process"
        else:
            return None
    if _endpoint_compatibility(proposed_type, endpoints) != "valid":
        return None
    return CandidateDecision(
        candidate_ref="",  # filled below; only the decision fields are consumed
        decision="keep" if proposed_type == edge.line_type else "retype",
        line_type=proposed_type,
        confidence="high",
        evidence=[*visual.evidence, "automatic visual/structural decision"][:8],
    )


def _endpoint_role(node: ReconciledNode | None) -> str:
    if node is None:
        return "missing"
    if node.kind == "opc":
        return "opc"
    text = " ".join(
        str(value or "")
        for value in (
            node.label,
            node.source_quote,
            node.attributes.get("equipment_class"),
            node.attributes.get("valve_type"),
            node.attributes.get("instrument_function"),
            node.attributes.get("structural_description"),
        )
    ).casefold()
    instrument_function = str(node.attributes.get("instrument_function") or "")
    if instrument_function == "valve_actuator":
        return "actuator_valve"
    valve_type = str(node.attributes.get("valve_type") or "")
    if valve_type in {"control", "control_valve"} or node.attributes.get(
        "actuation_type"
    ) or re.search(
        r"(?:control\s*valve|actuat|operated\s*valve|\b[fpalt]cv[- ]?\d)", text
    ):
        return "actuator_valve"
    if node.kind == "instrument":
        return "instrument"
    if re.search(r"(?:\bmotor\b|电机|motor[- ]?driven|\bm[- ]?\d{2,})", text):
        return "motor"
    if re.search(r"(?:power\s*(?:source|supply)|电源|switchgear|mcc)", text):
        return "power_source"
    return "equipment"


def _node_loop_number(node: ReconciledNode | None) -> str | None:
    if node is None:
        return None
    for key in ("loop_number", "tag_number"):
        value = re.sub(r"[^A-Z0-9]", "", str(node.attributes.get(key) or "").upper())
        if value:
            return value
    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        str(node.attributes.get("canonical_tag") or node.label or "").upper(),
    )
    match = re.search(r"(\d{2,6}[A-Z]?)$", compact)
    return match.group(1) if match is not None else None


def _node_instrument_function(node: ReconciledNode | None) -> str | None:
    if node is None or node.kind != "instrument":
        return None
    explicit = str(node.attributes.get("instrument_function") or "").strip().lower()
    if explicit and explicit != "unclassified_instrument":
        return explicit
    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        str(node.attributes.get("canonical_tag") or node.label or "").upper(),
    )
    match = re.match(r"([A-Z]{1,6})\d", compact)
    if match is None:
        return None
    functions = match.group(1)[1:]
    for letter, function in (
        ("C", "controller"),
        ("T", "transmitter"),
        ("I", "indicator"),
        ("R", "recorder"),
        ("E", "element"),
        ("S", "switch"),
        ("A", "alarm"),
        ("V", "valve_actuator"),
    ):
        if letter in functions:
            return function
    return None


def _node_measured_variable(node: ReconciledNode | None) -> str | None:
    if node is None or node.kind != "instrument":
        return None
    explicit = str(node.attributes.get("measured_variable") or "").strip().lower()
    if explicit and explicit != "other":
        return explicit
    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        str(node.attributes.get("canonical_tag") or node.label or "").upper(),
    )
    return {
        "A": "analysis",
        "F": "flow",
        "L": "level",
        "P": "pressure",
        "T": "temperature",
    }.get(compact[:1])


_INSTRUMENT_SIGNAL_FLOW: dict[str, set[str]] = {
    "element": {"transmitter", "indicator", "controller", "recorder"},
    "transmitter": {"indicator", "controller", "recorder", "alarm"},
    "switch": {"controller", "alarm"},
    "controller": {"indicator", "recorder"},
}


def _instrument_pair_evidence(
    endpoints: list[ReconciledNode | None],
) -> dict[str, Any] | None:
    if len(endpoints) != 2 or any(node is None or node.kind != "instrument" for node in endpoints):
        return None
    left, right = endpoints
    left_loop, right_loop = _node_loop_number(left), _node_loop_number(right)
    left_function = _node_instrument_function(left)
    right_function = _node_instrument_function(right)
    left_variable = _node_measured_variable(left)
    right_variable = _node_measured_variable(right)
    same_loop = bool(left_loop and right_loop and left_loop == right_loop)
    variable_compatible = not (
        left_variable and right_variable and left_variable != right_variable
    )
    direction = None
    if right_function in _INSTRUMENT_SIGNAL_FLOW.get(left_function or "", set()):
        direction = "forward"
    elif left_function in _INSTRUMENT_SIGNAL_FLOW.get(right_function or "", set()):
        direction = "reverse"
    return {
        "same_loop": same_loop,
        "loop_number": left_loop if same_loop else None,
        "functions": [left_function, right_function],
        "measured_variables": [left_variable, right_variable],
        "functional_direction": direction,
        "compatible": bool(same_loop and variable_compatible and direction),
    }


def _implicit_retype_supported(
    edge: ReconciledEdge,
    endpoints: list[ReconciledNode | None],
    *,
    proposed_line_type: LineType,
    visual: VisualCandidateAssessment | None,
) -> bool:
    """Require independent evidence for ``keep`` plus a changed line type."""

    if edge.line_type not in {"other", proposed_line_type}:
        return False
    if _endpoint_compatibility(proposed_line_type, endpoints) != "valid":
        return False
    visual_route = bool(
        visual is not None
        and visual.route_visible == "yes"
        and visual.endpoint_alignment == "both"
    )
    legend_match = bool(visual_route and visual.legend_class == proposed_line_type)
    if proposed_line_type in {
        "signal_electric",
        "signal_pneumatic",
        "instrument_capillary",
    }:
        pair = _instrument_pair_evidence(endpoints)
        return bool(legend_match and pair and pair["compatible"])
    if proposed_line_type == "process":
        return bool(
            visual_route
            and visual is not None
            and (visual.legend_class == "process" or visual.observed_style == "solid")
        )
    return legend_match


def _endpoint_compatibility(
    line_type: LineType | None,
    endpoints: list[ReconciledNode | None],
) -> Literal["valid", "uncertain", "invalid"]:
    if len(endpoints) != 2 or any(node is None for node in endpoints):
        return "invalid"
    roles = [_endpoint_role(node) for node in endpoints]
    role_set = set(roles)
    if line_type == "process":
        if roles == ["instrument", "instrument"]:
            return "invalid"
        if "instrument" in role_set:
            return "uncertain"
        return "valid"
    if line_type == "signal_electric":
        if roles == ["instrument", "instrument"]:
            return "valid"
        if "instrument" in role_set and "actuator_valve" in role_set:
            return "valid"
        return "invalid"
    if line_type == "signal_pneumatic":
        if "instrument" in role_set and "actuator_valve" in role_set:
            return "valid"
        if roles == ["instrument", "instrument"]:
            return "valid"
        return "invalid"
    if line_type == "instrument_capillary":
        if "instrument" not in role_set:
            return "invalid"
        return "valid" if role_set & {"equipment", "actuator_valve"} else "uncertain"
    if line_type == "electrical_power":
        if "motor" in role_set and role_set & {"equipment", "power_source"}:
            return "valid"
        return "invalid"
    return "uncertain"


def _score_edge_evidence(
    edge: ReconciledEdge,
    endpoints: list[ReconciledNode | None],
    *,
    visual: VisualCandidateAssessment | None,
    decision: CandidateDecision | None,
) -> ReconciledEdge:
    """Calculate system confidence from independent evidence contributions."""

    score = 0.50
    contributions: dict[str, float] = {}

    def add(name: str, value: float) -> None:
        nonlocal score
        score += value
        contributions[name] = round(value, 3)

    if edge.attributes.get("topology_source") == "deterministic_vector":
        add("vector_topology", 0.08)
    if len(set(edge.polyline_global)) >= 2:
        add("recovered_geometry", 0.06)
    else:
        add("missing_geometry", -0.14)
    compatibility = _endpoint_compatibility(edge.line_type, endpoints)
    add(
        "endpoint_compatibility",
        0.10 if compatibility == "valid" else -0.08 if compatibility == "uncertain" else -0.35,
    )
    instrument_pair = _instrument_pair_evidence(endpoints)
    if edge.line_type in _SIGNAL_TYPES and instrument_pair and instrument_pair["compatible"]:
        add("same_loop_instrument_semantics", 0.08)
    if visual is not None:
        add(
            "route_visibility",
            0.12
            if visual.route_visible == "yes"
            else -0.20
            if visual.route_visible == "no"
            else -0.04,
        )
        add(
            "endpoint_alignment",
            0.12
            if visual.endpoint_alignment == "both"
            else 0.02
            if visual.endpoint_alignment == "one"
            else -0.22
            if visual.endpoint_alignment == "neither"
            else -0.04,
        )
        deterministic_style = str(edge.attributes.get("visual_style") or "unknown")
        if visual.observed_style != "unknown" and deterministic_style != "unknown":
            add(
                "style_agreement",
                0.08 if visual.observed_style == deterministic_style else -0.10,
            )
        if visual.legend_class is not None:
            add("legend_match", 0.08 if visual.legend_class == edge.line_type else -0.08)
    if decision is not None:
        add(
            "semantic_decision",
            0.06
            if decision.confidence == "high"
            else 0.0
            if decision.confidence == "medium"
            else -0.08,
        )
    if edge.attributes.get("provisional_review_only"):
        contributions["structural_or_semantic_gate"] = round(min(0., .35-score), 3)
        score = min(score, .35)
    score = round(max(0.05, min(0.98, score)), 3)
    updated = edge.model_copy(deep=True)
    if edge.attributes.get("provisional_review_only"):
        updated.confidence = "low"
    updated.system_confidence = score
    updated.system_confidence_level = (
        "high" if score >= 0.80 else "medium" if score >= 0.55 else "low"
    )
    updated.attributes["system_confidence_evidence"] = {
        "score": score,
        "contributions": contributions,
        "endpoint_roles": [_endpoint_role(node) for node in endpoints],
        "instrument_pair": instrument_pair,
        "visual_assessment": visual.model_dump(mode="json") if visual is not None else None,
    }
    return updated


def _node_reference_table(
    page_index: int,
    local_nodes: list[ReconciledNode],
    all_nodes: list[ReconciledNode],
) -> tuple[dict[str, ReconciledNode], dict[str, ReconciledNode]]:
    ordered = sorted(
        local_nodes, key=lambda node: (node.bbox_global.y, node.bbox_global.x, node.id)
    )
    local = {f"N{index:03d}": node for index, node in enumerate(ordered, start=1)}
    combined = dict(local)
    remote_opcs = sorted(
        (node for node in all_nodes if node.kind == "opc" and node.page_index != page_index),
        key=lambda node: (node.page_index, node.bbox_global.y, node.bbox_global.x, node.id),
    )
    per_page: dict[int, int] = {}
    for node in remote_opcs:
        per_page[node.page_index] = per_page.get(node.page_index, 0) + 1
        combined[f"P{node.page_index + 1:03d}N{per_page[node.page_index]:03d}"] = node
    return local, combined


def _edge_reference_table(
    edges: list[ReconciledEdge],
) -> tuple[dict[str, ReconciledEdge], dict[str, ReconciledEdge]]:
    values = {
        f"E{index:03d}": edge
        for index, edge in enumerate(sorted(edges, key=lambda edge: edge.id), start=1)
    }
    return values, values


def _node_payload(ref: str, node: ReconciledNode) -> dict[str, Any]:
    return {
        "node_ref": ref,
        "page": node.page_index + 1,
        "kind": node.kind,
        "label": node.label,
        "printed_text": node.source_quote,
        "bbox": node.bbox_global.model_dump(),
        "attributes": {
            key: node.attributes.get(key)
            for key in (
                "instrument_function",
                "measured_variable",
                "loop_number",
                "valve_type",
                "direction",
                "service",
                "drawing_ref",
                "line_id",
            )
            if node.attributes.get(key) not in (None, "")
        },
    }


def _edge_payload(
    ref: str,
    edge: ReconciledEdge,
    node_refs: dict[str, ReconciledNode],
    *,
    visual: VisualCandidateAssessment | None = None,
) -> dict[str, Any]:
    by_id = {node.id: key for key, node in node_refs.items()}
    payload = {
        "candidate_ref": ref,
        "from_ref": by_id.get(edge.from_node),
        "to_ref": by_id.get(edge.to_node),
        "current_line_type": edge.line_type,
        "visual_style": edge.attributes.get("visual_style", "unknown"),
        "visual_style_confidence": edge.attributes.get("visual_style_confidence"),
        "style_evidence_ids": edge.attributes.get("style_evidence_ids", []),
        "polyline": edge.polyline_global[:: max(1, len(edge.polyline_global) // 12)],
        "source": edge.attributes.get("topology_source"),
    }
    if visual is not None:
        payload["visual_assessment"] = visual.model_dump(mode="json")
    return payload


def _sheet_table(pages: list[PageEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        references: list[str] = []
        for span in page.text_spans:
            if span.bbox.x < page.width * 0.55 or span.bbox.y < page.height * 0.75:
                continue
            references.extend(
                re.sub(r"\s+", "", match.group(0)).upper()
                for match in _DRAWING_REF_RE.finditer(span.text)
            )
        rows.append(
            {
                "page": page.page_index + 1,
                "role": page.role,
                "drawing_refs": sorted(set(references))[:8],
            }
        )
    return rows


def _annotated_overview(
    image: Image.Image,
    node_refs: dict[str, ReconciledNode],
    edge_refs: dict[str, ReconciledEdge],
    *,
    max_dim: int = 2000,
) -> Image.Image:
    scale = min(1.0, max_dim / max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    width = max(2, round(max(canvas.size) / 600))
    # Do not paint over the source route: preserving the original dash pattern
    # is more important than tracing it in a synthetic colour. A small midpoint
    # marker and label identify each candidate without hiding its appearance.
    for ref, edge in edge_refs.items():
        points = [(round(x * scale), round(y * scale)) for x, y in edge.polyline_global]
        if len(points) >= 2:
            mid = points[len(points) // 2]
            radius = max(2, width)
            draw.ellipse(
                (mid[0] - radius, mid[1] - radius, mid[0] + radius, mid[1] + radius),
                outline=(90, 30, 160),
                width=max(1, width // 2),
            )
            _draw_label(draw, (mid[0] + radius + 2, mid[1] - 14), ref, fill=(90, 30, 160))
    for ref, node in node_refs.items():
        box = node.bbox_global
        rect = tuple(round(value * scale) for value in (box.x, box.y, box.x2, box.y2))
        color = (
            (0, 105, 210)
            if node.kind == "instrument"
            else (0, 145, 90)
            if node.kind == "opc"
            else (220, 110, 0)
        )
        draw.rectangle(rect, outline=color, width=width)
        _draw_label(draw, (rect[0], max(0, rect[1] - 13)), ref, fill=color)
    return canvas


def _relationship_montage(
    image: Image.Image,
    node_refs: dict[str, ReconciledNode],
    edge_refs: dict[str, ReconciledEdge],
    *,
    limit: int = 18,
) -> Image.Image | None:
    by_id = {node.id: (ref, node) for ref, node in node_refs.items()}
    targets: list[tuple[int, str, BBox, str]] = []
    for ref, node in node_refs.items():
        if node.kind in {"instrument", "opc"}:
            priority = 2 if node.kind == "opc" else 3
            targets.append((priority, ref, node.bbox_global, "node"))
    for edge_ref, edge in edge_refs.items():
        endpoints = [by_id.get(edge.from_node), by_id.get(edge.to_node)]
        visual_style = str(edge.attributes.get("visual_style") or "unknown")
        touches_instrument = any(
            item and item[1].kind in {"instrument", "opc"} for item in endpoints
        )
        if visual_style not in {"dashed", "dotted", "dash_dot"} and not touches_instrument:
            continue
        points = edge.polyline_global
        if points:
            xs, ys = zip(*points, strict=False)
            box = BBox(
                x=min(xs), y=min(ys), w=max(1, max(xs) - min(xs)), h=max(1, max(ys) - min(ys))
            )
            priority = 0 if visual_style in {"dashed", "dotted", "dash_dot"} else 1
            targets.append((priority, edge_ref, box, "edge"))
    if not targets:
        return None
    cells: list[Image.Image] = []
    seen: set[tuple[int, int, int, int]] = set()
    for _, label, box, target_kind in sorted(
        targets,
        key=lambda row: (row[0], row[2].y, row[2].x, row[1]),
    ):
        padding = max(80, min(220, int(min(image.size) * 0.025))) if target_kind == "edge" else 180
        bounds = (
            max(0, box.x - padding),
            max(0, box.y - padding),
            min(image.width, box.x2 + padding),
            min(image.height, box.y2 + padding),
        )
        if bounds in seen or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            continue
        seen.add(bounds)
        crop = image.crop(bounds).convert("RGB")
        crop.thumbnail((600, 420), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (600, 450), "white")
        cell.paste(crop, ((600 - crop.width) // 2, 28))
        ImageDraw.Draw(cell).text((8, 7), label, fill=(20, 20, 20))
        cells.append(cell)
        if len(cells) >= limit:
            break
    columns = 3
    rows = (len(cells) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 600, rows * 450), "white")
    for index, cell in enumerate(cells):
        montage.paste(cell, ((index % columns) * 600, (index // columns) * 450))
    return montage


def _legend_line_montage(
    entries: list[dict[str, Any]],
    *,
    limit: int = 16,
) -> Image.Image | None:
    """Build one compact, labelled sheet of project-specific line examples.

    Legend crops are intentionally supplied only to the page relationship
    solver. Sending them to every object-perception crop would repeat the same
    image many times without helping object detection.
    """
    cells: list[Image.Image] = []
    ordered_entries = [
        entry
        for _, entry in sorted(
            enumerate(entries),
            key=lambda item: (_legend_line_priority(item[1]), item[0]),
        )
    ]
    for entry in ordered_entries:
        encoded = entry.get("image_b64")
        if not encoded:
            continue
        try:
            raw = base64.b64decode(str(encoded), validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                sample = source.convert("RGB")
        except Exception:  # noqa: BLE001 - ignore a malformed optional legend crop
            continue
        sample.thumbnail((520, 124), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (560, 160), "white")
        draw = ImageDraw.Draw(cell)
        label = " ".join(str(entry.get("label") or "line example").split())
        description = " ".join(str(entry.get("description") or "").split())
        caption = label if not description else f"{label} — {description}"
        draw.text((8, 7), caption[:90], fill=(20, 20, 20))
        cell.paste(sample, ((560 - sample.width) // 2, 30 + (124 - sample.height) // 2))
        cells.append(cell)
        if len(cells) >= limit:
            break
    if not cells:
        return None
    columns = 2
    rows = (len(cells) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 560, rows * 160), "white")
    for index, cell in enumerate(cells):
        montage.paste(cell, ((index % columns) * 560, (index // columns) * 160))
    return montage


def _legend_line_priority(entry: dict[str, Any]) -> int:
    text = " ".join(str(entry.get(key) or "").casefold() for key in ("label", "description"))
    if any(term in text for term in _LINE_LEGEND_SIGNAL_TERMS):
        return 0
    if any(term in text for term in _LINE_LEGEND_PROCESS_TERMS):
        return 1
    if any(term in text for term in _LINE_LEGEND_TOPOLOGY_TERMS):
        return 2
    return 3


def _draw_label(
    draw: ImageDraw.ImageDraw, point: tuple[int, int], text: str, *, fill: tuple[int, int, int]
) -> None:
    x, y = point
    box = draw.textbbox((x, y), text)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill="white")
    draw.text((x, y), text, fill=fill)


def _cross_sheet_supported(
    source: ReconciledNode,
    target: ReconciledNode,
    relation: ProposedRelation,
    pages: list[PageEvidence],
) -> bool:
    if source.kind != "opc" or target.kind != "opc":
        return False
    directions = {
        str(source.attributes.get("direction") or "").casefold(),
        str(target.attributes.get("direction") or "").casefold(),
    }
    if directions != {"in", "out"}:
        return False
    if not relation.evidence:
        return False
    page_refs = {row["page"]: set(row["drawing_refs"]) for row in _sheet_table(pages)}
    source_values = _visible_refs(source)
    target_values = _visible_refs(target)
    reciprocal = bool(
        source_values & page_refs.get(target.page_index + 1, set())
        and target_values & page_refs.get(source.page_index + 1, set())
    )
    same_line = bool(
        source.attributes.get("line_id")
        and source.attributes.get("line_id") == target.attributes.get("line_id")
    )
    same_service = bool(
        source.attributes.get("service")
        and str(source.attributes.get("service")).casefold()
        == str(target.attributes.get("service") or "").casefold()
    )
    return reciprocal or same_line or same_service


def _opc_update_supported(
    update: OpcUpdate,
    node: ReconciledNode,
    page: PageEvidence,
) -> bool:
    padding = max(180, int(max(node.bbox_global.w, node.bbox_global.h) * 5))
    box = node.bbox_global
    nearby = [
        span.text
        for span in page.text_spans
        if span.bbox.x < box.x2 + padding
        and span.bbox.x2 > box.x - padding
        and span.bbox.y < box.y2 + padding
        and span.bbox.y2 > box.y - padding
    ]
    source_text = " ".join([node.label, node.source_quote or "", *nearby])
    source_key = _evidence_text_key(source_text)
    proposed = [
        update.service,
        update.drawing_ref,
        update.line_id,
        update.raw_printed_text,
    ]
    proposed_keys = [_evidence_text_key(value) for value in proposed if value]
    return any(key and key in source_key for key in proposed_keys)


def _evidence_text_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").casefold())


def _visible_refs(node: ReconciledNode) -> set[str]:
    values = (node.label, node.source_quote, node.attributes.get("drawing_ref"))
    return {
        re.sub(r"\s+", "", match.group(0)).upper()
        for value in values
        for match in _DRAWING_REF_RE.finditer(str(value or ""))
    }


def _edge_key(
    left: str, right: str, line_type: LineType | None, cross_sheet: bool
) -> tuple[Any, ...]:
    return (*sorted((left, right)), line_type or "process", cross_sheet)


def _stable_relation_id(left: str, right: str, line_type: LineType, cross_sheet: bool) -> str:
    payload = repr((*sorted((left, right)), line_type, cross_sheet)).encode()
    return "e-pg-" + hashlib.sha256(payload).hexdigest()[:16]


def _thinking_payload(client: LLMClient) -> dict[str, Any]:
    mode = getattr(getattr(client, "config", None), "reasoning_mode", "enabled")
    if mode == "disabled":
        return {"type": "disabled"}
    return {"type": "adaptive", "display": "summarized"}


def _page_graph_reasoning_enabled(client: LLMClient) -> bool:
    config = getattr(client, "config", None)
    return getattr(config, "reasoning_mode", "enabled") != "disabled"


def _coerce_string_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        text = " ".join(value.split()).strip()
        return [text] if text else []
    return value


def _report_response_content(response: Any, reporter: ProgressReporter) -> None:
    for block in getattr(response, "content", None) or []:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type == "tool_use":
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            value = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
            reporter.on_tool_call(name=str(name or "tool"), input=value or {})
        elif block_type in {"thinking", "reasoning"}:
            value = (
                block.get("thinking")
                if isinstance(block, dict)
                else getattr(block, "thinking", None)
            )
            if value:
                reporter.on_stream_delta(kind="reasoning", text=str(value))
