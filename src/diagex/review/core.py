"""Persistent review sessions for the local P&ID review workbench.

The extractor's graph is immutable.  A review session is reconstructed by
replaying append-only events against that graph; ``state.json`` is only a
cache for humans and recovery tooling.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diagex.dexpi_schema import (
    ACTUATION_TYPE_KEYS,
    EQUIPMENT_CLASS_KEYS,
    INSTRUMENT_CLASS_KEYS,
    INSTRUMENT_FUNCTION_KEYS,
    VALVE_TYPE_KEYS,
)
from diagex.vision.models import EquipmentAssembly, ReconciledEdge, ReconciledGraph, ReconciledNode

SCHEMA_VERSION = "1.1.0"
REVIEW_STATES = {"unreviewed", "approved", "modified", "rejected", "waived", "resolved"}
EVIDENCE_STATES = {"unreviewed", "linked", "dismissed"}
EVIDENCE_DISPOSITIONS = {
    "line_number",
    "drawing_reference",
    "note",
    "dimension",
    "non_object",
    "other",
}
PAGE_ROLES = {"pid", "legend", "cover", "notes", "other"}
LINE_TYPES = {
    "process",
    "signal_electric",
    "signal_pneumatic",
    "instrument_capillary",
    "electrical_power",
    "other",
}
KINDS = {"equipment", "instrument", "line", "connection", "text", "note", "opc"}

_DRAWING_REFERENCE_RE = re.compile(r"^(?:DW|DWG|SH|SHT)[A-Z0-9]*[- ]?\d", re.IGNORECASE)
_LINE_NUMBER_RE = re.compile(
    r"^(?:\d{1,3}[\"”']?[- ])?[A-Z]{1,5}-\d{3,}(?:-[A-Z0-9]+){1,}$",
    re.IGNORECASE,
)
_EQUIPMENT_TAG_RE = re.compile(r"^\d{3,6}[- ]?[A-Z]{1,4}[- ]?\d{1,5}[A-Z]?$", re.IGNORECASE)
_INSTRUMENT_TAG_RE = re.compile(
    r"^(?:[A-Z]{2,5}[- ]?\d{1,6}|[A-Z][- ]?\d{2,6})[A-Z]?$", re.IGNORECASE
)
_TAG_EXCLUDED_PREFIXES = {"DN", "PN", "SCH", "CL", "NO", "REV", "PAGE"}


class ReviewError(RuntimeError):
    """Base class for review-session failures."""


class ReviewConflictError(ReviewError):
    """Raised when a stale browser revision or mismatched source is supplied."""


class ReviewIncompleteError(ReviewError):
    """Raised when export is requested before the review is complete."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class ReviewValidationError(ReviewError):
    """Raised when a reviewed graph cannot produce a valid DEXPI model."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any, length: int = 12) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def resolve_graph_path(target: Path | str) -> Path:
    path = Path(target).expanduser().resolve()
    if path.is_dir():
        path = path / "graph.json"
    if not path.is_file():
        raise FileNotFoundError(f"review graph not found: {path}")
    return path


def _load_build_issues(graph_path: Path) -> list[dict[str, Any]]:
    result_path = graph_path.with_name("result.json")
    if not result_path.exists():
        return []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[dict[str, Any]] = []
    for index, message in enumerate(result.get("dexpi_issues") or []):
        text = str(message)
        match = re.search(r"\b(?:node|edge) '([^']+)'", text)
        out.append(
            {
                "id": f"b-{index:04d}-{_canonical_hash(text, 8)}",
                "message": text,
                "target_id": match.group(1) if match else None,
            }
        )
    return out


def _load_page_manifest(graph_path: Path) -> dict[str, Any] | None:
    path = graph_path.with_name("pages.json")
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("pages"), list) else None


def _tag_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _text_candidate_kind(text: str) -> tuple[str, bool] | None:
    """Classify conservative native-text candidates without plant rules.

    Engineering tags are completeness-gating.  Drawing references and line
    numbers are retained as useful context, but do not block completion.
    """

    compact = " ".join(str(text).strip().split())
    key = _tag_key(compact)
    if not compact or len(key) < 3 or len(key) > 32:
        return None
    if _DRAWING_REFERENCE_RE.fullmatch(compact):
        return "drawing_reference", False
    if _LINE_NUMBER_RE.fullmatch(compact):
        return "line_number", False
    prefix = re.match(r"^[A-Z]+", key)
    if prefix and prefix.group(0) in _TAG_EXCLUDED_PREFIXES:
        return None
    if _EQUIPMENT_TAG_RE.fullmatch(compact):
        return "equipment_tag", True
    if _INSTRUMENT_TAG_RE.fullmatch(compact):
        return "instrument_tag", True
    return None


def _bbox_union(spans: list[dict[str, Any]]) -> dict[str, int]:
    boxes = [span["bbox"] for span in spans]
    x1 = min(int(box["x"]) for box in boxes)
    y1 = min(int(box["y"]) for box in boxes)
    x2 = max(int(box["x"]) + int(box["w"]) for box in boxes)
    y2 = max(int(box["y"]) + int(box["h"]) for box in boxes)
    return {"x": x1, "y": y1, "w": max(1, x2 - x1), "h": max(1, y2 - y1)}


def _candidate_span_groups(spans: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = [[span] for span in spans]
    by_line: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for span in spans:
        key = (int(span.get("block_index", -1)), int(span.get("line_index", -1)))
        by_line.setdefault(key, []).append(span)
    for line in by_line.values():
        ordered = sorted(line, key=lambda item: int(item.get("word_index", 0)))
        for size in (2, 3):
            for start in range(0, len(ordered) - size + 1):
                window = ordered[start : start + size]
                reasonable = all(
                    int(right["bbox"]["x"]) - (int(left["bbox"]["x"]) + int(left["bbox"]["w"]))
                    <= max(int(left["bbox"]["h"]), int(right["bbox"]["h"])) * 3
                    for left, right in zip(window, window[1:], strict=False)
                )
                if reasonable:
                    groups.append(window)
    return groups


def _node_text_keys(node: ReconciledNode) -> set[str]:
    attrs = node.attributes or {}
    values: list[Any] = [node.label, attrs.get("canonical_tag")]
    values.extend(attrs.get("raw_text_candidates") or [])
    values.extend(node.alternate_readings or [])
    return {key for value in values if (key := _tag_key(value))}


def _load_evidence_candidates(graph_path: Path, graph: ReconciledGraph) -> list[dict[str, Any]]:
    evidence_dir = graph_path.parent / "evidence"
    if not evidence_dir.is_dir():
        return []
    inventory_path = evidence_dir / "native-text-inventory.json"
    if inventory_path.is_file():
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            inventory = None
        if isinstance(inventory, dict) and isinstance(inventory.get("items"), list):
            return [
                {
                    "id": str(item["id"]),
                    "source_ids": list(item.get("source_ids") or []),
                    "text": str(item.get("text") or ""),
                    "normalised_text": str(item.get("normalised_text") or ""),
                    "page_index": int(item.get("page_index", -1)),
                    "bbox_global": copy.deepcopy(item.get("bbox_global") or {}),
                    "candidate_kind": str(item.get("candidate_kind") or "unknown_tag"),
                    "blocking": bool(item.get("blocking")),
                    "matched_node_ids": list(item.get("matched_node_ids") or []),
                    "matched_assembly_ids": list(item.get("matched_assembly_ids") or []),
                    "ownership_scope": item.get("ownership_scope", "physical_symbol"),
                    "match_method": item.get("match_method"),
                    "extraction_status": item.get("status"),
                    "extraction_reason": item.get("reason"),
                    "expected_kind": item.get("expected_kind"),
                    "tag_semantics": copy.deepcopy(item.get("tag_semantics")),
                }
                for item in inventory["items"]
                if isinstance(item, dict)
                and item.get("id")
                and int(item.get("page_index", -1)) >= 0
                and isinstance(item.get("bbox_global"), dict)
            ]
    nodes_by_page: dict[int, list[ReconciledNode]] = {}
    for node in graph.nodes:
        nodes_by_page.setdefault(node.page_index, []).append(node)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int, int]] = set()
    for page_path in sorted(evidence_dir.glob("page-*.json")):
        try:
            page = json.loads(page_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        page_index = int(page.get("page_index", -1))
        if page_index < 0 or page.get("role") not in {"pid", "other"}:
            continue
        spans = [
            span
            for span in page.get("text_spans") or []
            if isinstance(span, dict)
            and isinstance(span.get("bbox"), dict)
            and str(span.get("text") or "").strip()
        ]
        for group in _candidate_span_groups(spans):
            text = " ".join(str(span["text"]).strip() for span in group)
            classified = _text_candidate_kind(text)
            if classified is None:
                continue
            candidate_kind, blocking = classified
            bbox = _bbox_union(group)
            key = _tag_key(text)
            dedupe = (page_index, key, bbox["x"] // 8, bbox["y"] // 8)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            source_ids = [str(span.get("id")) for span in group if span.get("id")]
            referenced: list[str] = []
            exact: list[str] = []
            for node in nodes_by_page.get(page_index, []):
                attrs = node.attributes or {}
                node_sources = {
                    *node.source_evidence_ids,
                    *node.source_annotation_ids,
                    *(attrs.get("source_text_ids") or []),
                }
                if node_sources.intersection(source_ids):
                    referenced.append(node.id)
                elif key in _node_text_keys(node):
                    exact.append(node.id)
            matched = referenced or exact
            candidate_id = f"t-{_canonical_hash([page_index, source_ids, key], 12)}"
            candidates.append(
                {
                    "id": candidate_id,
                    "source_ids": source_ids,
                    "text": text,
                    "normalised_text": key,
                    "page_index": page_index,
                    "bbox_global": bbox,
                    "candidate_kind": candidate_kind,
                    "blocking": blocking,
                    "matched_node_ids": matched,
                    "match_method": (
                        "source_reference" if referenced else "exact_label" if exact else None
                    ),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["page_index"],
            item["bbox_global"]["y"],
            item["bbox_global"]["x"],
            item["id"],
        )
    )
    return candidates


def _conflict_records(graph: ReconciledGraph) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, conflict in enumerate(graph.conflicts):
        records.append(
            {
                "id": f"c-{index:04d}-{_canonical_hash(conflict, 8)}",
                "conflict": copy.deepcopy(conflict),
            }
        )
    return records


def _merge_review_findings(
    session: dict[str, Any], graph_path: Path, graph_sha: str, source_sha: str
) -> None:
    """Import source-audit findings without changing the graph or review events."""
    path = graph_path.parent / "review-findings.json"
    if not path.is_file():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("graph_sha256") != graph_sha or document.get("source_sha256") != source_sha:
        raise ReviewConflictError("review findings do not match this graph and source diagram")
    records = document.get("findings")
    if not isinstance(records, list):
        raise ReviewConflictError("review findings must contain a findings list")
    existing = {item["id"]: item for item in session["conflicts"]}
    seen: set[str] = set()
    additions = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ReviewConflictError("each review finding must have a stable id")
        key = "audit-" + record["id"]
        conflict = record.get("conflict")
        if key in seen or not isinstance(conflict, dict) or not conflict.get("type"):
            raise ReviewConflictError("review findings contain a duplicate id or invalid conflict")
        seen.add(key)
        if key in existing:
            if existing[key]["conflict"] != conflict:
                raise ReviewConflictError("an imported finding changed; assign a new finding id")
        else:
            additions.append({"id": key, "conflict": copy.deepcopy(conflict)})
    session["conflicts"].extend(additions)


def _conflict_candidates(graph: dict[str, Any], conflict: dict[str, Any]) -> dict[str, Any]:
    """Return compact, explicit choices referenced by a graph conflict.

    The persisted review state remains unchanged.  These candidates are a
    presentation aid derived from the current reviewed graph each time state
    is sent to the browser, so edits and rejected objects are reflected on
    reload.
    """

    node_ids: list[str] = []
    edge_ids: list[str] = []

    def add_id(items: list[str], value: Any) -> None:
        if isinstance(value, str) and value and value not in items:
            items.append(value)

    add_id(node_ids, conflict.get("node_id"))
    for value in conflict.get("node_ids") or []:
        add_id(node_ids, value)
    add_id(node_ids, conflict.get("primary_node_id"))
    for value in conflict.get("absorbed_node_ids") or []:
        add_id(node_ids, value)
    add_id(edge_ids, conflict.get("edge_id"))
    for value in conflict.get("edge_ids") or []:
        add_id(edge_ids, value)

    pairs: list[dict[str, Any]] = []
    for pair in conflict.get("candidate_pairs") or []:
        if not isinstance(pair, dict):
            continue
        pairs.append(copy.deepcopy(pair))
        for value in pair.get("node_ids") or []:
            add_id(node_ids, value)

    edge_by_id = {
        str(edge.get("id")): edge
        for edge in graph.get("edges") or []
        if isinstance(edge, dict) and edge.get("id")
    }
    for edge_id in edge_ids:
        edge = edge_by_id.get(edge_id)
        if edge:
            add_id(node_ids, edge.get("from_node"))
            add_id(node_ids, edge.get("to_node"))

    node_by_id = {
        str(node.get("id")): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    provenance_ids: set[str] = set()
    for key in ("detection_id", "valve_detection_id", "source_text_id"):
        value = conflict.get(key)
        if isinstance(value, str) and value:
            provenance_ids.add(value)
    for key in ("detection_ids", "candidate_valve_ids", "source_text_ids"):
        provenance_ids.update(
            str(value) for value in conflict.get(key) or [] if isinstance(value, str) and value
        )
    if provenance_ids:
        for node_id, node in node_by_id.items():
            attributes = node.get("attributes") or {}
            node_provenance = {
                *(node.get("source_annotation_ids") or []),
                *(node.get("source_evidence_ids") or []),
                *(attributes.get("source_text_ids") or []),
            }
            if provenance_ids.intersection(node_provenance):
                add_id(node_ids, node_id)
    nodes = []
    for node_id in node_ids:
        node = node_by_id.get(node_id)
        if node is None:
            continue
        nodes.append(
            {
                "id": node_id,
                "label": node.get("label") or node_id,
                "kind": node.get("kind"),
                "page_index": node.get("page_index"),
                "bbox_global": copy.deepcopy(node.get("bbox_global")),
                "confidence": node.get("confidence"),
            }
        )

    edges = []
    for edge_id in edge_ids:
        edge = edge_by_id.get(edge_id)
        if edge is None:
            continue
        edges.append(
            {
                "id": edge_id,
                "from_node": edge.get("from_node"),
                "to_node": edge.get("to_node"),
                "line_type": edge.get("line_type"),
                "polyline_global": copy.deepcopy(edge.get("polyline_global") or []),
                "confidence": edge.get("confidence"),
            }
        )

    labels = [
        str(value)
        for value in conflict.get("labels") or []
        if isinstance(value, (str, int, float)) and str(value).strip()
    ]
    kinds_value = conflict.get("kinds") or {}
    kinds = (
        [str(value) for value in kinds_value]
        if isinstance(kinds_value, list)
        else [str(value) for value in kinds_value]
        if isinstance(kinds_value, dict)
        else []
    )
    decisions = ["crossing", "junction"] if conflict.get("type") == "crossing_or_junction" else []
    return {
        "nodes": nodes,
        "edges": edges,
        "labels": labels,
        "kinds": kinds,
        "pairs": pairs,
        "decisions": decisions,
    }


def _initial_state(graph: ReconciledGraph, session: dict[str, Any]) -> dict[str, Any]:
    evidence_reviews: dict[str, dict[str, Any]] = {}
    for candidate in session.get("evidence_candidates") or []:
        matches = candidate.get("matched_node_ids") or []
        evidence_reviews[candidate["id"]] = {
            "status": "linked" if len(matches) == 1 else "unreviewed",
            "linked_node_id": matches[0] if len(matches) == 1 else None,
            "disposition": None,
            "reason": (
                f"automatically linked by {candidate.get('match_method')}"
                if len(matches) == 1
                else None
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 0,
        "base_graph_sha256": session["graph_sha256"],
        "graph": graph.model_dump(mode="json"),
        "page_reviews": {
            str(page["page_index"]): {
                "status": "unreviewed",
                "role": None,
                "reason": None,
            }
            for page in session["pages"]
        },
        "node_reviews": {node.id: "unreviewed" for node in graph.nodes},
        "edge_reviews": {edge.id: "unreviewed" for edge in graph.edges},
        "assembly_reviews": {assembly.id: "unreviewed" for assembly in graph.assemblies},
        "evidence_reviews": evidence_reviews,
        "conflict_reviews": {
            item["id"]: {
                "status": (
                    "resolved"
                    if item["conflict"].get("status") == "resolved"
                    else "waived"
                    if item["conflict"].get("status") == "waived"
                    else "unreviewed"
                ),
                "reason": (
                    item["conflict"].get("reason")
                    if item["conflict"].get("status") in {"resolved", "waived"}
                    else None
                ),
                "conflict": item["conflict"],
            }
            for item in session["conflicts"]
        },
        "warnings": [],
        "updated_at": session["created_at"],
    }


def _find_object(items: list[dict[str, Any]], object_id: str) -> dict[str, Any] | None:
    return next((item for item in items if item.get("id") == object_id), None)


_CONNECTION_CONFLICTS = {
    "unsupported_vector_route",
    "page_graph_uncertain_candidate",
    "endpoint_role_uncertain",
    "unsupported_endpoint_combination",
    "visual_topology_disagreement",
}


def _connection_target(conflict: dict[str, Any]) -> str | None:
    if conflict.get("type") in _CONNECTION_CONFLICTS and not conflict.get("edge_ids"):
        return conflict.get("edge_id")
    return None


def _confirm_provisional_edge(edge: dict[str, Any]) -> None:
    """Turn a review-only draft edge into an explicitly human-confirmed edge."""

    attributes = dict(edge.get("attributes") or {})
    for key in (
        "provisional_review_only",
        "requires_human_review",
        "review_conflict_type",
        "review_reason",
    ):
        attributes.pop(key, None)
    attributes["human_review_confirmed"] = True
    edge["attributes"] = attributes


def _replace_object(
    items: list[dict[str, Any]], object_id: str, value: dict[str, Any] | None
) -> None:
    position = next((i for i, item in enumerate(items) if item.get("id") == object_id), None)
    if value is None:
        if position is not None:
            items.pop(position)
        return
    if position is None:
        items.append(copy.deepcopy(value))
    else:
        items[position] = copy.deepcopy(value)


def _target_snapshot(state: dict[str, Any], target_type: str, target_id: str) -> Any:
    if target_type == "assembly":
        return {
            "object": copy.deepcopy(_find_object(state["graph"].get("assemblies", []), target_id)),
            "status": state.get("assembly_reviews", {}).get(target_id, "unreviewed"),
            "linked_conflicts": {
                cid: copy.deepcopy(review)
                for cid, review in state["conflict_reviews"].items()
                if (review.get("conflict") or {}).get("assembly_id") == target_id
            },
        }
    if target_type == "node":
        return {
            "object": copy.deepcopy(_find_object(state["graph"]["nodes"], target_id)),
            "status": state["node_reviews"].get(target_id),
        }
    if target_type == "edge":
        return {
            "object": copy.deepcopy(_find_object(state["graph"]["edges"], target_id)),
            "status": state["edge_reviews"].get(target_id),
            "linked_conflicts": {
                cid: copy.deepcopy(review)
                for cid, review in state["conflict_reviews"].items()
                if _connection_target(review.get("conflict") or {}) == target_id
            },
        }
    if target_type == "page":
        return copy.deepcopy(state["page_reviews"].get(target_id))
    if target_type == "conflict":
        return copy.deepcopy(state["conflict_reviews"].get(target_id))
    if target_type == "evidence":
        return copy.deepcopy(state["evidence_reviews"].get(target_id))
    raise ValueError(f"unknown review target type: {target_type}")


def _apply_snapshot(state: dict[str, Any], target_type: str, target_id: str, value: Any) -> None:
    if target_type == "assembly":
        _replace_object(
            state["graph"].setdefault("assemblies", []),
            target_id,
            value.get("object") if value else None,
        )
        statuses = state.setdefault("assembly_reviews", {})
        if not value or value.get("status") is None:
            statuses.pop(target_id, None)
        else:
            statuses[target_id] = value["status"]
        for cid, review in (value or {}).get("linked_conflicts", {}).items():
            state["conflict_reviews"][cid] = copy.deepcopy(review)
        return
    if target_type == "node":
        _replace_object(state["graph"]["nodes"], target_id, value.get("object") if value else None)
        if value is None or value.get("status") is None:
            state["node_reviews"].pop(target_id, None)
        else:
            state["node_reviews"][target_id] = value["status"]
        return
    if target_type == "edge":
        _replace_object(state["graph"]["edges"], target_id, value.get("object") if value else None)
        if value is None or value.get("status") is None:
            state["edge_reviews"].pop(target_id, None)
        else:
            state["edge_reviews"][target_id] = value["status"]
        for cid, review in (value or {}).get("linked_conflicts", {}).items():
            state["conflict_reviews"][cid] = copy.deepcopy(review)
        return
    if target_type == "page":
        if value is None:
            state["page_reviews"].pop(target_id, None)
        else:
            state["page_reviews"][target_id] = copy.deepcopy(value)
        return
    if target_type == "conflict":
        if value is None:
            state["conflict_reviews"].pop(target_id, None)
        else:
            state["conflict_reviews"][target_id] = copy.deepcopy(value)
        return
    if target_type == "evidence":
        if value is None:
            state["evidence_reviews"].pop(target_id, None)
        else:
            state["evidence_reviews"][target_id] = copy.deepcopy(value)
        return
    raise ValueError(f"unknown review target type: {target_type}")


def _apply_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    _apply_snapshot(state, event["target_type"], event["target_id"], event.get("after"))
    state["revision"] = int(event["revision"])
    state["updated_at"] = event["timestamp"]


def _read_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.append(f"ignored malformed events.jsonl line {line_number}")
    return events, warnings


class ReviewStore:
    """Thread-safe, event-sourced review session."""

    def __init__(
        self,
        *,
        graph_path: Path,
        source_path: Path,
        out_dir: Path,
        rater: str,
    ) -> None:
        from diagex.review.render import prepare_page_assets

        self.graph_path = resolve_graph_path(graph_path)
        self.source_path = Path(source_path).expanduser().resolve()
        if not self.source_path.is_file():
            raise FileNotFoundError(f"source diagram not found: {self.source_path}")
        if not rater.strip():
            raise ValueError("--rater must not be empty")
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = self.out_dir / "session.json"
        self.events_path = self.out_dir / "events.jsonl"
        self.state_path = self.out_dir / "state.json"
        self._lock = threading.RLock()

        graph = ReconciledGraph.model_validate_json(self.graph_path.read_text(encoding="utf-8"))
        graph_sha = _file_sha256(self.graph_path)
        source_sha = _file_sha256(self.source_path)
        if self.session_path.exists():
            session = json.loads(self.session_path.read_text(encoding="utf-8"))
            if session.get("graph_sha256") != graph_sha:
                raise ReviewConflictError(
                    "the source graph changed after this review session was created; "
                    "choose another --out-dir to start a new review"
                )
            if session.get("source_sha256") != source_sha:
                raise ReviewConflictError(
                    "the source diagram changed after this review session was created; "
                    "choose another --out-dir to start a new review"
                )
            if session.get("rater") != rater:
                raise ReviewConflictError(
                    f"this single-reviewer session belongs to {session.get('rater')!r}, not {rater!r}"
                )
            pages = prepare_page_assets(
                self.source_path, self.out_dir / "pages", cached=session.get("pages")
            )
            session["pages"] = pages
            if "evidence_candidates" not in session:
                session["evidence_candidates"] = _load_evidence_candidates(self.graph_path, graph)
        else:
            pages = prepare_page_assets(
                self.source_path,
                self.out_dir / "pages",
                expected=_load_page_manifest(self.graph_path),
            )
            session = {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "rater": rater,
                "graph_path": str(self.graph_path),
                "source_path": str(self.source_path),
                "graph_sha256": graph_sha,
                "source_sha256": source_sha,
                "pages": pages,
                "conflicts": _conflict_records(graph),
                "build_issues": _load_build_issues(self.graph_path),
                "evidence_candidates": _load_evidence_candidates(self.graph_path, graph),
            }
        _merge_review_findings(session, self.graph_path, graph_sha, source_sha)
        self.session = session
        _atomic_json(self.session_path, self.session)

        self.base_graph = graph
        self.events, event_warnings = _read_events(self.events_path)
        self.state = _initial_state(graph, self.session)
        for event in self.events:
            if event.get("base_graph_sha256") != graph_sha:
                raise ReviewConflictError("an event belongs to a different source graph")
            _apply_event(self.state, event)
        self.state["warnings"] = event_warnings + self._geometry_warnings()
        _atomic_json(self.state_path, self.state)

    @classmethod
    def open(
        cls,
        target: Path | str,
        *,
        source_path: Path | str,
        rater: str,
        out_dir: Path | str | None = None,
    ) -> ReviewStore:
        graph_path = resolve_graph_path(target)
        review_dir = Path(out_dir) if out_dir is not None else graph_path.parent / "review"
        return cls(
            graph_path=graph_path,
            source_path=Path(source_path),
            out_dir=review_dir,
            rater=rater,
        )

    def _geometry_warnings(self) -> list[str]:
        page_dims = {
            int(page["page_index"]): (int(page["width"]), int(page["height"]))
            for page in self.session["pages"]
        }
        warnings: list[str] = []
        for node in self.base_graph.nodes:
            dims = page_dims.get(node.page_index)
            if dims is None:
                warnings.append(f"node {node.id} references missing page {node.page_index + 1}")
                continue
            box = node.bbox_global
            if box.x < 0 or box.y < 0 or box.x2 > dims[0] or box.y2 > dims[1]:
                warnings.append(f"node {node.id} falls outside rendered page {node.page_index + 1}")
        return warnings

    def _append_event_record(self, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        needs_separator = self.events_path.exists() and self.events_path.stat().st_size > 0
        if needs_separator:
            with self.events_path.open("rb") as check:
                check.seek(-1, os.SEEK_END)
                needs_separator = check.read(1) != b"\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            if needs_separator:
                handle.write("\n")
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _normal_after(
        self,
        target_type: str,
        target_id: str,
        operation: str,
        payload: dict[str, Any],
        reason: str | None,
    ) -> Any:
        before = _target_snapshot(self.state, target_type, target_id)
        if target_type == "page":
            if before is None:
                raise KeyError(f"unknown page {target_id}")
            role = str(payload.get("role") or before.get("role") or "")
            if role not in PAGE_ROLES:
                raise ValueError(f"page role must be one of {sorted(PAGE_ROLES)}")
            if operation not in {"approve", "waive", "modify"}:
                raise ValueError("pages support approve, modify, or waive")
            return {
                "status": "waived" if operation == "waive" else "approved",
                "role": role,
                "reason": reason,
            }
        if target_type == "conflict":
            if before is None:
                raise KeyError(f"unknown conflict {target_id}")
            if operation not in {"resolve", "waive"}:
                raise ValueError("conflicts support resolve or waive")
            if target_id.startswith("audit-") and not (reason or "").strip():
                raise ValueError("record a decision or clarification for this audit finding")
            after = copy.deepcopy(before)
            after["status"] = "resolved" if operation == "resolve" else "waived"
            after["reason"] = reason
            return after
        if target_type == "evidence":
            if before is None:
                raise KeyError(f"unknown evidence candidate {target_id}")
            if operation == "link":
                node_id = str(payload.get("node_id") or "")
                node = _find_object(self.state["graph"]["nodes"], node_id)
                if node is None or self.state["node_reviews"].get(node_id) == "rejected":
                    raise ValueError("evidence must link to an active node")
                return {
                    "status": "linked",
                    "linked_node_id": node_id,
                    "disposition": None,
                    "reason": reason,
                }
            if operation == "dismiss":
                disposition = str(payload.get("disposition") or "")
                if disposition not in EVIDENCE_DISPOSITIONS:
                    raise ValueError(
                        f"evidence disposition must be one of {sorted(EVIDENCE_DISPOSITIONS)}"
                    )
                return {
                    "status": "dismissed",
                    "linked_node_id": None,
                    "disposition": disposition,
                    "reason": reason,
                }
            if operation == "reopen":
                return {
                    "status": "unreviewed",
                    "linked_node_id": None,
                    "disposition": None,
                    "reason": reason,
                }
            raise ValueError("evidence supports link, dismiss, or reopen")
        if target_type not in {"node", "edge", "assembly"}:
            raise ValueError(f"unknown target type: {target_type}")
        model_cls = {"node": ReconciledNode, "edge": ReconciledEdge, "assembly": EquipmentAssembly}[
            target_type
        ]
        if operation == "add":
            if before and before.get("object") is not None:
                raise ValueError(f"{target_type} {target_id} already exists")
            candidate = {**payload, "id": target_id}
            obj = model_cls.model_validate(candidate).model_dump(mode="json")
            return {"object": obj, "status": "modified"}
        if before is None or before.get("object") is None:
            raise KeyError(f"unknown {target_type} {target_id}")
        if operation == "approve":
            obj = copy.deepcopy(before["object"])
            if target_type == "assembly" and (
                obj.get("status") != "supported" or not obj.get("label")
            ):
                raise ValueError("choose an assembly identity before approving")
            if target_type == "edge":
                alternatives = (obj.get("attributes") or {}).get("route_alternatives") or []
                if alternatives:
                    choice = _find_object(
                        alternatives, str(payload.get("route_candidate_id") or "")
                    )
                    if choice is None:
                        raise ValueError("select a route before approving this connection")
                    obj["from_node"], obj["to_node"] = choice["from_node"], choice["to_node"]
                    obj["polyline_global"] = copy.deepcopy(choice["polyline_global"])
                    obj["source_evidence_ids"] = copy.deepcopy(choice["source_evidence_ids"])
                    obj["attributes"].update(copy.deepcopy(choice["attributes"]))
                    obj["attributes"]["selected_route_candidate_id"] = choice["id"]
                _confirm_provisional_edge(obj)
            return {"object": obj, "status": "approved"}
        if operation == "reject":
            return {"object": copy.deepcopy(before["object"]), "status": "rejected"}
        if operation == "modify":
            candidate = copy.deepcopy(before["object"])
            if target_type == "assembly":
                if not str(payload.get("label") or "").strip():
                    raise ValueError("assembly identity must not be empty")
                if set(payload) - {"label"}:
                    raise ValueError(
                        "assembly review changes the identity; physical members stay separate"
                    )
                candidate["status"] = "supported"
            candidate.update(payload)
            candidate["id"] = target_id
            obj = model_cls.model_validate(candidate).model_dump(mode="json")
            if target_type == "edge":
                _confirm_provisional_edge(obj)
            return {"object": obj, "status": "modified"}
        raise ValueError(f"unsupported {target_type} operation: {operation}")

    def append_action(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            expected = int(request.get("expected_revision", -1))
            if expected != int(self.state["revision"]):
                raise ReviewConflictError(
                    f"stale review revision {expected}; current revision is {self.state['revision']}"
                )
            operation = str(request.get("operation") or "")
            if operation == "undo":
                return self._undo(request)
            target_type = str(request.get("target_type") or "")
            target_id = str(request.get("target_id") or "")
            if operation == "add" and not target_id:
                prefix = "n" if target_type == "node" else "e"
                target_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
            if not target_id:
                raise ValueError("target_id is required")
            payload = request.get("after") or {}
            if not isinstance(payload, dict):
                raise ValueError("after must be an object")
            reason = str(request.get("reason") or "").strip() or None
            before = _target_snapshot(self.state, target_type, target_id)
            after = self._normal_after(target_type, target_id, operation, payload, reason)
            if target_type in {"edge", "assembly"} and operation in {"approve", "modify", "reject"}:
                after["linked_conflicts"] = copy.deepcopy(before.get("linked_conflicts") or {})
                for review in after["linked_conflicts"].values():
                    if review.get("status") not in {"resolved", "waived"}:
                        review.update(
                            status="resolved",
                            reason=reason,
                            decision=operation,
                            decision_target_type=target_type,
                        )
            event = {
                "schema_version": SCHEMA_VERSION,
                "revision": int(self.state["revision"]) + 1,
                "timestamp": _utc_now(),
                "reviewer": self.session["rater"],
                "base_graph_sha256": self.session["graph_sha256"],
                "target_type": target_type,
                "target_id": target_id,
                "operation": operation,
                "before": before,
                "after": after,
                "reason": reason,
            }
            self._commit_event(event)
            return {"event": event, "state": self.public_state()}

    def _undo(self, request: dict[str, Any]) -> dict[str, Any]:
        undone = {
            int(event["undo_of"]) for event in self.events if event.get("undo_of") is not None
        }
        candidate = next(
            (
                event
                for event in reversed(self.events)
                if event.get("operation") != "undo" and int(event["revision"]) not in undone
            ),
            None,
        )
        if candidate is None:
            raise ValueError("nothing to undo")
        target_type = candidate["target_type"]
        target_id = candidate["target_id"]
        event = {
            "schema_version": SCHEMA_VERSION,
            "revision": int(self.state["revision"]) + 1,
            "timestamp": _utc_now(),
            "reviewer": self.session["rater"],
            "base_graph_sha256": self.session["graph_sha256"],
            "target_type": target_type,
            "target_id": target_id,
            "operation": "undo",
            "undo_of": candidate["revision"],
            "before": _target_snapshot(self.state, target_type, target_id),
            "after": copy.deepcopy(candidate.get("before")),
            "reason": str(request.get("reason") or "").strip() or None,
        }
        self._commit_event(event)
        return {"event": event, "state": self.public_state()}

    def _commit_event(self, event: dict[str, Any]) -> None:
        self._append_event_record(event)
        self.events.append(event)
        _apply_event(self.state, event)
        _atomic_json(self.state_path, self.state)

    def reviewed_graph(self) -> ReconciledGraph:
        raw = copy.deepcopy(self.state["graph"])
        active_nodes = {
            node["id"]
            for node in raw["nodes"]
            if self.state["node_reviews"].get(node["id"]) != "rejected"
        }
        raw["nodes"] = [node for node in raw["nodes"] if node["id"] in active_nodes]
        raw["assemblies"] = [
            a
            for a in raw.get("assemblies", [])
            if self.state.get("assembly_reviews", {}).get(a["id"]) != "rejected"
        ]
        assembly_ids = {a["id"] for a in raw["assemblies"]}
        for assembly in raw["assemblies"]:
            assembly["member_node_ids"] = [
                nid for nid in assembly["member_node_ids"] if nid in active_nodes
            ]
            if assembly.get("parent_assembly_id") not in assembly_ids:
                assembly["parent_assembly_id"] = None
        for node in raw["nodes"]:
            if "assembly_ids" in node.get("attributes", {}):
                node["attributes"]["assembly_ids"] = [
                    aid for aid in node["attributes"]["assembly_ids"] if aid in assembly_ids
                ]
        for binding in raw.get("text_bindings", []):
            binding["assembly_ids"] = [
                aid for aid in binding.get("assembly_ids", []) if aid in assembly_ids
            ]
            binding["node_ids"] = [
                nid for nid in binding.get("node_ids", []) if nid in active_nodes
            ]
        raw["edges"] = [
            edge
            for edge in raw["edges"]
            if self.state["edge_reviews"].get(edge["id"]) != "rejected"
        ]
        raw["conflicts"] = [
            {
                **copy.deepcopy(item["conflict"]),
                "review_status": item["status"],
                "review_reason": item.get("reason"),
            }
            for item in self.state["conflict_reviews"].values()
        ]
        return ReconciledGraph.model_validate(raw)

    def completion(self) -> dict[str, Any]:
        unreviewed_pages = [
            page_id
            for page_id, review in self.state["page_reviews"].items()
            if review.get("status") not in {"approved", "waived"}
            or review.get("role") not in PAGE_ROLES
        ]
        unreviewed_nodes = [
            object_id
            for object_id, status in self.state["node_reviews"].items()
            if status == "unreviewed"
        ]
        unreviewed_edges = [
            object_id
            for object_id, status in self.state["edge_reviews"].items()
            if status == "unreviewed"
        ]
        unresolved_conflicts = [
            conflict_id
            for conflict_id, review in self.state["conflict_reviews"].items()
            if review.get("status") not in {"resolved", "waived"}
        ]
        graph = self.reviewed_graph()
        node_ids = {node.id for node in graph.nodes}
        dangling_edges = [
            edge.id
            for edge in graph.edges
            if edge.from_node not in node_ids or edge.to_node not in node_ids
        ]
        candidates = {item["id"]: item for item in self.session.get("evidence_candidates") or []}
        unreviewed_evidence = [
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate.get("blocking")
            and self.state["evidence_reviews"].get(candidate_id, {}).get("status") == "unreviewed"
        ]
        invalid_evidence_links = [
            candidate_id
            for candidate_id, review in self.state["evidence_reviews"].items()
            if review.get("status") == "linked" and review.get("linked_node_id") not in node_ids
        ]
        details = {
            "unreviewed_assemblies": [
                a["id"]
                for a in self.state["graph"].get("assemblies", [])
                if self.state.get("assembly_reviews", {}).get(a["id"], "unreviewed") == "unreviewed"
            ],
            "unreviewed_pages": unreviewed_pages,
            "unreviewed_nodes": unreviewed_nodes,
            "unreviewed_edges": unreviewed_edges,
            "unresolved_conflicts": unresolved_conflicts,
            "dangling_edges": dangling_edges,
            "unreviewed_evidence": unreviewed_evidence,
            "invalid_evidence_links": invalid_evidence_links,
        }
        details["complete"] = not any(details[key] for key in details if key != "complete")
        return details

    def _inventory(self) -> dict[str, Any]:
        graph = self.state["graph"]
        active_nodes = [
            node
            for node in graph["nodes"]
            if self.state["node_reviews"].get(node["id"]) != "rejected"
        ]
        node_by_id = {node["id"]: node for node in active_nodes}
        degrees = {node_id: 0 for node_id in node_by_id}
        for edge in graph["edges"]:
            if self.state["edge_reviews"].get(edge["id"]) == "rejected":
                continue
            if edge.get("from_node") in degrees:
                degrees[edge["from_node"]] += 1
            if edge.get("to_node") in degrees:
                degrees[edge["to_node"]] += 1

        pages: dict[str, dict[str, Any]] = {
            str(page["page_index"]): {
                "page_index": int(page["page_index"]),
                "nodes": 0,
                "reviewed_nodes": 0,
                "isolated_nodes": 0,
                "tag_candidates": 0,
                "unresolved_tag_candidates": 0,
                "page_review": copy.deepcopy(
                    self.state["page_reviews"].get(str(page["page_index"]))
                ),
            }
            for page in self.session["pages"]
        }
        for node in active_nodes:
            row = pages.setdefault(str(node["page_index"]), {"page_index": node["page_index"]})
            row["nodes"] = row.get("nodes", 0) + 1
            if self.state["node_reviews"].get(node["id"]) != "unreviewed":
                row["reviewed_nodes"] = row.get("reviewed_nodes", 0) + 1
            if degrees.get(node["id"], 0) == 0:
                row["isolated_nodes"] = row.get("isolated_nodes", 0) + 1

        evidence: list[dict[str, Any]] = []
        for candidate in self.session.get("evidence_candidates") or []:
            item = copy.deepcopy(candidate)
            review = copy.deepcopy(
                self.state["evidence_reviews"].get(
                    item["id"],
                    {
                        "status": "unreviewed",
                        "linked_node_id": None,
                        "disposition": None,
                        "reason": None,
                    },
                )
            )
            item["review"] = review
            cx = item["bbox_global"]["x"] + item["bbox_global"]["w"] / 2
            cy = item["bbox_global"]["y"] + item["bbox_global"]["h"] / 2
            nearby = sorted(
                (
                    (
                        (node["bbox_global"]["x"] + node["bbox_global"]["w"] / 2 - cx) ** 2
                        + (node["bbox_global"]["y"] + node["bbox_global"]["h"] / 2 - cy) ** 2,
                        node["id"],
                    )
                    for node in active_nodes
                    if node["page_index"] == item["page_index"]
                ),
                key=lambda value: (value[0], value[1]),
            )
            item["nearby_node_ids"] = [node_id for _, node_id in nearby[:6]]
            evidence.append(item)
            row = pages.setdefault(str(item["page_index"]), {"page_index": item["page_index"]})
            if item.get("blocking"):
                row["tag_candidates"] = row.get("tag_candidates", 0) + 1
                if review.get("status") == "unreviewed":
                    row["unresolved_tag_candidates"] = row.get("unresolved_tag_candidates", 0) + 1

        counts: dict[str, int] = {
            "nodes": len(active_nodes),
            "reviewed_nodes": sum(
                1
                for node in active_nodes
                if self.state["node_reviews"].get(node["id"]) != "unreviewed"
            ),
            "isolated_nodes": sum(1 for value in degrees.values() if value == 0),
            "evidence_candidates": len(evidence),
            "blocking_tag_candidates": sum(1 for item in evidence if item.get("blocking")),
            "unresolved_tag_candidates": sum(
                1
                for item in evidence
                if item.get("blocking") and item["review"].get("status") == "unreviewed"
            ),
        }
        return {
            "counts": counts,
            "degrees": degrees,
            "pages": [pages[key] for key in sorted(pages, key=int)],
            "evidence": evidence,
        }

    def _queue(self) -> list[dict[str, Any]]:
        graph = self.state["graph"]
        conflict_targets: set[str] = set()
        for review in self.state["conflict_reviews"].values():
            if review.get("status") in {"resolved", "waived"}:
                continue
            candidates = _conflict_candidates(graph, review.get("conflict") or {})
            conflict_targets.update(item["id"] for item in candidates["nodes"])
            conflict_targets.update(item["id"] for item in candidates["edges"])
        issue_targets = {
            item.get("target_id")
            for item in self.session.get("build_issues") or []
            if item.get("target_id")
        }
        partial_pages = {
            int(page)
            for page, status in (graph.get("per_page_status") or {}).items()
            if status in {"partial", "error", "cost_exhausted"}
        }
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        rows: list[dict[str, Any]] = []
        for assembly in graph.get("assemblies", []):
            box = assembly["bbox_global"]
            tier = 0 if assembly["status"] != "supported" else 1
            rows.append(
                {
                    "target_type": "assembly",
                    "target_id": assembly["id"],
                    "page_index": assembly["page_index"],
                    "label": assembly.get("label") or " / ".join(assembly["label_candidates"]),
                    "status": self.state.get("assembly_reviews", {}).get(
                        assembly["id"], "unreviewed"
                    ),
                    "tier": tier,
                    "reasons": ["assembly identity"],
                    "sort": [tier, assembly["page_index"], box["y"], box["x"], assembly["id"]],
                }
            )
        for node in graph["nodes"]:
            reasons: list[str] = []
            if node["page_index"] in partial_pages:
                reasons.append("partial page")
            if node["id"] in conflict_targets:
                reasons.append("graph conflict")
            if node["id"] in issue_targets:
                reasons.append("build issue")
            if node.get("confidence") in {"medium", "low"}:
                reasons.append(f"{node['confidence']} confidence")
            attrs = node.get("attributes") or {}
            if node.get("kind") == "equipment" and not attrs.get("equipment_class"):
                reasons.append("unclassified")
            tier = (
                0
                if any(reason in {"partial page", "graph conflict"} for reason in reasons)
                else (1 if reasons else 2)
            )
            box = node["bbox_global"]
            rows.append(
                {
                    "target_type": "node",
                    "target_id": node["id"],
                    "page_index": node["page_index"],
                    "label": node.get("label") or node["id"],
                    "status": self.state["node_reviews"].get(node["id"], "unreviewed"),
                    "tier": tier,
                    "reasons": reasons,
                    "sort": [tier, node["page_index"], box["y"], box["x"], node["id"]],
                }
            )
        for edge in graph["edges"]:
            start = node_by_id.get(edge["from_node"], {})
            page_index = int(start.get("page_index", 0))
            reasons = []
            if page_index in partial_pages:
                reasons.append("partial page")
            if edge["id"] in conflict_targets:
                reasons.append("graph conflict")
            if edge["id"] in issue_targets:
                reasons.append("build issue")
            if edge.get("confidence") in {"medium", "low"}:
                reasons.append(f"{edge['confidence']} confidence")
            if edge["from_node"] not in node_by_id or edge["to_node"] not in node_by_id:
                reasons.append("dangling endpoint")
            tier = (
                0
                if any(reason in {"partial page", "graph conflict"} for reason in reasons)
                else (1 if reasons else 2)
            )
            label = (edge.get("attributes") or {}).get(
                "line_id"
            ) or f"{edge['from_node']} → {edge['to_node']}"
            rows.append(
                {
                    "target_type": "edge",
                    "target_id": edge["id"],
                    "page_index": page_index,
                    "label": label,
                    "status": self.state["edge_reviews"].get(edge["id"], "unreviewed"),
                    "tier": tier,
                    "reasons": reasons,
                    "sort": [tier, page_index, 0, 0, edge["id"]],
                }
            )
        rows.sort(key=lambda row: row.pop("sort"))
        return rows

    def _findings(self, conflicts: dict[str, Any]) -> list[dict[str, Any]]:
        """One actionable queue entry per decision, distinct from blanket sign-off."""
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        graph = self.state["graph"]
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = {edge["id"]: edge for edge in graph["edges"]}
        for conflict_id, item in conflicts.items():
            if item["status"] in {"resolved", "waived"}:
                continue
            conflict = item["conflict"]
            target = item.get("decision_target") or {"type": "conflict", "id": conflict_id}
            key = (target["type"], target["id"])
            category = conflict.get("queue_category") or (
                "identifiers" if conflict["type"] == "source_identifier_conflict" else
                "symbols" if conflict["type"] in {"symbol_semantics", "symbols", "legends", "contextual_symbol_uncertainty"} else
                "connections" if target["type"] == "edge" or conflict["type"] in _CONNECTION_CONFLICTS | {"crossing_or_junction", "page_graph_uncertainty", "page_graph_missing"} or "opc" in conflict["type"] else
                "other"
            )
            if key not in rows:
                page = conflict.get("page_index")
                if page is None:
                    page = next((node["page_index"] for node in item["candidates"]["nodes"]), None)
                edge = edges.get(target["id"])
                label = conflict.get("title") or conflict["type"]
                if edge:
                    label = " → ".join(nodes.get(edge[end], {}).get("label") or edge[end] for end in ("from_node", "to_node"))
                rows[key] = {"target_type": target["type"], "target_id": target["id"],
                             "page_index": page, "label": label, "status": "unreviewed",
                             "category": category, "conflict_ids": [], "reasons": []}
            rows[key]["conflict_ids"].append(conflict_id)
            reason = conflict.get("question") or conflict.get("reason") or conflict["type"]
            if reason not in rows[key]["reasons"]:
                rows[key]["reasons"].append(reason)
        # Include provisional connections even when no conflict record was emitted.
        for edge in edges.values():
            key = ("edge", edge["id"])
            if key in rows or self.state["edge_reviews"].get(edge["id"]) != "unreviewed":
                continue
            attrs = edge.get("attributes") or {}
            if not (attrs.get("provisional_review_only") or attrs.get("requires_human_review")):
                continue
            rows[key] = {"target_type": "edge", "target_id": edge["id"],
                         "page_index": nodes.get(edge["from_node"], {}).get("page_index"),
                         "label": " → ".join(nodes.get(edge[end], {}).get("label") or edge[end] for end in ("from_node", "to_node")),
                         "status": "unreviewed", "category": "connections", "conflict_ids": [],
                         "reasons": [attrs.get("review_reason") or "Provisional connection; confirm its route and meaning."]}
        priority = {"identifiers": 0, "symbols": 1, "connections": 2, "other": 3}
        return sorted(rows.values(), key=lambda row: (priority.get(row["category"], 3), row["page_index"] if row["page_index"] is not None else 9999, row["target_id"]))

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            conflict_reviews = copy.deepcopy(self.state["conflict_reviews"])
            for review in conflict_reviews.values():
                review["candidates"] = _conflict_candidates(
                    self.state["graph"], review.get("conflict") or {}
                )
                edge_id = _connection_target(review.get("conflict") or {})
                assembly_id = (review.get("conflict") or {}).get("assembly_id")
                if assembly_id and _find_object(
                    self.state["graph"].get("assemblies", []), assembly_id
                ):
                    review["decision_target"] = {"type": "assembly", "id": assembly_id}
                if (
                    edge_id
                    and _find_object(self.state["graph"]["edges"], edge_id)
                    and (
                        self.state["edge_reviews"].get(edge_id, "unreviewed") == "unreviewed"
                        or review.get("status") in {"resolved", "waived"}
                    )
                ):
                    review["decision_target"] = {"type": "edge", "id": edge_id}
            return {
                "session": self.session,
                "revision": self.state["revision"],
                "graph": self.state["graph"],
                "reviews": {
                    "pages": self.state["page_reviews"],
                    "nodes": self.state["node_reviews"],
                    "edges": self.state["edge_reviews"],
                    "assemblies": self.state.get("assembly_reviews", {}),
                    "conflicts": conflict_reviews,
                },
                "warnings": self.state.get("warnings") or [],
                "completion": self.completion(),
                "queue": self._queue(),
                "findings": self._findings(conflict_reviews),
                "inventory": self._inventory(),
                "taxonomy": {
                    "kinds": sorted(KINDS),
                    "line_types": sorted(LINE_TYPES),
                    "page_roles": sorted(PAGE_ROLES),
                    "equipment_classes": list(EQUIPMENT_CLASS_KEYS),
                    "valve_types": list(VALVE_TYPE_KEYS),
                    "actuation_types": list(ACTUATION_TYPE_KEYS),
                    "instrument_functions": list(INSTRUMENT_FUNCTION_KEYS),
                    "instrument_classes": list(INSTRUMENT_CLASS_KEYS),
                    "evidence_dispositions": sorted(EVIDENCE_DISPOSITIONS),
                },
            }

    def finish(self) -> dict[str, Any]:
        with self._lock:
            completion = self.completion()
            if not completion["complete"]:
                raise ReviewIncompleteError("review coverage is incomplete", completion)

            from diagex.dexpi import validate as dexpi_validate
            from diagex.dexpi import xml_io
            from diagex.extractors.dexpi_builder import build_dexpi, serialize_model, validate_model

            graph = self.reviewed_graph()
            build = build_dexpi(graph)
            roundtrip_errors = validate_model(build.model)
            semantic = dexpi_validate.semantic_validate(build.model)
            semantic_rows = [asdict(issue) for issue in semantic]
            errors = [
                *roundtrip_errors,
                *[row for row in semantic_rows if row["severity"] == "error"],
            ]
            report: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "finished": False,
                "finished_at": _utc_now(),
                "reviewer": self.session["rater"],
                "base_graph_sha256": self.session["graph_sha256"],
                "revision": self.state["revision"],
                "coverage": completion,
                "review_counts": {
                    "pages": _status_counts(self.state["page_reviews"], nested=True),
                    "nodes": _status_counts(self.state["node_reviews"]),
                    "edges": _status_counts(self.state["edge_reviews"]),
                    "conflicts": _status_counts(self.state["conflict_reviews"], nested=True),
                    "evidence": _status_counts(self.state["evidence_reviews"], nested=True),
                },
                "inventory": self._inventory()["counts"],
                "graph_counts": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
                "dexpi_stats": build.stats,
                "build_issues": build.issues,
                "roundtrip_errors": roundtrip_errors,
                "semantic_issues": semantic_rows,
                "xsd": {"status": "pending", "issues": []},
                "outputs": {},
            }
            if errors:
                _atomic_json(self.out_dir / "review.report.json", report)
                raise ReviewValidationError("reviewed graph failed DEXPI validation", report)

            temp_dir = self.out_dir / ".export-tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            tmp_json = serialize_model(build.model, temp_dir, "pid.reviewed.dexpi")
            tmp_xml = temp_dir / "pid.reviewed.dexpi.xml"
            xml_io.dump(build.model, tmp_xml)
            try:
                xsd_issues = dexpi_validate.xsd_validate(tmp_xml)
                report["xsd"] = {"status": "ran", "issues": [asdict(issue) for issue in xsd_issues]}
            except dexpi_validate.XmlschemaUnavailableError as exc:
                report["xsd"] = {"status": "skipped", "message": str(exc), "issues": []}
                xsd_issues = []
            if dexpi_validate.has_errors(xsd_issues):
                _atomic_json(self.out_dir / "review.report.json", report)
                raise ReviewValidationError("reviewed XML failed XSD validation", report)

            graph_target = self.out_dir / "graph.reviewed.json"
            json_target = self.out_dir / "pid.reviewed.dexpi.json"
            xml_target = self.out_dir / "pid.reviewed.dexpi.xml"
            _atomic_json(graph_target, graph.model_dump(mode="json"))
            os.replace(tmp_json, json_target)
            os.replace(tmp_xml, xml_target)
            try:
                temp_dir.rmdir()
            except OSError:
                pass
            report["finished"] = True
            report["outputs"] = {
                "graph": str(graph_target),
                "dexpi_json": str(json_target),
                "dexpi_xml": str(xml_target),
            }
            _atomic_json(self.out_dir / "review.report.json", report)
            return report


def _status_counts(values: dict[str, Any], *, nested: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values.values():
        status = str(value.get("status")) if nested else str(value)
        counts[status] = counts.get(status, 0) + 1
    return counts
