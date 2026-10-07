"""Completeness audit and structural quality gates for evidence-v2."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from diagex.vision.evidence import PageEvidence
from diagex.vision.models import ReconciledGraph
from diagex.vision.native_text import NativeTextInventory, build_native_text_inventory
from diagex.vision.topology import TopologyResult

QualityStatus = Literal["pass", "partial", "error"]
_CONNECTABLE_KINDS = {"equipment", "instrument", "opc"}


class QualityReport(BaseModel):
    schema_version: str = "1.2.0"
    engine: Literal["evidence-v2"] = "evidence-v2"
    status: QualityStatus
    critical_violations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    unresolved_ambiguities: list[dict[str, Any]] = Field(default_factory=list)
    dexpi: dict[str, Any] = Field(default_factory=dict)


def assess_quality(
    *,
    graph: ReconciledGraph,
    pages: list[PageEvidence],
    topology: list[TopologyResult],
    native_text_inventory: NativeTextInventory | None = None,
) -> QualityReport:
    page_by_index = {page.page_index: page for page in pages}
    node_by_id = {node.id: node for node in graph.nodes}
    critical: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for node in graph.nodes:
        page = page_by_index.get(node.page_index)
        if page is None:
            critical.append(
                {"type": "node_missing_page", "node_id": node.id, "page_index": node.page_index}
            )
            continue
        bbox = node.bbox_global
        if (
            bbox.x < 0
            or bbox.y < 0
            or bbox.w <= 0
            or bbox.h <= 0
            or bbox.x2 > page.width
            or bbox.y2 > page.height
        ):
            critical.append(
                {
                    "type": "out_of_page_geometry",
                    "node_id": node.id,
                    "page_index": node.page_index,
                    "bbox": bbox.model_dump(),
                    "page_size": [page.width, page.height],
                }
            )

    for edge in graph.edges:
        if edge.from_node == edge.to_node:
            critical.append({"type": "unsupported_self_loop", "edge_id": edge.id})
        for field_name, endpoint in (("from_node", edge.from_node), ("to_node", edge.to_node)):
            node = node_by_id.get(endpoint)
            if node is None:
                critical.append(
                    {
                        "type": "dangling_endpoint",
                        "edge_id": edge.id,
                        "field": field_name,
                        "node_id": endpoint,
                    }
                )
            elif node.kind not in _CONNECTABLE_KINDS:
                critical.append(
                    {
                        "type": "non_connectable_endpoint",
                        "edge_id": edge.id,
                        "field": field_name,
                        "node_id": endpoint,
                        "kind": node.kind,
                    }
                )

    partial_pages = sorted(
        index
        for index, status in graph.per_page_status.items()
        if status in {"partial", "error", "cost_exhausted"}
    )
    if partial_pages:
        warnings.append({"type": "partial_pages", "pages": [index + 1 for index in partial_pages]})

    pid_pages = [page for page in pages if page.role == "pid"]
    nodes_per_page = {
        page.page_index: sum(node.page_index == page.page_index for node in graph.nodes)
        for page in pid_pages
    }
    empty_pid_pages = [page + 1 for page, count in nodes_per_page.items() if count == 0]
    if empty_pid_pages:
        warnings.append({"type": "empty_pid_pages", "pages": empty_pid_pages})

    candidate_paths = sum(len(result.candidate_path_ids) for result in topology)
    used_paths = sum(len(result.used_path_ids) for result in topology)
    raster_warnings = [
        {"page": result.page_index + 1, "detail": warning}
        for result in topology
        for warning in result.warnings
    ]
    if raster_warnings:
        warnings.append({"type": "topology_warnings", "items": raster_warnings})

    inventory = native_text_inventory or build_native_text_inventory(pages=pages, graph=graph)
    inventory_summary = inventory.summary
    unresolved = [
        conflict
        for conflict in graph.conflicts
        if str(conflict.get("status", "unresolved")) in {"unresolved", "uncertain"}
    ]
    attached_node_ids = {
        endpoint for edge in graph.edges for endpoint in (edge.from_node, edge.to_node)
    }
    isolated_node_ids = [node.id for node in graph.nodes if node.id not in attached_node_ids]
    accepted_edges = [
        edge for edge in graph.edges if not edge.attributes.get("provisional_review_only")
    ]
    accepted_attached_ids = {
        endpoint for edge in accepted_edges for endpoint in (edge.from_node, edge.to_node)
    }
    accepted_isolated_count = sum(node.id not in accepted_attached_ids for node in graph.nodes)
    duplicate_tag_groups = _duplicate_tag_groups(graph)
    visual_style_counts = Counter(
        evidence.visual_style for result in topology for evidence in result.line_style_evidence
    )
    edge_type_counts = Counter(str(edge.line_type or "other") for edge in graph.edges)

    if unresolved:
        threshold = max(25, int(len(graph.nodes) * 0.25))
        warnings.append(
            {
                "type": (
                    "excessive_unresolved_ambiguities"
                    if len(unresolved) > threshold
                    else "unresolved_ambiguities"
                ),
                "count": len(unresolved),
                "threshold": threshold,
            }
        )
    if graph.dangling_opcs:
        warnings.append({"type": "dangling_opcs", "count": len(graph.dangling_opcs)})
    if inventory_summary.get("unresolved_tag_count", 0):
        warnings.append(
            {
                "type": "unresolved_native_tags",
                "count": inventory_summary["unresolved_tag_count"],
                "reviewable_tag_count": inventory_summary.get("reviewable_tag_count", 0),
            }
        )
    if duplicate_tag_groups:
        warnings.append(
            {
                "type": "duplicate_same_page_tags",
                "count": len(duplicate_tag_groups),
                "groups": duplicate_tag_groups[:50],
            }
        )
    isolated_ratio = len(isolated_node_ids) / len(graph.nodes) if graph.nodes else 0.0
    if isolated_ratio > 0.15:
        warnings.append(
            {
                "type": "excessive_isolated_nodes",
                "count": len(isolated_node_ids),
                "ratio": round(isolated_ratio, 4),
            }
        )

    status: QualityStatus = "pass"
    if critical:
        status = "error"
    elif warnings:
        status = "partial"

    return QualityReport(
        status=status,
        critical_violations=critical,
        warnings=warnings,
        coverage={
            "page_count": len(pages),
            "pid_page_count": len(pid_pages),
            "page_roles": {str(page.page_index): page.role for page in pages},
            "candidate_path_count": candidate_paths,
            "used_path_count": used_paths,
            "used_path_ratio": round(used_paths / candidate_paths, 4) if candidate_paths else 0.0,
            "native_tag_evidence_count": inventory_summary.get("reviewable_tag_count", 0),
            "assigned_native_text_count": inventory_summary.get("assigned_tag_count", 0),
            "native_tag_assignment_ratio": inventory_summary.get("assignment_ratio", 1.0),
            "native_text_inventory": inventory_summary,
            "derived_vector_style_counts": dict(sorted(visual_style_counts.items())),
        },
        metrics={
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "accepted_edge_count": len(accepted_edges),
            "provisional_edge_count": len(graph.edges) - len(accepted_edges),
            "accepted_isolated_node_count": accepted_isolated_count,
            "accepted_isolated_node_ratio": round(accepted_isolated_count / len(graph.nodes), 4)
            if graph.nodes else 0.0,
            "edge_type_counts": dict(sorted(edge_type_counts.items())),
            "dangling_opc_count": len(graph.dangling_opcs),
            "conflict_count": len(graph.conflicts),
            "unresolved_conflict_count": len(unresolved),
            "duplicate_tag_group_count": len(duplicate_tag_groups),
            "isolated_node_count": len(isolated_node_ids),
            "isolated_node_ratio": round(isolated_ratio, 4),
            "critical_violation_count": len(critical),
            "unresolved_native_tag_count": inventory_summary.get("unresolved_tag_count", 0),
        },
        unresolved_ambiguities=unresolved,
    )


def add_dexpi_results(
    report: QualityReport,
    *,
    stats: dict[str, Any],
    build_issues: list[str],
    validation_issues: list[dict[str, Any]],
) -> QualityReport:
    dropped = int(stats.get("dropped_edges", 0) or 0)
    report.dexpi = {
        "stats": stats,
        "build_issues": build_issues,
        "validation_issues": validation_issues,
        "dropped_edges": dropped,
    }
    if dropped:
        report.critical_violations.append({"type": "dexpi_dropped_edges", "count": dropped})
    if validation_issues:
        report.critical_violations.append(
            {"type": "dexpi_validation_errors", "count": len(validation_issues)}
        )
    if build_issues:
        report.warnings.append({"type": "dexpi_build_issues", "count": len(build_issues)})
    if report.critical_violations:
        report.status = "error"
    elif report.warnings:
        report.status = "partial"
    return report


def _duplicate_tag_groups(graph: ReconciledGraph) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str, tuple[str, ...]], list[str]] = {}
    for node in graph.nodes:
        label = re.sub(r"[^a-z0-9]+", "", node.label.casefold())
        if not label or not any(character.isdigit() for character in label):
            continue
        scope = tuple(sorted(a.id for a in graph.assemblies if node.id in a.member_node_ids))
        groups.setdefault((node.page_index, node.kind, label, scope), []).append(node.id)
    return [
        {
            "page": page_index + 1,
            "kind": kind,
            "normalised_label": label,
            "node_ids": sorted(node_ids),
        }
        for (page_index, kind, label, scope), node_ids in sorted(groups.items())
        if len(node_ids) > 1
    ]
