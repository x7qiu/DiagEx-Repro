"""Read-only projection of saved extraction artifacts. No review/session writes."""
from __future__ import annotations

import io
import json
from pathlib import Path

from diagex.knowledge.sources import SOURCES

from .evidence_origin import evidence_origin, recognition_diagnostic

ARTIFACTS = (
    "graph.json", "detection.json", "legend.json", "result.json", "cost.json",
    "quality.report.json", "hypotheses.json", "knowledge.stages.json",
    "pid.dexpi.json", "pid.dexpi.xml", "pid.drawio", "pid.svg", "debug_report.md",
    "review/graph.reviewed.json",
)


def artifact(run: Path, name: str) -> Path:
    path = (run / name).resolve()
    if not path.is_relative_to(run):
        raise ValueError("Artifact is outside the selected run")
    return path


def read(run: Path, name: str) -> dict:
    path = artifact(run, name)
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Invalid saved object: {name}")
    return value


def run_path(workbench, value):
    root = Path(workbench.config.runs_dir).expanduser().resolve()
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Select a run inside the configured runs directory")
    if not any(artifact(path, name).is_file() for name in ("graph.json", "detection.json")):
        raise ValueError("No saved graph or detections in this run")
    return path


def source_for(workbench, run, graph):
    source, verified = workbench._resolve_run_source(run, graph)
    if source:
        return source, verified
    # Compatibility reader only: old source links do not recreate review state.
    session = read(run, "review/session.json")
    value = session.get("source_path")
    if value:
        candidate = Path(value).expanduser()
        expected = workbench._expected_source_hash(run) or session.get("source_sha256")
        if candidate.is_file() and candidate.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            if expected and workbench._source_hash(candidate) == expected:
                return candidate, True
    return None, None


def page_specs(run, detection, source):
    specs = {}
    # More precise evidence takes precedence over summaries.
    for record in [*read(run, "review/session.json").get("pages", []),
                   *detection.get("pages", []), *read(run, "pages.json").get("pages", [])]:
        if "page_index" in record:
            specs[int(record["page_index"])] = record
    for path in sorted((run / "evidence").glob("page-*.json")):
        record = read(run, str(path.relative_to(run)))
        if "page_index" in record:
            specs[int(record["page_index"])] = record
    if source:
        import fitz
        with fitz.open(source) as doc:
            for index, page in enumerate(doc):
                if index not in specs:
                    specs[index] = {"page_index": index, "width": page.rect.width,
                                    "height": page.rect.height, "geometry_inferred": True}
    return [{k: record.get(k) for k in (
        "page_index", "width", "height", "role", "rotation_deg", "geometry_inferred", "native_coordinate_frame"
    )} for _, record in sorted(specs.items())]


def details(workbench, value, version="extraction"):
    run = run_path(workbench, value)
    graph = read(run, "graph.json")
    detection = read(run, "detection.json")
    state = read(run, "review/state.json")
    saved_detection = read(run, "review/detection.json")
    saved_graph = read(run, "review/graph.reviewed.json") or state.get("graph", {})
    versions = [{"id": "extraction", "label": "原始提取"}]
    if saved_graph or saved_detection:
        versions.append({"id": "saved", "label": "历史保存的修改"})
    if version not in {v["id"] for v in versions}:
        raise ValueError("Unknown saved version")
    source, verified = source_for(workbench, run, graph)
    pages = page_specs(run, detection, source)
    legend = read(run, "legend.json") or detection.get("legend_pack", {})
    nodes, edges = graph.get("nodes", []), graph.get("edges", [])
    findings = list(graph.get("conflicts", [])) + list(graph.get("dangling_opcs", []))
    if version == "saved" and saved_graph:
        nodes, edges = saved_graph.get("nodes", []), saved_graph.get("edges", [])
        findings = list(saved_graph.get("conflicts", [])) + list(saved_graph.get("dangling_opcs", []))
    elif version == "saved":
        edges = []
        nodes = [{**row["detection"], "saved_status": row.get("status"), "saved_reason": row.get("reason")}
                 for row in saved_detection.get("symbols", []) if row.get("detection")]
        legend = {"entries": [{**row["entry"], "saved_status": row.get("status")}
                              for row in saved_detection.get("legends", []) if row.get("entry")]}
    elif not graph:
        nodes = list(detection.get("detections", []))
        for index, row in enumerate(detection.get("reviews", [])):
            if row.get("promoted_detection_id"):
                continue
            if row.get("bbox") and row.get("object"):
                nodes.append({**row["object"], "id": f"observation-{index}", "bbox": row["bbox"],
                              "page_index": row.get("page_index", 0), "candidate_only": True,
                              "saved_status": row.get("status"), "reason": row.get("reason"),
                              "legend_interpretation": row.get("legend_interpretation"),
                              "source_candidate_absent": not bool(row.get("candidate_id"))})
            elif row.get("status") in {"unreviewed", "uncertain", "unresolved"}:
                findings.append(row)
        reviews_by_candidate = {r.get("candidate_id"): r for r in detection.get("reviews", []) if r.get("candidate_id")}
        represented = {n.get("attributes", {}).get("symbol_candidate_id") for n in nodes}
        for candidate in detection.get("candidates", []):
            if candidate.get("id") not in represented:
                nodes.append({**candidate, "id": "candidate-" + candidate["id"],
                              "kind": "candidate", "candidate_only": True,
                              "saved_status": reviews_by_candidate.get(candidate["id"], {}).get("status"),
                              "reason": reviews_by_candidate.get(candidate["id"], {}).get("reason"),
                              "supplied_knowledge": reviews_by_candidate.get(candidate["id"], {}).get("supplied_knowledge")})
    if version == "saved" and saved_detection.get("legends"):
        legend = {"entries": [{**row["entry"], "saved_status": row.get("status")}
                              for row in saved_detection["legends"] if row.get("entry")]}
    result = read(run, "result.json")
    knowledge = read(run, "workbench.json").get("settings", {}).get("knowledge") or read(run, "knowledge.snapshot.json")
    selected_sources = knowledge.get("selected_sources")
    quality = read(run, "quality.report.json")
    hypotheses = read(run, "hypotheses.json").get("hypotheses", [])
    findings.extend(hypotheses)
    findings.extend(result.get("dexpi_issues", []))
    findings.extend(result.get("validation_issues", []))
    if result.get("stop_reason"):
        findings.insert(0, {"type": "extraction_stopped", "reason": result["stop_reason"]})
    # Without source geometry, expose values but never pretend guessed dimensions are exact.
    for index in sorted({int(n.get("page_index", 0)) for n in nodes} - {p["page_index"] for p in pages}):
        boxes = [n.get("bbox_global", n.get("bbox", {})) for n in nodes if n.get("page_index", 0) == index]
        pages.append({"page_index": index, "width": max([b.get("x", 0)+b.get("w", 0) for b in boxes]+[100]),
                      "height": max([b.get("y", 0)+b.get("h", 0) for b in boxes]+[100]), "geometry_inferred": True})
    return {
        "run_dir": str(run), "name": run.parent.name + " · " + run.name,
        "version": version, "versions": versions, "read_only": True,
        "summary": {k: result.get(k) for k in ("run_id", "model", "engine", "workflow_stage", "quality_status", "stats")},
        "source_available": source is not None, "source_verified": verified,
        "source_filename": source.name if source else None,
        "pages": sorted(pages, key=lambda p: p["page_index"]), "nodes": nodes, "edges": edges,
        "legend": legend.get("entries", []), "findings": findings, "quality": quality,
        "diagnostics": {"nodes": [recognition_diagnostic(n) for n in nodes],
                        "findings": [recognition_diagnostic(n) if isinstance(n, dict) else None for n in findings]},
        "evidence_origins": {"nodes": [evidence_origin(n) for n in nodes],
                             "legend": [evidence_origin(e, is_legend=True) for e in legend.get("entries", [])]},
        "knowledge_sources": selected_sources,
        "knowledge_source_names": [SOURCES.get(s, {}).get("name", s) for s in selected_sources] if selected_sources is not None else None,
        "downloads": [name for name in ARTIFACTS if artifact(run, name).is_file()],
    }


def page_image(workbench, value, index):
    import fitz
    from PIL import Image

    run = run_path(workbench, value)
    source, _ = source_for(workbench, run, read(run, "graph.json"))
    if source is None:
        raise ValueError("Original drawing is unavailable; saved extraction data is still viewable")
    pages = page_specs(run, read(run, "detection.json"), source)
    spec = next((p for p in pages if p["page_index"] == index), None)
    if spec is None or index < 0:
        raise ValueError("Unknown drawing page")
    with fitz.open(source) as doc:
        if index >= len(doc):
            raise ValueError("Page is missing from the original drawing")
        page = doc[index]
        scale = min(3, 2800 / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        if not spec.get("rotation_deg"):
            return pix.tobytes("png")
        # Extraction deskew uses a rotated raster frame; don't overlay on unrotated source.
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        image = image.rotate(float(spec["rotation_deg"]), resample=Image.BICUBIC, expand=False, fillcolor="white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def download(workbench, value, name):
    if name not in ARTIFACTS:
        raise ValueError("Unknown extraction artifact")
    path = artifact(run_path(workbench, value), name)
    if not path.is_file():
        raise ValueError("Artifact is unavailable")
    return path.read_bytes(), path.name
