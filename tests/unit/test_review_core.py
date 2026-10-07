from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from diagex.review.core import (
    ReviewConflictError,
    ReviewIncompleteError,
    ReviewStore,
    _confirm_provisional_edge,
)
from diagex.review.render import prepare_page_assets
from diagex.vision.models import BBox, ReconciledEdge, ReconciledGraph, ReconciledNode


def test_human_confirmation_promotes_provisional_edge() -> None:
    edge = {
        "attributes": {
            "provisional_review_only": True,
            "requires_human_review": True,
            "review_conflict_type": "page_graph_uncertain_candidate",
            "review_reason": "uncertain",
            "original_line_type": "process",
        }
    }

    _confirm_provisional_edge(edge)

    assert edge["attributes"]["human_review_confirmed"] is True
    assert "provisional_review_only" not in edge["attributes"]
    assert edge["attributes"]["original_line_type"] == "process"


def _write_source(path: Path, *, pages: int = 1) -> None:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=240, height=160)
        page.insert_text((24, 30), f"P&ID page {index + 1}")
        page.draw_rect(fitz.Rect(30, 50, 90, 100))
    document.save(path)
    document.close()


def _write_run(path: Path, *, dangling: bool = False) -> Path:
    path.mkdir()
    graph = ReconciledGraph(
        source_path="sample.pdf",
        nodes=[
            ReconciledNode(
                id="n-a",
                kind="equipment",
                label="T-101",
                bbox_global=BBox(x=100, y=120, w=120, h=80),
                page_index=0,
                attributes={"equipment_class": "tank"},
                confidence="high",
            ),
            ReconciledNode(
                id="n-b",
                kind="equipment",
                label="P-101",
                bbox_global=BBox(x=500, y=120, w=100, h=80),
                page_index=0,
                attributes={"equipment_class": "pump"},
                confidence="medium",
            ),
        ],
        edges=[
            ReconciledEdge(
                id="e-a",
                from_node="n-a",
                to_node="missing" if dangling else "n-b",
                line_type="process",
                polyline_global=[(220, 160), (500, 160)],
                confidence="high",
                attributes={"line_id": "L-101"},
            )
        ],
        conflicts=[{"type": "ambiguous_opc", "node_id": "n-b"}],
        per_page_status={0: "partial"},
    )
    (path / "graph.json").write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    (path / "result.json").write_text(
        json.dumps({"dexpi_issues": ["node 'n-b': unclassified equipment"]}),
        encoding="utf-8",
    )
    return path


def _write_native_evidence(run: Path) -> None:
    evidence = run / "evidence"
    evidence.mkdir()
    (evidence / "page-0001.json").write_text(
        json.dumps(
            {
                "page_index": 0,
                "role": "pid",
                "text_spans": [
                    {
                        "id": "txt-tank",
                        "text": "T-101",
                        "bbox": {"x": 100, "y": 100, "w": 50, "h": 20},
                        "block_index": 0,
                        "line_index": 0,
                        "word_index": 0,
                    },
                    {
                        "id": "txt-pt-prefix",
                        "text": "PT",
                        "bbox": {"x": 300, "y": 100, "w": 20, "h": 20},
                        "block_index": 1,
                        "line_index": 0,
                        "word_index": 0,
                    },
                    {
                        "id": "txt-pt-number",
                        "text": "102",
                        "bbox": {"x": 325, "y": 100, "w": 35, "h": 20},
                        "block_index": 1,
                        "line_index": 0,
                        "word_index": 1,
                    },
                    {
                        "id": "txt-line",
                        "text": "50-GA-00302-M15B-N",
                        "bbox": {"x": 100, "y": 300, "w": 150, "h": 20},
                        "block_index": 2,
                        "line_index": 0,
                        "word_index": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def review(tmp_path: Path) -> ReviewStore:
    source = tmp_path / "source.pdf"
    _write_source(source)
    run = _write_run(tmp_path / "run")
    return ReviewStore.open(run, source_path=source, rater="Ada", out_dir=tmp_path / "review")


def _action(
    store: ReviewStore,
    target_type: str,
    target_id: str,
    operation: str,
    after: dict | None = None,
    reason: str = "",
) -> dict:
    return store.append_action(
        {
            "expected_revision": store.state["revision"],
            "target_type": target_type,
            "target_id": target_id,
            "operation": operation,
            "after": after or {},
            "reason": reason,
        }
    )


def test_session_renders_both_frames_and_prioritises_risk(review: ReviewStore):
    state = review.public_state()
    assert (review.out_dir / "pages" / "p0000.source.png").is_file()
    assert (review.out_dir / "pages" / "p0000.inference.png").is_file()
    assert state["queue"][0]["tier"] == 0
    assert "partial page" in state["queue"][0]["reasons"]
    assert state["session"]["build_issues"][0]["target_id"] == "n-b"
    assert "solenoid" in state["taxonomy"]["actuation_types"]


def test_public_conflicts_include_compact_review_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    _write_source(source)
    run = _write_run(tmp_path / "run")
    graph_path = run / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["conflicts"] = [
        {
            "type": "ambiguous_opc_evidence",
            "node_id": "n-b",
            "edge_id": "e-a",
            "labels": ["P-101", "P-101A"],
            "kinds": {"equipment": 2, "opc": 1},
            "candidate_pairs": [
                {
                    "node_ids": ["n-a", "n-b"],
                    "score": 0.86,
                    "left_service": "feed",
                    "right_service": "feed",
                }
            ],
        }
    ]
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    store = ReviewStore.open(
        run,
        source_path=source,
        rater="Ada",
        out_dir=tmp_path / "review",
    )

    conflict = next(iter(store.public_state()["reviews"]["conflicts"].values()))
    candidates = conflict["candidates"]
    assert [node["id"] for node in candidates["nodes"]] == ["n-b", "n-a"]
    assert [edge["id"] for edge in candidates["edges"]] == ["e-a"]
    assert candidates["labels"] == ["P-101", "P-101A"]
    assert candidates["kinds"] == ["equipment", "opc"]
    assert candidates["pairs"][0]["score"] == 0.86
    assert (
        "candidates"
        not in store.state["conflict_reviews"][next(iter(store.state["conflict_reviews"]))]
    )


def test_conflict_candidates_resolve_detection_provenance_to_visible_node(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    _write_source(source)
    run = _write_run(tmp_path / "run")
    graph_path = run / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["nodes"][1]["source_annotation_ids"] = ["det-pump"]
    graph["conflicts"] = [
        {
            "type": "tag_kind_conflict",
            "detection_id": "det-pump",
            "page_index": 0,
            "status": "unresolved",
        }
    ]
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    store = ReviewStore.open(
        run,
        source_path=source,
        rater="Ada",
        out_dir=tmp_path / "review",
    )

    conflict = next(iter(store.public_state()["reviews"]["conflicts"].values()))
    assert [node["id"] for node in conflict["candidates"]["nodes"]] == ["n-b"]


def test_resolved_model_conflict_does_not_require_human_disposition(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    _write_source(source)
    run = _write_run(tmp_path / "run")
    graph_path = run / "graph.json"
    graph = json.loads(graph_path.read_text())
    graph["conflicts"][0]["status"] = "resolved"
    graph["conflicts"][0]["reason"] = "deterministically excluded"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    store = ReviewStore.open(
        run,
        source_path=source,
        rater="Ada",
        out_dir=tmp_path / "review",
    )

    conflict = next(iter(store.state["conflict_reviews"].values()))
    assert conflict["status"] == "resolved"
    assert store.completion()["unresolved_conflicts"] == []


def test_actions_are_persistent_and_stale_revisions_are_rejected(review: ReviewStore):
    _action(review, "page", "0", "approve", {"role": "pid"})
    _action(review, "node", "n-a", "approve")
    _action(review, "node", "n-b", "modify", {"label": "P-101A"})

    with pytest.raises(ReviewConflictError, match="stale review revision"):
        review.append_action(
            {
                "expected_revision": 0,
                "target_type": "edge",
                "target_id": "e-a",
                "operation": "approve",
            }
        )

    resumed = ReviewStore.open(
        review.graph_path,
        source_path=review.source_path,
        rater="Ada",
        out_dir=review.out_dir,
    )
    assert resumed.state["revision"] == 3
    assert next(n for n in resumed.state["graph"]["nodes"] if n["id"] == "n-b")["label"] == "P-101A"
    assert len((review.out_dir / "events.jsonl").read_text().splitlines()) == 3


def test_native_text_inventory_links_matches_and_audits_missing_tags(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _write_source(source)
    run = _write_run(tmp_path / "run")
    _write_native_evidence(run)
    store = ReviewStore.open(run, source_path=source, rater="Ada", out_dir=tmp_path / "review")

    state = store.public_state()
    candidates = state["inventory"]["evidence"]
    tank = next(item for item in candidates if item["normalised_text"] == "T101")
    missing = next(item for item in candidates if item["normalised_text"] == "PT102")
    line = next(item for item in candidates if item["candidate_kind"] == "line_number")
    assert tank["review"]["status"] == "linked"
    assert tank["review"]["linked_node_id"] == "n-a"
    assert missing["review"]["status"] == "unreviewed"
    assert line["blocking"] is False
    assert store.completion()["unreviewed_evidence"] == [missing["id"]]

    _action(
        store,
        "evidence",
        missing["id"],
        "link",
        {"node_id": "n-b"},
        "same printed tag",
    )
    assert store.completion()["unreviewed_evidence"] == []
    assert store.state["evidence_reviews"][missing["id"]]["linked_node_id"] == "n-b"

    _action(
        store,
        "evidence",
        line["id"],
        "dismiss",
        {"disposition": "line_number"},
    )
    resumed = ReviewStore.open(run, source_path=source, rater="Ada", out_dir=tmp_path / "review")
    assert resumed.state["evidence_reviews"][line["id"]]["status"] == "dismissed"


def test_reject_and_undo_append_a_compensating_event(review: ReviewStore):
    _action(review, "node", "n-a", "reject", reason="false positive")
    assert review.state["node_reviews"]["n-a"] == "rejected"
    review.append_action({"expected_revision": 1, "operation": "undo"})
    assert review.state["node_reviews"]["n-a"] == "unreviewed"
    events = [json.loads(line) for line in review.events_path.read_text().splitlines()]
    assert events[-1]["operation"] == "undo"
    assert events[-1]["undo_of"] == 1


def test_add_node_and_edge_then_edit_geometry(review: ReviewStore):
    added_node = _action(
        review,
        "node",
        "",
        "add",
        {
            "kind": "instrument",
            "label": "PI-101",
            "bbox_global": {"x": 300, "y": 250, "w": 60, "h": 60},
            "page_index": 0,
            "attributes": {"instrument_function": "indicator"},
            "confidence": "medium",
        },
    )["event"]["target_id"]
    added_edge = _action(
        review,
        "edge",
        "",
        "add",
        {
            "from_node": "n-a",
            "to_node": added_node,
            "line_type": "signal_electric",
            "polyline_global": [[160, 160], [330, 280]],
            "confidence": "medium",
        },
    )["event"]["target_id"]
    _action(review, "edge", added_edge, "modify", {"polyline_global": [[160, 160], [330, 260]]})
    graph = review.state["graph"]
    assert next(n for n in graph["nodes"] if n["id"] == added_node)["label"] == "PI-101"
    assert next(e for e in graph["edges"] if e["id"] == added_edge)["polyline_global"][-1] == [
        330,
        260,
    ]
    assert review.state["node_reviews"][added_node] == "modified"


def test_completion_gate_and_export(review: ReviewStore):
    with pytest.raises(ReviewIncompleteError):
        review.finish()
    _action(review, "page", "0", "approve", {"role": "pid"})
    _action(review, "node", "n-a", "approve")
    _action(review, "node", "n-b", "approve")
    _action(review, "edge", "e-a", "approve")
    conflict_id = next(iter(review.state["conflict_reviews"]))
    _action(review, "conflict", conflict_id, "waive", reason="source is ambiguous")

    assert review.completion()["complete"] is True
    report = review.finish()
    assert report["finished"] is True
    assert (review.out_dir / "graph.reviewed.json").is_file()
    assert (review.out_dir / "pid.reviewed.dexpi.json").is_file()
    assert (review.out_dir / "pid.reviewed.dexpi.xml").is_file()
    assert json.loads((review.out_dir / "review.report.json").read_text())["finished"] is True


def test_dangling_active_edge_blocks_completion(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _write_source(source)
    run = _write_run(tmp_path / "run", dangling=True)
    store = ReviewStore.open(run, source_path=source, rater="Ada")
    _action(store, "page", "0", "approve", {"role": "pid"})
    _action(store, "node", "n-a", "approve")
    _action(store, "node", "n-b", "approve")
    _action(store, "edge", "e-a", "approve")
    conflict_id = next(iter(store.state["conflict_reviews"]))
    _action(store, "conflict", conflict_id, "resolve")
    assert store.completion()["dangling_edges"] == ["e-a"]


def test_malformed_audit_line_is_reported_without_losing_valid_events(review: ReviewStore):
    _action(review, "node", "n-a", "approve")
    with review.events_path.open("a", encoding="utf-8") as handle:
        handle.write("{broken")
    resumed = ReviewStore.open(
        review.graph_path,
        source_path=review.source_path,
        rater="Ada",
        out_dir=review.out_dir,
    )
    assert resumed.state["revision"] == 1
    assert any("malformed" in warning for warning in resumed.state["warnings"])
    _action(resumed, "node", "n-b", "approve")
    lines = resumed.events_path.read_text().splitlines()
    assert json.loads(lines[-1])["revision"] == 2


def test_source_or_graph_hash_mismatch_refuses_resume(review: ReviewStore):
    review.source_path.write_bytes(review.source_path.read_bytes() + b"changed")
    with pytest.raises(ReviewConflictError, match="source diagram changed"):
        ReviewStore.open(
            review.graph_path,
            source_path=review.source_path,
            rater="Ada",
            out_dir=review.out_dir,
        )


def test_page_assets_support_multipage_and_persisted_coordinate_frames(tmp_path: Path):
    source = tmp_path / "source.pdf"
    _write_source(source, pages=2)
    inferred = prepare_page_assets(source, tmp_path / "inferred")
    assert [page["page_index"] for page in inferred] == [0, 1]

    expected = {
        "scan": {"deskew": False, "contrast": False, "despeckle": False},
        "pages": [
            {
                "page_index": 0,
                "width": 240,
                "height": 160,
                "dpi": 72.0,
                "effective_dpi": 72.0,
                "is_scanned": False,
                "rotation_deg": 0.0,
                "source_ref": "source#page=1",
            },
            {
                "page_index": 1,
                "width": 240,
                "height": 160,
                "dpi": 72.0,
                "effective_dpi": 72.0,
                "is_scanned": False,
                "rotation_deg": 0.0,
                "source_ref": "source#page=2",
            },
        ],
    }
    restored = prepare_page_assets(source, tmp_path / "restored", expected=expected)
    assert [(page["width"], page["height"], page["dpi"]) for page in restored] == [
        (240, 160, 72.0),
        (240, 160, 72.0),
    ]


@pytest.mark.parametrize("operation", ["approve", "reject", "modify"])
def test_connection_decision_resolves_linked_conflicts_and_undo_restores_them(review, operation):
    linked = {
        "status": "unreviewed",
        "conflict": {
            "type": "unsupported_vector_route",
            "edge_id": "e-a",
            "node_ids": ["n-a", "n-b"],
        },
    }
    review.state["conflict_reviews"]["linked"] = linked
    # Unrelated identity conflicts must remain independent of connection review.
    unrelated = next(iter(review.state["conflict_reviews"]))
    before = review.public_state()
    assert before["reviews"]["conflicts"]["linked"]["decision_target"] == {
        "type": "edge",
        "id": "e-a",
    }
    assert (
        "graph conflict" in next(r for r in before["queue"] if r["target_id"] == "n-a")["reasons"]
    )
    _action(review, "edge", "e-a", operation)
    assert review.state["conflict_reviews"]["linked"]["status"] == "resolved"
    assert review.state["conflict_reviews"][unrelated]["status"] == "unreviewed"
    assert (
        "graph conflict"
        not in next(r for r in review._queue() if r["target_id"] == "n-a")["reasons"]
    )
    _action(review, "edge", "e-a", "undo")
    assert review.state["conflict_reviews"]["linked"] == linked
    assert (
        "graph conflict" in next(r for r in review._queue() if r["target_id"] == "n-a")["reasons"]
    )


def test_grouped_connection_requires_choice_and_exports_only_chosen_route(review):
    import copy

    edge = review.state["graph"]["edges"][0]
    alternative = copy.deepcopy(edge)
    alternative.update(
        id="alternate",
        polyline_global=[[220, 160], [220, 200], [500, 160]],
        source_evidence_ids=["alternate-ink"],
    )
    edge["attributes"].update(
        provisional_review_only=True, route_alternatives=[copy.deepcopy(edge), alternative]
    )
    with pytest.raises(ValueError, match="select a route"):
        _action(review, "edge", "e-a", "approve")
    _action(review, "edge", "e-a", "approve", {"route_candidate_id": "alternate"})
    chosen = review.reviewed_graph().edges[0]
    assert chosen.polyline_global == [(220, 160), (220, 200), (500, 160)]
    assert chosen.source_evidence_ids == ["alternate-ink"]
    assert chosen.attributes["human_review_confirmed"]
    assert len(review.reviewed_graph().edges) == 1
    _action(review, "edge", "e-a", "undo")
    assert review.state["graph"]["edges"][0]["polyline_global"] == [[220, 160], [500, 160]]


def test_reviewed_legacy_edge_does_not_hide_unresolved_conflict(review):
    review.state["edge_reviews"]["e-a"] = "approved"
    review.state["conflict_reviews"]["legacy"] = {
        "status": "unreviewed",
        "conflict": {
            "type": "unsupported_vector_route",
            "edge_id": "e-a",
        },
    }
    assert "decision_target" not in review.public_state()["reviews"]["conflicts"]["legacy"]


def test_assembly_identity_review_is_atomic_and_does_not_create_pipe_objects(tmp_path):
    source = tmp_path / "source.pdf"
    _write_source(source)
    run = _write_run(tmp_path / "run")
    raw = json.loads((run / "graph.json").read_text())
    raw["assemblies"] = [
        {
            "id": "assembly-a",
            "page_index": 0,
            "bbox_global": {"x": 80, "y": 100, "w": 540, "h": 180},
            "label": None,
            "label_candidates": ["K-101", "K-102"],
            "member_node_ids": ["n-a", "n-b"],
            "status": "conflicting",
        }
    ]
    raw["text_bindings"] = [{"assembly_ids": ["assembly-a"], "node_ids": []}]
    raw["conflicts"] = [
        {
            "type": "assembly_identity_uncertainty",
            "assembly_id": "assembly-a",
            "node_ids": ["n-a", "n-b"],
            "labels": ["K-101", "K-102"],
        }
    ]
    (run / "graph.json").write_text(json.dumps(raw))
    store = ReviewStore.open(run, source_path=source, rater="Ada", out_dir=tmp_path / "review")
    cid = next(iter(store.state["conflict_reviews"]))
    assert store.completion()["unreviewed_assemblies"] == ["assembly-a"]
    assert (
        store.public_state()["reviews"]["conflicts"][cid]["decision_target"]["type"] == "assembly"
    )
    with pytest.raises(ValueError, match="choose an assembly identity"):
        _action(store, "assembly", "assembly-a", "approve")
    _action(store, "assembly", "assembly-a", "modify", {"label": "K-101"})
    assert store.state["conflict_reviews"][cid]["status"] == "resolved"
    assert store.state["node_reviews"] == {"n-a": "unreviewed", "n-b": "unreviewed"}
    reopened = ReviewStore.open(run, source_path=source, rater="Ada", out_dir=tmp_path / "review")
    assert reopened.state["graph"]["assemblies"][0]["label"] == "K-101"
    _action(reopened, "assembly", "assembly-a", "undo")
    assert reopened.state["graph"]["assemblies"][0]["label"] is None
    assert reopened.state["conflict_reviews"][cid]["status"] == "unreviewed"
    _action(reopened, "assembly", "assembly-a", "reject")
    rejected = reopened.reviewed_graph()
    assert not rejected.assemblies and not rejected.text_bindings[0]["assembly_ids"]
    assert len(rejected.nodes) == 2
    _action(reopened, "assembly", "assembly-a", "undo")
    _action(reopened, "assembly", "assembly-a", "modify", {"label": "K-102"})
    for nid in ["n-a", "n-b"]:
        _action(reopened, "node", nid, "approve")
    _action(reopened, "edge", "e-a", "approve")
    _action(reopened, "page", "0", "approve", {"role": "pid"})
    assert reopened.finish()["finished"]
    exported = json.loads((reopened.out_dir / "graph.reviewed.json").read_text())
    assert len(exported["nodes"]) == 2 and len(exported["edges"]) == 1
    assert exported["assemblies"][0]["label"] == "K-102"
    assert "assembly-a" not in (reopened.out_dir / "pid.reviewed.dexpi.xml").read_text()


def _write_review_findings(review: ReviewStore, records: list[dict]) -> Path:
    path = review.graph_path.parent / 'review-findings.json'
    path.write_text(json.dumps({
        'graph_sha256': review.session['graph_sha256'],
        'source_sha256': review.session['source_sha256'],
        'findings': records,
    }), encoding='utf-8')
    return path


def test_imported_findings_keep_review_history_and_human_gate(review: ReviewStore):
    records = [{'id': 'source-label', 'conflict': {
        'type': 'source_identifier_conflict', 'title': 'T-101 or T-102?',
        'question': 'Which printed label governs?', 'page_index': 0,
        'source_locations': [{'page_index': 0, 'bbox_global': {'x': 20, 'y': 30, 'w': 50, 'h': 20}, 'label': 'T-102'}],
    }}]
    path = _write_review_findings(review, records)
    def reopen():
        return ReviewStore.open(review.graph_path, source_path=review.source_path,
                                rater='Ada', out_dir=review.out_dir)
    current = reopen()
    state = current.public_state()
    assert state['findings'][0]['target_id'] == 'audit-source-label'
    assert state['findings'][0]['category'] == 'identifiers'
    assert set(state['reviews']['nodes'].values()) == {'unreviewed'}
    assert not state['completion']['complete']
    with pytest.raises(ValueError, match='record a decision'):
        _action(current, 'conflict', 'audit-source-label', 'resolve')
    _action(current, 'conflict', 'audit-source-label', 'resolve', reason='Retain T-101 per owner clarification')
    current = reopen()
    assert current.state['conflict_reviews']['audit-source-label']['reason'] == 'Retain T-101 per owner clarification'
    assert all(row['target_id'] != 'audit-source-label' for row in current.public_state()['findings'])
    assert len(current.session['conflicts']) == 2
    _action(current, '', '', 'undo')
    assert current.public_state()['findings'][0]['target_id'] == 'audit-source-label'
    records[0]['conflict']['title'] = 'Changed meaning must get a new ID'
    _write_review_findings(current, records)
    with pytest.raises(ReviewConflictError, match='changed'):
        reopen()
    payload = json.loads(path.read_text())
    payload['graph_sha256'] = 'wrong'
    path.write_text(json.dumps(payload))
    with pytest.raises(ReviewConflictError, match='do not match'):
        reopen()


def test_findings_group_connection_warnings_and_exclude_routine_signoff(review: ReviewStore):
    graph = review.state['graph']
    conflict = {'type': 'endpoint_role_uncertain', 'edge_id': 'e-a', 'page_index': 0, 'reason': 'Check line meaning'}
    for key in ['warning-one', 'warning-two']:
        review.state['conflict_reviews'][key] = {'conflict': conflict, 'status': 'unreviewed', 'reason': None}
    graph['edges'][0]['attributes']['provisional_review_only'] = True
    state = review.public_state()
    edge_rows = [row for row in state['findings'] if row['target_id'] == 'e-a']
    assert len(edge_rows) == 1
    assert edge_rows[0]['conflict_ids'] == ['warning-one', 'warning-two']
    assert edge_rows[0]['category'] == 'connections'
    assert all(row['target_type'] != 'node' for row in state['findings'])
    assert len(state['queue']) == 3  # Ordinary objects remain available for full sign-off.
    _action(review, 'edge', 'e-a', 'approve')
    assert all(row['target_id'] != 'e-a' for row in review.public_state()['findings'])
