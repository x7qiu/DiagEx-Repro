import copy

import fitz
import pytest

from diagex.knowledge.evaluate import evaluate
from diagex.knowledge.models import Context, Entry
from diagex.knowledge.profiles import ProfileStore, suggest
from diagex.knowledge.resolver import digest, knowledge_snapshot, reference_images, resolve
from diagex.vision.evidence import PageEvidence, PathEvidence, TextEvidence
from diagex.vision.fusion import _prepare_detections
from diagex.vision.models import BBox
from diagex.vision.symbol_interpretation import DetectionRecord


def profile(**context):
    return dict(
        id="utilities",
        name="Utilities",
        version=1,
        confirmed=True,
        context=Context(**context).model_dump(),
    )


def fixture(vertical=False, text="TO SHEET A-2", attached=True, glyph=True):
    points = [(400, 300), (440, 300), (450, 310), (440, 320), (400, 320), (400, 300)]
    line = [(250, 310), (400, 310)]
    if vertical:
        points = [(y + 90, x - 100) for x, y in points]
        line = [(y + 90, x - 100) for x, y in line]

    def path(identity, pts, closed):
        xs, ys = zip(*pts, strict=True)
        return PathEvidence(
            id=identity,
            page_index=0,
            points=pts,
            bbox=BBox(
                x=min(xs), y=min(ys), w=max(1, max(xs) - min(xs)), h=max(1, max(ys) - min(ys))
            ),
            origin="pdf_vector",
            primitive="line",
            closed=closed,
        )

    outline = path("glyph", points, True)
    page = PageEvidence(
        page_index=0,
        source_ref="synthetic",
        width=1000,
        height=800,
        dpi=100,
        effective_dpi=100,
        is_scanned=False,
        paths=([outline] if glyph else []) + ([path("pipe", line, False)] if attached else []),
        text_spans=[TextEvidence(id="label", text=text, bbox=BBox(x=400, y=350, w=100, h=15))]
        if text
        else [],
    )
    detection = DetectionRecord(
        id="candidate",
        page_index=0,
        tile_id="t",
        kind="opc",
        label="off-page connector",
        bbox=outline.bbox,
        confidence="medium",
    )
    return page, detection


def test_profile_revisions_confirmation_and_stale_write(tmp_path):
    store = ProfileStore(tmp_path)
    body = dict(
        name="Known project",
        context={"industry": "chemical", "unit_types": ["utilities", "storage"]},
    )
    with pytest.raises(ValueError, match="Confirm"):
        store.save(body)
    first = store.save({**body, "confirmed": True})
    assert first["version"] == 1 and first["context"]["standards"] == []
    second = store.save({**first, "expected_version": 1, "context": {"unit_types": ["utilities"]}})
    assert second["version"] == 2
    assert store.get(first["id"], 1) == first
    assert store.list() == [second]
    with pytest.raises(ValueError, match="changed"):
        store.save({**first, "expected_version": 1})
    with pytest.raises(ValueError):
        store.get("../escape", 1)


def test_suggestions_quote_native_evidence_without_guessing(tmp_path):
    path = tmp_path / "source.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text(
            (40, 40), "Industry: Chemical\nProcess unit: Utilities\nDrawing standard: ISA 5.1: 2024"
        )
        doc.save(path)
    suggestions = suggest(path)["suggestions"]
    assert {s["field"] for s in suggestions} == {"industry", "unit_types", "standards"}
    assert all(s["page"] == 1 and s["passage"] and s["source_sha256"] for s in suggestions)
    assert suggestions[-1]["value"] == [{"name": "ISA 5.1", "edition": "2024"}]
    snapshot = knowledge_snapshot("profile", profile(industry="chemical"))
    assert resolve(snapshot, "review")["context"]["standards"] == []


def test_scoped_retrieval_overrides_unknown_context_and_conflicts():
    snapshot = knowledge_snapshot("profile", profile(industry="chemical", unit_types=["utilities"]))
    entry = copy.deepcopy(next(e for e in snapshot["entries"] if e["id"] == "opc.placement-exceptions"))
    entry.update(
        id="utility.definition",
        kind="definition",
        definition_key="arrow",
        definition_value="connector",
    )
    entry["applicability"]["unit_types"] = ["utilities"]
    second = copy.deepcopy(entry)
    second.update(id="utility.conflict", definition_value="flow indicator")
    snapshot["entries"] += [entry, second]
    ctx = resolve(snapshot, "symbol_interpretation", query="arrow")
    assert ctx["conflicts"] and {entry["id"], second["id"]} <= set(ctx["reference_ids"])
    snapshot["overrides"] = [{"pages": [2], "context": {"unit_types": ["storage"]}}]
    assert not resolve(snapshot, "symbol_interpretation", 1, "arrow")["conflicts"]
    assert resolve(snapshot, "symbol_interpretation", 0, "arrow")["conflicts"]
    snapshot["profile"]["context"] = Context().model_dump()
    assert not resolve(snapshot, "symbol_interpretation", 0, "arrow")["conflicts"]


def test_standard_edition_requires_explicit_match():
    snapshot = knowledge_snapshot("profile", profile(industry="chemical"))
    entry = copy.deepcopy(next(e for e in snapshot["entries"] if e["id"] == "opc.placement-exceptions"))
    entry["id"] = "edition.rule"
    entry["applicability"]["standards"] = [{"name": "XYZ", "edition": "2024"}]
    snapshot["entries"].append(entry)
    assert entry["id"] not in resolve(snapshot, "connections", query="connector")["reference_ids"]
    snapshot["profile"]["context"]["standards"] = [{"name": "XYZ", "edition": "2024"}]
    assert entry["id"] in resolve(snapshot, "connections", query="connector")["reference_ids"]


@pytest.mark.parametrize("vertical", [False, True])
def test_supported_exception_retained_as_uncertain(vertical):
    page, d = fixture(vertical=vertical)
    baseline, _ = _prepare_detections([d], {0: page})
    kept, findings = _prepare_detections([d], {0: page}, knowledge=knowledge_snapshot("general"))
    assert not baseline
    assert len(kept) == 1 and kept[0].attributes["requires_human_review"]
    assert findings[0]["glyph_path_ids"] == ["glyph"]
    assert "knowledge_exception" not in d.attributes  # inputs immutable


@pytest.mark.parametrize(
    "kwargs", [{"text": ""}, {"text": "FLOW"}, {"attached": False}, {"glyph": False}]
)
def test_utility_context_does_not_rescue_unsupported_arrow(kwargs):
    page, d = fixture(**kwargs)
    accepted, _ = _prepare_detections(
        [d], {0: page}, knowledge=knowledge_snapshot("profile", profile(unit_types=["utilities"]))
    )
    assert not accepted


def test_snapshot_identity_profile_overrides_and_reference_images():
    general = knowledge_snapshot("general")
    assert (
        len(list(reference_images(resolve(general, "symbol_interpretation", query="connector"))))
        == 1
    )
    changed = knowledge_snapshot(
        "profile",
        profile(unit_types=["utilities"]),
        [{"pages": [1], "context": {"unit_types": []}}],
    )
    assert changed["identity"] != general["identity"]
    assert resolve(changed, "review", 0, "connector")["context"]["unit_types"] == []
    assert resolve(changed, "review", 1, "connector")["context"]["unit_types"] == ["utilities"]
    p2 = profile(unit_types=["utilities"])
    p2["version"] = 2
    assert (
        knowledge_snapshot("profile", p2)["identity"]
        != knowledge_snapshot("profile", profile(unit_types=["utilities"]))["identity"]
    )
    altered = copy.deepcopy(general)
    altered["entries"][0]["explanation"] += " updated"
    assert digest(altered) != digest(general)
    with pytest.raises(ValueError):
        knowledge_snapshot("general", overrides=[{"context": {}}])


def test_three_conditions_same_annotated_fixtures():
    rows = []
    for name, options, truth in [
        ("interior", {}, True),
        ("vertical", {"vertical": True}, True),
        ("arrow", {"text": "FLOW"}, False),
        ("no-line", {"attached": False}, False),
    ]:
        page, d = fixture(**options)
        rows.append(
            {
                "id": name,
                "page": page.model_dump(),
                "detections": [d.model_dump()],
                "expected_connector_ids": ["candidate"] if truth else [],
            }
        )
    report = evaluate(rows, profile(unit_types=["utilities"]))
    assert report["conditions"]["off"]["totals"]["false_negative"] == 2
    for mode in ["general", "profile"]:
        assert report["conditions"][mode]["delta_true_positive"] == 2
        assert report["conditions"][mode]["totals"]["false_positive"] == 0
        assert report["conditions"][mode]["totals"]["requires_review"] == 2


def test_library_entries_are_valid_and_missing_metadata_stays_unknown():
    snapshot = knowledge_snapshot("general")
    for entry in snapshot["entries"]:
        clean = {k: v for k, v in entry.items() if k != "image_sha256"}
        Entry.model_validate(clean)
        if entry["id"].startswith("opc."):
            assert entry["source"]["document"] is None and entry["source"]["page"] is None


def test_stage_cache_invalidates_on_profile_version(tmp_path):
    from diagex.vision.stage_contracts import StageRequest
    from diagex.vision.stages import run_stage

    page, _ = fixture()
    request = StageRequest(
        stage="text_assignment",
        backend="geometry",
        inputs={"nodes": [], "pages": [page.model_dump(mode="json")]},
        knowledge=knowledge_snapshot("profile", profile(unit_types=["utilities"])),
    )
    first, cached = run_stage(request, tmp_path)
    assert not cached and first["output"]["knowledge"][0]["profile_version"] == 1
    repeated, cached = run_stage(request, tmp_path)
    assert cached and repeated == first
    revised = profile(unit_types=["utilities"])
    revised["version"] = 2
    request.knowledge = knowledge_snapshot("profile", revised)
    second, cached = run_stage(request, tmp_path)
    assert not cached and second["identity_sha256"] != first["identity_sha256"]
    assert second["output"]["knowledge"][0]["profile_version"] == 2


def test_exception_cannot_create_cross_sheet_link():
    from diagex.vision.fusion import _match_opcs
    from diagex.vision.models import ReconciledNode

    page, detection = fixture()
    attrs = {
        "knowledge_exception": {"status": "requires_review"},
        "line_id": "L-1",
        "drawing_ref": "S-2",
        "sheet_id": "S-1",
        "direction": "out",
    }
    left = ReconciledNode(
        id="left",
        kind="opc",
        label="L-1",
        page_index=0,
        bbox_global=detection.bbox,
        confidence="high",
        attributes=attrs,
    )
    right = left.model_copy(
        deep=True,
        update={
            "id": "right",
            "page_index": 1,
            "attributes": {
                "line_id": "L-1",
                "drawing_ref": "S-1",
                "sheet_id": "S-2",
                "direction": "in",
            },
        },
    )
    edges, _, _ = _match_opcs(
        [left, right], pages_by_index={0: page, 1: page.model_copy(update={"page_index": 1})}
    )
    assert not edges


def test_http_profiles_suggestions_and_no_unconfirmed_extraction(tmp_path):
    import http.client
    import io
    import json
    import threading
    from http.server import ThreadingHTTPServer

    from diagex.config import Config, LLMConfig
    from diagex.web.server import Workbench, make_handler

    workbench = Workbench(
        Config(
            llm=LLMConfig(transport="openrouter", model="test", openrouter_api_key="test"),
            runs_dir=tmp_path / "runs",
        ),
        storage_dir=tmp_path / "web",
    )
    with fitz.open() as doc:
        doc.new_page().insert_text((40, 40), "Industry: Chemical")
        content = doc.tobytes()
    upload = workbench.save_upload(
        filename="test.pdf",
        content_type="application/pdf",
        source=io.BytesIO(content),
        length=len(content),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workbench))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def call(path, body=None):
        conn = http.client.HTTPConnection(*server.server_address)
        conn.request(
            "POST" if body is not None else "GET",
            path,
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        status = response.status
        data = json.loads(response.read())
        conn.close()
        return status, data

    try:
        assert call("/api/context/profiles")[1] == {"profiles": []}
        assert (
            call("/api/context/suggestions", {"upload_id": upload.id})[1]["suggestions"][0][
                "passage"
            ]
            == "Industry: Chemical"
        )
        assert call("/api/context/profiles", {"name": "Unconfirmed", "context": {}})[0] == 400
        status, data = call(
            "/api/context/profiles",
            {"name": "Confirmed", "context": {"unit_types": ["utilities"]}, "confirmed": True},
        )
        assert status == 200 and data["profile"]["version"] == 1
        assert (
            call(
                "/api/extractions",
                {
                    "upload_id": upload.id,
                    "knowledge_mode": "profile",
                    "context_profile_id": "missing",
                    "context_profile_version": 1,
                },
            )[0]
            == 400
        )
        assert not workbench.jobs
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_closed_equipment_box_is_not_connector_exception():
    page, detection = fixture()
    page.paths[0].points = [(400, 300), (450, 300), (450, 320), (400, 320), (400, 300)]
    accepted, _ = _prepare_detections(
        [detection], {0: page}, knowledge=knowledge_snapshot("general")
    )
    assert not accepted
