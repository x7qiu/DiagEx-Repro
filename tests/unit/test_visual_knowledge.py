import copy
import hashlib
import http.client
import json
import shutil
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from diagex.config import Config, SymbolPerceptionConfig
from diagex.knowledge import resolver
from diagex.knowledge.library import catalog_asset, load_library, model_assets, validate_match
from diagex.knowledge.models import Entry, Mapping
from diagex.knowledge.resolver import knowledge_snapshot, reference_images, resolve
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, Tile
from diagex.vision.symbol_interpretation import (
    PerceivedObject,
    PerceptionBatch,
    _bind_knowledge,
    _knowledge_image_blocks,
    perceive_tile,
)
from diagex.vision.views import ViewInfo
from diagex.web.server import Workbench, make_handler

PILOT = "isa-5.1-2009.t5.3.2-17"


def profile(edition="2009"):
    return {
        "id": "isa",
        "name": "ISA pilot",
        "version": 1,
        "confirmed": True,
        "context": {"standards": [{"name": "ISA 5.1", "edition": edition}]},
        "reference_ids": [PILOT],
    }


def context():
    return resolve(
        knowledge_snapshot("profile", profile()), "symbol_interpretation", query="signal connector"
    )


@pytest.fixture
def copied_library(tmp_path, monkeypatch):
    dest = tmp_path / "library"
    shutil.copytree(resolver.library_root(), dest)
    monkeypatch.setattr(resolver, "library_root", lambda: dest)
    return dest


def rewrite(path, mutate):
    value = yaml.safe_load(path.read_text())
    mutate(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_pilot_has_complete_variants_notes_and_document_without_original_pdf(copied_library):
    catalog = load_library(copied_library)
    assert any(d["id"] == "isa-5.1-2009" for d in catalog["documents"])
    assert catalog["mappings"] == []
    entries = [
        e
        for e in catalog["entries"]
        if e["id"] in {f"isa-5.1-2009.t5.3.2-{n}" for n in (17, 18, 19)}
    ]
    assert len(entries) == 3
    e = next(e for e in entries if e["id"] == PILOT)
    assert {a["id"] for a in e["assets"] if a["role"] == "variant"} == {"a1", "a2", "b1", "b2"}
    assert e["notes"][0]["source"]["pdf_page"] == 34
    assert {s["example_marker"] for s in e["text_slots"]} == {"(#)", "(##)"}
    assert not list(copied_library.rglob("*.pdf"))
    for entry in entries:
        for asset in entry["assets"]:
            assert catalog_asset(copied_library, entry["id"], asset["id"]).is_file()
            if asset["crop"]:
                assert asset["crop"]["dpi"] == 300
    legacy = next(e for e in catalog["entries"] if e["id"] == "opc.placement-exceptions")
    assert legacy["source"]["document"] is None
    assert catalog_asset(copied_library, legacy["id"]).is_file()


@pytest.mark.parametrize("problem", ["missing", "traversal", "symlink", "checksum", "dimensions"])
def test_bad_assets_fail_explicitly(copied_library, tmp_path, problem):
    record = copied_library / f"{PILOT}.yaml"
    entry = yaml.safe_load(record.read_text())
    asset = entry["assets"][0]
    path = copied_library / asset["path"]
    if problem == "missing":
        path.unlink()
    elif problem == "checksum":
        path.write_bytes(b"changed")
    elif problem == "symlink":
        outside = tmp_path / "outside.png"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        rewrite(
            record,
            lambda e: e["assets"][0].update(
                {"path": "../outside.png"} if problem == "traversal" else {"width": 1}
            ),
        )
    with pytest.raises(ValueError):
        load_library(copied_library)


def test_cache_identity_covers_notes_images_and_source_manifest(copied_library):
    first = knowledge_snapshot("general")["identity"]
    record = copied_library / f"{PILOT}.yaml"
    rewrite(record, lambda e: e["notes"][0].update(explanation="Revised note"))
    second = knowledge_snapshot("general")["identity"]
    assert first != second
    asset = yaml.safe_load(record.read_text())["assets"][2]
    path = copied_library / asset["path"]
    with Image.open(path) as im:
        im.putpixel((0, 0), (0, 0, 0))
        im.save(path)
    rewrite(
        record,
        lambda e: e["assets"][2].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    third = knowledge_snapshot("general")["identity"]
    assert third != second
    rewrite(
        copied_library / "sources" / "isa-5.1-2009.yaml",
        lambda d: d.update(title="Revised source title"),
    )
    assert knowledge_snapshot("general")["identity"] != third


def test_snapshot_rejects_images_changed_after_configuration(copied_library):
    ctx = context()
    path = copied_library / model_assets(ctx)[0]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since"):
        list(reference_images(ctx))


def test_scoped_retrieval_and_drawing_override():
    assert not any(
        e.startswith("isa-")
        for e in resolve(
            knowledge_snapshot("general"), "symbol_interpretation", query="signal connector"
        )["reference_ids"]
    )
    for edition in (None, "2024"):
        assert (
            PILOT
            not in resolve(
                knowledge_snapshot("profile", profile(edition)),
                "symbol_interpretation",
                query="signal connector",
            )["reference_ids"]
        )
    assert PILOT in context()["reference_ids"]
    snap = knowledge_snapshot("profile", profile(), [{"pages": [2], "context": {"standards": []}}])
    assert (
        PILOT
        not in resolve(snap, "symbol_interpretation", page_index=1, query="signal connector")[
            "reference_ids"
        ]
    )
    legend = {"connector": "Project-specific definition"}
    ctx = resolve(
        snap, "symbol_interpretation", query="signal connector", drawing_definitions=legend
    )
    assert ctx["drawing_definitions"] == legend
    assert "take precedence" in ctx["core_principles"][0]
    assert knowledge_snapshot("off") == {} and _knowledge_image_blocks({}) == []


def test_bounded_labeled_images_and_attribution_validation():
    ctx = context()
    blocks = _knowledge_image_blocks(ctx)
    assert sum(b["type"] == "image" for b in blocks) == 2
    assert PILOT in blocks[0]["text"] and "a1, a2, b1, b2" in blocks[0]["text"]
    assert len({(a["path"], a["sha256"]) for a in model_assets(ctx)}) == len(model_assets(ctx))
    match, error = validate_match(
        ctx, PILOT, "a1", "Two compartments with tag and sheet in the source"
    )
    assert not error and match["status"] == "model_reported"
    for ref, variant, reason, expected in [
        ("invented", None, "ink", "reference_not_supplied"),
        (PILOT, "invented", "ink", "variant_not_supplied"),
        (PILOT, None, " ", "missing_drawing_evidence"),
    ]:
        assert validate_match(ctx, ref, variant, reason) == (None, expected)


def test_match_does_not_assign_placeholder_or_change_geometry_and_invalid_match_retains_object():
    batch = PerceptionBatch(
        objects=[
            PerceivedObject(
                kind="opc",
                bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1},
                printed_tag="(#)",
                drawing_ref="(##)",
                confidence="low",
                knowledge_reference_id=PILOT,
                knowledge_variant_id="a1",
                knowledge_evidence="Source has two text compartments",
            )
        ]
    )
    before = batch.objects[0].bbox.model_dump()
    _bind_knowledge(batch, context())
    obj = batch.objects[0]
    assert obj.bbox.model_dump() == before and obj.confidence == "low"
    assert obj.printed_tag is None and obj.drawing_ref is None
    assert obj.attributes["knowledge_match"]["reference_version"] == next(
        e["version"] for e in context()["references"] if e["id"] == PILOT
    )
    obj.knowledge_reference_id = "invented"
    _bind_knowledge(batch, context())
    assert (
        "knowledge_match" not in obj.attributes
        and obj.attributes["knowledge_match_error"] == "reference_not_supplied"
    )
    assert len(batch.objects) == 1


def test_future_reference_types_and_separate_mapping():
    path = Path(__file__).parents[1] / "fixtures" / "knowledge" / "reference-types.yaml"
    entries = [Entry.model_validate(e) for e in yaml.safe_load(path.read_text())]
    assert {e.reference_type for e in entries} == {
        "line_style",
        "assembly_pattern",
        "scope_boundary",
    }
    mapping = Mapping(
        reference_id=entries[0].id,
        reference_version=1,
        target_model="example",
        target_version="1",
        target_kind="relationship",
        target="signal",
        conditions=["Actual signal evidence required"],
    )
    assert mapping.reference_version == 1 and "target" not in entries[0].model_dump()


def message(name, payload):
    from anthropic.types import Message

    return Message.model_validate(
        {
            "id": "mock-knowledge",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [{"type": "tool_use", "id": "tool-1", "name": name, "input": payload}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 20},
        }
    )


def drawing():
    page = PageEvidence(
        page_index=0,
        source_ref="test",
        width=500,
        height=400,
        dpi=300,
        effective_dpi=300,
        role="pid",
        is_scanned=True,
    )
    box = BBox(x=0, y=0, w=500, h=400)
    return (
        page,
        Tile(id="t1", page_index=0, bbox=box),
        ViewInfo("tile", (0, 0), 1, 1, (500, 400), box),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_vlm_request_and_output_use_traceable_reference_images(enabled):
    page, tile, view = drawing()
    ctx = context() if enabled else {}
    proposal = {
        "kind": "opc",
        "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.1},
        "printed_tag": "FT-101",
        "drawing_ref": "D-2",
        "legend_entry_ids": ["drawing-connector"],
        "legend_evidence": "Two text compartments match the source legend",
    }
    if enabled:
        proposal.update(
            knowledge_reference_id=PILOT,
            knowledge_variant_id="a1",
            knowledge_evidence="Rounded end, two compartments with FT-101 and D-2 in the drawing",
        )
    calls = []
    client = SimpleNamespace(
        messages_create=lambda **kw: (
            calls.append(kw)
            or message("submit_pid_objects", {"candidate_results": [], "proposals": [proposal]})
        )
    )
    outcome = perceive_tile(
        client=client,
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=page,
        tile=tile,
        view_image=Image.new("RGB", (500, 400), "white"),
        view_info=view,
        ownership_bbox=tile.bbox,
        legend_summary=[{"source_row_id": "drawing-connector", "source": "legend_extracted", "source_page_index": 0,
                         "label": "Connector", "kind": "connector", "symbol_class": "opc"}],
        step=1,
        candidates=[],
        page_context={"knowledge": ctx},
        reasoning_mode="disabled",
    )
    assert len(calls) == 1 and len(outcome.detections) == 1
    content = calls[0]["messages"][0]["content"]
    assert sum(b["type"] == "image" for b in content) == (3 if enabled else 1)
    node = outcome.detections[0]
    assert node.label == "FT-101" and node.attributes["drawing_ref"] == "D-2"
    assert node.bbox == BBox(x=50, y=80, w=150, h=40)
    assert node.attributes["legend_matches"][0]["source"] == "legend_extracted"
    assert node.attributes["legend_matches"][0]["legend_entry_id"] == "drawing-connector"
    if enabled:
        assert node.attributes["knowledge_match"]["variant_id"] == "a1"
        assert node.attributes["supplied_knowledge"]["profile_version"] == 1
    else:
        assert (
            "supplied_knowledge" not in node.attributes and "knowledge_match" not in node.attributes
        )


@pytest.mark.parametrize("reference_id", [PILOT, "invented"])
def test_cv_assisted_semantics_receives_images_and_validates_matches(reference_id):
    from diagex.vision.raster_semantics import interpret_raster_reviews, interpreted_detection

    page, tile, view = drawing()
    original = [
        {
            "status": "uncertain",
            "bbox": {"x": 50, "y": 80, "w": 150, "h": 40},
            "object": {
                "kind": "raster_symbol",
                "confidence": "low",
                "attributes": {
                    "broad_category": "connector",
                    "geometry_basis": "cv_proposal",
                    "raster_proposal_id": "cv-1",
                },
            },
        }
    ]
    row = {
        "symbol_id": "symbol-0",
        "decision": "interpreted",
        "legend_entry_ids": [],
        "reason": "Source has a rounded connector with two compartments",
        "knowledge_reference_id": reference_id,
        "knowledge_variant_id": "a1",
        "knowledge_evidence": "Source tag FT-101 and sheet D-2 within the connector",
        "symbol": {
            "kind": "opc",
            "connector_type": "signal",
            "printed_tag": "FT-101",
            "drawing_ref": "D-2",
        },
    }
    calls = []
    client = SimpleNamespace(
        messages_create=lambda **kw: (
            calls.append(kw) or message("submit_raster_semantics", {"results": [row]})
        )
    )
    before = copy.deepcopy(original)
    reviews, audit = interpret_raster_reviews(
        reviews=original,
        client=client,
        cost_tracker=CostTracker(),
        page=page,
        tile=tile,
        view_image=Image.new("RGB", (500, 400), "white"),
        view_info=view,
        legend_entries=[],
        policy=SymbolPerceptionConfig(),
        knowledge_context=context(),
    )
    assert original == before and reviews[0]["bbox"] == original[0]["bbox"]
    assert len(calls) == 1 and len(audit["supplied_knowledge"]["assets"]) == 2
    assert sum(b["type"] == "image" for b in calls[0]["messages"][0]["content"]) == 3
    decision = reviews[0]["legend_interpretation"]
    if reference_id == PILOT:
        assert decision["decision"] == "interpreted"
        result = interpreted_detection(
            {**original[0]["object"], "bbox": original[0]["bbox"]}, decision
        )
        assert result["attributes"]["knowledge_match"]["reference_id"] == PILOT
        assert result["bbox"] == original[0]["bbox"]
    else:
        assert decision["decision"] == "unresolved" and "symbol" not in decision


def test_web_catalog_assets_and_legacy_api(tmp_path):
    workbench = Workbench(Config(), storage_dir=tmp_path / "web")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(workbench))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def get(path):
        conn = http.client.HTTPConnection(*server.server_address)
        conn.request("GET", path)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data

    def preview(body):
        conn = http.client.HTTPConnection(*server.server_address)
        conn.request(
            "POST",
            "/api/context/applicable",
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        data = json.loads(response.read())
        conn.close()
        return response.status, data

    try:
        status, html = get("/knowledge")
        assert status == 200 and b"Knowledge Library" in html and b"typeFilter" in html
        status, body = get("/api/context/library")
        assert status == 200 and len(json.loads(body)["entries"]) >= 5
        matrices = [e for e in json.loads(body)["entries"] if e.get("letter_matrix")]
        assert len(matrices) == 2
        assert all(len(e["letter_matrix"]["rows"]) == 26 for e in matrices)
        assert b'option value="letters"' in html
        for identity, asset in [("isa-5.1-2009.t4.1-p30", "page-30"),
                                ("sht-3101-2017.letter-table", "complete-notes")]:
            status, data = get(f"/api/context/asset?id={identity}&asset={asset}")
            assert status == 200 and data.startswith(b"\x89PNG")
        for asset in ("a1", "a2", "b1", "b2", "context", "note-3", "variants"):
            status, data = get(f"/api/context/asset?id={PILOT}&asset={asset}")
            assert status == 200 and data.startswith(b"\x89PNG")
        assert get("/api/context/illustration?id=opc.placement-exceptions")[0] == 200
        assert get(f"/api/context/asset?id={PILOT}&asset=../sources/isa-5.1-2009.yaml")[0] == 400
        assert get("/api/context/asset?id=unknown&asset=a1")[0] == 400
        assert get(f"/api/context/asset?id={PILOT}")[0] == 400
        assert b"/knowledge" in get("/")[1]
        assert b"browseContextLibrary" in get("/")[1]
        assert preview({"knowledge_mode": "off"})[1]["reference_ids"] == []
        general_ids = preview({"knowledge_mode": "general"})[1]["reference_ids"]
        assert "opc.placement-exceptions" in general_ids
        assert not any(id_.startswith("isa-") for id_ in general_ids)
        assert any(id_.startswith("projectmaterials.") for id_ in general_ids)
        selected_ids = preview({"knowledge_mode": "general", "knowledge_sources": ["isa-5.1-2009", "sht-3101-2017"]})[1]["reference_ids"]
        assert any(i.startswith("isa-") for i in selected_ids) and any(i.startswith("sht-") for i in selected_ids)
        assert not any(i.startswith("projectmaterials.") for i in selected_ids)
        assert preview({"knowledge_mode": "general", "knowledge_sources": ["invented"]})[0] == 400
        assert preview({"knowledge_mode": "profile"})[0] == 400
        saved = workbench.context_profiles.save(
            {k: v for k, v in profile().items() if k != "version"}
        )
        payload = {
            "knowledge_mode": "profile",
            "context_profile_id": saved["id"],
            "context_profile_version": saved["version"],
        }
        assert len(preview(payload)[1]["reference_ids"]) == len(
            [e for e in load_library(resolver.library_root())["entries"]
             if (e.get("catalog") or {}).get("role") != "background"
             and (not e["applicability"]["standards"] or e["applicability"]["standards"] == [{"name": "ISA 5.1", "edition": "2009"}])]
        )
        overrides = [{"pages": [2], "context": {"standards": []}}]
        assert (
            PILOT
            not in preview({**payload, "drawing_overrides": overrides, "page": 2})[1][
                "reference_ids"
            ]
        )
        assert (
            PILOT
            in preview({**payload, "drawing_overrides": overrides, "page": 1})[1]["reference_ids"]
        )
        for page in (0, True, "1", 1.5):
            assert preview({**payload, "page": page})[0] == 400
        # Browsing and applicability preview must not write a profile revision.
        assert workbench.context_profiles.list() == [saved]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
