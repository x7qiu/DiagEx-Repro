"""Focused catalog contracts. These tests do not measure extraction accuracy."""

import copy

import pytest
from pydantic import ValidationError

from diagex.knowledge.library import load_library, model_assets, supplied_trace, validate_match
from diagex.knowledge.models import Entry
from diagex.knowledge.resolver import (
    applicable_references,
    digest,
    knowledge_snapshot,
    library_root,
    resolve,
)

FIELD = "isa-5.1-2009.t5.1.1-1"
NOTES = "isa-5.1-2009.clause-5.3.1"
BACKGROUND = "isa-5.1-2009.t5.7-3"


def profile(**kwargs):
    return {
        "id": "focused", "name": "Focused ISA", "version": 1, "confirmed": True,
        "context": {"standards": [{"name": "ISA 5.1", "edition": "2009"}]},
        **kwargs,
    }


@pytest.fixture(scope="module")
def catalog():
    return load_library(library_root())


@pytest.fixture(scope="module")
def snapshot():
    return knowledge_snapshot("profile", profile(reference_ids=[FIELD]))


def test_catalog_is_visual_and_archive_is_preserved(catalog):
    entries = [e for e in catalog["entries"] if e["source"].get("document_id") in (None, "isa-5.1-2009")]
    selected = [e for e in entries if e["catalog"]["role"] == "catalog"]
    assert len(selected) == 158 and len(entries) == 453
    assert {e["catalog"]["category"] for e in selected} == {
        "instruments", "valves_actuators", "measurement", "lines_signals", "connectors"
    }
    for e in selected:
        metadata = e["catalog"]
        images = {a["id"]: a for a in e["assets"]}
        assert metadata["short_name"] and metadata["interpretation"]
        for asset_id in metadata["displayed_asset_ids"]:
            asset = images[asset_id]
            assert asset["crop"]["dpi"] == 300
            assert asset["width"] > 0 and asset["height"] > 0
        assert images[metadata["model_asset_id"]]["derived_from"] == metadata["displayed_asset_ids"]
    field = next(e for e in entries if e["id"] == FIELD)
    selected_assets = {a["id"]: a for a in field["assets"]}
    assert [selected_assets[id_]["label"] for id_ in field["catalog"]["displayed_asset_ids"]] == [
        "Individual instrument", "Primary control system", "Alternate control system"
    ]
    assert any(a["label"] == "C: Computer systems" for a in field["assets"])
    assert field["notes"] and field["source"]["passage"]
    assert next(e for e in entries if e["id"] == BACKGROUND)["catalog"]["role"] == "background"


def test_focused_prompts_include_notes_but_not_source_archive(snapshot):
    result = resolve(snapshot, "symbol_interpretation", query="field instrument")
    assert FIELD in result["reference_ids"] and NOTES in result["reference_ids"]
    field = next(e for e in result["references"] if e["id"] == FIELD)
    assert field["concept"] == "Field instrument"
    assert not field["notes"] and not field["table_data"] and not field["source"]["passage"]
    assert not field["source"]["image"]
    assert all("panel-3" not in a["id"] for a in field["assets"])
    notes = next(e for e in result["references"] if e["id"] == NOTES)
    assert "Individual" in notes["explanation"] or "individual" in notes["explanation"]
    assert "interchangeable" in notes["explanation"]
    assert not notes["assets"] and not notes["source"]["image"]
    assets = model_assets(result)
    assert len(assets) <= 2 and len(result["references"]) <= 8
    assert all("panel-3" not in v for a in assets for v in a["variant_ids"])
    trace = supplied_trace(result)
    assert {e["id"] for e in trace["references"]} == set(result["reference_ids"])
    match, error = validate_match(result, FIELD, "catalog-t5.1.1-1-panel-4", "Circle in drawing")
    assert match and not error
    assert validate_match(result, FIELD, "t5.1.1-1-panel-3", "Hexagon")[1] == "variant_not_supplied"


def test_background_cannot_be_forced_or_enter_through_conflicts(snapshot):
    snap = copy.deepcopy(snapshot)
    snap["profile"]["reference_ids"] = [BACKGROUND]
    background = next(e for e in snap["entries"] if e["id"] == BACKGROUND)
    field = next(e for e in snap["entries"] if e["id"] == FIELD)
    background.update(definition_key="shape", definition_value="logic")
    field.update(definition_key="shape", definition_value="instrument")
    result = resolve(snap, "symbol_interpretation", query="NAND field instrument")
    assert BACKGROUND not in result["reference_ids"] and not result["conflicts"]
    assert BACKGROUND not in applicable_references(snap)["reference_ids"]
    alternate = copy.deepcopy(field)
    alternate.update(id="test.conflicting-instrument", definition_value="conflicting convention")
    snap["entries"].append(alternate)
    result = resolve(snap, "symbol_interpretation", query="field instrument")
    assert result["conflicts"]
    assert {FIELD, alternate["id"], NOTES} <= set(result["reference_ids"])


@pytest.mark.parametrize("edition", [None, "2024"])
def test_supporting_knowledge_respects_edition_and_sheet_overrides(edition):
    p = profile()
    p["context"]["standards"][0]["edition"] = edition
    result = resolve(knowledge_snapshot("profile", p), "symbol_interpretation", query="field instrument")
    assert FIELD not in result["reference_ids"] and NOTES not in result["reference_ids"]
    snap = knowledge_snapshot("profile", profile(reference_ids=[FIELD]), [
        {"pages": [2], "context": {"standards": []}}
    ])
    assert FIELD in resolve(snap, "symbol_interpretation", 0)["reference_ids"]
    assert FIELD not in resolve(snap, "symbol_interpretation", 1)["reference_ids"]
    legend = {"field instrument": "Project-specific meaning"}
    assert resolve(snap, "symbol_interpretation", drawing_definitions=legend)["drawing_definitions"] == legend
    assert knowledge_snapshot("off") == {} and resolve({}, "symbol_interpretation") == {}


def test_legacy_snapshot_keeps_original_images_and_attribution(snapshot):
    snap = copy.deepcopy(snapshot)
    # Historical serialized snapshots carry neither selection metadata nor new assets.
    for e in snap["entries"]:
        e.pop("catalog", None)
        e["assets"] = [a for a in e["assets"] if not a["id"].startswith("catalog-")]
    result = resolve(snap, "symbol_interpretation", query="field instrument")
    assert any("t5.1.1-1-panel-3" in a["variant_ids"] for a in model_assets(result))
    assert validate_match(result, FIELD, "t5.1.1-1-panel-3", "Visible hexagon")[0]
    raw = {k: v for k, v in next(e for e in snap["entries"] if e["id"] == FIELD).items()
           if k != "image_sha256"}
    assert Entry.model_validate(raw).catalog is None


def test_metadata_and_curated_text_participate_in_cache_identity(snapshot):
    first = resolve(snapshot, "symbol_interpretation", query="field instrument")
    for key, value in [("interpretation", "Updated interpretation"), ("short_name", "Updated name")]:
        snap = copy.deepcopy(snapshot)
        next(e for e in snap["entries"] if e["id"] == FIELD)["catalog"][key] = value
        assert digest(snap) != digest(snapshot)
        assert resolve(snap, "symbol_interpretation", query="field instrument")["identity"] != first["identity"]
    snap = copy.deepcopy(snapshot)
    next(e for e in snap["entries"] if e["id"] == FIELD)["catalog"]["role"] = "background"
    assert FIELD not in resolve(snap, "symbol_interpretation", query="field instrument")["reference_ids"]


def test_invalid_metadata_fails_instead_of_falling_back_to_whole_panel(catalog):
    field = next(e for e in catalog["entries"] if e["id"] == FIELD)
    field = {k: v for k, v in field.items() if k != "image_sha256"}
    for changes in [
        {"category": None}, {"displayed_asset_ids": ["missing"]},
        {"model_asset_id": "missing"}, {"model_asset_id": "variants"},
    ]:
        entry = copy.deepcopy(field)
        entry["catalog"].update(changes)
        with pytest.raises(ValidationError):
            Entry.model_validate(entry)


def test_note_citations_removed_without_erasing_geometry_or_symbol_text():
    import runpy
    from pathlib import Path

    import fitz

    tools = runpy.run_path(str(Path(__file__).parents[2] / "scripts/curate_isa_catalog.py"))
    with fitz.open() as original:
        page = original.new_page(width=220, height=160)
        page.insert_text((20, 26), "(2) ( (7) (8)", fontsize=8)
        page.insert_text((20, 50), "a) S (*) (#) (##) ?T", fontsize=10)
        page.insert_text((20, 120), "(2)", fontsize=10)  # Not a top-margin citation.
        page.draw_rect(fitz.Rect(15, 15, 205, 130))  # Crosses note-removal regions.
        rect = fitz.Rect(10, 10, 210, 140)
        before = [original.xref_stream(x) for x in page.get_contents()]
        working, cleaned, removed = tools["cleaned_panel_page"](original, 0, rect)
        try:
            assert {n["text"] for n in removed} == {"(2)", "(7)", "(8)", "("}
            text = cleaned.get_text()
            assert "(7)" not in text and "(8)" not in text and text.count("(2)") == 1
            for token in ["a)", "S", "(*)", "(#)", "(##)", "?T"]:
                assert token in text
            assert [p["items"] for p in cleaned.get_drawings()] == [
                p["items"] for p in page.get_drawings()
            ]
            assert [original.xref_stream(x) for x in page.get_contents()] == before
        finally:
            working.close()


def test_cleaned_assets_record_excluded_notes_and_preserve_originals(catalog):
    entry = next(e for e in catalog["entries"] if e["id"] == "isa-5.1-2009.t5.4.2-17")
    original = next(a for a in entry["assets"] if a["id"] == "t5.4.2-17-panel-1")
    cleaned = next(a for a in entry["assets"] if a["id"] == "catalog-t5.4.2-17-panel-1")
    assert not original["crop"]["excluded_annotations"]
    assert [n["text"] for n in cleaned["crop"]["excluded_annotations"]] == ["(8)"]
    assert "text-only PDF redaction" in cleaned["crop"]["processing"]
    assert original["sha256"] != cleaned["sha256"]
    assert cleaned["id"] in entry["catalog"]["displayed_asset_ids"]
