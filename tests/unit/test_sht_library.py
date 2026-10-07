"""Scanned-source curation and retrieval contracts, not extraction-accuracy claims."""

import copy
import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import yaml
from PIL import Image

from diagex.knowledge.library import catalog_asset, load_library, model_assets, validate_match
from diagex.knowledge.resolver import (
    applicable_references,
    knowledge_snapshot,
    library_root,
    resolve,
)

DOC = "sht-3101-2017"
HX = f"{DOC}.t4.16-4"
SCOPE = f"{DOC}.t4.1-8"


def profile(edition="2017", ids=()):
    return dict(id="sht", name="SH/T", version=1, confirmed=True,
                context={"standards": [{"name": "SH/T 3101", "edition": edition}]},
                reference_ids=list(ids))


@pytest.fixture(scope="module")
def library():
    return load_library(library_root())


@pytest.fixture(scope="module")
def entries(library):
    return {e["id"]: e for e in library["entries"] if e["id"].startswith(DOC)}


@pytest.fixture(scope="module")
def snapshot():
    return knowledge_snapshot("profile", profile())


def test_checked_coverage_native_evidence_and_source_identity(library, entries):
    recipe = yaml.safe_load((library_root() / "recipes" / f"{DOC}.yaml").read_text())
    assert len(entries) == 305
    assert Counter(e["catalog"]["role"] for e in entries.values()) == {
        "catalog": 276, "background": 26, "supporting": 3,
    }
    doc = next(d for d in library["documents"] if d["id"] == DOC)
    assert doc["sha256"] == recipe["source_sha256"] and "native pixels" in doc["scan_note"]
    for e in entries.values():
        assert e["applicability"]["standards"] == [{"name": "SH/T 3101", "edition": "2017"}]
        for a in e["assets"]:
            path = catalog_asset(library_root(), e["id"], a["id"])
            if a["crop"]:
                crop = a["crop"]
                x0, y0, x1, y1 = crop["rect"]
                assert crop["dpi"] == 72 and crop["printed_page"] == str(crop["pdf_page"] - 4)
                assert (a["width"], a["height"]) == (x1 - x0, y1 - y0)
                assert "no resampling" in crop["processing"]
            with Image.open(path) as image:
                assert image.size == (a["width"], a["height"])
    # Unrelated sources and the older ISA/website overlap decisions are preserved.
    assert any(e["id"].startswith("isa-") for e in library["entries"])
    assert len(library["replacements"]) == 91


def test_pfd_faint_scan_and_background_never_enter_new_prompts(entries, snapshot):
    eligible = set(applicable_references(snapshot)["reference_ids"])
    background = {e["id"] for e in entries.values() if e["catalog"]["role"] == "background"}
    assert not eligible & background
    assert f"{DOC}.t4.16-1" in background and HX in eligible
    faint = entries[f"{DOC}.t4.1-3"]
    assert any(a["quality"] == "ambiguous" for a in faint["assets"])
    snap = copy.deepcopy(snapshot)
    snap["profile"]["reference_ids"] = list(background)
    result = resolve(snap, "symbol_interpretation", query="PFD tracing furnace compressor")
    assert not background & set(result["reference_ids"])


@pytest.mark.parametrize("standard,edition", [("SH/T 3101", None), ("SH/T 3101", "2000"), ("ISA 5.1", "2009")])
def test_unknown_or_different_edition_cannot_activate_source(snapshot, standard, edition):
    snap = copy.deepcopy(snapshot)
    snap["profile"]["context"]["standards"] = [{"name": standard, "edition": edition}]
    snap["profile"]["reference_ids"] = [HX]
    assert not any(i.startswith(DOC) for i in applicable_references(snap)["reference_ids"])


def test_overrides_legend_precedence_disabled_and_two_image_budget(snapshot):
    snap = copy.deepcopy(snapshot)
    snap["profile"]["reference_ids"] = [HX]
    snap["overrides"] = [{"pages": [2], "context": {"standards": []}}]
    assert HX in applicable_references(snap, 0)["reference_ids"]
    assert HX not in applicable_references(snap, 1)["reference_ids"]
    legend = [{"concept": "heat exchanger", "meaning": "Project-specific glyph"}]
    result = resolve(snap, "symbol_interpretation", query="shell and tube heat exchanger", drawing_definitions=legend)
    assert result["drawing_definitions"] == legend
    assert any("take precedence" in rule for rule in result["core_principles"])
    images = model_assets(result)
    assert len(images) <= 2
    hx = next(a for a in images if a["reference_id"] == HX)
    assert len(hx["variant_ids"]) == 7
    assert validate_match(result, HX, "variant-7", "Observed exchanger outline in drawing")[0]
    assert validate_match(result, HX, "variant-99", "Observed outline")[1]
    assert knowledge_snapshot("off") == {} and resolve({}, "symbol_interpretation") == {}


def test_scope_assemblies_placeholders_and_family_meanings_are_distinct(entries, snapshot):
    assert entries[SCOPE]["reference_type"] == "scope_boundary"
    assert "Containment alone" in entries[SCOPE]["explanation"]
    assembly = entries[f"{DOC}.t4.7-17"]
    assert assembly["reference_type"] == "assembly_pattern"
    assert "not one atomic symbol" in assembly["explanation"]
    pump = entries[f"{DOC}.t4.18-1"]
    assert pump["text_slots"][0]["example_marker"] == "注"
    assert "not an instrument tag" in pump["text_slots"][0]["meaning"]
    connector = entries[f"{DOC}.t4.1-15"]
    assert {s["example_marker"] for s in connector["text_slots"]} == {"T1", "T2"}
    assert "optional" in connector["text_slots"][1]["meaning"]
    third = f"{DOC}.t4.6-1-third-party"
    snap = copy.deepcopy(snapshot)
    snap["profile"]["reference_ids"] = [third]
    result = resolve(snap, "symbol_interpretation", query="third-party instrument")
    assert f"{DOC}.instrument-markings" in result["reference_ids"]
    assert f"{DOC}.letter-rules" in result["reference_ids"]
    assert entries[third]["text_slots"][0]["example_marker"] == "***"
    assert "SIS" in entries[f"{DOC}.t4.6-1-sis"]["explanation"]
    assert "DCS" in entries[f"{DOC}.t4.6-1-dcs"]["explanation"]
    assert not any(a["asset_id"] == "context" for a in model_assets(result))


def test_annotated_column_panel_is_browsable_but_not_supplied_as_symbol(snapshot, entries):
    identity = f"{DOC}.t4.12-1"
    snap = copy.deepcopy(snapshot)
    snap["profile"]["reference_ids"] = [identity]
    result = resolve(snap, "symbol_interpretation", query="tray column")
    image = next(a for a in model_assets(result) if a["reference_id"] == identity)
    assert image["variant_ids"] == ["variant-1"]
    assert len(entries[identity]["catalog"]["displayed_asset_ids"]) == 2
    assert validate_match(result, identity, "variant-2", "A column")[1] == "variant_not_supplied"


def test_rebuild_refuses_a_different_scan_before_writing(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_sht3101_library.py"
    spec = importlib.util.spec_from_file_location("prepare_sht", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    other = tmp_path / "other.pdf"
    other.write_bytes(b"not the inspected source")
    with pytest.raises(ValueError, match="checksum"):
        module.prepare(other, root=tmp_path, recipe_path=library_root() / "recipes" / f"{DOC}.yaml")
    assert not (tmp_path / "knowledge").exists()
