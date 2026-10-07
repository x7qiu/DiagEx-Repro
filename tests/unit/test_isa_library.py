"""Whole-edition coverage and provenance contracts; not extraction accuracy tests."""

import hashlib

import pytest
import yaml

from diagex.knowledge.library import catalog_asset, load_library, model_assets
from diagex.knowledge.resolver import knowledge_snapshot, library_root, resolve


@pytest.fixture(scope="module")
def catalog():
    return load_library(library_root())


def test_complete_edition_has_no_missing_numbered_items_or_technical_pages(catalog):
    coverage = yaml.safe_load((library_root() / "coverage/isa-5.1-2009.yaml").read_text())
    ids = {e["id"] for e in catalog["entries"]}
    assert len(coverage["technical_pages"]) == 116
    assert set(map(int, coverage["pages"])) == set(coverage["technical_pages"])
    assert sum(len(items) for items in coverage["numbered_tables"].values()) == 263
    for table, items in coverage["numbered_tables"].items():
        assert items == list(range(1, max(items) + 1))
        assert all(f"isa-5.1-2009.t{table}-{item}" in ids for item in items)
    assert {f"isa-5.1-2009.clause-3.1.{i}" for i in range(1, 67)} <= ids
    assert {f"isa-5.1-2009.t4.1-letter-{letter}" for letter in "abcdefghijklmnopqrstuvwxyz"} <= ids
    assert {f"6.{i}" for i in range(1, 9)} <= set(coverage["tables"])
    assert all(set(entries) <= ids for entries in coverage["pages"].values())
    assert not set(coverage["excluded_pages"]) & set(coverage["pages"])


def test_panels_keep_alternatives_notes_and_rotated_source_coordinates(catalog):
    entries = {e["id"]: e for e in catalog["entries"]}
    location = entries["isa-5.1-2009.t5.1.1-1"]
    variants = [a for a in location["assets"] if a["role"] == "variant" and not a["id"].startswith("catalog-")]
    assert len(variants) == 4 and all(a["depiction"] == "source_panel" for a in variants)
    assert location["notes"][0]["source"]["passage"] == "Clause 5.3.1"
    landscape = entries["isa-5.1-2009.ta.2-p100"]
    crop = landscape["assets"][0]["crop"]
    assert crop["page_rotation"] == 90
    assert crop["coordinates"] == "unrotated_pdf_points_top_left"
    assert crop["rect"][2] <= 612 and crop["rect"][3] <= 792
    assert landscape["table_data"] and landscape["source_status"] == "informative"
    for item in (3, 4):
        e = entries[f"isa-5.1-2009.t5.7-{item}"]
        assert any("Source inconsistency" in text for text in e["exceptions"])
    assembly = entries["isa-5.1-2009.clause-b.13"]
    assert assembly["reference_type"] == "assembly_pattern"
    assert {a["crop"]["pdf_page"] for a in assembly["assets"]} == {124, 125, 126}
    assert all(
        e["source_status"] == "informative"
        for e in catalog["entries"]
        if e["id"].startswith("isa-5.1-2009.clause-b.")
    )


def test_large_library_retrieval_is_scoped_and_bounded():
    profile = {
        "id": "isa-full",
        "name": "ISA full",
        "version": 1,
        "confirmed": True,
        "context": {"standards": [{"name": "ISA 5.1", "edition": "2009"}]},
    }
    snapshot = knowledge_snapshot("profile", profile)
    result = resolve(snapshot, "symbol_interpretation", query="NAND gate")
    assert "isa-5.1-2009.t5.7-3" not in result["reference_ids"]
    assert all(e["catalog"]["role"] != "background" for e in result["references"])
    assert len(result["references"]) <= 8 and len(model_assets(result)) <= 2
    result = resolve(snapshot, "symbol_interpretation", query="solenoid actuator")
    assert any(e["source"]["table"] == "5.4.2" for e in result["references"])
    profile["context"]["standards"][0]["edition"] = "2024"
    assert not any(
        e.startswith("isa-")
        for e in resolve(
            knowledge_snapshot("profile", profile), "symbol_interpretation", query="NAND gate"
        )["reference_ids"]
    )
    assert knowledge_snapshot("off") == {}


def test_asset_fast_path_still_checks_hash_and_safe_record_location(tmp_path):
    from PIL import Image

    from diagex.knowledge.models import Entry

    png = tmp_path / "image.png"
    Image.new("RGB", (2, 3), "white").save(png)
    record = {
        "id": "test",
        "version": 1,
        "concept": "test",
        "kind": "definition",
        "explanation": "test",
        "tasks": [],
        "source": {},
        "assets": [
            {
                "id": "sample",
                "role": "context",
                "label": "source",
                "path": "image.png",
                "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                "width": 2,
                "height": 3,
                "crop": {
                    "document_id": "test",
                    "pdf_page": 1,
                    "printed_page": "1",
                    "rect": [0, 0, 1, 1],
                    "dpi": 300,
                    "renderer": "test",
                },
            }
        ],
    }
    entry = Entry.model_validate(record)
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(entry.model_dump()))
    assert catalog_asset(tmp_path, "test", "sample") == png
    Image.new("RGB", (2, 3), "black").save(png)
    with pytest.raises(ValueError, match="manifest"):
        catalog_asset(tmp_path, "test", "sample")
    with pytest.raises(ValueError):
        catalog_asset(tmp_path, "../test", "sample")
