"""Offline contracts for web evidence, independent of the live publisher."""

import copy
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from diagex.knowledge.library import catalog_asset, load_library, model_assets, validate_match
from diagex.knowledge.models import Entry
from diagex.knowledge.resolver import digest, knowledge_snapshot, library_root, resolve

DOC = "projectmaterials.pid-symbols"
HX = f"{DOC}.heat-exchangers.plate-heat-exchanger"


@pytest.fixture(scope="module")
def library():
    return load_library(library_root())


def test_web_source_preserves_names_images_and_separate_identity(library):
    document = next(d for d in library["documents"] if d["id"] == DOC)
    assert document["source_type"] == "web" and document["edition"] is None
    assert document["publisher"] == "Projectmaterials" and document["retrieved_at"]
    entries = [e for e in library["entries"] if e["source"]["document_id"] == DOC]
    assert len(entries) == 315
    assert len(library["replacements"]) == 91
    assert sum(e["source"]["section"] == "Heat Exchangers P&ID Symbols" for e in entries) == 46
    assert sum(e["catalog"]["role"] == "background" for e in entries) == 7
    for entry in entries:
        assert entry["source_status"] == "informative"
        assert not entry["applicability"]["standards"]
        asset = entry["assets"][0]
        assert asset["crop"] is None and asset["web"]["document_id"] == DOC
        assert asset["web"]["url"].startswith("https://blog.projectmaterials.com/")
        assert asset["web"]["original_sha256"] and asset["sha256"]
        assert catalog_asset(library_root(), entry["id"], "symbol").is_file()


def test_web_retrieval_trace_off_and_no_isa_scope_leak(library):
    snapshot = knowledge_snapshot("general")
    context = resolve(snapshot, "symbol_interpretation", query="plate heat exchanger")
    assert HX in context["reference_ids"]
    assert not any(id_.startswith("isa-") for id_ in context["reference_ids"])
    assert all(e["catalog"]["role"] != "background" for e in context["references"])
    assets = model_assets(context)
    assert 0 < len(assets) <= 2 and any(a["reference_id"] == HX for a in assets)
    assert validate_match(context, HX, "symbol", "Parallel plates visible in drawing crop")[0]
    assert validate_match(context, HX, "invented", "Plate geometry")[1]
    assert not validate_match(context, HX, "symbol", "")[0]
    assert resolve(knowledge_snapshot("off"), "symbol_interpretation", query="heat exchanger") == {}
    original = next(e for e in library["entries"] if e["id"] == HX)
    changed = copy.deepcopy(original)
    changed["assets"][0]["web"]["original_sha256"] = "0" * 64
    assert digest(changed) != digest(original)


def test_invalid_web_origin_and_urls(library):
    original = next(e for e in library["entries"] if e["id"] == HX)
    invalid = copy.deepcopy(original)
    invalid["assets"][0]["web"] = None
    with pytest.raises(ValidationError, match="provenance"):
        Entry.model_validate(invalid)
    invalid = copy.deepcopy(original)
    invalid["source"]["url"] = "javascript:alert(1)"
    with pytest.raises(ValidationError):
        Entry.model_validate(invalid)


def test_parser_excludes_overviews_navigation_and_prose():
    path = Path(__file__).resolve().parents[2] / "scripts/import_projectmaterials.py"
    spec = importlib.util.spec_from_file_location("projectmaterials_import_test", path)
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    result = importer.parse("""<img src="logo.png"><h2 id="heat">Heat Exchangers P&amp;ID Symbols</h2>
      <p>Article prose is not a symbol definition.</p><table><tr><td>
      <img alt="Plate HX" src="/_astro/plate.webp">Plate <b>Heat Exchanger</b>
      </td></tr></table><img src="/_astro/overview.webp">
      <h2>Frequently Asked Questions</h2><table><td><img src="ad.png"></td></table>""")
    assert len(result.rows) == 1
    assert result.rows[0]["label"] == "Plate Heat Exchanger"
    assert len(result.overviews) == 1
    with pytest.raises(ValueError, match="publisher host"):
        importer.fetch("https://example.com/evil.png", Path("/tmp/not-downloaded"))


def test_web_assets_require_a_web_manifest(library, tmp_path):
    import shutil

    import yaml

    entry = copy.deepcopy(next(e for e in library["entries"] if e["id"] == HX))
    asset = entry["assets"][0]
    destination = tmp_path / asset["path"]
    destination.parent.mkdir(parents=True)
    shutil.copyfile(library_root() / asset["path"], destination)
    record_path = tmp_path / f"{HX}.yaml"
    record_path.write_text(yaml.safe_dump(entry))
    (tmp_path / "sources").mkdir()
    document = copy.deepcopy(next(d for d in library["documents"] if d["id"] == DOC))
    document["source_type"] = "pdf"
    manifest_path = tmp_path / "sources/source.yaml"
    manifest_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="non-web document"):
        load_library(tmp_path)
    document["source_type"] = "web"
    manifest_path.write_text(yaml.safe_dump(document))
    entry["assets"][0]["web"]["document_id"] = "missing"
    record_path.write_text(yaml.safe_dump(entry))
    with pytest.raises(ValueError, match="unknown"):
        load_library(tmp_path)


@pytest.mark.parametrize("edition", [None, "2024", "2009"])
def test_replaced_entries_never_reenter_and_old_profile_respects_scope(edition):
    old_id = f"{DOC}.instruments-p-id-symbols.analyzer-transmitter"
    profile = dict(
        id="old",
        name="Older profile",
        version=1,
        confirmed=True,
        reference_ids=[old_id],
        context={"standards": [{"name": "ISA 5.1", "edition": edition}]},
    )
    snapshot = knowledge_snapshot("profile", profile)
    result = resolve(snapshot, "symbol_interpretation", query="analyzer transmitter")
    assert old_id not in {e["id"] for e in snapshot["entries"]}
    assert old_id not in result["reference_ids"]
    assert ("isa-5.1-2009.t5.1.1-1" in result["reference_ids"]) == (edition == "2009")
    # Never rewrite confirmed project context or silently declare ISA applicable.
    assert snapshot["profile"]["reference_ids"] == [old_id]
    assert snapshot["profile"]["context"]["standards"][0]["edition"] == edition
    snapshot["overrides"] = [{"pages": [2], "context": {"standards": []}}]
    assert (
        "isa-5.1-2009.t5.1.1-1"
        not in resolve(snapshot, "symbol_interpretation", 1, "analyzer transmitter")[
            "reference_ids"
        ]
    )


def test_overlap_history_assets_and_distinct_meanings(library):
    ids = {e["id"] for e in library["entries"]}
    for replacement in library["replacements"]:
        assert replacement["reference_id"] not in ids
        assert set(replacement["replacement_ids"]) <= ids
        assert catalog_asset(library_root(), replacement["reference_id"], "symbol").is_file()
    for suffix in [
        "strainers.flat-plate-strainer",
        "lines.electrical-supply",
        "lines.ultrasonic-signal",
        "instruments-p-id-symbols.computer-indicator",
        "heat-exchangers.plate-heat-exchanger",
        "pumps.centrifugal-pump-01",
    ]:
        assert f"{DOC}.{suffix}" in ids
    old = f"{DOC}.valve-symbols-for-p-id.motor-activated-valve"
    replacement = next(r for r in library["replacements"] if r["reference_id"] == old)
    assert len(replacement["replacement_ids"]) == 2  # Keep valve and actuator distinct.
    snapshot = knowledge_snapshot("general")
    changed = copy.deepcopy(snapshot)
    changed["replacements"][0]["reason"] += " Revised review."
    assert digest(changed) != digest(snapshot)
    assert validate_match(
        resolve(snapshot, "symbol_interpretation", query="motor valve"),
        old,
        "symbol",
        "M on drawing",
    )[1]


def test_changed_reviewed_image_requires_new_overlap_decision(library):
    import hashlib

    root = library_root()
    paths = [*root.glob("projectmaterials.*.yaml"), *root.glob("retired/projectmaterials.*.yaml")]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    script = Path(__file__).resolve().parents[2] / "scripts/import_projectmaterials.py"
    # Exercise the pure record guard without depending on ignored local downloads.
    spec = importlib.util.spec_from_file_location("overlap_import_test", script)
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    old_id = library["replacements"][0]["reference_id"]
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temporary:
        raw = catalog_asset(root, old_id, "symbol").read_bytes()
        with pytest.raises(ValueError, match="reconsider overlap"):
            importer.record(
                dict(
                    section="Instruments (P&ID Symbols)",
                    label="Changed",
                    alt="Changed",
                    url="https://blog.projectmaterials.com/image.png",
                    section_id="instruments",
                ),
                raw,
                "2026-10-02",
                Path(temporary),
                reviewed_hash="0" * 64,
            )
        assert not list(Path(temporary).rglob("*.png"))
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_offline_import_honors_review_on_every_run(library, tmp_path, monkeypatch):
    import hashlib
    import sys

    import yaml

    script = Path(__file__).resolve().parents[2] / "scripts/import_projectmaterials.py"
    spec = importlib.util.spec_from_file_location("reimport_overlap_test", script)
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    monkeypatch.setattr(importer, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script)])
    root = tmp_path / "knowledge/pid"
    cache = tmp_path / "data/knowledge-sources/projectmaterials"
    cache.mkdir(parents=True)
    replacement = next(
        r for r in library["replacements"] if r["reference_id"].endswith(".analyzer-transmitter")
    )
    old_id = replacement["reference_id"]
    archived = yaml.safe_load((library_root() / "retired" / f"{old_id}.yaml").read_text())
    asset = archived["assets"][0]
    raw = catalog_asset(library_root(), old_id, "symbol").read_bytes()
    url = asset["web"]["url"]
    (cache / (hashlib.sha256(url.encode()).hexdigest() + ".image")).write_bytes(raw)
    (cache / "page.html").write_text(
        f'<h2 id="instruments">Instruments (P&amp;ID Symbols)</h2><table><td><img alt="Analyzer Transmitter" src="{url}">Analyzer Transmitter</td></table>'
    )
    importer.dump(root / f"{old_id}.yaml", archived)
    importer.dump(
        root / "overlaps/test.yaml",
        dict(
            version=1,
            preferred_source="isa-5.1-2009",
            compared_source=DOC,
            policy="Prefer ISA",
            replacements=[replacement],
        ),
    )
    for target in replacement["replacement_ids"]:
        importer.dump(
            root / f"{target}.yaml", next(e for e in library["entries"] if e["id"] == target)
        )
    importer.main()
    assert not (root / f"{old_id}.yaml").exists()
    retired = root / "retired" / f"{old_id}.yaml"
    first = retired.read_bytes()
    importer.main()
    assert retired.read_bytes() == first and not (root / f"{old_id}.yaml").exists()
