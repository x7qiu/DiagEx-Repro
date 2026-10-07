"""Standards letter matrices: fidelity anchors, retrieval and compatibility."""

import copy
import json
import string

import pytest
from pydantic import ValidationError

from diagex.knowledge import resolver
from diagex.knowledge.library import load_library, model_assets, supplied_trace
from diagex.knowledge.models import LETTER_COLUMNS, Entry, LetterCell, LetterMatrix

ISA = "isa-5.1-2009.t4.1-p30"
SHT = "sht-3101-2017.letter-table"


@pytest.fixture(scope="module")
def library():
    return load_library(resolver.library_root())


@pytest.fixture
def catalog(library, monkeypatch):
    monkeypatch.setattr(resolver, "load_library", lambda root: copy.deepcopy(library))
    return {e["id"]: e for e in library["entries"]}


def meaning(entry, letter, column):
    return next(r for r in entry["letter_matrix"]["rows"] if r["letter"] == letter)["cells"][column]


def test_complete_tables_and_source_checked_differences(catalog):
    assert {e["id"] for e in catalog.values() if e.get("letter_matrix")} == {ISA, SHT}
    for identity in [ISA, SHT]:
        e = catalog[identity]
        assert e["catalog"]["role"] == "supporting"
        assert e["reference_type"] == "convention"
        assert [r["letter"] for r in e["letter_matrix"]["rows"]] == list(string.ascii_uppercase)
        assert all(set(r["cells"]) == set(LETTER_COLUMNS) for r in e["letter_matrix"]["rows"])
    # Independently checked against the original table images, not a shared template.
    assert meaning(catalog[ISA], "M", "measured_variable")["state"] == "user_choice"
    assert "Motor" in meaning(catalog[SHT], "M", "measured_variable")["text"]
    assert meaning(catalog[ISA], "I", "output_function")["state"] == "blank"
    assert meaning(catalog[SHT], "I", "output_function")["text"] == "输入 Input"
    assert "Integrate" in meaning(catalog[ISA], "Q", "variable_modifier")["text"]
    assert meaning(catalog[SHT], "Q", "variable_modifier")["state"] == "blank"
    assert meaning(catalog[ISA], "B", "output_function")["state"] == "user_choice"
    assert meaning(catalog[SHT], "B", "output_function")["state"] == "blank"
    assert "Unclassified" in meaning(catalog[ISA], "X", "measured_variable")["text"]
    assert "On off Valve" in meaning(catalog[SHT], "X", "measured_variable")["text"]
    assert "现场手动" not in json.dumps([catalog[ISA], catalog[SHT]], ensure_ascii=False)
    assert all(r["reference_id"] in catalog for r in catalog[ISA]["letter_matrix"]["rows"])
    assert len([i for i in catalog if i.startswith("isa-5.1-2009.t4.1-letter-")]) == 26


def test_notes_are_separate_and_dangling_scan_marker_is_explicit(catalog):
    c = meaning(catalog[ISA], "C", "measured_variable")
    assert c["text"] == "User’s Choice" and c["footnotes"] == ["3a", "5"]
    assert len(catalog[ISA]["letter_matrix"]["notes"]) == 30
    x = meaning(catalog[SHT], "X", "readout_function")
    assert x["footnotes"] == ["8"] and "(8)" not in x["text"]
    notes = {n["marker"]: n for n in catalog[SHT]["letter_matrix"]["notes"]}
    assert set(notes) == set("abcdefg") | {"8"}
    assert notes["8"]["status"] == "unresolved"
    assert notes["8"]["source"]["document_id"] == "sht-3101-2017"
    assert all(n["source"]["pdf_page"] == 15 for k, n in notes.items() if k != "8")


@pytest.mark.parametrize(
    "sources,expected",
    [
        (["isa-5.1-2009"], {ISA}),
        (["sht-3101-2017"], {SHT}),
        (["isa-5.1-2009", "sht-3101-2017"], {ISA, SHT}),
        ([], set()),
    ],
)
def test_selected_sources_keep_named_rules_and_budgets(catalog, sources, expected):
    snap = resolver.knowledge_snapshot("general", source_ids=sources)
    for task in ["symbol_interpretation", "text_assignment"]:
        result = resolver.resolve(snap, task, query="instrument letter table matrix")
        matrices = {e["id"]: e for e in result["references"] if e.get("letter_matrix")}
        assert set(matrices) == expected
        assert len(result["references"]) <= 8 and len(model_assets(result)) <= 2
        assert len(json.dumps(result["references"], ensure_ascii=False)) <= 40000
        for e in matrices.values():
            assert (
                meaning(e, "M", "measured_variable")["text"]
                == meaning(catalog[e["id"]], "M", "measured_variable")["text"]
            )
            assert e["letter_matrix"]["notes"]
        traced = {e["id"] for e in supplied_trace(result)["references"]}
        assert expected <= traced
    assert resolver.resolve({}, "symbol_interpretation") == {}


def test_unknown_conflicting_editions_overrides_and_drawing_precedence(catalog):
    assert not {ISA, SHT} & set(
        resolver.applicable_references(resolver.knowledge_snapshot("general"))["reference_ids"]
    )
    profile = dict(
        id="standards",
        name="Standards",
        version=1,
        confirmed=True,
        context={
            "standards": [
                {"name": "ISA 5.1", "edition": "2024"},
                {"name": "SH/T 3101", "edition": None},
            ]
        },
    )
    snap = resolver.knowledge_snapshot(
        "profile", profile, source_ids=["isa-5.1-2009", "sht-3101-2017"]
    )
    assert not {ISA, SHT} & set(resolver.applicable_references(snap)["reference_ids"])
    profile["context"]["standards"] = [{"name": "ISA 5.1", "edition": "2009"}]
    snap = resolver.knowledge_snapshot(
        "profile", profile, overrides=[{"pages": [2], "context": {"standards": []}}]
    )
    assert ISA in resolver.applicable_references(snap, 0)["reference_ids"]
    assert ISA not in resolver.applicable_references(snap, 1)["reference_ids"]
    definitions = [{"label": "M", "description": "Explicit project override"}]
    result = resolver.resolve(
        snap, "symbol_interpretation", drawing_definitions=definitions, query="letter matrix"
    )
    assert result["drawing_definitions"] == definitions
    assert any("take precedence" in s for s in result["core_principles"])
    assert resolver.knowledge_snapshot("off", source_ids=["isa-5.1-2009"]) == {}


def test_schema_preserves_legacy_and_rejects_broken_matrix(catalog):
    entry = {k: copy.deepcopy(v) for k, v in catalog[ISA].items() if k != "image_sha256"}
    old = copy.deepcopy(entry)
    old.pop("letter_matrix")
    assert Entry.model_validate(old).letter_matrix is None
    for mutate in [
        lambda m: m["rows"].pop(),
        lambda m: m["rows"][0]["cells"].pop("readout_function"),
        lambda m: m["rows"][0]["cells"]["measured_variable"].update(footnotes=["999"]),
    ]:
        matrix = copy.deepcopy(entry["letter_matrix"])
        mutate(matrix)
        with pytest.raises(ValidationError):
            LetterMatrix.model_validate(matrix)
    with pytest.raises(ValidationError):
        LetterCell(state="blank", text="guessed")
    with pytest.raises(ValidationError):
        LetterCell(state="user_choice")
    assert LetterCell(state="unreadable").text is None


def test_rule_and_note_changes_invalidate_identity_without_mutating_saved_snapshot(
    library, monkeypatch
):
    data = copy.deepcopy(library)
    monkeypatch.setattr(resolver, "load_library", lambda root: copy.deepcopy(data))
    before = resolver.knowledge_snapshot("general", source_ids=["isa-5.1-2009"])
    saved = json.loads(json.dumps(before))
    target = next(e for e in data["entries"] if e["id"] == ISA)
    target["letter_matrix"]["rows"][0]["cells"]["measured_variable"]["text"] = "Changed meaning"
    after = resolver.knowledge_snapshot("general", source_ids=["isa-5.1-2009"])
    assert before["identity"] != after["identity"]
    target["letter_matrix"]["notes"][0]["text"] += " Updated note."
    assert (
        resolver.knowledge_snapshot("general", source_ids=["isa-5.1-2009"])["identity"]
        != after["identity"]
    )
    assert before == saved
