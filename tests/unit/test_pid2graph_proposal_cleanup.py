from eval.pid2graph.proposal_cleanup import remove_contained


def box(identity, coords, label="general", confidence=0.9):
    return {"id": identity, "bbox": coords, "label": label, "confidence": confidence}


def test_fragments_removed_but_nested_other_classes_and_adjacent_symbols_survive():
    rows = [box("whole", [0, 0, 100, 100]), box("fragment", [2, 2, 98, 35], confidence=0.99),
            box("instrument", [30, 30, 55, 55], "instrumentation"), box("adjacent", [110, 0, 150, 50])]
    kept, rejected = remove_contained(rows)
    assert {r["id"] for r in kept} == {"whole", "instrument", "adjacent"}
    assert rejected[0]["id"] == "fragment"
    assert rejected[0]["retained_id"] == "whole"


def test_weak_large_box_cannot_erase_confident_smaller_object():
    rows = [box("weak", [0, 0, 100, 100], confidence=0.2), box("object", [25, 25, 60, 60])]
    kept, rejected = remove_contained(rows)
    assert len(kept) == 2
    assert rejected == []
