import copy

import pytest

from eval.pid2graph.training_report import exposure


def inputs():
    def sample(source, ids, labels, full):
        return {"source_id": source, "source_group": source, "reference_ids": ids,
                "labels": labels, "fully_covered_ids": full}
    data = {"sources": [{"id": s, "group": s, "split": "train"} for s in ("a", "b")],
            "samples": [sample("a", ["same", "v"], [1, 2], ["same"]),
                        sample("a", ["v"], [2], ["v"]),
                        sample("b", ["same"], [1], ["same"]), sample("b", [], [], [])],
            "symbol_counts": {"general": 2, "valve": 1}}
    schedule = {"seed": 1, "maximum_epochs": 3, "batch_size": 2,
                "sampling": {"base_crops": 4, "extra_background_draws": 1,
                             "rare_classes": {}, "epoch_presentations": 5}}
    return data, schedule


def test_overlap_repeats_and_same_ids_on_different_drawings_are_counted_correctly():
    data, schedule = inputs()
    result = exposure(data, schedule, {"epoch": 3, "cursor": 0, "presentations": 10, "step": 6})
    assert result["crop_presentations"] == 10 and result["unique_crops_presented"] == 4
    assert result["fully_seen_physical_symbols"] == result["physical_symbols"] == 3
    assert result["background_presentations"] == 4 and result["completed_sampling_epochs"] == 2
    assert result["per_class"]["general"]["fully_seen_symbols"] == 2
    assert result["per_class"]["valve"] == {"physical_symbols": 1, "fully_seen_symbols": 1,
                                          "partially_or_fully_seen_symbols": 1,
                                          "annotation_presentations_including_repeats": 4}
    assert result["per_drawing"]["a"]["physical_symbols"] == 2
    assert result["per_drawing"]["b"]["physical_symbols"] == 1


def test_final_short_batch_equals_next_epoch_zero_cursor():
    data, schedule = inputs()
    before = exposure(data, schedule, {"epoch": 1, "cursor": 5, "presentations": 5, "step": 3})
    after = exposure(data, schedule, {"epoch": 2, "cursor": 0, "presentations": 5, "step": 3})
    for key in ("fully_seen_physical_symbols", "per_class", "per_drawing", "completed_sampling_epochs"):
        assert before[key] == after[key]


@pytest.mark.parametrize("state", [
    {"epoch": 1, "cursor": 1, "presentations": 1, "step": 1},
    {"epoch": 2, "cursor": 0, "presentations": 6, "step": 3},
    {"epoch": 2, "cursor": 0, "presentations": 5, "step": 2},
    {"epoch": 4, "cursor": 2, "presentations": 17, "step": 10},
])
def test_invalid_committed_position_is_rejected(state):
    with pytest.raises(ValueError):
        exposure(*inputs(), state)


def test_nontraining_source_and_inconsistent_physical_inventory_are_rejected():
    data, schedule = inputs()
    state = {"epoch": 1, "cursor": 0, "presentations": 0, "step": 0}
    bad = copy.deepcopy(data)
    bad["sources"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="splits"):
        exposure(bad, schedule, state)
    bad = copy.deepcopy(data)
    bad["samples"][1]["labels"] = [1]
    with pytest.raises(ValueError, match="disagree"):
        exposure(bad, schedule, state)
    bad = copy.deepcopy(data)
    bad["symbol_counts"]["general"] = 3
    with pytest.raises(ValueError, match="inventory"):
        exposure(bad, schedule, state)
