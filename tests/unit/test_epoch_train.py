import copy
import math
from collections import Counter

import pytest

from eval.pid2graph.epoch_train import epoch_order, learning_rate, selection_state


def schedule():
    return {"seed": 123, "sampling": {"base_crops": 5, "extra_background_draws": 3,
            "rare_classes": {"3": {"available_crops": 2, "extra_draws": 4}}, "epoch_presentations": 12},
            "minimum_macro_f1_improvement": 0.002, "minimum_epochs": 2, "early_stopping_patience": 2}


def test_sampler_covers_every_base_crop_and_resumes_without_reordering():
    samples = [{"labels": labels} for labels in [[1], [], [3], [3, 2], [6]]]
    plan = schedule()
    order = epoch_order(samples, plan, 1)
    counts = Counter(order)
    assert len(order) == 12 and set(order) == set(range(5))
    assert counts[1] == 4 and counts[2] + counts[3] == 6
    assert counts[0] == counts[4] == 1
    assert order[:7] + epoch_order(samples, plan, 1)[7:] == order
    assert epoch_order(samples, plan, 2) != order
    broken = copy.deepcopy(plan)
    broken["sampling"]["rare_classes"]["3"]["available_crops"] = 1
    with pytest.raises(ValueError, match="coverage"):
        epoch_order(samples, broken, 1)


def test_early_stopping_uses_significant_improvement_but_selection_uses_best_score():
    history = [{"epoch": 0, "macro_drawing_f1": 0.5}, {"epoch": 1, "macro_drawing_f1": 0.501}]
    state = selection_state(history, schedule())
    assert state["best_epoch"] == 1 and not state["early_stop"]
    history.append({"epoch": 2, "macro_drawing_f1": 0.5015})
    state = selection_state(history, schedule())
    assert state["best_epoch"] == 2 and state["early_stop"]
    history[2]["macro_drawing_f1"] = 0.503
    state = selection_state(history, schedule())
    assert not state["early_stop"] and state["stale_epochs"] == 0
    history.append({"epoch": 3, "macro_drawing_f1": 0.503})
    assert selection_state(history, schedule())["best_epoch"] == 2


def test_learning_rate_does_not_restart_on_epoch_boundaries():
    assert learning_rate(1, 1000, 0.001) == 0.00001
    assert learning_rate(100, 1000, 0.001) == 0.001
    assert 0.0001 < learning_rate(500, 1000, 0.001) < learning_rate(101, 1000, 0.001)
    assert math.isclose(learning_rate(1000, 1000, 0.001), 0.0001)
