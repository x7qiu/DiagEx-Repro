import copy

import pytest

from eval.pid2graph.data import digest
from eval.pid2graph.decision_outcomes import GROUPS, join
from eval.pid2graph.scoring import score_drawing


def fixture():
    def p(i, label="valve"):
        return {"id": str(i), "bbox": [i * 20, 0, i * 20 + 10, 10], "label": label, "confidence": .9}

    graph = {"nodes": [p(i) for i in range(5)]}
    dconfig = {"variant": "supervised_detector"}
    hconfig = {"variant": "explicit_proposal_decisions", "guidance": {"detector_config": dconfig}}
    reports = []
    for config, predictions in [(dconfig, [p(i) for i in range(6)]),
                                (hconfig, [p(1), p(3, "pump"), p(4)])]:
        reports.append({"manifest_sha256": "m", "panel_sha256": "p", "split": "validation",
                        "config": config, "config_sha256": digest(config), "summary": {}, "runtime_seconds": 1,
                        "cases": [{**score_drawing(graph, predictions), "id": identity,
                                   "collection": "synthetic", "status": "partial"}
                                  for identity in ("a", "b")]})
    ids = dict(zip(GROUPS, [["0"], ["1", "5"], ["2"], ["3"], ["4"]], strict=True))
    audit = {"panel_sha256": "p", "split": "validation", "run_config_sha256": "c",
             "attempted_coverage_complete": True,
             "cases": [{"id": i, "detector_proposals": 6, "ids": copy.deepcopy(ids)} for i in ("a", "b")]}
    return *reports, audit


def test_rejected_truth_can_be_recovered_and_repeated_ids_are_scoped():
    result = join(*fixture(), "c")["groups"]
    assert result["rejected_only"]["proposals"] == 4
    assert result["rejected_only"]["detector_false_positives"] == 2
    assert result["rejected_only"]["truth_found_in_hybrid"] == 2
    assert result["rejected_only"]["truth_lost_in_hybrid"] == 0
    assert result["recognized_without_retained_output"]["lost_but_localized"] == 2
    assert result["unresolved_without_recognition"]["lost_without_localization"] == 2
    assert sum(g["detector_true_positives"] for g in result.values()) == 10


@pytest.mark.parametrize("change", ["duplicate", "missing", "foreign_run", "foreign_detector"])
def test_rejects_inconsistent_evidence(change):
    d, h, a = fixture()
    if change == "duplicate":
        a["cases"][0]["ids"]["retained_in_output"].append("1")
    elif change == "missing":
        a["cases"].pop()
    elif change == "foreign_run":
        a["run_config_sha256"] = "wrong"
    else:
        h["config"]["guidance"]["detector_config"] = {"different": True}
    with pytest.raises(ValueError):
        join(d, h, a, "c")
