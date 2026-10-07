import pytest

from eval.pid2graph.raster_decision_audit import summarize_case


def tile(ids, decisions):
    return {"requested_raster_proposal_ids": ids, "raster_guides_omitted_by_limit": 0,
            "raster_review_attempts": [{"tool_results": [{"decisions": [
                {"proposal_id": i, "status": s} for i, s in decisions], "unrecognized_rows": []}]}]}


def test_multiple_view_disagreement_is_not_called_a_rejection():
    proposals = [{"id": str(i), "label": "valve"} for i in range(6)]
    tiles = [tile(["0", "1", "2", "3", "4"], [("0", "symbol"), ("1", "reject"), ("2", "uncertain"), ("3", "symbol"), ("4", "symbol")]),
             tile(["0"], [("0", "reject")])]
    predictions = [{"attributes": {"raster_proposal_id": i, "raster_vlm_decision": "symbol"}} for i in ["0", "3"]]
    result = summarize_case(proposals, tiles, predictions)
    assert result["ids"] == {"never_requested": ["5"], "recognized": ["0", "3", "4"],
        "retained_in_output": ["0", "3"], "recognized_without_retained_output": ["4"], "rejected_only": ["1"],
        "unresolved_without_recognition": ["2"], "recognition_rejection_conflict": ["0"]}


def test_failed_request_does_not_become_model_rejection_or_confirmation():
    proposals = [{"id": "0", "label": "valve"}]
    failed = {"requested_raster_proposal_ids": ["0"], "raster_guides_omitted_by_limit": 0, "raster_review_attempts": []}
    result = summarize_case(proposals, [failed], [])
    assert result["counts"]["unresolved_without_recognition"] == result["proposal_views_without_response"] == 1
    forged = [{"attributes": {"raster_proposal_id": "0", "raster_vlm_decision": "symbol"}}]
    with pytest.raises(ValueError, match="explicit symbol decision"):
        summarize_case(proposals, [failed], forged)
