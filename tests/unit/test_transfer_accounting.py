import json

import pytest

from eval.pid2graph.transfer_accounting import transfer_billing


def test_snapshot_scopes_requests_and_separates_unknown_costs(tmp_path):
    path = tmp_path / "ledger.json"
    rows = [{"id": i, "status": status, "actual_billed_usd": actual, "exposure_usd": exposure}
            for i, status, actual, exposure in [("billed", "settled", .002, .002),
                                              ("active", "pending", None, .2),
                                              ("unknown", "unresolved", None, .3),
                                              ("unrelated", "settled", 8, 8)]]
    path.write_text(json.dumps({"schema_version": 2, "requests": rows}))
    original = path.read_bytes()
    result = transfer_billing(path, ["billed", "active", "unknown"])
    b = result["billing"]
    assert b["actual_billed_usd"] == .002
    assert b["active_reservations_usd"] == .2
    assert b["unresolved_upper_usd"] == .3
    assert b["active_request_count"] == b["finished_unresolved_request_count"] == 1
    assert not b["actual_billing_complete"]
    assert result["reserved_or_charged_usd"] == pytest.approx(.502)
    assert path.read_bytes() == original


def test_legacy_exposure_is_never_reported_as_actual(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"requests": [{"id": "x", "charged_usd": 58.01}]}))
    result = transfer_billing(path, ["x"])
    assert "billing" not in result
    assert result["reserved_or_charged_usd"] == 58.01
    assert "unavailable" in result["accounting"]["method"]


@pytest.mark.parametrize("ids", [["missing"], ["x", "x"]])
def test_missing_or_duplicate_ids_cannot_hide_charges(tmp_path, ids):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"schema_version": 2, "requests": [{"id": "x"}]}))
    with pytest.raises(ValueError):
        transfer_billing(path, ids)
