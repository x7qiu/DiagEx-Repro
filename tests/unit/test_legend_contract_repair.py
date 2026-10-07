"""One bounded model repair preserves source geometry and strict row semantics."""

import json
from types import SimpleNamespace

import pytest

from diagex.config import Config
from diagex.extractors.pid_legend import _extract_from_page
from diagex.llm.cost import CostTracker
from tests.unit.test_native_legend_rows import fixture


def accepted(row_id):
    return {
        "row_id": row_id,
        "decision": "accept",
        "kind": "equipment",
        "symbol_class": "actuator",
        "attributes": {"symbol_role": "actuator"},
    }


def invalid(row_id, mode):
    row = accepted(row_id)
    if mode == "actuator":
        return [{**row, "kind": "valve"}]
    if mode == "shape":
        return [{**row, "candidate_shapes": ["pipe_vent"]}]
    if mode == "missing":
        return []
    if mode == "conflicting":
        return [row, {"row_id": row_id, "decision": "reject", "reason": "metadata"}]
    raise AssertionError(mode)


class RepairClient:
    def __init__(self, first="actuator", second="accept"):
        self.first = first
        self.second = second
        self.calls = []

    def messages_create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = [
            json.loads(b["text"]) for b in kwargs["messages"][0]["content"] if b["type"] == "text"
        ]
        phase = self.first if len(self.calls) == 1 else self.second
        if phase == "transport":
            raise RuntimeError("transport unavailable")
        rows = []
        for item in inputs:
            row_id = item["row_id"]
            if phase == "accept":
                rows.append(accepted(row_id))
            elif phase in {"uncertain", "reject"}:
                rows.append(
                    {
                        "row_id": row_id,
                        "decision": phase,
                        "reason": "source interpretation remains ambiguous",
                    }
                )
            else:
                rows.extend(invalid(row_id, phase))
        return SimpleNamespace(
            content=[{"type": "tool_use", "name": "submit_legend_rows", "input": {"rows": rows}}],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def run(client):
    rendered, evidence = fixture(1)
    coverage = []
    entries = _extract_from_page(
        page=rendered,
        region=None,
        client=client,
        cost_tracker=CostTracker(),
        cfg=Config(),
        page_evidence=evidence,
        coverage=coverage,
    )
    return entries, coverage


@pytest.mark.parametrize("mode", ["actuator", "shape", "missing", "conflicting"])
def test_repairs_only_contract_errors_once_with_same_source_and_diagnostics(mode):
    client = RepairClient(mode)
    entries, coverage = run(client)
    assert len(client.calls) == 2
    first, second = [call["messages"][0]["content"] for call in client.calls]
    assert first[1] == second[1]  # Exact same drawing crop; no navigation.
    initial, repair = json.loads(first[0]["text"]), json.loads(second[0]["text"])
    assert repair["row_id"] == initial["row_id"]
    assert repair["printed_label"] == initial["printed_label"]
    assert repair["previous_response"] == invalid(initial["row_id"], mode)
    assert repair["validation_errors"] and "repair_instruction" in repair
    assert entries[0].kind == "equipment"
    assert entries[0].label == "Full definition"
    assert entries[0].source_bbox.w > 30
    trace = json.loads(entries[0].attributes["classification_repair_evidence"])
    assert trace["previous_response"] == repair["previous_response"]
    assert coverage[0].status == "complete" and coverage[0].verification_passes == 2


@pytest.mark.parametrize("second", ["actuator", "shape", "missing", "transport"])
def test_failed_repair_stays_partial_and_does_not_relax_completeness_gate(second):
    client = RepairClient(second=second)
    entries, coverage = run(client)
    assert len(client.calls) == 2
    assert entries[0].attributes["row_status"] == "uncertain"
    assert coverage[0].status == "partial"
    assert coverage[0].failure_kind == ("transport" if second == "transport" else "contract")
    assert coverage[0].verification_passes == 2
    assert "Actuator role conflicts" in coverage[0].reason


def test_valid_uncertainty_is_not_repaired_or_forced_into_acceptance():
    client = RepairClient(first="uncertain")
    entries, coverage = run(client)
    assert len(client.calls) == 1
    assert entries[0].attributes["row_status"] == "uncertain"
    assert coverage[0].failure_kind == "ambiguity"


def test_repair_can_preserve_uncertainty():
    client = RepairClient(second="uncertain")
    entries, coverage = run(client)
    assert len(client.calls) == 2
    assert entries[0].attributes["row_status"] == "uncertain"
    assert coverage[0].failure_kind == "ambiguity"


def test_repaired_rejection_retains_original_response_and_errors_in_coverage():
    client = RepairClient(second="reject")
    entries, coverage = run(client)
    assert len(client.calls) == 2 and entries == []
    assert coverage[0].status == "complete" and coverage[0].entry_count == 0
    evidence = json.loads(coverage[0].reason.split("Initial response repair evidence: ")[1])
    assert evidence["previous_response"][0]["kind"] == "valve"
    assert "Actuator role conflicts" in evidence["validation_errors"][0]["msg"]


def test_malformed_rejection_recheck_gets_no_third_attempt():
    client = RepairClient(first="reject", second="actuator")
    entries, coverage = run(client)
    assert len(client.calls) == 2
    assert entries[0].attributes["row_status"] == "uncertain"
    assert coverage[0].failure_kind == "contract" and coverage[0].verification_passes == 2


def test_transport_failure_uses_no_schema_repair():
    client = RepairClient(first="transport")
    entries, coverage = run(client)
    assert len(client.calls) == 1
    assert coverage[0].failure_kind == "transport"
    assert "classification_repair_evidence" not in entries[0].attributes
