import json

import pytest

from diagex.llm.diagnostics import error_code, failure_summary, safe_error
from diagex.web.evidence_origin import recognition_diagnostic


def saved_events():
    request = {"time_budget_s": 45, "max_attempts": 2, "system": "test", "messages": [{
        "content": [{"type": "image", "source": {"data_sha256": "abc"}},
                    {"type": "text", "text": 'Inspect\n' + json.dumps({
                        "page_index": 3, "native_symbol_candidates": [{"candidate_id": "pi201"}],
                        "legend_entries": [{"label": "PI"}],
                    })}],
    }]}
    return [
        {"phase": "request", "attempt": 1, "request": request},
        {"phase": "transport", "attempt": 1, "transport": {"event": "dispatch", "attempt": 1}},
        {"phase": "transport", "attempt": 1, "transport": {
            "event": "deadline", "attempt": 1, "elapsed_s": 47.7,
            "error": "stream exceeded request time budget",
        }},
        {"phase": "request_error", "attempt": 1, "error": "stream exceeded request time budget"},
    ]


def test_old_timeout_recovers_candidate_budget_and_does_not_invent_stream_progress():
    result = failure_summary(saved_events())
    assert result["candidate_ids"] == ["pi201"]
    assert result["time_budget_s"] == 45 and result["elapsed_s"] == 47.7
    assert result["attempts"] == 1 and result["max_attempts"] == 2
    assert result["image_count"] == 1 and result["legend_entry_count"] == 1
    assert "first_event_s" not in result
    assert recognition_diagnostic({"request_failure": result, "candidate_only": True})["code"] == "request_deadline"


@pytest.mark.parametrize("error,kind,status,expected", [
    ("stream exceeded request time budget", "TimeoutError", None, "request_deadline"),
    ("read timed out", "ReadTimeout", None, "transport_timeout"),
    ("rate limit", "APIStatusError", 429, "rate_limit"),
    ("unauthorized", "APIStatusError", 401, "authentication"),
    ("server error", "APIStatusError", 503, "provider_error"),
    ("connection failed", "APIConnectionError", None, "connection_error"),
    ("invalid response", "PerceptionResponseFormatError", None, "invalid_response"),
])
def test_error_categories(error, kind, status, expected):
    assert error_code(error, kind, status) == expected


def test_credentials_are_redacted_and_error_without_transport_still_has_cause():
    secret = 'Authorization: Bearer secret-token api_key=secret-value sk-or-private'
    value = safe_error(secret)
    assert all(s not in value for s in ('secret-token', 'secret-value', 'sk-or-private'))
    assert failure_summary([], ValueError('invalid schema'))['code'] == 'invalid_response'


def test_not_processed_is_distinct_from_uncertain_and_failed():
    assert recognition_diagnostic({'processing_status': 'not_processed'})['code'] == 'not_processed'
    assert recognition_diagnostic({'status': 'uncertain'})['code'] == 'uncertain'


def test_failed_crop_retains_specific_error_when_guard_stops_later_work(tmp_path, monkeypatch):
    import fitz

    from diagex.config import Config, LLMConfig
    from diagex.extractors.pid_evidence import run_pid_evidence_extract
    from diagex.vision.models import BBox
    from diagex.vision.symbol_candidates import SymbolCandidate
    from diagex.vision.symbol_detection import SymbolDetectionResult

    pdf = tmp_path / "drawing.pdf"
    with fitz.open() as doc:
        for _ in range(2):
            page = doc.new_page(width=400, height=200)
            page.insert_text((20, 20), "P&ID TEST")
            page.draw_circle((200, 100), 20)
            page.insert_text((191, 100), "PI", fontsize=8)
            page.insert_text((190, 110), "201", fontsize=8)
            page.draw_line((220, 100), (350, 100))
        doc.save(pdf)

    class FailedClient:
        def __init__(self, config, budgets=None):
            self.config, self.retries_total = config, 0

        def reset_retry_counter(self):
            self.retries_total = 0

        def messages_create(self, **kwargs):
            callback = kwargs.get("on_transport_event")
            if callback:
                callback({"event": "dispatch", "attempt": 1})
                callback({"event": "deadline", "attempt": 1, "elapsed_s": 45,
                          "error": "stream exceeded request time budget"})
            raise TimeoutError("stream exceeded request time budget")

    monkeypatch.setattr("diagex.extractors.pid_evidence.LLMClient", FailedClient)
    def candidates(*, page):
        return SymbolDetectionResult(page_index=page.page_index, native_candidates=[SymbolCandidate(
            id=f"pi-{page.page_index}", page_index=page.page_index, shape="round_symbol",
            bbox=BBox(x=180, y=80, w=40, h=40), source_path_ids=["circle"], text=["PI", "201"],
        )])
    monkeypatch.setattr("diagex.extractors.pid_evidence.detect_symbols", candidates)
    cfg = Config(llm=LLMConfig(model="fake", anthropic_api_key="unused"))
    cfg.runs_dir = tmp_path / "runs"
    cfg.symbol_perception.total_failure_limit = 1
    result = run_pid_evidence_extract(
        diagram=pdf, symbol_standard="none", legend_path=None, legend_pages=None,
        legend_region=None, no_legend=True, legend_key=None, effort="medium", config=cfg,
        persist=True, out_path=None, confidence_report_path=None, console=None, stop_after="detection",
    )
    saved = json.loads((result.run_dir / "detection.json").read_text())
    failed = [r for r in saved["reviews"] if r.get("request_failure")]
    skipped = [r for r in saved["reviews"] if r.get("processing_status") == "not_processed"]
    assert failed and skipped
    assert all(r["request_failure"]["code"] == "request_deadline" for r in failed)
    assert all("request_failure" not in r for r in skipped)
    errors = json.loads((result.run_dir / "request.errors.json").read_text())
    assert len(errors["failures"]) == 1
    assert list((result.run_dir / "checkpoints/perception_errors").glob("*.json"))
