"""Small, credential-free summaries of request failures and saved diagnostics."""

import json
import re


def safe_error(value):
    text = str(value)
    text = re.sub(r"(?i)(bearer\s+)[^\s\"',;]+", r"\1[redacted]", text)
    text = re.sub(r"sk-[\w-]+", "[redacted]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|authorization|access_token)[\"']?\s*[:=]\s*[\"']?)[^\s\"',;&]+", r"\1[redacted]", text)
    return text[:2000]


def error_code(error, error_type="", status=None):
    value = (str(error) + " " + error_type).lower()
    if "time budget" in value or "deadline" in value:
        return "request_deadline"
    if "timeout" in value or "timed out" in value:
        return "transport_timeout"
    if status == 429:
        return "rate_limit"
    if status in {401, 403}:
        return "authentication"
    if status is not None:
        return "provider_error"
    if any(s in value for s in ("connection", "protocol", "transport")):
        return "connection_error"
    if any(s in value for s in ("format", "validation", "structured", "schema")):
        return "invalid_response"
    return "request_error"


LABELS = {
    "request_deadline": "请求超过总时限，尚未获得完整识别结果",
    "transport_timeout": "网络连接或读取超时，尚未获得完整识别结果",
    "rate_limit": "模型服务限流，请求未完成",
    "authentication": "模型服务认证或权限错误",
    "provider_error": "模型服务返回错误",
    "connection_error": "网络连接中断，请求未完成",
    "invalid_response": "模型返回格式未通过校验",
    "request_error": "识别请求失败，需查看错误详情",
    "not_processed": "运行已停止，此候选尚未处理",
}


def request_summary(request):
    blocks = [b for m in request.get("messages", []) for b in m.get("content", []) if isinstance(b, dict)]
    summary = {
        "image_count": sum(b.get("type") == "image" for b in blocks),
        "text_characters": len(request.get("system", "")) + sum(len(b.get("text", "")) for b in blocks),
        "time_budget_s": request.get("time_budget_s"),
        "max_attempts": request.get("max_attempts"),
    }
    for b in blocks:
        text = b.get("text", "")
        start = text.find('{"page_index"')
        if start < 0:
            continue
        try:
            payload = json.JSONDecoder().raw_decode(text[start:])[0]
        except ValueError:
            continue
        summary.update(
            candidate_ids=[c["candidate_id"] for c in payload.get("native_symbol_candidates", [])],
            legend_entry_count=len(payload.get("legend_entries", [])),
        )
        break
    return summary


def failure_summary(events, exc=None):
    errors = [e for e in events if e.get("phase") == "request_error"]
    error = errors[-1] if errors else {}
    attempt = error.get("attempt")
    current = [e for e in events if attempt is None or e.get("attempt") == attempt]
    requests = [e for e in current if e.get("phase") == "request"]
    request = requests[-1] if requests else {}
    summary = request.get("summary") or request_summary(request.get("request", {}))
    transports = [e["transport"] for e in current if e.get("phase") == "transport"]
    terminal = next((e for e in reversed(transports) if e.get("event") in {"deadline", "error"}), {})
    detail = safe_error(error.get("error") or terminal.get("error") or exc or "request failed")
    code = terminal.get("error_code") or error_code(detail, type(exc).__name__ if exc else "", terminal.get("status_code"))
    return {
        **summary,
        **{k: terminal[k] for k in (
            "elapsed_s", "time_budget_s", "remaining_budget_s", "first_event_s", "first_content_s",
            "stream_events", "content_deltas", "response_id", "status_code", "error_type", "retry_reason",
        ) if k in terminal},
        "attempts": max((e.get("attempt", 0) for e in transports if e.get("event") == "dispatch"), default=0),
        "code": code, "label": LABELS[code], "error": detail,
    }
