"""Configuration and client-construction tests for the OpenRouter transport."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from diagex.config import LLMConfig, RuntimeBudgets
from diagex.llm.client import LLMClient, is_non_retryable_api_error


def test_request_deadline_stops_active_stream_without_retry(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("diagex.llm.client.time.monotonic", lambda: now[0])
    calls = []
    events = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for _ in range(20):
                now[0] += 1
                yield {"type": "ping"}

        def get_final_message(self):
            pytest.fail("An over-budget stream must not be accepted")

    client = LLMClient(LLMConfig(model="test", anthropic_api_key="test"))

    def stream(**kwargs):
        calls.append(kwargs)
        return Stream()

    monkeypatch.setattr(client._client.messages, "stream", stream)
    with pytest.raises(TimeoutError, match="stream exceeded"):
        client.messages_create(system="test", messages=[], max_tokens=100, time_budget_s=3, max_attempts=2, on_transport_event=events.append)
    assert len(calls) == 1 and client.retries_total == 0
    assert calls[0]["timeout"].read == 3
    last = events[-1]
    assert last["error_code"] == "request_deadline"
    assert last["stream_events"] == 3 and last["first_event_s"] == 1
    assert last["content_deltas"] == 0 and last["first_content_s"] is None
    assert last["time_budget_s"] == 3 and last["max_attempts"] == 2
    assert last["retry_reason"] == "budget_exhausted"
    client._client.close()


def test_bounded_request_does_not_wait_past_provider_cooldown(monkeypatch):
    monkeypatch.setattr("diagex.llm.client.time.monotonic", lambda: 100)
    monkeypatch.setattr("diagex.llm.client.time.sleep", lambda _: pytest.fail("must not wait"))
    client = LLMClient(LLMConfig(model="test", anthropic_api_key="test"))
    client._rate_limit_not_before = 200
    with pytest.raises(TimeoutError, match="cooldown"):
        client.messages_create(system="test", messages=[], max_tokens=100, time_budget_s=10)
    client._client.close()


def test_per_request_attempt_limit_overrides_long_global_retry_ladder(monkeypatch):
    calls = []
    events = []
    client = LLMClient(LLMConfig(model="test", anthropic_api_key="test"), RuntimeBudgets(retry_attempts=6, retry_base_s=0))

    def stream(**kwargs):
        calls.append(kwargs)
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://test.invalid"))

    monkeypatch.setattr(client._client.messages, "stream", stream)
    monkeypatch.setattr("diagex.llm.client.time.sleep", lambda _: None)
    with pytest.raises(anthropic.APIConnectionError):
        client.messages_create(system="test", messages=[], max_tokens=100, time_budget_s=45, max_attempts=2, on_transport_event=events.append)
    assert len(calls) == 2 and client.retries_total == 1
    assert [e["event"] for e in events] == ["dispatch", "error", "retry_wait", "dispatch", "error"]
    assert "Connection error" in events[-1]["error"]
    client._client.close()


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DIAGEX_LLM_PROVIDER",
        "DIAGEX_MODEL",
        "DIAGEX_REASONING",
        "OPENROUTER_MODEL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_APP_TITLE",
        "KIMI_API_KEY",
        "KIMI_BASE_URL",
        "KIMI_MODEL",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_api_error_classification(status: int) -> None:
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
    )
    error = anthropic.APIStatusError("fatal", response=response, body=None)

    assert is_non_retryable_api_error(error)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_api_error_classification(status: int) -> None:
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
    )
    error = anthropic.APIStatusError("retryable", response=response, body=None)

    assert not is_non_retryable_api_error(error)


def test_openrouter_config_from_explicit_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("DIAGEX_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DIAGEX_MODEL", "vendor/vision-model")

    config = LLMConfig.from_env()

    assert config.transport == "openrouter"
    assert config.model == "vendor/vision-model"
    assert config.openrouter_api_key == "test-key"
    assert config.openrouter_base_url == "https://openrouter.ai/api"


def test_openrouter_is_auto_detected_when_it_is_the_only_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/vision-model")

    assert LLMConfig.from_env().transport == "openrouter"


def test_openrouter_requires_an_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("DIAGEX_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(ValueError, match="requires a model slug"):
        LLMConfig.from_env()


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("auto", "auto"),
        ("enabled", "enabled"),
        ("on", "enabled"),
        ("disabled", "disabled"),
        ("false", "disabled"),
    ],
)
def test_reasoning_mode_is_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: str,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("DIAGEX_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DIAGEX_MODEL", "vendor/vision-model")
    monkeypatch.setenv("DIAGEX_REASONING", env_value)

    assert LLMConfig.from_env().reasoning_mode == expected


def test_invalid_reasoning_mode_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("DIAGEX_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("DIAGEX_MODEL", "vendor/vision-model")
    monkeypatch.setenv("DIAGEX_REASONING", "sometimes")

    with pytest.raises(ValueError, match="DIAGEX_REASONING"):
        LLMConfig.from_env()


def test_openrouter_client_uses_messages_endpoint_and_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_anthropic(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("diagex.llm.client.anthropic.Anthropic", fake_anthropic)

    LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
            openrouter_http_referer="https://example.test/diagex",
            openrouter_app_title="DiagEx Test",
        )
    )

    # The Anthropic SDK appends /v1/messages to this base URL.
    assert captured["base_url"] == "https://openrouter.ai/api"
    assert captured["auth_token"] == "test-key"
    assert captured["default_headers"] == {
        "HTTP-Referer": "https://example.test/diagex",
        "X-OpenRouter-Title": "DiagEx Test",
    }
    assert captured["max_retries"] == 0


def test_openrouter_preserves_native_images_tools_and_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        def stream(self, **kwargs: Any) -> FakeStream:
            captured.update(kwargs)
            return FakeStream()

    fake_client = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: fake_client),
    )
    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        )
    )
    image = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "aW1hZ2U=",
        },
    }
    tool = {
        "name": "get_overview",
        "description": "View the page",
        "input_schema": {"type": "object", "properties": {}},
    }

    result = client.messages_create(
        system=[{"type": "text", "text": "system"}],
        messages=[{"role": "user", "content": [image]}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "get_overview"},
        max_tokens=2048,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "medium"},
    )

    assert result is expected
    assert captured["model"] == "vendor/vision-model"
    assert captured["messages"][0]["content"][0] is image
    assert captured["tools"][0]["name"] == "get_overview"
    assert captured["tool_choice"] == {"type": "tool", "name": "get_overview"}
    assert captured["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert captured["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize(
    ("reasoning_mode", "expected_thinking"),
    [
        ("enabled", {"type": "adaptive", "display": "summarized"}),
        ("disabled", {"type": "disabled"}),
    ],
)
def test_reasoning_mode_overrides_caller_thinking(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_mode: str,
    expected_thinking: dict[str, str],
) -> None:
    captured: dict[str, Any] = {}
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        def stream(self, **kwargs: Any) -> FakeStream:
            captured.update(kwargs)
            return FakeStream()

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
            reasoning_mode=reasoning_mode,  # type: ignore[arg-type]
        )
    )

    client.messages_create(
        system="system",
        messages=[],
        max_tokens=32,
        thinking={"type": "disabled" if reasoning_mode == "enabled" else "adaptive"},
    )

    assert captured["thinking"] == expected_thinking


def test_per_call_reasoning_override_supports_structured_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        def stream(self, **kwargs: Any) -> FakeStream:
            captured.update(kwargs)
            return FakeStream()

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="deepseek/reasoner",
            openrouter_api_key="test-key",
            reasoning_mode="enabled",
        )
    )

    client.messages_create(
        system="system",
        messages=[],
        max_tokens=32,
        thinking={"type": "adaptive"},
        reasoning_mode_override="disabled",
    )

    assert captured["thinking"] == {"type": "disabled"}


def test_openrouter_honors_retry_after_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    sleeps: list[float] = []

    response = httpx.Response(
        429,
        headers={"Retry-After": "30"},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
    )
    rate_limit = anthropic.RateLimitError("rate limited", response=response, body=None)

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        calls = 0

        def stream(self, **kwargs: Any) -> FakeStream:
            self.calls += 1
            if self.calls == 1:
                raise rate_limit
            return FakeStream()

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    monkeypatch.setattr(LLMClient, "_sleep_for_attempt", lambda self, attempt: 2.0)
    monkeypatch.setattr("diagex.llm.client.time.sleep", sleeps.append)

    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        ),
        budgets=RuntimeBudgets(retry_attempts=2),
    )

    result = client.messages_create(system="system", messages=[], max_tokens=32)

    assert result is expected
    assert sleeps == [30.0]
    assert client.retries_total == 1


def test_retry_after_does_not_shorten_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        503,
        headers={"Retry-After": "1"},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
    )
    error = anthropic.APIStatusError("unavailable", response=response, body=None)
    sleeps: list[float] = []

    class FakeMessages:
        def stream(self, **kwargs: Any) -> Any:
            raise error

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    monkeypatch.setattr(LLMClient, "_sleep_for_attempt", lambda self, attempt: 5.0)
    monkeypatch.setattr("diagex.llm.client.time.sleep", sleeps.append)

    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        ),
        budgets=RuntimeBudgets(retry_attempts=2),
    )

    with pytest.raises(anthropic.APIStatusError):
        client.messages_create(system="system", messages=[], max_tokens=32)

    assert sleeps == [5.0]


def test_connection_retry_log_includes_underlying_transport_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/messages")
    connection_error = anthropic.APIConnectionError(request=request)
    connection_error.__cause__ = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body"
    )

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        calls = 0

        def stream(self, **kwargs: Any) -> FakeStream:
            self.calls += 1
            if self.calls == 1:
                raise connection_error
            return FakeStream()

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    monkeypatch.setattr(LLMClient, "_sleep_for_attempt", lambda self, attempt: 1.0)
    monkeypatch.setattr("diagex.llm.client.time.sleep", lambda _seconds: None)

    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        ),
        budgets=RuntimeBudgets(retry_attempts=2),
    )

    assert client.messages_create(system="system", messages=[], max_tokens=32) is expected
    retry_log = capsys.readouterr().err
    assert "APIConnectionError: Connection error." in retry_log
    assert "RemoteProtocolError: peer closed connection" in retry_log
    assert "retry 1/2 in 1.0s" in retry_log


def test_raw_stream_protocol_error_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")

    class FakeStream:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            if self.fail:
                raise httpx.RemoteProtocolError(
                    "peer closed connection without sending complete message body"
                )
            return expected

    class FakeMessages:
        calls = 0

        def stream(self, **kwargs: Any) -> FakeStream:
            self.calls += 1
            return FakeStream(fail=self.calls == 1)

    messages = FakeMessages()
    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=messages)),
    )
    monkeypatch.setattr(LLMClient, "_sleep_for_attempt", lambda self, attempt: 1.0)
    monkeypatch.setattr("diagex.llm.client.time.sleep", lambda _seconds: None)
    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        ),
        budgets=RuntimeBudgets(retry_attempts=2),
    )

    assert client.messages_create(system="system", messages=[], max_tokens=32) is expected
    assert messages.calls == 2
    assert client.retries_total == 1
    retry_log = capsys.readouterr().err
    assert "RemoteProtocolError: peer closed connection" in retry_log
    assert "retry 1/2 in 1.0s" in retry_log


def test_rate_limit_without_retry_after_uses_conservative_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    sleeps: list[float] = []
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
    )
    rate_limit = anthropic.RateLimitError("rate limited", response=response, body=None)

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        calls = 0

        def stream(self, **kwargs: Any) -> FakeStream:
            self.calls += 1
            if self.calls == 1:
                raise rate_limit
            return FakeStream()

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    monkeypatch.setattr(LLMClient, "_sleep_for_attempt", lambda self, attempt: 1.0)
    monkeypatch.setattr("diagex.llm.client.time.sleep", sleeps.append)

    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        ),
        budgets=RuntimeBudgets(retry_attempts=2),
    )

    assert client.messages_create(system="system", messages=[], max_tokens=32) is expected
    assert sleeps == [30.0]


def test_rate_limit_cooldown_persists_until_three_successes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
    )
    rate_limit = anthropic.RateLimitError("rate limited", response=response, body=None)

    class Clock:
        now = 0.0
        sleeps: list[float] = []

        @classmethod
        def monotonic(cls) -> float:
            return cls.now

        @classmethod
        def sleep(cls, seconds: float) -> None:
            cls.sleeps.append(seconds)
            cls.now += seconds

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        calls = 0

        def stream(self, **kwargs: Any) -> FakeStream:
            self.calls += 1
            if self.calls == 1:
                raise rate_limit
            return FakeStream()

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    monkeypatch.setattr(LLMClient, "_sleep_for_attempt", lambda self, attempt: 1.0)
    monkeypatch.setattr("diagex.llm.client.time.monotonic", Clock.monotonic)
    monkeypatch.setattr("diagex.llm.client.time.sleep", Clock.sleep)

    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        ),
        budgets=RuntimeBudgets(retry_attempts=2),
    )

    # The first call sleeps after its 429 and then succeeds. The next two calls
    # are paced; after the third consecutive success, the fourth starts at once.
    for _ in range(4):
        assert client.messages_create(system="system", messages=[], max_tokens=32) is expected

    assert Clock.sleeps == [30.0, 30.0, 30.0]


def test_repeated_rate_limits_escalate_shared_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = SimpleNamespace(content=[], usage=None, stop_reason="end_turn")
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
    )
    rate_limit = anthropic.RateLimitError("rate limited", response=response, body=None)
    sleeps: list[float] = []

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_final_message(self) -> SimpleNamespace:
            return expected

    class FakeMessages:
        calls = 0

        def stream(self, **kwargs: Any) -> FakeStream:
            self.calls += 1
            if self.calls <= 2:
                raise rate_limit
            return FakeStream()

    monkeypatch.setattr(
        LLMClient,
        "_build_client",
        staticmethod(lambda _config: SimpleNamespace(messages=FakeMessages())),
    )
    monkeypatch.setattr(LLMClient, "_sleep_for_attempt", lambda self, attempt: 1.0)
    monkeypatch.setattr("diagex.llm.client.time.sleep", sleeps.append)

    client = LLMClient(
        LLMConfig(
            transport="openrouter",
            model="vendor/vision-model",
            openrouter_api_key="test-key",
        ),
        budgets=RuntimeBudgets(retry_attempts=3),
    )

    assert client.messages_create(system="system", messages=[], max_tokens=32) is expected
    assert sleeps == [30.0, 60.0]
