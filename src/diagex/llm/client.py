"""LLM transport layer for Anthropic-compatible Messages APIs.

Spec refs: §4 (stack), §6.2 (prompt caching), §6.3 (budgets), §6.4 (resilience: retry),
§10 (cost).

Four transports are supported:
  - "anthropic": direct api.anthropic.com via anthropic.Anthropic().
  - "azure":    Azure AI Foundry's Anthropic-compatible endpoint, driven via
                anthropic.Anthropic(base_url=..., default_headers={"api-key": ...}).
  - "openrouter": OpenRouter's Anthropic Messages endpoint, authenticated with
                  an OpenRouter bearer token and accepting any compatible model slug.
  - "kimi":      Kimi Code's Anthropic-compatible endpoint, accepting K3 model IDs.

The SDK's own retry machinery is deliberately bypassed — this module enforces the
spec's backoff policy (base 2s, factor 2, max 60s, 6 attempts) from RuntimeBudgets so
the behaviour is configurable and re-projectable from config.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import anthropic
import httpx

from diagex.config import LLMConfig, RuntimeBudgets
from diagex.llm.billing import AccountedSpendingLedger
from diagex.llm.budget import BudgetExceeded, SpendingLedger
from diagex.llm.model_policy import validate_production_models


def is_non_retryable_api_error(exc: BaseException) -> bool:
    """True for provider HTTP failures that another item cannot repair.

    The transport retries 429 and server/connection failures internally. Other
    4xx responses represent invalid credentials, policy/routing restrictions,
    unsupported parameters, or missing models. Continuing a batch after one of
    those failures only repeats the same deterministic error.
    """
    if isinstance(exc, BudgetExceeded):
        return True
    if not isinstance(exc, anthropic.APIStatusError):
        return False
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429


_MALFORMED_TOOL_JSON_RE = re.compile(
    r"(?:key must be a string at line \d+ column \d+|"
    r"expected\s+[`'\"]?.+?[`'\"]?\s+at line \d+ column \d+|"
    r"expecting\s+.+?delimiter:\s*line \d+ column \d+|"
    r"unterminated string[^\n]*line \d+ column \d+|"
    r"invalid json[^\n]*(?:tool|input)|"
    r"(?:tool|input)[^\n]*invalid json)",
    re.IGNORECASE,
)

_REASONING_REQUIRED_MODEL_RE = re.compile(
    r"(?:^|/)glm-5\.3(?:$|[-:])",
    re.IGNORECASE,
)


def is_malformed_tool_json_error(exc: ValueError) -> bool:
    """Recognise provider/SDK failures while decoding tool arguments."""

    return isinstance(exc, json.JSONDecodeError) or bool(_MALFORMED_TOOL_JSON_RE.search(str(exc)))


def model_requires_reasoning(model: str | None) -> bool:
    """Return whether a known endpoint rejects requests with reasoning disabled.

    Keep this deliberately narrow.  A runtime error check complements the small
    registry so newly introduced provider aliases can recover without silently
    treating every reasoning-capable model as reasoning-mandatory.
    """

    normalised = str(model or "").strip().lstrip("~")
    return bool(_REASONING_REQUIRED_MODEL_RE.search(normalised))


def is_reasoning_required_error(exc: BaseException) -> bool:
    """Recognise provider validation errors that require reasoning to stay on."""

    detail = str(exc).casefold()
    mandatory = "reasoning is mandatory" in detail or "thinking is mandatory" in detail
    cannot_disable = "cannot be disabled" in detail or "must be enabled" in detail
    return mandatory and cannot_disable


class LLMClient:
    """Thin wrapper around the Anthropic SDK with caching-aware messages_create()."""

    def __init__(self, config: LLMConfig, budgets: RuntimeBudgets | None = None) -> None:
        self.config = config
        self.budgets = budgets or RuntimeBudgets()
        validate_production_models(config)
        if config.spending_ledger and not config.verified_prices:
            raise ValueError("A spending ledger requires verified model prices")
        self.spending = None
        if config.spending_ledger:
            state = json.loads(Path(config.spending_ledger).read_text())
            ledger_type = AccountedSpendingLedger if state.get("schema_version") == 2 else SpendingLedger
            self.spending = ledger_type(config.spending_ledger, config.verified_prices, config.spending_category)
        self._client = self._build_client(config)
        # Cumulative retry count: every backoff attempt increments this. The
        # extractor snapshots it before/after a run so the per-extractor row
        # in results.csv carries the retries that actually ate wall-clock.
        self.retries_total: int = 0
        # Provider-wide 429 state must outlive an individual messages_create()
        # call. The extraction tail issues many small, independent requests;
        # resetting backoff for each one otherwise creates a retry storm.
        self._rate_limit_cooldown_s: float = 0.0
        self._rate_limit_not_before: float = 0.0
        self._rate_limit_successes: int = 0

    def reset_retry_counter(self) -> None:
        self.retries_total = 0

    @staticmethod
    def _build_client(config: LLMConfig) -> anthropic.Anthropic:
        """Construct the underlying Anthropic SDK client.

        Azure AI Foundry exposes an Anthropic-compatible endpoint; the Python SDK
        talks to it when pointed at the right base_url with the `api-key` header
        (Azure) set alongside the SDK's usual `x-api-key` (which Azure ignores but
        the SDK insists on populating).
        """
        if config.transport == "azure":
            if not (config.azure_endpoint and config.azure_api_key):
                raise ValueError(
                    "Azure transport requires azure_endpoint and azure_api_key; "
                    "check .env for AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY."
                )
            # Azure appends ?api-version=... to every call; the SDK does not know about
            # that query param, so stamp it into default_query to avoid surgery per-call.
            default_query: dict[str, str] = {}
            if config.azure_api_version:
                default_query["api-version"] = config.azure_api_version
            return anthropic.Anthropic(
                base_url=config.azure_endpoint,
                api_key=config.azure_api_key,  # satisfies SDK; ignored by Azure
                default_headers={"api-key": config.azure_api_key},
                default_query=default_query or None,
                max_retries=0,  # we own retry policy
            )

        if config.transport == "openrouter":
            if not config.openrouter_api_key:
                raise ValueError(
                    "OpenRouter transport requires openrouter_api_key; set OPENROUTER_API_KEY."
                )
            headers = {"X-OpenRouter-Title": config.openrouter_app_title}
            if config.openrouter_http_referer:
                headers["HTTP-Referer"] = config.openrouter_http_referer
            return anthropic.Anthropic(
                base_url=config.openrouter_base_url,
                auth_token=config.openrouter_api_key,
                default_headers=headers,
                max_retries=0,  # we own retry policy
            )

        if config.transport == "kimi":
            if not config.kimi_api_key:
                raise ValueError("Kimi transport requires kimi_api_key; set KIMI_API_KEY.")
            return anthropic.Anthropic(
                base_url=LLMClient._kimi_anthropic_base_url(config.kimi_base_url),
                api_key=config.kimi_api_key,
                max_retries=0,  # we own retry policy
            )

        # Direct Anthropic transport.
        if not config.anthropic_api_key:
            raise ValueError(
                "Anthropic transport requires anthropic_api_key; set ANTHROPIC_API_KEY."
            )
        return anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            max_retries=0,  # we own retry policy
        )

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _kimi_anthropic_base_url(configured_url: str) -> str:
        """Convert Kimi's commonly published OpenAI base to its Anthropic base.

        Kimi documents ``/coding/v1`` for OpenAI-compatible clients and
        ``/coding`` for Anthropic-compatible clients. DiagEx uses the latter;
        the Anthropic SDK appends ``/v1/messages`` itself.
        """
        base_url = configured_url.rstrip("/")
        messages_suffix = "/coding/v1/messages"
        openai_suffix = "/coding/v1"
        if base_url.endswith(messages_suffix):
            return base_url[: -len("/v1/messages")]
        if base_url.endswith(openai_suffix):
            return base_url[: -len("/v1")]
        return base_url

    def _sleep_for_attempt(self, attempt: int) -> float:
        """Exponential backoff with jitter, clamped to retry_max_s."""
        base = self.budgets.retry_base_s
        factor = self.budgets.retry_factor
        cap = self.budgets.retry_max_s
        delay = min(cap, base * (factor**attempt))
        # Full jitter: sample in [0, delay] so coincident clients decorrelate.
        return random.uniform(0, delay)

    @staticmethod
    def _exception_detail(exc: BaseException, *, max_chars: int = 700) -> str:
        """Render a bounded, single-line exception chain for retry diagnostics.

        Anthropic's ``APIConnectionError`` often has the generic message
        ``Connection error.`` while the useful transport diagnosis (timeout,
        reset, incomplete chunked body, and so on) lives in ``__cause__``.
        Keep both without dumping request headers or an unbounded response body.
        """
        parts: list[str] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen and len(parts) < 5:
            seen.add(id(current))
            message = " ".join(str(current).split())
            name = type(current).__name__
            parts.append(f"{name}: {message}" if message else name)
            current = current.__cause__ or current.__context__

        rendered = " <- caused by ".join(parts)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max(0, max_chars - 1)].rstrip() + "…"

    @staticmethod
    def _retry_after_seconds(exc: anthropic.APIStatusError) -> float | None:
        """Return the server-requested retry delay, if it supplied a valid one.

        HTTP permits either a number of seconds or an absolute HTTP date.  The
        latter is uncommon on model APIs, but supporting it keeps this helper
        standards-compliant.  A Retry-After value is deliberately not capped by
        ``retry_max_s``: shortening it would immediately violate the provider's
        instruction and commonly cause another 429.
        """
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        raw = headers.get("retry-after")
        if raw is None:
            return None

        value = str(raw).strip()
        try:
            seconds = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = (retry_at - datetime.now(UTC)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return None

        if not isfinite(seconds) or seconds < 0:
            return None
        return seconds

    def _record_rate_limit(self, retry_after: float | None) -> float:
        """Advance and persist the shared cooldown after an HTTP 429."""
        requested = (
            retry_after if retry_after is not None else max(0.0, self.budgets.rate_limit_fallback_s)
        )
        previous = self._rate_limit_cooldown_s
        escalated = min(
            self.budgets.retry_max_s,
            previous * self.budgets.retry_factor,
        )
        # Never shorten an existing server-requested delay merely because the
        # normal local retry cap is lower.
        cooldown = max(requested, previous, escalated)
        self._rate_limit_cooldown_s = cooldown
        self._rate_limit_successes = 0
        self._rate_limit_not_before = max(
            self._rate_limit_not_before,
            time.monotonic() + cooldown,
        )
        return cooldown

    def _wait_for_rate_limit_cooldown(self, *, deadline: float | None = None) -> None:
        """Pace a new request according to 429 state from earlier requests."""
        remaining = self._rate_limit_not_before - time.monotonic()
        if remaining <= 0:
            return
        if deadline is not None and time.monotonic() + remaining >= deadline:
            raise TimeoutError("request time budget cannot accommodate provider cooldown")
        print(
            f"[diagex.llm] rate-limit cooldown; next request in {remaining:.1f}s",
            file=sys.stderr,
        )
        time.sleep(remaining)

    def _record_success_after_rate_limit(self) -> None:
        """Clear shared 429 state only after several consecutive successes."""
        if self._rate_limit_cooldown_s <= 0:
            return
        self._rate_limit_successes += 1
        required = max(1, self.budgets.rate_limit_successes_to_reset)
        if self._rate_limit_successes >= required:
            self._rate_limit_cooldown_s = 0.0
            self._rate_limit_not_before = 0.0
            self._rate_limit_successes = 0
            return

        # Pace the next independent request. This is intentionally scheduled
        # after the successful response, not after its start, because short
        # cleanup calls are what previously formed the end-of-run burst.
        self._rate_limit_not_before = time.monotonic() + self._rate_limit_cooldown_s

    @staticmethod
    def _stamp_cache(blocks: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Attach cache_control={type:ephemeral} to the final block of a list.

        Used for the system prompt and the tools list so a stable prefix is cached.
        We mutate the *last* element so callers can compose prefix chunks freely.
        """
        if not blocks:
            return blocks
        # Shallow copy so we don't mutate caller data structures.
        out = [dict(b) for b in blocks]
        out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
        return out

    # ---- public API -------------------------------------------------------

    def messages_create(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int,
        thinking: dict[str, Any] | None = None,
        reasoning_mode_override: Literal["enabled", "disabled"] | None = None,
        output_config: dict[str, Any] | None = None,
        extra_cache_breakpoints: list[dict[str, Any]] | None = None,
        on_stream_delta: Callable[[str, str], None] | None = None,
        time_budget_s: float | None = None,
        max_attempts: int | None = None,
        on_transport_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any:
        """Send a Messages request with caching + spec-compliant retry.

        `system` may be a plain string (treated as one text block, caching applied)
        or a pre-built list of system blocks where the caller has already placed
        cache_control markers. In the latter case we leave placement alone.

        `thinking` on Opus 4.7 must be `{"type": "adaptive"}` or `{"type": "disabled"}`
        — the legacy `{"type": "enabled", "budget_tokens": N}` shape is rejected.
        `output_config={"effort": "low|medium|high|xhigh|max"}` controls thinking depth.
        ``reasoning_mode_override`` supports explicit per-stage policy and bounded
        recovery calls. Without an override, calls follow ``DIAGEX_REASONING``.

        Opus 4.7 also rejects `temperature` / `top_p` / `top_k`, so we never send them.
        """
        # Normalize system → list form, stamp cache_control on the last block if the
        # caller passed a bare string or a list without its own markers.
        if isinstance(system, str):
            system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
            system_blocks = self._stamp_cache(system_blocks)  # type: ignore[assignment]
        else:
            # Respect caller-supplied cache_control if any block already has one.
            has_marker = any("cache_control" in b for b in system)
            system_blocks = system if has_marker else (self._stamp_cache(list(system)) or [])

        # Tools render at position 0; a breakpoint on the last tool caches the whole
        # tool list. Only stamp if the caller did not pre-place markers.
        tools_payload: list[dict[str, Any]] | None = None
        if tools:
            has_marker = any("cache_control" in t for t in tools)
            tools_payload = tools if has_marker else (self._stamp_cache(list(tools)) or [])

        # `extra_cache_breakpoints` are opaque structured blocks — e.g. a cached
        # overview image appended to the first user turn — that the caller has
        # already wired into `messages`. Kept in the signature as a hook so the
        # runtime can pass breakpoint metadata for logging/debugging without
        # modifying messages here.
        _ = extra_cache_breakpoints

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if self.spending and self.config.transport == "openrouter":
            price = self.spending.prices.get(self.config.model)
            if price:
                kwargs["extra_body"] = {"provider": {
                    "only": price["provider_tags"], "require_parameters": True,
                    "max_price": {"prompt": price["input_per_token"] * 1_000_000,
                                  "completion": price["output_per_token"] * 1_000_000},
                }}
        if tools_payload is not None:
            kwargs["tools"] = tools_payload
        if tool_choice is not None:
            if tools_payload is None:
                raise ValueError("tool_choice requires at least one tool")
            kwargs["tool_choice"] = tool_choice
        effective_thinking = thinking
        reasoning_mode = reasoning_mode_override or self.config.reasoning_mode
        if reasoning_mode == "enabled":
            effective_thinking = {"type": "adaptive", "display": "summarized"}
        elif reasoning_mode == "disabled":
            effective_thinking = {"type": "disabled"}

        if effective_thinking is not None:
            # ``display`` is an Anthropic-specific presentation option. Kimi K3
            # accepts adaptive/disabled thinking plus output_config.effort, but
            # rejects unknown thinking members with HTTP 400.
            kwargs["thinking"] = (
                {key: value for key, value in effective_thinking.items() if key != "display"}
                if self.config.transport == "kimi"
                else effective_thinking
            )
        # Qwen hosted endpoints expose binary thinking, not Anthropic effort.
        # Passing effort while requiring supported parameters makes OpenRouter
        # reject every route, even though image/tool smoke checks succeed.
        qwen_binary_thinking = (
            self.config.transport == "openrouter"
            and self.config.model.startswith("qwen/qwen3.5-")
        )
        # OpenRouter translates effort into a reasoning parameter even when
        # thinking is disabled. With require_parameters this contradictory
        # combination can eliminate otherwise valid tool/image providers.
        disabled_openrouter_thinking = (
            self.config.transport == "openrouter"
            and effective_thinking is not None
            and effective_thinking.get("type") == "disabled"
        )
        if output_config is not None and not qwen_binary_thinking and not disabled_openrouter_thinking:
            kwargs["output_config"] = output_config

        # Stream for long outputs. High `max_tokens` + adaptive thinking can exceed the
        # SDK's 10-minute non-streaming timeout; `.stream(...).get_final_message()` returns
        # the same Message object but keeps the connection alive via chunked transfer.
        if time_budget_s is not None and time_budget_s <= 0:
            raise TimeoutError("request time budget exhausted before dispatch")
        deadline = time.monotonic() + time_budget_s if time_budget_s is not None else None
        attempts = min(
            self.budgets.retry_attempts,
            max_attempts if max_attempts is not None else self.budgets.retry_attempts,
        )
        if attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._wait_for_rate_limit_cooldown(deadline=deadline)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            attempt_started = time.monotonic()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request time budget exhausted") from last_exc
                # The deadline is checked while consuming events. A stalled read
                # is bounded separately, including when no visible delta is emitted.
                kwargs["timeout"] = httpx.Timeout(min(remaining, 15.0))
            retry_after: float | None = None
            rate_limit_cooldown: float | None = None
            reservation = self.spending.reserve(self.config.model, max_tokens) if self.spending else None
            try:
                if on_transport_event is not None:
                    on_transport_event({"event": "dispatch", "attempt": attempt + 1, "started_at": time.time(), "model": self.config.model, "thinking": kwargs.get("thinking"), "max_tokens": max_tokens})
                with self._client.messages.stream(**kwargs) as stream:
                    if on_stream_delta is not None or deadline is not None or isinstance(self.spending, AccountedSpendingLedger):
                        for event in stream:
                            if reservation and isinstance(self.spending, AccountedSpendingLedger):
                                self.spending.observe_stream_event(reservation, event)
                            if deadline is not None and time.monotonic() >= deadline:
                                raise TimeoutError("stream exceeded request time budget")
                            if on_stream_delta is not None:
                                self._forward_stream_delta(event, on_stream_delta)
                    message = stream.get_final_message()
                    if reservation:
                        self.spending.settle(reservation, message)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("response exceeded request time budget")
                    self._record_success_after_rate_limit()
                    if on_transport_event is not None:
                        on_transport_event({"event": "response", "attempt": attempt + 1, "elapsed_s": time.monotonic() - attempt_started})
                    return message
            except anthropic.APIStatusError as exc:
                if reservation and isinstance(self.spending, AccountedSpendingLedger):
                    self.spending.observe_error(reservation, exc)
                status = getattr(exc, "status_code", None)
                # 4xx (except 429) = validation/auth/policy → surface immediately.
                if is_non_retryable_api_error(exc):
                    raise
                last_exc = exc
                if status in {429, 503}:
                    retry_after = self._retry_after_seconds(exc)
                if status == 429:
                    rate_limit_cooldown = self._record_rate_limit(retry_after)
            except anthropic.APIConnectionError as exc:
                # Transient network fault; same backoff ladder as 5xx.
                last_exc = exc
            except httpx.TransportError as exc:
                # A streamed response can fail while events are being consumed,
                # outside the SDK's APIConnectionError wrapper. Incomplete
                # chunked reads and raw protocol resets are still transient.
                last_exc = exc
            except TimeoutError as exc:
                if on_transport_event is not None:
                    on_transport_event({"event": "deadline", "attempt": attempt + 1, "elapsed_s": time.monotonic() - attempt_started, "error": str(exc)})
                raise
            finally:
                if reservation and isinstance(self.spending, AccountedSpendingLedger):
                    self.spending.finish_attempt(reservation)

            if on_transport_event is not None:
                on_transport_event({"event": "error", "attempt": attempt + 1, "elapsed_s": time.monotonic() - attempt_started, "error": self._exception_detail(last_exc)})

            if attempt == attempts - 1:
                break

            delay = self._sleep_for_attempt(attempt)
            delay = max(
                delay,
                retry_after or 0.0,
                rate_limit_cooldown or 0.0,
            )
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise TimeoutError("retry delay exceeds remaining request time budget") from last_exc
            self.retries_total += 1
            if on_transport_event is not None:
                on_transport_event({"event": "retry_wait", "attempt": attempt + 1, "seconds": delay})
            notes: list[str] = []
            if retry_after is not None:
                notes.append(f"Retry-After={retry_after:.1f}s")
            if rate_limit_cooldown is not None and (
                retry_after is None or rate_limit_cooldown > retry_after
            ):
                notes.append(f"rate-limit cooldown={rate_limit_cooldown:.1f}s")
            retry_note = f"; {'; '.join(notes)}" if notes else ""
            error_detail = self._exception_detail(last_exc)
            print(
                f"[diagex.llm] transient failure ({error_detail}); "
                f"retry {attempt + 1}/{attempts} in {delay:.1f}s"
                f"{retry_note}",
                file=sys.stderr,
            )
            time.sleep(delay)

        # Exhausted — re-raise the last observed error.
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _forward_stream_delta(event: Any, callback: Callable[[str, str], None]) -> None:
        """Forward visible text/reasoning deltas without coupling UI to the SDK."""
        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
        if event_type != "content_block_delta":
            return

        delta = event.get("delta") if isinstance(event, dict) else getattr(event, "delta", None)
        if delta is None:
            return
        delta_type = delta.get("type") if isinstance(delta, dict) else getattr(delta, "type", None)
        if delta_type in {"thinking_delta", "reasoning_delta"}:
            kind = "thinking"
            field_names = ("thinking", "reasoning", "text")
        elif delta_type == "text_delta":
            kind = "text"
            field_names = ("text",)
        else:
            return

        text = ""
        for field_name in field_names:
            value = (
                delta.get(field_name)
                if isinstance(delta, dict)
                else getattr(delta, field_name, None)
            )
            if isinstance(value, str) and value:
                text = value
                break
        if not text:
            return

        # Display failures must never abort and retry an otherwise healthy, costly
        # model request.
        try:
            callback(kind, text)
        except Exception:
            pass
