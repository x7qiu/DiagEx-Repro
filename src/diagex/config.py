"""Runtime configuration: model id, budgets, pricing, env vars.

Spec references: §4 (stack), §6.3 (budgets), §6.4 (resilience), §6.5 (data handling),
§10 (cost model). Every knob with a `[DECISION]` default in the spec lives here so a
config edit re-projects historical runs without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env once at import time; downstream os.environ lookups now see it.
load_dotenv()


EffortLevel = Literal["low", "medium", "high", "xhigh"]
ReasoningMode = Literal["auto", "enabled", "disabled"]
PidEngine = Literal["legacy", "evidence-v2"]


def _pid_engine_from_env() -> PidEngine:
    raw = os.environ.get("DIAGEX_PID_ENGINE", "legacy").strip().lower()
    aliases = {
        "": "legacy",
        "legacy": "legacy",
        "v1": "legacy",
        "evidence-v2": "evidence-v2",
        "evidence_v2": "evidence-v2",
        "v2": "evidence-v2",
    }
    try:
        return aliases[raw]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            "DIAGEX_PID_ENGINE must be one of: legacy, evidence-v2"
        ) from exc


@dataclass(frozen=True)
class EffortProfile:
    """Per-effort reasoning + step budget (spec §7.1).

    Opus 4.7 uses adaptive thinking plus `output_config.effort` — there is no
    explicit token budget for thinking. `api_effort` is the string passed to the
    API; `adaptive_thinking` toggles whether adaptive thinking is on at all.
    """

    max_steps: int
    api_effort: str                    # "low" | "medium" | "high" | "xhigh" | "max"
    adaptive_thinking: bool            # False for `low`, True otherwise
    max_output_tokens: int


EFFORT_PROFILES: dict[EffortLevel, EffortProfile] = {
    "low":    EffortProfile(max_steps=6,  api_effort="low",    adaptive_thinking=False, max_output_tokens=2048),
    "medium": EffortProfile(max_steps=20, api_effort="medium", adaptive_thinking=True,  max_output_tokens=16000),
    "high":   EffortProfile(max_steps=40, api_effort="high",   adaptive_thinking=True,  max_output_tokens=32000),
    "xhigh":  EffortProfile(max_steps=60, api_effort="xhigh",  adaptive_thinking=True,  max_output_tokens=64000),
}


@dataclass
class PricingConfig:
    """Per-million-token pricing, read from config so invoices can be re-projected (spec §10)."""

    input_per_mtok: float = 5.0
    output_per_mtok: float = 25.0
    cache_read_per_mtok: float = 0.50
    cache_write_per_mtok: float = 6.25

    def cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        return (
            input_tokens      * self.input_per_mtok      / 1_000_000
            + output_tokens     * self.output_per_mtok     / 1_000_000
            + cache_read_tokens * self.cache_read_per_mtok / 1_000_000
            + cache_write_tokens * self.cache_write_per_mtok / 1_000_000
        )


@dataclass
class TilingConfig:
    max_tokens_per_tile: int = 2200
    overlap_frac: float = 0.20
    token_per_pixel: float = 1.0 / 750.0    # Opus 4.7 ~ w*h/750
    max_page_dim_px: int = 6000             # rendering budget (spec §5.1)
    target_dpi: int = 300                   # clamped by max_page_dim_px


@dataclass
class ScanConfig:
    """Minimal, non-perceptual preprocessing for scanned PDFs (spec §12.1.6)."""

    deskew: bool = False
    contrast: bool = True
    despeckle: bool = True


@dataclass
class RuntimeBudgets:
    max_image_requests_per_tile: int = 3
    retry_base_s: float = 2.0
    retry_factor: float = 2.0
    retry_max_s: float = 60.0
    retry_attempts: int = 6
    # Some OpenRouter providers omit Retry-After on 429 responses. Keep a
    # conservative client-wide cooldown in that case so the next independent
    # cleanup/arbitration request does not immediately hit the same limit.
    rate_limit_fallback_s: float = 30.0
    rate_limit_successes_to_reset: int = 3


@dataclass
class SymbolPerceptionConfig:
    """Bounded raw-symbol requests; independent of downstream graph reasoning."""

    first_pass_tokens: int = 6000
    reasoning_tokens: int = 16000
    request_timeout_s: float = 45.0
    reasoning_timeout_s: float = 120.0
    transport_attempts: int = 2
    run_timeout_s: float = 1800.0
    consecutive_failure_limit: int = 3
    total_failure_limit: int = 5
    workflow: Literal["fixed", "adaptive"] = "fixed"


@dataclass
class LLMConfig:
    """LLM transport settings for Anthropic-compatible Messages endpoints.

    Note: Opus 4.7 rejects `temperature` / `top_p` / `top_k` with 400. Thinking
    depth is controlled via `output_config.effort`, not sampling parameters.
    """

    model: str = "claude-opus-4-7"
    # Evidence-v2 can use a fast perception model and reserve a stronger model
    # for the small ambiguity queue.  ``None`` retains the historical single-
    # model behaviour.
    vision_model: str | None = None
    reasoning_model: str | None = None
    escalation_model: str | None = field(default_factory=lambda: os.environ.get("DIAGEX_ESCALATION_MODEL") or None)
    production_open_weight: bool = field(default_factory=lambda: os.environ.get("DIAGEX_MODEL_POLICY", "evaluation") == "production-open-weight")
    spending_ledger: str | None = field(default_factory=lambda: os.environ.get("DIAGEX_SPENDING_LEDGER") or None)
    verified_prices: str | None = field(default_factory=lambda: os.environ.get("DIAGEX_VERIFIED_PRICES") or None)
    spending_category: str = field(default_factory=lambda: os.environ.get("DIAGEX_SPENDING_CATEGORY", "graph"))
    # Global override for model thinking. ``auto`` preserves each caller's
    # existing effort-based choice; enabled/disabled applies to every call,
    # including legend extraction and arbitration.
    reasoning_mode: ReasoningMode = "auto"
    # OpenRouter and Kimi expose Anthropic-compatible Messages endpoints, so all
    # transports retain the runtime's native image/tool/thinking block shape.
    transport: Literal["anthropic", "azure", "openrouter", "kimi"] = "anthropic"
    anthropic_api_key: str | None = None
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_deployment: str | None = None
    azure_api_version: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api"
    openrouter_http_referer: str | None = None
    openrouter_app_title: str = "DiagEx"
    kimi_api_key: str | None = None
    kimi_base_url: str = "https://api.kimi.com/coding/v1"

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Select a provider explicitly, retaining the historical auto-detection."""
        provider = os.environ.get("DIAGEX_LLM_PROVIDER", "").strip().lower()
        vision_model = os.environ.get("DIAGEX_VISION_MODEL", "").strip() or None
        reasoning_model = os.environ.get("DIAGEX_REASONING_MODEL", "").strip() or None
        reasoning_raw = os.environ.get("DIAGEX_REASONING", "auto").strip().lower()
        reasoning_aliases: dict[str, ReasoningMode] = {
            "": "auto",
            "auto": "auto",
            "on": "enabled",
            "true": "enabled",
            "1": "enabled",
            "enabled": "enabled",
            "reasoning": "enabled",
            "off": "disabled",
            "false": "disabled",
            "0": "disabled",
            "disabled": "disabled",
            "non-reasoning": "disabled",
            "non_reasoning": "disabled",
        }
        if reasoning_raw not in reasoning_aliases:
            raise ValueError(
                "DIAGEX_REASONING must be one of: auto, enabled, disabled "
                "(on/off and true/false are also accepted)"
            )
        reasoning_mode = reasoning_aliases[reasoning_raw]
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        kimi_key = os.environ.get("KIMI_API_KEY")

        if provider and provider not in {"anthropic", "azure", "openrouter", "kimi"}:
            raise ValueError(
                "DIAGEX_LLM_PROVIDER must be one of: anthropic, azure, openrouter, kimi"
            )

        use_kimi = provider == "kimi" or (
            not provider
            and bool(kimi_key)
            and not openrouter_key
            and not anthropic_key
            and not (azure_endpoint and azure_key and azure_deployment)
        )
        if use_kimi:
            model = os.environ.get("DIAGEX_MODEL") or os.environ.get("KIMI_MODEL")
            if not model:
                raise ValueError(
                    "Kimi requires a model ID; set DIAGEX_MODEL (for example, k3)."
                )
            return cls(
                transport="kimi",
                model=model,
                vision_model=vision_model,
                reasoning_model=reasoning_model,
                reasoning_mode=reasoning_mode,
                kimi_api_key=kimi_key,
                kimi_base_url=os.environ.get(
                    "KIMI_BASE_URL", "https://api.kimi.com/coding/v1"
                ).rstrip("/"),
            )

        use_openrouter = provider == "openrouter" or (
            not provider
            and bool(openrouter_key)
            and not anthropic_key
            and not (azure_endpoint and azure_key and azure_deployment)
        )
        if use_openrouter:
            model = os.environ.get("DIAGEX_MODEL") or os.environ.get("OPENROUTER_MODEL")
            if not model:
                raise ValueError(
                    "OpenRouter requires a model slug; set DIAGEX_MODEL "
                    "(for example, a model supporting both image input and tools)."
                )
            return cls(
                transport="openrouter",
                model=model,
                vision_model=vision_model,
                reasoning_model=reasoning_model,
                reasoning_mode=reasoning_mode,
                openrouter_api_key=openrouter_key,
                openrouter_base_url=os.environ.get(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api"
                ).rstrip("/"),
                openrouter_http_referer=os.environ.get("OPENROUTER_HTTP_REFERER"),
                openrouter_app_title=os.environ.get("OPENROUTER_APP_TITLE", "DiagEx"),
            )

        if provider == "azure" or (
            not provider and azure_endpoint and azure_key and azure_deployment
        ):
            return cls(
                transport="azure",
                model=azure_deployment,
                vision_model=vision_model,
                reasoning_model=reasoning_model,
                reasoning_mode=reasoning_mode,
                azure_endpoint=azure_endpoint,
                azure_api_key=azure_key,
                azure_deployment=azure_deployment,
                azure_api_version=azure_api_version or "2025-04-01-preview",
            )
        return cls(
            transport="anthropic",
            vision_model=vision_model,
            reasoning_model=reasoning_model,
            reasoning_mode=reasoning_mode,
            anthropic_api_key=anthropic_key,
        )


@dataclass
class PidConfig:
    """Phase 2 knobs: legend budget, few-shot cap, confidence report, DEXPI subset (spec §7.2).

    Budgets here shape the cached system prompt and the agent's per-tile effort.
    `legend_*` values encode spec §7.2.2's token-budget fallback.
    """

    # The evidence-first engine remains opt-in until it clears the eval gates.
    engine: PidEngine = field(default_factory=_pid_engine_from_env)

    # Few-shot legend block (inside the cached system prompt).
    legend_few_shot_tokens: int = 6000           # ≈ §12.1.2 30 symbols / 6k tokens
    legend_few_shot_max_entries: int = 30

    # Drawn-image-sidecar size (so the cache block is not dominated by one ornate glyph).
    legend_thumb_max_dim: int = 128

    # Shared-cache root for --legend-key runs; see §7.2.2.
    legends_dir: Path = field(default_factory=lambda: Path("legends"))

    # Confidence report: write an HTML digest alongside the DEXPI JSON.
    write_confidence_report: bool = True

    # Page-agent safety ceiling. Reasoning effort remains independent: dense
    # pages receive one step per planned tile plus a buffer, clamped here.
    page_min_steps: int = 20
    page_step_buffer: int = 15
    page_max_steps: int = 60
    page_min_tile_coverage: float = 1.0
    page_no_progress_steps: int = 4

    # LLM-arbitrated reconciliation (spec §5.5 final paragraph).
    # Enabled by default for P&IDs; capped per spec [DECISION] default 25.
    arbitrate_conflicts: bool = True
    max_arbitrations_per_run: int = 25

    # Second arbitration pass: re-crop every medium/low-confidence entity and
    # ask the model to confirm, revise, or reject the detection. Drastically
    # reduces run-to-run entity-count variance on accessory-heavy drawings
    # (DEXPI C01, grit-washer, OPEN100-3) without affecting the high-confidence
    # core. Typical cost: ~$0.005 per entity examined.
    arbitrate_low_confidence: bool = True
    arb_low_conf_max: int = 200                 # hard cap on entities examined per run
    arb_low_conf_crop_pad_px: int = 60          # larger than the main pass — tiny accessory symbols need context
    arb_low_conf_crop_max_dim: int = 400        # long side sent to the model; ~100 image tokens

    # Adaptive zoom-in pass for dangling / unsnapped edges. Runs between the
    # first reconcile and the arbitration pass; emits new line annotations that
    # feed back into reconcile so the resulting graph picks up missed
    # connectivity. Budget is per page (not per run) because dangling edges
    # scale with diagram density.
    resolve_dangling_edges: bool = True
    max_edge_resolves_per_page: int = 15
    edge_resolve_focus_window_px: int = 400
    edge_resolve_max_steps_per_edge: int = 5


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    budgets: RuntimeBudgets = field(default_factory=RuntimeBudgets)
    symbol_perception: SymbolPerceptionConfig = field(default_factory=SymbolPerceptionConfig)
    pid: PidConfig = field(default_factory=PidConfig)
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    process_context: list[dict] = field(default_factory=list)
    knowledge: dict = field(default_factory=dict)
    raster_proposals: dict | None = None
    raster_ink_filter: bool = False
    raster_symbol_mode: Literal["baseline", "broad_review"] = "baseline"


def load_config(*, production_open_weight: bool = False) -> Config:
    if not production_open_weight:
        return Config()
    from diagex.llm.model_policy import FAST_MODEL, apply_production_profile
    llm = LLMConfig(
        transport="openrouter", model=FAST_MODEL,
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api").rstrip("/"),
    )
    return apply_production_profile(Config(llm=llm))
