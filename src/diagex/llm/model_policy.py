"""Explicit open-weight allowlist for the production profile.

Model cards and Apache-2.0 licenses verified against the publisher's weight
repositories on 2026-09-08. Endpoint capability is checked separately before a
live run; a provider alias or a 'Flash' marketing name is not weight provenance.
"""

OPEN_WEIGHT_MODELS = {
    "qwen/qwen3.5-9b": "https://huggingface.co/Qwen/Qwen3.5-9B",
    "qwen/qwen3.5-35b-a3b": "https://huggingface.co/Qwen/Qwen3.5-35B-A3B",
    "qwen/qwen3.5-122b-a10b": "https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
}
FAST_MODEL = "qwen/qwen3.5-35b-a3b"
ESCALATION_MODEL = "qwen/qwen3.5-122b-a10b"


def apply_production_profile(config, *, workflow="fixed"):
    """Hosted default; adaptive remains opt-in until its empirical gate passes."""
    from dataclasses import replace

    config.llm = replace(
        config.llm,
        transport="openrouter",
        model=FAST_MODEL,
        vision_model=FAST_MODEL,
        reasoning_model=FAST_MODEL,
        escalation_model=ESCALATION_MODEL,
        production_open_weight=True,
    )
    config.pid.engine = "evidence-v2"
    config.symbol_perception = replace(config.symbol_perception, workflow=workflow)
    validate_production_models(config.llm)
    return config


def validate_production_models(config):
    if not config.production_open_weight:
        return
    for model in (
        config.model,
        config.vision_model,
        config.reasoning_model,
        config.escalation_model,
    ):
        if model is not None and model not in OPEN_WEIGHT_MODELS:
            raise ValueError(
                f"Production requires a verified open-weight model; {model!r} is not approved. Closed-source references are evaluation-only."
            )
