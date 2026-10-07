"""Web presets shared by the form and request configuration."""

from diagex.llm.model_policy import ESCALATION_MODEL, FAST_MODEL

DEEPSEEK_FLASH_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_MODEL_POLICY = "deepseek-flash"
MODEL_PROFILES = {
    "deepseek-flash": {
        "provider": "openrouter",
        "vision_model": DEEPSEEK_FLASH_MODEL,
        "reasoning_model": DEEPSEEK_FLASH_MODEL,
        "escalation_model": DEEPSEEK_FLASH_MODEL,
        "engine": "evidence-v2",
        "hint": "DeepSeek V4.1 Flash handles vision, reasoning, and bounded source reinspection.",
    },
    "production-open-weight": {
        "provider": "openrouter",
        "vision_model": FAST_MODEL,
        "reasoning_model": FAST_MODEL,
        "escalation_model": ESCALATION_MODEL,
        "engine": "evidence-v2",
        "hint": "Qwen 35B handles routine work; Qwen 122B handles bounded source reinspection.",
    },
}
