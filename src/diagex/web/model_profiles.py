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
        "hint": "Uses DeepSeek V4.1 Flash to read symbols, interpret connections, and recheck uncertain details.",
        "hint_zh": "使用 DeepSeek V4.1 Flash 识读符号、解释连接关系，并复查不确定的细节。",
    },
    "production-open-weight": {
        "provider": "openrouter",
        "vision_model": FAST_MODEL,
        "reasoning_model": FAST_MODEL,
        "escalation_model": ESCALATION_MODEL,
        "engine": "evidence-v2",
        "hint": "Uses Qwen 35B for extraction and the larger Qwen 122B to recheck uncertain details in the drawing.",
        "hint_zh": "使用 Qwen 35B 提取图纸信息，并用更大的 Qwen 122B 复查图中的不确定细节。",
    },
}
