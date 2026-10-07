from types import SimpleNamespace

import pytest

from diagex.config import LLMConfig
from diagex.llm.client import LLMClient


@pytest.mark.parametrize("transport,thinking,expected", [
    ("openrouter", "disabled", False),
    ("openrouter", "adaptive", True),
    ("anthropic", "disabled", True),
])
def test_disabled_openrouter_reasoning_omits_effort(monkeypatch, transport, thinking, expected):
    captured = {}
    class Stream:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get_final_message(self):
            return SimpleNamespace(content=[], usage=None)
    def stream(**kwargs):
        captured.update(kwargs)
        return Stream()
    client = LLMClient(LLMConfig(transport=transport, model="deepseek/deepseek-v4.1-flash",
                               openrouter_api_key="test", anthropic_api_key="test"))
    monkeypatch.setattr(client._client.messages, "stream", stream)
    client.messages_create(system="test", messages=[], max_tokens=100,
                           thinking={"type": thinking}, output_config={"effort": "low"})
    assert ("output_config" in captured) is expected
    assert captured["thinking"]["type"] == thinking
    client._client.close()
