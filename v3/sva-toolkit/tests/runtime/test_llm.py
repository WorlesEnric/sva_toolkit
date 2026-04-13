from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from sva_toolkit.runtime.llm import LLMClient, LLMConfig


def test_llm_config_uses_openai_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")

    config = LLMConfig(model="test-model")

    assert config.model == "test-model"
    assert config.api_key == "env-secret"
    assert config.base_url is None
    assert config.temperature == 0.7
    assert config.max_tokens == 4096


def test_llm_client_generate_uses_lazy_openai_import(monkeypatch) -> None:
    create_calls: list[dict[str, object]] = []
    client_calls: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, *, base_url: str | None = None, api_key: str) -> None:
            client_calls.append({"base_url": base_url, "api_key": api_key})
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs: object) -> object:
            create_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="synthetic response"))]
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    client = LLMClient(
        LLMConfig(
            model="test-model",
            api_key="secret",
            base_url="https://example.test/v1",
            temperature=0.2,
            max_tokens=128,
        )
    )

    assert client.generate("system prompt", "user prompt") == "synthetic response"
    assert client_calls == [{"base_url": "https://example.test/v1", "api_key": "secret"}]
    assert create_calls == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            "temperature": 0.2,
            "max_tokens": 128,
        }
    ]


def test_llm_client_generate_raises_helpful_error_without_openai(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai":
            raise ImportError("missing openai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    client = LLMClient(LLMConfig(model="test-model", api_key="secret"))

    with pytest.raises(RuntimeError, match="Install sva-toolkit\\[llm\\] for LLM support"):
        client.generate("system prompt", "user prompt")
