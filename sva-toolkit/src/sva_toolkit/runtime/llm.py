from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import ModuleType


@dataclass
class LLMConfig:
    model: str
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: object | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._get_client().chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _get_client(self) -> object:
        if self._client is None:
            openai = self._import_openai()
            self._client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    @staticmethod
    def _import_openai() -> ModuleType:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("Install sva-toolkit[llm] for LLM support") from exc
        return openai
