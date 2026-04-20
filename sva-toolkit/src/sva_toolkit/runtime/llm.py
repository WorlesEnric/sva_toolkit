from __future__ import annotations

from datetime import datetime, timezone
import email.utils
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from sva_toolkit.runtime.diagnostics import DEFAULT_DIAGNOSTICS, Diagnostics
from sva_toolkit.runtime.retry import RetryPolicy, TransientResponseError, retry


@dataclass
class LLMConfig:
    model: str
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    jitter: bool = True


class LLMClient:
    def __init__(self, config: LLMConfig, *, diagnostics: Diagnostics | None = None) -> None:
        self.config = config
        self._client: object | None = None
        self.diagnostics = diagnostics or DEFAULT_DIAGNOSTICS
        self.last_retry_count = 0

    @retry(lambda self, *_args, **_kwargs: self._retry_policy())
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._get_client().chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            retry_error = self._coerce_retryable_error(exc)
            if retry_error is not None:
                raise retry_error from exc
            raise
        return self._extract_content(response)

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

    def _retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.config.max_retries,
            backoff_base=self.config.backoff_base,
            backoff_cap=self.config.backoff_cap,
            jitter=self.config.jitter,
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        return response.choices[0].message.content or ""

    @classmethod
    def _coerce_retryable_error(cls, exc: Exception) -> TransientResponseError | None:
        if isinstance(exc, TransientResponseError):
            return exc

        status_code = cls._status_code_from_exception(exc)
        if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
            return TransientResponseError(
                str(exc),
                retry_after=cls._retry_after_from_exception(exc),
                status_code=status_code,
            )
        return None

    @staticmethod
    def _status_code_from_exception(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
        return None

    @classmethod
    def _retry_after_from_exception(cls, exc: Exception) -> float | None:
        retry_after = getattr(exc, "retry_after", None)
        parsed_retry_after = cls._parse_retry_after(retry_after)
        if parsed_retry_after is not None:
            return parsed_retry_after

        headers = getattr(exc, "headers", None)
        parsed_from_headers = cls._parse_retry_after_header(headers)
        if parsed_from_headers is not None:
            return parsed_from_headers

        response = getattr(exc, "response", None)
        return cls._parse_retry_after_header(getattr(response, "headers", None))

    @classmethod
    def _parse_retry_after_header(cls, headers: Any) -> float | None:
        if not headers:
            return None
        if isinstance(headers, dict):
            for key, value in headers.items():
                if str(key).lower() == "retry-after":
                    return cls._parse_retry_after(value)
            return None
        return cls._parse_retry_after(getattr(headers, "get", lambda _key, _default=None: None)("Retry-After"))

    @staticmethod
    def _parse_retry_after(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return max(0.0, float(value))

        text = str(value).strip()
        if not text:
            return None
        try:
            return max(0.0, float(text))
        except ValueError:
            pass

        try:
            parsed = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
