from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sva_toolkit.data import DatasetBuilder, DatasetEntry
from sva_toolkit.runtime.diagnostics import Diagnostics
from sva_toolkit.runtime.llm import LLMClient, LLMConfig
from sva_toolkit.runtime.retry import RetryExhaustedError, RetryPolicy, TransientResponseError, retry
import sva_toolkit.runtime.retry as retry_module


class _FakeHTTPError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class _FakeOpenAIClient:
    def __init__(self, scripted_results: list[object]) -> None:
        self._scripted_results = list(scripted_results)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls: list[dict[str, Any]] = []

    def _create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        result = self._scripted_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result))],
        )


class _FakeTranslator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[str] = []

    def translate(self, sva_code: str) -> str:
        self.calls.append(sva_code)
        return self.text


def test_llm_client_retries_rate_limits_and_respects_retry_after(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", lambda delay: sleeps.append(delay))
    llm_client = LLMClient(
        LLMConfig(model="fake-model", api_key="secret", max_retries=2, backoff_base=0.1),
    )
    transport = _FakeOpenAIClient(
        [
            _FakeHTTPError(429, "rate limited", headers={"Retry-After": "0.25"}),
            "Recovered response",
        ]
    )
    llm_client._client = transport

    result = llm_client.generate("system", "user")

    assert result == "Recovered response"
    assert llm_client.last_retry_count == 1
    assert len(transport.calls) == 2
    assert sleeps == [0.25]


def test_dataset_builder_uses_translator_fallback_after_retry_exhaustion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", lambda delay: sleeps.append(delay))

    diagnostics = Diagnostics()
    llm_client = LLMClient(
        LLMConfig(
            model="fake-model",
            api_key="secret",
            max_retries=1,
            backoff_base=0.0,
            jitter=False,
        ),
        diagnostics=diagnostics,
    )
    llm_client._client = _FakeOpenAIClient(
        [
            _FakeHTTPError(500, "server error"),
            _FakeHTTPError(500, "server error"),
        ]
    )
    translator = _FakeTranslator("Translator fallback")
    builder = DatasetBuilder(
        llm_client=llm_client,
        translator=translator,
        cache_dir=tmp_path / "cache",
        diagnostics=diagnostics,
    )

    processed = builder.process_entry(
        DatasetEntry(SVA="assert property (@(posedge clk) req |-> ack);"),
        generate_svad=True,
        generate_cot=False,
    )

    assert processed.SVAD == "Translator fallback"
    assert processed.metadata["svad_source"] == "translator_fallback"
    assert llm_client.last_retry_count == 1
    assert translator.calls == ["assert property (@(posedge clk) req |-> ack);"]
    assert sleeps == []
    snapshot = diagnostics.snapshot()
    assert snapshot["retry_exhausted"] == 1
    assert snapshot["translator_fallback"] == 1


def test_retry_decorator_raises_retry_exhausted_error_instead_of_transient_error(monkeypatch) -> None:
    monkeypatch.setattr(retry_module.time, "sleep", lambda _delay: None)

    @retry(RetryPolicy(max_retries=1, backoff_base=0.0, jitter=False))
    def always_transient() -> None:
        raise TransientResponseError("try again")

    with pytest.raises(RetryExhaustedError):
        always_transient()


def test_llm_client_respects_retry_tuning_from_config(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", lambda delay: sleeps.append(delay))

    llm_client = LLMClient(
        LLMConfig(
            model="fake-model",
            api_key="secret",
            max_retries=2,
            backoff_base=0.5,
            backoff_cap=5.0,
            jitter=False,
        ),
    )
    llm_client._client = _FakeOpenAIClient(
        [
            _FakeHTTPError(500, "server error"),
            _FakeHTTPError(500, "server error"),
            _FakeHTTPError(500, "server error"),
        ]
    )

    with pytest.raises(RetryExhaustedError):
        llm_client.generate("system", "user")

    assert llm_client.last_retry_count == 2
    assert sleeps == [0.5, 1.0]
