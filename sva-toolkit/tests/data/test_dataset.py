from __future__ import annotations

import json
from pathlib import Path

from sva_toolkit.data import DatasetBuilder, DatasetEntry
from sva_toolkit.runtime.llm import LLMConfig


class _FakeLLMClient:
    def __init__(self, responses: list[str] | None = None, *, error: Exception | None = None) -> None:
        self.config = LLMConfig(model="fake-model", api_key="secret")
        self._responses = list(responses or [])
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return "Synthetic SVAD"


class _FakeTranslator:
    def __init__(self, text: str = "Translator fallback") -> None:
        self.text = text
        self.calls: list[str] = []

    def translate(self, sva_code: str) -> str:
        self.calls.append(sva_code)
        return self.text


def test_dataset_entry_to_dict_excludes_empty_fields() -> None:
    entry = DatasetEntry(SVA="assert property (@(posedge clk) req |-> ack);")

    assert entry.to_dict() == {"SVA": "assert property (@(posedge clk) req |-> ack);"}


def test_dataset_builder_offline_mode_skips_svad_and_builds_cot(tmp_path: Path) -> None:
    builder = DatasetBuilder(cache_dir=tmp_path / "cache")

    entries = builder.build_dataset(
        [{"SVA": "assert property (@(posedge clk) req |-> ##1 ack);"}],
        generate_svad=True,
        generate_cot=True,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )

    assert len(entries) == 1
    assert entries[0].SVAD is None
    assert entries[0].CoT is not None
    assert entries[0].metadata["svad_skipped"] == "offline_mode"


def test_dataset_builder_uses_translator_fallback_when_llm_generation_fails(tmp_path: Path) -> None:
    llm_client = _FakeLLMClient(error=RuntimeError("network down"))
    translator = _FakeTranslator("Fallback SVAD")
    builder = DatasetBuilder(
        llm_client=llm_client,
        translator=translator,
        cache_dir=tmp_path / "cache",
    )

    processed = builder.process_entry(
        DatasetEntry(SVA="assert property (@(posedge clk) req |-> ack);"),
        generate_svad=True,
        generate_cot=False,
    )

    assert processed.SVAD == "Fallback SVAD"
    assert processed.metadata["svad_source"] == "translator_fallback"
    assert "network down" in processed.metadata["svad_error"]
    assert translator.calls == ["assert property (@(posedge clk) req |-> ack);"]


def test_dataset_builder_reuses_cache_without_reissuing_llm_calls(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first_client = _FakeLLMClient(["LLM SVAD"])
    first_builder = DatasetBuilder(llm_client=first_client, cache_dir=cache_dir)
    payload = [{"SVA": "assert property (@(posedge clk) req |-> ack);"}]

    first_entries = first_builder.build_dataset(
        payload,
        generate_svad=True,
        generate_cot=False,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )
    second_client = _FakeLLMClient(error=AssertionError("cache was not used"))
    second_builder = DatasetBuilder(llm_client=second_client, cache_dir=cache_dir)
    second_entries = second_builder.build_dataset(
        payload,
        generate_svad=True,
        generate_cot=False,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )

    assert first_entries[0].SVAD == "LLM SVAD"
    assert second_entries[0].SVAD == "LLM SVAD"
    assert len(first_client.calls) == 1
    assert second_client.calls == []
    assert first_builder.get_cache_stats()["cached_items"] == 1


def test_dataset_builder_build_from_file_writes_augmented_entries(tmp_path: Path) -> None:
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text(
        json.dumps([{"id": "row-1", "SVA": "assert property (@(posedge clk) req |-> ack);"}]),
        encoding="utf-8",
    )
    builder = DatasetBuilder(
        llm_client=_FakeLLMClient(["Generated description"]),
        cache_dir=tmp_path / "cache",
    )

    entries = builder.build_from_file(
        input_path=input_file,
        output_path=output_file,
        generate_svad=True,
        generate_cot=True,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )

    written = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert written[0]["id"] == "row-1"
    assert written[0]["SVAD"] == "Generated description"
    assert "CoT" in written[0]
