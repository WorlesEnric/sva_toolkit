from __future__ import annotations

import json
from pathlib import Path

import pytest

from sva_toolkit.cli.exit_codes import ExitCode
from sva_toolkit.cli.main import main
from sva_toolkit.data import DatasetBuilder
from sva_toolkit.runtime.diagnostics import Diagnostics
from sva_toolkit.runtime.llm import LLMConfig


pytestmark = pytest.mark.integration


class _FailingLLMClient:
    def __init__(self) -> None:
        self.config = LLMConfig(model="fake-model", api_key="dummy-key")

    def generate(self, *_args: object, **_kwargs: object) -> str:
        raise RuntimeError("429 Too Many Requests")


def test_dataset_builder_handles_duplicate_cache_keys_under_multiprocessing(tmp_path: Path) -> None:
    # R8 regression: concurrent writers to the same cache key must leave one coherent JSON payload behind.
    cache_dir = tmp_path / "cache"
    diagnostics = Diagnostics()
    builder = DatasetBuilder(cache_dir=cache_dir, num_workers=4, diagnostics=diagnostics)
    items = [
        {"id": f"row-{index}", "SVA": "assert property (@(posedge clk) req |-> ##1 ack);"}
        for index in range(64)
    ]

    entries = builder.build_dataset(
        items,
        generate_svad=False,
        generate_cot=False,
        use_multiprocessing=True,
        rate_limit_delay=0,
    )

    assert len(entries) == 64
    assert all(entry.SVA == items[0]["SVA"] for entry in entries)
    cache_files = sorted(cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    assert not list(cache_dir.glob("*.tmp.*"))
    for cache_file in cache_files:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        assert payload["SVA"] == items[0]["SVA"]
        assert payload["__cache_schema"] >= 1
    assert diagnostics.render_summary() == ""


def test_data_build_cli_surfaces_translator_fallback_metadata_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    runner,
    tmp_path: Path,
) -> None:
    # R7 regression: LLM fallback must remain visible in both the output row metadata and CLI diagnostics.
    input_path = tmp_path / "dataset.json"
    output_path = tmp_path / "dataset.jsonl"
    input_path.write_text(
        json.dumps([{"id": "row-1", "SVA": "assert property (@(posedge clk) req |-> ##1 ack);"}]),
        encoding="utf-8",
    )

    builder = DatasetBuilder(
        llm_client=_FailingLLMClient(),
        cache_dir=tmp_path / "cache",
        num_workers=1,
    )
    monkeypatch.setattr("sva_toolkit.cli.main._build_dataset_builder", lambda _model, _workers: builder)

    result = runner.invoke(
        main,
        ["data", "build", str(input_path), "--model", "fake-model", "--workers", "1", "-o", str(output_path)],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.SUCCESS
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["metadata"]["svad_source"] == "translator_fallback"
    assert rows[0]["metadata"]["svad_error"] == "429 Too Many Requests"
    assert rows[0]["SVAD"]
    assert "dataset entry 0 used translator fallback" in result.stderr
    assert "Diagnostics summary: translator_fallback=1" in result.stderr
