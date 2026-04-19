from __future__ import annotations

import json
from pathlib import Path

import pytest

from sva_toolkit.cli.main import main


pytestmark = pytest.mark.integration


def test_data_build_writes_jsonl_with_mocked_builder(monkeypatch: pytest.MonkeyPatch, runner, tmp_path: Path) -> None:
    input_path = tmp_path / "dataset.json"
    output_path = tmp_path / "dataset.jsonl"
    input_path.write_text(
        json.dumps([{"id": "row-1", "SVA": "assert property (@(posedge clk) req |-> ack);"}]), encoding="utf-8"
    )
    captured: dict[str, object] = {}

    class _FakeBuilder:
        def build_dataset(self, payload, **kwargs):
            captured["payload"] = payload
            captured["kwargs"] = kwargs
            return [
                {
                    "id": "row-1",
                    "SVA": payload[0]["SVA"],
                    "SVAD": "Mocked description",
                    "CoT": "Mocked reasoning",
                }
            ]

    def _fake_build_dataset_builder(model: str | None, workers: int) -> _FakeBuilder:
        captured["config"] = (model, workers)
        return _FakeBuilder()

    monkeypatch.setattr("sva_toolkit.cli.main._build_dataset_builder", _fake_build_dataset_builder)

    result = runner.invoke(
        main,
        ["data", "build", str(input_path), "--model", "fake-llm", "--workers", "1", "-o", str(output_path)],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert captured["config"] == ("fake-llm", 1)
    assert captured["kwargs"] == {
        "generate_svad": True,
        "generate_cot": True,
        "use_multiprocessing": False,
        "rate_limit_delay": 0,
    }
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written == [
        {
            "id": "row-1",
            "SVA": "assert property (@(posedge clk) req |-> ack);",
            "SVAD": "Mocked description",
            "CoT": "Mocked reasoning",
        }
    ]


def test_data_benchmark_writes_json_with_mocked_runner(monkeypatch: pytest.MonkeyPatch, runner, tmp_path: Path) -> None:
    input_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "benchmark-results.json"
    input_path.write_text(
        json.dumps(
            [{"SVAD": "Acknowledge request on next cycle.", "SVA": "assert property (@(posedge clk) req |-> ##1 ack);"}]
        ),
        encoding="utf-8",
    )

    class _FakeRunner:
        def __init__(self) -> None:
            self.llm_clients = [type("_Client", (), {"config": type("_Config", (), {"model": "fake-model"})()})()]

        def run_benchmark(self, dataset, llm_client, **kwargs):
            assert dataset[0]["SVAD"].startswith("Acknowledge")
            assert llm_client.config.model == "fake-model"
            assert kwargs == {"use_multiprocessing": False, "rate_limit_delay": 0}
            return type(
                "_BenchmarkResult",
                (),
                {
                    "to_dict": lambda self: {
                        "model_name": "fake-model",
                        "total_items": 1,
                        "equivalent_count": 1,
                    }
                },
            )()

    monkeypatch.setattr("sva_toolkit.cli.main._build_benchmark_runner", lambda model, workers: _FakeRunner())

    result = runner.invoke(
        main,
        ["data", "benchmark", str(input_path), "--model", "fake-model", "--workers", "1", "-o", str(output_path)],
        prog_name="sva",
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "model_name": "fake-model",
        "total_items": 1,
        "equivalent_count": 1,
    }
