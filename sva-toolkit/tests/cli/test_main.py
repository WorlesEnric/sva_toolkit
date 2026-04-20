from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sva_toolkit.cli.main import main
from sva_toolkit.formal.model import CheckResult, ImplicationResult


@pytest.mark.parametrize(
    ("args", "expected_text"),
    [
        (["--help"], "parse"),
        (["--help"], "formal"),
        (["--help"], "timing"),
        (["--help"], "generate"),
        (["--help"], "describe"),
        (["--help"], "data"),
        (["parse", "--help"], "Parse a SystemVerilog assertion"),
        (["formal", "--help"], "Formal verification commands."),
        (["formal", "check", "--help"], "--backend"),
        (["formal", "equivalent", "--help"], "--timeout"),
        (["formal", "relationship", "--help"], "--depth"),
        (["timing", "--help"], "Timing diagram commands."),
        (["timing", "render", "--help"], "--format"),
        (["timing", "validate", "--help"], "Validate"),
        (["timing", "emit-sva", "--help"], "--allow-lossy"),
        (["timing", "extract-sva", "--help"], "--depth"),
        (["timing", "bundle-sva", "--help"], "Bundle"),
        (["generate", "--help"], "--mode"),
        (["describe", "--help"], "Description commands."),
        (["describe", "svad", "--help"], "--format"),
        (["describe", "cot", "--help"], "--format"),
        (["data", "--help"], "Dataset and benchmark commands."),
        (["data", "build", "--help"], "--workers"),
        (["data", "benchmark", "--help"], "--model"),
    ],
)
def test_help_surfaces_expected_commands(args: list[str], expected_text: str) -> None:
    result = CliRunner().invoke(main, args, prog_name="sva")

    assert result.exit_code == 0
    assert expected_text in result.output


def test_parse_command_supports_json_output() -> None:
    result = CliRunner().invoke(
        main,
        ["parse", "@(posedge clk) req |-> ##1 ack", "--format", "json"],
        prog_name="sva",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["clocking"]["signal"]["name"] == "clk"
    assert payload["body"]["kind"] == "implication"


def test_formal_check_reports_service_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFormalService:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def check_implication(self, antecedent: str, consequent: str, **kwargs) -> CheckResult:
            assert antecedent == "req |-> ack"
            assert consequent == "req |-> ##1 ack"
            assert kwargs == {"clock": None, "clock_edge": None, "reset": None}
            return CheckResult(
                result=ImplicationResult.IMPLIES,
                message="proved",
            )

    monkeypatch.setattr("sva_toolkit.cli.formal_flags.FormalService", _FakeFormalService)

    result = CliRunner().invoke(
        main,
        ["formal", "check", "req |-> ack", "req |-> ##1 ack", "--backend", "ebmc", "--timeout", "10", "--depth", "4"],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert "Result: implies" in result.output
    assert "Message: proved" in result.output


def test_timing_validate_accepts_valid_dsl(tmp_path: Path) -> None:
    input_file = tmp_path / "valid.td"
    input_file.write_text(
        """
        diagram demo {
          clock posedge clk;
          lane req: bit = 0 1 1 1;
          ticks 4;
        }
        """,
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["timing", "validate", str(input_file)], prog_name="sva")

    assert result.exit_code == 0
    assert result.output.strip() == "valid"


def test_generate_command_emits_requested_count() -> None:
    result = CliRunner().invoke(main, ["generate", "--count", "2", "--seed", "7"], prog_name="sva")

    assert result.exit_code == 0
    assert "property p_gen_0;" in result.output
    assert "property p_gen_1;" in result.output


@pytest.mark.parametrize(
    ("subcommand", "expected_text"),
    [
        ("svad", "Relevant Signals"),
        ("cot", "Chain-of-Thought"),
    ],
)
def test_describe_commands_render_text(subcommand: str, expected_text: str) -> None:
    result = CliRunner().invoke(
        main,
        ["describe", subcommand, "assert property (@(posedge clk) req |-> ##1 ack);"],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert expected_text in result.output


def test_data_build_writes_jsonl_output(tmp_path: Path) -> None:
    input_file = tmp_path / "dataset.json"
    output_file = tmp_path / "dataset.jsonl"
    input_file.write_text(
        json.dumps(
            [
                {
                    "id": "row-1",
                    "SVA": "assert property (@(posedge clk) req |-> ##1 ack);",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["data", "build", str(input_file), "-o", str(output_file), "--workers", "1"],
        prog_name="sva",
    )

    assert result.exit_code == 0
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["SVA"].startswith("assert property")
    assert "CoT" in payload


def test_data_benchmark_writes_json_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sva_toolkit.data as data_module

    input_file = tmp_path / "benchmark.json"
    output_file = tmp_path / "results.json"
    input_file.write_text(
        json.dumps(
            [
                {
                    "SVAD": "Acknowledge a request in the next cycle.",
                    "SVA": "assert property (@(posedge clk) req |-> ##1 ack);",
                }
            ]
        ),
        encoding="utf-8",
    )

    class _FakeRunner:
        def __init__(self, *, llm_clients, formal_service=None, num_workers=4, cache_dir=None, **_kwargs) -> None:
            self.llm_clients = llm_clients
            self.formal_service = formal_service
            self.num_workers = num_workers
            self.cache_dir = cache_dir

        def run_benchmark(
            self, dataset, llm_client, *, use_multiprocessing=True, rate_limit_delay=0.5, progress_callback=None
        ):
            assert dataset[0]["SVAD"].startswith("Acknowledge")
            assert llm_client.config.model == "fake-model"
            return data_module.BenchmarkResult(
                model_name="fake-model",
                total_items=1,
                equivalent_count=1,
            )

    monkeypatch.setattr(data_module, "BenchmarkRunner", _FakeRunner)

    result = CliRunner().invoke(
        main,
        ["data", "benchmark", str(input_file), "--model", "fake-model", "--workers", "1", "-o", str(output_file)],
        prog_name="sva",
    )

    assert result.exit_code == 0
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["model_name"] == "fake-model"
    assert payload["equivalent_count"] == 1
