from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sva_toolkit.cli.exit_codes import BackendUnavailableError, ExitCode, exit_code_for
from sva_toolkit.cli.main import main
from sva_toolkit.formal.model import CheckResult, ImplicationResult


def test_backend_unavailable_error_maps_to_reserved_exit_code() -> None:
    assert exit_code_for(BackendUnavailableError("backend down")) is ExitCode.BACKEND_UNAVAILABLE


def test_formal_check_without_backend_exits_with_tool_missing_code(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFormalService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def check_implication(self, antecedent: str, consequent: str, **_kwargs: object) -> CheckResult:
            assert antecedent == "req |-> ack"
            assert consequent == "req |-> ##1 ack"
            return CheckResult(
                result=ImplicationResult.ERROR,
                message="No formal backend is available. Install 'vcf' or 'ebmc' and retry.",
            )

    monkeypatch.setattr("sva_toolkit.cli.formal_flags.FormalService", _FakeFormalService)

    result = CliRunner().invoke(
        main,
        ["formal", "check", "req |-> ack", "req |-> ##1 ack"],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "No formal backend is available" in result.stderr


def test_parse_command_returns_parse_exit_code() -> None:
    result = CliRunner().invoke(main, ["parse", "@(posedge clk) req |-> ##"], prog_name="sva")

    assert result.exit_code == ExitCode.PARSE_ERROR
    assert "Error:" in result.stderr


def test_data_build_timeout_returns_timeout_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_file = tmp_path / "dataset.json"
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

    def _raise_timeout(self, _system_prompt: str, _user_prompt: str) -> str:
        raise TimeoutError("LLM request timed out")

    monkeypatch.setattr("sva_toolkit.runtime.llm.LLMClient.generate", _raise_timeout)

    result = CliRunner().invoke(
        main,
        ["data", "build", str(input_file), "--model", "fake-model", "--workers", "1"],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.TIMEOUT
    assert "LLM request timed out" in result.stderr


def test_timing_extract_lossy_report_returns_lossy_exit_code(tmp_path: Path) -> None:
    input_file = tmp_path / "unsupported.sv"
    input_file.write_text(
        "property p_accept; @(posedge clk) disable iff (!rst_n) accept_on(abort) req |-> ack; endproperty",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["timing", "extract-sva", str(input_file)], prog_name="sva")

    assert result.exit_code == ExitCode.LOSSY_EXTRACTION
    assert result.stdout == ""
    assert "overall: unsupported" in result.stderr
    assert "WARNING Diagnostics summary: lossy_extraction=1" in result.stderr


def test_verbose_failure_prints_traceback_and_diagnostics_summary(tmp_path: Path) -> None:
    input_file = tmp_path / "unsupported.sv"
    input_file.write_text(
        "property p_accept; @(posedge clk) disable iff (!rst_n) accept_on(abort) req |-> ack; endproperty",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["--verbose", "timing", "extract-sva", str(input_file)],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.LOSSY_EXTRACTION
    assert "Traceback (most recent call last)" in result.stderr
    assert "LossyExtractionError" in result.stderr
    assert "WARNING Diagnostics summary: lossy_extraction=1" in result.stderr


def test_clean_run_has_success_exit_code_and_no_diagnostics_summary() -> None:
    result = CliRunner().invoke(
        main,
        ["parse", "@(posedge clk) req |-> ##1 ack"],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.SUCCESS
    assert "Diagnostics summary" not in result.stderr
