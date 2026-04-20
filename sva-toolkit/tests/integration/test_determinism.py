from __future__ import annotations

import json
from pathlib import Path

import pytest

from sva_toolkit.cli.exit_codes import ExitCode
from sva_toolkit.cli.main import main


pytestmark = pytest.mark.integration


def test_generate_cli_is_reproducible_for_a_fixed_seed(runner) -> None:
    # R5 regression: generated assertion batches must remain byte-identical for the same explicit seed.
    first = runner.invoke(main, ["generate", "--seed", "42", "--count", "5"], prog_name="sva")
    second = runner.invoke(main, ["generate", "--seed", "42", "--count", "5"], prog_name="sva")
    third = runner.invoke(main, ["generate", "--seed", "43", "--count", "5"], prog_name="sva")

    assert first.exit_code == ExitCode.SUCCESS
    assert second.exit_code == ExitCode.SUCCESS
    assert third.exit_code == ExitCode.SUCCESS
    assert first.output == second.output
    assert first.output != third.output
    assert first.stderr == ""
    assert second.stderr == ""
    assert third.stderr == ""


def test_generate_cli_reports_coverage_deterministically_for_seeded_batches(runner) -> None:
    # R15 regression: the generator's coverage summary must stay attached to the emitted batch and remain reproducible.
    first = runner.invoke(main, ["generate", "--seed", "11", "--count", "4", "--coverage"], prog_name="sva")
    second = runner.invoke(main, ["generate", "--seed", "11", "--count", "4", "--coverage"], prog_name="sva")

    assert first.exit_code == ExitCode.SUCCESS
    assert second.exit_code == ExitCode.SUCCESS
    assert first.output == second.output
    assert "Coverage:" in first.output
    assert "property p_gen_0;" in first.output


def test_generate_cli_reports_implicit_seed_for_reproduction(runner) -> None:
    # R5 regression: unseeded generation must continue to print the chosen seed for later reproduction.
    result = runner.invoke(main, ["generate", "--count", "1"], prog_name="sva")

    assert result.exit_code == ExitCode.SUCCESS
    assert "Using generation seed:" in result.stderr
    assert "property p_gen_0;" in result.output


def test_offline_data_build_is_byte_stable_for_identical_inputs(runner, tmp_path: Path) -> None:
    # R5 regression: downstream offline dataset builds must remain diff-friendly for identical inputs.
    input_path = tmp_path / "dataset.json"
    first_output = tmp_path / "dataset-a.jsonl"
    second_output = tmp_path / "dataset-b.jsonl"
    payload = [
        {"id": "row-1", "SVA": "assert property (@(posedge clk) req |-> ##1 ack);"},
        {"id": "row-2", "SVA": "assert property (@(posedge clk) disable iff (!rst_n) req |=> ack);"},
    ]
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    first = runner.invoke(main, ["data", "build", str(input_path), "--workers", "1", "-o", str(first_output)], prog_name="sva")
    second = runner.invoke(
        main,
        ["data", "build", str(input_path), "--workers", "1", "-o", str(second_output)],
        prog_name="sva",
    )

    assert first.exit_code == ExitCode.SUCCESS
    assert second.exit_code == ExitCode.SUCCESS
    assert first_output.read_text(encoding="utf-8") == second_output.read_text(encoding="utf-8")
