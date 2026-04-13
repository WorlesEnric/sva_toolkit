from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sva_toolkit.cli.main import main
from sva_toolkit.timing.frontend.parser import parse_diagram

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def test_sva_help_returns_zero() -> None:
    result = CliRunner().invoke(main, ["--help"], prog_name="sva")

    assert result.exit_code == 0
    assert "formal" in result.output
    assert "timing" in result.output


def test_sva_formal_help_returns_zero() -> None:
    result = CliRunner().invoke(main, ["formal", "--help"], prog_name="sva")

    assert result.exit_code == 0
    assert "Formal verification commands." in result.output


def test_sva_timing_help_returns_zero() -> None:
    result = CliRunner().invoke(main, ["timing", "--help"], prog_name="sva")

    assert result.exit_code == 0
    assert "Timing diagram commands." in result.output


def test_sva_timing_validate_accepts_valid_dsl(tmp_path) -> None:
    input_file = tmp_path / "valid.tdsl"
    input_file.write_text(
        """
        diagram test {
          clock posedge clk;
          lane req: bit = 0 0 1 1;
          ticks 4;
        }
        """,
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["timing", "validate", str(input_file)], prog_name="sva")

    assert result.exit_code == 0
    assert result.output.strip() == "valid"


def test_sva_timing_validate_rejects_invalid_dsl(tmp_path) -> None:
    input_file = tmp_path / "invalid.tdsl"
    input_file.write_text(
        """
        diagram broken {
          lane req: bit = 0 0 1 1;
          ticks 4;
        }
        """,
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["timing", "validate", str(input_file)], prog_name="sva")

    assert result.exit_code != 0
    assert "missing clock declaration" in result.output


def test_sva_timing_emit_sva_example_matches_fixture(tmp_path) -> None:
    input_file = EXAMPLES_DIR / "td" / "11_emit_sva_bridge.td"
    expected_output = (EXAMPLES_DIR / "sva" / "11_emit_sva_bridge.sv").read_text(encoding="utf-8").rstrip("\n")
    output_file = tmp_path / "emit_output.sv"

    result = CliRunner().invoke(
        main,
        ["timing", "emit-sva", str(input_file), "-o", str(output_file)],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert output_file.read_text(encoding="utf-8") == expected_output


def test_sva_timing_extract_sva_example_generates_valid_diagram(tmp_path) -> None:
    input_file = EXAMPLES_DIR / "sva" / "12_extract_sva_bridge.sv"
    output_file = tmp_path / "extract_output.td"

    result = CliRunner().invoke(
        main,
        ["timing", "extract-sva", str(input_file), "-o", str(output_file)],
        prog_name="sva",
    )

    assert result.exit_code == 0
    document = parse_diagram(output_file.read_text(encoding="utf-8"))
    assert len(document.properties) >= 1
    assert {signal.name for signal in document.signals} == {"ack", "req"}
