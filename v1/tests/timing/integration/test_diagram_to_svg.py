"""CLI integration tests for timing diagram rendering and extraction."""

from click.testing import CliRunner

from sva_toolkit.timing.cli.main import main

from ..unit.test_parser import SAMPLE_DIAGRAM


def test_render_command_outputs_svg(temp_dir):
    input_file = temp_dir / "req_ack.tdg"
    input_file.write_text(SAMPLE_DIAGRAM, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["render", str(input_file)])

    assert result.exit_code == 0
    assert result.output.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "req_ack" in result.output


def test_extract_sva_command_outputs_symbolic_dsl(temp_dir):
    input_file = temp_dir / "ready_wait.sva"
    input_file.write_text(
        "property ready_wait; @(posedge clk) req |-> ##[1:$] ack; endproperty",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["extract-sva", str(input_file)])

    assert result.exit_code == 0
    assert "diagram ready_wait {" in result.output
    assert "window ready_wait__delay_window = between trigger and ready_wait__response [1:$];" in result.output
