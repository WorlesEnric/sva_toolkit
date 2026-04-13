"""CLI integration tests for timing/SVA conversion."""

from click.testing import CliRunner

from sva_toolkit.timing.cli.main import main

from ..unit.test_parser import SAMPLE_DIAGRAM, SYMBOLIC_DIAGRAM


def test_emit_sva_command_outputs_properties_for_v1(temp_dir):
    input_file = temp_dir / "req_ack.tdg"
    input_file.write_text(SAMPLE_DIAGRAM, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["emit-sva", str(input_file)])

    assert result.exit_code == 0
    assert "property p_ack_after_req(int LAT_MIN, int LAT_MAX);" in result.output
    assert "$rose(req) |-> ##[LAT_MIN:LAT_MAX] $rose(ack);" in result.output


def test_emit_sva_command_outputs_exact_symbolic_property(temp_dir):
    input_file = temp_dir / "axi_wait.tdg"
    input_file.write_text(SYMBOLIC_DIAGRAM, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["emit-sva", str(input_file)])

    assert result.exit_code == 0
    assert "(valid && !ready) |-> ##[0:READY_WAIT_MAX] (valid && ready);" in result.output
