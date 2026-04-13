from __future__ import annotations

from pathlib import Path

import pytest

from sva_toolkit.cli.main import main


pytestmark = pytest.mark.integration


def test_timing_validate_accepts_sample_diagram(runner, timing_diagram_path: Path) -> None:
    result = runner.invoke(main, ["timing", "validate", str(timing_diagram_path)], prog_name="sva")

    assert result.exit_code == 0
    assert result.output.strip() == "valid"


def test_timing_render_writes_svg(runner, timing_diagram_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "diagram.svg"

    result = runner.invoke(
        main,
        ["timing", "render", str(timing_diagram_path), "-o", str(output_path)],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "<svg" in output_path.read_text(encoding="utf-8")


def test_timing_emit_sva_emits_parameterized_properties(runner, timing_diagram_path: Path) -> None:
    result = runner.invoke(main, ["timing", "emit-sva", str(timing_diagram_path)], prog_name="sva")

    assert result.exit_code == 0
    assert "property ready_window(int MAX_WAIT);" in result.output
    assert "$rose(valid) |-> ##[0:MAX_WAIT] (valid && ready);" in result.output
    assert "stable_data_from_asserted_until_handshake" in result.output
