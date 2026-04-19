from __future__ import annotations

from pathlib import Path

import pytest

from sva_toolkit.cli.main import main


pytestmark = pytest.mark.integration


def test_describe_svad_renders_text_for_sample_sva(runner, sample_sva: str) -> None:
    result = runner.invoke(main, ["describe", "svad", sample_sva], prog_name="sva")

    assert result.exit_code == 0
    assert "Relevant Signals" in result.output
    assert "Check Condition" in result.output


def test_describe_cot_reads_sva_from_file(runner, sample_sva: str, tmp_path: Path) -> None:
    input_path = tmp_path / "sample.sva"
    input_path.write_text(sample_sva, encoding="utf-8")

    result = runner.invoke(main, ["describe", "cot", str(input_path)], prog_name="sva")

    assert result.exit_code == 0
    assert "Chain-of-Thought" in result.output
    assert "Step 1" in result.output
