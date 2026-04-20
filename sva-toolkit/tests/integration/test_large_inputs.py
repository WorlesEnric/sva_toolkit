from __future__ import annotations

import time

import pytest

from sva_toolkit.cli.exit_codes import ExitCode
from sva_toolkit.cli.main import main
from sva_toolkit.sva.lexer import tokenize
from sva_toolkit.sva.parser import parse_property_text


pytestmark = pytest.mark.integration


def test_large_property_parses_under_two_seconds(long_property_text: str) -> None:
    # R16 regression: large properties should continue to parse within a modest smoke-test latency budget.
    token_count = len(tokenize(long_property_text)) - 1

    started = time.perf_counter()
    spec = parse_property_text(long_property_text)
    elapsed = time.perf_counter() - started

    assert token_count > 1000
    assert elapsed < 2.0, elapsed
    assert spec.clocking is not None
    assert spec.clocking.signal.name == "clk"


@pytest.mark.parametrize("command", ["parse", "describe"])
def test_large_property_cli_smoke_remains_green(runner, long_property_text: str, command: str, tmp_path) -> None:
    input_path = tmp_path / "long_property.sv"
    input_path.write_text(long_property_text, encoding="utf-8")

    if command == "parse":
        args = ["parse", str(input_path), "--format", "json"]
        expected = '"kind": "property"'
    else:
        args = ["describe", "svad", str(input_path)]
        expected = "Relevant Signals"

    result = runner.invoke(main, args, prog_name="sva")

    assert result.exit_code == ExitCode.SUCCESS
    assert expected in result.output
