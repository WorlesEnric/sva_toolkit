from __future__ import annotations

import random

import pytest

from sva_toolkit.cli.main import main
from sva_toolkit.sva.parser import parse_property_text

from .conftest import extract_property_blocks


pytestmark = pytest.mark.integration


def test_generate_command_emits_parseable_property_blocks(runner) -> None:
    random.seed(1)
    result = runner.invoke(main, ["generate", "--count", "2"], prog_name="sva")

    assert result.exit_code == 0
    assert "module generated_sva" in result.output

    blocks = extract_property_blocks(result.output)
    assert len(blocks) == 2
    for block in blocks:
        parsed = parse_property_text(block)
        assert parsed.name is not None
