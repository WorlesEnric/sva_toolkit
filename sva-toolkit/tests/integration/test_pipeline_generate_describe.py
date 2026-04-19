from __future__ import annotations

import random

import pytest

from sva_toolkit.cli.main import main

from .conftest import extract_property_blocks


pytestmark = pytest.mark.integration


def test_generate_then_describe_pipeline_produces_human_readable_output(runner) -> None:
    random.seed(1)
    generate_result = runner.invoke(main, ["generate", "--count", "1"], prog_name="sva")

    assert generate_result.exit_code == 0
    blocks = extract_property_blocks(generate_result.output)
    assert len(blocks) == 1

    describe_result = runner.invoke(main, ["describe", "svad", blocks[0]], prog_name="sva")

    assert describe_result.exit_code == 0
    assert "Relevant Signals" in describe_result.output
    assert "Expected Results" in describe_result.output
