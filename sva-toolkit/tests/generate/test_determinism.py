from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

import pytest
from click.testing import CliRunner

from sva_toolkit.cli.main import main
from sva_toolkit.generate import GenerationRng, SVASynthesizer, StratifiedGenerator
from sva_toolkit.generate.synthesizer import ValidationResult


@pytest.fixture
def render_module() -> Callable[[int], str]:
    def _render(seed: int) -> str:
        synthesizer = SVASynthesizer(
            signals=["req", "ack", "gnt"],
            max_depth=2,
            rng=GenerationRng(seed=seed),
        )
        module_code, _properties = synthesizer.generate_module("demo_sva", 4)
        return module_code

    return _render


def test_synthesizer_generates_identical_module_text_for_same_seed(render_module: Callable[[int], str]) -> None:
    assert render_module(42) == render_module(42)


def test_synthesizer_generates_different_module_text_for_different_seeds(
    render_module: Callable[[int], str],
) -> None:
    assert render_module(42) != render_module(43)


def test_stratified_generator_is_deterministic_for_same_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_valid(_self, _prop) -> ValidationResult:
        return ValidationResult(is_valid=True)

    monkeypatch.setattr(SVASynthesizer, "validate_single_property", always_valid)

    first = StratifiedGenerator(
        signals=["req", "ack", "gnt"],
        samples_per_construct=1,
        rng=GenerationRng(seed=42),
    ).generate_stratified_dataset()
    second = StratifiedGenerator(
        signals=["req", "ack", "gnt"],
        samples_per_construct=1,
        rng=GenerationRng(seed=42),
    ).generate_stratified_dataset()

    assert [prop.property_block for prop in first] == [prop.property_block for prop in second]


def test_generate_package_avoids_module_level_random_imports() -> None:
    generate_dir = Path(__file__).resolve().parents[2] / "src" / "sva_toolkit" / "generate"

    for path in generate_dir.rglob("*.py"):
        contents = path.read_text(encoding="utf-8")
        assert re.search(r"^import random", contents, re.MULTILINE) is None, path


def test_cli_generate_is_deterministic_with_explicit_seed() -> None:
    runner = CliRunner()

    first = runner.invoke(main, ["generate", "--seed", "42", "--count", "3"], prog_name="sva")
    second = runner.invoke(main, ["generate", "--seed", "42", "--count", "3"], prog_name="sva")

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert first.output == second.output
    assert first.stderr == ""
    assert second.stderr == ""


def test_cli_generate_reports_chosen_seed_on_stderr() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["generate", "--count", "1"], prog_name="sva")

    assert result.exit_code == 0
    assert "Using generation seed:" in result.stderr
    assert "property p_gen_0;" in result.output
