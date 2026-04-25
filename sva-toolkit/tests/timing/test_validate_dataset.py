"""Tests for generated timing dataset utility validation."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sva_toolkit.cli.main import main
from sva_toolkit.timing.generate import generate_dataset
from sva_toolkit.timing.generate.validate_dataset import validate_dataset


def _read_records(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "records.jsonl").read_text().splitlines() if line]


def test_validate_generated_dataset_passes(tmp_path: Path) -> None:
    generate_dataset(count=5, seed=123, out_dir=tmp_path, max_retries=120)
    result = validate_dataset(tmp_path)
    assert result["records_total"] == 5
    assert result["records_failed"] == 0


def test_validate_dataset_reports_corrupt_dsl_file(tmp_path: Path) -> None:
    generate_dataset(count=3, seed=124, out_dir=tmp_path, max_retries=120)
    record = _read_records(tmp_path)[0]
    (tmp_path / record["dsl_path"]).write_text("diagram corrupt {\n", encoding="utf-8")

    result = validate_dataset(tmp_path)

    assert result["records_failed"] >= 1
    assert any(failure["reason"] == "dsl_file_mismatch" for failure in result["failures"])


def test_validate_dataset_reports_corrupt_svg_file(tmp_path: Path) -> None:
    generate_dataset(count=3, seed=125, out_dir=tmp_path, max_retries=120)
    record = _read_records(tmp_path)[0]
    (tmp_path / record["svg_path"]).write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
        encoding="utf-8",
    )

    result = validate_dataset(tmp_path)

    assert result["records_failed"] >= 1
    assert any(failure["reason"] == "visual_recoverability" for failure in result["failures"])


def test_validate_dataset_coverage_summary_has_expected_buckets(tmp_path: Path) -> None:
    generate_dataset(
        count=20,
        seed=2024,
        out_dir=tmp_path,
        max_retries=300,
        cuts_probability=0.8,
    )

    result = validate_dataset(tmp_path)

    assert result["records_failed"] == 0
    assert result["coverage_summary"]["predicate"] >= 5
    assert result["coverage_summary"]["region"] >= 4
    assert result["coverage_summary"]["cut"] >= 2


def test_validate_dataset_cli_outputs_summary(tmp_path: Path) -> None:
    generate_dataset(count=5, seed=126, out_dir=tmp_path, max_retries=120)

    result = CliRunner().invoke(
        main,
        [
            "timing",
            "validate-dataset",
            "--dataset",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "records_total" in result.output
