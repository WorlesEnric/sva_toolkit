from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sva_toolkit.cli.main import main


def test_cli_generate_dataset_v2_artifacts(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "3",
            "--seed",
            "0",
            "--out",
            str(tmp_path),
            "--render-profile-set",
            "train_v2",
            "--emit-render-specs",
            "--audit-strict",
            "--format",
            "png",
            "--max-retries",
            "220",
        ],
        prog_name="sva",
    )

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 3
    for record in records:
        assert (tmp_path / record["image_path"]).is_file()
        assert (tmp_path / record["render_spec_path"]).is_file()
        assert (tmp_path / record["audits_path"]).is_file()
