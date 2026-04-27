from __future__ import annotations

import json
from pathlib import Path

from sva_toolkit.timing.generate import generate_dataset


def _records(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_generate_dataset_uses_render2_pipeline(tmp_path: Path) -> None:
    generate_dataset(count=4, seed=0, out_dir=tmp_path, render_profile_set="train_v2", max_retries=200)

    records = _records(tmp_path)
    assert len(records) == 4
    assert len({record["renderer_id"] for record in records}) >= 2
    for record in records:
        assert (tmp_path / record["render_spec_path"]).is_file()
        assert (tmp_path / record["audits_path"]).is_file()
        assert (tmp_path / record["image_path"]).is_file()
