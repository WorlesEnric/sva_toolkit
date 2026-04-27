from __future__ import annotations

import json
from pathlib import Path

from sva_toolkit.timing.generate import generate_dataset


def _records(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_style_holdout_excludes_profile(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=3,
        seed=0,
        out_dir=tmp_path,
        render_profile_set="train_v2",
        style_holdout="native_random",
        max_retries=220,
    )

    for record in _records(tmp_path):
        assert record["profile"] != "native-random"
        assert record["style_family"] != "native_random"
    assert "holdout_style" not in summary["rejection_reasons"] or summary["rejection_reasons"]["holdout_style"] > 0
