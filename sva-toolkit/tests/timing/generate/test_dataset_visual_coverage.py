from __future__ import annotations

import json
from pathlib import Path

from sva_toolkit.timing.generate import generate_dataset


def test_visual_coverage_is_reported(tmp_path: Path) -> None:
    generate_dataset(count=3, seed=1, out_dir=tmp_path, render_profile_set="train_v2", max_retries=160)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    visual = summary["coverage"]["visual"]

    assert any(count >= 1 for count in visual["renderer_id"].values())
    assert visual["annotation_policy"]
