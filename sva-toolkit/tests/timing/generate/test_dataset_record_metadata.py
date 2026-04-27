from __future__ import annotations

import json
from pathlib import Path

from sva_toolkit.timing.generate import generate_dataset


REQUIRED_RECORD_FIELDS = {
    "id",
    "seed",
    "split",
    "dsl_path",
    "semantic_dsl_path",
    "image_path",
    "svg_path",
    "render_spec_path",
    "audits_path",
    "renderer_id",
    "profile",
    "style_family",
    "annotation_policy",
    "degradation_profile",
    "target",
    "features",
    "visual_features",
    "visibility",
    "difficulty",
    "audit_status",
}


def test_record_metadata_schema_is_complete(tmp_path: Path) -> None:
    generate_dataset(count=2, seed=2, out_dir=tmp_path, render_profile_set="train_v2", max_retries=160)

    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    for record in records:
        assert REQUIRED_RECORD_FIELDS <= set(record)
        assert record["target"]["policy"] == "visual"
        assert (tmp_path / record["dsl_path"]).is_file()
        assert (tmp_path / record["semantic_dsl_path"]).is_file()
        assert (tmp_path / record["image_path"]).is_file()
        assert (tmp_path / record["render_spec_path"]).is_file()
        assert (tmp_path / record["audits_path"]).is_file()
