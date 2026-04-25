"""Tests for the timing dataset generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    LaneConstraint,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.generate import GenerationError, GenerationRng, generate_dataset
from sva_toolkit.timing.generate.dataset import _bound_feature
from sva_toolkit.timing.generate.model import EventNode, ScenarioComponents
from sva_toolkit.timing.generate.waveform import _verify_waveform_semantics


@pytest.fixture
def small_dataset(tmp_path: Path) -> dict:
    return generate_dataset(
        count=5,
        seed=12345,
        out_dir=tmp_path,
        max_retries=30,
        cuts_probability=0.5,
        distractor_probability=0.3,
        concrete_ratio=0.8,
        symbolic_ratio=0.1,
        mixed_ratio=0.1,
    )


def _read_records(out_dir: Path) -> list[dict]:
    records_path = out_dir / "records.jsonl"
    return [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]


def test_generates_requested_count(tmp_path: Path) -> None:
    summary = generate_dataset(count=4, seed=1, out_dir=tmp_path, max_retries=20)
    assert summary["count"] == 4
    records = _read_records(tmp_path)
    assert len(records) == 4
    for record in records:
        assert (tmp_path / record["dsl_path"]).is_file()
        assert "svg_path" in record
        assert (tmp_path / record["svg_path"]).is_file()


def test_tick_bounds_are_enforced(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=4,
        seed=101,
        out_dir=tmp_path,
        min_ticks=6,
        max_ticks=6,
        max_retries=80,
    )
    assert summary["count"] == 4
    for record in _read_records(tmp_path):
        assert record["features"]["ticks"] == 6


def test_lane_bounds_are_enforced(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=4,
        seed=102,
        out_dir=tmp_path,
        min_lanes=5,
        max_lanes=6,
        max_retries=80,
    )
    assert summary["count"] == 4
    for record in _read_records(tmp_path):
        assert 5 <= record["features"]["lane_count"] <= 6


def test_invalid_size_bounds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_ticks"):
        generate_dataset(count=1, seed=1, out_dir=tmp_path / "ticks", min_ticks=8, max_ticks=6)
    with pytest.raises(ValueError, match="max_lanes"):
        generate_dataset(count=1, seed=1, out_dir=tmp_path / "lanes", min_lanes=5, max_lanes=4)


def test_canonical_dsl_round_trips(tmp_path: Path) -> None:
    generate_dataset(count=3, seed=2, out_dir=tmp_path, max_retries=20)
    for record in _read_records(tmp_path):
        canonical = record["target"]["canonical_dsl"]
        parsed = parse_diagram(canonical)
        assert emit_timing_dsl(parsed) == canonical


def test_generated_windows_have_svg_response_overlays(tmp_path: Path) -> None:
    generate_dataset(count=2, seed=22, out_dir=tmp_path, max_retries=40)
    for record in _read_records(tmp_path):
        canonical = record["target"]["canonical_dsl"]
        document = parse_diagram(canonical)
        assert document.windows
        svg_text = (tmp_path / record["svg_path"]).read_text(encoding="utf-8")
        assert "timing-rule-summary" in svg_text


def test_parameterized_bound_feature_is_preserved() -> None:
    window = TimeWindow(
        name="latency",
        start_anchor="start",
        end_anchor="done",
        bound=TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay="MAX_LAT"),
    )
    assert _bound_feature(window) == "parameterized"


def test_waveform_semantic_verifier_rejects_anchor_mismatch() -> None:
    components = ScenarioComponents(
        name="bad_anchor",
        clock_signal="clk",
        anchor_node_map={
            "start": EventNode(
                id="start",
                role="trigger",
                predicate_kind="rise",
                primary_signal="req",
            )
        },
    )
    with pytest.raises(GenerationError, match="predicate does not hold"):
        _verify_waveform_semantics(
            components,
            {"start": 1},
            {"req": ("0", "0", "1")},
            total_ticks=3,
        )


def test_waveform_semantic_verifier_rejects_constraint_mismatch() -> None:
    components = ScenarioComponents(
        name="bad_constraint",
        clock_signal="clk",
        anchor_node_map={
            "start": EventNode(
                id="start",
                role="trigger",
                predicate_kind="high",
                primary_signal="valid",
            ),
            "done": EventNode(
                id="done",
                role="response",
                predicate_kind="high",
                primary_signal="ready",
            ),
        },
        lane_constraints=[
            LaneConstraint(
                name="hold_valid",
                signals=("valid",),
                relation="high",
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor="start",
                end_anchor="done",
            )
        ],
    )
    with pytest.raises(GenerationError, match="hold_valid"):
        _verify_waveform_semantics(
            components,
            {"start": 1, "done": 2},
            {"valid": ("0", "1", "0"), "ready": ("0", "0", "1")},
            total_ticks=3,
        )


def test_bit_samples_are_legal(tmp_path: Path) -> None:
    generate_dataset(count=4, seed=3, out_dir=tmp_path, max_retries=30, cuts_probability=0.4)
    for record in _read_records(tmp_path):
        canonical = record["target"]["canonical_dsl"]
        document = parse_diagram(canonical)
        for signal in document.signals:
            if signal.kind.value == "bit":
                for sample in signal.samples:
                    assert sample.lower() in {"0", "1", "x", "z"}, (
                        f"illegal bit sample {sample!r} in lane {signal.name}"
                    )
            if signal.samples and document.ticks is not None:
                assert len(signal.samples) == document.ticks


def test_svg_is_non_empty(small_dataset: dict, tmp_path: Path) -> None:
    for record in _read_records(tmp_path):
        svg_text = (tmp_path / record["svg_path"]).read_text(encoding="utf-8")
        assert svg_text.startswith("<?xml") or svg_text.lstrip().startswith("<svg")
        assert len(svg_text) > 200


def test_coverage_buckets_populated(small_dataset: dict) -> None:
    coverage = small_dataset["coverage"]
    assert coverage.get("topology"), "topology coverage must be non-empty"
    assert coverage.get("rendering"), "rendering coverage must be non-empty"
    assert coverage.get("predicate"), "predicate coverage must be non-empty"


def test_determinism_with_same_seed(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    summary_a = generate_dataset(count=3, seed=99, out_dir=out_a, max_retries=20)
    summary_b = generate_dataset(count=3, seed=99, out_dir=out_b, max_retries=20)
    records_a = _read_records(out_a)
    records_b = _read_records(out_b)
    assert len(records_a) == len(records_b) == 3
    for ra, rb in zip(records_a, records_b):
        assert ra["target"]["canonical_dsl"] == rb["target"]["canonical_dsl"]
        assert ra["features"] == rb["features"]
    assert summary_a["coverage"] == summary_b["coverage"]


def test_holdout_topology_is_excluded(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=8,
        seed=4,
        out_dir=tmp_path,
        max_retries=30,
        holdout_topology="burst",
    )
    for record in _read_records(tmp_path):
        assert record["features"]["topology"] != "burst"
    assert "burst" not in summary["coverage"].get("topology", {})


def test_holdout_flavor_is_excluded(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=8,
        seed=5,
        out_dir=tmp_path,
        max_retries=30,
        holdout_flavor="axi_like",
    )
    for record in _read_records(tmp_path):
        assert record["features"]["flavor"] != "axi_like"


def test_generation_rng_derives_stable_children() -> None:
    parent_a = GenerationRng(seed=11)
    parent_b = GenerationRng(seed=11)
    child_a = parent_a.derive_child("item:0")
    child_b = parent_b.derive_child("item:0")
    assert [child_a.random() for _ in range(5)] == [child_b.random() for _ in range(5)]


def test_cli_invocation(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from sva_toolkit.cli.main import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "2",
            "--seed",
            "21",
            "--out",
            str(tmp_path),
            "--format",
            "svg",
        ],
    )
    assert result.exit_code == 0, result.output
    records_path = tmp_path / "records.jsonl"
    assert records_path.is_file()
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    assert len(records) == 2
