"""Tests for the timing dataset generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sva_toolkit.timing.generate.dataset as dataset_module
from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.core.conditions import collect_signals
from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    CutMeaning,
    LaneConstraint,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.generate import GeneratedItem, GenerationError, GenerationRng, generate_dataset
from sva_toolkit.timing.generate.dataset import _bound_feature, _passes_visual_filter, _with_response_overlays
from sva_toolkit.timing.generate.validate_dataset import validate_dataset
from sva_toolkit.timing.generate.model import EventNode, ScenarioComponents
from sva_toolkit.timing.generate.waveform import _constraint_ranges, _verify_waveform_semantics
from sva_toolkit.timing.projection.wavedrom_view import build_wavedrom_view, evaluate_condition
from sva_toolkit.timing.render.svg import render_diagram_svg


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


@pytest.fixture
def cut_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("cut_dataset")
    generate_dataset(
        count=6,
        seed=2,
        out_dir=out_dir,
        max_retries=120,
        cuts_probability=1.0,
    )
    return out_dir


def _read_records(out_dir: Path) -> list[dict]:
    records_path = out_dir / "records.jsonl"
    return [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]


def _first_anchor_tick(document, anchor_name: str) -> int | None:
    anchor = document.anchor_map[anchor_name]
    samples = {signal.name: signal.samples for signal in document.signals}
    if document.ticks is None:
        return None
    for tick in range(document.ticks):
        if evaluate_condition(anchor.condition, samples, tick):
            return tick
    return None


def _anchor_ticks(document) -> dict[str, int]:
    return {
        anchor.name: tick
        for anchor in document.anchors
        if (tick := _first_anchor_tick(document, anchor.name)) is not None
    }


def _bound_allows(bound: TimeBound, delay: int) -> bool:
    if bound.kind == WindowBoundKind.EXACT:
        return bound.min_delay is not None and bound.min_delay.isdigit() and delay == int(bound.min_delay)
    if bound.kind == WindowBoundKind.RANGE:
        lo = int(bound.min_delay) if bound.min_delay and bound.min_delay.isdigit() else 0
        hi = int(bound.max_delay) if bound.max_delay and bound.max_delay.isdigit() else None
        return delay >= lo if hi is None else lo <= delay <= hi
    if bound.kind == WindowBoundKind.UNBOUNDED:
        lo = int(bound.min_delay) if bound.min_delay and bound.min_delay.isdigit() else 0
        return delay >= lo
    return True


def _condition_occurrences(document, anchor_name: str) -> list[int]:
    anchor = document.anchor_map[anchor_name]
    samples = {signal.name: signal.samples for signal in document.signals if signal.samples}
    if document.ticks is None:
        return []
    return [
        tick
        for tick in range(document.ticks)
        if evaluate_condition(anchor.condition, samples, tick)
    ]


def _constraint_text(constraint: LaneConstraint) -> str:
    signal = constraint.signals[0] if len(constraint.signals) == 1 else "{" + ", ".join(constraint.signals) + "}"
    if constraint.relation == "eq":
        head = f"{signal} = {constraint.value}"
    elif constraint.relation == "neq":
        head = f"neq({signal}, {constraint.value})"
    else:
        head = f"{constraint.relation}({signal})"

    if constraint.region == ConstraintRegion.AT:
        return f"show {head} at {constraint.anchor};"
    if constraint.region == ConstraintRegion.IN:
        return f"show {head} in {constraint.window};"
    if constraint.region == ConstraintRegion.BEFORE:
        return f"show {head} before {constraint.anchor};"
    if constraint.region == ConstraintRegion.AFTER:
        return f"show {head} after {constraint.anchor};"
    return f"show {head} from {constraint.start_anchor} until {constraint.end_anchor};"


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


def test_parameterized_bound_coverage_is_reported_for_generation(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=20,
        seed=2024,
        out_dir=tmp_path,
        max_retries=200,
        render_svg=False,
    )
    assert "parameterized" in summary["coverage"]["bound_kind"]


def test_generated_anchor_predicates_are_recoverable(tmp_path: Path) -> None:
    generate_dataset(
        count=20,
        seed=2024,
        out_dir=tmp_path,
        max_retries=200,
        render_svg=False,
    )
    checked = 0
    for record in _read_records(tmp_path):
        document = parse_diagram(record["target"]["canonical_dsl"])
        if document.ticks is None:
            continue
        signal_samples = {signal.name: signal.samples for signal in document.signals}
        for anchor in document.anchors:
            signal_names = collect_signals(anchor.condition)
            if not signal_names or any(not signal_samples.get(name) for name in signal_names):
                continue
            occurrences = [
                tick
                for tick in range(document.ticks)
                if evaluate_condition(anchor.condition, signal_samples, tick)
            ]
            assert occurrences, f"{record['id']} anchor {anchor.name} is not visible in samples"
            checked += 1
    assert checked


def test_generated_windows_have_svg_response_overlays(tmp_path: Path) -> None:
    generate_dataset(count=2, seed=22, out_dir=tmp_path, max_retries=40)
    for record in _read_records(tmp_path):
        canonical = record["target"]["canonical_dsl"]
        document = parse_diagram(canonical)
        assert document.windows
        svg_text = (tmp_path / record["svg_path"]).read_text(encoding="utf-8")
        assert "timing-rule-summary" in svg_text


def test_generated_windows_have_sweeping_svg_response_overlays(tmp_path: Path) -> None:
    generate_dataset(count=10, seed=22, out_dir=tmp_path, max_retries=100)
    for record in _read_records(tmp_path):
        document = parse_diagram(record["target"]["canonical_dsl"])
        svg_text = (tmp_path / record["svg_path"]).read_text(encoding="utf-8")
        assert "timing-rule-summary" in svg_text
        for window in document.windows:
            assert (
                f"p_{window.name}" in svg_text
                or "timing-rule-overlays" in svg_text
                or "timing-hold-fill" in svg_text
            )


def test_join_topology_window_bounds_match_actual_delta(tmp_path: Path) -> None:
    generate_dataset(count=20, seed=8675309, out_dir=tmp_path, max_retries=300)
    checked = 0
    for record in _read_records(tmp_path):
        if record["features"]["topology"] != "join":
            continue
        document = parse_diagram(record["target"]["canonical_dsl"])
        first_ticks = _anchor_ticks(document)
        for window in document.windows:
            start = first_ticks[window.start_anchor]
            end = first_ticks[window.end_anchor]
            assert start < end
            assert _bound_allows(window.bound, end - start), (
                f"{record['id']} {window.name} delta={end - start} bound={window.bound.label}"
            )
            checked += 1
    assert checked


def test_mixed_rendering_keeps_anchor_signal_samples(tmp_path: Path) -> None:
    generate_dataset(count=30, seed=42, out_dir=tmp_path, max_retries=300)
    checked = 0
    for record in _read_records(tmp_path):
        if record["features"]["rendering"] != "mixed":
            continue
        document = parse_diagram(record["target"]["canonical_dsl"])
        signal_map = document.signal_map
        for anchor in document.anchors:
            for signal_name in collect_signals(anchor.condition):
                assert signal_map[signal_name].samples, f"{record['id']} redacted anchor lane {signal_name}"
                checked += 1
    assert checked


def test_anchor_first_occurrence_equals_assigned(tmp_path: Path) -> None:
    generate_dataset(
        count=30,
        seed=99,
        out_dir=tmp_path,
        max_retries=300,
        concrete_ratio=1.0,
        symbolic_ratio=0.0,
        mixed_ratio=0.0,
    )
    checked = 0
    for record in _read_records(tmp_path):
        document = parse_diagram(record["target"]["canonical_dsl"])
        view = build_wavedrom_view(_with_response_overlays(document))
        first_ticks = {
            anchor.name: occurrences[0]
            for anchor in document.anchors
            if (occurrences := _condition_occurrences(document, anchor.name))
        }
        for occurrence in view.anchor_occurrences:
            assert first_ticks[occurrence.anchor_name] <= occurrence.tick
            checked += 1
        for window in document.windows:
            start = first_ticks[window.start_anchor]
            end = first_ticks[window.end_anchor]
            assert start < end
            assert _bound_allows(window.bound, end - start)
        for span in view.response_spans:
            window_name = span.name.removeprefix("p_")
            window = document.window_map[window_name]
            assert span.trigger_tick == first_ticks[window.start_anchor]
            assert span.response_tick == first_ticks[window.end_anchor]
    assert checked


def test_validate_dataset_strict_zero_failures(tmp_path: Path) -> None:
    generate_dataset(
        count=30,
        seed=2024,
        out_dir=tmp_path,
        max_retries=400,
        cuts_probability=0.55,
        coverage_target=2,
    )
    result = validate_dataset(tmp_path, strict=True)
    assert result["records_failed"] == 0


def test_generates_eq_or_change_or_stable_predicate(tmp_path: Path) -> None:
    generate_dataset(count=20, seed=2024, out_dir=tmp_path, max_retries=80)
    predicates = {
        predicate
        for record in _read_records(tmp_path)
        for predicate in record["features"]["predicates"]
    }
    assert predicates & {"eq", "stable", "change", "fall"}


def test_generates_at_in_or_after_constraint_region(tmp_path: Path) -> None:
    generate_dataset(count=20, seed=4242, out_dir=tmp_path, max_retries=120)
    regions = {
        region
        for record in _read_records(tmp_path)
        for region in record["features"]["constraint_regions"]
    }
    assert regions & {"at", "in", "after"}


def test_between_window_cut_generated(cut_dataset: Path) -> None:
    assert any("between" in record["features"]["cuts"] for record in _read_records(cut_dataset))


def test_compressed_or_lookback_cut_generated(cut_dataset: Path) -> None:
    meanings = {
        cut.meaning
        for record in _read_records(cut_dataset)
        for cut in parse_diagram(record["target"]["canonical_dsl"]).cuts
    }
    assert meanings & {CutMeaning.SYMBOLIC_GAP, CutMeaning.LOOKBACK}


def test_unlabeled_cut_generated(cut_dataset: Path) -> None:
    assert any(
        cut.label is None
        for record in _read_records(cut_dataset)
        for cut in parse_diagram(record["target"]["canonical_dsl"]).cuts
    )


def test_generates_eq_or_neq_constraint_relation(tmp_path: Path) -> None:
    generate_dataset(count=30, seed=99, out_dir=tmp_path, max_retries=120)
    saw_relation = False
    saw_rendered_text = False
    for record in _read_records(tmp_path):
        canonical = record["target"]["canonical_dsl"]
        parsed = parse_diagram(canonical)
        saw_relation = saw_relation or any(
            constraint.relation in {"eq", "neq"} for constraint in parsed.lane_constraints
        )
        saw_rendered_text = saw_rendered_text or "eq(" in canonical or "neq(" in canonical or " = " in canonical
    assert saw_relation
    assert saw_rendered_text


def test_constraint_regions_and_relations_round_trip_in_dsl(tmp_path: Path) -> None:
    generate_dataset(
        count=20,
        seed=311,
        out_dir=tmp_path,
        max_retries=200,
        render_svg=False,
    )
    seen_regions: set[str] = set()
    seen_relations: set[str] = set()
    saw_bus_stable = False
    for record in _read_records(tmp_path):
        canonical = record["target"]["canonical_dsl"]
        document = parse_diagram(canonical)
        emitted_lines = {line.strip() for line in canonical.splitlines()}
        for constraint in document.lane_constraints:
            assert _constraint_text(constraint) in emitted_lines
            seen_regions.add(constraint.region.value)
            seen_relations.add(constraint.relation)
            saw_bus_stable = saw_bus_stable or (
                constraint.relation == "stable"
                and any(document.signal_map[name].kind.value == "bus" for name in constraint.signals)
            )
    assert {"at", "in", "after"}.issubset(seen_regions)
    assert {"eq", "neq", "fall", "change"}.issubset(seen_relations)
    assert saw_bus_stable


def test_compact_flavor_short_names(tmp_path: Path) -> None:
    generate_dataset(
        count=8,
        seed=7,
        out_dir=tmp_path,
        max_retries=80,
        holdout_flavor=None,
    )
    compact_records = [
        record for record in _read_records(tmp_path) if record["features"]["flavor"] == "compact"
    ]
    assert compact_records
    for record in compact_records:
        document = parse_diagram(record["target"]["canonical_dsl"])
        assert all(len(signal.name) <= 6 for signal in document.signals)


def test_burst_middle_beat_or_response_after_last(tmp_path: Path) -> None:
    generate_dataset(count=15, seed=21, out_dir=tmp_path, max_retries=200)
    burst_records = [
        record for record in _read_records(tmp_path) if record["features"]["topology"] == "burst"
    ]
    assert burst_records
    assert any(
        ("middle_beat" in record["target"]["canonical_dsl"])
        or ("response_after_last" in record["target"]["canonical_dsl"])
        for record in burst_records
    )


def test_backpressure_stall_cycles_visible(tmp_path: Path) -> None:
    generate_dataset(
        count=15,
        seed=11,
        out_dir=tmp_path,
        max_retries=200,
        concrete_ratio=1.0,
        symbolic_ratio=0.0,
        mixed_ratio=0.0,
    )
    saw_backpressure = False
    for record in _read_records(tmp_path):
        if record["features"]["topology"] != "backpressure":
            continue
        saw_backpressure = True
        document = parse_diagram(record["target"]["canonical_dsl"])
        svg = render_diagram_svg(_with_response_overlays(document))
        assert "<svg" in svg
        start = _first_anchor_tick(document, "valid_rise")
        end = _first_anchor_tick(document, "handshake")
        assert start is not None and end is not None and start < end
        signal_map = document.signal_map
        stall_constraints = [
            constraint
            for constraint in document.lane_constraints
            if constraint.relation == "low"
            and constraint.region == ConstraintRegion.FROM_UNTIL
            and constraint.start_anchor == "valid_rise"
            and constraint.end_anchor == "handshake"
        ]
        assert stall_constraints
        assert any(
            any(sample == "0" for sample in signal_map[constraint.signals[0]].samples[start:end])
            for constraint in stall_constraints
        )
    assert saw_backpressure


def test_clock_edge_can_be_negedge(tmp_path: Path) -> None:
    generate_dataset(count=20, seed=8, out_dir=tmp_path, max_retries=80)
    assert any(
        parse_diagram(record["target"]["canonical_dsl"]).clocking.edge == "negedge"
        for record in _read_records(tmp_path)
    )


def test_bus_distractor_outside_stable_range(tmp_path: Path) -> None:
    generate_dataset(
        count=10,
        seed=33,
        out_dir=tmp_path,
        max_retries=120,
        concrete_ratio=1.0,
        symbolic_ratio=0.0,
        mixed_ratio=0.0,
    )
    checked_lane = False
    for record in _read_records(tmp_path):
        document = parse_diagram(record["target"]["canonical_dsl"])
        if document.ticks is None:
            continue
        anchor_ticks = _anchor_ticks(document)
        bus_names = {signal.name for signal in document.signals if signal.kind.value == "bus" and signal.samples}
        for signal_name in bus_names:
            constrained_ticks: set[int] = set()
            for constraint in document.lane_constraints:
                if signal_name not in constraint.signals or constraint.relation not in {"stable", "eq"}:
                    continue
                for start, end in _constraint_ranges(constraint, anchor_ticks, document, document.ticks):
                    constrained_ticks.update(range(start, end + 1))
            outside_ticks = [tick for tick in range(document.ticks) if tick not in constrained_ticks]
            if len(outside_ticks) < 2 or not constrained_ticks:
                continue
            checked_lane = True
            samples = document.signal_map[signal_name].samples
            assert any(samples[tick] != "x" for tick in outside_ticks)
    assert checked_lane


def test_parameterized_bound_feature_is_preserved() -> None:
    window = TimeWindow(
        name="latency",
        start_anchor="start",
        end_anchor="done",
        bound=TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay="MAX_LAT"),
    )
    assert _bound_feature(window) == "parameterized"


def test_cut_coverage_tracks_multiple_placements(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=25,
        seed=77,
        out_dir=tmp_path,
        max_retries=300,
        cuts_probability=0.8,
        render_svg=False,
    )
    placements = set(summary["coverage"]["cut"]) & {"before", "after", "between"}
    assert len(placements) >= 2


def test_coverage_target_zero_and_unmet_are_reported(tmp_path: Path) -> None:
    zero_summary = generate_dataset(
        count=3,
        seed=1,
        out_dir=tmp_path / "zero",
        max_retries=80,
        coverage_target=0,
        render_svg=False,
    )
    assert zero_summary["count"] == 3

    try:
        target_summary = generate_dataset(
            count=3,
            seed=1,
            out_dir=tmp_path / "target",
            max_retries=120,
            coverage_target=2,
            render_svg=False,
        )
    except GenerationError as exc:
        assert "coverage" in str(exc).lower() or exc.reason == "coverage_bucket_required"
    else:
        assert target_summary.get("coverage_target_unmet")


def test_rendering_quotas_cover_small_count(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=10,
        seed=3,
        out_dir=tmp_path,
        max_retries=200,
        concrete_ratio=0.5,
        symbolic_ratio=0.3,
        mixed_ratio=0.2,
        render_svg=False,
    )
    assert {"concrete", "symbolic", "mixed"}.issubset(summary["coverage"]["rendering"])


def test_mixed_records_keep_sampled_and_symbolic_lanes(tmp_path: Path) -> None:
    generate_dataset(
        count=20,
        seed=3,
        out_dir=tmp_path,
        max_retries=200,
        concrete_ratio=0.2,
        symbolic_ratio=0.1,
        mixed_ratio=0.7,
        render_svg=False,
    )
    mixed_seen = False
    for record in _read_records(tmp_path):
        if record["features"]["rendering"] != "mixed":
            continue
        mixed_seen = True
        document = parse_diagram(record["target"]["canonical_dsl"])
        if document.ticks is None:
            continue
        sampled = [signal for signal in document.signals if signal.samples]
        symbolic = [signal for signal in document.signals if not signal.samples]
        assert sampled
        assert symbolic
    assert mixed_seen


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
    with pytest.raises(GenerationError, match="first match"):
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


def test_coverage_target_creates_distinct_predicate_values(tmp_path: Path) -> None:
    summary = generate_dataset(
        count=6,
        seed=2026,
        out_dir=tmp_path,
        max_retries=200,
        coverage_target=2,
        render_svg=False,
    )
    assert len(summary["coverage"].get("predicate", {})) >= 2


def test_visual_filter_rejects_tiny_svg() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40">'
        '<g class="timing-test"><text x="4" y="12">tiny</text></g>'
        f"<!-- {'x' * 220} --></svg>"
    )
    assert not _passes_visual_filter(svg)


def test_duplicate_svg_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120" viewBox="0 0 200 120">'
        '<g class="timing-test"><text x="20" y="30">same</text></g>'
        f"<!-- {'x' * 220} --></svg>"
    )
    features = {
        "topology": "single_response",
        "flavor": "generic",
        "idioms": ["response"],
        "ticks": 6,
        "lane_count": 3,
        "anchor_count": 2,
        "window_count": 1,
        "has_bus": False,
        "has_params": False,
        "bound_kinds": ["exact"],
        "constraint_regions": [],
        "predicates": ["rise"],
        "lane_kind": "bit_only",
        "rendering": "concrete",
        "naming": "snake_case",
        "cut": "none",
        "cuts": ["none"],
    }

    def fake_generate_one(spec, item_rng, output_path, *, render_svg: bool, render_png: bool):
        return GeneratedItem(
            id=spec.item_id,
            seed=spec.seed,
            canonical_dsl=f"diagram d_{spec.item_id} {{\n  clock posedge clk;\n}}",
            svg_text=svg,
            features=dict(features),
        )

    monkeypatch.setattr(dataset_module, "_generate_one", fake_generate_one)
    with pytest.raises(GenerationError, match="duplicate_svg"):
        generate_dataset(count=2, seed=1, out_dir=tmp_path, max_retries=2)


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


def test_holdout_bound_excludes_parameterized(tmp_path: Path) -> None:
    generate_dataset(
        count=8,
        seed=6,
        out_dir=tmp_path,
        max_retries=100,
        holdout_bound="parameterized",
    )
    for record in _read_records(tmp_path):
        assert "parameterized" not in record["features"]["bound_kinds"]


def test_holdout_rendering_excludes_symbolic(tmp_path: Path) -> None:
    generate_dataset(
        count=5,
        seed=7,
        out_dir=tmp_path,
        max_retries=80,
        concrete_ratio=0.0,
        symbolic_ratio=1.0,
        mixed_ratio=0.0,
        holdout_rendering="symbolic",
        render_svg=False,
    )
    for record in _read_records(tmp_path):
        assert record["features"]["rendering"] != "symbolic"


def test_streaming_jsonl_writes_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_update = dataset_module.CoverageTracker.update
    observed_line_counts: list[int] = []

    def update_and_peek(self, features):
        lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
        assert lines
        json.loads(lines[-1])
        observed_line_counts.append(len(lines))
        return original_update(self, features)

    monkeypatch.setattr(dataset_module.CoverageTracker, "update", update_and_peek)
    generate_dataset(count=2, seed=17, out_dir=tmp_path, max_retries=80)
    assert observed_line_counts[0] == 1
    assert observed_line_counts[-1] == 2


def test_rejection_reason_codes_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120" viewBox="0 0 200 120">'
        '<g class="timing-test"><text x="20" y="30">ok</text></g>'
        f"<!-- {'x' * 220} --></svg>"
    )
    features = {
        "topology": "single_response",
        "flavor": "generic",
        "idioms": ["response"],
        "ticks": 6,
        "lane_count": 3,
        "anchor_count": 2,
        "window_count": 1,
        "has_bus": True,
        "has_params": False,
        "bound_kinds": ["exact"],
        "constraint_regions": ["from_until"],
        "predicates": ["eq"],
        "lane_kind": "mixed",
        "rendering": "concrete",
        "naming": "snake_case",
        "cut": "none",
        "cuts": ["none"],
    }
    calls = {"count": 0}

    def fake_generate_one(spec, item_rng, output_path, *, render_svg: bool, render_png: bool):
        calls["count"] += 1
        if calls["count"] == 1:
            raise GenerationError("contradictory bus value constraint", reason="bus_value_conflict")
        return GeneratedItem(
            id=spec.item_id,
            seed=spec.seed,
            canonical_dsl=f"diagram d_{spec.item_id} {{\n  clock posedge clk;\n}}",
            svg_text=svg,
            features=dict(features),
        )

    monkeypatch.setattr(dataset_module, "_generate_one", fake_generate_one)
    summary = generate_dataset(count=1, seed=1, out_dir=tmp_path, max_retries=2)
    assert summary["rejections"]["bus_value_conflict"] == 1
    assert not any(reason.startswith("generation_error:") for reason in summary["rejections"])


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


def test_cli_rejects_min_ticks_greater_than_max_ticks(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from sva_toolkit.cli.main import main

    result = CliRunner().invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "1",
            "--seed",
            "1",
            "--out",
            str(tmp_path),
            "--min-ticks",
            "9",
            "--max-ticks",
            "6",
        ],
    )
    assert result.exit_code != 0
    assert "max_ticks" in result.output


def test_cli_rejects_min_lanes_greater_than_max_lanes(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from sva_toolkit.cli.main import main

    result = CliRunner().invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "1",
            "--seed",
            "1",
            "--out",
            str(tmp_path),
            "--min-lanes",
            "8",
            "--max-lanes",
            "4",
        ],
    )
    assert result.exit_code != 0
    assert "max_lanes" in result.output


def test_cli_rejects_negative_coverage_target(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from sva_toolkit.cli.main import main

    result = CliRunner().invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "1",
            "--seed",
            "1",
            "--out",
            str(tmp_path),
            "--coverage-target",
            "-1",
        ],
    )
    assert result.exit_code != 0
    assert "coverage_target" in result.output


def test_cli_all_zero_rendering_ratios_fall_back_to_concrete(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from sva_toolkit.cli.main import main

    result = CliRunner().invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "2",
            "--seed",
            "31",
            "--out",
            str(tmp_path),
            "--format",
            "none",
            "--concrete-ratio",
            "0",
            "--symbolic-ratio",
            "0",
            "--mixed-ratio",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert {record["features"]["rendering"] for record in _read_records(tmp_path)} == {"concrete"}


def test_cli_unknown_holdout_topology_is_clear(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from sva_toolkit.cli.main import main

    result = CliRunner().invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "1",
            "--seed",
            "1",
            "--out",
            str(tmp_path),
            "--holdout-topology",
            "does_not_exist",
        ],
    )
    assert result.exit_code != 0
    assert "unknown holdout topology" in result.output


def test_cli_unknown_holdout_flavor_is_clear(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from sva_toolkit.cli.main import main

    result = CliRunner().invoke(
        main,
        [
            "timing",
            "generate-dataset",
            "--count",
            "1",
            "--seed",
            "1",
            "--out",
            str(tmp_path),
            "--holdout-flavor",
            "nope",
        ],
    )
    assert result.exit_code != 0
    assert "unknown holdout flavor" in result.output
