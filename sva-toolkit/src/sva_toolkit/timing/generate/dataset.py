"""Top-level orchestration for the timing diagram dataset generator."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.core.conditions import Predicate, collect_signals
from sva_toolkit.timing.core.scenario import (
    Anchor,
    ClockingSpec,
    LaneConstraint,
    PropertyOverlay,
    ScenarioDocument,
    SignalKind,
    TimeWindow,
)
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.generate.coverage import (
    LANE_COUNT_BUCKETS,
    TICK_COUNT_BUCKETS,
    CoverageTracker,
    bucket_range,
)
from sva_toolkit.timing.generate.idioms import apply_idioms
from sva_toolkit.timing.generate.model import (
    GeneratedItem,
    GenerationError,
    GenerationSpec,
    ScenarioComponents,
)
from sva_toolkit.timing.generate.names import FLAVORS
from sva_toolkit.timing.generate.topology import TOPOLOGIES, build_topology
from sva_toolkit.timing.generate.waveform import (
    assign_ticks,
    attach_samples,
    synthesize_waveforms,
)
from sva_toolkit.timing.render.svg import render_diagram_svg


def _probe_png_renderer() -> None:
    try:
        import cairosvg
    except ImportError as exc:
        raise GenerationError(
            "PNG output requested but cairosvg is not installed. "
            "Install sva-toolkit[timing-render].",
            reason="png_renderer_unavailable",
        ) from exc
    except OSError as exc:
        raise GenerationError(
            "PNG output requested but cairosvg cannot load its native dependencies. "
            "Install libcairo on the system (e.g., 'brew install cairo' on macOS, "
            "'apt install libcairo2' on Debian/Ubuntu). "
            "Underlying error: see traceback above.",
            reason="png_renderer_unavailable",
        ) from exc
    try:
        cairosvg.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>')
    except Exception as exc:
        raise GenerationError(
            "PNG output requested but cairosvg cannot rasterize on this system. "
            "Make sure libcairo is available. "
            f"Underlying error: {exc.__class__.__name__}",
            reason="png_renderer_unavailable",
        ) from exc


_DEFAULT_IDIOMS = ("hold_until", "stable_while", "not_before")
_BOUND_KINDS = ("exact", "range", "parameterized", "unbounded")
_RENDERING_MODES = ("concrete", "symbolic", "mixed")
_RECOVERABILITY = {"concrete": "visual", "symbolic": "partial_visual", "mixed": "partial_visual"}
_SPLIT_POLICIES = ("random", "topology", "flavor", "bound", "size", "rendering")
_PREDICATE_KINDS = ("rise", "fall", "high", "low", "change", "stable", "eq", "neq")
_REGION_KINDS = ("at", "in", "before", "after", "from_until")
_CUT_KINDS = ("none", "before", "after", "between")
_HOLDOUT_SIZE_BUCKETS = {
    "small": ("0-6", "0-3"),
    "medium": ("7-12", "4-6"),
    "large": ("13-20", "7-12"),
}


class GenerationRng:
    """Deterministic random source with a stable child-derivation contract."""

    def __init__(self, seed: int) -> None:
        self.seed_value = int(seed)
        self._random = random.Random(self.seed_value)

    @property
    def random(self) -> random.Random:
        return self._random

    def derive_child(self, label: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed_value}:{label}".encode("utf-8")).digest()
        sub_seed = int.from_bytes(digest[:8], "big")
        return random.Random(sub_seed)


def generate_dataset(
    count: int,
    seed: int,
    out_dir: str | Path,
    *,
    min_ticks: int = 6,
    max_ticks: int = 20,
    min_lanes: int = 3,
    max_lanes: int = 12,
    concrete_ratio: float = 0.8,
    symbolic_ratio: float = 0.1,
    mixed_ratio: float = 0.1,
    max_retries: int = 100,
    coverage_target: int = 0,
    coverage_required: Iterable[str] | str | None = None,
    holdout_topology: str | None = None,
    holdout_flavor: str | None = None,
    holdout_bound: Iterable[str] | str | None = None,
    holdout_size: str | None = None,
    holdout_rendering: str | None = None,
    split_policy: str = "random",
    render_svg: bool = True,
    render_png: bool = False,
    cuts_probability: float = 0.3,
    distractor_probability: float = 0.3,
    split: str = "train",
) -> dict[str, Any]:
    """Generate a dataset of timing Image-DSL pairs and write records.jsonl."""

    if count < 1:
        raise ValueError("count must be at least 1")
    if min_ticks < 2:
        raise ValueError("min_ticks must be at least 2")
    if max_ticks < min_ticks:
        raise ValueError("max_ticks must be greater than or equal to min_ticks")
    if min_lanes < 1:
        raise ValueError("min_lanes must be at least 1")
    if max_lanes < min_lanes:
        raise ValueError("max_lanes must be greater than or equal to min_lanes")
    if coverage_target < 0:
        raise ValueError("coverage_target must be non-negative")
    if split_policy not in _SPLIT_POLICIES:
        raise ValueError(f"split_policy must be one of {', '.join(_SPLIT_POLICIES)}")
    if holdout_topology is not None and holdout_topology not in TOPOLOGIES:
        raise ValueError(f"unknown holdout topology: {holdout_topology}")
    if holdout_flavor is not None and holdout_flavor not in FLAVORS:
        raise ValueError(f"unknown holdout flavor: {holdout_flavor}")
    holdout_bound_values = _parse_csv_values(holdout_bound)
    unknown_bounds = sorted(holdout_bound_values - set(_BOUND_KINDS))
    if unknown_bounds:
        raise ValueError(f"unknown holdout bound kind: {', '.join(unknown_bounds)}")
    if holdout_size is not None and holdout_size not in _HOLDOUT_SIZE_BUCKETS:
        raise ValueError("holdout_size must be one of small, medium, large")
    if holdout_rendering is not None and holdout_rendering not in _RENDERING_MODES:
        raise ValueError("holdout_rendering must be one of concrete, symbolic, mixed")
    required_coverage = _parse_coverage_required(coverage_required)

    rng = GenerationRng(seed)
    output_path = Path(out_dir)
    (output_path / "dsl").mkdir(parents=True, exist_ok=True)
    if render_svg:
        (output_path / "svg").mkdir(parents=True, exist_ok=True)
    if render_png:
        _probe_png_renderer()
        (output_path / "png").mkdir(parents=True, exist_ok=True)

    available_topologies = tuple(t for t in TOPOLOGIES if t != holdout_topology)
    available_flavors = tuple(f for f in FLAVORS if f != holdout_flavor)
    available_bound_kinds = tuple(kind for kind in _BOUND_KINDS if kind not in holdout_bound_values)
    available_rendering_modes = tuple(mode for mode in _RENDERING_MODES if mode != holdout_rendering)
    if not available_topologies or not available_flavors or not available_bound_kinds or not available_rendering_modes:
        raise ValueError("holdouts removed all topologies, flavors, bound kinds, or rendering modes")

    rendering_choices, rendering_weights = _normalize_rendering_weights(
        concrete_ratio,
        symbolic_ratio,
        mixed_ratio,
        allowed_modes=available_rendering_modes,
    )
    rendering_quotas = _rendering_quotas(count, rendering_choices, rendering_weights)
    rendering_counts = {mode: 0 for mode in _RENDERING_MODES}

    coverage = CoverageTracker()
    accepted_records: list[dict[str, Any]] = []
    seen_canonical_hashes: set[str] = set()
    seen_svg_hashes: set[str] = set()
    seen_feature_signatures: set[str] = set()
    seen_svg_feature_pairs: set[tuple[str, str]] = set()
    rejection_counts: dict[str, int] = {}
    records_path = output_path / "records.jsonl"

    item_index = 0
    with records_path.open("w", encoding="utf-8") as records_handle:
        while len(accepted_records) < count:
            item_id = f"td_{len(accepted_records):06d}"
            item_rng = rng.derive_child(f"item:{item_index}")

            accepted = False
            for attempt in range(max_retries):
                spec = _sample_spec(
                    item_rng,
                    item_id=item_id,
                    topologies=available_topologies,
                    flavors=available_flavors,
                    rendering_choices=rendering_choices,
                    rendering_weights=rendering_weights,
                    rendering_counts=rendering_counts,
                    rendering_quotas=rendering_quotas,
                    bound_kinds=available_bound_kinds,
                    min_ticks=min_ticks,
                    max_ticks=max_ticks,
                    cuts_probability=cuts_probability,
                    distractor_probability=distractor_probability,
                    coverage=coverage,
                    coverage_target=coverage_target,
                )
                try:
                    item = _generate_one(spec, item_rng, output_path, render_svg=render_svg, render_png=render_png)
                except GenerationError as exc:
                    _bump(rejection_counts, exc.reason)
                    continue
                except Exception as exc:
                    _bump(rejection_counts, f"unexpected_{exc.__class__.__name__}")
                    continue

                canonical_hash = hashlib.sha256(item.canonical_dsl.encode("utf-8")).hexdigest()
                if canonical_hash in seen_canonical_hashes:
                    _bump(rejection_counts, "duplicate_canonical_dsl")
                    continue

                lane_count = item.features.get("lane_count")
                if not isinstance(lane_count, int) or not (min_lanes <= lane_count <= max_lanes):
                    _bump(rejection_counts, "lane_count_out_of_range")
                    continue

                holdout_reason = _holdout_rejection_reason(
                    item.features,
                    holdout_bounds=holdout_bound_values,
                    holdout_size=holdout_size,
                    holdout_rendering=holdout_rendering,
                )
                if holdout_reason:
                    _bump(rejection_counts, holdout_reason)
                    continue

                item_rendering = item.features.get("rendering", spec.rendering)
                if _rendering_mode_over_quota(str(item_rendering), rendering_counts, rendering_quotas):
                    _bump(rejection_counts, "rendering_mode_quota")
                    continue

                if render_svg and not _passes_visual_filter(item.svg_text):
                    _bump(rejection_counts, "visual_filter_failed")
                    continue

                if coverage_target > 0:
                    coverage.score(item.features)
                    if not coverage.feature_increases_deficient_bucket(item.features, coverage_target):
                        _bump(rejection_counts, "coverage_bucket_required")
                        continue

                svg_hash = _stable_svg_hash(item.svg_text) if render_svg else ""
                feature_signature = _feature_signature(item.features)
                if render_svg and (svg_hash, feature_signature) in seen_svg_feature_pairs:
                    _bump(rejection_counts, "duplicate_svg")
                    continue

                seen_canonical_hashes.add(canonical_hash)
                if render_svg:
                    seen_svg_hashes.add(svg_hash)
                    seen_svg_feature_pairs.add((svg_hash, feature_signature))
                seen_feature_signatures.add(feature_signature)
                record = _persist_item(item, spec, output_path, render_svg=render_svg, render_png=render_png, split=split)
                records_handle.write(json.dumps(record))
                records_handle.write("\n")
                records_handle.flush()
                accepted_records.append(record)
                coverage.update(item.features)
                rendering_counts[str(item_rendering)] = rendering_counts.get(str(item_rendering), 0) + 1
                accepted = True
                break

            if not accepted:
                raise GenerationError(
                    f"failed to generate valid item {item_id} after {max_retries} retries; "
                    f"rejection counts: {rejection_counts}",
                    reason="max_retries_exhausted",
                )

            item_index += 1

    missing_required = _missing_required_coverage(coverage, required_coverage)
    if missing_required:
        missing_text = ", ".join(f"{bucket}={value}" for bucket, value in missing_required)
        raise GenerationError(
            f"required coverage bucket values were not generated: {missing_text}",
            reason="coverage_bucket_required",
        )

    coverage_target_unmet: dict[str, dict[str, int]] = {}
    if coverage_target > 0:
        coverage_target_unmet = {
            bucket: {
                "distinct_count": len(coverage.counts.get(bucket, {})),
                "target": coverage_target,
            }
            for bucket in sorted(coverage.deficient_buckets(coverage_target))
        }

    summary = {
        "count": len(accepted_records),
        "seed": seed,
        "split": split,
        "split_policy": split_policy,
        "out_dir": str(output_path),
        "records_path": str(records_path),
        "summary_path": str(output_path / "summary.json"),
        "coverage": {bucket: dict(values) for bucket, values in coverage.counts.items()},
        "rejections": rejection_counts,
    }
    if coverage_target_unmet:
        summary["coverage_target_unmet"] = coverage_target_unmet
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _parse_csv_values(values: Iterable[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = []
        for value in values:
            raw_values.extend(str(value).split(","))
    return {value.strip() for value in raw_values if value.strip()}


def _parse_coverage_required(values: Iterable[str] | str | None) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for entry in _parse_csv_values(values):
        if "=" in entry:
            bucket, value = entry.split("=", 1)
        elif ":" in entry:
            bucket, value = entry.split(":", 1)
        else:
            raise ValueError("coverage_required entries must use bucket=value")
        bucket = bucket.strip()
        value = value.strip()
        if not bucket or not value:
            raise ValueError("coverage_required entries must use bucket=value")
        required.setdefault(bucket, set()).add(value)
    return required


def _missing_required_coverage(
    coverage: CoverageTracker,
    required: dict[str, set[str]],
) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for bucket, values in sorted(required.items()):
        counts = coverage.counts.get(bucket, {})
        for value in sorted(values):
            if counts.get(value, 0) <= 0:
                missing.append((bucket, value))
    return missing


def _normalize_rendering_weights(
    concrete: float,
    symbolic: float,
    mixed: float,
    *,
    allowed_modes: tuple[str, ...] = _RENDERING_MODES,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    raw_weights = {
        "concrete": max(0.0, concrete),
        "symbolic": max(0.0, symbolic),
        "mixed": max(0.0, mixed),
    }
    choices = tuple(mode for mode in _RENDERING_MODES if mode in allowed_modes)
    weights = [raw_weights[mode] for mode in choices]
    total = sum(weights)
    if total <= 0:
        fallback = "concrete" if "concrete" in choices else choices[0]
        weights = [1.0 if mode == fallback else 0.0 for mode in choices]
        total = 1.0
    return choices, tuple(w / total for w in weights)


def _rendering_quotas(
    count: int,
    rendering_choices: tuple[str, ...],
    rendering_weights: tuple[float, ...],
) -> dict[str, int]:
    return {
        mode: int(math.ceil(count * weight))
        for mode, weight in zip(rendering_choices, rendering_weights)
    }


def _rendering_mode_over_quota(
    rendering: str,
    rendering_counts: dict[str, int],
    rendering_quotas: dict[str, int],
) -> bool:
    quota = rendering_quotas.get(rendering)
    if quota is None or rendering_counts.get(rendering, 0) < quota:
        return False
    return any(
        rendering_counts.get(mode, 0) < mode_quota
        for mode, mode_quota in rendering_quotas.items()
        if mode != rendering
    )


def _sample_spec(
    rng: random.Random,
    *,
    item_id: str,
    topologies: tuple[str, ...],
    flavors: tuple[str, ...],
    rendering_choices: tuple[str, ...],
    rendering_weights: tuple[float, ...],
    rendering_counts: dict[str, int],
    rendering_quotas: dict[str, int],
    bound_kinds: tuple[str, ...],
    min_ticks: int,
    max_ticks: int,
    cuts_probability: float,
    distractor_probability: float,
    coverage: CoverageTracker | None = None,
    coverage_target: int = 0,
) -> GenerationSpec:
    item_number = int(item_id.rsplit("_", 1)[-1])
    default_topology = topologies[item_number % len(topologies)] if item_number < len(topologies) else rng.choice(topologies)
    default_flavor = flavors[item_number % len(flavors)] if item_number < len(flavors) else rng.choice(flavors)
    missing_topologies = _missing_if_deficient(coverage, coverage_target, "topology", topologies)
    missing_flavors = _missing_if_deficient(coverage, coverage_target, "naming", tuple(FLAVORS[f].naming_style for f in flavors))
    topology = _biased_choice(rng, topologies, missing_topologies, fallback=default_topology)
    flavor_preference = tuple(
        flavor_name
        for flavor_name in flavors
        if FLAVORS[flavor_name].naming_style in set(missing_flavors)
    )
    flavor = _biased_choice(rng, flavors, flavor_preference, fallback=default_flavor)
    rendering = _sample_rendering(
        rng,
        rendering_choices,
        rendering_weights,
        rendering_counts,
        rendering_quotas,
        preferred=_missing_if_deficient(coverage, coverage_target, "rendering", rendering_choices),
    )
    missing_bounds = _missing_if_deficient(coverage, coverage_target, "bound_kind", bound_kinds)
    sampled_bound_kinds = _sample_bound_kinds(rng, allowed=bound_kinds, preferred=missing_bounds)
    missing_cuts = _missing_if_deficient(coverage, coverage_target, "cut", _CUT_KINDS)
    cuts_enabled = _sample_cuts_enabled(rng, cuts_probability, missing_cuts)
    missing_idioms = _missing_if_deficient(
        coverage,
        coverage_target,
        "idiom",
        (*_DEFAULT_IDIOMS, "response", "burst", "backpressure", "cut"),
    )
    idioms = _sample_idioms(rng, topology, cuts_enabled=cuts_enabled, preferred=missing_idioms)
    naming = FLAVORS[flavor].naming_style
    distractor_lanes = rng.randint(0, 2) if rng.random() < distractor_probability else 0
    tick_budget = _sample_tick_budget(
        rng,
        min_ticks,
        max_ticks,
        preferred=_missing_if_deficient(
            coverage,
            coverage_target,
            "tick_count",
            tuple(bucket_range(value, TICK_COUNT_BUCKETS) for value in range(min_ticks, max_ticks + 1)),
        ),
    )
    clock_edge = "negedge" if rng.random() < 0.35 else "posedge"
    return GenerationSpec(
        item_id=item_id,
        seed=rng.randint(0, 2**31 - 1),
        topology=topology,
        flavor=flavor,
        idioms=idioms,
        rendering=rendering,
        bound_kinds=sampled_bound_kinds,
        naming=naming,
        cuts_enabled=cuts_enabled,
        distractor_lanes=distractor_lanes,
        tick_budget=tick_budget,
        clock_edge=clock_edge,
        predicate_bias=_missing_if_deficient(coverage, coverage_target, "predicate", _PREDICATE_KINDS),
        region_bias=_missing_if_deficient(coverage, coverage_target, "region", _REGION_KINDS),
        cut_placement_bias=tuple(value for value in missing_cuts if value != "none"),
    )


def _missing_if_deficient(
    coverage: CoverageTracker | None,
    target: int,
    bucket: str,
    values: tuple[str, ...],
) -> tuple[str, ...]:
    if coverage is None or not coverage.is_bucket_deficient(bucket, target):
        return ()
    unique_values = tuple(dict.fromkeys(values))
    return coverage.missing_values(bucket, unique_values)


def _biased_choice(
    rng: random.Random,
    choices: tuple[str, ...],
    preferred: tuple[str, ...],
    *,
    fallback: str,
    bias: float = 0.70,
) -> str:
    preferred_set = set(preferred)
    if preferred_set and rng.random() < bias:
        weights = [5.0 if choice in preferred_set else 1.0 for choice in choices]
        return rng.choices(choices, weights=weights, k=1)[0]
    return fallback if fallback in choices else rng.choice(choices)


def _sample_rendering(
    rng: random.Random,
    choices: tuple[str, ...],
    weights: tuple[float, ...],
    counts: dict[str, int],
    quotas: dict[str, int],
    *,
    preferred: tuple[str, ...],
) -> str:
    under_quota = tuple(mode for mode in choices if counts.get(mode, 0) < quotas.get(mode, 0))
    active_choices = under_quota or choices
    base_weights = dict(zip(choices, weights))
    preferred_set = set(preferred)
    active_weights = []
    for mode in active_choices:
        weight = base_weights.get(mode, 0.0)
        if mode in preferred_set:
            weight = max(weight, 0.05) * 5.0
        active_weights.append(weight)
    if sum(active_weights) <= 0:
        active_weights = [1.0] * len(active_choices)
    return rng.choices(active_choices, weights=active_weights, k=1)[0]


def _sample_cuts_enabled(
    rng: random.Random,
    probability: float,
    preferred_cuts: tuple[str, ...],
) -> bool:
    preferred_set = set(preferred_cuts)
    if preferred_set and rng.random() < 0.75:
        if preferred_set == {"none"}:
            return False
        return True
    return rng.random() < probability


def _sample_tick_budget(
    rng: random.Random,
    min_ticks: int,
    max_ticks: int,
    *,
    preferred: tuple[str, ...],
) -> int:
    candidates = tuple(range(min_ticks, max_ticks + 1))
    preferred_set = set(preferred)
    if preferred_set and rng.random() < 0.70:
        weights = [
            5.0 if bucket_range(candidate, TICK_COUNT_BUCKETS) in preferred_set else 1.0
            for candidate in candidates
        ]
        return rng.choices(candidates, weights=weights, k=1)[0]
    return rng.choice(candidates)


def _sample_bound_kinds(
    rng: random.Random,
    *,
    allowed: tuple[str, ...],
    preferred: tuple[str, ...],
) -> tuple[str, ...]:
    primary_pool = tuple(kind for kind in ("range", "exact") if kind in allowed) or allowed
    primary = _biased_choice(
        rng,
        primary_pool,
        tuple(kind for kind in preferred if kind in primary_pool),
        fallback=rng.choice(primary_pool),
    )
    extras = []
    if "parameterized" in allowed and rng.random() < (0.75 if "parameterized" in preferred else 0.4):
        extras.append("parameterized")
    if "unbounded" in allowed and rng.random() < (0.75 if "unbounded" in preferred else 0.15):
        extras.append("unbounded")
    for kind in preferred:
        if kind in allowed and kind not in extras and rng.random() < 0.70:
            extras.append(kind)
    chosen = [primary, *extras]
    seen = []
    for kind in chosen:
        if kind not in seen:
            seen.append(kind)
    return tuple(seen)


def _sample_idioms(
    rng: random.Random,
    topology: str,
    *,
    cuts_enabled: bool,
    preferred: tuple[str, ...] = (),
) -> tuple[str, ...]:
    available = list(_DEFAULT_IDIOMS)
    if topology in {"single_response", "chain", "fork", "join", "parallel"}:
        available.append("response")
    if topology == "burst":
        available.append("burst")
    if topology == "backpressure":
        available.append("backpressure")
    chosen = []
    preferred_set = set(preferred)
    for idiom in available:
        probability = 0.85 if idiom in preferred_set else 0.65
        if rng.random() < probability:
            chosen.append(idiom)
    if not chosen:
        chosen.append("hold_until")
    if cuts_enabled:
        chosen.append("cut")
    return tuple(dict.fromkeys(chosen))


def _generate_one(
    spec: GenerationSpec,
    item_rng: random.Random,
    output_path: Path,
    *,
    render_svg: bool,
    render_png: bool,
) -> GeneratedItem:
    graph = build_topology(spec.topology, spec.flavor, item_rng, predicate_bias=spec.predicate_bias)
    components = apply_idioms(graph, spec, item_rng)

    anchor_ticks, total_ticks = assign_ticks(components, spec, item_rng)

    if spec.rendering == "symbolic":
        sampled_components = components
        actual_spec = spec
    else:
        samples = synthesize_waveforms(components, anchor_ticks, total_ticks, spec, item_rng)
        actual_spec = spec
        if spec.rendering == "mixed":
            samples, actual_rendering = _redact_mixed(
                samples,
                item_rng,
                anchors=components.anchors,
                lane_constraints=components.lane_constraints,
            )
            actual_spec = replace(spec, rendering=actual_rendering)
        sampled_components = attach_samples(components, samples)

    document = _build_document(sampled_components, total_ticks, actual_spec)
    dsl_text = emit_timing_dsl(document)

    parsed = parse_diagram(dsl_text)
    canonical_dsl = emit_timing_dsl(parsed)

    svg_text = ""
    if render_svg:
        svg_text = render_diagram_svg(_with_response_overlays(parsed))

    features = _extract_features(parsed, actual_spec)
    return GeneratedItem(
        id=spec.item_id,
        seed=spec.seed,
        canonical_dsl=canonical_dsl,
        svg_text=svg_text,
        features=features,
    )


def _build_document(
    components: ScenarioComponents, total_ticks: int, spec: GenerationSpec
) -> ScenarioDocument:
    has_samples = any(signal.samples for signal in components.signals)
    return ScenarioDocument(
        name=components.name,
        clocking=ClockingSpec(edge=spec.clock_edge, signal=components.clock_signal),
        params=tuple(components.params),
        signals=tuple(components.signals),
        anchors=tuple(components.anchors),
        windows=tuple(components.windows),
        cuts=tuple(components.cuts),
        lane_constraints=tuple(components.lane_constraints),
        ticks=total_ticks if has_samples else (total_ticks if spec.rendering == "symbolic" else None),
    )


def _with_response_overlays(document: ScenarioDocument) -> ScenarioDocument:
    if not document.windows:
        return document
    properties = (
        PropertyOverlay(name="p_render_context", body="1'b1"),
        *(_response_property_for_window(window) for window in document.windows),
    )
    return replace(document, properties=properties)


def _response_property_for_window(window: TimeWindow) -> PropertyOverlay:
    body = f"{window.start_anchor} |-> ##{window.bound.label} {window.end_anchor}"
    return PropertyOverlay(
        name=f"p_{window.name}",
        body=body,
        related_anchors=(window.start_anchor, window.end_anchor),
        related_windows=(window.name,),
    )


def _redact_mixed(
    samples: dict[str, tuple[str, ...]],
    rng: random.Random,
    *,
    anchors: Iterable[Anchor],
    lane_constraints: Iterable[LaneConstraint],
) -> tuple[dict[str, tuple[str, ...]], str]:
    keys = list(samples.keys())
    if not keys:
        return samples, "concrete"
    must_keep = _must_keep_sample_lanes(anchors, lane_constraints)
    eligible = [name for name in keys if name not in must_keep]
    if not eligible:
        return dict(samples), "concrete"
    keep = max(1, len(keys) // 2)
    redact_count = min(len(eligible), max(0, len(keys) - keep))
    redact = set(rng.sample(eligible, redact_count)) if redact_count else set()
    redacted = {}
    for name, values in samples.items():
        if name in redact:
            redacted[name] = ()
        else:
            redacted[name] = values
    return redacted, "mixed" if any(not values for values in redacted.values()) else "concrete"


def _must_keep_sample_lanes(
    anchors: Iterable[Anchor],
    lane_constraints: Iterable[LaneConstraint],
) -> set[str]:
    keep: set[str] = set()
    for anchor in anchors:
        keep.update(collect_signals(anchor.condition))
    for constraint in lane_constraints:
        keep.update(constraint.signals)
    return keep


def _holdout_rejection_reason(
    features: dict[str, Any],
    *,
    holdout_bounds: set[str],
    holdout_size: str | None,
    holdout_rendering: str | None,
) -> str | None:
    if holdout_bounds and holdout_bounds.intersection(features.get("bound_kinds", ()) or ()):
        return "holdout_bound"
    if holdout_rendering and features.get("rendering") == holdout_rendering:
        return "holdout_rendering"
    if holdout_size:
        expected_tick, expected_lane = _HOLDOUT_SIZE_BUCKETS[holdout_size]
        ticks = features.get("ticks")
        lanes = features.get("lane_count")
        if (
            isinstance(ticks, int)
            and isinstance(lanes, int)
            and bucket_range(ticks, TICK_COUNT_BUCKETS) == expected_tick
            and bucket_range(lanes, LANE_COUNT_BUCKETS) == expected_lane
        ):
            return "holdout_size"
    return None


def _passes_visual_filter(svg_text: str) -> bool:
    if not svg_text or not (200 <= len(svg_text) <= 400_000):
        return False
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return False

    dimensions = _declared_svg_dimensions(root)
    if dimensions is None:
        return False
    width, height = dimensions
    if width < 80 or height < 80 or width > 8000 or height > 8000:
        return False

    timing_groups = [
        element
        for element in root.iter()
        if _local_name(element.tag) == "g"
        and any(token.startswith("timing-") for token in element.attrib.get("class", "").split())
    ]
    if not timing_groups:
        return False

    text_elements = [element for element in root.iter() if _local_name(element.tag) == "text"]
    if not text_elements:
        return False
    return _severe_adjacent_label_overlap_count(text_elements) <= 5


def _declared_svg_dimensions(root: ET.Element) -> tuple[float, float] | None:
    view_box = root.attrib.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                return None
    width = _parse_svg_length(root.attrib.get("width"))
    height = _parse_svg_length(root.attrib.get("height"))
    if width is None or height is None:
        return None
    return width, height


def _parse_svg_length(value: str | None) -> float | None:
    if not value:
        return None
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", value)
    if not match:
        return None
    return float(match.group(1))


def _severe_adjacent_label_overlap_count(text_elements: list[ET.Element]) -> int:
    by_y: dict[float, list[float]] = {}
    for element in text_elements:
        x = _parse_svg_length(element.attrib.get("x"))
        y = _parse_svg_length(element.attrib.get("y"))
        if x is None or y is None:
            continue
        by_y.setdefault(round(y, 2), []).append(x)

    overlap_count = 0
    for xs in by_y.values():
        sorted_xs = sorted(xs)
        for left, right in zip(sorted_xs, sorted_xs[1:]):
            if abs(right - left) < 8.0:
                overlap_count += 1
    return overlap_count


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _stable_svg_hash(svg_text: str) -> str:
    normalized = re.sub(r'\s(?:viewBox|width|height)="[^"]*"', "", svg_text)
    normalized = re.sub(r"\s+", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _feature_signature(features: dict[str, Any]) -> str:
    ticks = features.get("ticks")
    lanes = features.get("lane_count")
    tick_count_bucket = bucket_range(ticks, TICK_COUNT_BUCKETS) if isinstance(ticks, int) else "unknown"
    lane_count_bucket = bucket_range(lanes, LANE_COUNT_BUCKETS) if isinstance(lanes, int) else "unknown"
    cut_values = features.get("cuts")
    if not cut_values:
        cut = features.get("cut")
        cut_values = (cut,) if cut else ()
    payload = [
        features.get("topology"),
        sorted(features.get("idioms", ()) or ()),
        sorted(features.get("predicates", ()) or ()),
        sorted(features.get("constraint_regions", ()) or ()),
        sorted(features.get("bound_kinds", ()) or ()),
        tick_count_bucket,
        lane_count_bucket,
        features.get("lane_kind"),
        sorted(cut_values),
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _extract_features(document: ScenarioDocument, spec: GenerationSpec) -> dict[str, Any]:
    bus_lanes = [signal for signal in document.signals if signal.kind == SignalKind.BUS]
    bit_lanes = [signal for signal in document.signals if signal.kind == SignalKind.BIT]
    if bus_lanes and bit_lanes:
        lane_kind = "mixed"
    elif bus_lanes:
        lane_kind = "bus_only"
    else:
        lane_kind = "bit_only"

    bound_kinds = sorted({_bound_feature(window) for window in document.windows})
    constraint_regions = sorted({constraint.region.value for constraint in document.lane_constraints})
    predicates = sorted(_collect_anchor_predicates(document.anchors))
    cut_placements = sorted({_cut_feature(cut.placement.value) for cut in document.cuts})
    if not cut_placements:
        cut_placements = ["none"]

    return {
        "topology": spec.topology,
        "flavor": spec.flavor,
        "idioms": list(spec.idioms),
        "ticks": document.ticks,
        "lane_count": len(document.signals),
        "anchor_count": len(document.anchors),
        "window_count": len(document.windows),
        "has_bus": bool(bus_lanes),
        "has_params": bool(document.params),
        "bound_kinds": bound_kinds,
        "constraint_regions": constraint_regions,
        "predicates": predicates,
        "lane_kind": lane_kind,
        "rendering": spec.rendering,
        "naming": spec.naming,
        "cut": cut_placements[0] if len(cut_placements) == 1 else "multiple",
        "cuts": cut_placements,
    }


def _bound_feature(window: TimeWindow) -> str:
    tokens = (window.bound.min_delay, window.bound.max_delay)
    if any(token and not token.isdigit() and token != "$" for token in tokens):
        return "parameterized"
    return window.bound.kind.value


def _cut_feature(placement: str) -> str:
    return {
        "before_anchor": "before",
        "after_anchor": "after",
        "between_windows": "between",
    }.get(placement, placement)


def _collect_anchor_predicates(anchors: Iterable[Anchor]) -> set[str]:
    seen: set[str] = set()
    for anchor in anchors:
        _walk_condition_predicates(anchor.condition, seen)
    return seen


def _walk_condition_predicates(condition, seen: set[str]) -> None:
    if condition is None:
        return
    if condition.kind == "predicate" and condition.predicate is not None:
        predicate: Predicate = condition.predicate
        if predicate.op:
            seen.add(predicate.op)
    for child in condition.items:
        _walk_condition_predicates(child, seen)


def _persist_item(
    item: GeneratedItem,
    spec: GenerationSpec,
    out_dir: Path,
    *,
    render_svg: bool,
    render_png: bool,
    split: str,
) -> dict[str, Any]:
    dsl_relative = f"dsl/{item.id}.td"
    (out_dir / dsl_relative).write_text(item.canonical_dsl + "\n", encoding="utf-8")

    svg_relative: str | None = None
    if render_svg and item.svg_text:
        svg_relative = f"svg/{item.id}.svg"
        (out_dir / svg_relative).write_text(item.svg_text, encoding="utf-8")

    png_relative: str | None = None
    if render_png:
        from sva_toolkit.timing.render.png import render_diagram_png

        parsed = parse_diagram(item.canonical_dsl)
        png_relative = f"png/{item.id}.png"
        render_diagram_png(parsed, out_dir / png_relative)

    record: dict[str, Any] = {
        "id": item.id,
        "seed": item.seed,
        "split": split,
        "dsl_path": dsl_relative,
        "features": item.features,
        "target": {
            "canonical_dsl": item.canonical_dsl,
            "recoverability": _RECOVERABILITY.get(str(item.features.get("rendering", spec.rendering)), "visual"),
        },
    }
    if svg_relative:
        record["svg_path"] = svg_relative
    if png_relative:
        record["png_path"] = png_relative
    return record
