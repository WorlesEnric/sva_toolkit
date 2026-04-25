"""Top-level orchestration for the timing diagram dataset generator."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.core.conditions import Predicate
from sva_toolkit.timing.core.scenario import (
    Anchor,
    ClockingSpec,
    PropertyOverlay,
    ScenarioDocument,
    SignalKind,
    TimeWindow,
)
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.generate.coverage import CoverageTracker
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
            "Install sva-toolkit[timing-render]."
        ) from exc
    except OSError as exc:
        raise GenerationError(
            "PNG output requested but cairosvg cannot load its native dependencies. "
            "Install libcairo on the system (e.g., 'brew install cairo' on macOS, "
            "'apt install libcairo2' on Debian/Ubuntu). "
            "Underlying error: see traceback above."
        ) from exc
    try:
        cairosvg.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>')
    except Exception as exc:
        raise GenerationError(
            "PNG output requested but cairosvg cannot rasterize on this system. "
            "Make sure libcairo is available. "
            f"Underlying error: {exc.__class__.__name__}"
        ) from exc


_DEFAULT_IDIOMS = ("hold_until", "stable_while", "not_before")
_BOUND_KINDS = ("exact", "range", "parameterized", "unbounded")
_RENDERING_MODES = ("concrete", "symbolic", "mixed")
_RECOVERABILITY = {"concrete": "visual", "symbolic": "partial_visual", "mixed": "partial_visual"}


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
    holdout_topology: str | None = None,
    holdout_flavor: str | None = None,
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
    if not available_topologies or not available_flavors:
        raise ValueError("holdouts removed all topologies or flavors")

    rendering_choices, rendering_weights = _normalize_rendering_weights(
        concrete_ratio, symbolic_ratio, mixed_ratio
    )

    coverage = CoverageTracker()
    accepted_records: list[dict[str, Any]] = []
    seen_canonical_hashes: set[str] = set()
    rejection_counts: dict[str, int] = {}

    item_index = 0
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
                min_ticks=min_ticks,
                max_ticks=max_ticks,
                cuts_probability=cuts_probability,
                distractor_probability=distractor_probability,
            )
            try:
                item = _generate_one(spec, item_rng, output_path, render_svg=render_svg, render_png=render_png)
            except GenerationError as exc:
                _bump(rejection_counts, f"generation_error:{exc.__class__.__name__}")
                continue
            except Exception as exc:
                _bump(rejection_counts, f"unexpected:{exc.__class__.__name__}")
                continue

            canonical_hash = hashlib.sha256(item.canonical_dsl.encode("utf-8")).hexdigest()
            if canonical_hash in seen_canonical_hashes:
                _bump(rejection_counts, "duplicate")
                continue

            lane_count = item.features.get("lane_count")
            if not isinstance(lane_count, int) or not (min_lanes <= lane_count <= max_lanes):
                _bump(rejection_counts, "lane_count")
                continue

            if render_svg and not _passes_visual_filter(item.svg_text):
                _bump(rejection_counts, "visual_filter")
                continue

            seen_canonical_hashes.add(canonical_hash)
            record = _persist_item(item, spec, output_path, render_svg=render_svg, render_png=render_png, split=split)
            accepted_records.append(record)
            coverage.update(item.features)
            accepted = True
            break

        if not accepted:
            raise GenerationError(
                f"failed to generate valid item {item_id} after {max_retries} retries; "
                f"rejection counts: {rejection_counts}"
            )

        item_index += 1

    records_path = output_path / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in accepted_records:
            handle.write(json.dumps(record))
            handle.write("\n")

    summary = {
        "count": len(accepted_records),
        "seed": seed,
        "split": split,
        "out_dir": str(output_path),
        "records_path": str(records_path),
        "coverage": {bucket: dict(values) for bucket, values in coverage.counts.items()},
        "rejections": rejection_counts,
    }
    return summary


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _normalize_rendering_weights(
    concrete: float, symbolic: float, mixed: float
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    weights = [max(0.0, concrete), max(0.0, symbolic), max(0.0, mixed)]
    total = sum(weights)
    if total <= 0:
        weights = [1.0, 0.0, 0.0]
        total = 1.0
    return _RENDERING_MODES, tuple(w / total for w in weights)


def _sample_spec(
    rng: random.Random,
    *,
    item_id: str,
    topologies: tuple[str, ...],
    flavors: tuple[str, ...],
    rendering_choices: tuple[str, ...],
    rendering_weights: tuple[float, ...],
    min_ticks: int,
    max_ticks: int,
    cuts_probability: float,
    distractor_probability: float,
) -> GenerationSpec:
    topology = rng.choice(topologies)
    flavor = rng.choice(flavors)
    rendering = rng.choices(rendering_choices, weights=rendering_weights, k=1)[0]
    bound_kinds = _sample_bound_kinds(rng)
    cuts_enabled = rng.random() < cuts_probability
    idioms = _sample_idioms(rng, topology, cuts_enabled=cuts_enabled)
    naming = FLAVORS[flavor].naming_style
    distractor_lanes = rng.randint(0, 2) if rng.random() < distractor_probability else 0
    tick_budget = rng.randint(min_ticks, max(min_ticks, max_ticks))
    return GenerationSpec(
        item_id=item_id,
        seed=rng.randint(0, 2**31 - 1),
        topology=topology,
        flavor=flavor,
        idioms=idioms,
        rendering=rendering,
        bound_kinds=bound_kinds,
        naming=naming,
        cuts_enabled=cuts_enabled,
        distractor_lanes=distractor_lanes,
        tick_budget=tick_budget,
    )


def _sample_bound_kinds(rng: random.Random) -> tuple[str, ...]:
    primary = rng.choice(["range", "exact"])
    extras = []
    if rng.random() < 0.4:
        extras.append("parameterized")
    if rng.random() < 0.15:
        extras.append("unbounded")
    chosen = [primary, *extras]
    seen = []
    for kind in chosen:
        if kind not in seen:
            seen.append(kind)
    return tuple(seen)


def _sample_idioms(rng: random.Random, topology: str, *, cuts_enabled: bool) -> tuple[str, ...]:
    available = list(_DEFAULT_IDIOMS)
    if topology in {"single_response", "chain", "fork", "join", "parallel"}:
        available.append("response")
    if topology == "burst":
        available.append("burst")
    if topology == "backpressure":
        available.append("backpressure")
    chosen = []
    for idiom in available:
        if rng.random() < 0.65:
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
    graph = build_topology(spec.topology, spec.flavor, item_rng)
    components = apply_idioms(graph, spec, item_rng)

    anchor_ticks, total_ticks = assign_ticks(components, spec, item_rng)

    if spec.rendering == "symbolic":
        sampled_components = components
    else:
        samples = synthesize_waveforms(components, anchor_ticks, total_ticks, spec, item_rng)
        if spec.rendering == "mixed":
            samples = _redact_mixed(samples, item_rng)
        sampled_components = attach_samples(components, samples)

    document = _build_document(sampled_components, total_ticks, spec)
    dsl_text = emit_timing_dsl(document)

    parsed = parse_diagram(dsl_text)
    canonical_dsl = emit_timing_dsl(parsed)

    svg_text = ""
    if render_svg:
        svg_text = render_diagram_svg(_with_response_overlays(parsed))

    features = _extract_features(parsed, spec)
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
        clocking=ClockingSpec(edge="posedge", signal=components.clock_signal),
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


def _redact_mixed(samples: dict[str, tuple[str, ...]], rng: random.Random) -> dict[str, tuple[str, ...]]:
    keys = list(samples.keys())
    if not keys:
        return samples
    keep = max(1, len(keys) // 2)
    redact = set(rng.sample(keys, len(keys) - keep)) if len(keys) > keep else set()
    redacted = {}
    for name, values in samples.items():
        if name in redact:
            redacted[name] = ()
        else:
            redacted[name] = values
    return redacted


def _passes_visual_filter(svg_text: str) -> bool:
    if not svg_text:
        return False
    return 200 <= len(svg_text) <= 400_000


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
            "recoverability": _RECOVERABILITY.get(spec.rendering, "visual"),
        },
    }
    if svg_relative:
        record["svg_path"] = svg_relative
    if png_relative:
        record["png_path"] = png_relative
    return record
