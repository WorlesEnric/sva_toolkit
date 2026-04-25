"""Tick assignment and waveform synthesis for generated scenarios."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    SignalDecl,
    SignalKind,
)
from sva_toolkit.timing.generate.model import (
    EventNode,
    GenerationError,
    GenerationSpec,
    ScenarioComponents,
    TemporalEdge,
)


_HEX_TOKENS = (
    "00",
    "01",
    "02",
    "0A",
    "10",
    "1F",
    "33",
    "55",
    "7E",
    "A1",
    "A5",
    "BE",
    "D0",
    "EE",
    "F0",
    "FF",
)


def assign_ticks(components: ScenarioComponents, spec: GenerationSpec, rng: random.Random) -> tuple[dict[str, int], int]:
    """Topologically order anchors and assign concrete absolute ticks."""

    incoming: dict[str, list[TemporalEdge]] = defaultdict(list)
    outgoing_count: dict[str, int] = defaultdict(int)
    nodes = {node.id for node in components.anchor_node_map.values()}

    for edge in components.edges:
        incoming[edge.end].append(edge)
        outgoing_count[edge.start] += 1

    roots = [node for node in nodes if not incoming.get(node)]
    if not roots:
        raise GenerationError("graph has no root anchor")

    ticks: dict[str, int] = {}
    base_offset = max(1, rng.randint(1, 2))
    for root in roots:
        ticks[root] = base_offset

    pending = set(nodes) - set(ticks)
    safety = 0
    while pending and safety < 1000:
        safety += 1
        progressed = False
        for node in list(pending):
            edges_in = incoming.get(node, [])
            if not edges_in:
                continue
            if any(edge.start not in ticks for edge in edges_in):
                continue
            candidate = max(ticks[edge.start] + max(1, edge.sampled_delay) for edge in edges_in)
            ticks[node] = candidate
            pending.discard(node)
            progressed = True
        if not progressed:
            raise GenerationError("graph contains a cycle or unreachable nodes")

    if pending:
        raise GenerationError("failed to fully order graph")

    max_tick = max(ticks.values())
    total_ticks = spec.tick_budget
    if max_tick >= total_ticks:
        raise GenerationError(
            f"assigned anchor tick {max_tick} exceeds tick budget {spec.tick_budget}"
        )
    return ticks, total_ticks


def synthesize_waveforms(
    components: ScenarioComponents,
    anchor_ticks: dict[str, int],
    total_ticks: int,
    spec: GenerationSpec,
    rng: random.Random,
) -> dict[str, tuple[str, ...]]:
    """Build per-signal sample tuples that satisfy anchors and constraints."""

    bit_must_high: dict[str, set[int]] = defaultdict(set)
    bit_must_low: dict[str, set[int]] = defaultdict(set)
    bus_value_ranges: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    for node in components.anchor_node_map.values():
        if node.id not in anchor_ticks:
            continue
        tick = anchor_ticks[node.id]
        if tick < 0 or tick >= total_ticks:
            raise GenerationError(f"anchor '{node.id}' tick {tick} out of range")
        kind = node.predicate_kind
        if kind == "rise":
            sig = node.primary_signal
            if sig is None:
                continue
            if tick - 1 >= 0:
                bit_must_low[sig].add(tick - 1)
            bit_must_high[sig].add(tick)
        elif kind == "fall":
            sig = node.primary_signal
            if sig is None:
                continue
            if tick - 1 >= 0:
                bit_must_high[sig].add(tick - 1)
            bit_must_low[sig].add(tick)
        elif kind == "high":
            sig = node.primary_signal
            if sig:
                bit_must_high[sig].add(tick)
        elif kind == "low":
            sig = node.primary_signal
            if sig:
                bit_must_low[sig].add(tick)
        elif kind == "all_high":
            signals = [node.primary_signal, *node.extra_signals]
            for sig in signals:
                if sig:
                    bit_must_high[sig].add(tick)

    for constraint in components.lane_constraints:
        if constraint.region == ConstraintRegion.FROM_UNTIL:
            t_start = anchor_ticks.get(constraint.start_anchor)
            t_end = anchor_ticks.get(constraint.end_anchor)
            if t_start is None or t_end is None:
                continue
            if t_end < t_start:
                continue
            if constraint.relation == "high":
                for sig in constraint.signals:
                    for tt in range(t_start, t_end + 1):
                        bit_must_high[sig].add(tt)
            elif constraint.relation == "low":
                for sig in constraint.signals:
                    for tt in range(t_start, t_end + 1):
                        bit_must_low[sig].add(tt)
            elif constraint.relation == "stable":
                value = _pick_bus_value(constraint.signals[0], rng)
                for sig in constraint.signals:
                    bus_value_ranges[sig].append((t_start, t_end, value))
        elif constraint.region == ConstraintRegion.BEFORE:
            t = anchor_ticks.get(constraint.anchor)
            if t is None:
                continue
            if constraint.relation == "low":
                for sig in constraint.signals:
                    for tt in range(0, t):
                        bit_must_low[sig].add(tt)
            elif constraint.relation == "high":
                for sig in constraint.signals:
                    for tt in range(0, t):
                        bit_must_high[sig].add(tt)
        elif constraint.region == ConstraintRegion.AFTER:
            t = anchor_ticks.get(constraint.anchor)
            if t is None:
                continue
            if constraint.relation == "high":
                for sig in constraint.signals:
                    for tt in range(t, total_ticks):
                        bit_must_high[sig].add(tt)
            elif constraint.relation == "low":
                for sig in constraint.signals:
                    for tt in range(t, total_ticks):
                        bit_must_low[sig].add(tt)
        elif constraint.region == ConstraintRegion.AT:
            t = anchor_ticks.get(constraint.anchor)
            if t is None:
                continue
            if constraint.relation == "high":
                for sig in constraint.signals:
                    bit_must_high[sig].add(t)
            elif constraint.relation == "low":
                for sig in constraint.signals:
                    bit_must_low[sig].add(t)

    for sig, highs in bit_must_high.items():
        clash = highs & bit_must_low.get(sig, set())
        if clash:
            raise GenerationError(f"bit signal '{sig}' has conflicting high/low requirements at {sorted(clash)}")

    samples: dict[str, tuple[str, ...]] = {}
    for signal in components.signals:
        if signal.kind == SignalKind.BIT:
            timeline = ["0"] * total_ticks
            for tick in bit_must_high.get(signal.name, set()):
                if 0 <= tick < total_ticks:
                    timeline[tick] = "1"
            for tick in bit_must_low.get(signal.name, set()):
                if 0 <= tick < total_ticks:
                    timeline[tick] = "0"
            _add_bit_distractors(timeline, signal.name, bit_must_high, bit_must_low, rng)
            samples[signal.name] = tuple(timeline)
        else:
            timeline = ["x"] * total_ticks
            ranges = bus_value_ranges.get(signal.name, [])
            ranges.sort(key=lambda r: (r[0], r[1]))
            assigned: dict[int, str] = {}
            for start, end, value in ranges:
                for tt in range(start, min(end + 1, total_ticks)):
                    if tt in assigned and assigned[tt] != value:
                        raise GenerationError(
                            f"bus signal '{signal.name}' has conflicting stable values at tick {tt}"
                        )
                    assigned[tt] = value
                    timeline[tt] = value
            samples[signal.name] = tuple(timeline)

    _verify_waveform_semantics(components, anchor_ticks, samples, total_ticks)
    return samples


def _verify_waveform_semantics(
    components: ScenarioComponents,
    anchor_ticks: dict[str, int],
    samples: dict[str, tuple[str, ...]],
    total_ticks: int,
) -> None:
    for node in components.anchor_node_map.values():
        tick = anchor_ticks.get(node.id)
        if tick is None:
            raise GenerationError(f"anchor '{node.id}' has no assigned tick")
        if not _node_predicate_holds(node, samples, tick):
            raise GenerationError(f"anchor '{node.id}' predicate does not hold at tick {tick}")

    for constraint in components.lane_constraints:
        ranges = _constraint_ranges(constraint, anchor_ticks, components, total_ticks)
        for start, end in ranges:
            for signal_name in constraint.signals:
                signal_samples = samples.get(signal_name)
                if signal_samples is None:
                    raise GenerationError(f"constraint '{constraint.name}' references unsampled lane '{signal_name}'")
                if not _constraint_holds(constraint.relation, constraint.value, signal_samples, start, end):
                    raise GenerationError(
                        f"constraint '{constraint.name}' relation '{constraint.relation}' "
                        f"does not hold on lane '{signal_name}' from tick {start} to {end}"
                    )


def _node_predicate_holds(node: EventNode, samples: dict[str, tuple[str, ...]], tick: int) -> bool:
    if node.predicate_kind == "all_high":
        signals = (node.primary_signal, *node.extra_signals)
        return all(_sample_at(samples, signal, tick) == "1" for signal in signals if signal)
    if node.primary_signal is None:
        return False
    values = samples.get(node.primary_signal)
    if values is None or tick < 0 or tick >= len(values):
        return False
    current = values[tick]
    previous = values[tick - 1] if tick > 0 else None
    if node.predicate_kind == "rise":
        return previous == "0" and current == "1"
    if node.predicate_kind == "fall":
        return previous == "1" and current == "0"
    if node.predicate_kind == "high":
        return current == "1"
    if node.predicate_kind == "low":
        return current == "0"
    return False


def _sample_at(samples: dict[str, tuple[str, ...]], signal: str | None, tick: int) -> str | None:
    if signal is None:
        return None
    values = samples.get(signal)
    if values is None or tick < 0 or tick >= len(values):
        return None
    return values[tick]


def _constraint_ranges(
    constraint,
    anchor_ticks: dict[str, int],
    components: ScenarioComponents,
    total_ticks: int,
) -> list[tuple[int, int]]:
    if constraint.region == ConstraintRegion.FROM_UNTIL:
        start = anchor_ticks.get(constraint.start_anchor)
        end = anchor_ticks.get(constraint.end_anchor)
        if start is None or end is None:
            raise GenerationError(f"constraint '{constraint.name}' references unassigned anchors")
        return [(max(0, start), min(total_ticks - 1, end))]
    if constraint.region == ConstraintRegion.BEFORE:
        end = anchor_ticks.get(constraint.anchor)
        if end is None:
            raise GenerationError(f"constraint '{constraint.name}' references an unassigned anchor")
        return [(0, min(total_ticks - 1, end - 1))] if end > 0 else []
    if constraint.region == ConstraintRegion.AFTER:
        start = anchor_ticks.get(constraint.anchor)
        if start is None:
            raise GenerationError(f"constraint '{constraint.name}' references an unassigned anchor")
        return [(max(0, start), total_ticks - 1)]
    if constraint.region == ConstraintRegion.AT:
        tick = anchor_ticks.get(constraint.anchor)
        if tick is None:
            raise GenerationError(f"constraint '{constraint.name}' references an unassigned anchor")
        return [(tick, tick)]
    if constraint.region == ConstraintRegion.IN:
        window = next((window for window in components.windows if window.name == constraint.window), None)
        if window is None:
            raise GenerationError(f"constraint '{constraint.name}' references unknown window '{constraint.window}'")
        start = anchor_ticks.get(window.start_anchor)
        end = anchor_ticks.get(window.end_anchor)
        if start is None or end is None:
            raise GenerationError(f"constraint '{constraint.name}' references a window with unassigned anchors")
        return [(max(0, start), min(total_ticks - 1, end))]
    return []


def _constraint_holds(
    relation: str,
    value: str | None,
    samples: tuple[str, ...],
    start: int,
    end: int,
) -> bool:
    if end < start:
        return True
    region = samples[start : end + 1]
    if relation == "high":
        return all(sample == "1" for sample in region)
    if relation == "low":
        return all(sample == "0" for sample in region)
    if relation == "stable":
        return len(set(region)) <= 1
    if relation == "eq":
        return value is not None and all(sample == value for sample in region)
    if relation == "neq":
        return value is not None and all(sample != value for sample in region)
    return True


def _add_bit_distractors(
    timeline: list[str],
    signal_name: str,
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
    rng: random.Random,
) -> None:
    constrained = bit_must_high.get(signal_name, set()) | bit_must_low.get(signal_name, set())
    if constrained:
        return
    if rng.random() > 0.4:
        return
    if not timeline:
        return
    pulse = rng.randint(0, len(timeline) - 1)
    timeline[pulse] = "1"


def _pick_bus_value(signal_name: str, rng: random.Random) -> str:
    return rng.choice(_HEX_TOKENS)


def attach_samples(components: ScenarioComponents, samples: dict[str, tuple[str, ...]]) -> ScenarioComponents:
    """Return a copy of components with concrete samples attached to each lane."""

    new_signals: list[SignalDecl] = []
    for signal in components.signals:
        sample = samples.get(signal.name, ())
        new_signals.append(replace(signal, samples=sample))
    new_components = ScenarioComponents(
        name=components.name,
        clock_signal=components.clock_signal,
        params=list(components.params),
        signals=new_signals,
        anchors=list(components.anchors),
        windows=list(components.windows),
        cuts=list(components.cuts),
        lane_constraints=list(components.lane_constraints),
        anchor_node_map=dict(components.anchor_node_map),
        edges=list(components.edges),
    )
    return new_components
