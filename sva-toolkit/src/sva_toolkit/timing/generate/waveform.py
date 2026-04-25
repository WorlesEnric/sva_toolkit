"""Tick assignment and waveform synthesis for generated scenarios."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    SignalDecl,
    SignalKind,
    TimeBound,
    WindowBoundKind,
)
from sva_toolkit.timing.generate.model import (
    EventNode,
    GenerationError,
    GenerationSpec,
    ScenarioComponents,
    TemporalEdge,
)
from sva_toolkit.timing.generate.names import VALUE_TOKENS


_HEX_TOKENS = VALUE_TOKENS


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
        raise GenerationError("graph has no root anchor", reason="tick_budget_exceeded")

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
            raise GenerationError("graph contains a cycle or unreachable nodes", reason="tick_budget_exceeded")

    if pending:
        raise GenerationError("failed to fully order graph", reason="tick_budget_exceeded")

    max_tick = max(ticks.values())
    total_ticks = spec.tick_budget
    if max_tick >= total_ticks:
        raise GenerationError(
            f"assigned anchor tick {max_tick} exceeds tick budget {spec.tick_budget}",
            reason="tick_budget_exceeded",
        )
    _validate_assigned_window_bounds(components, ticks)
    return ticks, total_ticks


def _validate_assigned_window_bounds(components: ScenarioComponents, anchor_ticks: dict[str, int]) -> None:
    for window in components.windows:
        start = anchor_ticks.get(window.start_anchor)
        end = anchor_ticks.get(window.end_anchor)
        if start is None or end is None:
            raise GenerationError(
                f"window '{window.name}' references unassigned anchors",
                reason="window_bound_inconsistent",
            )
        delta = end - start
        if delta <= 0 or not _window_bound_allows(window.bound, delta):
            raise GenerationError(
                f"window '{window.name}' actual delay {delta} violates bound {window.bound.label}",
                reason="window_bound_inconsistent",
            )


def _window_bound_allows(bound: TimeBound, delay: int) -> bool:
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


def synthesize_waveforms(
    components: ScenarioComponents,
    anchor_ticks: dict[str, int],
    total_ticks: int,
    spec: GenerationSpec,
    rng: random.Random,
) -> dict[str, tuple[str, ...]]:
    """Build per-signal sample tuples that satisfy anchors and constraints."""

    signal_kinds = {signal.name: signal.kind for signal in components.signals}
    bit_must_high: dict[str, set[int]] = defaultdict(set)
    bit_must_low: dict[str, set[int]] = defaultdict(set)
    bus_must_values: dict[str, dict[int, str]] = defaultdict(dict)
    anchor_signal_names = _anchor_signal_names(components)

    for node in components.anchor_node_map.values():
        if node.id not in anchor_ticks:
            continue
        tick = anchor_ticks[node.id]
        if tick < 0 or tick >= total_ticks:
            raise GenerationError(
                f"anchor '{node.id}' tick {tick} out of range",
                reason="anchor_predicate_unsatisfied",
            )
        _require_anchor_first_occurrence(
            node,
            tick,
            total_ticks,
            signal_kinds,
            bit_must_high,
            bit_must_low,
            bus_must_values,
            rng,
        )
        kind = node.predicate_kind
        if kind == "rise":
            sig = node.primary_signal
            if sig is None:
                continue
            _require_bit_event(bit_must_high, bit_must_low, sig, tick, "rise", total_ticks)
        elif kind == "fall":
            sig = node.primary_signal
            if sig is None:
                continue
            _require_bit_event(bit_must_high, bit_must_low, sig, tick, "fall", total_ticks)
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
        elif kind == "all_high_eq":
            for sig in (node.primary_signal, *node.extra_signals):
                if sig:
                    bit_must_high[sig].add(tick)
            if node.bus_signal and node.eq_value:
                _require_bus_value(bus_must_values, node.bus_signal, tick, node.eq_value, total_ticks)
        elif kind == "change":
            sig = node.primary_signal
            if sig is None:
                continue
            if signal_kinds.get(sig) == SignalKind.BUS:
                _require_bus_change(bus_must_values, sig, tick, total_ticks, rng)
            else:
                _require_bit_event(bit_must_high, bit_must_low, sig, tick, "change", total_ticks)
        elif kind == "stable":
            sig = node.primary_signal
            if sig is None:
                continue
            if signal_kinds.get(sig) == SignalKind.BUS:
                _require_bus_stable_point(bus_must_values, sig, tick, total_ticks, rng)
            else:
                _require_bit_stable_point(bit_must_high, bit_must_low, sig, tick, total_ticks, rng)
        elif kind == "eq":
            if node.primary_signal and node.eq_value:
                _require_bus_value(bus_must_values, node.primary_signal, tick, node.eq_value, total_ticks)
        elif kind == "neq":
            if node.primary_signal and node.eq_value:
                _require_bus_neq(bus_must_values, node.primary_signal, tick, node.eq_value, total_ticks, rng)

    for constraint in components.lane_constraints:
        ranges = _constraint_ranges(constraint, anchor_ticks, components, total_ticks)
        for start, end in ranges:
            if end < start:
                continue
            for sig in constraint.signals:
                signal_kind = signal_kinds.get(sig, SignalKind.BIT)
                if constraint.relation == "high":
                    for tt in range(start, end + 1):
                        bit_must_high[sig].add(tt)
                elif constraint.relation == "low":
                    for tt in range(start, end + 1):
                        bit_must_low[sig].add(tt)
                elif constraint.relation == "stable":
                    if signal_kind == SignalKind.BUS:
                        value = _stable_bus_value(bus_must_values, sig, start, end, rng)
                        for tt in range(start, end + 1):
                            _require_bus_value(bus_must_values, sig, tt, value, total_ticks)
                    else:
                        value = _stable_bit_value(sig, start, end, bit_must_high, bit_must_low, rng)
                        target = bit_must_high if value == "1" else bit_must_low
                        for tt in range(start, end + 1):
                            target[sig].add(tt)
                elif constraint.relation == "eq" and constraint.value is not None:
                    if signal_kind == SignalKind.BUS:
                        for tt in range(start, end + 1):
                            _require_bus_value(bus_must_values, sig, tt, constraint.value, total_ticks)
                elif constraint.relation == "neq" and constraint.value is not None:
                    if signal_kind == SignalKind.BUS:
                        for tt in range(start, end + 1):
                            _require_bus_neq(bus_must_values, sig, tt, constraint.value, total_ticks, rng)
                elif constraint.relation in {"rise", "fall"}:
                    event_tick = _event_tick_in_range(start, end, total_ticks)
                    if event_tick is not None:
                        _require_bit_event(
                            bit_must_high,
                            bit_must_low,
                            sig,
                            event_tick,
                            constraint.relation,
                            total_ticks,
                        )
                elif constraint.relation == "change":
                    event_tick = _event_tick_in_range(start, end, total_ticks)
                    if event_tick is not None:
                        if signal_kind == SignalKind.BUS:
                            _require_bus_change(bus_must_values, sig, event_tick, total_ticks, rng)
                        else:
                            _require_bit_event(bit_must_high, bit_must_low, sig, event_tick, "change", total_ticks)

    for sig, highs in bit_must_high.items():
        clash = highs & bit_must_low.get(sig, set())
        if clash:
            raise GenerationError(
                f"bit signal '{sig}' has conflicting high/low requirements at {sorted(clash)}",
                reason="lane_constraint_unsatisfied",
            )

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
            _add_bit_distractors(timeline, signal.name, bit_must_high, bit_must_low, anchor_signal_names, rng)
            samples[signal.name] = tuple(timeline)
        else:
            timeline = ["x"] * total_ticks
            assigned = bus_must_values.get(signal.name, {})
            for tt, value in sorted(assigned.items()):
                if 0 <= tt < total_ticks:
                    timeline[tt] = value
            _add_bus_distractor(timeline, signal.name, assigned, anchor_signal_names, rng)
            samples[signal.name] = tuple(timeline)

    _verify_waveform_semantics(components, anchor_ticks, samples, total_ticks)
    return samples


def _anchor_signal_names(components: ScenarioComponents) -> set[str]:
    names: set[str] = set()
    for node in components.anchor_node_map.values():
        if node.primary_signal:
            names.add(node.primary_signal)
        names.update(signal for signal in node.extra_signals if signal)
        if node.bus_signal:
            names.add(node.bus_signal)
    return names


def _require_anchor_first_occurrence(
    node: EventNode,
    tick: int,
    total_ticks: int,
    signal_kinds: dict[str, SignalKind],
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
    bus_must_values: dict[str, dict[int, str]],
    rng: random.Random,
) -> None:
    kind = node.predicate_kind

    def require_bit_prefix(signal: str | None, value: str) -> None:
        if signal is None:
            return
        for tt in range(tick):
            _require_bit_value(bit_must_high, bit_must_low, signal, tt, value, total_ticks)

    def require_bit_value(signal: str | None, tt: int, value: str) -> None:
        if signal is not None:
            _require_bit_value(bit_must_high, bit_must_low, signal, tt, value, total_ticks)

    def require_bus_prefix(signal: str | None, value: str) -> None:
        if signal is None:
            return
        for tt in range(tick):
            _require_bus_value(bus_must_values, signal, tt, value, total_ticks)

    if kind in {"rise", "high"}:
        require_bit_prefix(node.primary_signal, "0")
        require_bit_value(node.primary_signal, tick, "1")
        return
    if kind in {"fall", "low"}:
        require_bit_prefix(node.primary_signal, "1")
        require_bit_value(node.primary_signal, tick, "0")
        return
    if kind == "all_high":
        for signal in (node.primary_signal, *node.extra_signals):
            require_bit_prefix(signal, "0")
            require_bit_value(signal, tick, "1")
        return
    if kind == "all_high_eq":
        for signal in (node.primary_signal, *node.extra_signals):
            require_bit_prefix(signal, "0")
            require_bit_value(signal, tick, "1")
        if node.bus_signal and node.eq_value:
            require_bus_prefix(node.bus_signal, _pick_bus_value_excluding({node.eq_value}, rng))
            _require_bus_value(bus_must_values, node.bus_signal, tick, node.eq_value, total_ticks)
        return
    if kind == "change" and node.primary_signal:
        if signal_kinds.get(node.primary_signal) == SignalKind.BUS:
            previous = _pick_bus_value(node.primary_signal, rng)
            require_bus_prefix(node.primary_signal, previous)
            _require_bus_value(
                bus_must_values,
                node.primary_signal,
                tick,
                _pick_bus_value_excluding({previous}, rng),
                total_ticks,
            )
        else:
            require_bit_prefix(node.primary_signal, "0")
            require_bit_value(node.primary_signal, tick, "1")
        return
    if kind == "stable" and node.primary_signal:
        if tick <= 0:
            return
        if signal_kinds.get(node.primary_signal) == SignalKind.BUS:
            first = _pick_bus_value(node.primary_signal, rng)
            second = _pick_bus_value_excluding({first}, rng)
            for tt in range(tick):
                _require_bus_value(
                    bus_must_values,
                    node.primary_signal,
                    tt,
                    first if tt % 2 == 0 else second,
                    total_ticks,
                )
            _require_bus_value(
                bus_must_values,
                node.primary_signal,
                tick,
                first if (tick - 1) % 2 == 0 else second,
                total_ticks,
            )
        else:
            for tt in range(tick):
                require_bit_value(node.primary_signal, tt, "0" if tt % 2 == 0 else "1")
            require_bit_value(node.primary_signal, tick, "0" if (tick - 1) % 2 == 0 else "1")
        return
    if kind == "eq" and node.primary_signal and node.eq_value:
        require_bus_prefix(node.primary_signal, _pick_bus_value_excluding({node.eq_value}, rng))
        _require_bus_value(bus_must_values, node.primary_signal, tick, node.eq_value, total_ticks)
        return
    if kind == "neq" and node.primary_signal and node.eq_value:
        require_bus_prefix(node.primary_signal, node.eq_value)
        _require_bus_value(
            bus_must_values,
            node.primary_signal,
            tick,
            _pick_bus_value_excluding({node.eq_value}, rng),
            total_ticks,
        )


def _require_bit_value(
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
    signal: str,
    tick: int,
    value: str,
    total_ticks: int,
) -> None:
    if tick < 0 or tick >= total_ticks:
        return
    if value == "1":
        bit_must_high[signal].add(tick)
    else:
        bit_must_low[signal].add(tick)


def _known_bit_value(
    signal: str,
    tick: int,
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
) -> str | None:
    high = tick in bit_must_high.get(signal, set())
    low = tick in bit_must_low.get(signal, set())
    if high and low:
        raise GenerationError(
            f"bit signal '{signal}' has conflicting requirements at tick {tick}",
            reason="lane_constraint_unsatisfied",
        )
    if high:
        return "1"
    if low:
        return "0"
    return None


def _require_bit_event(
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
    signal: str,
    tick: int,
    relation: str,
    total_ticks: int,
) -> None:
    if tick <= 0:
        raise GenerationError(
            f"bit event '{relation}' on '{signal}' cannot be placed at tick {tick}",
            reason="anchor_predicate_unsatisfied",
        )
    if relation == "rise":
        _require_bit_value(bit_must_high, bit_must_low, signal, tick - 1, "0", total_ticks)
        _require_bit_value(bit_must_high, bit_must_low, signal, tick, "1", total_ticks)
        return
    if relation == "fall":
        _require_bit_value(bit_must_high, bit_must_low, signal, tick - 1, "1", total_ticks)
        _require_bit_value(bit_must_high, bit_must_low, signal, tick, "0", total_ticks)
        return
    previous = _known_bit_value(signal, tick - 1, bit_must_high, bit_must_low)
    current = _known_bit_value(signal, tick, bit_must_high, bit_must_low)
    if previous is not None and current is not None:
        if previous == current:
            raise GenerationError(
                f"bit signal '{signal}' cannot change at tick {tick}",
                reason="anchor_predicate_unsatisfied",
            )
        return
    if previous is not None:
        _require_bit_value(bit_must_high, bit_must_low, signal, tick, "0" if previous == "1" else "1", total_ticks)
        return
    if current is not None:
        _require_bit_value(bit_must_high, bit_must_low, signal, tick - 1, "0" if current == "1" else "1", total_ticks)
        return
    _require_bit_value(bit_must_high, bit_must_low, signal, tick - 1, "0", total_ticks)
    _require_bit_value(bit_must_high, bit_must_low, signal, tick, "1", total_ticks)


def _require_bit_stable_point(
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
    signal: str,
    tick: int,
    total_ticks: int,
    rng: random.Random,
) -> None:
    if tick <= 0:
        raise GenerationError(
            f"bit stable predicate on '{signal}' cannot be placed at tick {tick}",
            reason="anchor_predicate_unsatisfied",
        )
    previous = _known_bit_value(signal, tick - 1, bit_must_high, bit_must_low)
    current = _known_bit_value(signal, tick, bit_must_high, bit_must_low)
    if previous is not None and current is not None and previous != current:
        raise GenerationError(
            f"bit signal '{signal}' cannot be stable at tick {tick}",
            reason="anchor_predicate_unsatisfied",
        )
    value = previous or current or rng.choice(("0", "1"))
    _require_bit_value(bit_must_high, bit_must_low, signal, tick - 1, value, total_ticks)
    _require_bit_value(bit_must_high, bit_must_low, signal, tick, value, total_ticks)


def _require_bus_value(
    bus_must_values: dict[str, dict[int, str]],
    signal: str,
    tick: int,
    value: str,
    total_ticks: int,
) -> None:
    if tick < 0 or tick >= total_ticks:
        return
    assigned = bus_must_values[signal]
    existing = assigned.get(tick)
    if existing is not None and existing != value:
        raise GenerationError(
            f"bus signal '{signal}' has conflicting values at tick {tick}: {existing} vs {value}",
            reason="bus_value_conflict",
        )
    assigned[tick] = value


def _require_bus_neq(
    bus_must_values: dict[str, dict[int, str]],
    signal: str,
    tick: int,
    forbidden: str,
    total_ticks: int,
    rng: random.Random,
) -> None:
    if tick < 0 or tick >= total_ticks:
        return
    existing = bus_must_values[signal].get(tick)
    if existing is not None:
        if existing == forbidden:
            raise GenerationError(
                f"bus signal '{signal}' is forced to forbidden value {forbidden} at tick {tick}",
                reason="bus_value_conflict",
            )
        return
    _require_bus_value(
        bus_must_values,
        signal,
        tick,
        _pick_bus_value_excluding({forbidden}, rng),
        total_ticks,
    )


def _require_bus_change(
    bus_must_values: dict[str, dict[int, str]],
    signal: str,
    tick: int,
    total_ticks: int,
    rng: random.Random,
) -> None:
    if tick <= 0:
        raise GenerationError(
            f"bus change predicate on '{signal}' cannot be placed at tick {tick}",
            reason="anchor_predicate_unsatisfied",
        )
    assigned = bus_must_values[signal]
    previous = assigned.get(tick - 1)
    current = assigned.get(tick)
    if previous is not None and current is not None:
        if previous == current:
            raise GenerationError(
                f"bus signal '{signal}' cannot change at tick {tick}",
                reason="anchor_predicate_unsatisfied",
            )
        return
    if previous is not None:
        current = _pick_bus_value_excluding({previous}, rng)
    elif current is not None:
        previous = _pick_bus_value_excluding({current}, rng)
    else:
        previous = _pick_bus_value(signal, rng)
        current = _pick_bus_value_excluding({previous}, rng)
    _require_bus_value(bus_must_values, signal, tick - 1, previous, total_ticks)
    _require_bus_value(bus_must_values, signal, tick, current, total_ticks)


def _require_bus_stable_point(
    bus_must_values: dict[str, dict[int, str]],
    signal: str,
    tick: int,
    total_ticks: int,
    rng: random.Random,
) -> None:
    if tick <= 0:
        raise GenerationError(
            f"bus stable predicate on '{signal}' cannot be placed at tick {tick}",
            reason="anchor_predicate_unsatisfied",
        )
    assigned = bus_must_values[signal]
    previous = assigned.get(tick - 1)
    current = assigned.get(tick)
    if previous is not None and current is not None and previous != current:
        raise GenerationError(
            f"bus signal '{signal}' cannot be stable at tick {tick}",
            reason="anchor_predicate_unsatisfied",
        )
    value = previous or current or _pick_bus_value(signal, rng)
    _require_bus_value(bus_must_values, signal, tick - 1, value, total_ticks)
    _require_bus_value(bus_must_values, signal, tick, value, total_ticks)


def _stable_bus_value(
    bus_must_values: dict[str, dict[int, str]],
    signal: str,
    start: int,
    end: int,
    rng: random.Random,
) -> str:
    values = {value for tick, value in bus_must_values.get(signal, {}).items() if start <= tick <= end}
    if len(values) > 1:
        raise GenerationError(
            f"bus signal '{signal}' cannot be stable across conflicting values",
            reason="bus_value_conflict",
        )
    if values:
        return next(iter(values))
    return _pick_bus_value(signal, rng)


def _stable_bit_value(
    signal: str,
    start: int,
    end: int,
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
    rng: random.Random,
) -> str:
    values = {
        value
        for tick in range(start, end + 1)
        if (value := _known_bit_value(signal, tick, bit_must_high, bit_must_low)) is not None
    }
    if len(values) > 1:
        raise GenerationError(
            f"bit signal '{signal}' cannot be stable across conflicting values",
            reason="lane_constraint_unsatisfied",
        )
    if values:
        return next(iter(values))
    return rng.choice(("0", "1"))


def _event_tick_in_range(start: int, end: int, total_ticks: int) -> int | None:
    if end < start:
        return None
    if start > 0:
        return min(start, total_ticks - 1)
    if end >= 1:
        return 1
    return None


def _verify_waveform_semantics(
    components: ScenarioComponents,
    anchor_ticks: dict[str, int],
    samples: dict[str, tuple[str, ...]],
    total_ticks: int,
) -> None:
    for node in components.anchor_node_map.values():
        assigned = anchor_ticks.get(node.id)
        if assigned is None:
            raise GenerationError(
                f"anchor '{node.id}' has no assigned tick",
                reason="anchor_predicate_unsatisfied",
            )
        first = _first_predicate_tick(node, samples, total_ticks)
        if first is None:
            raise GenerationError(
                f"anchor '{node.id}' predicate holds at no tick",
                reason="anchor_predicate_unsatisfied",
            )
        if first != assigned:
            raise GenerationError(
                f"anchor '{node.id}' assigned tick {assigned} but first match at {first}",
                reason="anchor_first_occurrence_drift",
            )

    first_occurrences = _first_occurrence_map(components, samples, total_ticks)
    for constraint in components.lane_constraints:
        ranges = _constraint_ranges(constraint, first_occurrences, components, total_ticks)
        for start, end in ranges:
            for signal_name in constraint.signals:
                signal_samples = samples.get(signal_name)
                if signal_samples is None:
                    raise GenerationError(
                        f"constraint '{constraint.name}' references unsampled lane '{signal_name}'",
                        reason="lane_constraint_unsatisfied",
                    )
                if not _constraint_holds(constraint.relation, constraint.value, signal_samples, start, end):
                    raise GenerationError(
                        f"constraint '{constraint.name}' relation '{constraint.relation}' "
                        f"does not hold on lane '{signal_name}' from tick {start} to {end}",
                        reason="lane_constraint_unsatisfied",
                    )


def _first_predicate_tick(node: EventNode, samples: dict[str, tuple[str, ...]], total_ticks: int) -> int | None:
    for tick in range(total_ticks):
        if _node_predicate_holds(node, samples, tick):
            return tick
    return None


def _first_occurrence_map(
    components: ScenarioComponents,
    samples: dict[str, tuple[str, ...]],
    total_ticks: int,
) -> dict[str, int | None]:
    return {
        node.id: _first_predicate_tick(node, samples, total_ticks)
        for node in components.anchor_node_map.values()
    }


def _node_predicate_holds(node: EventNode, samples: dict[str, tuple[str, ...]], tick: int) -> bool:
    if node.predicate_kind == "all_high":
        signals = (node.primary_signal, *node.extra_signals)
        return all(_sample_at(samples, signal, tick) == "1" for signal in signals if signal)
    if node.predicate_kind == "all_high_eq":
        signals = (node.primary_signal, *node.extra_signals)
        bits_hold = all(_sample_at(samples, signal, tick) == "1" for signal in signals if signal)
        return bits_hold and node.bus_signal is not None and _sample_at(samples, node.bus_signal, tick) == node.eq_value
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
    if node.predicate_kind == "change":
        return previous is not None and current != previous
    if node.predicate_kind == "stable":
        return previous is not None and current == previous
    if node.predicate_kind == "eq":
        return current == node.eq_value
    if node.predicate_kind == "neq":
        return node.eq_value is not None and current != node.eq_value
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
    anchor_ticks: dict[str, int | None],
    components: ScenarioComponents,
    total_ticks: int,
) -> list[tuple[int, int]]:
    if constraint.region == ConstraintRegion.FROM_UNTIL:
        start = anchor_ticks.get(constraint.start_anchor)
        end = anchor_ticks.get(constraint.end_anchor)
        if start is None or end is None:
            raise GenerationError(
                f"constraint '{constraint.name}' references unassigned anchors",
                reason="lane_constraint_unsatisfied",
            )
        return [(max(0, start), min(total_ticks - 1, end))]
    if constraint.region == ConstraintRegion.BEFORE:
        end = anchor_ticks.get(constraint.anchor)
        if end is None:
            raise GenerationError(
                f"constraint '{constraint.name}' references an unassigned anchor",
                reason="lane_constraint_unsatisfied",
            )
        return [(0, min(total_ticks - 1, end - 1))] if end > 0 else []
    if constraint.region == ConstraintRegion.AFTER:
        start = anchor_ticks.get(constraint.anchor)
        if start is None:
            raise GenerationError(
                f"constraint '{constraint.name}' references an unassigned anchor",
                reason="lane_constraint_unsatisfied",
            )
        return [(max(0, start), total_ticks - 1)]
    if constraint.region == ConstraintRegion.AT:
        tick = anchor_ticks.get(constraint.anchor)
        if tick is None:
            raise GenerationError(
                f"constraint '{constraint.name}' references an unassigned anchor",
                reason="lane_constraint_unsatisfied",
            )
        return [(tick, tick)]
    if constraint.region == ConstraintRegion.IN:
        window = next((window for window in components.windows if window.name == constraint.window), None)
        if window is None:
            raise GenerationError(
                f"constraint '{constraint.name}' references unknown window '{constraint.window}'",
                reason="lane_constraint_unsatisfied",
            )
        start = anchor_ticks.get(window.start_anchor)
        end = anchor_ticks.get(window.end_anchor)
        if start is None or end is None:
            raise GenerationError(
                f"constraint '{constraint.name}' references a window with unassigned anchors",
                reason="lane_constraint_unsatisfied",
            )
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
        if start == end and start > 0:
            return samples[start - 1] == samples[start]
        return len(set(region)) <= 1
    if relation == "eq":
        return value is not None and all(sample == value for sample in region)
    if relation == "neq":
        return value is not None and all(sample != value for sample in region)
    if relation == "rise":
        return _event_holds(samples, start, end, lambda previous, current: previous == "0" and current == "1")
    if relation == "fall":
        return _event_holds(samples, start, end, lambda previous, current: previous == "1" and current == "0")
    if relation == "change":
        return _event_holds(samples, start, end, lambda previous, current: previous is not None and current != previous)
    return True


def _event_holds(samples: tuple[str, ...], start: int, end: int, predicate) -> bool:
    lo = max(1, start)
    hi = min(end, len(samples) - 1)
    if hi < lo:
        return False
    return any(predicate(samples[tick - 1], samples[tick]) for tick in range(lo, hi + 1))


def _add_bit_distractors(
    timeline: list[str],
    signal_name: str,
    bit_must_high: dict[str, set[int]],
    bit_must_low: dict[str, set[int]],
    anchor_signal_names: set[str],
    rng: random.Random,
) -> None:
    if signal_name in anchor_signal_names:
        return
    constrained = bit_must_high.get(signal_name, set()) | bit_must_low.get(signal_name, set())
    probability = 0.3 if constrained else 0.4
    if rng.random() > probability:
        return
    if not timeline:
        return
    free_ticks = [tick for tick in range(len(timeline)) if tick not in constrained]
    if len(free_ticks) < 2:
        return
    pulse = rng.choice(free_ticks)
    timeline[pulse] = "0" if timeline[pulse] == "1" else "1"


def _add_bus_distractor(
    timeline: list[str],
    signal_name: str,
    assigned: dict[int, str],
    anchor_signal_names: set[str],
    rng: random.Random,
) -> None:
    if signal_name in anchor_signal_names:
        return
    if not assigned:
        return
    free_ticks = [tick for tick in range(len(timeline)) if tick not in assigned]
    if len(free_ticks) < 2:
        return
    tick = rng.choice(free_ticks)
    neighbor_values = {
        timeline[index]
        for index in (tick - 1, tick + 1)
        if 0 <= index < len(timeline) and timeline[index] != "x"
    }
    forbidden = set(assigned.values()) | neighbor_values
    timeline[tick] = _pick_bus_value_excluding(forbidden, rng)


def _pick_bus_value(signal_name: str, rng: random.Random) -> str:
    return rng.choice(_HEX_TOKENS)


def _pick_bus_value_excluding(forbidden: set[str], rng: random.Random) -> str:
    choices = [token for token in _HEX_TOKENS if token not in forbidden]
    if not choices:
        choices = list(_HEX_TOKENS)
    return rng.choice(choices)


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
        response_overlay_targets=list(components.response_overlay_targets),
    )
    return new_components
