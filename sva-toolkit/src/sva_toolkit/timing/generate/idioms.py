"""Apply idiom decorations to a topology to produce concrete scenario IR fragments."""

from __future__ import annotations

import random

from sva_toolkit.timing.core.conditions import all_of, predicate_condition
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    ConstraintRegion,
    Cut,
    CutMeaning,
    CutPlacement,
    LaneConstraint,
    ParameterDecl,
    SignalDecl,
    SignalKind,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
    normalize_signal_width,
)
from sva_toolkit.timing.generate.model import (
    DecoratedGraph,
    EventNode,
    GenerationError,
    GenerationSpec,
    ScenarioComponents,
    TemporalEdge,
)
from sva_toolkit.timing.generate.names import FLAVORS, NameAllocator, PARAM_NAMES_BY_FLAVOR


_ROLE_TO_ANCHOR = {
    "trigger": AnchorRole.TRIGGER,
    "response": AnchorRole.RESPONSE,
    "state": AnchorRole.STATE,
}


def apply_idioms(graph: DecoratedGraph, spec: GenerationSpec, rng: random.Random) -> ScenarioComponents:
    """Allocate signals and produce the scenario IR fragments for the given graph."""

    flavor = FLAVORS[graph.flavor]
    allocator = NameAllocator(flavor, rng)

    components = ScenarioComponents(name=_scenario_name(spec, rng), clock_signal=flavor.clock)
    decorated_nodes: dict[str, EventNode] = {}

    for node in graph.nodes:
        decorated, extra_lanes = _decorate_node(node, allocator, rng)
        decorated_nodes[node.id] = decorated
        for lane in extra_lanes:
            components.signals.append(lane)

    for node_id, node in decorated_nodes.items():
        bit_signals = _node_bit_signals(node)
        for signal_name in bit_signals:
            if not _has_signal(components, signal_name):
                components.signals.append(_make_bit_signal(signal_name))

    for node_id, node in decorated_nodes.items():
        components.anchors.append(_anchor_from_node(node))
        components.anchor_node_map[node.id] = node

    for edge in graph.edges:
        sampled_edge, window = _build_window(edge, decorated_nodes, spec, rng, allocator, components)
        components.windows.append(window)
        components.edges.append(sampled_edge)

    _add_idiom_constraints(graph, components, decorated_nodes, spec, rng, allocator)

    if spec.cuts_enabled:
        _add_cuts(graph, components, decorated_nodes, rng)

    _add_distractor_lanes(components, allocator, spec)

    return components


def _scenario_name(spec: GenerationSpec, rng: random.Random) -> str:
    base = f"td_{spec.topology}_{spec.flavor}_{spec.item_id}"
    return base.replace("-", "_")


def _decorate_node(
    node: EventNode, allocator: NameAllocator, rng: random.Random
) -> tuple[EventNode, list[SignalDecl]]:
    extra_lanes: list[SignalDecl] = []

    if node.predicate_kind in {"rise", "fall", "high", "low"}:
        primary = allocator.take_bit(_pick_signal_hint(node.id, allocator))
        return (
            EventNode(
                id=node.id,
                role=node.role,
                predicate_kind=node.predicate_kind,
                primary_signal=primary,
            ),
            extra_lanes,
        )

    if node.predicate_kind == "all_high":
        primary = allocator.take_bit(_pick_signal_hint(node.id, allocator))
        extras: list[str] = []
        extras.append(allocator.take_bit())
        if node.id in {"last_beat"}:
            extras.append(allocator.take_bit())
        return (
            EventNode(
                id=node.id,
                role=node.role,
                predicate_kind=node.predicate_kind,
                primary_signal=primary,
                extra_signals=tuple(extras),
            ),
            extra_lanes,
        )

    primary = allocator.take_bit()
    return (
        EventNode(
            id=node.id,
            role=node.role,
            predicate_kind="rise",
            primary_signal=primary,
        ),
        extra_lanes,
    )


def _pick_signal_hint(node_id: str, allocator: NameAllocator) -> str | None:
    flavor = allocator.flavor
    if flavor.name == "axi_like":
        if node_id in {"trigger", "valid_rise", "first_beat", "a_start"}:
            for hint in ("AWVALID", "WVALID", "ARVALID"):
                if hint in allocator._available_bits:
                    return hint
        if node_id in {"response", "handshake", "last_beat"}:
            for hint in ("AWREADY", "WREADY", "ARREADY"):
                if hint in allocator._available_bits:
                    return hint
    return None


def _node_bit_signals(node: EventNode) -> tuple[str, ...]:
    if node.primary_signal is None:
        return ()
    return (node.primary_signal, *node.extra_signals)


def _has_signal(components: ScenarioComponents, name: str) -> bool:
    return any(signal.name == name for signal in components.signals)


def _make_bit_signal(name: str) -> SignalDecl:
    return SignalDecl(
        name=name,
        kind=SignalKind.BIT,
        width=normalize_signal_width(SignalKind.BIT, None),
    )


def _make_bus_signal(name: str, width: str) -> SignalDecl:
    return SignalDecl(name=name, kind=SignalKind.BUS, width=width)


def _anchor_from_node(node: EventNode) -> Anchor:
    role = _ROLE_TO_ANCHOR.get(node.role, AnchorRole.STATE)
    if node.predicate_kind == "all_high":
        signals = (node.primary_signal, *node.extra_signals)
        condition = all_of(predicate_condition("high", signal) for signal in signals if signal)
    else:
        if node.primary_signal is None:
            raise GenerationError(f"node '{node.id}' has no primary signal")
        condition = predicate_condition(node.predicate_kind, node.primary_signal)
    return Anchor(name=node.id, condition=condition, role=role)


def _build_window(
    edge: TemporalEdge,
    nodes: dict[str, EventNode],
    spec: GenerationSpec,
    rng: random.Random,
    allocator: NameAllocator,
    components: ScenarioComponents,
) -> tuple[TemporalEdge, TimeWindow]:
    bound_kind = rng.choice(spec.bound_kinds) if spec.bound_kinds else "range"

    if bound_kind == "exact":
        delay = rng.randint(1, 3)
        bound = TimeBound(kind=WindowBoundKind.EXACT, min_delay=str(delay), max_delay=str(delay))
        sampled = delay
    elif bound_kind == "parameterized":
        param_pool = PARAM_NAMES_BY_FLAVOR.get(allocator.flavor.name, ("MAX_LAT",))
        param_name = allocator.take_param(param_pool)
        if not _has_param(components, param_name):
            components.params.append(ParameterDecl(name=param_name, kind="int"))
        delay = rng.randint(1, 3)
        bound = TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay=param_name)
        sampled = delay
    elif bound_kind == "unbounded":
        delay = rng.randint(1, 3)
        bound = TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay="1", max_delay="$")
        sampled = delay
    else:
        lo = rng.randint(1, 2)
        hi = lo + rng.randint(1, 3)
        bound = TimeBound(kind=WindowBoundKind.RANGE, min_delay=str(lo), max_delay=str(hi))
        sampled = rng.randint(lo, hi)

    window = TimeWindow(name=edge.id, start_anchor=edge.start, end_anchor=edge.end, bound=bound)
    new_edge = TemporalEdge(
        id=edge.id,
        start=edge.start,
        end=edge.end,
        bound_kind=bound_kind,
        min_delay=int(bound.min_delay) if (bound.min_delay or "").isdigit() else 1,
        max_delay=sampled,
        parameter=bound.max_delay if bound_kind == "parameterized" else None,
        sampled_delay=sampled,
    )
    return new_edge, window


def _has_param(components: ScenarioComponents, name: str) -> bool:
    return any(p.name == name for p in components.params)


def _add_idiom_constraints(
    graph: DecoratedGraph,
    components: ScenarioComponents,
    nodes: dict[str, EventNode],
    spec: GenerationSpec,
    rng: random.Random,
    allocator: NameAllocator,
) -> None:
    counter = [0]

    def add(constraint: LaneConstraint) -> None:
        components.lane_constraints.append(constraint)
        counter[0] += 1

    if "hold_until" in spec.idioms:
        for edge in components.edges:
            start_node = nodes[edge.start]
            end_node = nodes[edge.end]
            if start_node.predicate_kind == "rise" and start_node.primary_signal:
                add(
                    LaneConstraint(
                        name=f"show_{counter[0]}_0",
                        signals=(start_node.primary_signal,),
                        relation="high",
                        region=ConstraintRegion.FROM_UNTIL,
                        start_anchor=start_node.id,
                        end_anchor=end_node.id,
                    )
                )

    if "stable_while" in spec.idioms:
        for edge in components.edges[:1]:
            bus_name = allocator.take_bus()
            if not _has_signal(components, bus_name):
                components.signals.append(_make_bus_signal(bus_name, allocator.take_bus_width()))
            add(
                LaneConstraint(
                    name=f"show_{counter[0]}_0",
                    signals=(bus_name,),
                    relation="stable",
                    region=ConstraintRegion.FROM_UNTIL,
                    start_anchor=edge.start,
                    end_anchor=edge.end,
                )
            )

    if "not_before" in spec.idioms:
        target_node = _find_response_node(graph, nodes)
        gate_node = _find_trigger_node(graph, nodes)
        if target_node and gate_node and target_node.primary_signal:
            add(
                LaneConstraint(
                    name=f"show_{counter[0]}_0",
                    signals=(target_node.primary_signal,),
                    relation="low",
                    region=ConstraintRegion.BEFORE,
                    anchor=gate_node.id,
                )
            )


def _find_trigger_node(graph: DecoratedGraph, nodes: dict[str, EventNode]) -> EventNode | None:
    for node in nodes.values():
        if node.role == "trigger":
            return node
    return None


def _find_response_node(graph: DecoratedGraph, nodes: dict[str, EventNode]) -> EventNode | None:
    for node in nodes.values():
        if node.role == "response":
            return node
    return None


def _add_cuts(
    graph: DecoratedGraph,
    components: ScenarioComponents,
    nodes: dict[str, EventNode],
    rng: random.Random,
) -> None:
    placements = []
    trigger = _find_trigger_node(graph, nodes)
    response = _find_response_node(graph, nodes)
    if trigger:
        placements.append(("before", trigger.id))
    if response:
        placements.append(("after", response.id))

    for index, (placement, anchor_name) in enumerate(placements):
        cut_name = f"cut_{placement}_{index}"
        meaning = CutMeaning.OMITTED_HISTORY if placement == "before" else CutMeaning.OMITTED_FUTURE
        cut_placement = CutPlacement.BEFORE_ANCHOR if placement == "before" else CutPlacement.AFTER_ANCHOR
        label = "idle" if placement == "before" else "next transaction"
        components.cuts.append(
            Cut(
                name=cut_name,
                placement=cut_placement,
                meaning=meaning,
                anchor=anchor_name,
                label=label,
            )
        )


def _add_distractor_lanes(
    components: ScenarioComponents, allocator: NameAllocator, spec: GenerationSpec
) -> None:
    for _ in range(spec.distractor_lanes):
        try:
            name = allocator.take_bit()
        except Exception:
            break
        if not _has_signal(components, name):
            components.signals.append(_make_bit_signal(name))
