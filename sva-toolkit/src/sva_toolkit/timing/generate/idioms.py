"""Apply idiom decorations to a topology to produce concrete scenario IR fragments."""

from __future__ import annotations

import random
from dataclasses import replace

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
from sva_toolkit.timing.generate.names import FLAVORS, NameAllocator, PARAM_NAMES_BY_FLAVOR, VALUE_TOKENS


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

    edge_count = max(1, len(graph.edges))
    incoming_counts: dict[str, int] = {}
    for edge in graph.edges:
        incoming_counts[edge.end] = incoming_counts.get(edge.end, 0) + 1
    for edge in graph.edges:
        sampled_edge, window = _build_window(
            edge,
            decorated_nodes,
            spec,
            rng,
            allocator,
            components,
            edge_count=edge_count,
            incoming_counts=incoming_counts,
        )
        components.windows.append(window)
        components.edges.append(sampled_edge)

    if "response" in spec.idioms:
        components.response_overlay_targets.extend(window.name for window in components.windows)

    _add_idiom_constraints(graph, components, decorated_nodes, spec, rng, allocator)

    if spec.cuts_enabled:
        _add_cuts(graph, components, decorated_nodes, spec, rng)

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

    if node.predicate_kind == "all_high_eq":
        primary = allocator.take_bit(_pick_signal_hint(node.id, allocator))
        extras = (allocator.take_bit(),)
        bus_name = allocator.take_bus()
        extra_lanes.append(_make_bus_signal(bus_name, allocator.take_bus_width()))
        return (
            EventNode(
                id=node.id,
                role=node.role,
                predicate_kind=node.predicate_kind,
                primary_signal=primary,
                extra_signals=extras,
                bus_signal=bus_name,
                eq_value=rng.choice(VALUE_TOKENS),
            ),
            extra_lanes,
        )

    if node.predicate_kind in {"change", "stable", "eq", "neq"}:
        bus_name = allocator.take_bus()
        extra_lanes.append(_make_bus_signal(bus_name, allocator.take_bus_width()))
        return (
            EventNode(
                id=node.id,
                role=node.role,
                predicate_kind=node.predicate_kind,
                primary_signal=bus_name,
                eq_value=rng.choice(VALUE_TOKENS) if node.predicate_kind in {"eq", "neq"} else None,
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
    if node.predicate_kind in {"change", "stable", "eq", "neq"}:
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
    elif node.predicate_kind == "all_high_eq":
        predicates = [
            predicate_condition("high", signal)
            for signal in (node.primary_signal, *node.extra_signals)
            if signal
        ]
        if node.bus_signal is None or node.eq_value is None:
            raise GenerationError(
                f"node '{node.id}' has incomplete equality conjunction metadata",
                reason="anchor_predicate_unsatisfied",
            )
        predicates.append(predicate_condition("eq", node.bus_signal, value=node.eq_value))
        condition = all_of(predicates)
    else:
        if node.primary_signal is None:
            raise GenerationError(
                f"node '{node.id}' has no primary signal",
                reason="anchor_predicate_unsatisfied",
            )
        if node.predicate_kind in {"eq", "neq"}:
            condition = predicate_condition(node.predicate_kind, node.primary_signal, value=node.eq_value)
        else:
            condition = predicate_condition(node.predicate_kind, node.primary_signal)
    return Anchor(name=node.id, condition=condition, role=role)


def _sample_window_bound_kind(
    edge: TemporalEdge,
    spec: GenerationSpec,
    rng: random.Random,
    incoming_counts: dict[str, int],
) -> str:
    allowed = tuple(spec.bound_kinds) if spec.bound_kinds else ("range",)
    if len(allowed) == 1:
        return allowed[0]

    participates_in_join = incoming_counts.get(edge.end, 0) > 1
    participates_in_chain = spec.topology in {"chain", "setup_hold"}
    weights: list[float] = []
    for kind in allowed:
        if kind == "range":
            weight = 8.0 if participates_in_join else 4.0 if participates_in_chain else 2.0
        elif kind == "exact":
            weight = 0.05 if participates_in_join else 0.35 if participates_in_chain else 1.0
        elif kind == "parameterized":
            weight = 2.5 if participates_in_join else 1.2
        elif kind == "unbounded":
            weight = 1.4 if participates_in_join else 0.6
        else:
            weight = 1.0
        weights.append(weight)
    return rng.choices(allowed, weights=weights, k=1)[0]


def _build_window(
    edge: TemporalEdge,
    nodes: dict[str, EventNode],
    spec: GenerationSpec,
    rng: random.Random,
    allocator: NameAllocator,
    components: ScenarioComponents,
    *,
    edge_count: int,
    incoming_counts: dict[str, int],
) -> tuple[TemporalEdge, TimeWindow]:
    bound_kind = _sample_window_bound_kind(edge, spec, rng, incoming_counts)
    delay_cap = max(1, spec.tick_budget // (edge_count + 1))
    if delay_cap < 1:
        raise GenerationError("tick budget cannot fit graph edges", reason="tick_budget_exceeded")

    if bound_kind == "exact":
        delay = rng.randint(1, min(3, delay_cap))
        bound = TimeBound(kind=WindowBoundKind.EXACT, min_delay=str(delay), max_delay=str(delay))
        sampled = delay
    elif bound_kind == "parameterized":
        param_pool = PARAM_NAMES_BY_FLAVOR.get(allocator.flavor.name, ("MAX_LAT",))
        param_name = allocator.take_param(param_pool)
        if not _has_param(components, param_name):
            components.params.append(ParameterDecl(name=param_name, kind="int"))
        delay = rng.randint(1, min(3, delay_cap))
        bound = TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay=param_name)
        sampled = delay
    elif bound_kind == "unbounded":
        delay = rng.randint(1, min(3, delay_cap))
        bound = TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay="1", max_delay="$")
        sampled = delay
    else:
        if incoming_counts.get(edge.end, 0) > 1:
            lo = 1
            hi = delay_cap
        else:
            lo = rng.randint(1, min(2, delay_cap))
            hi = min(delay_cap, lo + rng.randint(1, 3))
        if hi < lo:
            raise GenerationError("tick budget cannot fit sampled window delay", reason="tick_budget_exceeded")
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

    def add(
        *,
        signals: tuple[str, ...],
        relation: str,
        region: ConstraintRegion,
        value: str | None = None,
        anchor: str | None = None,
        window: str | None = None,
        start_anchor: str | None = None,
        end_anchor: str | None = None,
    ) -> None:
        components.lane_constraints.append(
            LaneConstraint(
                name=f"show_{counter[0]}_0",
                signals=signals,
                relation=relation,
                region=region,
                value=value,
                anchor=anchor,
                window=window,
                start_anchor=start_anchor,
                end_anchor=end_anchor,
            )
        )
        counter[0] += 1

    if "hold_until" in spec.idioms:
        emitted = False
        for index, edge in enumerate(list(components.edges)):
            if emitted and rng.random() >= 0.5:
                continue
            start_node = nodes[edge.start]
            if "parameterized" in spec.bound_kinds and rng.random() < 0.10:
                _parameterize_window(edge, components, allocator)
            roll = rng.random()
            if roll < 0.10:
                signals = tuple(_ensure_bus_signal(components, allocator) for _ in range(2))
                add(
                    signals=signals,
                    relation="stable",
                    region=ConstraintRegion.FROM_UNTIL,
                    start_anchor=edge.start,
                    end_anchor=edge.end,
                )
            elif roll < 0.25:
                bus_name = _ensure_bus_signal(components, allocator)
                add(
                    signals=(bus_name,),
                    relation="stable",
                    region=ConstraintRegion.FROM_UNTIL,
                    start_anchor=edge.start,
                    end_anchor=edge.end,
                )
            elif roll < 0.35:
                bus_name = _ensure_bus_signal(components, allocator)
                add(
                    signals=(bus_name,),
                    relation="eq",
                    value=rng.choice(VALUE_TOKENS),
                    region=ConstraintRegion.FROM_UNTIL,
                    start_anchor=edge.start,
                    end_anchor=edge.end,
                )
            else:
                relation = rng.choices(("high", "low", "stable"), weights=(0.65, 0.25, 0.10), k=1)[0]
                signal = (
                    start_node.primary_signal
                    if relation == "high" and _node_primary_is_high_compatible_bit(start_node, components)
                    else _ensure_bit_signal(components, allocator)
                )
                add(
                    signals=(signal,),
                    relation=relation,
                    region=ConstraintRegion.FROM_UNTIL,
                    start_anchor=edge.start,
                    end_anchor=edge.end,
                )
            emitted = True

    if "stable_while" in spec.idioms:
        for edge in components.edges[:1]:
            window_form = bool(components.windows) and rng.random() < 0.4
            target_window = edge.id
            bus_count = 2 if rng.random() < 0.3 else 1
            for _ in range(bus_count):
                bus_name = _ensure_bus_signal(components, allocator)
                if window_form:
                    add(
                        signals=(bus_name,),
                        relation="stable",
                        region=ConstraintRegion.IN,
                        window=target_window,
                    )
                else:
                    add(
                        signals=(bus_name,),
                        relation="stable",
                        region=ConstraintRegion.FROM_UNTIL,
                        start_anchor=edge.start,
                        end_anchor=edge.end,
                    )

    if "not_before" in spec.idioms:
        target_node = _find_response_node(graph, nodes)
        gate_node = _find_trigger_node(graph, nodes)
        if target_node and gate_node:
            roll = rng.random()
            if roll < 0.50:
                signal = (
                    target_node.primary_signal
                    if _signal_kind(components, target_node.primary_signal) == SignalKind.BIT
                    else _ensure_bit_signal(components, allocator)
                )
                add(
                    signals=(signal,),
                    relation="low",
                    region=ConstraintRegion.BEFORE,
                    anchor=gate_node.id,
                )
            elif roll < 0.80:
                signal = (
                    target_node.primary_signal
                    if _signal_kind(components, target_node.primary_signal) == SignalKind.BUS
                    else _ensure_bus_signal(components, allocator)
                )
                add(
                    signals=(signal,),
                    relation="neq",
                    value=rng.choice(VALUE_TOKENS),
                    region=ConstraintRegion.BEFORE,
                    anchor=gate_node.id,
                )
            elif components.edges:
                signal = _ensure_bit_signal(components, allocator)
                add(
                    signals=(signal,),
                    relation="low",
                    region=ConstraintRegion.BEFORE,
                    anchor=components.edges[-1].end,
                )

    if "burst" in spec.idioms or graph.topology == "burst":
        payload = _ensure_bus_signal(components, allocator, hint=_payload_hint(allocator))
        beat_nodes = [node for node in nodes.values() if "beat" in node.id]
        for node in beat_nodes:
            add(
                signals=(payload,),
                relation="change",
                region=ConstraintRegion.AT,
                anchor=node.id,
            )

    if "backpressure" in spec.idioms or graph.topology == "backpressure":
        valid = nodes.get("valid_rise")
        handshake = nodes.get("handshake")
        if valid and handshake and valid.primary_signal:
            ready_low = _ensure_bit_signal(components, allocator, hint=_ready_hint(allocator))
            add(
                signals=(valid.primary_signal,),
                relation="high",
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor=valid.id,
                end_anchor=handshake.id,
            )
            add(
                signals=(ready_low,),
                relation="low",
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor=valid.id,
                end_anchor=handshake.id,
            )
            if rng.random() < 0.8:
                bus_name = _ensure_bus_signal(components, allocator)
                add(
                    signals=(bus_name,),
                    relation="stable",
                    region=ConstraintRegion.FROM_UNTIL,
                    start_anchor=valid.id,
                    end_anchor=handshake.id,
                )

    _add_regional_constraints(components, nodes, spec, rng, allocator, add)


def _add_regional_constraints(
    components: ScenarioComponents,
    nodes: dict[str, EventNode],
    spec: GenerationSpec,
    rng: random.Random,
    allocator: NameAllocator,
    add,
) -> None:
    anchor_names = [node.id for node in nodes.values()]
    response_or_state = [node.id for node in nodes.values() if node.role in {"response", "state"}]
    preferred_regions = set(spec.region_bias)
    item_number = int(spec.item_id.rsplit("_", 1)[-1])
    if anchor_names and rng.random() < (0.85 if "at" in preferred_regions else 0.4):
        anchor = rng.choice(response_or_state or anchor_names)
        if rng.random() < 0.55:
            relation = ("fall", "high", "low")[item_number % 3]
            add(
                signals=(_ensure_bit_signal(components, allocator),),
                relation=relation,
                region=ConstraintRegion.AT,
                anchor=anchor,
            )
        else:
            relation = rng.choice(("eq", "neq", "change", "stable"))
            value = rng.choice(VALUE_TOKENS) if relation in {"eq", "neq"} else None
            add(
                signals=(_ensure_bus_signal(components, allocator),),
                relation=relation,
                value=value,
                region=ConstraintRegion.AT,
                anchor=anchor,
            )

    if components.windows and rng.random() < (0.85 if "in" in preferred_regions else 0.4):
        window = rng.choice(components.windows)
        relation = rng.choice(("stable", "eq", "neq", "change"))
        value = rng.choice(VALUE_TOKENS) if relation in {"eq", "neq"} else None
        add(
            signals=(_ensure_bus_signal(components, allocator),),
            relation=relation,
            value=value,
            region=ConstraintRegion.IN,
            window=window.name,
        )

    if anchor_names and rng.random() < (0.85 if "after" in preferred_regions else 0.4):
        anchor = components.edges[-1].end if components.edges else rng.choice(anchor_names)
        if rng.random() < 0.5:
            relation = rng.choice(("high", "low"))
            add(
                signals=(_ensure_bit_signal(components, allocator),),
                relation=relation,
                region=ConstraintRegion.AFTER,
                anchor=anchor,
            )
        else:
            relation = rng.choice(("stable", "eq", "neq"))
            value = rng.choice(VALUE_TOKENS) if relation in {"eq", "neq"} else None
            add(
                signals=(_ensure_bus_signal(components, allocator),),
                relation=relation,
                value=value,
                region=ConstraintRegion.AFTER,
                anchor=anchor,
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


def _signal_kind(components: ScenarioComponents, signal_name: str | None) -> SignalKind | None:
    if signal_name is None:
        return None
    for signal in components.signals:
        if signal.name == signal_name:
            return signal.kind
    return None


def _ensure_bit_signal(
    components: ScenarioComponents,
    allocator: NameAllocator,
    hint: str | None = None,
) -> str:
    name = allocator.take_bit(hint)
    if not _has_signal(components, name):
        components.signals.append(_make_bit_signal(name))
    return name


def _ensure_bus_signal(
    components: ScenarioComponents,
    allocator: NameAllocator,
    hint: str | None = None,
) -> str:
    name = allocator.take_bus(hint)
    if not _has_signal(components, name):
        components.signals.append(_make_bus_signal(name, allocator.take_bus_width()))
    return name


def _node_primary_is_high_compatible_bit(node: EventNode, components: ScenarioComponents) -> bool:
    if node.primary_signal is None:
        return False
    if _signal_kind(components, node.primary_signal) != SignalKind.BIT:
        return False
    return node.predicate_kind in {"rise", "high", "all_high", "all_high_eq"}


def _payload_hint(allocator: NameAllocator) -> str | None:
    for hint in ("data", "WDATA", "TX_DATA", "wdata", "rdata", "da"):
        if hint in allocator._available_bus:
            return hint
    return None


def _ready_hint(allocator: NameAllocator) -> str | None:
    for hint in ("ready", "AWREADY", "WREADY", "TX_READY", "cmd_ready", "r"):
        if hint in allocator._available_bits:
            return hint
    return None


def _parameterize_window(edge: TemporalEdge, components: ScenarioComponents, allocator: NameAllocator) -> None:
    param_pool = PARAM_NAMES_BY_FLAVOR.get(allocator.flavor.name, ("MAX_LAT",))
    param_name = allocator.take_param(param_pool)
    if not _has_param(components, param_name):
        components.params.append(ParameterDecl(name=param_name, kind="int"))
    for index, window in enumerate(components.windows):
        if window.name == edge.id:
            components.windows[index] = replace(
                window,
                bound=TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay=param_name),
            )
            break
    for index, sampled_edge in enumerate(components.edges):
        if sampled_edge.id == edge.id:
            components.edges[index] = replace(
                sampled_edge,
                bound_kind="parameterized",
                parameter=param_name,
            )
            break


def _add_cuts(
    graph: DecoratedGraph,
    components: ScenarioComponents,
    nodes: dict[str, EventNode],
    spec: GenerationSpec,
    rng: random.Random,
) -> None:
    placement_options: list[tuple[str, str | None, str | None, str | None]] = []
    trigger = _find_trigger_node(graph, nodes)
    response = _find_response_node(graph, nodes)
    if trigger:
        placement_options.append(("before", trigger.id, None, None))
    if response:
        placement_options.append(("after", response.id, None, None))
    if len(components.windows) >= 2:
        placement_options.append(("between", None, components.windows[0].name, components.windows[1].name))

    if not placement_options:
        return

    item_number = int(spec.item_id.rsplit("_", 1)[-1])
    preferred_order = ("before", "after", "between")
    preferred_placement = preferred_order[item_number % len(preferred_order)]
    if spec.cut_placement_bias and rng.random() < 0.75:
        preferred_placement = rng.choice(spec.cut_placement_bias)
    matching = [option for option in placement_options if option[0] == preferred_placement]
    placement, anchor_name, left_window, right_window = matching[0] if matching else placement_options[item_number % len(placement_options)]

    cut_name = f"cut_{placement}_{item_number % 1000}"
    meaning = _cut_meaning_for(placement, item_number)
    cut_placement = {
        "before": CutPlacement.BEFORE_ANCHOR,
        "after": CutPlacement.AFTER_ANCHOR,
        "between": CutPlacement.BETWEEN_WINDOWS,
    }[placement]
    components.cuts.append(
        Cut(
            name=cut_name,
            placement=cut_placement,
            meaning=meaning,
            anchor=anchor_name if placement != "between" else None,
            left_window=left_window if placement == "between" else None,
            right_window=right_window if placement == "between" else None,
            label=_cut_label_for(placement, item_number),
        )
    )


def _cut_meaning_for(placement: str, item_number: int) -> CutMeaning:
    if placement == "between":
        return CutMeaning.SYMBOLIC_GAP if item_number % 2 == 0 else CutMeaning.LOOKBACK
    variants = (CutMeaning.SYMBOLIC_GAP, CutMeaning.LOOKBACK)
    if placement == "before":
        variants = (CutMeaning.OMITTED_HISTORY, *variants)
    else:
        variants = (CutMeaning.OMITTED_FUTURE, *variants)
    return variants[item_number % len(variants)]


def _cut_label_for(placement: str, item_number: int) -> str | None:
    variant = item_number % 4
    if variant == 0:
        return None
    if variant == 1:
        return "idle" if placement == "before" else "gap"
    if variant == 2:
        return "compressed lookback context" if placement == "between" else "next transaction"
    return f"{placement} region intentionally hidden for timing context"


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
