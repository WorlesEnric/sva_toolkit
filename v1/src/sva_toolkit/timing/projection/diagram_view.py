"""Diagram-oriented projection with evaluated event occurrences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sva_toolkit.timing.core.model import DiagramSpec, EventExpr, EventPredicate, HoldUntilRule, LaneSpec, NotBeforeRule, ResponseRule


@dataclass(frozen=True)
class EventOccurrence:
    """Occurrence of a named event on a specific tick."""

    name: str
    tick: int
    anchor: str


@dataclass(frozen=True)
class ResponseOverlay:
    """Arrow-like overlay shown between two event occurrences."""

    name: str
    start_tick: int
    start_anchor: str
    end_tick: int
    end_anchor: str
    label: str


@dataclass(frozen=True)
class HoldOverlay:
    """Highlighted interval on a lane."""

    name: str
    lane_names: Tuple[str, ...]
    start_tick: int
    end_tick: int
    label: str


@dataclass(frozen=True)
class DiagramView:
    """Rendering input derived from the core timing model."""

    spec: DiagramSpec
    occurrences: Tuple[EventOccurrence, ...]
    response_overlays: Tuple[ResponseOverlay, ...]
    hold_overlays: Tuple[HoldOverlay, ...]


def build_diagram_view(diagram: DiagramSpec) -> DiagramView:
    """Evaluate event occurrences and backend-friendly overlays."""

    occurrence_map: Dict[str, List[EventOccurrence]] = {
        event.name: _evaluate_event(diagram, event.name) for event in diagram.events
    }

    responses = []
    holds = []
    for rule in diagram.rules:
        if isinstance(rule, ResponseRule):
            start, end = _first_matching_pair(
                occurrence_map.get(rule.trigger_event, []),
                occurrence_map.get(rule.response_event, []),
            )
            if start is not None and end is not None:
                responses.append(
                    ResponseOverlay(
                        name=rule.name,
                        start_tick=start.tick,
                        start_anchor=start.anchor,
                        end_tick=end.tick,
                        end_anchor=end.anchor,
                        label=f"[{rule.min_delay}:{rule.max_delay}]",
                    )
                )
        elif isinstance(rule, HoldUntilRule):
            start, end = _first_matching_pair(
                occurrence_map.get(rule.start_event, []),
                occurrence_map.get(rule.end_event, []),
            )
            if start is not None and end is not None:
                holds.append(
                    HoldOverlay(
                        name=rule.name,
                        lane_names=tuple(dict.fromkeys(predicate.signal for predicate in rule.predicate_expr.predicates)),
                        start_tick=start.tick,
                        end_tick=end.tick,
                        label=_expr_label(rule.predicate_expr),
                    )
                )
        elif isinstance(rule, NotBeforeRule):
            continue

    flattened = tuple(occ for occs in occurrence_map.values() for occ in occs)
    return DiagramView(
        spec=diagram,
        occurrences=flattened,
        response_overlays=tuple(responses),
        hold_overlays=tuple(holds),
    )


def _evaluate_event(diagram: DiagramSpec, event_name: str) -> List[EventOccurrence]:
    event = diagram.event_map[event_name]
    occurrences: List[EventOccurrence] = []
    for tick in range(diagram.ticks):
        if all(_predicate_matches(diagram.lane_map[p.signal], p, tick) for p in event.expr.predicates):
            anchor = "boundary" if all(p.op in {"rise", "fall", "change"} for p in event.expr.predicates) else "center"
            occurrences.append(EventOccurrence(name=event_name, tick=tick, anchor=anchor))
    return occurrences


def _predicate_matches(lane: LaneSpec, predicate: EventPredicate, tick: int) -> bool:
    current = lane.samples[tick]
    previous = lane.samples[tick - 1] if tick > 0 else None
    op = predicate.op
    if op == "high":
        return current == "1"
    if op == "low":
        return current == "0"
    if op == "rise":
        return previous == "0" and current == "1"
    if op == "fall":
        return previous == "1" and current == "0"
    if op == "change":
        return previous is not None and current != previous
    if op == "stable":
        return previous is not None and current == previous
    if op == "eq":
        return current == predicate.value
    if op == "neq":
        return current != predicate.value
    raise ValueError(f"unsupported predicate op: {op}")


def _first_matching_pair(
    starts: List[EventOccurrence],
    ends: List[EventOccurrence],
) -> Tuple[Optional[EventOccurrence], Optional[EventOccurrence]]:
    for start in starts:
        for end in ends:
            if end.tick >= start.tick:
                return start, end
    return None, None


def _expr_label(expr: EventExpr) -> str:
    parts = []
    for predicate in expr.predicates:
        if predicate.value is None:
            parts.append(f"{predicate.op}({predicate.signal})")
        else:
            parts.append(f"{predicate.op}({predicate.signal}, {predicate.value})")
    return " and ".join(parts)
