"""Assertion-oriented projection of the timing core model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sva_toolkit.timing.core.model import DiagramSpec, EventExpr, EventPredicate, HoldUntilRule, NotBeforeRule, ParameterDecl, ResponseRule


@dataclass(frozen=True)
class AssertionProperty:
    """A parameterized SVA property ready for pretty-printing."""

    name: str
    params: Tuple[ParameterDecl, ...]
    body: str


def build_assertion_view(diagram: DiagramSpec) -> Tuple[AssertionProperty, ...]:
    """Build assertion projection objects from the semantic core model."""

    properties = []
    param_map = {param.name: param for param in diagram.params}

    for rule in diagram.rules:
        if isinstance(rule, ResponseRule):
            used_params = _collect_params(param_map, rule.min_delay, rule.max_delay)
            body = (
                f"{_wrap_expr(_event_to_sva(diagram, rule.trigger_event))} |-> "
                f"##[{rule.min_delay}:{rule.max_delay}] {_wrap_expr(_event_to_sva(diagram, rule.response_event))}"
            )
        elif isinstance(rule, NotBeforeRule):
            used_params = ()
            body = (
                f"!{_wrap_expr(_event_to_sva(diagram, rule.forbidden_event))} "
                f"until {_wrap_expr(_event_to_sva(diagram, rule.reference_event))}"
            )
        elif isinstance(rule, HoldUntilRule):
            used_params = ()
            body = (
                f"{_wrap_expr(_event_to_sva(diagram, rule.start_event))} |-> "
                f"{_wrap_expr(_expr_to_sva(rule.predicate_expr))} "
                f"until_with {_wrap_expr(_event_to_sva(diagram, rule.end_event))}"
            )
        else:
            continue
        properties.append(AssertionProperty(name=f"p_{rule.name}", params=used_params, body=body))

    return tuple(properties)


def _event_to_sva(diagram: DiagramSpec, event_name: str) -> str:
    event = diagram.event_map[event_name]
    return _expr_to_sva(event.expr)


def _expr_to_sva(expr: EventExpr) -> str:
    return " && ".join(_predicate_to_sva(predicate) for predicate in expr.predicates)


def _predicate_to_sva(predicate: EventPredicate) -> str:
    op = predicate.op
    signal = predicate.signal
    if op == "rise":
        return f"$rose({signal})"
    if op == "fall":
        return f"$fell({signal})"
    if op == "high":
        return signal
    if op == "low":
        return f"!{signal}"
    if op == "change":
        return f"$changed({signal})"
    if op == "stable":
        return f"$stable({signal})"
    if op == "eq":
        return f"({signal} == {predicate.value})"
    if op == "neq":
        return f"({signal} != {predicate.value})"
    raise ValueError(f"unsupported predicate op: {op}")


def _collect_params(param_map, *names: str) -> Tuple[ParameterDecl, ...]:
    ordered = []
    seen = set()
    for name in names:
        if name in param_map and name not in seen:
            ordered.append(param_map[name])
            seen.add(name)
    return tuple(ordered)


def _wrap_expr(expr: str) -> str:
    if " && " in expr or " until" in expr:
        return f"({expr})"
    return expr
