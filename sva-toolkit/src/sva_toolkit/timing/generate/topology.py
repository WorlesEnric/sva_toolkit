"""Topology generators for the timing dataset event graphs."""

from __future__ import annotations

import random

from sva_toolkit.timing.generate.model import DecoratedGraph, EventNode, TemporalEdge


TOPOLOGIES: tuple[str, ...] = (
    "single_response",
    "chain",
    "fork",
    "join",
    "parallel",
    "burst",
    "backpressure",
    "setup_hold",
)


def build_topology(
    topology: str,
    flavor: str,
    rng: random.Random,
    *,
    predicate_bias: tuple[str, ...] = (),
) -> DecoratedGraph:
    """Return an undecorated graph (no concrete signals yet) for the chosen shape."""

    if topology == "single_response":
        return _single_response(flavor, rng, predicate_bias)
    if topology == "chain":
        return _chain(flavor, rng, predicate_bias)
    if topology == "fork":
        return _fork(flavor, rng, predicate_bias)
    if topology == "join":
        return _join(flavor, rng, predicate_bias)
    if topology == "parallel":
        return _parallel(flavor, rng, predicate_bias)
    if topology == "burst":
        return _burst(flavor, rng)
    if topology == "backpressure":
        return _backpressure(flavor, rng)
    if topology == "setup_hold":
        return _setup_hold(flavor, rng, predicate_bias)
    raise ValueError(f"unknown topology: {topology}")


def _node(
    node_id: str,
    *,
    role: str,
    rng: random.Random,
    predicate_kind: str | None = None,
    predicate_bias: tuple[str, ...] = (),
) -> EventNode:
    if predicate_kind is None:
        predicate_kind = _sample_predicate_kind(role, rng, predicate_bias)
    return EventNode(id=node_id, role=role, predicate_kind=predicate_kind)


def _edge(edge_id: str, start: str, end: str) -> TemporalEdge:
    return TemporalEdge(id=edge_id, start=start, end=end, bound_kind="range", min_delay=1, max_delay=4)


def _sample_predicate_kind(role: str, rng: random.Random, predicate_bias: tuple[str, ...] = ()) -> str:
    if role in {"state", "response"} and rng.random() < 0.30:
        return "all_high_eq" if rng.random() < 0.45 else "all_high"
    choices = ("rise", "fall", "high", "low", "change", "stable", "eq", "neq")
    weights = (0.50, 0.12, 0.12, 0.08, 0.07, 0.05, 0.04, 0.02)
    preferred = set(predicate_bias)
    if preferred:
        weights = tuple(weight * (5.0 if choice in preferred else 1.0) for choice, weight in zip(choices, weights))
    return rng.choices(choices, weights=weights, k=1)[0]


def _single_response(flavor: str, rng: random.Random, predicate_bias: tuple[str, ...]) -> DecoratedGraph:
    return DecoratedGraph(
        topology="single_response",
        flavor=flavor,
        nodes=[
            _node("trigger", role="trigger", rng=rng, predicate_bias=predicate_bias),
            _node("response", role="response", rng=rng, predicate_bias=predicate_bias),
        ],
        edges=[_edge("response_window", "trigger", "response")],
    )


def _chain(flavor: str, rng: random.Random, predicate_bias: tuple[str, ...]) -> DecoratedGraph:
    return DecoratedGraph(
        topology="chain",
        flavor=flavor,
        nodes=[
            _node("trigger", role="trigger", rng=rng, predicate_bias=predicate_bias),
            _node("middle", role="state", rng=rng, predicate_bias=predicate_bias),
            _node("response", role="response", rng=rng, predicate_bias=predicate_bias),
        ],
        edges=[
            _edge("first_window", "trigger", "middle"),
            _edge("second_window", "middle", "response"),
        ],
    )


def _fork(flavor: str, rng: random.Random, predicate_bias: tuple[str, ...]) -> DecoratedGraph:
    return DecoratedGraph(
        topology="fork",
        flavor=flavor,
        nodes=[
            _node("trigger", role="trigger", rng=rng, predicate_bias=predicate_bias),
            _node("branch_a", role="response", rng=rng, predicate_bias=predicate_bias),
            _node("branch_b", role="response", rng=rng, predicate_bias=predicate_bias),
        ],
        edges=[
            _edge("fork_a", "trigger", "branch_a"),
            _edge("fork_b", "trigger", "branch_b"),
        ],
    )


def _join(flavor: str, rng: random.Random, predicate_bias: tuple[str, ...]) -> DecoratedGraph:
    return DecoratedGraph(
        topology="join",
        flavor=flavor,
        nodes=[
            _node("first", role="trigger", rng=rng, predicate_bias=predicate_bias),
            _node("second", role="trigger", rng=rng, predicate_bias=predicate_bias),
            _node("done", role="response", rng=rng, predicate_bias=predicate_bias),
        ],
        edges=[
            _edge("join_a", "first", "done"),
            _edge("join_b", "second", "done"),
        ],
    )


def _parallel(flavor: str, rng: random.Random, predicate_bias: tuple[str, ...]) -> DecoratedGraph:
    return DecoratedGraph(
        topology="parallel",
        flavor=flavor,
        nodes=[
            _node("a_start", role="trigger", rng=rng, predicate_bias=predicate_bias),
            _node("a_end", role="response", rng=rng, predicate_bias=predicate_bias),
            _node("b_start", role="trigger", rng=rng, predicate_bias=predicate_bias),
            _node("b_end", role="response", rng=rng, predicate_bias=predicate_bias),
        ],
        edges=[
            _edge("a_window", "a_start", "a_end"),
            _edge("b_window", "b_start", "b_end"),
        ],
    )


def _burst(flavor: str, rng: random.Random) -> DecoratedGraph:
    nodes = [
        _node("first_beat", role="trigger", rng=rng, predicate_kind="all_high"),
    ]
    edges: list[TemporalEdge] = []
    previous = "first_beat"
    include_middle = rng.random() < 0.5
    include_response_after_last = rng.random() < 0.5
    if not include_middle and not include_response_after_last:
        include_middle = True
    if include_middle:
        nodes.append(_node("middle_beat", role="state", rng=rng, predicate_kind="all_high"))
        edges.append(_edge("burst_first_to_middle", "first_beat", "middle_beat"))
        previous = "middle_beat"
    nodes.append(_node("last_beat", role="response", rng=rng, predicate_kind="all_high_eq"))
    edges.append(_edge("burst_window", previous, "last_beat"))
    if include_response_after_last:
        nodes.append(_node("response_after_last", role="response", rng=rng, predicate_kind="rise"))
        edges.append(_edge("response_after_last_window", "last_beat", "response_after_last"))
    return DecoratedGraph(
        topology="burst",
        flavor=flavor,
        nodes=nodes,
        edges=edges,
    )


def _backpressure(flavor: str, rng: random.Random) -> DecoratedGraph:
    return DecoratedGraph(
        topology="backpressure",
        flavor=flavor,
        nodes=[
            _node("valid_rise", role="trigger", rng=rng, predicate_kind="rise"),
            _node("handshake", role="response", rng=rng, predicate_kind="all_high"),
        ],
        edges=[_edge("wait_window", "valid_rise", "handshake")],
    )


def _setup_hold(flavor: str, rng: random.Random, predicate_bias: tuple[str, ...]) -> DecoratedGraph:
    variant = rng.choices(("both", "pre_only", "post_only"), weights=(0.50, 0.25, 0.25), k=1)[0]
    if variant == "pre_only":
        return DecoratedGraph(
            topology="setup_hold",
            flavor=flavor,
            nodes=[
                _node("launch", role="trigger", rng=rng, predicate_bias=predicate_bias),
                _node("capture", role="response", rng=rng, predicate_bias=predicate_bias),
            ],
            edges=[_edge("setup_window", "launch", "capture")],
        )
    if variant == "post_only":
        return DecoratedGraph(
            topology="setup_hold",
            flavor=flavor,
            nodes=[
                _node("capture", role="trigger", rng=rng, predicate_bias=predicate_bias),
                _node("hold_end", role="response", rng=rng, predicate_kind="high"),
            ],
            edges=[_edge("hold_window", "capture", "hold_end")],
        )
    return DecoratedGraph(
            topology="setup_hold",
            flavor=flavor,
            nodes=[
                _node("launch", role="trigger", rng=rng, predicate_bias=predicate_bias),
                _node("capture", role="state", rng=rng, predicate_bias=predicate_bias),
                _node("hold_end", role="response", rng=rng, predicate_kind="high"),
            ],
        edges=[
            _edge("setup_window", "launch", "capture"),
            _edge("hold_window", "capture", "hold_end"),
        ],
    )
