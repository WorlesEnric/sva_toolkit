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


def build_topology(topology: str, flavor: str, rng: random.Random) -> DecoratedGraph:
    """Return an undecorated graph (no concrete signals yet) for the chosen shape."""

    if topology == "single_response":
        return _single_response(flavor)
    if topology == "chain":
        return _chain(flavor)
    if topology == "fork":
        return _fork(flavor)
    if topology == "join":
        return _join(flavor)
    if topology == "parallel":
        return _parallel(flavor)
    if topology == "burst":
        return _burst(flavor)
    if topology == "backpressure":
        return _backpressure(flavor)
    if topology == "setup_hold":
        return _setup_hold(flavor)
    raise ValueError(f"unknown topology: {topology}")


def _node(node_id: str, *, role: str, predicate_kind: str = "rise") -> EventNode:
    return EventNode(id=node_id, role=role, predicate_kind=predicate_kind)


def _edge(edge_id: str, start: str, end: str) -> TemporalEdge:
    return TemporalEdge(id=edge_id, start=start, end=end, bound_kind="range", min_delay=1, max_delay=4)


def _single_response(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="single_response",
        flavor=flavor,
        nodes=[
            _node("trigger", role="trigger", predicate_kind="rise"),
            _node("response", role="response", predicate_kind="rise"),
        ],
        edges=[_edge("response_window", "trigger", "response")],
    )


def _chain(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="chain",
        flavor=flavor,
        nodes=[
            _node("trigger", role="trigger", predicate_kind="rise"),
            _node("middle", role="state", predicate_kind="rise"),
            _node("response", role="response", predicate_kind="rise"),
        ],
        edges=[
            _edge("first_window", "trigger", "middle"),
            _edge("second_window", "middle", "response"),
        ],
    )


def _fork(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="fork",
        flavor=flavor,
        nodes=[
            _node("trigger", role="trigger", predicate_kind="rise"),
            _node("branch_a", role="response", predicate_kind="rise"),
            _node("branch_b", role="response", predicate_kind="rise"),
        ],
        edges=[
            _edge("fork_a", "trigger", "branch_a"),
            _edge("fork_b", "trigger", "branch_b"),
        ],
    )


def _join(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="join",
        flavor=flavor,
        nodes=[
            _node("first", role="trigger", predicate_kind="rise"),
            _node("second", role="trigger", predicate_kind="rise"),
            _node("done", role="response", predicate_kind="rise"),
        ],
        edges=[
            _edge("join_a", "first", "done"),
            _edge("join_b", "second", "done"),
        ],
    )


def _parallel(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="parallel",
        flavor=flavor,
        nodes=[
            _node("a_start", role="trigger", predicate_kind="rise"),
            _node("a_end", role="response", predicate_kind="rise"),
            _node("b_start", role="trigger", predicate_kind="rise"),
            _node("b_end", role="response", predicate_kind="rise"),
        ],
        edges=[
            _edge("a_window", "a_start", "a_end"),
            _edge("b_window", "b_start", "b_end"),
        ],
    )


def _burst(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="burst",
        flavor=flavor,
        nodes=[
            _node("first_beat", role="trigger", predicate_kind="all_high"),
            _node("last_beat", role="response", predicate_kind="all_high"),
        ],
        edges=[_edge("burst_window", "first_beat", "last_beat")],
    )


def _backpressure(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="backpressure",
        flavor=flavor,
        nodes=[
            _node("valid_rise", role="trigger", predicate_kind="rise"),
            _node("handshake", role="response", predicate_kind="all_high"),
        ],
        edges=[_edge("wait_window", "valid_rise", "handshake")],
    )


def _setup_hold(flavor: str) -> DecoratedGraph:
    return DecoratedGraph(
        topology="setup_hold",
        flavor=flavor,
        nodes=[
            _node("launch", role="trigger", predicate_kind="rise"),
            _node("capture", role="state", predicate_kind="rise"),
            _node("hold_end", role="response", predicate_kind="high"),
        ],
        edges=[
            _edge("setup_window", "launch", "capture"),
            _edge("hold_window", "capture", "hold_end"),
        ],
    )
