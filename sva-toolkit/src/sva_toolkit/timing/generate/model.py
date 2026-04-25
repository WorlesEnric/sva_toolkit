"""Internal data structures used by the timing diagram dataset generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sva_toolkit.timing.core.scenario import (
    Anchor,
    Cut,
    LaneConstraint,
    ParameterDecl,
    SignalDecl,
    TimeWindow,
)


@dataclass(frozen=True)
class EventNode:
    """Abstract event in the temporal graph; later realized as an Anchor."""

    id: str
    role: str = "state"
    predicate_kind: str = "rise"  # rise|fall|high|low|change|stable|eq|neq|all_high|all_high_eq
    primary_signal: str | None = None
    extra_signals: tuple[str, ...] = ()
    bus_signal: str | None = None
    eq_value: str | None = None


@dataclass(frozen=True)
class TemporalEdge:
    """Abstract temporal relationship between two events; realized as a TimeWindow."""

    id: str
    start: str
    end: str
    bound_kind: str  # exact|range|parameterized|unbounded
    min_delay: int = 1
    max_delay: int = 1
    parameter: str | None = None
    sampled_delay: int = 1
    omitted: bool = False  # rendered as a hold-only window without a finite bound


@dataclass
class DecoratedGraph:
    """Topology graph plus the decorations needed to lower into the scenario IR."""

    topology: str
    flavor: str
    nodes: list[EventNode] = field(default_factory=list)
    edges: list[TemporalEdge] = field(default_factory=list)


@dataclass
class ScenarioComponents:
    """Concrete IR fragments produced by the idiom layer for one item."""

    name: str
    clock_signal: str
    params: list[ParameterDecl] = field(default_factory=list)
    signals: list[SignalDecl] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    windows: list[TimeWindow] = field(default_factory=list)
    cuts: list[Cut] = field(default_factory=list)
    lane_constraints: list[LaneConstraint] = field(default_factory=list)
    anchor_node_map: dict[str, EventNode] = field(default_factory=dict)
    edges: list[TemporalEdge] = field(default_factory=list)
    response_overlay_targets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationSpec:
    """Per-item generation parameters chosen by the coverage-guided sampler."""

    item_id: str
    seed: int
    topology: str
    flavor: str
    idioms: tuple[str, ...]
    rendering: str  # concrete|symbolic|mixed
    bound_kinds: tuple[str, ...]
    naming: str  # snake_case|uppercase|protocol_like|short
    cuts_enabled: bool
    distractor_lanes: int
    tick_budget: int
    clock_edge: str = "posedge"
    predicate_bias: tuple[str, ...] = ()
    region_bias: tuple[str, ...] = ()
    cut_placement_bias: tuple[str, ...] = ()


@dataclass
class GeneratedItem:
    """One accepted dataset item ready to write to disk."""

    id: str
    seed: int
    canonical_dsl: str
    svg_text: str
    features: dict[str, Any]


class GenerationError(RuntimeError):
    """Raised when an internal generator step cannot produce a valid candidate."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason or _reason_from_message(message)


def _reason_from_message(message: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in message.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized[:80] or "generation_error"
