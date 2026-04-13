"""Core timing model shared by rendering and SVA emission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union


class LaneKind(str, Enum):
    """Supported lane kinds for the first timing MVP."""

    BIT = "bit"
    BUS = "bus"


@dataclass(frozen=True)
class ClockingSpec:
    """Clocking context for a diagram."""

    edge: str
    signal: str
    disable_iff: Optional[str] = None


@dataclass(frozen=True)
class ParameterDecl:
    """Parameterized symbol preserved into SVA output."""

    name: str
    kind: str = "int"


@dataclass(frozen=True)
class LaneSpec:
    """A rendered timing lane with one sample per discrete tick."""

    name: str
    kind: LaneKind
    width: Optional[str]
    samples: Tuple[str, ...]

    @property
    def display_name(self) -> str:
        if self.width and not (self.kind == LaneKind.BIT and self.width == "1"):
            return f"{self.name}[{self.width}]"
        return self.name


@dataclass(frozen=True)
class EventPredicate:
    """Primitive event predicate evaluated against a lane."""

    op: str
    signal: str
    value: Optional[str] = None


@dataclass(frozen=True)
class EventExpr:
    """Conjunction of event predicates on the same cycle."""

    predicates: Tuple[EventPredicate, ...]


@dataclass(frozen=True)
class EventSpec:
    """Named event for rule and rendering references."""

    name: str
    expr: EventExpr


@dataclass(frozen=True)
class RuleSpec:
    """Base class for supported timing rules."""

    name: str


@dataclass(frozen=True)
class NotBeforeRule(RuleSpec):
    """Forbid one event from happening before another."""

    forbidden_event: str
    reference_event: str


@dataclass(frozen=True)
class ResponseRule(RuleSpec):
    """Require a response event in a ranged delay window."""

    trigger_event: str
    min_delay: str
    max_delay: str
    response_event: str


@dataclass(frozen=True)
class HoldUntilRule(RuleSpec):
    """Require a boolean predicate to hold between two events."""

    predicate_expr: EventExpr
    start_event: str
    end_event: str


SupportedRule = Union[NotBeforeRule, ResponseRule, HoldUntilRule]


@dataclass(frozen=True)
class DiagramSpec:
    """Full timing diagram specification."""

    name: str
    clocking: ClockingSpec
    ticks: int
    params: Tuple[ParameterDecl, ...]
    lanes: Tuple[LaneSpec, ...]
    events: Tuple[EventSpec, ...]
    rules: Tuple[SupportedRule, ...]

    @property
    def lane_map(self) -> Dict[str, LaneSpec]:
        return {lane.name: lane for lane in self.lanes}

    @property
    def event_map(self) -> Dict[str, EventSpec]:
        return {event.name: event for event in self.events}
