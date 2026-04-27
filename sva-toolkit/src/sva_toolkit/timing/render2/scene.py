"""Renderer-independent timing scene IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from sva_toolkit.timing.core.scenario import ScenarioDocument
from sva_toolkit.timing.visual import VisibilityClass

if TYPE_CHECKING:
    from sva_toolkit.timing.render2.decorations import Decoration


class LaneType(str, Enum):
    BIT = "bit"
    BUS = "bus"
    CLOCK = "clock"
    ANALOG = "analog"
    UNKNOWN = "unknown"
    HIGH_Z = "high_z"


@dataclass(frozen=True)
class SampleRun:
    """A contiguous run of identical sample values on a lane."""

    start_tick: int
    end_tick: int
    value: str
    is_unknown: bool = False
    is_high_z: bool = False

    def __post_init__(self) -> None:
        if self.end_tick < self.start_tick:
            raise ValueError("sample run end_tick must be >= start_tick")


@dataclass(frozen=True)
class LaneScene:
    name: str
    lane_type: LaneType
    runs: tuple[SampleRun, ...]
    width_bits: str | None = None
    visibility: VisibilityClass = VisibilityClass.VISIBLE_TEXT

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_type", LaneType(self.lane_type))
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "visibility", VisibilityClass(self.visibility))


@dataclass(frozen=True)
class TickModel:
    total_ticks: int
    tick_origin: int = 0
    grid_pitch_hint: float = 1.0

    def __post_init__(self) -> None:
        if self.total_ticks < 0:
            raise ValueError("total_ticks must be non-negative")
        if self.grid_pitch_hint <= 0:
            raise ValueError("grid_pitch_hint must be positive")


@dataclass(frozen=True)
class CutRegion:
    name: str
    start_tick: int
    end_tick: int
    meaning: str
    label: str | None = None

    def __post_init__(self) -> None:
        if self.end_tick < self.start_tick:
            raise ValueError("cut end_tick must be >= start_tick")


@dataclass(frozen=True)
class VisualEvent:
    name: str
    tick: int
    placement: str
    target_visibility: VisibilityClass

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_visibility", VisibilityClass(self.target_visibility))


@dataclass(frozen=True)
class VisualConstraint:
    name: str
    kind: str
    region: str
    lane_names: tuple[str, ...]
    start_tick: int | None
    end_tick: int | None
    anchor_ref: str | None
    window_ref: str | None
    visibility: VisibilityClass

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_names", tuple(self.lane_names))
        object.__setattr__(self, "visibility", VisibilityClass(self.visibility))


@dataclass(frozen=True)
class TimingScene:
    """Renderer-independent timing scene.

    ``visible_target`` and ``semantic_document`` retain source context for audit
    tooling, but they are excluded from equality and hashing so serialized scene
    comparisons stay structural and renderer-facing.
    """

    name: str
    clocking_edge: str
    clocking_signal: str
    lanes: tuple[LaneScene, ...]
    ticks: TickModel
    cuts: tuple[CutRegion, ...]
    events: tuple[VisualEvent, ...]
    constraints: tuple[VisualConstraint, ...]
    decorations: tuple["Decoration", ...] = ()
    visible_target: ScenarioDocument | None = field(default=None, compare=False, hash=False)
    semantic_document: ScenarioDocument | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lanes", tuple(self.lanes))
        object.__setattr__(self, "cuts", tuple(self.cuts))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "decorations", tuple(self.decorations))
