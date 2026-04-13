"""Canonical timing scenario IR used by parsing, extraction, emission, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from sva_toolkit.sva.ast import ExprNode, PropertyNode
from sva_toolkit.timing.core.conditions import Condition


class ExtractionStatus(str, Enum):
    """Reverse-extraction confidence level."""

    EXACT = "exact"
    LOSSY = "lossy"
    UNSUPPORTED = "unsupported"


class SignalKind(str, Enum):
    """Supported rendered signal kinds."""

    BIT = "bit"
    BUS = "bus"


class AnchorRole(str, Enum):
    """Semantic role attached to a scenario anchor."""

    TRIGGER = "trigger"
    RESPONSE = "response"
    STATE = "state"
    LOOKBACK = "lookback"
    SYNTHETIC = "synthetic"


class WindowBoundKind(str, Enum):
    """Kinds of timeline window bounds."""

    EXACT = "exact"
    RANGE = "range"
    UNBOUNDED = "unbounded"
    OMITTED = "omitted"


class CutPlacement(str, Enum):
    """Placement of a cut marker on the timeline."""

    BEFORE_ANCHOR = "before_anchor"
    AFTER_ANCHOR = "after_anchor"
    BETWEEN_WINDOWS = "between_windows"


class CutMeaning(str, Enum):
    """Meaning rendered by a cut marker."""

    OMITTED_HISTORY = "omitted_history"
    OMITTED_FUTURE = "omitted_future"
    SYMBOLIC_GAP = "symbolic_gap"
    LOOKBACK = "lookback"


class ConstraintRegion(str, Enum):
    """Where a lane constraint applies."""

    AT = "at"
    IN = "in"
    BEFORE = "before"
    AFTER = "after"
    FROM_UNTIL = "from_until"


@dataclass(frozen=True)
class ClockingSpec:
    """Clocking context for a scenario."""

    edge: str
    signal: str
    disable_iff: str | None = None
    disable_iff_ast: ExprNode | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ParameterDecl:
    """Parameterized symbol used by scenarios and emitted SVA."""

    name: str
    kind: str = "int"


@dataclass(frozen=True)
class SignalDecl:
    """Signal declaration with optional concrete samples."""

    name: str
    kind: SignalKind
    width: str | None = None
    samples: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        if self.width and not (self.kind == SignalKind.BIT and self.width == "1"):
            return f"{self.name}[{self.width}]"
        return self.name

    @property
    def is_symbolic(self) -> bool:
        return not self.samples


@dataclass(frozen=True)
class Anchor:
    """Named origin point on the scenario timeline."""

    name: str
    condition: Condition
    role: AnchorRole = AnchorRole.STATE
    derived_from: str | None = None
    role_metadata: str | None = None
    absolute_tick: int | None = None


@dataclass(frozen=True)
class TimeBound:
    """Exact, ranged, or unbounded time distance between anchors."""

    kind: WindowBoundKind
    min_delay: str | None = None
    max_delay: str | None = None
    inclusive: bool = True

    @property
    def label(self) -> str:
        if self.kind == WindowBoundKind.EXACT:
            return self.min_delay or "0"
        if self.kind == WindowBoundKind.RANGE:
            return f"[{self.min_delay}:{self.max_delay}]"
        if self.kind == WindowBoundKind.UNBOUNDED:
            return f"[{self.min_delay}:$]"
        return "omitted"


@dataclass(frozen=True)
class TimeWindow:
    """Named temporal interval between two anchors."""

    name: str
    start_anchor: str
    end_anchor: str
    bound: TimeBound


@dataclass(frozen=True)
class Cut:
    """Explicit omitted or compressed region marker."""

    name: str
    placement: CutPlacement
    meaning: CutMeaning
    anchor: str | None = None
    left_window: str | None = None
    right_window: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class LaneConstraint:
    """Typed semantic/display fact attached to one or more signals."""

    name: str
    signals: tuple[str, ...]
    relation: str
    region: ConstraintRegion
    value: str | None = None
    anchor: str | None = None
    window: str | None = None
    start_anchor: str | None = None
    end_anchor: str | None = None
    display_only: bool = True


@dataclass(frozen=True)
class PropertyOverlay:
    """Stored property semantics and extraction metadata."""

    name: str
    body: str
    status: ExtractionStatus = ExtractionStatus.EXACT
    source: str | None = None
    related_anchors: tuple[str, ...] = ()
    related_windows: tuple[str, ...] = ()
    related_constraints: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    body_ast: PropertyNode | None = field(default=None, compare=False)


@dataclass(frozen=True)
class BundleMetadata:
    """Metadata attached to grouped scenarios."""

    source_names: tuple[str, ...] = ()
    group_key: tuple[str, ...] = ()
    signal_overlap: float = 1.0


@dataclass(frozen=True)
class ScenarioDocument:
    """Canonical dual-direction timing scenario document."""

    name: str
    clocking: ClockingSpec
    params: tuple[ParameterDecl, ...] = ()
    signals: tuple[SignalDecl, ...] = ()
    anchors: tuple[Anchor, ...] = ()
    windows: tuple[TimeWindow, ...] = ()
    cuts: tuple[Cut, ...] = ()
    lane_constraints: tuple[LaneConstraint, ...] = ()
    properties: tuple[PropertyOverlay, ...] = ()
    extraction_status: ExtractionStatus = ExtractionStatus.EXACT
    bundle: BundleMetadata = field(default_factory=BundleMetadata)
    notes: tuple[str, ...] = ()
    ticks: int | None = None
    ceg: object | None = None  # CanonicalEventGraph (type deferred to avoid circular import)

    @property
    def signal_map(self) -> dict[str, SignalDecl]:
        return {signal.name: signal for signal in self.signals}

    @property
    def anchor_map(self) -> dict[str, Anchor]:
        return {anchor.name: anchor for anchor in self.anchors}

    @property
    def window_map(self) -> dict[str, TimeWindow]:
        return {window.name: window for window in self.windows}

    @property
    def constraint_map(self) -> dict[str, LaneConstraint]:
        return {constraint.name: constraint for constraint in self.lane_constraints}

    @property
    def property_map(self) -> dict[str, PropertyOverlay]:
        return {prop.name: prop for prop in self.properties}

    @property
    def is_concrete(self) -> bool:
        return self.ticks is not None and all(signal.samples for signal in self.signals)

    @property
    def has_symbolic_features(self) -> bool:
        if self.windows or self.cuts:
            return True
        if any(signal.is_symbolic for signal in self.signals):
            return True
        return False

    @property
    def effective_status(self) -> ExtractionStatus:
        statuses = [self.extraction_status, *(prop.status for prop in self.properties)]
        if ExtractionStatus.UNSUPPORTED in statuses:
            return ExtractionStatus.UNSUPPORTED
        if ExtractionStatus.LOSSY in statuses:
            return ExtractionStatus.LOSSY
        return ExtractionStatus.EXACT


def merge_bundle_metadata(documents: tuple[ScenarioDocument, ...], signal_overlap: float) -> BundleMetadata:
    """Build deterministic bundle metadata from grouped documents."""

    if not documents:
        return BundleMetadata()
    group_key = (
        documents[0].clocking.edge,
        documents[0].clocking.signal,
        documents[0].clocking.disable_iff or "",
    )
    return BundleMetadata(
        source_names=tuple(document.name for document in documents),
        group_key=group_key,
        signal_overlap=signal_overlap,
    )


def normalize_signal_width(kind: SignalKind, width: str | None) -> str | None:
    """Normalize absent bit widths to `1` while preserving symbolic bus widths."""

    if kind == SignalKind.BIT:
        return width or "1"
    return width


def rebind_names(mapping: Mapping[str, str], text: str) -> str:
    """Apply a deterministic textual identifier rename map."""

    updated = text
    for old_name, new_name in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        updated = updated.replace(old_name, new_name)
    return updated
