"""Canonical timing scenario IR used by parsing, extraction, emission, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

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
    disable_iff: Optional[str] = None


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
    width: Optional[str] = None
    samples: Tuple[str, ...] = ()

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
    derived_from: Optional[str] = None


@dataclass(frozen=True)
class TimeBound:
    """Exact, ranged, or unbounded time distance between anchors."""

    kind: WindowBoundKind
    min_delay: Optional[str] = None
    max_delay: Optional[str] = None
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
    anchor: Optional[str] = None
    left_window: Optional[str] = None
    right_window: Optional[str] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class LaneConstraint:
    """Typed semantic/display fact attached to one or more signals."""

    name: str
    signals: Tuple[str, ...]
    relation: str
    region: ConstraintRegion
    value: Optional[str] = None
    anchor: Optional[str] = None
    window: Optional[str] = None
    start_anchor: Optional[str] = None
    end_anchor: Optional[str] = None
    display_only: bool = True


@dataclass(frozen=True)
class PropertyOverlay:
    """Stored property semantics and extraction metadata."""

    name: str
    body: str
    status: ExtractionStatus = ExtractionStatus.EXACT
    source: Optional[str] = None
    related_anchors: Tuple[str, ...] = ()
    related_windows: Tuple[str, ...] = ()
    related_constraints: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BundleMetadata:
    """Metadata attached to grouped scenarios."""

    source_names: Tuple[str, ...] = ()
    group_key: Tuple[str, ...] = ()
    signal_overlap: float = 1.0


@dataclass(frozen=True)
class ScenarioDocument:
    """Canonical dual-direction timing scenario document."""

    name: str
    clocking: ClockingSpec
    params: Tuple[ParameterDecl, ...] = ()
    signals: Tuple[SignalDecl, ...] = ()
    anchors: Tuple[Anchor, ...] = ()
    windows: Tuple[TimeWindow, ...] = ()
    cuts: Tuple[Cut, ...] = ()
    lane_constraints: Tuple[LaneConstraint, ...] = ()
    properties: Tuple[PropertyOverlay, ...] = ()
    extraction_status: ExtractionStatus = ExtractionStatus.EXACT
    bundle: BundleMetadata = field(default_factory=BundleMetadata)
    notes: Tuple[str, ...] = ()
    ticks: Optional[int] = None
    legacy_diagram: object | None = None

    @property
    def signal_map(self) -> Dict[str, SignalDecl]:
        return {signal.name: signal for signal in self.signals}

    @property
    def anchor_map(self) -> Dict[str, Anchor]:
        return {anchor.name: anchor for anchor in self.anchors}

    @property
    def window_map(self) -> Dict[str, TimeWindow]:
        return {window.name: window for window in self.windows}

    @property
    def constraint_map(self) -> Dict[str, LaneConstraint]:
        return {constraint.name: constraint for constraint in self.lane_constraints}

    @property
    def property_map(self) -> Dict[str, PropertyOverlay]:
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

    def with_legacy_diagram(self, legacy_diagram: object) -> "ScenarioDocument":
        """Return a copy with the legacy concrete diagram adapter attached."""

        return ScenarioDocument(
            name=self.name,
            clocking=self.clocking,
            params=self.params,
            signals=self.signals,
            anchors=self.anchors,
            windows=self.windows,
            cuts=self.cuts,
            lane_constraints=self.lane_constraints,
            properties=self.properties,
            extraction_status=self.extraction_status,
            bundle=self.bundle,
            notes=self.notes,
            ticks=self.ticks,
            legacy_diagram=legacy_diagram,
        )


def merge_bundle_metadata(documents: Tuple[ScenarioDocument, ...], signal_overlap: float) -> BundleMetadata:
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


def normalize_signal_width(kind: SignalKind, width: Optional[str]) -> Optional[str]:
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
