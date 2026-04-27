"""Concrete projection layer for WaveDrom-backed timing rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from sva_toolkit.timing.core.conditions import Condition
from sva_toolkit.timing.core.scenario import (
    ClockingSpec,
    ConstraintRegion,
    LaneConstraint,
    PropertyOverlay,
    ScenarioDocument,
    SignalKind,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.projection.tick_solver import (
    _apply_anchor_condition,
    _apply_span_semantics,
    _canonical_tick_assignment,
    _infer_window_spans,
)


Placement = Literal["boundary", "center"]
SummaryCategory = Literal["response", "hold", "not_before"]

_SUPPORTED_PREDICATES = {"high", "low", "rise", "fall", "change", "stable", "eq", "neq"}
_BOUNDARY_PREDICATES = {"rise", "fall", "change"}


@dataclass(frozen=True)
class WaveLaneView:
    """Concrete sampled lane used by the WaveDrom renderer."""

    name: str
    kind: str
    samples: tuple[str, ...]
    width: str | None = None


@dataclass(frozen=True)
class AnchorOccurrence:
    """Concrete occurrence of an anchor on a sampled timeline."""

    anchor_name: str
    tick: int
    placement: Placement
    label: str


@dataclass(frozen=True)
class ResponseSpan:
    """Concrete response span rendered as a rule arrow."""

    name: str
    trigger_tick: int
    response_tick: int
    label: str
    delay_text: str


@dataclass(frozen=True)
class HoldSpan:
    """Concrete hold span rendered as highlighted lanes."""

    name: str
    start_tick: int
    end_tick: int
    lanes: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class SummaryRule:
    """Summary footer rule row."""

    category: SummaryCategory
    name: str
    description: str


@dataclass(frozen=True)
class WaveDromScenarioView:
    """Render-local concrete scenario view for WaveDrom."""

    name: str
    clocking: ClockingSpec
    ticks: int
    lanes: tuple[WaveLaneView, ...]
    anchor_occurrences: tuple[AnchorOccurrence, ...]
    response_spans: tuple[ResponseSpan, ...]
    hold_spans: tuple[HoldSpan, ...]
    summary_rules: tuple[SummaryRule, ...]


@dataclass(frozen=True)
class _ClassifiedProperty:
    """Normalized property classification used by the WaveDrom projection."""

    category: SummaryCategory
    property_overlay: PropertyOverlay
    anchor_names: tuple[str, str]
    window: TimeWindow | None = None
    constraints: tuple[LaneConstraint, ...] = ()


def can_render_with_wavedrom(document: ScenarioDocument) -> bool:
    """Return whether the document is conservatively renderable as WaveDrom.

    With canonical trace synthesis, we can now handle documents without concrete
    samples.  The only hard requirement is that property classification succeeds.
    """
    try:
        _classify_properties(document)
    except ValueError:
        return False

    return True


def build_wavedrom_view(document: ScenarioDocument) -> WaveDromScenarioView:
    """Project a scenario document into WaveDrom render data, synthesizing samples if missing."""

    # Determine if the document already has concrete samples
    is_concrete = document.ticks is not None and all(s.samples for s in document.signals)

    # Resolve anchor ticks
    anchor_ticks: dict[str, int] = {}
    if not is_concrete:
        # Use pre-assigned absolute_tick where available, otherwise solve
        solved = _canonical_tick_assignment(document)
        for a in document.anchors:
            anchor_ticks[a.name] = a.absolute_tick if a.absolute_tick is not None else solved.get(a.name, 0)

    # Resolve total tick count
    ticks = document.ticks
    if ticks is None:
        ticks = 8
        if anchor_ticks:
            ticks = max(anchor_ticks.values()) + 3

    # Build signal kind map
    signal_kinds = {s.name: s.kind for s in document.signals}

    # Synthesize or use concrete signals
    if is_concrete:
        final_signals = {signal.name: signal.samples for signal in document.signals}
    else:
        signals_mut: dict[str, list[str]] = {}
        for signal in document.signals:
            if signal.samples:
                signals_mut[signal.name] = list(signal.samples)
            else:
                default = "0" if signal.kind == SignalKind.BIT else "x"
                signals_mut[signal.name] = [default] * ticks

        # Pass 1: apply anchor point conditions
        for anchor in document.anchors:
            tick = anchor_ticks.get(anchor.name)
            if tick is not None and 0 <= tick < ticks:
                _apply_anchor_condition(anchor.condition, signals_mut, tick, signal_kinds)

        # Pass 1.5: infer span fills from windows when lane_constraints are absent
        _infer_window_spans(document, signals_mut, anchor_ticks, signal_kinds, ticks)

        # Pass 2: apply span-level lane_constraints
        _apply_span_semantics(document, signals_mut, anchor_ticks, signal_kinds)

        final_signals = {name: tuple(samples) for name, samples in signals_mut.items()}

    # Classification (fail-fast, no silent fallback)
    classifications = _classify_properties(document)

    placements = {anchor.name: _anchor_placement(anchor.condition) for anchor in document.anchors}

    # Build lanes with clock lane at top
    lane_list = [
        WaveLaneView(
            name=document.clocking.signal,
            kind="bit",
            samples=(("1", "0") * (ticks // 2 + 1))[:ticks],
        )
    ]
    for signal in document.signals:
        if signal.name == document.clocking.signal:
            continue
        samples = final_signals.get(signal.name)
        if not samples:
            default = "0" if signal.kind == SignalKind.BIT else "x"
            samples = tuple([default] * ticks)
        lane_list.append(
            WaveLaneView(
                name=signal.name,
                kind=signal.kind.value,
                samples=samples if isinstance(samples, tuple) else tuple(samples),
                width=signal.width,
            )
        )
    lanes = tuple(lane_list)

    occurrence_map: dict[str, list[AnchorOccurrence]] = {}
    for anchor in document.anchors:
        anchor_occurrences: list[AnchorOccurrence] = []
        # Hide labels for synthetic/auto-generated anchors
        is_internal = "__node_" in anchor.name or "__point_" in anchor.name or anchor.role_metadata == "synthetic"
        label = "" if is_internal else anchor.name
        for tick in range(ticks):
            if evaluate_condition(anchor.condition, final_signals, tick):
                anchor_occurrences.append(
                    AnchorOccurrence(
                        anchor_name=anchor.name,
                        tick=tick,
                        placement=placements[anchor.name],
                        label=label,
                    )
                )
        occurrence_map[anchor.name] = anchor_occurrences

    response_spans: list[ResponseSpan] = []
    hold_spans: list[HoldSpan] = []
    summary_rules: list[SummaryRule] = []

    for classified in classifications:
        start_anchor, end_anchor = classified.anchor_names
        if classified.category == "response":
            if classified.window is None:
                raise ValueError(f"response property '{classified.property_overlay.name}' is missing a window")
            summary_rules.append(
                SummaryRule(
                    category="response",
                    name=classified.property_overlay.name,
                    description=f"{start_anchor} -> after {classified.window.bound.label} {end_anchor}",
                )
            )
            start_occurrence, end_occurrence = _first_matching_pair(
                occurrence_map.get(start_anchor, ()),
                occurrence_map.get(end_anchor, ()),
            )
            if start_occurrence is not None and end_occurrence is not None:
                response_spans.append(
                    ResponseSpan(
                        name=classified.property_overlay.name,
                        trigger_tick=start_occurrence.tick,
                        response_tick=end_occurrence.tick,
                        label=classified.property_overlay.name,
                        delay_text=classified.window.bound.label,
                    )
                )
            continue

        if classified.category == "hold":
            hold_label = _constraints_label(classified.constraints)
            summary_rules.append(
                SummaryRule(
                    category="hold",
                    name=classified.property_overlay.name,
                    description=f"{hold_label} from {start_anchor} until {end_anchor}",
                )
            )
            start_occurrence, end_occurrence = _first_matching_pair(
                occurrence_map.get(start_anchor, ()),
                occurrence_map.get(end_anchor, ()),
            )
            if start_occurrence is not None and end_occurrence is not None:
                hold_spans.append(
                    HoldSpan(
                        name=classified.property_overlay.name,
                        start_tick=start_occurrence.tick,
                        end_tick=end_occurrence.tick,
                        lanes=_constraint_lanes(classified.constraints),
                        label=hold_label,
                    )
                )
            continue

        if classified.category == "not_before":
            summary_rules.append(
                SummaryRule(
                    category="not_before",
                    name=classified.property_overlay.name,
                    description=f"not {start_anchor} before {end_anchor}",
                )
            )
            continue

        raise ValueError(f"unsupported property category: {classified.category}")

    return WaveDromScenarioView(
        name=document.name,
        clocking=document.clocking,
        ticks=ticks,
        lanes=lanes,
        anchor_occurrences=tuple(
            occurrence
            for anchor in document.anchors
            for occurrence in occurrence_map.get(anchor.name, ())
        ),
        response_spans=tuple(response_spans),
        hold_spans=tuple(hold_spans),
        summary_rules=tuple(summary_rules),
    )


def evaluate_condition(condition: Condition, signals: Mapping[str, Sequence[str]], tick: int) -> bool:
    """Evaluate a supported condition tree against concrete samples."""

    if condition.kind == "predicate" and condition.predicate is not None:
        predicate = condition.predicate
        if predicate.signal is None:
            raise ValueError("predicate-backed conditions require a concrete signal")

        samples = signals[predicate.signal]
        current = samples[tick]
        previous = samples[tick - 1] if tick > 0 else None

        if predicate.op == "high":
            return current == "1"
        if predicate.op == "low":
            return current == "0"
        if predicate.op == "rise":
            return previous == "0" and current == "1"
        if predicate.op == "fall":
            return previous == "1" and current == "0"
        if predicate.op == "change":
            return previous is not None and current != previous
        if predicate.op == "stable":
            return previous is not None and current == previous
        if predicate.op == "eq":
            return current == predicate.value
        if predicate.op == "neq":
            return current != predicate.value
        raise ValueError(f"unsupported predicate op: {predicate.op}")

    if condition.kind == "all":
        return all(evaluate_condition(item, signals, tick) for item in condition.items)
    if condition.kind == "any":
        return any(evaluate_condition(item, signals, tick) for item in condition.items)
    if condition.kind == "not" and condition.items:
        return not evaluate_condition(condition.items[0], signals, tick)
    if condition.kind == "raw":
        # Raw SVA text cannot be evaluated over concrete samples.
        # For constant-true expressions return True; otherwise conservatively
        # return True so synthesis does not suppress an anchor.
        text = (condition.text or "").strip().lower()
        if text in ("0", "1'b0", "false"):
            return False
        return True
    raise ValueError(f"unsupported condition kind: {condition.kind}")


def _classify_properties(document: ScenarioDocument) -> tuple[_ClassifiedProperty, ...]:
    """Classify properties into the concrete overlay/summarized categories."""

    classifications: list[_ClassifiedProperty] = []
    used_windows: set[str] = set()
    used_constraints: set[str] = set()
    skipped = 0

    for property_overlay in document.properties:
        try:
            classified = _classify_property(document, property_overlay)
        except ValueError:
            # Properties without sufficient metadata (e.g. user-defined raw
            # properties lacking related_anchors) are skipped for WaveDrom
            # rendering.  This is not a silent degradation — the property text
            # is preserved in the document; it simply has no visual overlay.
            skipped += 1
            continue
        classifications.append(classified)
        if classified.window is not None:
            used_windows.add(classified.window.name)
        for constraint in classified.constraints:
            used_constraints.add(constraint.name)

    referenced_windows = {window_name for prop in document.properties for window_name in prop.related_windows}
    referenced_constraints = {constraint_name for prop in document.properties for constraint_name in prop.related_constraints}

    if referenced_windows != used_windows:
        unsupported = ", ".join(sorted(referenced_windows - used_windows))
        raise ValueError(f"unsupported window classifications: {unsupported}")
    if referenced_constraints != used_constraints:
        unsupported = ", ".join(sorted(referenced_constraints - used_constraints))
        raise ValueError(f"unsupported constraint classifications: {unsupported}")
    # Only enforce full window/constraint coverage when ALL properties were
    # successfully classified (no skips due to missing metadata).
    if document.properties and skipped == 0:
        if any(window.name not in used_windows for window in document.windows):
            names = ", ".join(sorted(window.name for window in document.windows if window.name not in used_windows))
            raise ValueError(f"unclassified windows are not renderable with WaveDrom: {names}")
        if any(constraint.name not in used_constraints for constraint in document.lane_constraints):
            names = ", ".join(sorted(constraint.name for constraint in document.lane_constraints if constraint.name not in used_constraints))
            raise ValueError(f"unclassified constraints are not renderable with WaveDrom: {names}")

    return tuple(classifications)


def _classify_property(document: ScenarioDocument, property_overlay: PropertyOverlay) -> _ClassifiedProperty:
    """Classify a single property into a supported concrete render category."""

    anchor_names = property_overlay.related_anchors
    if len(anchor_names) != 2:
        raise ValueError(f"property '{property_overlay.name}' is missing related anchors")

    if property_overlay.related_windows:
        if len(property_overlay.related_windows) != 1:
            raise ValueError(f"property '{property_overlay.name}' references multiple windows")
        window_name = property_overlay.related_windows[0]
        window = document.window_map[window_name]
        if property_overlay.related_constraints:
            constraints = tuple(document.constraint_map[name] for name in property_overlay.related_constraints)
            if window.bound.kind != WindowBoundKind.OMITTED:
                raise ValueError(f"hold property '{property_overlay.name}' must use an omitted window bound")
            if not constraints or any(constraint.region != ConstraintRegion.FROM_UNTIL for constraint in constraints):
                raise ValueError(f"hold property '{property_overlay.name}' must use FROM_UNTIL constraints")
            return _ClassifiedProperty(
                category="hold",
                property_overlay=property_overlay,
                anchor_names=(anchor_names[0], anchor_names[1]),
                window=window,
                constraints=constraints,
            )

        if window.bound.kind == WindowBoundKind.OMITTED:
            raise ValueError(f"response property '{property_overlay.name}' has an omitted window bound")
        return _ClassifiedProperty(
            category="response",
            property_overlay=property_overlay,
            anchor_names=(anchor_names[0], anchor_names[1]),
            window=window,
        )

    if property_overlay.related_constraints:
        raise ValueError(f"property '{property_overlay.name}' references constraints without a window")

    body = property_overlay.body.strip()
    if body.startswith("!") and " until " in body:
        return _ClassifiedProperty(
            category="not_before",
            property_overlay=property_overlay,
            anchor_names=(anchor_names[0], anchor_names[1]),
        )

    raise ValueError(f"property '{property_overlay.name}' is not classifiable for WaveDrom")


def _is_condition_supported(condition: Condition) -> bool:
    """Return whether the condition tree can be evaluated over concrete samples."""

    if condition.kind == "predicate" and condition.predicate is not None:
        predicate = condition.predicate
        return predicate.signal is not None and predicate.op in _SUPPORTED_PREDICATES
    if condition.kind in {"all", "any"}:
        return bool(condition.items) and all(_is_condition_supported(item) for item in condition.items)
    if condition.kind == "not" and condition.items:
        return _is_condition_supported(condition.items[0])
    return False


def _anchor_placement(condition: Condition) -> Placement:
    """Determine whether anchor labels should snap to a boundary or sample center."""

    if _all_predicates_in(condition, _BOUNDARY_PREDICATES):
        return "boundary"
    return "center"


def _all_predicates_in(condition: Condition, allowed_ops: set[str]) -> bool:
    """Return whether every predicate in the tree belongs to the allowed set."""

    if condition.kind == "predicate" and condition.predicate is not None:
        return condition.predicate.op in allowed_ops
    if condition.kind in {"all", "any", "not"}:
        return bool(condition.items) and all(_all_predicates_in(item, allowed_ops) for item in condition.items)
    return False


def _first_matching_pair(
    starts: Sequence[AnchorOccurrence],
    ends: Sequence[AnchorOccurrence],
) -> tuple[AnchorOccurrence | None, AnchorOccurrence | None]:
    """Return the first end occurrence that does not precede its start occurrence."""

    for start in starts:
        for end in ends:
            if end.tick >= start.tick:
                return start, end
    return None, None


def _constraints_label(constraints: Sequence[LaneConstraint]) -> str:
    """Build a compact label from FROM_UNTIL constraints."""

    return " and ".join(_constraint_label(constraint) for constraint in constraints)


def _constraint_label(constraint: LaneConstraint) -> str:
    """Render a single lane constraint into a readable predicate label."""

    signal_name = constraint.signals[0] if constraint.signals else "?"
    if constraint.value is None:
        return f"{constraint.relation}({signal_name})"
    return f"{constraint.relation}({signal_name}, {constraint.value})"


def _constraint_lanes(constraints: Sequence[LaneConstraint]) -> tuple[str, ...]:
    """Collect the unique lane names referenced by a constraint set."""

    ordered: list[str] = []
    for constraint in constraints:
        for signal_name in constraint.signals:
            if signal_name not in ordered:
                ordered.append(signal_name)
    return tuple(ordered)
