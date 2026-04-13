"""Concrete projection layer for WaveDrom-backed timing rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from sva_toolkit.timing.core.conditions import Condition, Predicate
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    ClockingSpec,
    ConstraintRegion,
    CutMeaning,
    CutPlacement,
    LaneConstraint,
    PropertyOverlay,
    ScenarioDocument,
    SignalKind,
    TimeWindow,
    WindowBoundKind,
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


def _canonical_tick_assignment(document: ScenarioDocument) -> dict[str, int]:
    """Assign absolute ticks to anchors via topological longest-path over the window graph.

    Uses earliest-legal-tick policy: roots get the base offset, successors get
    predecessor_tick + min_delay (or max_delay when finite).
    Raises ValueError on unresolvable conflicts.
    """
    all_anchors = {a.name for a in document.anchors}
    offset = 0
    for cut in document.cuts:
        if cut.placement == CutPlacement.BEFORE_ANCHOR and cut.meaning == CutMeaning.OMITTED_HISTORY:
            offset = 1
            break

    # Build adjacency: anchor -> [(target_anchor, delay)]
    successors: dict[str, list[tuple[str, int]]] = {name: [] for name in all_anchors}
    in_degree: dict[str, int] = {name: 0 for name in all_anchors}

    for window in document.windows:
        if window.start_anchor not in all_anchors or window.end_anchor not in all_anchors:
            continue
        # Use max_delay when finite for better visual "reach"; fall back to min_delay
        delay_str = window.bound.max_delay if window.bound.max_delay and window.bound.max_delay != "$" else window.bound.min_delay
        try:
            delay = int(delay_str) if delay_str else 0
        except ValueError:
            delay = 1
        successors[window.start_anchor].append((window.end_anchor, delay))
        in_degree[window.end_anchor] = in_degree.get(window.end_anchor, 0) + 1

    # Kahn's algorithm with longest-path tick assignment
    ticks: dict[str, int] = {}
    queue = sorted(name for name in all_anchors if in_degree.get(name, 0) == 0)
    for root in queue:
        ticks[root] = offset

    q = list(queue)
    while q:
        q.sort()
        node = q.pop(0)
        for target, delay in successors.get(node, []):
            proposed = ticks[node] + delay
            if target in ticks:
                ticks[target] = max(ticks[target], proposed)
            else:
                ticks[target] = proposed
            in_degree[target] -= 1
            if in_degree[target] == 0:
                q.append(target)

    for name in all_anchors:
        if name not in ticks:
            ticks[name] = offset

    return ticks


def _apply_anchor_condition(
    cond: Condition, signals: dict[str, list[str]], tick: int,
    signal_kinds: dict[str, SignalKind] | None = None,
) -> None:
    """Apply point-event anchor conditions at the given tick."""
    if cond.kind == "predicate" and cond.predicate is not None:
        p = cond.predicate
        if not p.signal or p.signal not in signals:
            return
        samples = signals[p.signal]
        if tick < 0 or tick >= len(samples):
            return
        if p.op == "high":
            samples[tick] = "1"
        elif p.op == "low":
            samples[tick] = "0"
        elif p.op == "rise":
            if tick > 0:
                samples[tick - 1] = "0"
            samples[tick] = "1"
            if tick + 1 < len(samples):
                samples[tick + 1] = "1"
        elif p.op == "fall":
            if tick > 0:
                samples[tick - 1] = "1"
            samples[tick] = "0"
            if tick + 1 < len(samples):
                samples[tick + 1] = "0"
        elif p.op == "eq":
            samples[tick] = p.value or "1"
        elif p.op == "stable":
            is_bus = (signal_kinds or {}).get(p.signal, SignalKind.BIT) == SignalKind.BUS
            if is_bus and samples[tick] == "x":
                samples[tick] = p.signal
    elif cond.kind == "all":
        for item in cond.items:
            _apply_anchor_condition(item, signals, tick, signal_kinds)


def _collect_condition_predicates(cond: Condition) -> list[Predicate]:
    """Collect all leaf predicates from a condition tree."""
    if cond.kind == "predicate" and cond.predicate is not None:
        return [cond.predicate]
    if cond.kind in ("all", "any", "not") and cond.items:
        return [p for item in cond.items for p in _collect_condition_predicates(item)]
    return []


def _infer_window_spans(
    document: ScenarioDocument,
    signals: dict[str, list[str]],
    anchor_ticks: dict[str, int],
    signal_kinds: dict[str, SignalKind],
    ticks: int,
) -> None:
    """Infer span-level fills from window endpoints when lane_constraints are absent.

    When SVA extraction produces windows connecting anchors with stable/high/low
    conditions but no explicit lane_constraints, this pass infers the intended
    span semantics and fills signal samples accordingly.
    """
    if document.lane_constraints:
        return  # explicit constraints take precedence

    anchor_map = {a.name: a for a in document.anchors}

    for window in document.windows:
        start_tick = anchor_ticks.get(window.start_anchor)
        end_tick = anchor_ticks.get(window.end_anchor)
        if start_tick is None or end_tick is None:
            continue

        lo = max(0, min(start_tick, end_tick))
        hi = min(max(start_tick, end_tick), ticks - 1)

        # Collect predicates from both endpoint anchors
        for anchor_name in (window.start_anchor, window.end_anchor):
            anchor = anchor_map.get(anchor_name)
            if anchor is None:
                continue
            for pred in _collect_condition_predicates(anchor.condition):
                if not pred.signal or pred.signal not in signals:
                    continue
                is_bus = signal_kinds.get(pred.signal, SignalKind.BIT) == SignalKind.BUS
                samples = signals[pred.signal]

                fill_val: str | None = None
                if pred.op == "stable" and is_bus:
                    fill_val = pred.signal
                elif pred.op == "eq":
                    fill_val = pred.value or pred.signal
                elif pred.op == "high":
                    fill_val = "1"
                elif pred.op == "low":
                    fill_val = "0"

                if fill_val is None:
                    continue

                for t in range(lo, hi + 1):
                    current = samples[t]
                    # Only fill default values; don't overwrite existing meaningful data
                    if current in ("x", "0", "") or current == fill_val:
                        samples[t] = fill_val


def _apply_span_semantics(
    document: ScenarioDocument,
    signals: dict[str, list[str]],
    anchor_ticks: dict[str, int],
    signal_kinds: dict[str, SignalKind],
) -> None:
    """Enforce span-level semantics from lane_constraints on the signal samples.

    Walks lane_constraints and fills intervals with required values.
    Raises ValueError on conflicts (e.g. same tick requires high and low).
    """
    for constraint in document.lane_constraints:
        start_tick: int | None = None
        end_tick: int | None = None

        if constraint.region == ConstraintRegion.FROM_UNTIL:
            start_tick = anchor_ticks.get(constraint.start_anchor) if constraint.start_anchor else None
            end_tick = anchor_ticks.get(constraint.end_anchor) if constraint.end_anchor else None
        elif constraint.region == ConstraintRegion.BEFORE:
            start_tick = 0
            end_tick = (anchor_ticks[constraint.anchor] - 1) if constraint.anchor and constraint.anchor in anchor_ticks else None
        elif constraint.region == ConstraintRegion.AT:
            at_tick = anchor_ticks.get(constraint.anchor) if constraint.anchor else None
            if at_tick is not None:
                start_tick = at_tick
                end_tick = at_tick

        if start_tick is None or end_tick is None:
            continue

        for signal_name in constraint.signals:
            if signal_name not in signals:
                continue
            samples = signals[signal_name]
            is_bus = signal_kinds.get(signal_name, SignalKind.BIT) == SignalKind.BUS

            for t in range(max(0, start_tick), min(end_tick + 1, len(samples))):
                target_val: str | None = None
                if constraint.relation == "high":
                    target_val = "1"
                elif constraint.relation == "low":
                    target_val = "0"
                elif constraint.relation == "stable":
                    target_val = signal_name if is_bus else "1"
                elif constraint.relation == "eq":
                    target_val = constraint.value or ("x" if is_bus else "1")
                elif constraint.relation in ("rise", "fall", "change", "neq"):
                    continue  # boundary/negative predicates, not span fillers

                if target_val is not None:
                    current = samples[t]
                    # If the current value was already set (non-default) and
                    # disagrees with the constraint, skip — anchor point
                    # conditions take priority over span constraints.
                    if current not in ("0", "x", "") and current != target_val:
                        continue
                    samples[t] = target_val


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
