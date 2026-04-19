"""Render-oriented projection for symbolic timing scenarios."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sva_toolkit.timing.core.conditions import condition_to_dsl
from sva_toolkit.timing.core.scenario import (
    AnchorRole,
    ConstraintRegion,
    Cut,
    CutMeaning,
    CutPlacement,
    ExtractionStatus,
    LaneConstraint,
    ScenarioDocument,
)


@dataclass(frozen=True)
class TimelineItem:
    """One visible x-axis primitive in the symbolic renderer."""

    kind: str
    name: str
    label: str
    detail: str | None = None
    anchor: str | None = None
    window: str | None = None
    start_anchor: str | None = None
    end_anchor: str | None = None
    role: AnchorRole | None = None
    placement: CutPlacement | None = None
    meaning: CutMeaning | None = None


@dataclass(frozen=True)
class LaneConstraintView:
    """Resolved lane constraint metadata for rendering."""

    name: str
    label: str
    detail: str
    relation: str
    region: ConstraintRegion
    value: str | None = None
    anchor: str | None = None
    window: str | None = None
    start_anchor: str | None = None
    end_anchor: str | None = None
    display_only: bool = True
    property_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class SignalLaneView:
    """Render-oriented grouping of lane facts."""

    signal_name: str
    display_name: str
    constraints: tuple[LaneConstraintView, ...]


@dataclass(frozen=True)
class PropertyView:
    """Property footer row used by the symbolic renderer."""

    name: str
    body: str
    status: ExtractionStatus
    notes: tuple[str, ...]
    related_anchors: tuple[str, ...]
    related_windows: tuple[str, ...]
    related_constraints: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioView:
    """Normalized symbolic render view."""

    document: ScenarioDocument
    timeline: tuple[TimelineItem, ...]
    lanes: tuple[SignalLaneView, ...]
    properties: tuple[PropertyView, ...]
    anchor_order: tuple[str, ...]
    clocking_label: str


def build_scenario_view(document: ScenarioDocument) -> ScenarioView:
    """Build a symbolic view of the scenario timeline and lane facts."""

    anchor_order = _order_anchors(document)
    anchor_map = document.anchor_map

    before_cuts: dict[str, list[TimelineItem]] = defaultdict(list)
    after_cuts: dict[str, list[TimelineItem]] = defaultdict(list)
    before_window_cuts: dict[str, list[TimelineItem]] = defaultdict(list)
    after_window_cuts: dict[str, list[TimelineItem]] = defaultdict(list)
    floating_cuts: list[TimelineItem] = []

    for cut in document.cuts:
        item = TimelineItem(
            kind="cut",
            name=cut.name,
            label=cut.label or _cut_label(cut),
            detail=_cut_detail(cut),
            anchor=cut.anchor,
            placement=cut.placement,
            meaning=cut.meaning,
        )
        if cut.placement == CutPlacement.BEFORE_ANCHOR and cut.anchor:
            before_cuts[cut.anchor].append(item)
        elif cut.placement == CutPlacement.AFTER_ANCHOR and cut.anchor:
            after_cuts[cut.anchor].append(item)
        elif cut.placement == CutPlacement.BETWEEN_WINDOWS:
            if cut.left_window:
                after_window_cuts[cut.left_window].append(item)
            elif cut.right_window:
                before_window_cuts[cut.right_window].append(item)
            else:
                floating_cuts.append(item)
        else:
            floating_cuts.append(item)

    windows_by_start: dict[str, list[TimelineItem]] = defaultdict(list)
    for window in document.windows:
        windows_by_start[window.start_anchor].append(
            TimelineItem(
                kind="window",
                name=window.name,
                label=window.name,
                detail=window.bound.label,
                window=window.name,
                start_anchor=window.start_anchor,
                end_anchor=window.end_anchor,
            )
        )

    timeline: list[TimelineItem] = []
    seen_windows: set[str] = set()
    seen_cuts: set[str] = set()
    anchor_order_index = {name: index for index, name in enumerate(anchor_order)}

    for anchor_name in anchor_order:
        timeline.extend(_sorted_timeline_items(before_cuts.get(anchor_name, ())))
        seen_cuts.update(item.name for item in before_cuts.get(anchor_name, ()))

        anchor = anchor_map.get(anchor_name)
        timeline.append(
            TimelineItem(
                kind="anchor",
                name=anchor_name,
                label=anchor_name,
                detail=condition_to_dsl(anchor.condition) if anchor is not None else "undefined anchor",
                anchor=anchor_name,
                role=anchor.role if anchor is not None else AnchorRole.SYNTHETIC,
            )
        )

        for window_item in sorted(
            windows_by_start.get(anchor_name, ()),
            key=lambda item: (anchor_order_index.get(item.end_anchor or "", len(anchor_order_index)), item.name),
        ):
            timeline.extend(_sorted_timeline_items(before_window_cuts.get(window_item.window or "", ())))
            seen_cuts.update(item.name for item in before_window_cuts.get(window_item.window or "", ()))
            timeline.append(window_item)
            seen_windows.add(window_item.name)
            timeline.extend(_sorted_timeline_items(after_window_cuts.get(window_item.window or "", ())))
            seen_cuts.update(item.name for item in after_window_cuts.get(window_item.window or "", ()))

        timeline.extend(_sorted_timeline_items(after_cuts.get(anchor_name, ())))
        seen_cuts.update(item.name for item in after_cuts.get(anchor_name, ()))

    for window in sorted(document.windows, key=lambda item: item.name):
        if window.name in seen_windows:
            continue
        timeline.append(
            TimelineItem(
                kind="window",
                name=window.name,
                label=window.name,
                detail=window.bound.label,
                window=window.name,
                start_anchor=window.start_anchor,
                end_anchor=window.end_anchor,
            )
        )

    for cut_item in sorted(floating_cuts, key=lambda item: item.name):
        if cut_item.name in seen_cuts:
            continue
        timeline.append(cut_item)

    lanes = _build_lane_views(document)
    properties = tuple(
        PropertyView(
            name=prop.name,
            body=prop.body,
            status=prop.status,
            notes=prop.notes,
            related_anchors=prop.related_anchors,
            related_windows=prop.related_windows,
            related_constraints=prop.related_constraints,
        )
        for prop in document.properties
    )

    return ScenarioView(
        document=document,
        timeline=tuple(timeline),
        lanes=lanes,
        properties=properties,
        anchor_order=anchor_order,
        clocking_label=_clocking_label(document),
    )


def _build_lane_views(document: ScenarioDocument) -> tuple[SignalLaneView, ...]:
    properties_by_constraint: dict[str, list[str]] = defaultdict(list)
    for prop in document.properties:
        for constraint_name in prop.related_constraints:
            properties_by_constraint[constraint_name].append(prop.name)

    constraints_by_signal: dict[str, list[LaneConstraint]] = defaultdict(list)
    for constraint in document.lane_constraints:
        for signal in constraint.signals:
            constraints_by_signal[signal].append(constraint)

    signal_order = [signal.name for signal in document.signals]
    extras = sorted(name for name in constraints_by_signal if name not in document.signal_map)
    signal_order.extend(extras)

    lanes: list[SignalLaneView] = []
    for signal_name in signal_order:
        signal = document.signal_map.get(signal_name)
        display_name = signal.display_name if signal is not None else signal_name
        constraints = tuple(
            _constraint_view(constraint, tuple(properties_by_constraint.get(constraint.name, ())))
            for constraint in sorted(constraints_by_signal.get(signal_name, ()), key=_constraint_sort_key)
        )
        lanes.append(
            SignalLaneView(
                signal_name=signal_name,
                display_name=display_name,
                constraints=constraints,
            )
        )
    return tuple(lanes)


def _constraint_view(constraint: LaneConstraint, property_names: tuple[str, ...]) -> LaneConstraintView:
    return LaneConstraintView(
        name=constraint.name,
        label=_constraint_label(constraint),
        detail=_constraint_detail(constraint, property_names),
        relation=constraint.relation,
        region=constraint.region,
        value=constraint.value,
        anchor=constraint.anchor,
        window=constraint.window,
        start_anchor=constraint.start_anchor,
        end_anchor=constraint.end_anchor,
        display_only=constraint.display_only,
        property_names=property_names,
    )


def _order_anchors(document: ScenarioDocument) -> tuple[str, ...]:
    dependencies: dict[str, set[str]] = {}
    followers: dict[str, set[str]] = {}

    for anchor in document.anchors:
        dependencies.setdefault(anchor.name, set())
        followers.setdefault(anchor.name, set())

    for window in document.windows:
        dependencies.setdefault(window.start_anchor, set())
        dependencies.setdefault(window.end_anchor, set()).add(window.start_anchor)
        followers.setdefault(window.start_anchor, set()).add(window.end_anchor)
        followers.setdefault(window.end_anchor, set())

    original_index = {anchor.name: index for index, anchor in enumerate(document.anchors)}
    pending = {name: set(values) for name, values in dependencies.items()}
    ready = sorted((name for name, values in pending.items() if not values), key=lambda name: (original_index.get(name, len(original_index)), name))

    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for follower in sorted(followers.get(current, ()), key=lambda name: (original_index.get(name, len(original_index)), name)):
            if current not in pending[follower]:
                continue
            pending[follower].remove(current)
            if not pending[follower] and follower not in ordered and follower not in ready:
                ready.append(follower)
        ready.sort(key=lambda name: (original_index.get(name, len(original_index)), name))

    for anchor in document.anchors:
        if anchor.name not in ordered:
            ordered.append(anchor.name)
    for window in document.windows:
        if window.start_anchor not in ordered:
            ordered.append(window.start_anchor)
        if window.end_anchor not in ordered:
            ordered.append(window.end_anchor)
    return tuple(ordered)


def _clocking_label(document: ScenarioDocument) -> str:
    label = f"@({document.clocking.edge} {document.clocking.signal})"
    if document.clocking.disable_iff:
        return f"{label} disable iff ({document.clocking.disable_iff})"
    return label


def _constraint_label(constraint: LaneConstraint) -> str:
    if constraint.relation == "eq":
        return constraint.value or "="
    if constraint.relation == "neq":
        return f"!={constraint.value}" if constraint.value else "!="
    relation_labels = {
        "high": "1",
        "low": "0",
        "rise": "rise",
        "fall": "fall",
        "stable": "stable",
        "change": "change",
        "unknown": "unknown",
        "dontcare": "?",
        "raw": constraint.value or "expr",
    }
    return relation_labels.get(constraint.relation, constraint.relation)


def _constraint_detail(constraint: LaneConstraint, property_names: tuple[str, ...]) -> str:
    detail = _constraint_region_label(constraint)
    if property_names:
        detail = f"{detail} | {', '.join(property_names)}"
    return detail


def _constraint_region_label(constraint: LaneConstraint) -> str:
    if constraint.region == ConstraintRegion.AT and constraint.anchor:
        return f"at {constraint.anchor}"
    if constraint.region == ConstraintRegion.BEFORE and constraint.anchor:
        return f"before {constraint.anchor}"
    if constraint.region == ConstraintRegion.AFTER and constraint.anchor:
        return f"after {constraint.anchor}"
    if constraint.region == ConstraintRegion.IN and constraint.window:
        return f"in {constraint.window}"
    if constraint.region == ConstraintRegion.FROM_UNTIL and constraint.start_anchor and constraint.end_anchor:
        return f"from {constraint.start_anchor} until {constraint.end_anchor}"
    return constraint.region.value.replace("_", " ")


def _constraint_sort_key(constraint: LaneConstraint) -> tuple[int, str, str]:
    order = {
        ConstraintRegion.AT: 0,
        ConstraintRegion.IN: 1,
        ConstraintRegion.FROM_UNTIL: 2,
        ConstraintRegion.BEFORE: 3,
        ConstraintRegion.AFTER: 4,
    }
    target = constraint.anchor or constraint.window or constraint.start_anchor or constraint.end_anchor or ""
    return (order.get(constraint.region, 99), target, constraint.name)


def _cut_label(cut: Cut) -> str:
    meaning_labels = {
        CutMeaning.OMITTED_HISTORY: "history",
        CutMeaning.OMITTED_FUTURE: "future",
        CutMeaning.SYMBOLIC_GAP: "gap",
        CutMeaning.LOOKBACK: "lookback",
    }
    return meaning_labels.get(cut.meaning, cut.meaning.value.replace("_", " "))


def _cut_detail(cut: Cut) -> str:
    if cut.placement == CutPlacement.BEFORE_ANCHOR and cut.anchor:
        return f"before {cut.anchor}"
    if cut.placement == CutPlacement.AFTER_ANCHOR and cut.anchor:
        return f"after {cut.anchor}"
    if cut.left_window and cut.right_window:
        return f"between {cut.left_window} and {cut.right_window}"
    return cut.placement.value.replace("_", " ")


def _sorted_timeline_items(items: tuple[TimelineItem, ...] | list[TimelineItem]) -> list[TimelineItem]:
    return sorted(items, key=lambda item: item.name)
