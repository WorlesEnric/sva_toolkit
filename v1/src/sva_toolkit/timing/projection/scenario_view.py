"""Render-oriented projection for symbolic timing scenarios."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Dict, List, Tuple

from sva_toolkit.timing.core.scenario import CutPlacement, LaneConstraint, ScenarioDocument, WindowBoundKind


@dataclass(frozen=True)
class TimelineItem:
    """One visible x-axis primitive in the symbolic renderer."""

    kind: str
    name: str
    label: str
    anchor: str | None = None
    window: str | None = None


@dataclass(frozen=True)
class SignalLaneView:
    """Render-oriented grouping of lane facts."""

    signal_name: str
    constraints: Tuple[LaneConstraint, ...]


@dataclass(frozen=True)
class ScenarioView:
    """Normalized view used by the symbolic SVG renderer."""

    document: ScenarioDocument
    timeline: Tuple[TimelineItem, ...]
    lanes: Tuple[SignalLaneView, ...]
    anchor_order: Tuple[str, ...]


def build_scenario_view(document: ScenarioDocument) -> ScenarioView:
    """Build a symbolic view of the scenario timeline and lane facts."""

    anchor_order = _order_anchors(document)
    before_cuts = defaultdict(list)
    after_cuts = defaultdict(list)
    between_windows = []
    for cut in document.cuts:
        item = TimelineItem(kind="cut", name=cut.name, label=cut.label or cut.meaning.value, anchor=cut.anchor)
        if cut.placement == CutPlacement.BEFORE_ANCHOR and cut.anchor:
            before_cuts[cut.anchor].append(item)
        elif cut.placement == CutPlacement.AFTER_ANCHOR and cut.anchor:
            after_cuts[cut.anchor].append(item)
        else:
            between_windows.append(TimelineItem(kind="cut", name=cut.name, label=cut.label or cut.meaning.value))

    windows_by_start = defaultdict(list)
    for window in document.windows:
        label = window.bound.label if window.bound.kind != WindowBoundKind.OMITTED else "omitted"
        windows_by_start[window.start_anchor].append(
            TimelineItem(kind="window", name=window.name, label=label, window=window.name)
        )

    timeline: List[TimelineItem] = []
    for anchor_name in anchor_order:
        timeline.extend(sorted(before_cuts.get(anchor_name, []), key=lambda item: item.name))
        timeline.append(TimelineItem(kind="anchor", name=anchor_name, label=anchor_name, anchor=anchor_name))
        timeline.extend(sorted(windows_by_start.get(anchor_name, []), key=lambda item: item.name))
        timeline.extend(sorted(after_cuts.get(anchor_name, []), key=lambda item: item.name))
    timeline.extend(sorted(between_windows, key=lambda item: item.name))

    constraints_by_signal: Dict[str, List[LaneConstraint]] = defaultdict(list)
    for constraint in document.lane_constraints:
        for signal in constraint.signals:
            constraints_by_signal[signal].append(constraint)

    lanes = tuple(
        SignalLaneView(signal_name=signal.name, constraints=tuple(constraints_by_signal.get(signal.name, ())))
        for signal in document.signals
    )
    return ScenarioView(document=document, timeline=tuple(timeline), lanes=lanes, anchor_order=anchor_order)


def _order_anchors(document: ScenarioDocument) -> Tuple[str, ...]:
    graph = {}
    for anchor in document.anchors:
        graph.setdefault(anchor.name, set())
    for window in document.windows:
        graph.setdefault(window.start_anchor, set())
        graph.setdefault(window.end_anchor, set()).add(window.start_anchor)
    sorter = TopologicalSorter(graph)
    return tuple(sorter.static_order())
