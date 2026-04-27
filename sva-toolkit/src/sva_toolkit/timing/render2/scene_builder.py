"""Project visual timing documents into renderer-independent timing scenes."""

from __future__ import annotations

from typing import Mapping

from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    Cut,
    CutPlacement,
    LaneConstraint,
    ScenarioDocument,
    SignalDecl,
    SignalKind,
    TimeWindow,
)
from sva_toolkit.timing.projection.tick_solver import (
    anchor_placement,
    resolve_anchor_ticks,
    resolve_total_ticks,
    synthesize_signal_samples,
)
from sva_toolkit.timing.render2.decorations import Decoration, DecorationKind
from sva_toolkit.timing.render2.scene import (
    CutRegion,
    LaneScene,
    LaneType,
    SampleRun,
    TickModel,
    TimingScene,
    VisualConstraint,
    VisualEvent,
)
from sva_toolkit.timing.visual import VisibilityClass


def build_timing_scene(
    visual_document: ScenarioDocument,
    *,
    semantic_document: ScenarioDocument | None = None,
    layout: object | None = None,
) -> TimingScene:
    """Build a renderer-independent scene from an already-lowered visual document.

    The input contract is the Phase 1 visual target contract: anchor, window,
    and constraint names are already canonicalized by ``lower_to_visual_document``.
    This function intentionally does not call lowering because renderers must
    not silently change target semantics.
    """

    del layout
    anchor_ticks = resolve_anchor_ticks(visual_document)
    total_ticks = resolve_total_ticks(visual_document, anchor_ticks)
    signal_samples = synthesize_signal_samples(
        visual_document,
        anchor_ticks=anchor_ticks,
        ticks=total_ticks,
    )

    lanes = _build_lanes(visual_document, signal_samples, total_ticks)
    cuts = tuple(_build_cut_region(cut, visual_document.windows, anchor_ticks, total_ticks) for cut in visual_document.cuts)
    events = tuple(
        VisualEvent(
            name=anchor.name,
            tick=anchor_ticks.get(anchor.name, 0),
            placement=anchor_placement(anchor.condition),
            target_visibility=VisibilityClass.VISIBLE_CONVENTION,
        )
        for anchor in visual_document.anchors
    )
    constraints = tuple(
        _build_visual_constraint(constraint, visual_document.windows, anchor_ticks, total_ticks)
        for constraint in visual_document.lane_constraints
    )
    decorations = _build_decorations(visual_document, events, constraints, anchor_ticks)

    return TimingScene(
        name=visual_document.name,
        clocking_edge=visual_document.clocking.edge,
        clocking_signal=visual_document.clocking.signal,
        lanes=lanes,
        ticks=TickModel(total_ticks=total_ticks, tick_origin=_tick_origin(visual_document)),
        cuts=cuts,
        events=events,
        constraints=constraints,
        decorations=decorations,
        visible_target=visual_document,
        semantic_document=semantic_document,
    )


def _build_lanes(
    document: ScenarioDocument,
    signal_samples: Mapping[str, tuple[str, ...]],
    total_ticks: int,
) -> tuple[LaneScene, ...]:
    lanes: list[LaneScene] = [
        LaneScene(
            name=document.clocking.signal,
            lane_type=LaneType.CLOCK,
            runs=_runs((("1", "0") * (total_ticks // 2 + 1))[:total_ticks]),
            visibility=VisibilityClass.VISIBLE_GEOMETRY,
        )
    ]
    for signal in document.signals:
        if signal.name == document.clocking.signal:
            continue
        samples = signal_samples.get(signal.name, (_default_value(signal.kind),) * total_ticks)
        lanes.append(
            LaneScene(
                name=signal.name,
                lane_type=_lane_type(signal),
                runs=_runs(samples),
                width_bits=_width_bits(signal),
                visibility=VisibilityClass.VISIBLE_TEXT,
            )
        )
    return tuple(lanes)


def _runs(samples: tuple[str, ...]) -> tuple[SampleRun, ...]:
    if not samples:
        return ()

    runs: list[SampleRun] = []
    start = 0
    value = samples[0]
    for tick, sample in enumerate(samples[1:], start=1):
        if sample != value:
            runs.append(_sample_run(start, tick - 1, value))
            start = tick
            value = sample
    runs.append(_sample_run(start, len(samples) - 1, value))
    return tuple(runs)


def _sample_run(start_tick: int, end_tick: int, value: str) -> SampleRun:
    normalized = value.lower()
    return SampleRun(
        start_tick=start_tick,
        end_tick=end_tick,
        value=value,
        is_unknown=normalized == "x",
        is_high_z=normalized == "z",
    )


def _lane_type(signal: SignalDecl) -> LaneType:
    if signal.kind == SignalKind.BIT:
        return LaneType.BIT
    if signal.kind == SignalKind.BUS:
        return LaneType.BUS
    return LaneType.UNKNOWN


def _width_bits(signal: SignalDecl) -> str | None:
    if signal.kind == SignalKind.BIT and (signal.width is None or signal.width == "1"):
        return None
    return signal.width


def _default_value(kind: SignalKind) -> str:
    return "0" if kind == SignalKind.BIT else "x"


def _build_visual_constraint(
    constraint: LaneConstraint,
    windows: tuple[TimeWindow, ...],
    anchor_ticks: Mapping[str, int],
    total_ticks: int,
) -> VisualConstraint:
    start_tick, end_tick = _constraint_span(constraint, windows, anchor_ticks, total_ticks)
    return VisualConstraint(
        name=constraint.name,
        kind=constraint.relation,
        region=constraint.region.value,
        lane_names=constraint.signals,
        start_tick=start_tick,
        end_tick=end_tick,
        anchor_ref=constraint.anchor,
        window_ref=constraint.window,
        visibility=VisibilityClass.VISIBLE_CONVENTION,
    )


def _constraint_span(
    constraint: LaneConstraint,
    windows: tuple[TimeWindow, ...],
    anchor_ticks: Mapping[str, int],
    total_ticks: int,
) -> tuple[int | None, int | None]:
    if constraint.region == ConstraintRegion.AT and constraint.anchor:
        tick = anchor_ticks.get(constraint.anchor)
        return tick, tick
    if constraint.region == ConstraintRegion.IN and constraint.window:
        return _window_span(_window_map(windows).get(constraint.window), anchor_ticks)
    if constraint.region == ConstraintRegion.BEFORE and constraint.anchor:
        tick = anchor_ticks.get(constraint.anchor)
        return (0, max(0, tick - 1)) if tick is not None else (None, None)
    if constraint.region == ConstraintRegion.AFTER and constraint.anchor:
        tick = anchor_ticks.get(constraint.anchor)
        return (min(total_ticks - 1, tick + 1), total_ticks - 1) if tick is not None else (None, None)
    if constraint.region == ConstraintRegion.FROM_UNTIL:
        start = anchor_ticks.get(constraint.start_anchor) if constraint.start_anchor else None
        end = anchor_ticks.get(constraint.end_anchor) if constraint.end_anchor else None
        return start, end
    return None, None


def _window_span(window: TimeWindow | None, anchor_ticks: Mapping[str, int]) -> tuple[int | None, int | None]:
    if window is None:
        return None, None
    return anchor_ticks.get(window.start_anchor), anchor_ticks.get(window.end_anchor)


def _window_map(windows: tuple[TimeWindow, ...]) -> dict[str, TimeWindow]:
    return {window.name: window for window in windows}


def _build_cut_region(
    cut: Cut,
    windows: tuple[TimeWindow, ...],
    anchor_ticks: Mapping[str, int],
    total_ticks: int,
) -> CutRegion:
    start_tick, end_tick = _cut_span(cut, windows, anchor_ticks, total_ticks)
    return CutRegion(
        name=cut.name,
        start_tick=start_tick,
        end_tick=end_tick,
        meaning=cut.meaning.value,
        label=cut.label,
    )


def _cut_span(
    cut: Cut,
    windows: tuple[TimeWindow, ...],
    anchor_ticks: Mapping[str, int],
    total_ticks: int,
) -> tuple[int, int]:
    if total_ticks <= 0:
        return 0, 0
    if cut.placement == CutPlacement.BEFORE_ANCHOR and cut.anchor:
        tick = anchor_ticks.get(cut.anchor, 0)
        return 0, max(0, min(total_ticks - 1, tick - 1))
    if cut.placement == CutPlacement.AFTER_ANCHOR and cut.anchor:
        tick = anchor_ticks.get(cut.anchor, total_ticks - 1)
        start = min(total_ticks - 1, max(0, tick + 1))
        return start, total_ticks - 1
    if cut.placement == CutPlacement.BETWEEN_WINDOWS:
        windows_by_name = _window_map(windows)
        left_start, left_end = _window_span(windows_by_name.get(cut.left_window or ""), anchor_ticks)
        right_start, _right_end = _window_span(windows_by_name.get(cut.right_window or ""), anchor_ticks)
        if left_end is not None and right_start is not None:
            start = min(total_ticks - 1, max(0, left_end + 1))
            end = max(start, min(total_ticks - 1, right_start - 1))
            return start, end
        if left_start is not None:
            return left_start, left_start
    return 0, 0


def _tick_origin(document: ScenarioDocument) -> int:
    return int(
        any(
            cut.placement == CutPlacement.BEFORE_ANCHOR and cut.meaning.value == "omitted_history"
            for cut in document.cuts
        )
    )


def _build_decorations(
    document: ScenarioDocument,
    events: tuple[VisualEvent, ...],
    constraints: tuple[VisualConstraint, ...],
    anchor_ticks: Mapping[str, int],
) -> tuple[Decoration, ...]:
    decorations: list[Decoration] = [
        Decoration(
            kind=DecorationKind.VERTICAL_GUIDE,
            semantic=True,
            target_ref=f"anchor:{event.name}",
            anchor_tick=event.tick,
            visibility_class=VisibilityClass.VISIBLE_GEOMETRY,
        )
        for event in events
    ]

    for window in document.windows:
        start_tick, end_tick = _window_span(window, anchor_ticks)
        if start_tick is None or end_tick is None:
            continue
        decorations.append(
            Decoration(
                kind=DecorationKind.MEASUREMENT_BRACKET,
                semantic=True,
                target_ref=f"window:{window.name}",
                text=window.bound.label,
                visibility_class=VisibilityClass.VISIBLE_TEXT,
                span=(start_tick, end_tick),
            )
        )

    for constraint in constraints:
        if constraint.start_tick is None or constraint.end_tick is None:
            continue
        decorations.append(
            Decoration(
                kind=DecorationKind.HIGHLIGHT_REGION,
                semantic=True,
                target_ref=f"constraint:{constraint.name}",
                span=(constraint.start_tick, constraint.end_tick),
                lane_names=constraint.lane_names,
                visibility_class=VisibilityClass.VISIBLE_GEOMETRY,
            )
        )

    return tuple(decorations)
