"""Debug-only legacy WaveDrom overlays.

These helpers intentionally preserve the target-leaking anchor pills, response
labels, hold bands, and RULES footer for human inspection profiles only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Sequence
from xml.etree import ElementTree as ET

from sva_toolkit.timing.projection.wavedrom_view import SummaryRule, WaveDromScenarioView


SVG_NS = "http://www.w3.org/2000/svg"

EVENT_TRACK_PITCH = 24.0
EVENT_BOX_HEIGHT = 20.0
RULE_TRACK_PITCH = 22.0
OVERLAY_TOP_PADDING = 8.0
OVERLAY_BAND_GAP = 8.0
OVERLAY_BOTTOM_PADDING = 8.0
SUMMARY_TOP_GAP = 14.0
SUMMARY_LINE_PITCH = 18.0

RESPONSE_COLOR = "#b45309"
HOLD_COLOR = "#16a34a"
NOT_BEFORE_COLOR = "#b91c1c"

ET.register_namespace("", SVG_NS)


@dataclass(frozen=True)
class LegacyOverlayLayout:
    width: float
    height: float
    plot_origin_x: float
    plot_origin_y: float
    tick_width: float
    lane_height: float
    lane_pitch: float
    lane_count: int


@dataclass(frozen=True)
class EventLabelPlacement:
    label: str
    x: float
    width: float
    track: int


@dataclass(frozen=True)
class RuleTrackPlacement:
    label: str
    start_x: float
    end_x: float
    label_width: float
    loop: bool
    track: int
    color: str


@dataclass(frozen=True)
class SummaryLine:
    color: str
    text: str
    width: float


@dataclass(frozen=True)
class LegacyOverlayResult:
    layout: LegacyOverlayLayout
    overlay_height: float
    summary_height: float


def attach_legacy_overlays(
    svg_tree: ET.Element,
    view: WaveDromScenarioView,
    layout: LegacyOverlayLayout,
) -> LegacyOverlayResult:
    """Attach the old target-leaking overlay groups to a WaveDrom SVG tree."""

    event_labels = _plan_event_labels(view, layout)
    rule_tracks = _plan_rule_tracks(view, layout)
    summary_lines = _build_summary_lines(view)
    overlay_height = _overlay_height(event_labels, rule_tracks)
    summary_height = _summary_height(summary_lines)

    required_width = _required_inner_width(layout, event_labels, rule_tracks, summary_lines)
    if required_width > layout.width:
        layout = replace(layout, width=required_width)

    _ensure_renderer_defs(svg_tree)
    _ensure_renderer_style(svg_tree)

    if overlay_height > 0:
        _shift_waves_group(svg_tree, overlay_height)
        svg_tree.append(_build_event_overlay_group(view, layout, event_labels, rule_tracks, overlay_height))
        svg_tree.append(_build_rule_overlay_group(rule_tracks))
    else:
        _shift_waves_group(svg_tree, 0.0)

    hold_group = _build_hold_highlight_group(view, layout, overlay_height)
    if hold_group is not None:
        _inject_hold_group(svg_tree, hold_group)

    if summary_lines:
        svg_tree.append(_build_summary_group(overlay_height + layout.height + SUMMARY_TOP_GAP, summary_lines))

    total_height = layout.height + overlay_height + summary_height
    svg_tree.set("width", _fmt(layout.width))
    svg_tree.set("height", _fmt(total_height))
    svg_tree.set("viewBox", f"0 0 {_fmt(layout.width)} {_fmt(total_height)}")
    return LegacyOverlayResult(layout=replace(layout, height=total_height), overlay_height=overlay_height, summary_height=summary_height)


def _plan_event_labels(view: WaveDromScenarioView, layout: LegacyOverlayLayout) -> list[EventLabelPlacement]:
    counts = Counter(occurrence.anchor_name for occurrence in view.anchor_occurrences)
    planned = []
    for occurrence in sorted(view.anchor_occurrences, key=lambda item: (item.tick, item.anchor_name, item.placement)):
        if not occurrence.label:
            continue
        label = occurrence.label if counts[occurrence.anchor_name] == 1 else f"{occurrence.label} @ t{occurrence.tick}"
        width = max(64.0, _text_width(label, 11) + 18.0)
        x = _event_x(layout, occurrence.tick, occurrence.placement)
        planned.append((label, x, width))

    tracks = _assign_tracks(planned, lambda item: (item[1] - item[2] / 2.0, item[1] + item[2] / 2.0))
    return [
        EventLabelPlacement(label=label, x=x, width=width, track=track)
        for (label, x, width), track in zip(planned, tracks)
    ]


def _plan_rule_tracks(view: WaveDromScenarioView, layout: LegacyOverlayLayout) -> list[RuleTrackPlacement]:
    preliminary = []
    for span in view.response_spans:
        start_x = _event_x(layout, span.trigger_tick, "boundary")
        end_x = _event_x(layout, span.response_tick, "center")
        loop = abs(end_x - start_x) < layout.tick_width * 0.35
        label_width = max(56.0, _text_width(span.delay_text, 11) + 16.0)
        label_center = start_x + 14.0 if loop else (start_x + end_x) / 2.0
        preliminary.append(
            (
                span.delay_text,
                start_x,
                end_x,
                loop,
                label_width,
                (
                    min(start_x, end_x, label_center - label_width / 2.0) - 10.0,
                    max(start_x, end_x, label_center + label_width / 2.0) + 10.0,
                ),
            )
        )

    tracks = _assign_tracks(preliminary, lambda item: item[5])
    return [
        RuleTrackPlacement(
            label=label,
            start_x=start_x,
            end_x=end_x,
            label_width=label_width,
            loop=loop,
            track=track,
            color=RESPONSE_COLOR,
        )
        for (label, start_x, end_x, loop, label_width, _), track in zip(preliminary, tracks)
    ]


def _assign_tracks(items: Sequence[object], width_fn) -> list[int]:
    track_ends: list[float] = []
    assignments = [0] * len(items)
    ordered = sorted(enumerate(items), key=lambda item: width_fn(item[1]))

    for original_index, item in ordered:
        start_x, end_x = width_fn(item)
        for track_index, previous_end in enumerate(track_ends):
            if start_x >= previous_end:
                track_ends[track_index] = end_x + 10.0
                assignments[original_index] = track_index
                break
        else:
            assignments[original_index] = len(track_ends)
            track_ends.append(end_x + 10.0)

    return assignments


def _build_event_overlay_group(
    view: WaveDromScenarioView,
    layout: LegacyOverlayLayout,
    placements: Sequence[EventLabelPlacement],
    rule_tracks: Sequence[RuleTrackPlacement],
    overlay_height: float,
) -> ET.Element:
    group = ET.Element(_svg("g"), {"id": "timing-event-overlays", "class": "timing-event-overlays"})
    if not placements:
        return group

    rule_band_height = (max((track.track for track in rule_tracks), default=-1) + 1) * RULE_TRACK_PITCH
    event_band_top = OVERLAY_TOP_PADDING + rule_band_height + (OVERLAY_BAND_GAP if rule_band_height else 0.0)
    guide_y = overlay_height + layout.plot_origin_y
    plot_bottom = guide_y + max(0.0, (layout.lane_count - 1) * layout.lane_pitch + layout.lane_height)
    occurrence_xs = {_event_x(layout, occurrence.tick, occurrence.placement) for occurrence in view.anchor_occurrences}

    for x in sorted(occurrence_xs):
        ET.SubElement(
            group,
            _svg("line"),
            {
                "x1": _fmt(x),
                "y1": _fmt(guide_y),
                "x2": _fmt(x),
                "y2": _fmt(plot_bottom),
                "class": "timing-event-guide",
                "data-role": "debug_overlay",
            },
        )

    for placement in placements:
        box_y = event_band_top + placement.track * EVENT_TRACK_PITCH
        box_x = placement.x - placement.width / 2.0
        connector_y = box_y + EVENT_BOX_HEIGHT

        ET.SubElement(
            group,
            _svg("line"),
            {
                "x1": _fmt(placement.x),
                "y1": _fmt(connector_y),
                "x2": _fmt(placement.x),
                "y2": _fmt(guide_y - 4.0),
                "class": "timing-event-connector",
                "data-role": "debug_overlay",
            },
        )
        ET.SubElement(
            group,
            _svg("rect"),
            {
                "x": _fmt(box_x),
                "y": _fmt(box_y),
                "width": _fmt(placement.width),
                "height": _fmt(EVENT_BOX_HEIGHT),
                "rx": "10",
                "ry": "10",
                "class": "timing-event-box",
                "data-role": "debug_overlay",
            },
        )
        text = ET.SubElement(
            group,
            _svg("text"),
            {
                "x": _fmt(placement.x),
                "y": _fmt(box_y + EVENT_BOX_HEIGHT / 2.0),
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "class": "timing-event-label",
                "data-role": "debug_overlay",
            },
        )
        text.text = placement.label

    return group


def _build_rule_overlay_group(placements: Sequence[RuleTrackPlacement]) -> ET.Element:
    group = ET.Element(_svg("g"), {"id": "timing-rule-overlays", "class": "timing-rule-overlays"})
    for placement in placements:
        y = OVERLAY_TOP_PADDING + placement.track * RULE_TRACK_PITCH + 12.0
        label_center = placement.start_x + 14.0 if placement.loop else (placement.start_x + placement.end_x) / 2.0

        if placement.loop:
            path_d = (
                f"M {_fmt(placement.start_x)} {_fmt(y)} "
                f"C {_fmt(placement.start_x)} {_fmt(y - 14.0)}, "
                f"{_fmt(placement.start_x + 20.0)} {_fmt(y - 14.0)}, "
                f"{_fmt(placement.start_x + 20.0)} {_fmt(y)}"
            )
        else:
            path_d = f"M {_fmt(placement.start_x)} {_fmt(y)} L {_fmt(placement.end_x)} {_fmt(y)}"

        ET.SubElement(
            group,
            _svg("path"),
            {
                "d": path_d,
                "stroke": placement.color,
                "fill": "none",
                "stroke-width": "2",
                "marker-end": "url(#timing-arrow-head)",
                "data-role": "debug_overlay",
            },
        )
        text = ET.SubElement(
            group,
            _svg("text"),
            {
                "x": _fmt(label_center),
                "y": _fmt(y - 7.0),
                "text-anchor": "middle",
                "class": "timing-rule-label",
                "data-role": "debug_overlay",
            },
        )
        text.text = placement.label

    return group


def _build_hold_highlight_group(
    view: WaveDromScenarioView,
    layout: LegacyOverlayLayout,
    overlay_height: float,
) -> ET.Element | None:
    del overlay_height
    if not view.hold_spans:
        return None

    lane_index = {lane.name: index for index, lane in enumerate(view.lanes)}
    group = ET.Element(_svg("g"), {"id": "timing-hold-highlights", "class": "timing-hold-highlights"})

    for span in view.hold_spans:
        x = layout.plot_origin_x + span.start_tick * layout.tick_width + 1.0
        width = max((span.end_tick - span.start_tick + 1) * layout.tick_width - 2.0, layout.tick_width / 2.0)
        for lane_name in span.lanes:
            if lane_name not in lane_index:
                continue
            top = layout.plot_origin_y + lane_index[lane_name] * layout.lane_pitch + 2.0
            ET.SubElement(
                group,
                _svg("rect"),
                {
                    "x": _fmt(x),
                    "y": _fmt(top),
                    "width": _fmt(width),
                    "height": _fmt(layout.lane_height - 4.0),
                    "rx": "6",
                    "ry": "6",
                    "class": "timing-hold-fill",
                    "data-role": "debug_overlay",
                },
            )

    return group


def _build_summary_group(top_y: float, lines: Sequence[SummaryLine]) -> ET.Element:
    group = ET.Element(
        _svg("g"),
        {
            "id": "timing-rule-summary",
            "class": "timing-rule-summary",
            "transform": f"translate(0,{_fmt(top_y)})",
        },
    )
    heading = ET.SubElement(
        group,
        _svg("text"),
        {
            "x": "0",
            "y": "0",
            "class": "timing-summary-heading",
            "data-role": "debug_overlay",
        },
    )
    heading.text = "RULES"

    for index, line in enumerate(lines):
        row_y = 18.0 + index * SUMMARY_LINE_PITCH
        ET.SubElement(
            group,
            _svg("rect"),
            {
                "x": "0",
                "y": _fmt(row_y - 8.0),
                "width": "10",
                "height": "10",
                "rx": "2",
                "ry": "2",
                "fill": line.color,
                "data-role": "debug_overlay",
            },
        )
        text = ET.SubElement(
            group,
            _svg("text"),
            {
                "x": "18",
                "y": _fmt(row_y),
                "class": "timing-summary-line",
                "data-role": "debug_overlay",
            },
        )
        text.text = line.text

    return group


def _build_summary_lines(view: WaveDromScenarioView) -> list[SummaryLine]:
    return [
        SummaryLine(
            color=_summary_color(rule),
            text=rule.description,
            width=_text_width(rule.description, 11),
        )
        for rule in view.summary_rules
    ]


def _summary_color(rule: SummaryRule) -> str:
    if rule.category == "response":
        return RESPONSE_COLOR
    if rule.category == "hold":
        return HOLD_COLOR
    return NOT_BEFORE_COLOR


def _overlay_height(
    event_labels: Sequence[EventLabelPlacement],
    rule_tracks: Sequence[RuleTrackPlacement],
) -> float:
    if not event_labels and not rule_tracks:
        return 0.0

    rule_height = (max((track.track for track in rule_tracks), default=-1) + 1) * RULE_TRACK_PITCH
    event_height = (max((label.track for label in event_labels), default=-1) + 1) * EVENT_TRACK_PITCH
    overlay_height = OVERLAY_TOP_PADDING + rule_height
    if rule_height and event_height:
        overlay_height += OVERLAY_BAND_GAP
    if event_height:
        overlay_height += event_height + OVERLAY_BOTTOM_PADDING + EVENT_BOX_HEIGHT
    else:
        overlay_height += OVERLAY_BOTTOM_PADDING
    return overlay_height


def _summary_height(lines: Sequence[SummaryLine]) -> float:
    if not lines:
        return 0.0
    return SUMMARY_TOP_GAP + 24.0 + len(lines) * SUMMARY_LINE_PITCH + 8.0


def _required_inner_width(
    layout: LegacyOverlayLayout,
    event_labels: Sequence[EventLabelPlacement],
    rule_tracks: Sequence[RuleTrackPlacement],
    summary_lines: Sequence[SummaryLine],
) -> float:
    required = layout.width
    for label in event_labels:
        required = max(required, label.x + label.width / 2.0 + 14.0)
    for track in rule_tracks:
        label_center = track.start_x + 14.0 if track.loop else (track.start_x + track.end_x) / 2.0
        required = max(required, label_center + track.label_width / 2.0 + 14.0)
    for line in summary_lines:
        required = max(required, line.width + 26.0)
    return math.ceil(required)


def _ensure_renderer_defs(svg_tree: ET.Element) -> None:
    defs = svg_tree.find(_svg("defs"))
    if defs is None:
        defs = ET.Element(_svg("defs"))
        svg_tree.insert(0, defs)

    if defs.find(f"{_svg('marker')}[@id='timing-arrow-head']") is not None:
        return

    marker = ET.SubElement(
        defs,
        _svg("marker"),
        {
            "id": "timing-arrow-head",
            "markerWidth": "10",
            "markerHeight": "7",
            "refX": "8",
            "refY": "3.5",
            "orient": "auto",
            "markerUnits": "strokeWidth",
            "viewBox": "0 0 10 7",
        },
    )
    ET.SubElement(marker, _svg("path"), {"d": "M 0 0 L 10 3.5 L 0 7 z", "fill": RESPONSE_COLOR})


def _ensure_renderer_style(svg_tree: ET.Element) -> None:
    style = ET.SubElement(svg_tree, _svg("style"), {"type": "text/css"})
    style.text = """
.timing-event-guide { stroke: #93c5fd; stroke-width: 1; stroke-dasharray: 4 4; }
.timing-event-connector { stroke: #93c5fd; stroke-width: 1; }
.timing-event-box { fill: #eff6ff; stroke: #2563eb; stroke-width: 1; }
.timing-event-label { fill: #1d4ed8; font-size: 11px; font-weight: 600; }
.timing-rule-label { fill: #92400e; font-size: 11px; font-weight: 600; }
.timing-hold-fill { fill: #bbf7d0; fill-opacity: 0.42; stroke: #16a34a; stroke-opacity: 0.14; }
.timing-summary-heading { fill: #334155; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; }
.timing-summary-line { fill: #334155; font-size: 11px; }
"""


def _shift_waves_group(svg_tree: ET.Element, overlay_height: float) -> None:
    waves_group = _find_required(svg_tree, _svg("g"), "waves_0")
    if overlay_height <= 0:
        waves_group.attrib.pop("transform", None)
        return
    waves_group.set("transform", f"translate(0,{_fmt(overlay_height)})")


def _inject_hold_group(svg_tree: ET.Element, hold_group: ET.Element) -> None:
    waves_group = _find_required(svg_tree, _svg("g"), "waves_0")
    waves_group.insert(0, hold_group)


def _event_x(layout: LegacyOverlayLayout, tick: int, placement: str) -> float:
    x = layout.plot_origin_x + tick * layout.tick_width
    if placement == "center":
        x += layout.tick_width / 2.0
    return x


def _text_width(text: str, size: int) -> float:
    try:
        from wavedrom.waveform import WaveDrom
    except ImportError:
        return len(text) * size * 0.62
    return float(WaveDrom().text_width(text, size))


def _find_required(root: ET.Element, tag: str, element_id: str) -> ET.Element:
    element = root.find(f"{tag}[@id='{element_id}']")
    if element is None:
        raise RuntimeError(f"WaveDrom output is missing expected element '{element_id}'")
    return element


def _svg(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def _fmt(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    rounded = round(float(value), 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")
