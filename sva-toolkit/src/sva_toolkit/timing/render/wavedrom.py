"""WaveDrom-backed SVG renderer for concrete timing scenarios."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Sequence
from xml.etree import ElementTree as ET

from sva_toolkit.timing.projection.wavedrom_view import (
    SummaryRule,
    WaveDromScenarioView,
    WaveLaneView,
)


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

OUTER_MARGIN_X = 20.0
OUTER_MARGIN_Y = 16.0
HEADER_HEIGHT = 54.0
CARD_PADDING = 12.0

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
ET.register_namespace("xlink", XLINK_NS)


@dataclass(frozen=True)
class WaveDromLayout:
    """Geometric layout derived from the WaveDrom renderer."""

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
    """Top-band label for a named event occurrence."""

    label: str
    x: float
    width: float
    track: int


@dataclass(frozen=True)
class RuleTrackPlacement:
    """Top-band span annotation for response rules."""

    label: str
    start_x: float
    end_x: float
    label_width: float
    loop: bool
    track: int
    color: str


@dataclass(frozen=True)
class SummaryLine:
    """Footer summary row for a classified rule."""

    color: str
    text: str
    width: float


def render_wavedrom_svg(view: WaveDromScenarioView) -> str:
    """Render a concrete timing view to SVG via WaveDrom plus semantic overlays."""

    renderer_cls = _require_wavedrom()

    wavedrom_source = _build_wavedrom_source(view)
    renderer = renderer_cls()
    drawing = renderer.render_waveform(0, wavedrom_source)
    inner_root = ET.fromstring(drawing.tostring())
    _stash_renderer_layout(inner_root, renderer)

    layout = _extract_layout(inner_root, view)
    event_labels = _plan_event_labels(view, layout)
    rule_tracks = _plan_rule_tracks(view, layout)
    summary_lines = _build_summary_lines(view)
    overlay_height = _overlay_height(event_labels, rule_tracks)
    summary_height = _summary_height(summary_lines)

    required_width = _required_inner_width(layout, event_labels, rule_tracks, summary_lines)
    if required_width > layout.width:
        layout = WaveDromLayout(
            width=required_width,
            height=layout.height,
            plot_origin_x=layout.plot_origin_x,
            plot_origin_y=layout.plot_origin_y,
            tick_width=layout.tick_width,
            lane_height=layout.lane_height,
            lane_pitch=layout.lane_pitch,
            lane_count=layout.lane_count,
        )

    _ensure_renderer_defs(inner_root)
    _ensure_renderer_style(inner_root)

    if overlay_height > 0:
        _shift_waves_group(inner_root, overlay_height)
        inner_root.append(_build_event_overlay_group(view, layout, event_labels, rule_tracks, overlay_height))
        inner_root.append(_build_rule_overlay_group(layout, rule_tracks))
    else:
        _shift_waves_group(inner_root, 0.0)

    hold_group = _build_hold_highlight_group(view, layout)
    if hold_group is not None:
        _inject_hold_group(inner_root, hold_group)

    if summary_lines:
        inner_root.append(_build_summary_group(overlay_height + layout.height + SUMMARY_TOP_GAP, summary_lines))

    total_inner_height = layout.height + overlay_height + summary_height
    inner_root.set("width", _fmt(layout.width))
    inner_root.set("height", _fmt(total_inner_height))
    inner_root.set("viewBox", f"0 0 {_fmt(layout.width)} {_fmt(total_inner_height)}")

    inner_svg = ET.tostring(inner_root, encoding="unicode")
    return _wrap_outer_svg(inner_svg, view, overlay_height, summary_height)


def _build_wavedrom_source(view: WaveDromScenarioView) -> dict:
    """Build the WaveDrom source dictionary from concrete lanes."""

    return {
        "signal": [_build_lane_signal(lane) for lane in view.lanes],
        "head": {"tick": 0},
        "config": {"hscale": _compute_hscale(view)},
    }


def _build_lane_signal(lane: WaveLaneView) -> dict:
    """Convert a lane view into a WaveDrom signal object."""

    signal = {"name": _lane_display_name(lane)}
    if lane.kind == "bit":
        signal["wave"] = _encode_bit_wave(lane.samples)
        return signal

    wave, data = _encode_bus_wave(lane.samples)
    signal["wave"] = wave
    if data:
        signal["data"] = data
    return signal


def _encode_bit_wave(samples: Sequence[str]) -> str:
    """Encode concrete bit samples into WaveDrom wave syntax."""

    wave: list[str] = []
    previous = None
    for raw_sample in samples:
        sample = raw_sample.lower()
        if sample not in {"0", "1", "x", "z"}:
            sample = "x"
        wave.append(sample if sample != previous else ".")
        previous = sample
    return "".join(wave)


def _encode_bus_wave(samples: Sequence[str]) -> tuple[str, list[str]]:
    """Encode concrete bus samples into WaveDrom wave/data pairs."""

    wave: list[str] = []
    data: list[str] = []

    run_start = 0
    while run_start < len(samples):
        run_value = samples[run_start]
        run_end = run_start + 1
        while run_end < len(samples) and samples[run_end] == run_value:
            run_end += 1

        normalized = run_value.lower()
        if normalized in {"x", "z"}:
            symbol = normalized
        else:
            symbol = "="
            data.append(run_value)

        wave.append(symbol)
        wave.extend("." for _ in range(run_end - run_start - 1))
        run_start = run_end

    return "".join(wave), data


def _compute_hscale(view: WaveDromScenarioView) -> float:
    """Choose an hscale that gives bus labels enough space without oversizing short traces."""

    widest_value = 0.0
    for lane in view.lanes:
        if lane.kind != "bus":
            continue
        for sample in lane.samples:
            if sample.lower() in {"x", "z"}:
                continue
            widest_value = max(widest_value, _text_width(sample, 11))

    target_tick_width = max(56.0, min(100.0, widest_value + 24.0))
    hscale = max(1, min(4, int(math.ceil(target_tick_width / 40.0))))

    if view.ticks >= 18 and widest_value <= 18.0:
        return 1
    return max(hscale, 2 if view.ticks <= 14 else 1)


def _extract_layout(svg_tree: ET.Element, view: WaveDromScenarioView) -> WaveDromLayout:
    """Extract renderer geometry needed for semantic overlay placement."""

    xg = float(svg_tree.attrib.pop("data-timing-xg"))
    yh0 = float(svg_tree.attrib.pop("data-timing-yh0"))
    yh1 = float(svg_tree.attrib.pop("data-timing-yh1"))
    xs = float(svg_tree.attrib.pop("data-timing-xs"))
    ys = float(svg_tree.attrib.pop("data-timing-ys"))
    yo = float(svg_tree.attrib.pop("data-timing-yo"))
    hscale = float(svg_tree.attrib.pop("data-timing-hscale"))

    return WaveDromLayout(
        width=_to_float(svg_tree.attrib["width"]),
        height=_to_float(svg_tree.attrib["height"]),
        plot_origin_x=xg + 0.5,
        plot_origin_y=yh0 + yh1 + 0.5,
        tick_width=2.0 * xs * hscale,
        lane_height=ys,
        lane_pitch=yo,
        lane_count=len(view.lanes),
    )


def _plan_event_labels(view: WaveDromScenarioView, layout: WaveDromLayout) -> list[EventLabelPlacement]:
    """Plan non-overlapping label boxes for anchor occurrences."""

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


def _plan_rule_tracks(view: WaveDromScenarioView, layout: WaveDromLayout) -> list[RuleTrackPlacement]:
    """Plan non-overlapping arrow tracks for response spans."""

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
    """Greedily assign tracks so projected spans do not overlap."""

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
    layout: WaveDromLayout,
    placements: Sequence[EventLabelPlacement],
    rule_tracks: Sequence[RuleTrackPlacement],
    overlay_height: float,
) -> ET.Element:
    """Build the event-label overlay group."""

    group = ET.Element(_svg("g"), {"id": "timing-event-overlays"})
    if not placements:
        return group

    rule_band_height = (max((track.track for track in rule_tracks), default=-1) + 1) * RULE_TRACK_PITCH
    event_band_top = OVERLAY_TOP_PADDING + rule_band_height + (OVERLAY_BAND_GAP if rule_band_height else 0.0)
    guide_y = overlay_height + layout.plot_origin_y
    plot_bottom = guide_y + max(0.0, (layout.lane_count - 1) * layout.lane_pitch + layout.lane_height)
    occurrence_xs = {
        _event_x(layout, occurrence.tick, occurrence.placement)
        for occurrence in view.anchor_occurrences
    }

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
            },
        )
        text.text = placement.label

    return group


def _build_rule_overlay_group(
    layout: WaveDromLayout,
    placements: Sequence[RuleTrackPlacement],
) -> ET.Element:
    """Build the response-arrow overlay group."""

    group = ET.Element(_svg("g"), {"id": "timing-rule-overlays"})
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
            },
        )
        text.text = placement.label

    return group


def _build_hold_highlight_group(view: WaveDromScenarioView, layout: WaveDromLayout) -> ET.Element | None:
    """Build the lane highlight group for hold spans."""

    if not view.hold_spans:
        return None

    lane_index = {lane.name: index for index, lane in enumerate(view.lanes)}
    group = ET.Element(_svg("g"), {"id": "timing-hold-highlights"})

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
                },
            )

    return group


def _build_summary_group(top_y: float, lines: Sequence[SummaryLine]) -> ET.Element:
    """Build the footer summary group for rule text."""

    group = ET.Element(_svg("g"), {"id": "timing-rule-summary", "transform": f"translate(0,{_fmt(top_y)})"})
    heading = ET.SubElement(
        group,
        _svg("text"),
        {
            "x": "0",
            "y": "0",
            "class": "timing-summary-heading",
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
            },
        )
        text = ET.SubElement(
            group,
            _svg("text"),
            {
                "x": "18",
                "y": _fmt(row_y),
                "class": "timing-summary-line",
            },
        )
        text.text = line.text

    return group


def _ensure_renderer_defs(svg_tree: ET.Element) -> None:
    """Ensure the WaveDrom SVG contains arrow-marker definitions."""

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
    ET.SubElement(
        marker,
        _svg("path"),
        {
            "d": "M 0 0 L 10 3.5 L 0 7 z",
            "fill": RESPONSE_COLOR,
        },
    )


def _ensure_renderer_style(svg_tree: ET.Element) -> None:
    """Inject the CSS used by the semantic overlays and waveform styling."""

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

/* Override WaveDrom default label color to match native renderer */
#lanes_0 text { fill: #0066CC; font-size: 13px; font-family: Helvetica, Arial, sans-serif; }
#waves_0 path { stroke: #000000; stroke-width: 1.5; }
"""


def _wrap_outer_svg(
    inner_svg: str,
    view: WaveDromScenarioView,
    overlay_height: float,
    summary_height: float,
) -> str:
    """Wrap the inner WaveDrom SVG in the timing-shell card."""

    inner_root = ET.fromstring(inner_svg)
    inner_width = _to_float(inner_root.attrib["width"])
    inner_height = _to_float(inner_root.attrib["height"])
    card_width = inner_width + CARD_PADDING * 2.0
    card_height = inner_height + CARD_PADDING * 2.0

    meta_text = f"@({view.clocking.edge} {view.clocking.signal})"
    if view.clocking.disable_iff:
        meta_text += f" disable iff ({view.clocking.disable_iff})"

    outer_width = math.ceil(
        max(
            OUTER_MARGIN_X * 2.0 + card_width,
            OUTER_MARGIN_X * 2.0 + _text_width(view.name, 20) + 12.0,
            OUTER_MARGIN_X * 2.0 + _text_width(meta_text, 12) + 12.0,
        )
    )
    outer_height = math.ceil(OUTER_MARGIN_Y * 2.0 + HEADER_HEIGHT + card_height)

    card_x = OUTER_MARGIN_X
    card_y = OUTER_MARGIN_Y + HEADER_HEIGHT
    inner_x = card_x + CARD_PADDING
    inner_y = card_y + CARD_PADDING

    del overlay_height
    del summary_height

    return "\n".join(
        [
            f'<svg xmlns="{SVG_NS}" width="{_fmt(outer_width)}" height="{_fmt(outer_height)}" '
            f'viewBox="0 0 {_fmt(outer_width)} {_fmt(outer_height)}" class="timing-shell" '
            f'font-family="Helvetica, Arial, sans-serif">',
            '<style type="text/css">',
            ".timing-bg { fill: #ffffff; }",
            ".timing-card { fill: #ffffff; stroke: #d8d8d8; stroke-width: 1; }",
            ".timing-title { fill: #111111; font-size: 16px; font-weight: 700; }",
            ".timing-meta { fill: #555555; font-size: 12px; }",
            "</style>",
            f'<rect class="timing-bg" x="0" y="0" width="{_fmt(outer_width)}" height="{_fmt(outer_height)}"/>',
            f'<text x="{_fmt(OUTER_MARGIN_X)}" y="{_fmt(OUTER_MARGIN_Y + 18.0)}" class="timing-title">{_xml(view.name)}</text>',
            f'<text x="{_fmt(OUTER_MARGIN_X)}" y="{_fmt(OUTER_MARGIN_Y + 38.0)}" class="timing-meta">{_xml(meta_text)}</text>',
            f'<rect class="timing-card" x="{_fmt(card_x)}" y="{_fmt(card_y)}" width="{_fmt(card_width)}" height="{_fmt(card_height)}" rx="12" ry="12"/>',
            f'<g transform="translate({_fmt(inner_x)},{_fmt(inner_y)})">',
            inner_svg,
            "</g>",
            "</svg>",
        ]
    )


def _build_summary_lines(view: WaveDromScenarioView) -> list[SummaryLine]:
    """Convert summary rules into rendered footer lines."""

    return [
        SummaryLine(
            color=_summary_color(rule),
            text=rule.description,
            width=_text_width(rule.description, 11),
        )
        for rule in view.summary_rules
    ]


def _summary_color(rule: SummaryRule) -> str:
    """Map a summary rule category to its display color."""

    if rule.category == "response":
        return RESPONSE_COLOR
    if rule.category == "hold":
        return HOLD_COLOR
    return NOT_BEFORE_COLOR


def _overlay_height(
    event_labels: Sequence[EventLabelPlacement],
    rule_tracks: Sequence[RuleTrackPlacement],
) -> float:
    """Compute the top-band height needed for labels and arrows."""

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
    """Compute the footer height needed for summary rows."""

    if not lines:
        return 0.0
    return SUMMARY_TOP_GAP + 24.0 + len(lines) * SUMMARY_LINE_PITCH + 8.0


def _required_inner_width(
    layout: WaveDromLayout,
    event_labels: Sequence[EventLabelPlacement],
    rule_tracks: Sequence[RuleTrackPlacement],
    summary_lines: Sequence[SummaryLine],
) -> float:
    """Return the minimum inner width that avoids clipping overlays or summary text."""

    required = layout.width
    for label in event_labels:
        required = max(required, label.x + label.width / 2.0 + 14.0)
    for track in rule_tracks:
        label_center = track.start_x + 14.0 if track.loop else (track.start_x + track.end_x) / 2.0
        required = max(required, label_center + track.label_width / 2.0 + 14.0)
    for line in summary_lines:
        required = max(required, line.width + 26.0)
    return math.ceil(required)


def _shift_waves_group(svg_tree: ET.Element, overlay_height: float) -> None:
    """Shift the base waveform group down to make room for top overlays."""

    waves_group = _find_required(svg_tree, _svg("g"), "waves_0")
    if overlay_height <= 0:
        waves_group.attrib.pop("transform", None)
        return
    waves_group.set("transform", f"translate(0,{_fmt(overlay_height)})")


def _inject_hold_group(svg_tree: ET.Element, hold_group: ET.Element) -> None:
    """Insert the hold highlight group behind the rendered lanes."""

    waves_group = _find_required(svg_tree, _svg("g"), "waves_0")
    waves_group.insert(0, hold_group)


def _lane_display_name(lane: WaveLaneView) -> str:
    """Return the lane label shown in WaveDrom."""

    if lane.width and not (lane.kind == "bit" and lane.width == "1"):
        return f"{lane.name}[{lane.width}]"
    return lane.name


def _event_x(layout: WaveDromLayout, tick: int, placement: str) -> float:
    """Convert a tick and placement hint into an x coordinate."""

    x = layout.plot_origin_x + tick * layout.tick_width
    if placement == "center":
        x += layout.tick_width / 2.0
    return x


def _stash_renderer_layout(svg_tree: ET.Element, renderer) -> None:
    """Store renderer geometry on the SVG root until layout extraction runs."""

    svg_tree.set("data-timing-xg", _fmt(float(renderer.lane.xg)))
    svg_tree.set("data-timing-yh0", _fmt(float(renderer.lane.yh0)))
    svg_tree.set("data-timing-yh1", _fmt(float(renderer.lane.yh1)))
    svg_tree.set("data-timing-xs", _fmt(float(renderer.lane.xs)))
    svg_tree.set("data-timing-ys", _fmt(float(renderer.lane.ys)))
    svg_tree.set("data-timing-yo", _fmt(float(renderer.lane.yo)))
    svg_tree.set("data-timing-hscale", _fmt(float(renderer.lane.hscale)))


def _text_width(text: str, size: int) -> float:
    """Measure text using WaveDrom's font metrics when available."""

    renderer_cls = _load_wavedrom()
    if renderer_cls is None:
        return len(text) * size * 0.62
    return float(renderer_cls().text_width(text, size))


def _load_wavedrom():
    try:
        from wavedrom.waveform import WaveDrom
    except ImportError:
        return None
    return WaveDrom


def _require_wavedrom():
    """Raise the documented runtime error when WaveDrom is unavailable."""

    renderer_cls = _load_wavedrom()
    if renderer_cls is None:
        raise RuntimeError("Install sva-toolkit[timing-render] for PNG/WaveDrom support")
    return renderer_cls


def _find_required(root: ET.Element, tag: str, element_id: str) -> ET.Element:
    """Find a required element or fail loudly if WaveDrom output drifted."""

    element = root.find(f"{tag}[@id='{element_id}']")
    if element is None:
        raise RuntimeError(f"WaveDrom output is missing expected element '{element_id}'")
    return element


def _svg(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def _to_float(value) -> float:
    return float(str(value).replace("px", "").replace(",", ""))


def _fmt(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    rounded = round(float(value), 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
