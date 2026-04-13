"""WaveDrom-backed SVG renderer for timing diagrams."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import List, Sequence
from xml.etree import ElementTree as ET

from sva_toolkit.timing.core.model import DiagramSpec, EventExpr, EventPredicate, HoldUntilRule, LaneKind, NotBeforeRule, ResponseRule
from sva_toolkit.timing.projection.diagram_view import DiagramView, build_diagram_view

try:
    from wavedrom.waveform import WaveDrom
except ImportError as exc:  # pragma: no cover - exercised via runtime error path
    WaveDrom = None
    _WAVEDROM_IMPORT_ERROR = exc
else:  # pragma: no cover - import branch selection is environment dependent
    _WAVEDROM_IMPORT_ERROR = None


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
HOLD_STROKE = "#16a34a"
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
    """Footer summary row for a rule."""

    color: str
    text: str
    width: float


def render_diagram_wavedrom_svg(diagram: DiagramSpec) -> str:
    """Render a timing diagram to production SVG via WaveDrom plus semantic overlays."""

    if WaveDrom is None:  # pragma: no cover - depends on environment packaging
        raise RuntimeError(
            "SVG rendering requires the 'wavedrom' package. Install the timing renderer dependencies first."
        ) from _WAVEDROM_IMPORT_ERROR

    view = build_diagram_view(diagram)
    metrics = WaveDrom()
    wavedrom_source = _build_wavedrom_source(diagram, metrics)

    renderer = WaveDrom()
    drawing = renderer.render_waveform(0, wavedrom_source)
    inner_root = ET.fromstring(drawing.tostring())
    layout = _extract_layout(renderer, drawing)

    event_labels = _plan_event_labels(view, layout, metrics)
    rule_tracks = _plan_rule_tracks(view, layout, metrics)
    summary_lines = _build_summary_lines(diagram, metrics)
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
    return _wrap_outer_svg(diagram, inner_svg, layout.width, total_inner_height, metrics)


def _build_wavedrom_source(diagram: DiagramSpec, metrics: WaveDrom) -> dict:
    signals = [_build_lane_signal(lane) for lane in diagram.lanes]
    return {
        "signal": signals,
        "head": {"tick": 0},
        "config": {"hscale": _compute_hscale(diagram, metrics)},
    }


def _build_lane_signal(lane) -> dict:
    signal = {"name": lane.display_name}
    if lane.kind == LaneKind.BIT:
        signal["wave"] = _encode_bit_wave(lane.samples)
        return signal

    wave, data = _encode_bus_wave(lane.samples)
    signal["wave"] = wave
    if data:
        signal["data"] = data
    return signal


def _encode_bit_wave(samples: Sequence[str]) -> str:
    wave: List[str] = []
    previous = None
    for raw_sample in samples:
        sample = raw_sample.lower()
        if sample not in {"0", "1", "x", "z"}:
            sample = "x"
        wave.append(sample if sample != previous else ".")
        previous = sample
    return "".join(wave)


def _encode_bus_wave(samples: Sequence[str]) -> tuple[str, List[str]]:
    wave: List[str] = []
    data: List[str] = []

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


def _compute_hscale(diagram: DiagramSpec, metrics: WaveDrom) -> int:
    widest_value = 0.0
    for lane in diagram.lanes:
        if lane.kind == LaneKind.BUS:
            for sample in lane.samples:
                if sample.lower() not in {"x", "z"}:
                    widest_value = max(widest_value, metrics.text_width(sample, 11))

    target_tick_width = max(56.0, min(100.0, widest_value + 24.0))
    hscale = max(1, min(4, int(math.ceil(target_tick_width / 40.0))))

    if diagram.ticks >= 18 and widest_value <= 18.0:
        return 1
    return max(hscale, 2 if diagram.ticks <= 14 else 1)


def _extract_layout(renderer: WaveDrom, drawing) -> WaveDromLayout:
    return WaveDromLayout(
        width=_to_float(drawing["width"]),
        height=_to_float(drawing["height"]),
        plot_origin_x=float(renderer.lane.xg) + 0.5,
        plot_origin_y=float(renderer.lane.yh0 + renderer.lane.yh1) + 0.5,
        tick_width=2.0 * float(renderer.lane.xs) * float(renderer.lane.hscale),
        lane_height=float(renderer.lane.ys),
        lane_pitch=float(renderer.lane.yo),
    )


def _plan_event_labels(view: DiagramView, layout: WaveDromLayout, metrics: WaveDrom) -> List[EventLabelPlacement]:
    counts = Counter(occurrence.name for occurrence in view.occurrences)
    labels = []
    spans = []
    for occurrence in sorted(view.occurrences, key=lambda item: (item.tick, item.name, item.anchor)):
        label = occurrence.name if counts[occurrence.name] == 1 else f"{occurrence.name} @ t{occurrence.tick}"
        width = max(64.0, metrics.text_width(label, 11) + 18.0)
        x = _event_x(layout, occurrence.tick, occurrence.anchor)
        labels.append((label, x, width))
        spans.append((x - width / 2.0, x + width / 2.0))

    tracks = _assign_tracks(spans)
    return [
        EventLabelPlacement(label=label, x=x, width=width, track=track)
        for (label, x, width), track in zip(labels, tracks)
    ]


def _plan_rule_tracks(view: DiagramView, layout: WaveDromLayout, metrics: WaveDrom) -> List[RuleTrackPlacement]:
    preliminary = []
    spans = []
    for overlay in view.response_overlays:
        start_x = _event_x(layout, overlay.start_tick, overlay.start_anchor)
        end_x = _event_x(layout, overlay.end_tick, overlay.end_anchor)
        loop = abs(end_x - start_x) < layout.tick_width * 0.35
        label_width = max(56.0, metrics.text_width(overlay.label, 11) + 16.0)
        label_center = start_x + 14.0 if loop else (start_x + end_x) / 2.0
        spans.append(
            (
                min(start_x, end_x, label_center - label_width / 2.0) - 10.0,
                max(start_x, end_x, label_center + label_width / 2.0) + 10.0,
            )
        )
        preliminary.append((overlay, start_x, end_x, loop, label_width))

    tracks = _assign_tracks(spans)
    return [
        RuleTrackPlacement(
            label=overlay.label,
            start_x=start_x,
            end_x=end_x,
            label_width=label_width,
            loop=loop,
            track=track,
            color=RESPONSE_COLOR,
        )
        for (overlay, start_x, end_x, loop, label_width), track in zip(preliminary, tracks)
    ]


def _assign_tracks(spans: Sequence[tuple[float, float]]) -> List[int]:
    track_ends: List[float] = []
    assignments = [0] * len(spans)
    ordered = sorted(enumerate(spans), key=lambda item: (item[1][0], item[1][1]))

    for original_index, (start_x, end_x) in ordered:
        for track_index, previous_end in enumerate(track_ends):
            if start_x >= previous_end:
                track_ends[track_index] = end_x + 10.0
                assignments[original_index] = track_index
                break
        else:
            assignments[original_index] = len(track_ends)
            track_ends.append(end_x + 10.0)

    return assignments


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


def _summary_height(summary_lines: Sequence[SummaryLine]) -> float:
    if not summary_lines:
        return 0.0
    return SUMMARY_TOP_GAP + 24.0 + len(summary_lines) * SUMMARY_LINE_PITCH + 8.0


def _required_inner_width(
    layout: WaveDromLayout,
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


def _build_event_overlay_group(
    view: DiagramView,
    layout: WaveDromLayout,
    event_labels: Sequence[EventLabelPlacement],
    rule_tracks: Sequence[RuleTrackPlacement],
    overlay_height: float,
) -> ET.Element:
    group = ET.Element(_svg("g"), {"id": "timing-event-overlays"})
    if not event_labels:
        return group

    rule_band_height = (max((track.track for track in rule_tracks), default=-1) + 1) * RULE_TRACK_PITCH
    event_band_top = OVERLAY_TOP_PADDING + rule_band_height + (OVERLAY_BAND_GAP if rule_band_height else 0.0)
    plot_top = overlay_height
    plot_bottom = plot_top + layout.plot_origin_y + (len(view.spec.lanes) - 1) * layout.lane_pitch + layout.lane_height

    guide_y = plot_top + layout.plot_origin_y
    for x in sorted({_event_x(layout, occurrence.tick, occurrence.anchor) for occurrence in view.occurrences}):
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

    for placement in event_labels:
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
    rule_tracks: Sequence[RuleTrackPlacement],
) -> ET.Element:
    group = ET.Element(_svg("g"), {"id": "timing-rule-overlays"})
    for track in rule_tracks:
        y = OVERLAY_TOP_PADDING + track.track * RULE_TRACK_PITCH + 12.0
        label_center = track.start_x + 14.0 if track.loop else (track.start_x + track.end_x) / 2.0

        if track.loop:
            path_d = (
                f"M {_fmt(track.start_x)} {_fmt(y)} "
                f"C {_fmt(track.start_x)} {_fmt(y - 14.0)}, "
                f"{_fmt(track.start_x + 20.0)} {_fmt(y - 14.0)}, "
                f"{_fmt(track.start_x + 20.0)} {_fmt(y)}"
            )
        else:
            path_d = f"M {_fmt(track.start_x)} {_fmt(y)} L {_fmt(track.end_x)} {_fmt(y)}"

        ET.SubElement(
            group,
            _svg("path"),
            {
                "d": path_d,
                "stroke": track.color,
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
        text.text = track.label

    return group


def _build_hold_highlight_group(view: DiagramView, layout: WaveDromLayout) -> ET.Element | None:
    if not view.hold_overlays:
        return None

    lane_index = {lane.name: index for index, lane in enumerate(view.spec.lanes)}
    group = ET.Element(_svg("g"), {"id": "timing-hold-highlights"})

    for overlay in view.hold_overlays:
        x = layout.plot_origin_x + overlay.start_tick * layout.tick_width + 1.0
        width = max((overlay.end_tick - overlay.start_tick + 1) * layout.tick_width - 2.0, layout.tick_width / 2.0)
        for lane_name in overlay.lane_names:
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


def _inject_hold_group(root: ET.Element, hold_group: ET.Element) -> None:
    waves_group = _find_required(root, _svg("g"), "waves_0")
    waves_group.insert(0, hold_group)


def _build_summary_group(top_y: float, summary_lines: Sequence[SummaryLine]) -> ET.Element:
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

    for index, line in enumerate(summary_lines):
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


def _build_summary_lines(diagram: DiagramSpec, metrics: WaveDrom) -> List[SummaryLine]:
    summary_lines = []
    for rule in diagram.rules:
        if isinstance(rule, ResponseRule):
            color = RESPONSE_COLOR
            detail = f"{rule.trigger_event} -> after [{rule.min_delay}:{rule.max_delay}] {rule.response_event}"
        elif isinstance(rule, HoldUntilRule):
            color = HOLD_STROKE
            detail = f"{_expr_to_label(rule.predicate_expr)} from {rule.start_event} until {rule.end_event}"
        elif isinstance(rule, NotBeforeRule):
            color = NOT_BEFORE_COLOR
            detail = f"not {rule.forbidden_event} before {rule.reference_event}"
        else:  # pragma: no cover - guarded by core model types
            continue

        text = f"{rule.name}: {detail}"
        summary_lines.append(SummaryLine(color=color, text=text, width=metrics.text_width(text, 11)))

    return summary_lines


def _expr_to_label(expr: EventExpr) -> str:
    if expr.predicates and all(predicate.op == "stable" and predicate.value is None for predicate in expr.predicates):
        if len(expr.predicates) == 1:
            return f"stable({expr.predicates[0].signal})"
        joined = ", ".join(predicate.signal for predicate in expr.predicates)
        return f"stable({{{joined}}})"
    return " and ".join(_predicate_to_label(predicate) for predicate in expr.predicates)


def _predicate_to_label(predicate: EventPredicate) -> str:
    if predicate.value is None:
        return f"{predicate.op}({predicate.signal})"
    return f"{predicate.op}({predicate.signal}, {predicate.value})"


def _ensure_renderer_defs(root: ET.Element) -> None:
    defs = root.find(_svg("defs"))
    if defs is None:
        defs = ET.Element(_svg("defs"))
        root.insert(0, defs)

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


def _ensure_renderer_style(root: ET.Element) -> None:
    style = ET.SubElement(root, _svg("style"), {"type": "text/css"})
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


def _shift_waves_group(root: ET.Element, overlay_height: float) -> None:
    waves_group = _find_required(root, _svg("g"), "waves_0")
    if overlay_height <= 0:
        waves_group.attrib.pop("transform", None)
        return
    waves_group.set("transform", f"translate(0,{_fmt(overlay_height)})")


def _find_required(root: ET.Element, tag: str, element_id: str) -> ET.Element:
    element = root.find(f"{tag}[@id='{element_id}']")
    if element is None:
        raise RuntimeError(f"WaveDrom output is missing expected element '{element_id}'")
    return element


def _wrap_outer_svg(diagram: DiagramSpec, inner_svg: str, inner_width: float, inner_height: float, metrics: WaveDrom) -> str:
    card_width = inner_width + CARD_PADDING * 2.0
    card_height = inner_height + CARD_PADDING * 2.0
    meta_text = f"@({diagram.clocking.edge} {diagram.clocking.signal})"
    if diagram.clocking.disable_iff:
        meta_text += f" disable iff ({diagram.clocking.disable_iff})"

    outer_width = math.ceil(
        max(
            OUTER_MARGIN_X * 2.0 + card_width,
            OUTER_MARGIN_X * 2.0 + metrics.text_width(diagram.name, 20) + 12.0,
            OUTER_MARGIN_X * 2.0 + metrics.text_width(meta_text, 12) + 12.0,
        )
    )
    outer_height = math.ceil(OUTER_MARGIN_Y * 2.0 + HEADER_HEIGHT + card_height)

    card_x = OUTER_MARGIN_X
    card_y = OUTER_MARGIN_Y + HEADER_HEIGHT
    inner_x = card_x + CARD_PADDING
    inner_y = card_y + CARD_PADDING

    return "\n".join(
        [
            f'<svg xmlns="{SVG_NS}" width="{_fmt(outer_width)}" height="{_fmt(outer_height)}" '
            f'viewBox="0 0 {_fmt(outer_width)} {_fmt(outer_height)}" class="timing-shell" '
            f'font-family="IBM Plex Sans, Segoe UI, Arial, sans-serif">',
            '<style type="text/css">',
            ".timing-bg { fill: #f4f7fb; }",
            ".timing-card { fill: #ffffff; stroke: #d7dde7; stroke-width: 1; }",
            ".timing-title { fill: #0f172a; font-size: 20px; font-weight: 700; }",
            ".timing-meta { fill: #475569; font-size: 12px; }",
            "</style>",
            f'<rect class="timing-bg" x="0" y="0" width="{_fmt(outer_width)}" height="{_fmt(outer_height)}"/>',
            f'<text x="{_fmt(OUTER_MARGIN_X)}" y="{_fmt(OUTER_MARGIN_Y + 18.0)}" class="timing-title">{_xml(diagram.name)}</text>',
            f'<text x="{_fmt(OUTER_MARGIN_X)}" y="{_fmt(OUTER_MARGIN_Y + 38.0)}" class="timing-meta">{_xml(meta_text)}</text>',
            f'<rect class="timing-card" x="{_fmt(card_x)}" y="{_fmt(card_y)}" width="{_fmt(card_width)}" height="{_fmt(card_height)}" rx="12" ry="12"/>',
            f'<g transform="translate({_fmt(inner_x)},{_fmt(inner_y)})">',
            inner_svg,
            "</g>",
            "</svg>",
        ]
    )


def _event_x(layout: WaveDromLayout, tick: int, anchor: str) -> float:
    x = layout.plot_origin_x + tick * layout.tick_width
    if anchor == "center":
        x += layout.tick_width / 2.0
    return x


def _svg(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def _to_float(value) -> float:
    return float(str(value).replace("px", ""))


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
