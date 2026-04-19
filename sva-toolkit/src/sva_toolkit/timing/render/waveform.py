"""Native SVG waveform renderer for concrete timing scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

from sva_toolkit.timing.core.scenario import ScenarioDocument, SignalDecl, SignalKind


SVG_NS = "http://www.w3.org/2000/svg"

LEFT_MARGIN = 20.0
RIGHT_MARGIN = 20.0
TOP_MARGIN = 18.0
BOTTOM_MARGIN = 20.0

NAME_COL_WIDTH = 120.0
TITLE_HEIGHT = 52.0
TITLE_GAP = 14.0

LANE_HEIGHT = 28.0
LANE_GAP = 4.0
LANE_INSET = 4.0
TICK_WIDTH = 40.0

GRID_COLOR = "#E8E8E8"
NAME_COLOR = "#0066CC"
UNKNOWN_COLOR = "#02D98A"
HIGHZ_COLOR = "#FF8C00"
BUS_FILL = "#FFFFB0"
BLACK = "#000000"
TRANSITION_WIDTH = 6.0
STROKE_WIDTH = 1.5


@dataclass(frozen=True)
class LaneSpec:
    """Render-local lane configuration."""

    name: str
    kind: str
    display_name: str
    samples: tuple[str, ...] = ()


def render_waveform_svg(document: ScenarioDocument) -> str:
    """Render a concrete timing scenario into a native SVG waveform diagram."""

    ticks = _resolve_ticks(document)
    lanes = _build_lanes(document, ticks)
    lane_count = max(1, len(lanes))

    waveform_left = LEFT_MARGIN + NAME_COL_WIDTH
    waveform_top = TOP_MARGIN + TITLE_HEIGHT + TITLE_GAP
    waveform_width = max(float(ticks) * TICK_WIDTH, TICK_WIDTH)
    waveform_height = lane_count * LANE_HEIGHT + max(lane_count - 1, 0) * LANE_GAP
    width = int(LEFT_MARGIN + NAME_COL_WIDTH + waveform_width + RIGHT_MARGIN)
    height = int(TOP_MARGIN + TITLE_HEIGHT + TITLE_GAP + waveform_height + BOTTOM_MARGIN)
    clocking = _clocking_label(document)

    lines = [
        (
            f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMinYMin meet" '
            'style="max-width: 100%; height: auto;" role="img" aria-labelledby="title desc">'
        ),
        "  <defs>",
        _defs(),
        "  </defs>",
        "  <style>",
        _styles(),
        "  </style>",
        f"  <title id=\"title\">{escape(document.name)}</title>",
        f"  <desc id=\"desc\">{escape(_diagram_summary(document, lanes, ticks))}</desc>",
        f"  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" class=\"diagram-bg\" />",
        f"  <text id=\"diagram-title\" x=\"{_fmt(LEFT_MARGIN)}\" y=\"{_fmt(TOP_MARGIN + 20.0)}\" class=\"diagram-title\">{escape(document.name)}</text>",
        f"  <text x=\"{_fmt(LEFT_MARGIN)}\" y=\"{_fmt(TOP_MARGIN + 40.0)}\" class=\"diagram-meta\">{escape(clocking)}</text>",
        (
            f"  <line x1=\"{_fmt(LEFT_MARGIN)}\" y1=\"{_fmt(TOP_MARGIN + TITLE_HEIGHT)}\" "
            f"x2=\"{_fmt(width - RIGHT_MARGIN)}\" y2=\"{_fmt(TOP_MARGIN + TITLE_HEIGHT)}\" class=\"frame-line\" />"
        ),
        (
            f"  <line x1=\"{_fmt(waveform_left)}\" y1=\"{_fmt(waveform_top - 8.0)}\" "
            f"x2=\"{_fmt(waveform_left)}\" y2=\"{_fmt(waveform_top + waveform_height)}\" class=\"name-divider\" />"
        ),
        "  <g class=\"grid\">",
    ]

    _append_grid(lines, waveform_left, waveform_top, waveform_height, ticks)
    lines.append("  </g>")
    lines.append("  <g class=\"lanes\">")

    for index, lane in enumerate(lanes):
        lane_top = waveform_top + index * (LANE_HEIGHT + LANE_GAP)
        lane_bottom = lane_top + LANE_HEIGHT
        lane_mid = lane_top + LANE_HEIGHT / 2.0

        lines.append(
            f"    <text x=\"{_fmt(waveform_left - 10.0)}\" y=\"{_fmt(lane_mid)}\" class=\"signal-name\">{escape(lane.display_name)}</text>"
        )
        lines.append(
            f"    <line x1=\"{_fmt(waveform_left)}\" y1=\"{_fmt(lane_bottom)}\" "
            f"x2=\"{_fmt(waveform_left + waveform_width)}\" y2=\"{_fmt(lane_bottom)}\" class=\"lane-divider\" />"
        )
        lines.append(f"    <g data-lane=\"{escape(lane.name)}\">")
        if lane.kind == "clock":
            _append_clock_lane(lines, waveform_left, lane_top, ticks)
        elif lane.kind == SignalKind.BIT.value:
            _append_bit_lane(lines, waveform_left, lane_top, lane.samples, ticks)
        else:
            _append_bus_lane(lines, waveform_left, lane_top, lane.samples, ticks)
        lines.append("    </g>")

    lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines)


def _resolve_ticks(document: ScenarioDocument) -> int:
    tick_count = document.ticks or max((len(signal.samples) for signal in document.signals), default=0)
    return max(1, tick_count)


def _build_lanes(document: ScenarioDocument, ticks: int) -> list[LaneSpec]:
    lanes = [
        LaneSpec(
            name=document.clocking.signal,
            kind="clock",
            display_name=document.clocking.signal,
        )
    ]
    for signal in document.signals:
        if signal.name == document.clocking.signal:
            continue
        lanes.append(
            LaneSpec(
                name=signal.name,
                kind=signal.kind.value,
                display_name=signal.display_name,
                samples=_normalize_samples(signal, ticks),
            )
        )
    return lanes


def _normalize_samples(signal: SignalDecl, ticks: int) -> tuple[str, ...]:
    if len(signal.samples) >= ticks:
        raw_samples = signal.samples[:ticks]
    else:
        fill = "x" if signal.kind == SignalKind.BIT else ""
        raw_samples = signal.samples + (fill,) * (ticks - len(signal.samples))

    if signal.kind == SignalKind.BIT:
        return tuple(_normalize_bit_sample(sample) for sample in raw_samples)
    return tuple(sample for sample in raw_samples)


def _normalize_bit_sample(sample: str) -> str:
    normalized = sample.strip().lower()
    if normalized in {"0", "1", "x", "z"}:
        return normalized
    return "x"


def _append_grid(lines: list[str], left: float, top: float, height: float, ticks: int) -> None:
    for boundary in range(ticks + 1):
        x = left + boundary * TICK_WIDTH
        lines.append(
            f"    <line x1=\"{_fmt(x)}\" y1=\"{_fmt(top)}\" x2=\"{_fmt(x)}\" y2=\"{_fmt(top + height)}\" class=\"grid-line\" />"
        )


def _append_clock_lane(lines: list[str], left: float, lane_top: float, ticks: int) -> None:
    y_high, y_mid, y_low = _bit_levels(lane_top)
    half_tick = TICK_WIDTH / 2.0

    for tick in range(ticks):
        x = left + tick * TICK_WIDTH
        lines.append(
            f"      <rect x=\"{_fmt(x)}\" y=\"{_fmt(y_high)}\" width=\"{_fmt(half_tick)}\" height=\"{_fmt(y_low - y_high)}\" class=\"clock-high\" />"
        )

    commands = [f"M {_fmt(left)} {_fmt(y_low)}"]
    for tick in range(ticks):
        tick_left = left + tick * TICK_WIDTH
        tick_mid = tick_left + half_tick
        tick_right = tick_left + TICK_WIDTH
        commands.extend(
            [
                f"L {_fmt(tick_left)} {_fmt(y_high)}",
                f"L {_fmt(tick_mid)} {_fmt(y_high)}",
                f"L {_fmt(tick_mid)} {_fmt(y_low)}",
                f"L {_fmt(tick_right)} {_fmt(y_low)}",
            ]
        )

    lines.append(f"      <path d=\"{' '.join(commands)}\" class=\"wave-line\" />")
    lines.append(f"      <line x1=\"{_fmt(left)}\" y1=\"{_fmt(y_low)}\" x2=\"{_fmt(left + ticks * TICK_WIDTH)}\" y2=\"{_fmt(y_low)}\" class=\"wave-line\" />")
    lines.append(f"      <line x1=\"{_fmt(left)}\" y1=\"{_fmt(y_high)}\" x2=\"{_fmt(left)}\" y2=\"{_fmt(y_low)}\" class=\"wave-line\" />")


def _append_bit_lane(lines: list[str], left: float, lane_top: float, samples: tuple[str, ...], ticks: int) -> None:
    y_high, y_mid, y_low = _bit_levels(lane_top)

    for index, sample in enumerate(samples[:ticks]):
        x0 = left + index * TICK_WIDTH
        x1 = x0 + TICK_WIDTH

        if sample == "x":
            lines.append(
                f"      <rect x=\"{_fmt(x0)}\" y=\"{_fmt(y_high)}\" width=\"{_fmt(TICK_WIDTH)}\" height=\"{_fmt(y_low - y_high)}\" class=\"unknown-region\" />"
            )
            lines.append(f"      <line x1=\"{_fmt(x0)}\" y1=\"{_fmt(y_high)}\" x2=\"{_fmt(x1)}\" y2=\"{_fmt(y_high)}\" class=\"wave-line\" />")
            lines.append(f"      <line x1=\"{_fmt(x0)}\" y1=\"{_fmt(y_low)}\" x2=\"{_fmt(x1)}\" y2=\"{_fmt(y_low)}\" class=\"wave-line\" />")
            if index == 0 or samples[index - 1] != "x":
                _append_x_cross(lines, x0, y_high, y_low)
            if index == ticks - 1 or samples[index + 1] != "x":
                _append_x_cross(lines, x1, y_high, y_low)
            continue

        if sample == "z":
            lines.append(
                f"      <rect x=\"{_fmt(x0)}\" y=\"{_fmt(y_high)}\" width=\"{_fmt(TICK_WIDTH)}\" height=\"{_fmt(y_low - y_high)}\" class=\"highz-region\" />"
            )
            lines.append(f"      <line x1=\"{_fmt(x0)}\" y1=\"{_fmt(y_mid)}\" x2=\"{_fmt(x1)}\" y2=\"{_fmt(y_mid)}\" class=\"wave-line\" />")
            continue

        y = y_high if sample == "1" else y_low
        lines.append(f"      <line x1=\"{_fmt(x0)}\" y1=\"{_fmt(y)}\" x2=\"{_fmt(x1)}\" y2=\"{_fmt(y)}\" class=\"wave-line\" />")

    for index in range(ticks - 1):
        current = samples[index]
        nxt = samples[index + 1]
        if current == nxt or "x" in {current, nxt}:
            continue

        boundary_x = left + (index + 1) * TICK_WIDTH
        if "z" in {current, nxt}:
            lines.append(
                f"      <line x1=\"{_fmt(boundary_x)}\" y1=\"{_fmt(_bit_sample_y(current, y_high, y_mid, y_low))}\" "
                f"x2=\"{_fmt(boundary_x)}\" y2=\"{_fmt(_bit_sample_y(nxt, y_high, y_mid, y_low))}\" class=\"wave-line\" />"
            )
            continue

        lines.append(f"      <line x1=\"{_fmt(boundary_x)}\" y1=\"{_fmt(y_low if current == '0' else y_high)}\" x2=\"{_fmt(boundary_x)}\" y2=\"{_fmt(y_high if nxt == '1' else y_low)}\" class=\"wave-line\" />")


def _append_bus_lane(lines: list[str], left: float, lane_top: float, samples: tuple[str, ...], ticks: int) -> None:
    y_high, _, y_low = _bit_levels(lane_top)
    runs = _group_runs(samples[:ticks])
    half_transition = TRANSITION_WIDTH / 2.0

    for run_index, (start, end, value) in enumerate(runs):
        left_edge = left + start * TICK_WIDTH
        right_edge = left + end * TICK_WIDTH
        region_left = left_edge if run_index == 0 else left_edge + half_transition
        region_right = right_edge if run_index == len(runs) - 1 else right_edge - half_transition

        if region_right < region_left:
            midpoint = (left_edge + right_edge) / 2.0
            region_left = midpoint - 0.5
            region_right = midpoint + 0.5

        fill_class = "unknown-region" if _is_unknown_bus_value(value) else "bus-fill"
        lines.append(
            f"      <rect x=\"{_fmt(region_left)}\" y=\"{_fmt(y_high)}\" width=\"{_fmt(region_right - region_left)}\" height=\"{_fmt(y_low - y_high)}\" class=\"{fill_class}\" />"
        )
        lines.append(f"      <line x1=\"{_fmt(region_left)}\" y1=\"{_fmt(y_high)}\" x2=\"{_fmt(region_right)}\" y2=\"{_fmt(y_high)}\" class=\"wave-line\" />")
        lines.append(f"      <line x1=\"{_fmt(region_left)}\" y1=\"{_fmt(y_low)}\" x2=\"{_fmt(region_right)}\" y2=\"{_fmt(y_low)}\" class=\"wave-line\" />")

        if not _is_unknown_bus_value(value):
            text_x = (region_left + region_right) / 2.0
            text_y = (y_high + y_low) / 2.0
            lines.append(
                f"      <text x=\"{_fmt(text_x)}\" y=\"{_fmt(text_y)}\" class=\"bus-text\">{escape(value)}</text>"
            )

        if _is_unknown_bus_value(value):
            if run_index == 0:
                _append_x_cross(lines, left_edge, y_high, y_low)
            if run_index == len(runs) - 1:
                _append_x_cross(lines, right_edge, y_high, y_low)

    for start, end, _ in runs[:-1]:
        boundary_x = left + end * TICK_WIDTH
        _append_x_cross(lines, boundary_x, y_high, y_low)


def _group_runs(samples: tuple[str, ...]) -> list[tuple[int, int, str]]:
    if not samples:
        return [(0, 1, "x")]

    runs: list[tuple[int, int, str]] = []
    start = 0
    current = samples[0]
    for index, sample in enumerate(samples[1:], start=1):
        if sample == current:
            continue
        runs.append((start, index, current))
        start = index
        current = sample
    runs.append((start, len(samples), current))
    return runs


def _append_x_cross(lines: list[str], x: float, y_high: float, y_low: float) -> None:
    half = TRANSITION_WIDTH / 2.0
    lines.append(
        f"      <line x1=\"{_fmt(x - half)}\" y1=\"{_fmt(y_high)}\" x2=\"{_fmt(x + half)}\" y2=\"{_fmt(y_low)}\" class=\"wave-line\" />"
    )
    lines.append(
        f"      <line x1=\"{_fmt(x - half)}\" y1=\"{_fmt(y_low)}\" x2=\"{_fmt(x + half)}\" y2=\"{_fmt(y_high)}\" class=\"wave-line\" />"
    )


def _bit_levels(lane_top: float) -> tuple[float, float, float]:
    y_high = lane_top + LANE_INSET
    y_low = lane_top + LANE_HEIGHT - LANE_INSET
    y_mid = (y_high + y_low) / 2.0
    return y_high, y_mid, y_low


def _bit_sample_y(sample: str, y_high: float, y_mid: float, y_low: float) -> float:
    if sample == "1":
        return y_high
    if sample == "0":
        return y_low
    return y_mid


def _is_unknown_bus_value(value: str) -> bool:
    normalized = value.strip().lower()
    return bool(normalized) and set(normalized) <= {"x"}


def _clocking_label(document: ScenarioDocument) -> str:
    label = f"@({document.clocking.edge} {document.clocking.signal})"
    if document.clocking.disable_iff:
        return f"{label} disable iff ({document.clocking.disable_iff})"
    return label


def _diagram_summary(document: ScenarioDocument, lanes: list[LaneSpec], ticks: int) -> str:
    return f"{document.name}: {len(lanes)} waveform lanes across {ticks} ticks, clocked by {_clocking_label(document)}."


def _defs() -> str:
    return "\n".join(
        [
            "    <pattern id=\"unknown-hatch\" patternUnits=\"userSpaceOnUse\" width=\"8\" height=\"8\">",
            "      <rect x=\"0\" y=\"0\" width=\"8\" height=\"8\" fill=\"#F2FFF9\" />",
            f"      <path d=\"M -2 8 L 4 2 M 0 10 L 10 0 M 6 10 L 10 6\" stroke=\"{UNKNOWN_COLOR}\" stroke-width=\"1\" fill=\"none\" />",
            "    </pattern>",
            "    <pattern id=\"highz-hatch\" patternUnits=\"userSpaceOnUse\" width=\"8\" height=\"8\">",
            "      <rect x=\"0\" y=\"0\" width=\"8\" height=\"8\" fill=\"#FFF7EA\" />",
            f"      <path d=\"M -2 8 L 4 2 M 0 10 L 10 0 M 6 10 L 10 6\" stroke=\"{HIGHZ_COLOR}\" stroke-width=\"1\" fill=\"none\" />",
            "    </pattern>",
        ]
    )


def _styles() -> str:
    return "\n".join(
        [
            "    svg { background: #FFFFFF; }",
            "    text { font-family: Helvetica, Arial, sans-serif; }",
            "    .diagram-bg { fill: #FFFFFF; }",
            "    .diagram-title { fill: #111111; font-size: 16px; font-weight: 700; }",
            "    .diagram-meta { fill: #555555; font-size: 12px; }",
            f"    .signal-name {{ fill: {NAME_COLOR}; font-size: 13px; text-anchor: end; dominant-baseline: middle; }}",
            f"    .grid-line {{ stroke: {GRID_COLOR}; stroke-width: 1; stroke-dasharray: 3 3; }}",
            "    .frame-line, .name-divider { stroke: #D8D8D8; stroke-width: 1; }",
            "    .lane-divider { stroke: #F1F1F1; stroke-width: 1; }",
            f"    .wave-line {{ fill: none; stroke: {BLACK}; stroke-width: {STROKE_WIDTH}; stroke-linecap: square; stroke-linejoin: miter; }}",
            f"    .clock-high {{ fill: {BLACK}; }}",
            f"    .bus-fill {{ fill: {BUS_FILL}; }}",
            "    .unknown-region { fill: url(#unknown-hatch); }",
            "    .highz-region { fill: url(#highz-hatch); }",
            "    .bus-text { fill: #000000; font-size: 12px; text-anchor: middle; dominant-baseline: middle; }",
        ]
    )


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")
