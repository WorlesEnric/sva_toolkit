"""Waveform primitive emitters for the native SVG renderer."""

from __future__ import annotations

from collections.abc import Iterable

from sva_toolkit.timing.render2.primitives import BBox, Fill, FontSpec, Line, Path, Point, Polyline, Rect, Stroke, Text
from sva_toolkit.timing.render2.scene import CutRegion, LaneScene, LaneType, SampleRun
from sva_toolkit.timing.render2.spec import StyleSpec
from sva_toolkit.timing.visual import VisibilityClass

from sva_toolkit.timing.render2.native.geometry import NativeGeometry


def emit_lane_wave(lane: LaneScene, lane_index: int, geometry: NativeGeometry, style: StyleSpec) -> tuple[object, ...]:
    if lane.lane_type == LaneType.BUS or lane.width_bits is not None:
        return emit_bus_wave(lane, lane_index, geometry, style)
    return emit_bit_wave(lane, lane_index, geometry, style)


def emit_bit_wave(lane: LaneScene, lane_index: int, geometry: NativeGeometry, style: StyleSpec) -> tuple[object, ...]:
    primitives: list[object] = []
    wave_stroke = style.waveform_stroke
    high_y = geometry.high_y(lane_index)
    low_y = geometry.low_y(lane_index)

    for run in lane.runs:
        x0 = geometry.tick_to_x(run.start_tick)
        x1 = geometry.tick_to_x(run.end_tick + 1)
        value = _normalized_value(run)
        if value == "x":
            primitives.extend(_unknown_region(x0, x1, lane_index, geometry, style, role="unknown_region", z=50))
            continue
        if value == "z":
            primitives.extend(_unknown_region(x0, x1, lane_index, geometry, style, role="hiz_region", z=51))
            continue
        y = high_y if value == "1" else low_y
        primitives.append(
            Polyline(
                role="bit_wave_high" if value == "1" else "bit_wave_low",
                z=70,
                points=(Point(x0, y), Point(x1, y)),
                stroke=wave_stroke,
            )
        )

    primitives.extend(_bit_transitions(lane.runs, lane_index, geometry, style))
    return tuple(primitives)


def emit_bus_wave(lane: LaneScene, lane_index: int, geometry: NativeGeometry, style: StyleSpec) -> tuple[object, ...]:
    primitives: list[object] = []
    colors = _palette(style)
    top = geometry.bus_top_y(lane_index)
    bottom = geometry.bus_bottom_y(lane_index)
    stroke = style.waveform_stroke
    bus_style = style.bus_style
    fill = _bus_fill(style, colors)
    font = FontSpec(
        family=style.primary_font.family,
        size_px=max(8.0, min(style.primary_font.size_px, geometry.layout.lane_height * 0.42)),
        weight=style.primary_font.weight,
        style=style.primary_font.style,
        color=colors["wave"],
    )

    for run in lane.runs:
        x0 = geometry.tick_to_x(run.start_tick)
        x1 = geometry.tick_to_x(run.end_tick + 1)
        value = _normalized_value(run)
        if value == "x":
            primitives.extend(_unknown_region(x0, x1, lane_index, geometry, style, role="unknown_region", z=52))
            continue
        if value == "z":
            primitives.extend(_unknown_region(x0, x1, lane_index, geometry, style, role="hiz_region", z=53))
            continue

        if bus_style == "boxed":
            primitives.append(
                Rect(
                    role="bus_region",
                    z=62,
                    bbox=BBox(x=x0, y=top, width=max(0.0, x1 - x0), height=bottom - top),
                    stroke=stroke,
                    fill=fill,
                    radius=2.0,
                )
            )
        else:
            primitives.append(
                Path(
                    role="bus_region",
                    z=62,
                    d=_bus_region_path(x0, x1, top, bottom, geometry.layout.tick_width * 0.14),
                    stroke=stroke,
                    fill=fill,
                )
            )

        if bus_style == "hatched":
            primitives.extend(
                _hatch_lines(
                    x0,
                    top,
                    x1 - x0,
                    bottom - top,
                    role="bus_region_edge",
                    stroke=Stroke(color=colors["wave"], width=max(0.5, stroke.width * 0.55), opacity=0.45),
                    z=63,
                )
            )
        if _can_print_bus_value(run.value):
            run_width = max(0.0, x1 - x0)
            value_text = _fit_bus_label(str(run.value), run_width, font)
            if value_text:
                primitives.append(
                    Text(
                        role="bus_value_text",
                        z=72,
                        text=value_text,
                        anchor=Point((x0 + x1) / 2, geometry.lane_center_y(lane_index) + font.size_px * 0.35),
                        font=font,
                        text_anchor="middle",
                        visibility_class=VisibilityClass.VISIBLE_TEXT.value,
                    )
                )

    return tuple(primitives)


def emit_cut_markers(cuts: Iterable[CutRegion], geometry: NativeGeometry, style: StyleSpec) -> tuple[object, ...]:
    primitives: list[object] = []
    colors = _palette(style)
    stroke = Stroke(
        color=colors["accent"],
        width=max(1.0, style.waveform_stroke.width * 0.9),
        linecap=style.waveform_stroke.linecap,
        linejoin="round",
        opacity=0.85,
    )
    top = geometry.plot_origin.y
    bottom = geometry.plot_bottom

    for cut in cuts:
        x = geometry.tick_to_x(max(0, min(geometry.total_ticks, cut.start_tick)))
        if style.cut_style == "gray_band":
            x1 = geometry.tick_to_x(min(geometry.total_ticks, cut.end_tick + 1))
            primitives.append(
                Rect(
                    role="cut_marker",
                    z=80,
                    bbox=BBox(x=x, y=top, width=max(4.0, x1 - x), height=bottom - top),
                    stroke=None,
                    fill=Fill(color=colors["grid"], opacity=0.35),
                )
            )
        elif style.cut_style == "double_slash":
            offset = 5.0
            primitives.append(
                Path(
                    role="cut_marker",
                    z=82,
                    d=(
                        f"M {_fmt(x - offset)} {_fmt(bottom)} L {_fmt(x + offset)} {_fmt(top)} "
                        f"M {_fmt(x + offset)} {_fmt(bottom)} L {_fmt(x + offset * 3)} {_fmt(top)}"
                    ),
                    stroke=stroke,
                    fill=None,
                )
            )
        elif style.cut_style == "zigzag":
            step = max(4.0, geometry.layout.lane_height * 0.18)
            y = top
            parts = [f"M {_fmt(x)} {_fmt(y)}"]
            toggle = 1
            while y < bottom:
                y = min(bottom, y + step)
                parts.append(f"L {_fmt(x + toggle * 5.0)} {_fmt(y)}")
                toggle *= -1
            primitives.append(Path(role="cut_marker", z=82, d=" ".join(parts), stroke=stroke, fill=None))
        else:
            primitives.append(
                Text(
                    role="cut_marker",
                    z=82,
                    text="...",
                    anchor=Point(x, geometry.plot_origin.y + geometry.plot_height / 2),
                    font=FontSpec(
                        family=style.primary_font.family,
                        size_px=style.primary_font.size_px * 1.2,
                        weight=style.primary_font.weight,
                        color=colors["accent"],
                    ),
                    text_anchor="middle",
                    visibility_class=VisibilityClass.VISIBLE_CONVENTION.value,
                )
            )

    return tuple(primitives)


def _bit_transitions(
    runs: tuple[SampleRun, ...],
    lane_index: int,
    geometry: NativeGeometry,
    style: StyleSpec,
) -> tuple[object, ...]:
    transitions: list[object] = []
    for previous, current in zip(runs, runs[1:]):
        prev_value = _normalized_value(previous)
        current_value = _normalized_value(current)
        if prev_value not in {"0", "1"} or current_value not in {"0", "1"} or prev_value == current_value:
            continue
        x = geometry.tick_to_x(current.start_tick)
        y0 = geometry.high_y(lane_index) if prev_value == "1" else geometry.low_y(lane_index)
        y1 = geometry.high_y(lane_index) if current_value == "1" else geometry.low_y(lane_index)
        transitions.append(_transition_primitive(x, y0, y1, geometry.layout.tick_width, style))
    return tuple(transitions)


def _transition_primitive(x: float, y0: float, y1: float, tick_width: float, style: StyleSpec) -> object:
    width = max(3.0, min(tick_width * 0.18, 12.0))
    stroke = style.waveform_stroke
    if style.transition_shape == "curved":
        mid_y = (y0 + y1) / 2
        curve_y = mid_y - (y1 - y0) * 0.24
        return Path(
            role="bit_transition",
            z=71,
            d=f"M {_fmt(x - width)} {_fmt(y0)} Q {_fmt(x)} {_fmt(curve_y)} {_fmt(x + width)} {_fmt(y1)}",
            stroke=stroke,
            fill=None,
        )
    if style.transition_shape == "slanted":
        return Line(role="bit_transition", z=71, p0=Point(x - width, y0), p1=Point(x + width, y1), stroke=stroke)
    if style.transition_shape == "step":
        mid_x = x
        return Path(
            role="bit_transition",
            z=71,
            d=f"M {_fmt(x - width)} {_fmt(y0)} L {_fmt(mid_x)} {_fmt(y0)} "
            f"L {_fmt(mid_x)} {_fmt(y1)} L {_fmt(x + width)} {_fmt(y1)}",
            stroke=stroke,
            fill=None,
        )
    return Line(role="bit_transition", z=71, p0=Point(x, y0), p1=Point(x, y1), stroke=stroke)


def _unknown_region(
    x0: float,
    x1: float,
    lane_index: int,
    geometry: NativeGeometry,
    style: StyleSpec,
    *,
    role: str,
    z: int,
) -> tuple[object, ...]:
    colors = _palette(style)
    bbox = BBox(
        x=x0,
        y=geometry.lane_top(lane_index) + geometry.layout.lane_height * 0.18,
        width=max(0.0, x1 - x0),
        height=geometry.layout.lane_height * 0.64,
    )
    if role == "hiz_region":
        fill_color = colors["hiz"]
        hatch_color = colors["wave"]
        opacity = 0.24
        hatch = True
    else:
        fill_color = _unknown_fill_color(style, colors)
        hatch_color = colors["accent"]
        opacity = 0.30 if style.unknown_style != "dashed_outline" else 0.0
        hatch = style.unknown_style in {"x_hatch", "green_hatch", "orange_hatch", "diagonal_stripes"}

    stroke = Stroke(
        color=hatch_color,
        width=max(0.6, style.waveform_stroke.width * 0.75),
        dasharray=(4.0, 3.0) if style.unknown_style == "dashed_outline" else (),
        opacity=0.75,
    )
    primitives: list[object] = [
        Rect(
            role=role,
            z=z,
            bbox=bbox,
            stroke=stroke if style.unknown_style == "dashed_outline" else None,
            fill=Fill(color=fill_color, opacity=opacity),
            radius=2.0,
        )
    ]
    if hatch:
        primitives.extend(
            _hatch_lines(
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height,
                role=role,
                stroke=Stroke(color=hatch_color, width=max(0.45, style.waveform_stroke.width * 0.5), opacity=0.55),
                z=z + 1,
            )
        )
    return tuple(primitives)


def _hatch_lines(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    role: str,
    stroke: Stroke,
    z: int,
) -> tuple[Line, ...]:
    lines: list[Line] = []
    spacing = max(5.0, height * 0.45)
    offset = -height
    while offset < width + height:
        x0 = x + max(0.0, offset)
        y0 = y + max(0.0, -offset)
        x1 = x + min(width, offset + height)
        y1 = y + min(height, width - offset)
        lines.append(Line(role=role, z=z, p0=Point(x0, y0), p1=Point(x1, y1), stroke=stroke))
        offset += spacing
    return tuple(lines)


def _bus_region_path(x0: float, x1: float, top: float, bottom: float, bevel: float) -> str:
    bevel = min(bevel, max(0.0, (x1 - x0) / 3))
    mid = (top + bottom) / 2
    return (
        f"M {_fmt(x0)} {_fmt(mid)} L {_fmt(x0 + bevel)} {_fmt(top)} L {_fmt(x1 - bevel)} {_fmt(top)} "
        f"L {_fmt(x1)} {_fmt(mid)} L {_fmt(x1 - bevel)} {_fmt(bottom)} "
        f"L {_fmt(x0 + bevel)} {_fmt(bottom)} Z"
    )


def _bus_fill(style: StyleSpec, colors: dict[str, str]) -> Fill | None:
    if style.bus_style in {"filled", "hatched", "inline_text", "boxed"}:
        return Fill(color=colors["bus"], opacity=0.55 if style.bus_style != "filled" else 0.78)
    return None


def _unknown_fill_color(style: StyleSpec, colors: dict[str, str]) -> str:
    if style.unknown_style == "green_hatch":
        return colors["unknown_green"]
    if style.unknown_style == "orange_hatch":
        return colors["unknown_orange"]
    return colors["unknown"]


def _normalized_value(run: SampleRun) -> str:
    if run.is_high_z:
        return "z"
    if run.is_unknown:
        return "x"
    value = str(run.value).strip().lower()
    if value in {"1", "h", "high"}:
        return "1"
    if value in {"0", "l", "low"}:
        return "0"
    if value in {"z", "hz", "highz"}:
        return "z"
    if value in {"x", "?", "unknown"}:
        return "x"
    return value


def _can_print_bus_value(value: str) -> bool:
    normalized = str(value).strip().lower()
    return bool(normalized) and normalized not in {"x", "z", "?", "unknown", "highz"}


def _fit_bus_label(value: str, run_width: float, font: FontSpec) -> str:
    """Return the largest prefix of ``value`` that fits in ``run_width``.

    Bus envelopes shorter than a single character get an empty string back so
    the caller can skip the Text primitive entirely. Long bus labels in tight
    runs are truncated with a trailing ``…`` so the model still recovers a
    distinguishing prefix.
    """

    from sva_toolkit.timing.render2.native.text_metrics import estimate_text_width

    text = str(value)
    if not text:
        return ""
    padding = max(2.0, font.size_px * 0.4)
    available = max(0.0, run_width - padding)
    if available <= 0.0:
        return ""
    if estimate_text_width(text, font) <= available:
        return text
    if estimate_text_width("…", font) > available:
        return ""
    for cut in range(len(text) - 1, 0, -1):
        candidate = f"{text[:cut]}…"
        if estimate_text_width(candidate, font) <= available:
            return candidate
    return ""


def _palette(style: StyleSpec) -> dict[str, str]:
    palette = tuple(style.palette)

    def color(index: int, fallback: str) -> str:
        return palette[index] if index < len(palette) else fallback

    return {
        "background": color(0, "#ffffff"),
        "card": color(1, "#ffffff"),
        "wave": color(2, "#000000"),
        "grid": color(3, "#d7d7d7"),
        "label": color(4, "#1f4e8c"),
        "lane": color(5, "#f6f8fb"),
        "bus": color(6, "#fff2b3"),
        "unknown": color(7, "#d9d9d9"),
        "hiz": color(8, "#cfe9d6"),
        "accent": color(9, "#666666"),
        "unknown_green": color(12, "#d9f3df"),
        "unknown_orange": color(13, "#fde4c6"),
    }


def _fmt(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
