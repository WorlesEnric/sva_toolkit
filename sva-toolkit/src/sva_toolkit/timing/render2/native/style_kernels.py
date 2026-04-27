"""Randomized style kernels for the native SVG renderer."""

from __future__ import annotations

import colorsys
from collections.abc import Mapping, Sequence
from typing import Any

from sva_toolkit.timing.render2.primitives import FontSpec, Stroke
from sva_toolkit.timing.render2.spec import StyleSpec


WEB_SAFE_FONTS = (
    "Helvetica, Arial, sans-serif",
    "Times New Roman, Times, serif",
    "Courier New, Courier, monospace",
    "Verdana, Geneva, sans-serif",
    "Georgia, serif",
    "Tahoma, Geneva, sans-serif",
    "Consolas, Courier New, monospace",
    "Arial Narrow, Arial, sans-serif",
)
GRID_DASHES = ((), (4.0, 4.0), (2.0, 2.0), (6.0, 2.0, 2.0, 2.0))
TRANSITION_SHAPES = ("sharp", "slanted", "curved", "step")
BUS_STYLES = ("filled", "empty", "hatched", "boxed", "inline_text")
UNKNOWN_STYLES = ("x_hatch", "gray_block", "green_hatch", "orange_hatch", "diagonal_stripes", "dashed_outline")
CUT_STYLES = ("zigzag", "ellipsis", "gray_band", "double_slash")
GRID_MODES = ("none", "sparse", "major_only", "major_minor", "dense")
COLOR_MODES = ("color", "grayscale", "monochrome", "low_contrast", "inverted")
LABEL_POSITIONS = ("left", "right", "inside_left", "inside_right")
LINECAPS = ("butt", "round", "square")


def debug_style() -> StyleSpec:
    palette = (
        "#ffffff",
        "#ffffff",
        "#000000",
        "#d8d8d8",
        "#0041a8",
        "#f7f7f7",
        "#fff7b8",
        "#e4e4e4",
        "#dbeedc",
        "#666666",
        "#777777",
        "#c00000",
        "#dff2e2",
        "#ffe4c2",
    )
    return StyleSpec(
        family="debug-current",
        palette=palette,
        primary_font=FontSpec(family="Helvetica, Arial, sans-serif", size_px=12.0, color="#000000"),
        label_font=FontSpec(family="Helvetica, Arial, sans-serif", size_px=12.0, weight="600", color="#0041a8"),
        waveform_stroke=Stroke(color="#000000", width=1.5, linecap="butt", linejoin="miter"),
        grid_stroke=Stroke(color="#d8d8d8", width=0.75, dasharray=()),
        grid_mode="major_only",
        bus_style="filled",
        unknown_style="x_hatch",
        cut_style="ellipsis",
        transition_shape="sharp",
        color_mode="color",
    )


def sample_style_kernel(rng: Any, overrides: Mapping[str, Any]) -> StyleSpec:
    font_family = str(overrides.get("font_family") or _weighted_choice(rng, _font_weights(overrides)))
    font_size = _clipped_gauss(
        rng,
        float(overrides.get("font_size_mean", 12.0)),
        float(overrides.get("font_size_std", 2.2)),
        9.0,
        18.0,
    )
    wave_width = _clipped_gauss(
        rng,
        float(overrides.get("stroke_width_mean", 1.45)),
        float(overrides.get("stroke_width_std", 0.55)),
        0.6,
        3.0,
    )
    color_mode = str(overrides.get("color_mode") or _weighted_choice(rng, _mode_weights(overrides, "color_mode")))
    base_palette = _sample_base_palette(rng)
    palette = derive_palette(base_palette, color_mode)
    wave_color = palette[2]
    grid_color = palette[3]
    label_color = palette[4]

    return StyleSpec(
        family=str(overrides.get("family", "native_random")),
        palette=palette,
        primary_font=FontSpec(
            family=font_family,
            size_px=font_size,
            weight=str(overrides.get("font_weight", "400")),
            color=wave_color,
        ),
        label_font=FontSpec(
            family=str(overrides.get("label_font_family", font_family)),
            size_px=_clamp(font_size + float(overrides.get("label_size_delta", 0.0)), 9.0, 18.0),
            weight=str(overrides.get("label_weight", _weighted_choice(rng, (("400", 0.65), ("600", 0.25), ("700", 0.10))))),
            color=label_color,
        ),
        waveform_stroke=Stroke(
            color=wave_color,
            width=wave_width,
            linecap=str(overrides.get("linecap") or _weighted_choice(rng, tuple((cap, 1.0) for cap in LINECAPS))),
            linejoin="round" if rng.random() < float(overrides.get("round_join_probability", 0.35)) else "miter",
        ),
        grid_stroke=Stroke(
            color=grid_color,
            width=_clipped_gauss(rng, float(overrides.get("grid_width_mean", 0.65)), 0.22, 0.35, 1.5),
            dasharray=tuple(overrides.get("grid_dasharray") or GRID_DASHES[int(rng.random() * len(GRID_DASHES))]),
            opacity=float(overrides.get("grid_opacity", 0.72)),
        ),
        grid_mode=str(overrides.get("grid_mode") or _weighted_choice(rng, _mode_weights(overrides, "grid_mode"))),
        bus_style=str(overrides.get("bus_style") or _weighted_choice(rng, _mode_weights(overrides, "bus_style"))),
        unknown_style=str(overrides.get("unknown_style") or _weighted_choice(rng, _mode_weights(overrides, "unknown_style"))),
        cut_style=str(overrides.get("cut_style") or _weighted_choice(rng, _mode_weights(overrides, "cut_style"))),
        transition_shape=str(
            overrides.get("transition_shape") or _weighted_choice(rng, _mode_weights(overrides, "transition_shape"))
        ),
        color_mode=color_mode,
    )


def derive_palette(base_palette: Sequence[str], color_mode: str) -> tuple[str, ...]:
    palette = tuple(base_palette)
    if color_mode == "grayscale":
        return tuple(_gray(color) for color in palette)
    if color_mode == "monochrome":
        return tuple(_mono(color) for color in palette)
    if color_mode == "low_contrast":
        background = palette[0]
        return tuple(_mix(color, background, 0.66 if index not in {2, 4, 9} else 0.47) for index, color in enumerate(palette))
    if color_mode == "inverted":
        return tuple(_invert(color) for color in palette)
    return palette


def contrast_ratio(foreground: str, background: str) -> float:
    fg = relative_luminance(foreground)
    bg = relative_luminance(background)
    lighter = max(fg, bg)
    darker = min(fg, bg)
    return (lighter + 0.05) / (darker + 0.05)


def relative_luminance(color: str) -> float:
    r, g, b = _hex_to_rgb(color)
    channels = []
    for value in (r, g, b):
        linear = value / 255.0
        channels.append(linear / 12.92 if linear <= 0.04045 else ((linear + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _font_weights(overrides: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    configured = overrides.get("font_family_weights")
    if configured is not None:
        return tuple((str(name), float(weight)) for name, weight in configured)
    return tuple((font, 1.0) for font in WEB_SAFE_FONTS)


def _mode_weights(overrides: Mapping[str, Any], key: str) -> tuple[tuple[str, float], ...]:
    configured = overrides.get(f"{key}_weights")
    if configured is not None:
        return tuple((str(name), float(weight)) for name, weight in configured)
    values: tuple[str, ...]
    if key == "color_mode":
        values = COLOR_MODES
        return (("color", 0.72), ("grayscale", 0.10), ("monochrome", 0.07), ("low_contrast", 0.06), ("inverted", 0.05))
    if key == "grid_mode":
        values = GRID_MODES
    elif key == "bus_style":
        values = BUS_STYLES
    elif key == "unknown_style":
        values = UNKNOWN_STYLES
    elif key == "cut_style":
        values = CUT_STYLES
    elif key == "transition_shape":
        values = TRANSITION_SHAPES
    else:
        values = ()
    return tuple((value, 1.0) for value in values)


def _sample_base_palette(rng: Any) -> tuple[str, ...]:
    hue = rng.random()
    accent_hue = (hue + 0.11 + rng.random() * 0.12) % 1.0
    label_hue = (hue + 0.55 + rng.random() * 0.1) % 1.0
    return (
        _hsl(hue, 0.10 + rng.random() * 0.08, 0.96 + rng.random() * 0.025),
        _hsl(hue, 0.06 + rng.random() * 0.08, 0.985),
        _hsl(hue, 0.42 + rng.random() * 0.32, 0.16 + rng.random() * 0.18),
        _hsl(hue, 0.08 + rng.random() * 0.12, 0.78 + rng.random() * 0.12),
        _hsl(label_hue, 0.36 + rng.random() * 0.32, 0.30 + rng.random() * 0.18),
        _hsl(hue, 0.10 + rng.random() * 0.15, 0.91 + rng.random() * 0.05),
        _hsl((hue + 0.16) % 1.0, 0.36 + rng.random() * 0.26, 0.78 + rng.random() * 0.10),
        _hsl(hue, 0.03 + rng.random() * 0.10, 0.78 + rng.random() * 0.10),
        _hsl((hue + 0.32) % 1.0, 0.20 + rng.random() * 0.22, 0.80 + rng.random() * 0.10),
        _hsl(accent_hue, 0.26 + rng.random() * 0.24, 0.34 + rng.random() * 0.18),
        _hsl((hue + 0.05) % 1.0, 0.10 + rng.random() * 0.16, 0.42 + rng.random() * 0.18),
        _hsl((hue + 0.90) % 1.0, 0.50 + rng.random() * 0.22, 0.36 + rng.random() * 0.20),
        _hsl((hue + 0.30) % 1.0, 0.28 + rng.random() * 0.20, 0.82 + rng.random() * 0.08),
        _hsl((hue + 0.08) % 1.0, 0.40 + rng.random() * 0.24, 0.82 + rng.random() * 0.08),
    )


def _weighted_choice(rng: Any, values: Sequence[tuple[str, float]]) -> str:
    total = sum(max(0.0, weight) for _value, weight in values)
    if total <= 0:
        return values[0][0]
    cursor = rng.random() * total
    upto = 0.0
    for value, weight in values:
        upto += max(0.0, weight)
        if cursor <= upto:
            return value
    return values[-1][0]


def _clipped_gauss(rng: Any, mean: float, std: float, lower: float, upper: float) -> float:
    return _clamp(rng.gauss(mean, std), lower, upper)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _hsl(hue: float, saturation: float, lightness: float) -> str:
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return _rgb_to_hex(round(r * 255), round(g * 255), round(b * 255))


def _gray(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    lum = round(0.2126 * r + 0.7152 * g + 0.0722 * b)
    return _rgb_to_hex(lum, lum, lum)


def _mono(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 232:
        value = 255
    elif lum > 170:
        value = 238
    elif lum > 96:
        value = 120
    else:
        value = 20
    return _rgb_to_hex(value, value, value)


def _invert(color: str) -> str:
    r, g, b = _hex_to_rgb(color)
    return _rgb_to_hex(255 - r, 255 - g, 255 - b)


def _mix(color: str, target: str, amount: float) -> str:
    r, g, b = _hex_to_rgb(color)
    tr, tg, tb = _hex_to_rgb(target)
    return _rgb_to_hex(
        round(r * (1.0 - amount) + tr * amount),
        round(g * (1.0 - amount) + tg * amount),
        round(b * (1.0 - amount) + tb * amount),
    )


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    stripped = color.strip().lstrip("#")
    if len(stripped) == 3:
        stripped = "".join(char * 2 for char in stripped)
    return int(stripped[0:2], 16), int(stripped[2:4], 16), int(stripped[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{_clamp_int(r):02x}{_clamp_int(g):02x}{_clamp_int(b):02x}"


def _clamp_int(value: int) -> int:
    return max(0, min(255, int(value)))
