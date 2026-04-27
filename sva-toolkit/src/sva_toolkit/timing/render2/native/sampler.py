"""RenderSpec sampler for the native SVG renderer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sva_toolkit.timing.render2.decorations import AnnotationPolicy
from sva_toolkit.timing.render2.primitives import BBox
from sva_toolkit.timing.render2.spec import (
    AnnotationSpec,
    DegradationSpec,
    LayoutSpec,
    PageSpec,
    RasterSpec,
    RenderSpec,
)

from sva_toolkit.timing.render2.native.style_kernels import (
    BUS_STYLES,
    GRID_MODES,
    LABEL_POSITIONS,
    TRANSITION_SHAPES,
    UNKNOWN_STYLES,
    debug_style,
    sample_style_kernel,
)


_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "debug-current": {
        "debug": True,
        "policy": AnnotationPolicy.GEOMETRIC_GUIDES,
        "semantic_guides_enabled": True,
        "helper_line_density": 0.15,
    },
    "clean-native": {
        "family": "clean-native",
        "policy": AnnotationPolicy.GEOMETRIC_GUIDES,
        "semantic_guides_enabled": True,
        "helper_line_density_range": (0.15, 0.55),
    },
    "datasheet-native": {
        "family": "datasheet-native",
        "policy": AnnotationPolicy.NATURAL_MEASUREMENTS,
        "font_family_weights": (
            ("Times New Roman, Times, serif", 3.0),
            ("Georgia, serif", 2.0),
            ("Helvetica, Arial, sans-serif", 1.0),
            ("Courier New, Courier, monospace", 0.6),
        ),
        "color_mode_weights": (("monochrome", 0.44), ("grayscale", 0.30), ("color", 0.18), ("low_contrast", 0.08)),
        "grid_mode_weights": (("dense", 0.45), ("major_minor", 0.40), ("major_only", 0.10), ("sparse", 0.05)),
        "transition_shape_weights": (("sharp", 0.45), ("slanted", 0.30), ("step", 0.20), ("curved", 0.05)),
        "bus_style_weights": (("empty", 0.36), ("boxed", 0.30), ("inline_text", 0.20), ("filled", 0.10), ("hatched", 0.04)),
        "label_position_weights": (("left", 0.62), ("right", 0.20), ("inside_left", 0.12), ("inside_right", 0.06)),
        "helper_line_density_range": (0.25, 0.65),
        "font_size_mean": 11.5,
        "grid_width_mean": 0.55,
    },
    "document-native": {
        "family": "document-native",
        "policy": AnnotationPolicy.GEOMETRIC_GUIDES,
        "semantic_guides_enabled": True,
        "helper_line_density_range": (0.10, 0.45),
        "page_enabled": True,
        "caption_above": True,
        "caption_below": True,
        "surrounding_paragraph": True,
        "page_header": True,
        "page_footer": True,
        "table_border_probability": 0.65,
    },
    "ood-native": {
        "family": "ood-native",
        "policy": AnnotationPolicy.NATURAL_MEASUREMENTS,
        "semantic_guides_enabled": True,
        "color_mode_weights": (("low_contrast", 0.62), ("inverted", 0.18), ("color", 0.12), ("grayscale", 0.08)),
        "transition_shape_weights": (("curved", 0.52), ("step", 0.22), ("slanted", 0.18), ("sharp", 0.08)),
        "bus_style_weights": (("hatched", 0.46), ("inline_text", 0.22), ("boxed", 0.18), ("filled", 0.10), ("empty", 0.04)),
        "unknown_style_weights": (("orange_hatch", 0.32), ("green_hatch", 0.26), ("diagonal_stripes", 0.18), ("x_hatch", 0.14), ("dashed_outline", 0.10)),
        "grid_mode_weights": (("dense", 0.36), ("major_minor", 0.30), ("sparse", 0.20), ("major_only", 0.12), ("none", 0.02)),
        "helper_line_density_range": (0.65, 1.0),
        "nuisance_text_range": (1, 4),
        "stroke_width_mean": 2.1,
        "stroke_width_std": 0.55,
        "font_size_mean": 13.5,
    },
}


def sample_native_render_spec(
    rng: Any,
    *,
    profile: str,
    layout_hint: object | None = None,
    target_canvas_width_hint: float | None = None,
) -> RenderSpec:
    """Sample a complete native renderer spec using only the supplied RNG."""

    overrides = _PROFILE_OVERRIDES[profile]
    if overrides.get("debug"):
        return _debug_spec(rng, profile)

    style = sample_style_kernel(rng, overrides)
    layout = _sample_layout(rng, style.label_font.size_px, overrides, layout_hint, target_canvas_width_hint)
    annotations = _sample_annotations(rng, overrides)
    page = _sample_page(rng, overrides)
    raster = RasterSpec(
        dpi=int(_clamp(round(rng.gauss(float(overrides.get("dpi_mean", 132)), 28)), 72, 240)),
        antialias=rng.random() > float(overrides.get("pixelated_probability", 0.12)),
        output_format="png",
        jpeg_quality=int(_clamp(round(rng.gauss(90, 7)), 65, 100)),
    )
    degradation = DegradationSpec(
        family="native",
        blur_sigma=_clamp(rng.gauss(0.08, 0.08), 0.0, 0.45),
        noise_sigma=_clamp(rng.gauss(0.02, 0.025), 0.0, 0.14),
        contrast=_clamp(rng.gauss(1.0, 0.06), 0.82, 1.18),
        brightness=_clamp(rng.gauss(1.0, 0.045), 0.88, 1.12),
        rotation_deg=_clamp(rng.gauss(0.0, 0.18), -0.8, 0.8),
        perspective=_clamp(rng.gauss(0.0, 0.01), 0.0, 0.04),
        jpeg_quality=int(_clamp(round(rng.gauss(92, 5)), 70, 100)),
    )

    return RenderSpec(
        renderer_id="native_svg",
        style=style,
        layout=layout,
        annotations=annotations,
        page=page,
        raster=raster,
        degradation=degradation,
        seed=rng.randrange(0, 2**31),
        profile=profile,
        extras=_sample_extras(rng, overrides),
    )


def native_render_spec_sampler(rng: Any, profile: str) -> RenderSpec:
    return sample_native_render_spec(rng, profile=profile)


def _debug_spec(rng: Any, profile: str) -> RenderSpec:
    del rng
    return RenderSpec(
        renderer_id="native_svg",
        style=debug_style(),
        layout=LayoutSpec(
            lane_height=28.0,
            lane_pitch=34.0,
            tick_width=42.0,
            label_position="left",
            label_alignment="end",
            margin=BBox(16.0, 22.0, 18.0, 18.0),
        ),
        annotations=AnnotationSpec(
            policy=AnnotationPolicy.GEOMETRIC_GUIDES,
            measurement_label_style="debug",
            helper_line_density=0.15,
            nuisance_text_count=0,
            semantic_guides_enabled=True,
        ),
        page=PageSpec(
            enabled=False,
            caption_above=False,
            caption_below=False,
            surrounding_paragraph=False,
            table_border=False,
            page_header=False,
            page_footer=False,
            crop_mode="tight",
        ),
        raster=RasterSpec(dpi=96, antialias=True, output_format="png", jpeg_quality=95),
        degradation=DegradationSpec(family="none"),
        seed=0,
        profile=profile,
        extras={
            "outer_radius": "4.0",
            "outer_stroke_width": "1.0",
            "lane_bands": "0",
            "lane_band_opacity": "0.0",
        },
    )


def _sample_layout(
    rng: Any,
    label_font_size: float,
    overrides: Mapping[str, Any],
    layout_hint: object | None,
    target_canvas_width_hint: float | None,
) -> LayoutSpec:
    lane_height = _hint(layout_hint, "lane_height", _clamp(rng.gauss(28.0, 5.6), 18.0, 40.0))
    lane_pitch = _hint(layout_hint, "lane_pitch", lane_height + rng.uniform(2.0, 8.5))
    tick_width = _hint(layout_hint, "tick_width", rng.uniform(16.0, 80.0))
    if target_canvas_width_hint is not None:
        tick_width = _clamp((target_canvas_width_hint - 120.0) / rng.uniform(5.0, 10.0), 16.0, 80.0)
    label_position = str(
        _hint(
            layout_hint,
            "label_position",
            overrides.get("label_position")
            or _weighted_choice(
                rng,
                overrides.get(
                    "label_position_weights",
                    tuple((position, 1.0) for position in LABEL_POSITIONS),
                ),
            ),
        )
    )
    label_alignment = "start" if label_position in {"right", "inside_left"} else "end"
    margin = BBox(
        x=_hint(layout_hint, "margin_left", rng.uniform(10.0, 30.0) + label_font_size * 0.15),
        y=_hint(layout_hint, "margin_top", rng.uniform(12.0, 34.0)),
        width=_hint(layout_hint, "margin_right", rng.uniform(10.0, 30.0)),
        height=_hint(layout_hint, "margin_bottom", rng.uniform(12.0, 34.0)),
    )
    return LayoutSpec(
        lane_height=lane_height,
        lane_pitch=lane_pitch,
        tick_width=tick_width,
        label_position=label_position,
        label_alignment=label_alignment,
        margin=margin,
        grouped_lanes=rng.random() < float(overrides.get("grouped_lanes_probability", 0.18)),
        multiline_labels=rng.random() < float(overrides.get("multiline_labels_probability", 0.08)),
    )


def _sample_annotations(rng: Any, overrides: Mapping[str, Any]) -> AnnotationSpec:
    density_range = overrides.get("helper_line_density_range")
    if density_range is None:
        helper_density = float(overrides.get("helper_line_density", rng.uniform(0.0, 0.6)))
    else:
        helper_density = rng.uniform(float(density_range[0]), float(density_range[1]))
    nuisance_range = overrides.get("nuisance_text_range", (0, 2))
    return AnnotationSpec(
        policy=overrides.get("policy", AnnotationPolicy.GEOMETRIC_GUIDES),
        measurement_label_style=str(overrides.get("measurement_label_style", "native")),
        helper_line_density=_clamp(helper_density, 0.0, 1.0),
        nuisance_text_count=rng.randrange(int(nuisance_range[0]), int(nuisance_range[1]) + 1),
        semantic_guides_enabled=bool(overrides.get("semantic_guides_enabled", True)),
    )


def _sample_page(rng: Any, overrides: Mapping[str, Any]) -> PageSpec:
    enabled = bool(overrides.get("page_enabled", False))
    return PageSpec(
        enabled=enabled,
        caption_above=bool(overrides.get("caption_above", enabled and rng.random() < 0.55)),
        caption_below=bool(overrides.get("caption_below", enabled and rng.random() < 0.35)),
        surrounding_paragraph=bool(overrides.get("surrounding_paragraph", enabled and rng.random() < 0.45)),
        table_border=enabled and rng.random() < float(overrides.get("table_border_probability", 0.25)),
        page_header=bool(overrides.get("page_header", enabled and rng.random() < 0.35)),
        page_footer=bool(overrides.get("page_footer", enabled and rng.random() < 0.35)),
        crop_mode=str(overrides.get("crop_mode", "page_fragment" if enabled else "tight")),
    )


def _sample_extras(rng: Any, overrides: Mapping[str, Any]) -> dict[str, str]:
    del overrides
    return {
        "outer_radius": _fmt(rng.uniform(0.0, 8.0)),
        "outer_stroke_width": _fmt(rng.uniform(0.35, 1.6)),
        "lane_bands": "1" if rng.random() < 0.55 else "0",
        "lane_band_opacity": _fmt(rng.uniform(0.04, 0.18)),
        "major_grid_every": str(rng.randrange(2, 6)),
    }


def _hint(layout_hint: object | None, name: str, default: Any) -> Any:
    if layout_hint is None:
        return default
    if isinstance(layout_hint, Mapping):
        return layout_hint.get(name, default)
    return getattr(layout_hint, name, default)


def _weighted_choice(rng: Any, values: Any) -> str:
    values = tuple((str(value), float(weight)) for value, weight in values)
    total = sum(max(0.0, weight) for _value, weight in values)
    cursor = rng.random() * total
    upto = 0.0
    for value, weight in values:
        upto += max(0.0, weight)
        if cursor <= upto:
            return value
    return values[-1][0]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


__all__ = [
    "BUS_STYLES",
    "GRID_MODES",
    "TRANSITION_SHAPES",
    "UNKNOWN_STYLES",
    "native_render_spec_sampler",
    "sample_native_render_spec",
]
