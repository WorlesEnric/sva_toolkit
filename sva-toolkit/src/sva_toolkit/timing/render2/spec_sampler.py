"""Seeded render2 profile and RenderSpec sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
import random
from typing import Any

from sva_toolkit.timing.render2.decorations import AnnotationPolicy
from sva_toolkit.timing.render2.native.sampler import sample_native_render_spec
from sva_toolkit.timing.render2.native.style_kernels import derive_palette
from sva_toolkit.timing.render2.primitives import FontSpec, Stroke
from sva_toolkit.timing.render2.profiles import ProfileSet, RenderProfile
from sva_toolkit.timing.render2.protocol import DEFAULT_REGISTRY
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import (
    DegradationSpec,
    LayoutSpec,
    PageSpec,
    RenderSpec,
    StyleSpec,
)


def sample_profile(rng: random.Random, profile_set: ProfileSet) -> RenderProfile:
    """Pick a profile from the set with deterministic rng draws."""

    total = sum(profile_set.weights)
    if total <= 0.0:
        raise ValueError("profile set must have positive total weight")
    cursor = rng.random() * total
    upto = 0.0
    for profile, weight in zip(profile_set.profiles, profile_set.weights, strict=True):
        upto += weight
        if cursor <= upto:
            return profile
    return profile_set.profiles[-1]


def sample_render_spec(
    rng: random.Random,
    *,
    profile: RenderProfile,
    scene: TimingScene,
    target_canvas_hint: tuple[int, int] | None = None,
) -> RenderSpec:
    """Sample a complete RenderSpec using only the supplied RNG."""

    spec = _renderer_default_spec(rng, profile=profile, target_canvas_hint=target_canvas_hint)
    spec = _apply_profile_overrides(rng, spec, profile=profile)
    spec = replace(spec, layout=_fit_layout_to_scene(spec.layout, scene, target_canvas_hint))
    return spec


def _renderer_default_spec(
    rng: random.Random,
    *,
    profile: RenderProfile,
    target_canvas_hint: tuple[int, int] | None,
) -> RenderSpec:
    renderer = _registered_renderer(profile.renderer_id)
    default_render_spec = getattr(renderer, "default_render_spec", None) if renderer is not None else None
    if callable(default_render_spec):
        return default_render_spec(profile, rng)

    native_profile = _native_sampler_profile(profile)
    return sample_native_render_spec(
        rng,
        profile=native_profile,
        target_canvas_width_hint=float(target_canvas_hint[0]) if target_canvas_hint is not None else None,
    )


def _registered_renderer(renderer_id: str) -> object | None:
    try:
        return DEFAULT_REGISTRY.get(renderer_id)
    except KeyError:
        return None


def _native_sampler_profile(profile: RenderProfile) -> str:
    if profile.id == "debug-current":
        return "debug-current"
    if profile.id == "document-native":
        return "document-native"
    if profile.id in {"datasheet-native", "tikz-datasheet", "ascii-rfc"}:
        return "datasheet-native"
    if profile.id in {"ood-native", "plantuml-ood", "gtkwave-ood"}:
        return "ood-native"
    return "clean-native"


def _apply_profile_overrides(rng: random.Random, spec: RenderSpec, *, profile: RenderProfile) -> RenderSpec:
    style = _apply_style_overrides(spec.style, profile.style_overrides)
    style = replace(
        style,
        family=profile.style_family,
        color_mode=profile.color_mode,
        palette=derive_palette(style.palette, profile.color_mode),
    )
    annotations = replace(
        spec.annotations,
        policy=profile.annotation_policy,
        measurement_label_style=profile.style_family,
        helper_line_density=_helper_line_density(spec.annotations.helper_line_density, profile.annotation_policy),
        nuisance_text_count=_nuisance_text_count(spec.annotations.nuisance_text_count, profile.annotation_policy),
        semantic_guides_enabled=_semantic_guides_enabled(profile.annotation_policy),
    )
    return replace(
        spec,
        renderer_id=profile.renderer_id,
        style=style,
        annotations=annotations,
        page=_page_spec(spec.page, enabled=profile.page_enabled),
        degradation=_sample_degradation(rng, spec.degradation, family=profile.degradation_family),
        profile=profile.id,
        extras={**dict(spec.extras), "style_family": profile.style_family},
    )


def _apply_style_overrides(style: StyleSpec, overrides: Mapping[str, Any]) -> StyleSpec:
    if not overrides:
        return style

    style_fields = {field.name for field in fields(StyleSpec)}
    nested_fields = {"primary_font", "label_font", "waveform_stroke", "grid_stroke"}
    direct = {key: value for key, value in overrides.items() if key in style_fields - nested_fields}
    primary_font = _font_with_overrides(style.primary_font, "primary_font", overrides)
    label_font = _font_with_overrides(style.label_font, "label_font", overrides)
    waveform_stroke = _stroke_with_overrides(style.waveform_stroke, "waveform_stroke", overrides)
    grid_stroke = _stroke_with_overrides(style.grid_stroke, "grid_stroke", overrides)

    if "font_family" in overrides:
        primary_font = replace(primary_font, family=str(overrides["font_family"]))
    if "label_font_family" in overrides:
        label_font = replace(label_font, family=str(overrides["label_font_family"]))
    if "font_size_px" in overrides:
        primary_font = replace(primary_font, size_px=float(overrides["font_size_px"]))
    if "label_font_size_px" in overrides:
        label_font = replace(label_font, size_px=float(overrides["label_font_size_px"]))
    if "stroke_width" in overrides:
        waveform_stroke = replace(waveform_stroke, width=float(overrides["stroke_width"]))
    if "grid_stroke_width" in overrides:
        grid_stroke = replace(grid_stroke, width=float(overrides["grid_stroke_width"]))

    return replace(
        style,
        **direct,
        primary_font=primary_font,
        label_font=label_font,
        waveform_stroke=waveform_stroke,
        grid_stroke=grid_stroke,
    )


def _font_with_overrides(font: FontSpec, prefix: str, overrides: Mapping[str, Any]) -> FontSpec:
    values: dict[str, Any] = {}
    for field in fields(FontSpec):
        key = f"{prefix}_{field.name}"
        if key in overrides:
            values[field.name] = overrides[key]
    return replace(font, **values) if values else font


def _stroke_with_overrides(stroke: Stroke, prefix: str, overrides: Mapping[str, Any]) -> Stroke:
    values: dict[str, Any] = {}
    for field in fields(Stroke):
        key = f"{prefix}_{field.name}"
        if key in overrides:
            values[field.name] = overrides[key]
    return replace(stroke, **values) if values else stroke


def _helper_line_density(density: float, policy: AnnotationPolicy) -> float:
    if policy == AnnotationPolicy.NONE:
        return 0.0
    if policy == AnnotationPolicy.DEBUG_LEAKY:
        return max(density, 0.15)
    if policy == AnnotationPolicy.NATURAL_MEASUREMENTS:
        return max(density, 0.25)
    return density


def _nuisance_text_count(count: int, policy: AnnotationPolicy) -> int:
    if policy == AnnotationPolicy.NONE:
        return 0
    if policy == AnnotationPolicy.DEBUG_LEAKY:
        return count
    return count


def _semantic_guides_enabled(policy: AnnotationPolicy) -> bool:
    return policy in {
        AnnotationPolicy.GEOMETRIC_GUIDES,
        AnnotationPolicy.NATURAL_MEASUREMENTS,
        AnnotationPolicy.DEBUG_LEAKY,
    }


def _page_spec(page: PageSpec, *, enabled: bool) -> PageSpec:
    if enabled:
        return replace(page, enabled=True)
    return PageSpec(
        enabled=False,
        caption_above=False,
        caption_below=False,
        surrounding_paragraph=False,
        table_border=False,
        page_header=False,
        page_footer=False,
        crop_mode="tight",
    )


def _sample_degradation(rng: random.Random, degradation: DegradationSpec, *, family: str) -> DegradationSpec:
    if family == "none":
        return DegradationSpec(family="none")
    if family in {"clean", "plain_text"}:
        return replace(
            degradation,
            family=family,
            blur_sigma=min(degradation.blur_sigma, 0.16),
            noise_sigma=min(degradation.noise_sigma, 0.06),
            rotation_deg=_clamp(degradation.rotation_deg, -0.25, 0.25),
            perspective=min(degradation.perspective, 0.012),
            jpeg_quality=max(degradation.jpeg_quality, 90),
        )
    if family in {"document", "screenshot"}:
        return DegradationSpec(
            family=family,
            blur_sigma=rng.uniform(0.08, 0.70),
            noise_sigma=rng.uniform(0.02, 0.16),
            contrast=rng.uniform(0.82, 1.12),
            brightness=rng.uniform(0.90, 1.10),
            rotation_deg=rng.uniform(-0.9, 0.9),
            perspective=rng.uniform(0.0, 0.045),
            jpeg_quality=rng.randrange(72, 94),
            morphology="none",
        )
    if family == "ood":
        return DegradationSpec(
            family=family,
            blur_sigma=rng.uniform(0.12, 0.95),
            noise_sigma=rng.uniform(0.04, 0.22),
            contrast=rng.uniform(0.72, 1.20),
            brightness=rng.uniform(0.82, 1.16),
            rotation_deg=rng.uniform(-1.5, 1.5),
            perspective=rng.uniform(0.01, 0.08),
            jpeg_quality=rng.randrange(58, 90),
            morphology="dilate" if rng.random() < 0.35 else "none",
        )
    return replace(degradation, family=family)


def _fit_layout_to_scene(
    layout: LayoutSpec,
    scene: TimingScene,
    target_canvas_hint: tuple[int, int] | None,
) -> LayoutSpec:
    lanes = max(1, len(scene.lanes))
    ticks = max(1, scene.ticks.total_ticks)
    lane_height = layout.lane_height
    lane_pitch = layout.lane_pitch
    tick_width = layout.tick_width

    if lanes >= 20:
        lane_height = min(lane_height, 22.0)
    elif lanes >= 12:
        lane_height = min(lane_height, 26.0)

    if ticks >= 40:
        tick_width = min(tick_width, 34.0)
    elif ticks >= 28:
        tick_width = min(tick_width, 48.0)
    elif ticks >= 18:
        tick_width = min(tick_width, 64.0)

    if target_canvas_hint is not None:
        target_width, target_height = target_canvas_hint
        horizontal_overhead = layout.margin.x + layout.margin.width + 96.0
        max_tick_width = max(8.0, (target_width - horizontal_overhead) / ticks)
        tick_width = min(tick_width, max_tick_width)

        vertical_overhead = layout.margin.y + layout.margin.height + 24.0
        max_pitch = max(14.0, (target_height - vertical_overhead) / lanes)
        lane_pitch = min(lane_pitch, max_pitch)
        lane_height = min(lane_height, max(12.0, lane_pitch - 2.0))

    lane_pitch = max(lane_pitch, lane_height + 2.0)
    return replace(
        layout,
        lane_height=max(12.0, lane_height),
        lane_pitch=max(14.0, lane_pitch),
        tick_width=max(8.0, tick_width),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = ["sample_profile", "sample_render_spec"]
