"""Decoration-to-primitive emitters for the native renderer."""

from __future__ import annotations

import random

from sva_toolkit.timing.render2.decorations import AnnotationPolicy, Decoration, DecorationKind, select_decorations
from sva_toolkit.timing.render2.primitives import BBox, Fill, FontSpec, Line, Point, Rect, Stroke, Text
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.visual import VisibilityClass

from sva_toolkit.timing.render2.native.geometry import NativeGeometry


def emit_decorations(
    scene: TimingScene,
    spec: RenderSpec,
    geometry: NativeGeometry,
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    rng = random.Random(spec.seed ^ 0x4D3C2B1A)
    policy = AnnotationPolicy(spec.annotations.policy)
    selected = select_decorations(scene, policy, rng)
    primitives: list[object] = []
    warnings: list[str] = []

    for decoration in selected:
        emitted, emitted_warnings = _emit_decoration(decoration, scene, spec, geometry)
        primitives.extend(emitted)
        warnings.extend(emitted_warnings)

    if policy == AnnotationPolicy.NUISANCE_ONLY:
        primitives.extend(_synthetic_nuisance_text(spec, geometry, rng))

    if policy == AnnotationPolicy.DEBUG_LEAKY:
        primitives.extend(_debug_anchor_labels(scene, spec, geometry))

    return tuple(primitives), tuple(warnings)


def _emit_decoration(
    decoration: Decoration,
    scene: TimingScene,
    spec: RenderSpec,
    geometry: NativeGeometry,
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    policy = AnnotationPolicy(spec.annotations.policy)
    if decoration.kind == DecorationKind.VERTICAL_GUIDE:
        if not spec.annotations.semantic_guides_enabled and policy != AnnotationPolicy.DEBUG_LEAKY:
            return (), ()
        return _vertical_guide(decoration, spec, geometry), ()
    if decoration.kind == DecorationKind.HORIZONTAL_GUIDE:
        return _horizontal_guide(decoration, spec, geometry), ()
    if decoration.kind == DecorationKind.MEASUREMENT_BRACKET:
        return _measurement_bracket(decoration, spec, geometry), ()
    if decoration.kind == DecorationKind.HIGHLIGHT_REGION:
        return _highlight_region(decoration, scene, spec, geometry), ()
    if decoration.kind == DecorationKind.NUISANCE_TEXT:
        return _nuisance_text(decoration, spec, geometry), ()
    if decoration.kind == DecorationKind.CAPTION:
        return _caption(decoration, spec, geometry), ()
    return (), (f"downgraded unsupported decoration kind: {decoration.kind.value}",)


def _vertical_guide(decoration: Decoration, spec: RenderSpec, geometry: NativeGeometry) -> tuple[object, ...]:
    if decoration.anchor_tick is None:
        return ()
    colors = _palette(spec)
    x = geometry.tick_to_x(decoration.anchor_tick)
    return (
        Line(
            role="vertical_helper_line",
            z=90,
            p0=Point(x, geometry.plot_origin.y),
            p1=Point(x, geometry.plot_bottom),
            stroke=Stroke(
                color=colors["helper"],
                width=max(0.6, spec.style.waveform_stroke.width * 0.65),
                dasharray=(4.0, 3.0),
                opacity=0.68,
            ),
        ),
    )


def _horizontal_guide(decoration: Decoration, spec: RenderSpec, geometry: NativeGeometry) -> tuple[object, ...]:
    colors = _palette(spec)
    lanes = decoration.lane_names or geometry.lane_names
    primitives: list[object] = []
    for lane_name in lanes:
        if lane_name not in geometry.lane_names:
            continue
        y = geometry.lane_center_y(geometry.lane_names.index(lane_name))
        primitives.append(
            Line(
                role="horizontal_helper_line",
                z=91,
                p0=Point(geometry.plot_origin.x, y),
                p1=Point(geometry.plot_right, y),
                stroke=Stroke(color=colors["helper"], width=0.7, dasharray=(3.0, 3.0), opacity=0.45),
            )
        )
    return tuple(primitives)


def _measurement_bracket(decoration: Decoration, spec: RenderSpec, geometry: NativeGeometry) -> tuple[object, ...]:
    if decoration.span is None:
        return ()
    colors = _palette(spec)
    x0 = geometry.tick_to_x(decoration.span[0])
    x1 = geometry.tick_to_x(decoration.span[1])
    if x1 < x0:
        x0, x1 = x1, x0
    y = max(8.0, geometry.plot_origin.y - spec.style.primary_font.size_px * 0.9)
    stroke = Stroke(color=colors["accent"], width=max(0.8, spec.style.waveform_stroke.width * 0.75), linecap="butt")
    primitives: list[object] = [
        Line(role="measurement_bracket", z=94, p0=Point(x0, y), p1=Point(x1, y), stroke=stroke),
        Line(
            role="measurement_bracket",
            z=94,
            p0=Point(x0, y - 4.0),
            p1=Point(x0, y + 4.0),
            stroke=stroke,
        ),
        Line(
            role="measurement_bracket",
            z=94,
            p0=Point(x1, y - 4.0),
            p1=Point(x1, y + 4.0),
            stroke=stroke,
        ),
    ]

    can_show_text = (
        AnnotationPolicy(spec.annotations.policy) in {AnnotationPolicy.NATURAL_MEASUREMENTS, AnnotationPolicy.DEBUG_LEAKY}
        and decoration.text
        and decoration.visibility_class == VisibilityClass.VISIBLE_TEXT
    )
    if can_show_text:
        primitives.append(
            Text(
                role="measurement_bracket",
                z=95,
                text=decoration.text or "",
                anchor=Point((x0 + x1) / 2, y - 6.0),
                font=FontSpec(
                    family=spec.style.primary_font.family,
                    size_px=max(8.0, spec.style.primary_font.size_px * 0.82),
                    weight=spec.style.primary_font.weight,
                    color=colors["accent"],
                ),
                text_anchor="middle",
                visibility_class=VisibilityClass.VISIBLE_TEXT.value,
            )
        )
    return tuple(primitives)


def _highlight_region(
    decoration: Decoration,
    scene: TimingScene,
    spec: RenderSpec,
    geometry: NativeGeometry,
) -> tuple[object, ...]:
    del scene
    if decoration.span is None:
        return ()
    colors = _palette(spec)
    lane_names = decoration.lane_names or geometry.lane_names
    primitives: list[object] = []
    x0 = geometry.tick_to_x(decoration.span[0])
    x1 = geometry.tick_to_x(decoration.span[1] + 1)
    for lane_name in lane_names:
        lane_bbox = geometry.bbox_for_lane(lane_name)
        if lane_bbox is None:
            continue
        primitives.append(
            Rect(
                role="hold_highlight",
                z=45,
                bbox=BBox(x=x0, y=lane_bbox.y, width=max(0.0, x1 - x0), height=lane_bbox.height),
                stroke=None,
                fill=Fill(color=colors["accent"], opacity=0.08),
                radius=0.0,
            )
        )
    return tuple(primitives)


def _nuisance_text(decoration: Decoration, spec: RenderSpec, geometry: NativeGeometry) -> tuple[object, ...]:
    if not decoration.text:
        return ()
    colors = _palette(spec)
    return (
        Text(
            role="nuisance_text",
            z=120,
            text=decoration.text,
            anchor=Point(geometry.plot_origin.x, max(12.0, geometry.plot_origin.y - 6.0)),
            font=FontSpec(
                family=spec.style.primary_font.family,
                size_px=max(8.0, spec.style.primary_font.size_px * 0.78),
                color=colors["nuisance"],
            ),
            visibility_class=VisibilityClass.HIDDEN_SEMANTIC.value,
        ),
    )


def _caption(decoration: Decoration, spec: RenderSpec, geometry: NativeGeometry) -> tuple[object, ...]:
    if not decoration.text:
        return ()
    colors = _palette(spec)
    return (
        Text(
            role="caption_text",
            z=125,
            text=decoration.text,
            anchor=Point(geometry.plot_origin.x, geometry.plot_bottom + spec.style.primary_font.size_px * 1.25),
            font=FontSpec(
                family=spec.style.primary_font.family,
                size_px=max(8.0, spec.style.primary_font.size_px * 0.9),
                color=colors["nuisance"],
            ),
            visibility_class=VisibilityClass.HIDDEN_SEMANTIC.value,
        ),
    )


def _synthetic_nuisance_text(
    spec: RenderSpec,
    geometry: NativeGeometry,
    rng: random.Random,
) -> tuple[Text, ...]:
    colors = _palette(spec)
    count = max(1, spec.annotations.nuisance_text_count)
    font = FontSpec(
        family=spec.style.primary_font.family,
        size_px=max(8.0, spec.style.primary_font.size_px * 0.76),
        color=colors["nuisance"],
    )
    choices = ("note", "ref", "timing", "sample", "scope", "margin")
    y = max(font.size_px + 2.0, geometry.plot_origin.y - 7.0)
    texts: list[Text] = []
    for index in range(count):
        token = choices[int(rng.random() * len(choices)) % len(choices)]
        x = geometry.plot_origin.x + rng.random() * max(1.0, geometry.plot_width * 0.75)
        texts.append(
            Text(
                role="nuisance_text",
                z=120 + index,
                text=f"{token} {index + 1}",
                anchor=Point(x, y - index * (font.size_px * 1.1)),
                font=font,
                visibility_class=VisibilityClass.HIDDEN_SEMANTIC.value,
            )
        )
    return tuple(texts)


def _debug_anchor_labels(scene: TimingScene, spec: RenderSpec, geometry: NativeGeometry) -> tuple[Text, ...]:
    colors = _palette(spec)
    font = FontSpec(
        family=spec.style.primary_font.family,
        size_px=max(8.0, spec.style.primary_font.size_px * 0.8),
        weight="600",
        color=colors["debug"],
    )
    labels: list[Text] = []
    y = geometry.plot_origin.y + font.size_px
    for event in scene.events:
        labels.append(
            Text(
                role="debug_overlay",
                z=140,
                text=event.name,
                anchor=Point(geometry.tick_to_x(event.tick) + 3.0, y),
                font=font,
                visibility_class=VisibilityClass.DEBUG_OVERLAY.value,
            )
        )
    return tuple(labels)


def _palette(spec: RenderSpec) -> dict[str, str]:
    palette = tuple(spec.style.palette)

    def color(index: int, fallback: str) -> str:
        return palette[index] if index < len(palette) else fallback

    return {
        "helper": color(9, "#666666"),
        "accent": color(9, "#666666"),
        "nuisance": color(10, "#7a7a7a"),
        "debug": color(11, "#c00000"),
    }
