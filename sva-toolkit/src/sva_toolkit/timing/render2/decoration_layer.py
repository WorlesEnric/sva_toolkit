"""Shared decoration-to-primitive rendering for timing renderers."""

from __future__ import annotations

import random
import warnings

from sva_toolkit.timing.render2.decorations import AnnotationPolicy, Decoration, DecorationKind, select_decorations
from sva_toolkit.timing.render2.primitives import BBox, Fill, FontSpec, Line, Path, Point, Primitive, Rect, Stroke, Text
from sva_toolkit.timing.render2.result import DiagramLayout
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.visual import VisibilityClass


def render_decorations(scene: TimingScene, spec: RenderSpec, layout: DiagramLayout) -> tuple[Primitive, ...]:
    """Render policy-selected scene decorations as overlay primitives."""

    rng = random.Random(spec.seed ^ 0xDEC047)
    primitives: list[Primitive] = []
    for decoration in select_decorations(scene, spec.annotations.policy, rng):
        if not _target_ref_exists(scene, decoration):
            warnings.warn(
                f"dropped decoration with missing target_ref: {decoration.target_ref}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        primitives.extend(_emit_decoration(scene, spec, layout, decoration, rng))
    return tuple(primitives)


def _emit_decoration(
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
    rng: random.Random,
) -> tuple[Primitive, ...]:
    if decoration.kind == DecorationKind.VERTICAL_GUIDE:
        return _vertical_guide(scene, spec, layout, decoration)
    if decoration.kind == DecorationKind.MEASUREMENT_BRACKET:
        return _measurement_bracket(scene, spec, layout, decoration)
    if decoration.kind == DecorationKind.HIGHLIGHT_REGION:
        return _highlight_region(scene, spec, layout, decoration)
    if decoration.kind == DecorationKind.CALLOUT_ARROW:
        return _callout_arrow(scene, spec, layout, decoration)
    if decoration.kind == DecorationKind.NUISANCE_TEXT:
        return _furniture_text(spec, layout, decoration, role="nuisance_text")
    if decoration.kind == DecorationKind.CAPTION:
        return _furniture_text(spec, layout, decoration, role="caption_text")
    if decoration.kind == DecorationKind.HANDDRAWN_MARK:
        return _handdrawn_mark(scene, spec, layout, decoration, rng)
    if decoration.kind == DecorationKind.HORIZONTAL_GUIDE:
        return _horizontal_guide(scene, spec, layout, decoration)
    return ()


def _vertical_guide(
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
) -> tuple[Primitive, ...]:
    if decoration.anchor_tick is None:
        return ()
    x = _tick_to_x(layout, decoration.anchor_tick)
    return (
        Line(
            role="vertical_helper_line",
            z=90,
            p0=Point(x, _lane_band_top(layout)),
            p1=Point(x, _lane_band_bottom(scene, layout)),
            stroke=_decoration_stroke(spec, decoration, dasharray=(4.0, 3.0), opacity=0.62),
        ),
    )


def _horizontal_guide(
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
) -> tuple[Primitive, ...]:
    lanes = decoration.lane_names or tuple(lane.name for lane in scene.lanes)
    primitives: list[Primitive] = []
    for lane_name in lanes:
        lane_box = _lane_bbox(scene, layout, lane_name)
        if lane_box is None:
            continue
        y = lane_box.y + lane_box.height / 2.0
        primitives.append(
            Line(
                role="horizontal_helper_line",
                z=91,
                p0=Point(layout.plot_origin.x, y),
                p1=Point(_plot_right(scene, layout), y),
                stroke=_decoration_stroke(spec, decoration, dasharray=(3.0, 3.0), opacity=0.45),
            )
        )
    return tuple(primitives)


def _measurement_bracket(
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
) -> tuple[Primitive, ...]:
    del scene
    if decoration.span is None:
        return ()

    x0 = _tick_to_x(layout, decoration.span[0])
    x1 = _tick_to_x(layout, decoration.span[1])
    if x1 < x0:
        x0, x1 = x1, x0
    y = max(8.0, layout.plot_origin.y - spec.style.primary_font.size_px * 0.9)
    notch = 5.0
    path = Path(
        role="measurement_bracket",
        z=94,
        d=f"M {_fmt(x0)} {_fmt(y + notch)} L {_fmt(x0)} {_fmt(y)} L {_fmt(x1)} {_fmt(y)} "
        f"L {_fmt(x1)} {_fmt(y + notch)}",
        stroke=_decoration_stroke(spec, decoration, opacity=0.85),
        fill=None,
    )
    primitives: list[Primitive] = [path]
    policy = AnnotationPolicy(spec.annotations.policy)
    if (
        policy in {AnnotationPolicy.NATURAL_MEASUREMENTS, AnnotationPolicy.DEBUG_LEAKY}
        and decoration.text
        and decoration.visibility_class == VisibilityClass.VISIBLE_TEXT
    ):
        primitives.append(
            Text(
                role="measurement_bracket",
                z=95,
                text=decoration.text,
                anchor=Point((x0 + x1) / 2.0, y - 6.0),
                font=_decoration_font(spec, decoration),
                text_anchor="middle",
                visibility_class=VisibilityClass.VISIBLE_TEXT.value,
            )
        )
    return tuple(primitives)


def _highlight_region(
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
) -> tuple[Primitive, ...]:
    if decoration.span is None:
        return ()
    x0 = _tick_to_x(layout, decoration.span[0])
    x1 = _tick_to_x(layout, decoration.span[1] + 1)
    if x1 < x0:
        x0, x1 = x1, x0
    fill = decoration.style.fill or Fill(color=_palette_color(spec, 12, "#dff2e2"), opacity=0.22)
    lanes = decoration.lane_names or tuple(lane.name for lane in scene.lanes)
    primitives: list[Primitive] = []
    for lane_name in lanes:
        lane_box = _lane_bbox(scene, layout, lane_name)
        if lane_box is None:
            continue
        primitives.append(
            Rect(
                role="hold_highlight",
                z=45,
                bbox=BBox(x=x0, y=lane_box.y + 2.0, width=max(0.0, x1 - x0), height=max(1.0, lane_box.height - 4.0)),
                stroke=decoration.style.stroke,
                fill=fill,
                radius=0.0,
            )
        )
    return tuple(primitives)


def _callout_arrow(
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
) -> tuple[Primitive, ...]:
    policy = AnnotationPolicy(spec.annotations.policy)
    if policy not in {AnnotationPolicy.NATURAL_MEASUREMENTS, AnnotationPolicy.DEBUG_LEAKY}:
        return ()
    anchor_tick = decoration.anchor_tick
    if anchor_tick is None and decoration.span is not None:
        anchor_tick = decoration.span[0]
    if anchor_tick is None:
        return ()

    x = _tick_to_x(layout, anchor_tick)
    target_y = _lane_band_top(layout) + layout.lane_height * 0.35
    text_y = max(10.0, _lane_band_top(layout) - spec.style.primary_font.size_px * 0.75)
    text_x = min(_plot_right(scene, layout) - 6.0, x + layout.tick_width * 0.85)
    path = Path(
        role="annotation_arrow",
        z=96,
        d=f"M {_fmt(text_x)} {_fmt(text_y + 4.0)} L {_fmt(x)} {_fmt(target_y)}",
        stroke=_decoration_stroke(spec, decoration, opacity=0.78),
        fill=None,
    )
    primitives: list[Primitive] = [path]
    if decoration.text and decoration.visibility_class == VisibilityClass.VISIBLE_TEXT:
        primitives.append(
            Text(
                role="measurement_bracket",
                z=97,
                text=decoration.text,
                anchor=Point(text_x, text_y),
                font=_decoration_font(spec, decoration),
                text_anchor="middle",
                visibility_class=VisibilityClass.VISIBLE_TEXT.value,
            )
        )
    return tuple(primitives)


def _furniture_text(
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
    *,
    role: str,
) -> tuple[Primitive, ...]:
    if not decoration.text:
        return ()
    above = role == "nuisance_text"
    y = max(10.0, layout.plot_origin.y - 7.0) if above else min(layout.height - 4.0, _plot_bottom(layout) + 16.0)
    return (
        Text(
            role=role,
            z=120 if role == "nuisance_text" else 125,
            text=decoration.text,
            anchor=Point(layout.plot_origin.x, y),
            font=_decoration_font(spec, decoration, fallback_color=_palette_color(spec, 10, "#777777")),
            visibility_class=VisibilityClass.HIDDEN_SEMANTIC.value,
        ),
    )


def _handdrawn_mark(
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
    decoration: Decoration,
    rng: random.Random,
) -> tuple[Primitive, ...]:
    x0, x1 = _decoration_span_x(layout, decoration)
    y0 = _lane_band_top(layout) + layout.lane_height * 0.18
    y1 = min(_lane_band_bottom(scene, layout) - 2.0, y0 + layout.lane_height * 1.4)
    jitter = 4.0 if decoration.style.handdrawn else 0.0

    def j(value: float) -> float:
        return value + rng.uniform(-jitter, jitter)

    d = (
        f"M {_fmt(j(x0))} {_fmt(j((y0 + y1) / 2.0))} "
        f"C {_fmt(j(x0))} {_fmt(j(y0))}, {_fmt(j(x1))} {_fmt(j(y0))}, {_fmt(j(x1))} {_fmt(j((y0 + y1) / 2.0))} "
        f"C {_fmt(j(x1))} {_fmt(j(y1))}, {_fmt(j(x0))} {_fmt(j(y1))}, {_fmt(j(x0))} {_fmt(j((y0 + y1) / 2.0))}"
    )
    return (
        Path(
            role="nuisance_text" if not decoration.semantic else "annotation_arrow",
            z=118,
            d=d,
            stroke=_decoration_stroke(spec, decoration, dasharray=decoration.style.stroke.dasharray if decoration.style.stroke else ()),
            fill=None,
        ),
    )


def _target_ref_exists(scene: TimingScene, decoration: Decoration) -> bool:
    if decoration.target_ref is None:
        return True
    kind, _, name = decoration.target_ref.partition(":")
    if not kind or not name:
        return False
    if kind == "anchor":
        return name in {event.name for event in scene.events}
    if kind == "constraint":
        return name in {constraint.name for constraint in scene.constraints}
    if kind == "window":
        return scene.visible_target is None or name in scene.visible_target.window_map
    return False


def _decoration_stroke(
    spec: RenderSpec,
    decoration: Decoration,
    *,
    dasharray: tuple[float, ...] = (),
    opacity: float = 0.72,
) -> Stroke:
    if decoration.style.stroke is not None:
        return decoration.style.stroke
    return Stroke(
        color=_palette_color(spec, 9, "#666666"),
        width=max(0.7, spec.style.waveform_stroke.width * 0.65),
        dasharray=dasharray,
        opacity=opacity,
    )


def _decoration_font(
    spec: RenderSpec,
    decoration: Decoration,
    *,
    fallback_color: str | None = None,
) -> FontSpec:
    if decoration.style.font is not None:
        return decoration.style.font
    return FontSpec(
        family=spec.style.primary_font.family,
        size_px=max(8.0, spec.style.primary_font.size_px * 0.82),
        weight=spec.style.primary_font.weight,
        color=fallback_color or _palette_color(spec, 9, "#666666"),
    )


def _decoration_span_x(layout: DiagramLayout, decoration: Decoration) -> tuple[float, float]:
    if decoration.span is not None:
        x0 = _tick_to_x(layout, decoration.span[0])
        x1 = _tick_to_x(layout, decoration.span[1] + 1)
        if x1 < x0:
            x0, x1 = x1, x0
        return x0, x1
    if decoration.anchor_tick is not None:
        x = _tick_to_x(layout, decoration.anchor_tick)
        return x - layout.tick_width * 0.35, x + layout.tick_width * 0.35
    return layout.plot_origin.x, min(_plot_right_from_layout(layout), layout.plot_origin.x + layout.tick_width)


def _lane_bbox(scene: TimingScene, layout: DiagramLayout, lane_name: str) -> BBox | None:
    for lane_index, lane in enumerate(scene.lanes):
        if lane.name == lane_name:
            return BBox(
                x=layout.plot_origin.x,
                y=layout.plot_origin.y + lane_index * layout.lane_pitch,
                width=max(0.0, scene.ticks.total_ticks * layout.tick_width),
                height=layout.lane_height,
            )
    return None


def _tick_to_x(layout: DiagramLayout, tick: float) -> float:
    return layout.plot_origin.x + tick * layout.tick_width


def _lane_band_top(layout: DiagramLayout) -> float:
    return layout.plot_origin.y


def _lane_band_bottom(scene: TimingScene, layout: DiagramLayout) -> float:
    if not scene.lanes:
        return layout.plot_origin.y
    return layout.plot_origin.y + (len(scene.lanes) - 1) * layout.lane_pitch + layout.lane_height


def _plot_right(scene: TimingScene, layout: DiagramLayout) -> float:
    return layout.plot_origin.x + max(1, scene.ticks.total_ticks) * layout.tick_width


def _plot_right_from_layout(layout: DiagramLayout) -> float:
    return layout.width - max(0.0, layout.plot_origin.x)


def _plot_bottom(layout: DiagramLayout) -> float:
    return layout.plot_origin.y + layout.lane_height


def _palette_color(spec: RenderSpec, index: int, fallback: str) -> str:
    return spec.style.palette[index] if index < len(spec.style.palette) else fallback


def _fmt(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"
