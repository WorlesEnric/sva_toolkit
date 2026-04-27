"""Self-contained native SVG renderer for render2 TimingScene objects."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from sva_toolkit.timing.render2.decorations import AnnotationPolicy
from sva_toolkit.timing.render2.primitives import BBox, Fill, FontSpec, Line, Point, Primitive, Rect, Stroke, Text
from sva_toolkit.timing.render2.result import DiagramLayout, RenderResult, TextPrimitive, VisualVisibilityReport
from sva_toolkit.timing.render2.scene import LaneScene, LaneType, TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.visual import VisibilityClass

from sva_toolkit.timing.render2.native.decorations import emit_decorations
from sva_toolkit.timing.render2.native.drawing import primitive_bbox, render_svg
from sva_toolkit.timing.render2.native.geometry import NativeGeometry, compute_geometry
from sva_toolkit.timing.render2.native.style_kernels import contrast_ratio
from sva_toolkit.timing.render2.native.waves import emit_cut_markers, emit_lane_wave


class NativeSvgRenderer:
    id = "native_svg"
    capabilities = frozenset(
        {
            "bit",
            "bus",
            "clock",
            "cuts",
            "unknown",
            "high_z",
            "annotations",
            "style_overrides",
            "measurement_brackets",
            "page_composition",
            "vector_text",
            "ascii_only",
        }
    )

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool:
        if spec.renderer_id != self.id:
            return False
        return all(lane.lane_type != LaneType.ANALOG for lane in scene.lanes)

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult:
        geometry = compute_geometry(scene, spec.layout, spec.page, spec.style.label_font)
        primitives: list[Primitive] = []
        warnings: list[str] = []

        primitives.extend(_background_primitives(geometry, spec))
        primitives.extend(_lane_primitives(scene.lanes, geometry, spec))
        primitives.extend(_grid_primitives(geometry, spec))
        primitives.extend(_label_primitives(scene.lanes, geometry, spec))
        for lane_index, lane in enumerate(scene.lanes):
            primitives.extend(_as_primitives(emit_lane_wave(lane, lane_index, geometry, spec.style)))
        primitives.extend(_as_primitives(emit_cut_markers(scene.cuts, geometry, spec.style)))
        decoration_primitives, decoration_warnings = emit_decorations(scene, spec, geometry)
        primitives.extend(_as_primitives(decoration_primitives))
        warnings.extend(decoration_warnings)
        primitives.extend(_page_primitives(scene, geometry, spec))

        bbox_by_role = _bbox_by_role(primitives)
        layout = DiagramLayout(
            width=geometry.width,
            height=geometry.height,
            plot_origin=geometry.plot_origin,
            tick_width=spec.layout.tick_width,
            lane_height=spec.layout.lane_height,
            lane_pitch=spec.layout.lane_pitch,
            bbox_by_role=bbox_by_role,
        )
        visibility = _visibility_report(scene, spec, primitives, bbox_by_role, geometry)
        svg_text = render_svg(primitives, width=geometry.width, height=geometry.height, antialias=spec.raster.antialias)

        return RenderResult(
            svg_text=svg_text,
            png_bytes=None,
            layout=layout,
            visibility=visibility,
            render_spec=spec,
            warnings=tuple(warnings),
        )


def _background_primitives(geometry: NativeGeometry, spec: RenderSpec) -> tuple[Primitive, ...]:
    colors = _palette(spec)
    card_margin_x = max(4.0, spec.layout.margin.x * 0.45)
    card_top = max(2.0, geometry.plot_origin.y - 12.0)
    card_bottom = min(geometry.height - 2.0, geometry.plot_bottom + 12.0)
    return (
        Rect(
            role="background",
            z=0,
            bbox=BBox(x=0.0, y=0.0, width=geometry.width, height=geometry.height),
            stroke=None,
            fill=Fill(color=colors["background"]),
        ),
        Rect(
            role="outer_card",
            z=5,
            bbox=BBox(
                x=card_margin_x,
                y=card_top,
                width=max(1.0, geometry.width - card_margin_x * 2),
                height=max(1.0, card_bottom - card_top),
            ),
            stroke=Stroke(
                color=colors["grid"],
                width=_extra_float(spec.extras, "outer_stroke_width", 0.8),
                opacity=0.9,
            ),
            fill=Fill(color=colors["card"]),
            radius=_extra_float(spec.extras, "outer_radius", 4.0),
        ),
    )


def _lane_primitives(lanes: tuple[LaneScene, ...], geometry: NativeGeometry, spec: RenderSpec) -> tuple[Primitive, ...]:
    colors = _palette(spec)
    primitives: list[Primitive] = []
    lane_bands_enabled = spec.extras.get("lane_bands", "1") == "1"
    band_opacity = _extra_float(spec.extras, "lane_band_opacity", 0.08)
    for lane_index, _lane in enumerate(lanes):
        bbox = geometry.lane_bbox(lane_index)
        if lane_bands_enabled and lane_index % 2 == 1:
            primitives.append(
                Rect(
                    role="lane_separator",
                    z=12,
                    bbox=bbox,
                    stroke=None,
                    fill=Fill(color=colors["lane"], opacity=band_opacity),
                )
            )
        if lane_index > 0:
            y = geometry.lane_top(lane_index) - (geometry.layout.lane_pitch - geometry.layout.lane_height) / 2
            primitives.append(
                Line(
                    role="lane_separator",
                    z=16,
                    p0=Point(geometry.plot_origin.x, y),
                    p1=Point(geometry.plot_right, y),
                    stroke=Stroke(color=colors["grid"], width=0.55, opacity=0.45),
                )
            )
    return tuple(primitives)


def _grid_primitives(geometry: NativeGeometry, spec: RenderSpec) -> tuple[Primitive, ...]:
    mode = spec.style.grid_mode
    if mode == "none":
        return ()

    major_every = max(1, int(float(spec.extras.get("major_grid_every", "4"))))
    if mode == "sparse":
        ticks = tuple(range(0, geometry.total_ticks + 1, major_every))
        minor = ()
    elif mode == "major_only":
        ticks = tuple(range(0, geometry.total_ticks + 1, major_every))
        minor = ()
    elif mode == "major_minor":
        ticks = tuple(range(0, geometry.total_ticks + 1, major_every))
        minor = tuple(tick for tick in range(geometry.total_ticks + 1) if tick % major_every != 0)
    else:
        ticks = tuple(range(0, geometry.total_ticks + 1, max(1, major_every // 2)))
        minor = tuple(tick for tick in range(geometry.total_ticks + 1) if tick not in ticks)

    primitives: list[Primitive] = []
    minor_stroke = Stroke(
        color=spec.style.grid_stroke.color,
        width=max(0.35, spec.style.grid_stroke.width * 0.75),
        dasharray=spec.style.grid_stroke.dasharray,
        opacity=min(0.55, spec.style.grid_stroke.opacity),
    )
    for tick in minor:
        x = geometry.tick_to_x(tick)
        primitives.append(
            Line(
                role="grid_minor",
                z=20,
                p0=Point(x, geometry.plot_origin.y),
                p1=Point(x, geometry.plot_bottom),
                stroke=minor_stroke,
            )
        )
    for tick in ticks:
        x = geometry.tick_to_x(tick)
        primitives.append(
            Line(
                role="grid_major",
                z=22,
                p0=Point(x, geometry.plot_origin.y),
                p1=Point(x, geometry.plot_bottom),
                stroke=spec.style.grid_stroke,
            )
        )
    return tuple(primitives)


def _label_primitives(lanes: tuple[LaneScene, ...], geometry: NativeGeometry, spec: RenderSpec) -> tuple[Primitive, ...]:
    labels: list[Primitive] = []
    for lane_index, lane in enumerate(lanes):
        anchor, text_anchor = geometry.label_anchor(lane_index)
        labels.append(
            Text(
                role="lane_label",
                z=40,
                text=lane.name,
                anchor=anchor,
                font=spec.style.label_font,
                text_anchor=text_anchor,
                visibility_class=VisibilityClass.VISIBLE_TEXT.value,
            )
        )
    return tuple(labels)


def _page_primitives(scene: TimingScene, geometry: NativeGeometry, spec: RenderSpec) -> tuple[Primitive, ...]:
    if not spec.page.enabled:
        return ()

    colors = _palette(spec)
    font = FontSpec(
        family=spec.style.primary_font.family,
        size_px=max(9.0, spec.style.primary_font.size_px * 0.84),
        color=colors["nuisance"],
    )
    primitives: list[Primitive] = []
    x = spec.layout.margin.x
    cursor = max(font.size_px + 2.0, spec.layout.margin.y)
    if spec.page.table_border:
        primitives.append(
            Rect(
                role="page_table_border",
                z=128,
                bbox=BBox(
                    x=max(1.0, spec.layout.margin.x * 0.45),
                    y=max(1.0, spec.layout.margin.y * 0.45),
                    width=max(1.0, geometry.width - spec.layout.margin.x * 0.9),
                    height=max(1.0, geometry.height - spec.layout.margin.y * 0.9),
                ),
                stroke=Stroke(color=colors["grid"], width=0.75, opacity=0.75),
                fill=None,
            )
    )
    if spec.page.page_header:
        primitives.append(_page_text("page_header", f"REF-TMG-{abs(spec.seed) % 41 + 1:02d}", Point(x, cursor), font))
        cursor += font.size_px * 1.35
    if spec.page.surrounding_paragraph:
        primitives.append(_page_text("page_paragraph", f"NOTE-{_stable_index(scene.name, 83):02d}", Point(x, cursor), font))
        cursor += font.size_px * 1.35
    if spec.page.caption_above:
        primitives.append(_page_text("page_caption", f"FIG-TD-{abs(spec.seed) % 97 + 1:02d}", Point(x, cursor), font))

    below = geometry.plot_bottom + font.size_px * 1.4
    if spec.page.caption_below:
        primitives.append(_page_text("page_caption", f"TD-{_stable_index(scene.name, 29):02d}", Point(x, below), font))
        below += font.size_px * 1.35
    if spec.page.page_footer:
        footer_y = min(geometry.height - 4.0, max(below, geometry.height - spec.layout.margin.height * 0.5))
        primitives.append(_page_text("page_footer", f"PG-{abs(spec.seed) % 13 + 1:02d}", Point(x, footer_y), font))
    return tuple(primitives)


def _page_text(role: str, text: str, anchor: Point, font: FontSpec) -> Text:
    return Text(
        role=role,
        z=130,
        text=text,
        anchor=anchor,
        font=font,
        visibility_class=VisibilityClass.HIDDEN_SEMANTIC.value,
    )


def _visibility_report(
    scene: TimingScene,
    spec: RenderSpec,
    primitives: Iterable[Primitive],
    bbox_by_role: Mapping[str, tuple[BBox, ...]],
    geometry: NativeGeometry,
) -> VisualVisibilityReport:
    target_tokens = _canonical_target_tokens(scene)
    text_primitives: list[TextPrimitive] = []
    target_visible: set[str] = set()
    nuisance_tokens: set[str] = set()
    debug_tokens: set[str] = set()
    leaked_tokens: set[str] = set()

    for primitive in _flatten(primitives):
        if not isinstance(primitive, Text):
            continue
        bbox = primitive_bbox(primitive)
        if bbox is None:
            continue
        visibility_class = str(primitive.visibility_class)
        text_primitives.append(
            TextPrimitive(
                text=primitive.text,
                bbox=bbox,
                role=primitive.role,
                visibility_class=visibility_class,
            )
        )
        if primitive.role in {"nuisance_text", "caption_text"} or primitive.role.startswith("page_"):
            nuisance_tokens.add(primitive.text)
        if primitive.role == "debug_overlay" and spec.annotations.policy == AnnotationPolicy.DEBUG_LEAKY:
            debug_tokens.add(primitive.text)
        if (
            primitive.text in target_tokens
            and visibility_class in {VisibilityClass.VISIBLE_TEXT.value, VisibilityClass.VISIBLE_CONVENTION.value}
        ):
            target_visible.add(primitive.text)
        elif primitive.text in target_tokens and primitive.role != "debug_overlay":
            leaked_tokens.add(primitive.text)

    return VisualVisibilityReport(
        rendered_text=tuple(text_primitives),
        target_tokens_visible=frozenset(target_visible),
        nuisance_tokens=frozenset(nuisance_tokens),
        debug_overlay_tokens=frozenset(debug_tokens if spec.annotations.policy == AnnotationPolicy.DEBUG_LEAKY else ()),
        leaked_tokens=frozenset(leaked_tokens),
        occluded_lane_fractions=_occluded_lane_fractions(bbox_by_role, geometry),
        minimum_contrast=_normalized_contrast(spec.style.waveform_stroke.color, _palette(spec)["background"]),
    )


def _canonical_target_tokens(scene: TimingScene) -> frozenset[str]:
    tokens: set[str] = set()
    document = scene.visible_target
    if document is not None:
        tokens.add(document.name)
        tokens.add(document.clocking.signal)
        for signal in document.signals:
            tokens.add(signal.name)
            tokens.add(signal.display_name)
            tokens.update(str(sample) for sample in signal.samples if _printable_token(sample))
        for anchor in document.anchors:
            tokens.add(anchor.name)
        for window in document.windows:
            tokens.add(window.name)
            tokens.add(window.bound.label)
        for cut in document.cuts:
            tokens.add(cut.name)
            if cut.label:
                tokens.add(cut.label)
        for constraint in document.lane_constraints:
            tokens.add(constraint.name)
            if constraint.value:
                tokens.add(constraint.value)
        for parameter in document.params:
            tokens.add(parameter.name)
    else:
        tokens.update(lane.name for lane in scene.lanes)
        tokens.update(event.name for event in scene.events)

    for lane in scene.lanes:
        tokens.add(lane.name)
        for run in lane.runs:
            if lane.lane_type == LaneType.BUS and _printable_token(run.value):
                tokens.add(str(run.value))
    for decoration in scene.decorations:
        if decoration.text:
            tokens.add(decoration.text)
    return frozenset(token for token in tokens if token)


def _occluded_lane_fractions(
    bbox_by_role: Mapping[str, tuple[BBox, ...]],
    geometry: NativeGeometry,
) -> Mapping[str, float]:
    occluding_roles = {
        "nuisance_text",
        "caption_text",
        "page_caption",
        "page_paragraph",
        "page_header",
        "page_footer",
        "debug_overlay",
        "vertical_helper_line",
        "horizontal_helper_line",
        "annotation_arrow",
    }
    boxes = tuple(box for role in occluding_roles for box in bbox_by_role.get(role, ()))
    fractions: dict[str, float] = {}
    for lane_index, lane_name in enumerate(geometry.lane_names):
        lane = geometry.lane_bbox(lane_index)
        intervals: list[tuple[float, float]] = []
        for box in boxes:
            if not _overlaps_y(lane, box):
                continue
            left = max(lane.x, box.x)
            right = min(lane.x + lane.width, box.x + box.width)
            if right > left:
                intervals.append((left, right))
        fractions[lane_name] = min(1.0, _union_interval_width(intervals) / lane.width) if lane.width > 0 else 0.0
    return fractions


def _bbox_by_role(primitives: Iterable[Primitive]) -> Mapping[str, tuple[BBox, ...]]:
    boxes: defaultdict[str, list[BBox]] = defaultdict(list)
    for primitive in _flatten(primitives):
        bbox = primitive_bbox(primitive)
        if bbox is not None:
            boxes[primitive.role].append(bbox)
    return {role: tuple(role_boxes) for role, role_boxes in boxes.items()}


def _flatten(primitives: Iterable[Primitive]) -> Iterable[Primitive]:
    for primitive in primitives:
        yield primitive
        children = getattr(primitive, "children", ())
        if children:
            yield from _flatten(children)


def _as_primitives(items: Iterable[object]) -> tuple[Primitive, ...]:
    return tuple(item for item in items if isinstance(item, Primitive))


def _overlaps_y(left: BBox, right: BBox) -> bool:
    return left.y < right.y + right.height and right.y < left.y + left.height


def _union_interval_width(intervals: Iterable[tuple[float, float]]) -> float:
    sorted_intervals = sorted(intervals)
    if not sorted_intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted_intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _printable_token(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in {"x", "z", "?", "unknown", "highz"}


def _palette(spec: RenderSpec) -> dict[str, str]:
    palette = tuple(spec.style.palette)

    def color(index: int, fallback: str) -> str:
        return palette[index] if index < len(palette) else fallback

    return {
        "background": color(0, "#ffffff"),
        "card": color(1, "#ffffff"),
        "wave": color(2, "#000000"),
        "grid": color(3, "#d7d7d7"),
        "label": color(4, "#1f4e8c"),
        "lane": color(5, "#f6f8fb"),
        "nuisance": color(10, "#777777"),
    }


def _extra_float(extras: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(extras.get(key, str(default)))
    except ValueError:
        return default


def _stable_index(text: str, modulo: int) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % modulo + 1


def _normalized_contrast(foreground: str, background: str) -> float:
    return max(0.0, min(1.0, (contrast_ratio(foreground, background) - 1.0) / 20.0))
