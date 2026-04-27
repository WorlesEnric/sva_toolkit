"""WaveDrom adapter for render2 TimingScene objects."""

from __future__ import annotations

from dataclasses import replace
import math
import re
from collections.abc import Iterable
from xml.etree import ElementTree as ET

from sva_toolkit.timing.core.scenario import ClockingSpec, ScenarioDocument
from sva_toolkit.timing.projection.wavedrom_view import (
    AnchorOccurrence,
    ResponseSpan,
    SummaryRule,
    WaveDromScenarioView,
    WaveLaneView,
    build_wavedrom_view,
)
from sva_toolkit.timing.render.wavedrom import _encode_bit_wave, _encode_bus_wave
from sva_toolkit.timing.render2.decoration_layer import render_decorations
from sva_toolkit.timing.render2.decorations import AnnotationPolicy
from sva_toolkit.timing.render2.legacy.debug_overlays import LegacyOverlayLayout, attach_legacy_overlays
from sva_toolkit.timing.render2.native.drawing import append_primitive, primitive_bbox
from sva_toolkit.timing.render2.native.style_kernels import contrast_ratio
from sva_toolkit.timing.render2.native.text_metrics import estimate_text_bbox
from sva_toolkit.timing.render2.primitives import BBox, FontSpec, Point, Primitive
from sva_toolkit.timing.render2.result import DiagramLayout, RenderResult, TextPrimitive, VisualVisibilityReport
from sva_toolkit.timing.render2.scene import LaneScene, LaneType, TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.visual import VisibilityClass


SVG_NS = "http://www.w3.org/2000/svg"
_TRANSLATE_RE = re.compile(r"translate\(\s*([+-]?\d+(?:\.\d+)?)(?:[,\s]+([+-]?\d+(?:\.\d+)?))?\s*\)")

ET.register_namespace("", SVG_NS)


class WaveDromAdapter:
    id = "wavedrom"
    capabilities = frozenset(
        {
            "bit",
            "bus",
            "clock",
            "cuts",
            "unknown",
            "high_z",
            "annotations",
            "wavejson_subset",
            "vector_text",
        }
    )

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool:
        return spec.renderer_id == self.id and all(lane.lane_type != LaneType.ANALOG for lane in scene.lanes)

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult:
        renderer_cls = _require_wavedrom()
        source = _build_wavedrom_source(scene)
        renderer = renderer_cls()
        drawing = renderer.render_waveform(0, source)
        root = ET.fromstring(drawing.tostring())
        _stash_renderer_layout(root, renderer)

        wave_layout = _extract_wave_layout(root, scene)
        layout = _diagram_layout_from_wave_layout(wave_layout)
        debug_enabled = spec.profile == "debug-current" or spec.annotations.policy == AnnotationPolicy.DEBUG_LEAKY

        if debug_enabled:
            debug_view = _debug_view(scene)
            debug_result = attach_legacy_overlays(root, debug_view, wave_layout)
            wave_layout = debug_result.layout
            layout = _diagram_layout_from_wave_layout(
                replace(
                    wave_layout,
                    plot_origin_y=wave_layout.plot_origin_y + debug_result.overlay_height,
                )
            )

        decoration_primitives = render_decorations(scene, spec, layout)
        if decoration_primitives:
            overlay_group = ET.Element(_svg("g"), {"id": "render2-decoration-layer"})
            for primitive in sorted(decoration_primitives, key=lambda item: item.z):
                append_primitive(overlay_group, primitive, antialias=spec.raster.antialias)
            root.append(overlay_group)

        root.set("width", root.attrib.get("width", _fmt(layout.width)))
        root.set("height", root.attrib.get("height", _fmt(layout.height)))
        if "viewBox" not in root.attrib:
            root.set("viewBox", f"0 0 {root.attrib['width']} {root.attrib['height']}")

        svg_text = ET.tostring(root, encoding="unicode", short_empty_elements=True)
        rendered_text = _collect_text(root, scene, spec)
        bbox_by_role = _bbox_by_role(rendered_text, decoration_primitives)
        final_layout = DiagramLayout(
            width=_to_float(root.attrib.get("width", layout.width)),
            height=_to_float(root.attrib.get("height", layout.height)),
            plot_origin=layout.plot_origin,
            tick_width=layout.tick_width,
            lane_height=layout.lane_height,
            lane_pitch=layout.lane_pitch,
            bbox_by_role=bbox_by_role,
        )
        visibility = _visibility_report(scene, spec, rendered_text)

        return RenderResult(
            svg_text=svg_text,
            png_bytes=None,
            layout=final_layout,
            visibility=visibility,
            render_spec=spec,
            warnings=(),
        )


def _build_wavedrom_source(scene: TimingScene) -> dict:
    return {
        "signal": [_build_lane_signal(lane, scene.ticks.total_ticks) for lane in scene.lanes],
        "head": {"tick": 0},
        "config": {"hscale": _compute_hscale(scene)},
    }


def _build_lane_signal(lane: LaneScene, total_ticks: int) -> dict:
    samples = _samples_from_runs(lane, total_ticks)
    signal = {"name": _lane_display_name(lane)}
    if lane.lane_type in {LaneType.BIT, LaneType.CLOCK, LaneType.UNKNOWN, LaneType.HIGH_Z}:
        signal["wave"] = _encode_bit_wave(samples)
        return signal

    wave, data = _encode_bus_wave(samples)
    signal["wave"] = wave
    if data:
        signal["data"] = data
    return signal


def _samples_from_runs(lane: LaneScene, total_ticks: int) -> tuple[str, ...]:
    tick_count = max(1, total_ticks)
    samples = ["x"] * tick_count
    for run in lane.runs:
        for tick in range(max(0, run.start_tick), min(tick_count - 1, run.end_tick) + 1):
            samples[tick] = run.value
    return tuple(samples)


def _compute_hscale(scene: TimingScene) -> float:
    widest_value = 0.0
    for lane in scene.lanes:
        if lane.lane_type != LaneType.BUS:
            continue
        for run in lane.runs:
            if run.value.lower() in {"x", "z"}:
                continue
            widest_value = max(widest_value, _text_width(run.value, 11))

    target_tick_width = max(56.0, min(100.0, widest_value + 24.0))
    hscale = max(1, min(4, int(math.ceil(target_tick_width / 40.0))))
    if scene.ticks.total_ticks >= 18 and widest_value <= 18.0:
        return 1
    return max(hscale, 2 if scene.ticks.total_ticks <= 14 else 1)


def _stash_renderer_layout(svg_tree: ET.Element, renderer) -> None:
    svg_tree.set("data-timing-xg", _fmt(float(renderer.lane.xg)))
    svg_tree.set("data-timing-yh0", _fmt(float(renderer.lane.yh0)))
    svg_tree.set("data-timing-yh1", _fmt(float(renderer.lane.yh1)))
    svg_tree.set("data-timing-xs", _fmt(float(renderer.lane.xs)))
    svg_tree.set("data-timing-ys", _fmt(float(renderer.lane.ys)))
    svg_tree.set("data-timing-yo", _fmt(float(renderer.lane.yo)))
    svg_tree.set("data-timing-hscale", _fmt(float(renderer.lane.hscale)))


def _extract_wave_layout(svg_tree: ET.Element, scene: TimingScene) -> LegacyOverlayLayout:
    xg = float(svg_tree.attrib.pop("data-timing-xg"))
    yh0 = float(svg_tree.attrib.pop("data-timing-yh0"))
    yh1 = float(svg_tree.attrib.pop("data-timing-yh1"))
    xs = float(svg_tree.attrib.pop("data-timing-xs"))
    ys = float(svg_tree.attrib.pop("data-timing-ys"))
    yo = float(svg_tree.attrib.pop("data-timing-yo"))
    hscale = float(svg_tree.attrib.pop("data-timing-hscale"))
    return LegacyOverlayLayout(
        width=_to_float(svg_tree.attrib["width"]),
        height=_to_float(svg_tree.attrib["height"]),
        plot_origin_x=xg + 0.5,
        plot_origin_y=yh0 + yh1 + 0.5,
        tick_width=2.0 * xs * hscale,
        lane_height=ys,
        lane_pitch=yo,
        lane_count=len(scene.lanes),
    )


def _diagram_layout_from_wave_layout(layout: LegacyOverlayLayout) -> DiagramLayout:
    return DiagramLayout(
        width=layout.width,
        height=layout.height,
        plot_origin=Point(layout.plot_origin_x, layout.plot_origin_y),
        tick_width=layout.tick_width,
        lane_height=layout.lane_height,
        lane_pitch=layout.lane_pitch,
    )


def _debug_view(scene: TimingScene) -> WaveDromScenarioView:
    if scene.semantic_document is not None:
        return _augment_debug_view_with_windows(build_wavedrom_view(scene.semantic_document), scene.semantic_document)
    if scene.visible_target is not None:
        return _augment_debug_view_with_windows(build_wavedrom_view(scene.visible_target), scene.visible_target)
    return WaveDromScenarioView(
        name=scene.name,
        clocking=ClockingSpec(edge=scene.clocking_edge, signal=scene.clocking_signal),
        ticks=scene.ticks.total_ticks,
        lanes=tuple(
            WaveLaneView(
                name=lane.name,
                kind="bus" if lane.lane_type == LaneType.BUS else "bit",
                samples=_samples_from_runs(lane, scene.ticks.total_ticks),
                width=lane.width_bits,
            )
            for lane in scene.lanes
        ),
        anchor_occurrences=tuple(
            AnchorOccurrence(anchor_name=event.name, tick=event.tick, placement="boundary", label=event.name)
            for event in scene.events
        ),
        response_spans=(),
        hold_spans=(),
        summary_rules=(),
    )


def _augment_debug_view_with_windows(view: WaveDromScenarioView, document: ScenarioDocument) -> WaveDromScenarioView:
    if view.summary_rules or view.response_spans:
        return view

    occurrence_by_anchor: dict[str, list[AnchorOccurrence]] = {}
    for occurrence in view.anchor_occurrences:
        occurrence_by_anchor.setdefault(occurrence.anchor_name, []).append(occurrence)

    response_spans: list[ResponseSpan] = []
    summary_rules: list[SummaryRule] = []
    for window in document.windows:
        summary_rules.append(
            SummaryRule(
                category="response",
                name=window.name,
                description=f"{window.start_anchor} -> after {window.bound.label} {window.end_anchor}",
            )
        )
        start_occurrence, end_occurrence = _first_matching_pair(
            occurrence_by_anchor.get(window.start_anchor, ()),
            occurrence_by_anchor.get(window.end_anchor, ()),
        )
        if start_occurrence is None or end_occurrence is None:
            continue
        response_spans.append(
            ResponseSpan(
                name=window.name,
                trigger_tick=start_occurrence.tick,
                response_tick=end_occurrence.tick,
                label=window.name,
                delay_text=window.bound.label,
            )
        )
    return replace(view, response_spans=tuple(response_spans), summary_rules=tuple(summary_rules))


def _first_matching_pair(
    starts: Iterable[AnchorOccurrence],
    ends: Iterable[AnchorOccurrence],
) -> tuple[AnchorOccurrence | None, AnchorOccurrence | None]:
    start_items = tuple(sorted(starts, key=lambda item: item.tick))
    end_items = tuple(sorted(ends, key=lambda item: item.tick))
    for start in start_items:
        for end in end_items:
            if end.tick >= start.tick:
                return start, end
    return (start_items[0], end_items[0]) if start_items and end_items else (None, None)


def _collect_text(root: ET.Element, scene: TimingScene, spec: RenderSpec) -> tuple[TextPrimitive, ...]:
    text_primitives: list[TextPrimitive] = []
    lane_labels = {_lane_display_name(lane): lane.name for lane in scene.lanes}
    lane_labels.update({lane.name: lane.name for lane in scene.lanes})
    bus_values = {
        str(run.value)
        for lane in scene.lanes
        if lane.lane_type == LaneType.BUS
        for run in lane.runs
        if run.value.lower() not in {"x", "z"}
    }

    def walk(element: ET.Element, tx: float, ty: float) -> None:
        local_tx, local_ty = _translate(element.attrib.get("transform", ""))
        tx += local_tx
        ty += local_ty
        if element.tag == _svg("text"):
            text = "".join(element.itertext()).strip()
            if text:
                role, visibility = _classify_text(element, text, lane_labels, bus_values, spec)
                x = _to_float(element.attrib.get("x", 0.0)) + tx
                y = _to_float(element.attrib.get("y", 0.0)) + ty
                font = _font_from_element(element, spec, role)
                bbox = estimate_text_bbox(
                    text,
                    font,
                    Point(x, y),
                    text_anchor=element.attrib.get("text-anchor", "start"),
                )
                text_primitives.append(TextPrimitive(text=text, bbox=bbox, role=role, visibility_class=visibility))
        for child in element:
            walk(child, tx, ty)

    walk(root, 0.0, 0.0)
    return tuple(text_primitives)


def _classify_text(
    element: ET.Element,
    text: str,
    lane_labels: dict[str, str],
    bus_values: set[str],
    spec: RenderSpec,
) -> tuple[str, str]:
    del spec
    css_class = set(element.attrib.get("class", "").split())
    data_role = element.attrib.get("data-role")
    if data_role == "debug_overlay" or css_class & {
        "timing-event-label",
        "timing-rule-label",
        "timing-summary-heading",
        "timing-summary-line",
    }:
        return "debug_overlay", VisibilityClass.DEBUG_OVERLAY.value
    if data_role in {"measurement_bracket", "annotation_arrow"}:
        return "measurement_bracket", VisibilityClass.VISIBLE_TEXT.value
    if data_role in {"nuisance_text", "caption_text"}:
        return data_role, VisibilityClass.HIDDEN_SEMANTIC.value
    if "muted" in css_class and text.isdigit():
        return "tick_label", VisibilityClass.VISIBLE_GEOMETRY.value
    if text in lane_labels or "info" in css_class:
        return "lane_label", VisibilityClass.VISIBLE_TEXT.value
    if text in bus_values:
        return "bus_value_text", VisibilityClass.VISIBLE_TEXT.value
    return "nuisance_text", VisibilityClass.HIDDEN_SEMANTIC.value


def _font_from_element(element: ET.Element, spec: RenderSpec, role: str) -> FontSpec:
    size = _font_size(element.attrib.get("font-size"))
    if size is None:
        size = spec.style.label_font.size_px if role == "lane_label" else spec.style.primary_font.size_px
    color = element.attrib.get("fill") or (spec.style.label_font.color if role == "lane_label" else spec.style.primary_font.color)
    return FontSpec(
        family=element.attrib.get("font-family", spec.style.primary_font.family),
        size_px=size,
        weight=element.attrib.get("font-weight", "400"),
        color=color,
    )


def _visibility_report(
    scene: TimingScene,
    spec: RenderSpec,
    rendered_text: tuple[TextPrimitive, ...],
) -> VisualVisibilityReport:
    target_visible: set[str] = set()
    nuisance: set[str] = set()
    debug: set[str] = set()
    lane_by_display = {_lane_display_name(lane): lane.name for lane in scene.lanes}
    lane_by_display.update({lane.name: lane.name for lane in scene.lanes})
    bus_values = {
        str(run.value)
        for lane in scene.lanes
        if lane.lane_type == LaneType.BUS
        for run in lane.runs
        if run.value.lower() not in {"x", "z"}
    }

    for text in rendered_text:
        if text.role == "lane_label":
            target_visible.add(lane_by_display.get(text.text, text.text))
        elif text.role == "bus_value_text" and text.text in bus_values:
            target_visible.add(text.text)
        elif text.role in {"nuisance_text", "caption_text"}:
            nuisance.add(text.text)
        elif text.role == "debug_overlay":
            debug.add(text.text)

    return VisualVisibilityReport(
        rendered_text=rendered_text,
        target_tokens_visible=frozenset(target_visible),
        nuisance_tokens=frozenset(nuisance),
        debug_overlay_tokens=frozenset(debug),
        leaked_tokens=frozenset(),
        occluded_lane_fractions={lane.name: 0.0 for lane in scene.lanes},
        minimum_contrast=_normalized_contrast("#000000", "#ffffff"),
    )


def _bbox_by_role(
    rendered_text: Iterable[TextPrimitive],
    decoration_primitives: Iterable[Primitive],
) -> dict[str, tuple[BBox, ...]]:
    boxes: dict[str, list[BBox]] = {}
    for text in rendered_text:
        boxes.setdefault(text.role, []).append(text.bbox)
    for primitive in decoration_primitives:
        box = primitive_bbox(primitive)
        if box is not None:
            boxes.setdefault(primitive.role, []).append(box)
    return {role: tuple(role_boxes) for role, role_boxes in boxes.items()}


def _lane_display_name(lane: LaneScene) -> str:
    if lane.width_bits and not (lane.lane_type == LaneType.BIT and lane.width_bits == "1"):
        return f"{lane.name}[{lane.width_bits}]"
    return lane.name


def _require_wavedrom():
    try:
        from wavedrom.waveform import WaveDrom
    except ImportError as exc:
        raise RuntimeError("wavedrom is required for the render2 WaveDromAdapter") from exc
    return WaveDrom


def _text_width(text: str, size: int) -> float:
    try:
        from wavedrom.waveform import WaveDrom
    except ImportError:
        return len(text) * size * 0.62
    return float(WaveDrom().text_width(text, size))


def _translate(transform: str) -> tuple[float, float]:
    match = _TRANSLATE_RE.search(transform)
    if match is None:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2) or 0.0)


def _font_size(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace("px", ""))
    except ValueError:
        return None


def _normalized_contrast(foreground: str, background: str) -> float:
    return max(0.0, min(1.0, (contrast_ratio(foreground, background) - 1.0) / 20.0))


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
