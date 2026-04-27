"""Shared helpers for render2 external adapters."""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
import re
from xml.etree import ElementTree as ET

from sva_toolkit.timing.render2.decoration_layer import render_decorations
from sva_toolkit.timing.render2.native.drawing import SVG_NS, append_primitive, primitive_bbox
from sva_toolkit.timing.render2.native.geometry import compute_geometry
from sva_toolkit.timing.render2.native.text_metrics import estimate_text_bbox, estimate_text_width
from sva_toolkit.timing.render2.primitives import BBox, FontSpec, Point, Primitive
from sva_toolkit.timing.render2.protocol import _required_capabilities
from sva_toolkit.timing.render2.result import DiagramLayout, TextPrimitive, VisualVisibilityReport
from sva_toolkit.timing.render2.scene import LaneScene, LaneType, TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.visual import VisibilityClass


_TRANSLATE_RE = re.compile(r"translate\(\s*([+-]?\d+(?:\.\d+)?)(?:[,\s]+([+-]?\d+(?:\.\d+)?))?\s*\)")


def supports_scene(
    renderer_id: str,
    capabilities: frozenset[str],
    scene: TimingScene,
    spec: RenderSpec,
    *,
    dependency_available: bool = True,
) -> bool:
    if spec.renderer_id != renderer_id or not dependency_available:
        return False
    if any(lane.lane_type == LaneType.ANALOG for lane in scene.lanes):
        return False
    return _required_capabilities(scene, spec) <= capabilities


def samples_from_runs(lane: LaneScene, total_ticks: int) -> tuple[str, ...]:
    tick_count = max(1, total_ticks)
    samples = ["x"] * tick_count
    for run in lane.runs:
        for tick in range(max(0, run.start_tick), min(tick_count - 1, run.end_tick) + 1):
            samples[tick] = str(run.value)
    return tuple(samples)


def lane_display_name(lane: LaneScene) -> str:
    if lane.width_bits and not (lane.lane_type == LaneType.BIT and lane.width_bits == "1"):
        return f"{lane.name}[{lane.width_bits}]"
    return lane.name


def printable_bus_values(scene: TimingScene) -> frozenset[str]:
    return frozenset(
        str(run.value)
        for lane in scene.lanes
        if lane.lane_type == LaneType.BUS
        for run in lane.runs
        if _printable_token(run.value)
    )


def base_layout(scene: TimingScene, spec: RenderSpec, *, width: float | None = None, height: float | None = None) -> DiagramLayout:
    geometry = compute_geometry(scene, spec.layout, spec.page, spec.style.label_font)
    return DiagramLayout(
        width=geometry.width if width is None else width,
        height=geometry.height if height is None else height,
        plot_origin=geometry.plot_origin,
        tick_width=spec.layout.tick_width,
        lane_height=spec.layout.lane_height,
        lane_pitch=spec.layout.lane_pitch,
    )


def with_bbox_by_role(layout: DiagramLayout, bbox_by_role: dict[str, tuple[BBox, ...]]) -> DiagramLayout:
    return DiagramLayout(
        width=layout.width,
        height=layout.height,
        plot_origin=layout.plot_origin,
        tick_width=layout.tick_width,
        lane_height=layout.lane_height,
        lane_pitch=layout.lane_pitch,
        bbox_by_role=bbox_by_role,
    )


def add_decorations_to_svg(
    root: ET.Element,
    scene: TimingScene,
    spec: RenderSpec,
    layout: DiagramLayout,
) -> tuple[Primitive, ...]:
    primitives = render_decorations(scene, spec, layout)
    if not primitives:
        return ()
    overlay = ET.Element(f"{{{SVG_NS}}}g", {"id": "render2-decoration-layer"})
    for primitive in sorted(primitives, key=lambda item: item.z):
        append_primitive(overlay, primitive, antialias=spec.raster.antialias)
    root.append(overlay)
    return primitives


def collect_svg_text(root: ET.Element, scene: TimingScene, spec: RenderSpec) -> tuple[TextPrimitive, ...]:
    text_primitives: list[TextPrimitive] = []
    lane_labels = {lane_display_name(lane): lane.name for lane in scene.lanes}
    lane_labels.update({lane.name: lane.name for lane in scene.lanes})
    bus_values = printable_bus_values(scene)

    def walk(element: ET.Element, tx: float, ty: float) -> None:
        local_tx, local_ty = _translate(element.attrib.get("transform", ""))
        tx += local_tx
        ty += local_ty
        if _local_name(element.tag) == "text":
            text = "".join(element.itertext()).strip()
            if text:
                role, visibility = classify_text(element, text, lane_labels, bus_values)
                x = _to_float(element.attrib.get("x", 0.0)) + tx
                y = _to_float(element.attrib.get("y", 0.0)) + ty
                bbox = estimate_text_bbox(
                    text,
                    _font_from_svg_element(element, spec, role),
                    Point(x, y),
                    text_anchor=element.attrib.get("text-anchor", "start"),
                )
                text_primitives.append(TextPrimitive(text, bbox, role, visibility))
        for child in element:
            walk(child, tx, ty)

    walk(root, 0.0, 0.0)
    return tuple(text_primitives)


def classify_text(
    element: ET.Element,
    text: str,
    lane_labels: dict[str, str],
    bus_values: frozenset[str],
) -> tuple[str, str]:
    css_class = set(element.attrib.get("class", "").split())
    data_role = element.attrib.get("data-role")
    if data_role == "debug_overlay" or "debug_overlay" in css_class:
        return "debug_overlay", VisibilityClass.DEBUG_OVERLAY.value
    if data_role in {"measurement_bracket", "annotation_arrow"}:
        return "measurement_bracket", VisibilityClass.VISIBLE_TEXT.value
    if data_role in {"nuisance_text", "caption_text"}:
        return data_role, VisibilityClass.HIDDEN_SEMANTIC.value
    if text in lane_labels:
        return "lane_label", VisibilityClass.VISIBLE_TEXT.value
    if text in bus_values:
        return "bus_value_text", VisibilityClass.VISIBLE_TEXT.value
    return "nuisance_text", VisibilityClass.HIDDEN_SEMANTIC.value


def text_primitives_from_tokens(
    tokens: Iterable[tuple[str, str, str, Point]],
    spec: RenderSpec,
) -> tuple[TextPrimitive, ...]:
    primitives: list[TextPrimitive] = []
    for text, role, visibility, anchor in tokens:
        font = spec.style.label_font if role == "lane_label" else spec.style.primary_font
        primitives.append(TextPrimitive(text, estimate_text_bbox(text, font, anchor), role, visibility))
    return tuple(primitives)


def visibility_report(scene: TimingScene, rendered_text: Iterable[TextPrimitive]) -> VisualVisibilityReport:
    rendered = tuple(rendered_text)
    target_visible: set[str] = set()
    nuisance: set[str] = set()
    debug: set[str] = set()
    bus_values = printable_bus_values(scene)
    lane_by_display = {lane_display_name(lane): lane.name for lane in scene.lanes}
    lane_by_display.update({lane.name: lane.name for lane in scene.lanes})

    for text in rendered:
        if text.role == "lane_label":
            target_visible.add(lane_by_display.get(text.text, text.text))
        elif text.role == "bus_value_text" and text.text in bus_values:
            target_visible.add(text.text)
        elif text.role in {"nuisance_text", "caption_text"}:
            nuisance.add(text.text)
        elif text.role == "debug_overlay" or text.visibility_class == VisibilityClass.DEBUG_OVERLAY.value:
            debug.add(text.text)

    return VisualVisibilityReport(
        rendered_text=rendered,
        target_tokens_visible=frozenset(target_visible),
        nuisance_tokens=frozenset(nuisance),
        debug_overlay_tokens=frozenset(debug),
        leaked_tokens=frozenset(),
        occluded_lane_fractions={lane.name: 0.0 for lane in scene.lanes},
        minimum_contrast=1.0,
    )


def bbox_by_role(
    rendered_text: Iterable[TextPrimitive],
    decoration_primitives: Iterable[Primitive] = (),
) -> dict[str, tuple[BBox, ...]]:
    boxes: dict[str, list[BBox]] = {}
    for text in rendered_text:
        boxes.setdefault(text.role, []).append(text.bbox)
    for primitive in decoration_primitives:
        box = primitive_bbox(primitive)
        if box is not None:
            boxes.setdefault(primitive.role, []).append(box)
    return {role: tuple(role_boxes) for role, role_boxes in boxes.items()}


def svg_size(root: ET.Element, fallback: DiagramLayout) -> tuple[float, float]:
    return _to_float(root.attrib.get("width", fallback.width)), _to_float(root.attrib.get("height", fallback.height))


def render_ascii_png(ascii_text: str, spec: RenderSpec) -> bytes | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    lines = ascii_text.splitlines() or [""]
    font = ImageFont.load_default()
    char_width = max(1, int(estimate_text_width("M", spec.style.primary_font) * 0.72))
    line_height = max(14, int(spec.style.primary_font.size_px * 1.35))
    width = max(1, max(len(line) for line in lines) * char_width + 16)
    height = max(1, len(lines) * line_height + 16)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((8, 8 + index * line_height), line, fill="black", font=font)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _font_from_svg_element(element: ET.Element, spec: RenderSpec, role: str) -> FontSpec:
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


def _to_float(value: object) -> float:
    text = str(value).replace("px", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _printable_token(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in {"x", "z", "?", "unknown", "highz"}
