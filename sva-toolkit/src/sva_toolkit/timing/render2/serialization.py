"""JSON-stable serialization helpers for render2 dataclasses."""

from __future__ import annotations

import base64
from typing import Any, Mapping

from sva_toolkit.timing.render2.decorations import Decoration, DecorationKind, DecorationStyle
from sva_toolkit.timing.render2.primitives import BBox, Fill, FontSpec, Point, Stroke
from sva_toolkit.timing.render2.result import (
    DiagramLayout,
    RenderResult,
    TextPrimitive,
    VisualVisibilityReport,
)
from sva_toolkit.timing.render2.scene import (
    CutRegion,
    LaneScene,
    LaneType,
    SampleRun,
    TickModel,
    TimingScene,
    VisualConstraint,
    VisualEvent,
)
from sva_toolkit.timing.render2.spec import (
    AnnotationSpec,
    DegradationSpec,
    LayoutSpec,
    PageSpec,
    RasterSpec,
    RenderSpec,
    StyleSpec,
)
from sva_toolkit.timing.visual import VisibilityClass


def spec_to_dict(spec: RenderSpec) -> dict[str, Any]:
    return {
        "renderer_id": spec.renderer_id,
        "style": to_dict(spec.style),
        "layout": to_dict(spec.layout),
        "annotations": to_dict(spec.annotations),
        "page": to_dict(spec.page),
        "raster": to_dict(spec.raster),
        "degradation": to_dict(spec.degradation),
        "seed": spec.seed,
        "profile": spec.profile,
        "extras": dict(spec.extras),
    }


def spec_from_dict(data: Mapping[str, Any]) -> RenderSpec:
    return RenderSpec(
        renderer_id=str(data["renderer_id"]),
        style=style_spec_from_dict(data["style"]),
        layout=layout_spec_from_dict(data["layout"]),
        annotations=annotation_spec_from_dict(data["annotations"]),
        page=page_spec_from_dict(data["page"]),
        raster=raster_spec_from_dict(data["raster"]),
        degradation=degradation_spec_from_dict(data["degradation"]),
        seed=int(data["seed"]),
        profile=str(data["profile"]),
        extras={str(key): str(value) for key, value in data.get("extras", {}).items()},
    )


def result_to_dict(result: RenderResult) -> dict[str, Any]:
    return {
        "svg_text": result.svg_text,
        "png_bytes_b64": base64.b64encode(result.png_bytes).decode("ascii") if result.png_bytes is not None else None,
        "ascii_text": result.ascii_text,
        "layout": diagram_layout_to_dict(result.layout),
        "visibility": visibility_report_to_dict(result.visibility),
        "render_spec": spec_to_dict(result.render_spec),
        "warnings": list(result.warnings),
    }


def result_from_dict(data: Mapping[str, Any]) -> RenderResult:
    png_text = data.get("png_bytes_b64")
    return RenderResult(
        svg_text=data.get("svg_text"),
        png_bytes=base64.b64decode(png_text) if png_text is not None else None,
        layout=diagram_layout_from_dict(data["layout"]),
        visibility=visibility_report_from_dict(data["visibility"]),
        render_spec=spec_from_dict(data["render_spec"]),
        warnings=tuple(data.get("warnings", ())),
        ascii_text=data.get("ascii_text"),
    )


def scene_to_dict(scene: TimingScene) -> dict[str, Any]:
    return {
        "name": scene.name,
        "clocking_edge": scene.clocking_edge,
        "clocking_signal": scene.clocking_signal,
        "lanes": [lane_scene_to_dict(lane) for lane in scene.lanes],
        "ticks": tick_model_to_dict(scene.ticks),
        "cuts": [cut_region_to_dict(cut) for cut in scene.cuts],
        "events": [visual_event_to_dict(event) for event in scene.events],
        "constraints": [visual_constraint_to_dict(constraint) for constraint in scene.constraints],
        "decorations": [decoration_to_dict(decoration) for decoration in scene.decorations],
    }


def scene_from_dict(data: Mapping[str, Any]) -> TimingScene:
    return TimingScene(
        name=str(data["name"]),
        clocking_edge=str(data["clocking_edge"]),
        clocking_signal=str(data["clocking_signal"]),
        lanes=tuple(lane_scene_from_dict(item) for item in data.get("lanes", ())),
        ticks=tick_model_from_dict(data["ticks"]),
        cuts=tuple(cut_region_from_dict(item) for item in data.get("cuts", ())),
        events=tuple(visual_event_from_dict(item) for item in data.get("events", ())),
        constraints=tuple(visual_constraint_from_dict(item) for item in data.get("constraints", ())),
        decorations=tuple(decoration_from_dict(item) for item in data.get("decorations", ())),
        visible_target=None,
        semantic_document=None,
    )


def to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, RenderSpec):
        return spec_to_dict(obj)
    if isinstance(obj, RenderResult):
        return result_to_dict(obj)
    if isinstance(obj, TimingScene):
        return scene_to_dict(obj)
    if isinstance(obj, StyleSpec):
        return style_spec_to_dict(obj)
    if isinstance(obj, LayoutSpec):
        return layout_spec_to_dict(obj)
    if isinstance(obj, AnnotationSpec):
        return annotation_spec_to_dict(obj)
    if isinstance(obj, PageSpec):
        return page_spec_to_dict(obj)
    if isinstance(obj, RasterSpec):
        return raster_spec_to_dict(obj)
    if isinstance(obj, DegradationSpec):
        return degradation_spec_to_dict(obj)
    if isinstance(obj, Point):
        return point_to_dict(obj)
    if isinstance(obj, BBox):
        return bbox_to_dict(obj)
    if isinstance(obj, Stroke):
        return stroke_to_dict(obj)
    if isinstance(obj, Fill):
        return fill_to_dict(obj)
    if isinstance(obj, FontSpec):
        return font_spec_to_dict(obj)
    raise TypeError(f"unsupported render2 object for serialization: {type(obj).__name__}")


def from_dict(cls: type[Any], data: Mapping[str, Any]) -> Any:
    if cls is RenderSpec:
        return spec_from_dict(data)
    if cls is RenderResult:
        return result_from_dict(data)
    if cls is TimingScene:
        return scene_from_dict(data)
    if cls is StyleSpec:
        return style_spec_from_dict(data)
    if cls is LayoutSpec:
        return layout_spec_from_dict(data)
    if cls is AnnotationSpec:
        return annotation_spec_from_dict(data)
    if cls is PageSpec:
        return page_spec_from_dict(data)
    if cls is RasterSpec:
        return raster_spec_from_dict(data)
    if cls is DegradationSpec:
        return degradation_spec_from_dict(data)
    raise TypeError(f"unsupported render2 class for deserialization: {cls!r}")


def style_spec_to_dict(spec: StyleSpec) -> dict[str, Any]:
    return {
        "family": spec.family,
        "palette": list(spec.palette),
        "primary_font": font_spec_to_dict(spec.primary_font),
        "label_font": font_spec_to_dict(spec.label_font),
        "waveform_stroke": stroke_to_dict(spec.waveform_stroke),
        "grid_stroke": stroke_to_dict(spec.grid_stroke),
        "grid_mode": spec.grid_mode,
        "bus_style": spec.bus_style,
        "unknown_style": spec.unknown_style,
        "cut_style": spec.cut_style,
        "transition_shape": spec.transition_shape,
        "color_mode": spec.color_mode,
    }


def style_spec_from_dict(data: Mapping[str, Any]) -> StyleSpec:
    return StyleSpec(
        family=str(data["family"]),
        palette=tuple(str(color) for color in data["palette"]),
        primary_font=font_spec_from_dict(data["primary_font"]),
        label_font=font_spec_from_dict(data["label_font"]),
        waveform_stroke=stroke_from_dict(data["waveform_stroke"]),
        grid_stroke=stroke_from_dict(data["grid_stroke"]),
        grid_mode=str(data["grid_mode"]),
        bus_style=str(data["bus_style"]),
        unknown_style=str(data["unknown_style"]),
        cut_style=str(data["cut_style"]),
        transition_shape=str(data["transition_shape"]),
        color_mode=str(data["color_mode"]),
    )


def layout_spec_to_dict(spec: LayoutSpec) -> dict[str, Any]:
    return {
        "lane_height": spec.lane_height,
        "lane_pitch": spec.lane_pitch,
        "tick_width": spec.tick_width,
        "label_position": spec.label_position,
        "label_alignment": spec.label_alignment,
        "margin": bbox_to_dict(spec.margin),
        "grouped_lanes": spec.grouped_lanes,
        "multiline_labels": spec.multiline_labels,
    }


def layout_spec_from_dict(data: Mapping[str, Any]) -> LayoutSpec:
    return LayoutSpec(
        lane_height=float(data["lane_height"]),
        lane_pitch=float(data["lane_pitch"]),
        tick_width=float(data["tick_width"]),
        label_position=str(data["label_position"]),
        label_alignment=str(data["label_alignment"]),
        margin=bbox_from_dict(data["margin"]),
        grouped_lanes=bool(data.get("grouped_lanes", False)),
        multiline_labels=bool(data.get("multiline_labels", False)),
    )


def annotation_spec_to_dict(spec: AnnotationSpec) -> dict[str, Any]:
    return {
        "policy": spec.policy.value,
        "measurement_label_style": spec.measurement_label_style,
        "helper_line_density": spec.helper_line_density,
        "nuisance_text_count": spec.nuisance_text_count,
        "semantic_guides_enabled": spec.semantic_guides_enabled,
    }


def annotation_spec_from_dict(data: Mapping[str, Any]) -> AnnotationSpec:
    return AnnotationSpec(
        policy=data["policy"],
        measurement_label_style=str(data["measurement_label_style"]),
        helper_line_density=float(data["helper_line_density"]),
        nuisance_text_count=int(data["nuisance_text_count"]),
        semantic_guides_enabled=bool(data["semantic_guides_enabled"]),
    )


def page_spec_to_dict(spec: PageSpec) -> dict[str, Any]:
    return {
        "enabled": spec.enabled,
        "caption_above": spec.caption_above,
        "caption_below": spec.caption_below,
        "surrounding_paragraph": spec.surrounding_paragraph,
        "table_border": spec.table_border,
        "page_header": spec.page_header,
        "page_footer": spec.page_footer,
        "crop_mode": spec.crop_mode,
    }


def page_spec_from_dict(data: Mapping[str, Any]) -> PageSpec:
    return PageSpec(
        enabled=bool(data["enabled"]),
        caption_above=bool(data["caption_above"]),
        caption_below=bool(data["caption_below"]),
        surrounding_paragraph=bool(data["surrounding_paragraph"]),
        table_border=bool(data["table_border"]),
        page_header=bool(data["page_header"]),
        page_footer=bool(data["page_footer"]),
        crop_mode=str(data["crop_mode"]),
    )


def raster_spec_to_dict(spec: RasterSpec) -> dict[str, Any]:
    return {
        "dpi": spec.dpi,
        "antialias": spec.antialias,
        "output_format": spec.output_format,
        "jpeg_quality": spec.jpeg_quality,
    }


def raster_spec_from_dict(data: Mapping[str, Any]) -> RasterSpec:
    return RasterSpec(
        dpi=int(data["dpi"]),
        antialias=bool(data["antialias"]),
        output_format=str(data["output_format"]),
        jpeg_quality=int(data.get("jpeg_quality", 85)),
    )


def degradation_spec_to_dict(spec: DegradationSpec) -> dict[str, Any]:
    return {
        "family": spec.family,
        "blur_sigma": spec.blur_sigma,
        "noise_sigma": spec.noise_sigma,
        "contrast": spec.contrast,
        "brightness": spec.brightness,
        "rotation_deg": spec.rotation_deg,
        "perspective": spec.perspective,
        "jpeg_quality": spec.jpeg_quality,
        "morphology": spec.morphology,
        "augraphy_pipeline": spec.augraphy_pipeline,
    }


def degradation_spec_from_dict(data: Mapping[str, Any]) -> DegradationSpec:
    return DegradationSpec(
        family=str(data["family"]),
        blur_sigma=float(data.get("blur_sigma", 0.0)),
        noise_sigma=float(data.get("noise_sigma", 0.0)),
        contrast=float(data.get("contrast", 1.0)),
        brightness=float(data.get("brightness", 1.0)),
        rotation_deg=float(data.get("rotation_deg", 0.0)),
        perspective=float(data.get("perspective", 0.0)),
        jpeg_quality=int(data.get("jpeg_quality", 95)),
        morphology=str(data.get("morphology", "none")),
        augraphy_pipeline=data.get("augraphy_pipeline"),
    )


def point_to_dict(point: Point) -> dict[str, float]:
    return {"x": point.x, "y": point.y}


def point_from_dict(data: Mapping[str, Any]) -> Point:
    return Point(x=float(data["x"]), y=float(data["y"]))


def bbox_to_dict(bbox: BBox) -> dict[str, float]:
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


def bbox_from_dict(data: Mapping[str, Any]) -> BBox:
    return BBox(
        x=float(data["x"]),
        y=float(data["y"]),
        width=float(data["width"]),
        height=float(data["height"]),
    )


def stroke_to_dict(stroke: Stroke) -> dict[str, Any]:
    return {
        "color": stroke.color,
        "width": stroke.width,
        "dasharray": list(stroke.dasharray),
        "linecap": stroke.linecap,
        "linejoin": stroke.linejoin,
        "opacity": stroke.opacity,
    }


def stroke_from_dict(data: Mapping[str, Any]) -> Stroke:
    return Stroke(
        color=str(data.get("color", "#000000")),
        width=float(data.get("width", 1.0)),
        dasharray=tuple(float(value) for value in data.get("dasharray", ())),
        linecap=str(data.get("linecap", "butt")),
        linejoin=str(data.get("linejoin", "miter")),
        opacity=float(data.get("opacity", 1.0)),
    )


def fill_to_dict(fill: Fill) -> dict[str, Any]:
    return {"color": fill.color, "opacity": fill.opacity}


def fill_from_dict(data: Mapping[str, Any]) -> Fill:
    return Fill(color=str(data.get("color", "#000000")), opacity=float(data.get("opacity", 1.0)))


def font_spec_to_dict(font: FontSpec) -> dict[str, Any]:
    return {
        "family": font.family,
        "size_px": font.size_px,
        "weight": font.weight,
        "style": font.style,
        "color": font.color,
    }


def font_spec_from_dict(data: Mapping[str, Any]) -> FontSpec:
    return FontSpec(
        family=str(data.get("family", "Helvetica, Arial, sans-serif")),
        size_px=float(data.get("size_px", 12.0)),
        weight=str(data.get("weight", "400")),
        style=str(data.get("style", "normal")),
        color=str(data.get("color", "#000000")),
    )


def tick_model_to_dict(ticks: TickModel) -> dict[str, Any]:
    return {
        "total_ticks": ticks.total_ticks,
        "tick_origin": ticks.tick_origin,
        "grid_pitch_hint": ticks.grid_pitch_hint,
    }


def tick_model_from_dict(data: Mapping[str, Any]) -> TickModel:
    return TickModel(
        total_ticks=int(data["total_ticks"]),
        tick_origin=int(data.get("tick_origin", 0)),
        grid_pitch_hint=float(data.get("grid_pitch_hint", 1.0)),
    )


def sample_run_to_dict(run: SampleRun) -> dict[str, Any]:
    return {
        "start_tick": run.start_tick,
        "end_tick": run.end_tick,
        "value": run.value,
        "is_unknown": run.is_unknown,
        "is_high_z": run.is_high_z,
    }


def sample_run_from_dict(data: Mapping[str, Any]) -> SampleRun:
    return SampleRun(
        start_tick=int(data["start_tick"]),
        end_tick=int(data["end_tick"]),
        value=str(data["value"]),
        is_unknown=bool(data.get("is_unknown", False)),
        is_high_z=bool(data.get("is_high_z", False)),
    )


def lane_scene_to_dict(lane: LaneScene) -> dict[str, Any]:
    return {
        "name": lane.name,
        "lane_type": lane.lane_type.value,
        "runs": [sample_run_to_dict(run) for run in lane.runs],
        "width_bits": lane.width_bits,
        "visibility": lane.visibility.value,
    }


def lane_scene_from_dict(data: Mapping[str, Any]) -> LaneScene:
    return LaneScene(
        name=str(data["name"]),
        lane_type=LaneType(data["lane_type"]),
        runs=tuple(sample_run_from_dict(item) for item in data.get("runs", ())),
        width_bits=data.get("width_bits"),
        visibility=VisibilityClass(data.get("visibility", VisibilityClass.VISIBLE_TEXT.value)),
    )


def cut_region_to_dict(cut: CutRegion) -> dict[str, Any]:
    return {
        "name": cut.name,
        "start_tick": cut.start_tick,
        "end_tick": cut.end_tick,
        "meaning": cut.meaning,
        "label": cut.label,
    }


def cut_region_from_dict(data: Mapping[str, Any]) -> CutRegion:
    return CutRegion(
        name=str(data["name"]),
        start_tick=int(data["start_tick"]),
        end_tick=int(data["end_tick"]),
        meaning=str(data["meaning"]),
        label=data.get("label"),
    )


def visual_event_to_dict(event: VisualEvent) -> dict[str, Any]:
    return {
        "name": event.name,
        "tick": event.tick,
        "placement": event.placement,
        "target_visibility": event.target_visibility.value,
    }


def visual_event_from_dict(data: Mapping[str, Any]) -> VisualEvent:
    return VisualEvent(
        name=str(data["name"]),
        tick=int(data["tick"]),
        placement=str(data["placement"]),
        target_visibility=VisibilityClass(data["target_visibility"]),
    )


def visual_constraint_to_dict(constraint: VisualConstraint) -> dict[str, Any]:
    return {
        "name": constraint.name,
        "kind": constraint.kind,
        "region": constraint.region,
        "lane_names": list(constraint.lane_names),
        "start_tick": constraint.start_tick,
        "end_tick": constraint.end_tick,
        "anchor_ref": constraint.anchor_ref,
        "window_ref": constraint.window_ref,
        "visibility": constraint.visibility.value,
    }


def visual_constraint_from_dict(data: Mapping[str, Any]) -> VisualConstraint:
    return VisualConstraint(
        name=str(data["name"]),
        kind=str(data["kind"]),
        region=str(data["region"]),
        lane_names=tuple(str(name) for name in data.get("lane_names", ())),
        start_tick=_optional_int(data.get("start_tick")),
        end_tick=_optional_int(data.get("end_tick")),
        anchor_ref=data.get("anchor_ref"),
        window_ref=data.get("window_ref"),
        visibility=VisibilityClass(data["visibility"]),
    )


def decoration_to_dict(decoration: Decoration) -> dict[str, Any]:
    return {
        "kind": decoration.kind.value,
        "semantic": decoration.semantic,
        "target_ref": decoration.target_ref,
        "text": decoration.text,
        "visibility_class": decoration.visibility_class.value,
        "anchor_tick": decoration.anchor_tick,
        "span": list(decoration.span) if decoration.span is not None else None,
        "lane_names": list(decoration.lane_names),
        "style": decoration_style_to_dict(decoration.style),
    }


def decoration_from_dict(data: Mapping[str, Any]) -> Decoration:
    span = data.get("span")
    return Decoration(
        kind=DecorationKind(data["kind"]),
        semantic=bool(data["semantic"]),
        target_ref=data.get("target_ref"),
        text=data.get("text"),
        visibility_class=VisibilityClass(data.get("visibility_class", VisibilityClass.VISIBLE_GEOMETRY.value)),
        anchor_tick=_optional_int(data.get("anchor_tick")),
        span=tuple(int(value) for value in span) if span is not None else None,
        lane_names=tuple(str(name) for name in data.get("lane_names", ())),
        style=decoration_style_from_dict(data.get("style", {})),
    )


def decoration_style_to_dict(style: DecorationStyle) -> dict[str, Any]:
    return {
        "stroke": stroke_to_dict(style.stroke) if style.stroke is not None else None,
        "fill": fill_to_dict(style.fill) if style.fill is not None else None,
        "font": font_spec_to_dict(style.font) if style.font is not None else None,
        "dashed": style.dashed,
        "handdrawn": style.handdrawn,
    }


def decoration_style_from_dict(data: Mapping[str, Any]) -> DecorationStyle:
    return DecorationStyle(
        stroke=stroke_from_dict(data["stroke"]) if data.get("stroke") is not None else None,
        fill=fill_from_dict(data["fill"]) if data.get("fill") is not None else None,
        font=font_spec_from_dict(data["font"]) if data.get("font") is not None else None,
        dashed=bool(data.get("dashed", False)),
        handdrawn=bool(data.get("handdrawn", False)),
    )


def text_primitive_to_dict(text: TextPrimitive) -> dict[str, Any]:
    return {
        "text": text.text,
        "bbox": bbox_to_dict(text.bbox),
        "role": text.role,
        "visibility_class": text.visibility_class,
    }


def text_primitive_from_dict(data: Mapping[str, Any]) -> TextPrimitive:
    return TextPrimitive(
        text=str(data["text"]),
        bbox=bbox_from_dict(data["bbox"]),
        role=str(data["role"]),
        visibility_class=str(data["visibility_class"]),
    )


def visibility_report_to_dict(report: VisualVisibilityReport) -> dict[str, Any]:
    return {
        "rendered_text": [text_primitive_to_dict(text) for text in report.rendered_text],
        "target_tokens_visible": sorted(report.target_tokens_visible),
        "nuisance_tokens": sorted(report.nuisance_tokens),
        "debug_overlay_tokens": sorted(report.debug_overlay_tokens),
        "leaked_tokens": sorted(report.leaked_tokens),
        "occluded_lane_fractions": dict(report.occluded_lane_fractions),
        "minimum_contrast": report.minimum_contrast,
    }


def visibility_report_from_dict(data: Mapping[str, Any]) -> VisualVisibilityReport:
    return VisualVisibilityReport(
        rendered_text=tuple(text_primitive_from_dict(item) for item in data.get("rendered_text", ())),
        target_tokens_visible=frozenset(str(token) for token in data.get("target_tokens_visible", ())),
        nuisance_tokens=frozenset(str(token) for token in data.get("nuisance_tokens", ())),
        debug_overlay_tokens=frozenset(str(token) for token in data.get("debug_overlay_tokens", ())),
        leaked_tokens=frozenset(str(token) for token in data.get("leaked_tokens", ())),
        occluded_lane_fractions={
            str(key): float(value) for key, value in data.get("occluded_lane_fractions", {}).items()
        },
        minimum_contrast=float(data["minimum_contrast"]),
    )


def diagram_layout_to_dict(layout: DiagramLayout) -> dict[str, Any]:
    return {
        "width": layout.width,
        "height": layout.height,
        "plot_origin": point_to_dict(layout.plot_origin),
        "tick_width": layout.tick_width,
        "lane_height": layout.lane_height,
        "lane_pitch": layout.lane_pitch,
        "bbox_by_role": {
            role: [bbox_to_dict(bbox) for bbox in boxes]
            for role, boxes in sorted(layout.bbox_by_role.items())
        },
    }


def diagram_layout_from_dict(data: Mapping[str, Any]) -> DiagramLayout:
    return DiagramLayout(
        width=float(data["width"]),
        height=float(data["height"]),
        plot_origin=point_from_dict(data["plot_origin"]),
        tick_width=float(data["tick_width"]),
        lane_height=float(data["lane_height"]),
        lane_pitch=float(data["lane_pitch"]),
        bbox_by_role={
            str(role): tuple(bbox_from_dict(item) for item in boxes)
            for role, boxes in data.get("bbox_by_role", {}).items()
        },
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
