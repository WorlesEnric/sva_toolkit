"""Primitive-to-SVG translation for the native renderer."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable

from sva_toolkit.timing.render2.primitives import (
    BBox,
    Fill,
    Group,
    Line,
    Path,
    Point,
    Polyline,
    Primitive,
    Rect,
    Stroke,
    Text,
)

from sva_toolkit.timing.render2.native.text_metrics import estimate_text_bbox


SVG_NS = "http://www.w3.org/2000/svg"
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def render_svg(primitives: Iterable[Primitive], *, width: float, height: float, antialias: bool) -> str:
    ET.register_namespace("", SVG_NS)
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": _fmt(width),
            "height": _fmt(height),
            "viewBox": f"0 0 {_fmt(width)} {_fmt(height)}",
            "role": "img",
        },
    )
    if not antialias:
        root.set("shape-rendering", "crispEdges")
        root.set("text-rendering", "geometricPrecision")

    for primitive in sorted(primitives, key=lambda item: item.z):
        append_primitive(root, primitive, antialias=antialias)

    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


def append_primitive(parent: ET.Element, primitive: Primitive, *, antialias: bool) -> None:
    if isinstance(primitive, Rect):
        _append_rect(parent, primitive, antialias=antialias)
    elif isinstance(primitive, Line):
        _append_line(parent, primitive, antialias=antialias)
    elif isinstance(primitive, Polyline):
        _append_polyline(parent, primitive, antialias=antialias)
    elif isinstance(primitive, Path):
        _append_path(parent, primitive)
    elif isinstance(primitive, Text):
        _append_text(parent, primitive, antialias=antialias)
    elif isinstance(primitive, Group):
        element = ET.SubElement(parent, f"{{{SVG_NS}}}g", _base_attrs(primitive))
        if primitive.transform:
            element.set("transform", primitive.transform)
        for child in primitive.children:
            append_primitive(element, child, antialias=antialias)


def primitive_bbox(primitive: Primitive) -> BBox | None:
    if isinstance(primitive, Rect):
        return primitive.bbox
    if isinstance(primitive, Text):
        return estimate_text_bbox(
            primitive.text,
            primitive.font,
            primitive.anchor,
            text_anchor=primitive.text_anchor,
        )
    if isinstance(primitive, Line):
        return _bbox_from_points((primitive.p0, primitive.p1), pad=primitive.stroke.width / 2)
    if isinstance(primitive, Polyline):
        return _bbox_from_points(primitive.points, pad=primitive.stroke.width / 2)
    if isinstance(primitive, Path):
        return _path_bbox(primitive)
    if isinstance(primitive, Group):
        boxes = tuple(box for child in primitive.children if (box := primitive_bbox(child)) is not None)
        return _union_boxes(boxes)
    return None


def _append_rect(parent: ET.Element, primitive: Rect, *, antialias: bool) -> None:
    bbox = _snap_bbox(primitive.bbox) if not antialias else primitive.bbox
    attrs = {
        **_base_attrs(primitive),
        "x": _fmt(bbox.x),
        "y": _fmt(bbox.y),
        "width": _fmt(bbox.width),
        "height": _fmt(bbox.height),
        "rx": _fmt(primitive.radius),
        "ry": _fmt(primitive.radius),
    }
    _apply_fill(attrs, primitive.fill)
    _apply_stroke(attrs, primitive.stroke)
    ET.SubElement(parent, f"{{{SVG_NS}}}rect", attrs)


def _append_line(parent: ET.Element, primitive: Line, *, antialias: bool) -> None:
    p0 = _snap_point(primitive.p0) if not antialias else primitive.p0
    p1 = _snap_point(primitive.p1) if not antialias else primitive.p1
    attrs = {
        **_base_attrs(primitive),
        "x1": _fmt(p0.x),
        "y1": _fmt(p0.y),
        "x2": _fmt(p1.x),
        "y2": _fmt(p1.y),
    }
    _apply_stroke(attrs, primitive.stroke)
    ET.SubElement(parent, f"{{{SVG_NS}}}line", attrs)


def _append_polyline(parent: ET.Element, primitive: Polyline, *, antialias: bool) -> None:
    points = tuple(_snap_point(point) for point in primitive.points) if not antialias else primitive.points
    attrs = {
        **_base_attrs(primitive),
        "points": " ".join(f"{_fmt(point.x)},{_fmt(point.y)}" for point in points),
        "fill": "none",
    }
    _apply_stroke(attrs, primitive.stroke)
    ET.SubElement(parent, f"{{{SVG_NS}}}polyline", attrs)


def _append_path(parent: ET.Element, primitive: Path) -> None:
    attrs = {
        **_base_attrs(primitive),
        "d": primitive.d,
    }
    _apply_fill(attrs, primitive.fill)
    _apply_stroke(attrs, primitive.stroke)
    ET.SubElement(parent, f"{{{SVG_NS}}}path", attrs)


def _append_text(parent: ET.Element, primitive: Text, *, antialias: bool) -> None:
    anchor = _snap_point(primitive.anchor) if not antialias else primitive.anchor
    attrs = {
        **_base_attrs(primitive),
        "x": _fmt(anchor.x),
        "y": _fmt(anchor.y),
        "font-family": primitive.font.family,
        "font-size": _fmt(primitive.font.size_px),
        "font-weight": primitive.font.weight,
        "font-style": primitive.font.style,
        "fill": primitive.font.color,
        "text-anchor": primitive.text_anchor,
        "dominant-baseline": "alphabetic",
    }
    element = ET.SubElement(parent, f"{{{SVG_NS}}}text", attrs)
    element.text = primitive.text


def _base_attrs(primitive: Primitive) -> dict[str, str]:
    attrs = {"data-role": primitive.role}
    if primitive.id:
        attrs["id"] = primitive.id
    return attrs


def _apply_stroke(attrs: dict[str, str], stroke: Stroke | None) -> None:
    if stroke is None:
        attrs["stroke"] = "none"
        return
    attrs["stroke"] = stroke.color
    attrs["stroke-width"] = _fmt(stroke.width)
    attrs["stroke-linecap"] = stroke.linecap
    attrs["stroke-linejoin"] = stroke.linejoin
    if stroke.dasharray:
        attrs["stroke-dasharray"] = " ".join(_fmt(value) for value in stroke.dasharray)
    if stroke.opacity != 1.0:
        attrs["stroke-opacity"] = _fmt(stroke.opacity)


def _apply_fill(attrs: dict[str, str], fill: Fill | None) -> None:
    if fill is None:
        attrs["fill"] = "none"
        return
    attrs["fill"] = fill.color
    if fill.opacity != 1.0:
        attrs["fill-opacity"] = _fmt(fill.opacity)


def _bbox_from_points(points: Iterable[Point], *, pad: float = 0.0) -> BBox | None:
    points = tuple(points)
    if not points:
        return None
    xs = tuple(point.x for point in points)
    ys = tuple(point.y for point in points)
    return BBox(
        x=min(xs) - pad,
        y=min(ys) - pad,
        width=max(xs) - min(xs) + pad * 2,
        height=max(ys) - min(ys) + pad * 2,
    )


def _path_bbox(path: Path) -> BBox | None:
    values = tuple(float(match.group(0)) for match in _NUMBER_RE.finditer(path.d))
    points = tuple(Point(values[index], values[index + 1]) for index in range(0, len(values) - 1, 2))
    return _bbox_from_points(points, pad=path.stroke.width / 2)


def _union_boxes(boxes: Iterable[BBox]) -> BBox | None:
    boxes = tuple(boxes)
    if not boxes:
        return None
    min_x = min(box.x for box in boxes)
    min_y = min(box.y for box in boxes)
    max_x = max(box.x + box.width for box in boxes)
    max_y = max(box.y + box.height for box in boxes)
    return BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)


def _snap_point(point: Point) -> Point:
    return Point(x=round(point.x) + 0.5, y=round(point.y) + 0.5)


def _snap_bbox(bbox: BBox) -> BBox:
    return BBox(x=round(bbox.x), y=round(bbox.y), width=round(bbox.width), height=round(bbox.height))


def _fmt(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"
