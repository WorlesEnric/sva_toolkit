"""Renderer-independent vector primitives for timing diagrams."""

from __future__ import annotations

from dataclasses import dataclass


ALLOWED_PRIMITIVE_ROLES = frozenset(
    {
        "background",
        "outer_card",
        "lane_label",
        "lane_separator",
        "grid_major",
        "grid_minor",
        "tick_label",
        "bit_wave_high",
        "bit_wave_low",
        "bit_transition",
        "bus_region",
        "bus_region_edge",
        "bus_value_text",
        "unknown_region",
        "hiz_region",
        "cut_marker",
        "measurement_bracket",
        "vertical_helper_line",
        "horizontal_helper_line",
        "annotation_arrow",
        "nuisance_text",
        "caption_text",
        "anchor_marker",
        "anchor_label",
        "response_arrow",
        "response_label",
        "hold_highlight",
        "debug_overlay",
        "page_caption",
        "page_paragraph",
        "page_table_border",
        "page_header",
        "page_footer",
    }
)


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Stroke:
    color: str = "#000000"
    width: float = 1.0
    dasharray: tuple[float, ...] = ()
    linecap: str = "butt"
    linejoin: str = "miter"
    opacity: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dasharray", tuple(self.dasharray))


@dataclass(frozen=True)
class Fill:
    color: str = "#000000"
    opacity: float = 1.0


@dataclass(frozen=True)
class FontSpec:
    family: str = "Helvetica, Arial, sans-serif"
    size_px: float = 12.0
    weight: str = "400"
    style: str = "normal"
    color: str = "#000000"


@dataclass(frozen=True)
class Primitive:
    role: str
    z: int = 0
    id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ALLOWED_PRIMITIVE_ROLES:
            raise ValueError(f"invalid primitive role: {self.role}")


@dataclass(frozen=True)
class Line(Primitive):
    p0: Point = Point(0, 0)
    p1: Point = Point(0, 0)
    stroke: Stroke = Stroke()


@dataclass(frozen=True)
class Polyline(Primitive):
    points: tuple[Point, ...] = ()
    stroke: Stroke = Stroke()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "points", tuple(self.points))


@dataclass(frozen=True)
class Path(Primitive):
    d: str = ""
    stroke: Stroke = Stroke()
    fill: Fill | None = None


@dataclass(frozen=True)
class Text(Primitive):
    text: str = ""
    anchor: Point = Point(0, 0)
    font: FontSpec = FontSpec()
    text_anchor: str = "start"
    bbox_policy: str = "tight"
    visibility_class: str = "visible_text"


@dataclass(frozen=True)
class Rect(Primitive):
    bbox: BBox = BBox(0, 0, 0, 0)
    stroke: Stroke | None = None
    fill: Fill | None = None
    radius: float = 0.0


@dataclass(frozen=True)
class Group(Primitive):
    children: tuple[Primitive, ...] = ()
    transform: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "children", tuple(self.children))
