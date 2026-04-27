"""Geometry helpers for the native SVG timing renderer."""

from __future__ import annotations

from dataclasses import dataclass

from sva_toolkit.timing.render2.primitives import BBox, FontSpec, Point
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import LayoutSpec, PageSpec

from sva_toolkit.timing.render2.native.text_metrics import estimate_text_width


_LABEL_GAP = 12.0
_PAGE_LINE_GAP = 8.0


@dataclass(frozen=True)
class NativeGeometry:
    width: float
    height: float
    plot_origin: Point
    plot_width: float
    plot_height: float
    label_width: float
    page_top: float
    page_bottom: float
    layout: LayoutSpec
    lane_names: tuple[str, ...]
    total_ticks: int

    @property
    def plot_right(self) -> float:
        return self.plot_origin.x + self.plot_width

    @property
    def plot_bottom(self) -> float:
        return self.plot_origin.y + self.plot_height

    def tick_to_x(self, tick: float) -> float:
        return self.plot_origin.x + tick * self.layout.tick_width

    def lane_top(self, lane_index: int) -> float:
        return self.plot_origin.y + lane_index * self.layout.lane_pitch

    def lane_center_y(self, lane_index: int) -> float:
        return self.lane_top(lane_index) + self.layout.lane_height / 2

    def lane_bbox(self, lane_index: int) -> BBox:
        return BBox(
            x=self.plot_origin.x,
            y=self.lane_top(lane_index),
            width=self.plot_width,
            height=self.layout.lane_height,
        )

    def bbox_for_lane(self, name: str) -> BBox | None:
        try:
            return self.lane_bbox(self.lane_names.index(name))
        except ValueError:
            return None

    def high_y(self, lane_index: int) -> float:
        return self.lane_center_y(lane_index) - self.layout.lane_height * 0.23

    def low_y(self, lane_index: int) -> float:
        return self.lane_center_y(lane_index) + self.layout.lane_height * 0.23

    def bus_top_y(self, lane_index: int) -> float:
        return self.lane_top(lane_index) + self.layout.lane_height * 0.22

    def bus_bottom_y(self, lane_index: int) -> float:
        return self.lane_top(lane_index) + self.layout.lane_height * 0.78

    def label_anchor(self, lane_index: int) -> tuple[Point, str]:
        center = self.lane_center_y(lane_index) + self.layout.lane_height * 0.12
        position = self.layout.label_position
        if position == "right":
            return Point(self.plot_right + _LABEL_GAP, center), "start"
        if position == "inside_left":
            return Point(self.plot_origin.x + 6.0, center), "start"
        if position == "inside_right":
            return Point(self.plot_right - 6.0, center), "end"
        return Point(self.plot_origin.x - _LABEL_GAP, center), "end"


def compute_geometry(scene: TimingScene, layout: LayoutSpec, page: PageSpec, label_font: FontSpec) -> NativeGeometry:
    """Compute the renderer layout from scene lanes, ticks, page options, and margins."""

    lane_names = tuple(lane.name for lane in scene.lanes)
    total_ticks = max(1, scene.ticks.total_ticks)
    label_width = max((estimate_text_width(name, label_font) for name in lane_names), default=0.0)
    left_margin = layout.margin.x
    top_margin = layout.margin.y
    right_margin = layout.margin.width
    bottom_margin = layout.margin.height

    page_top = _page_top_height(page, label_font)
    page_bottom = _page_bottom_height(page, label_font)
    plot_width = total_ticks * layout.tick_width
    plot_height = 0.0
    if scene.lanes:
        plot_height = (len(scene.lanes) - 1) * layout.lane_pitch + layout.lane_height

    position = layout.label_position
    left_label_space = label_width + _LABEL_GAP if position == "left" else 0.0
    right_label_space = label_width + _LABEL_GAP if position == "right" else 0.0
    plot_origin = Point(
        x=left_margin + left_label_space,
        y=top_margin + page_top,
    )
    width = left_margin + left_label_space + plot_width + right_label_space + right_margin
    height = top_margin + page_top + plot_height + page_bottom + bottom_margin

    return NativeGeometry(
        width=width,
        height=height,
        plot_origin=plot_origin,
        plot_width=plot_width,
        plot_height=plot_height,
        label_width=label_width,
        page_top=page_top,
        page_bottom=page_bottom,
        layout=layout,
        lane_names=lane_names,
        total_ticks=total_ticks,
    )


def _page_top_height(page: PageSpec, font: FontSpec) -> float:
    if not page.enabled:
        return 0.0
    lines = int(page.page_header) + int(page.surrounding_paragraph) + int(page.caption_above)
    return lines * (font.size_px * 1.25 + _PAGE_LINE_GAP)


def _page_bottom_height(page: PageSpec, font: FontSpec) -> float:
    if not page.enabled:
        return 0.0
    lines = int(page.caption_below) + int(page.page_footer)
    return lines * (font.size_px * 1.25 + _PAGE_LINE_GAP)
