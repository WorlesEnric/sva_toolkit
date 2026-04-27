"""Layout overflow audit for render2 results."""

from __future__ import annotations

from dataclasses import dataclass

from sva_toolkit.timing.render2.primitives import BBox
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import TimingScene


@dataclass(frozen=True)
class OverflowReport:
    primitive_count: int
    overflow_count: int
    overflowing_roles: tuple[str, ...]
    union_bbox: BBox | None
    canvas_width: float
    canvas_height: float
    passed: bool
    reason: str | None


def audit_layout_overflow(scene: TimingScene, render_result: RenderResult) -> OverflowReport:
    del scene
    boxes = _boxes_by_role(render_result)
    union = _union(box for _role, box in boxes)
    overflowing_roles = tuple(
        role for role, box in boxes if _is_outside_canvas(box, render_result.layout.width, render_result.layout.height)
    )
    passed = not overflowing_roles
    return OverflowReport(
        primitive_count=len(boxes),
        overflow_count=len(overflowing_roles),
        overflowing_roles=overflowing_roles,
        union_bbox=union,
        canvas_width=render_result.layout.width,
        canvas_height=render_result.layout.height,
        passed=passed,
        reason=None if passed else "layout_overflow",
    )


def _boxes_by_role(render_result: RenderResult) -> tuple[tuple[str, BBox], ...]:
    boxes: list[tuple[str, BBox]] = []
    for role, role_boxes in render_result.layout.bbox_by_role.items():
        boxes.extend((role, box) for box in role_boxes)
    known = {(role, box) for role, box in boxes}
    for text in render_result.visibility.rendered_text:
        item = (text.role, text.bbox)
        if item not in known:
            boxes.append(item)
            known.add(item)
    return tuple(boxes)


def _is_outside_canvas(box: BBox, width: float, height: float) -> bool:
    tolerance = 1e-6
    return (
        box.x < -tolerance
        or box.y < -tolerance
        or box.x + box.width > width + tolerance
        or box.y + box.height > height + tolerance
    )


def _union(boxes: tuple[BBox, ...] | list[BBox] | object) -> BBox | None:
    boxes = tuple(boxes)
    if not boxes:
        return None
    min_x = min(box.x for box in boxes)
    min_y = min(box.y for box in boxes)
    max_x = max(box.x + box.width for box in boxes)
    max_y = max(box.y + box.height for box in boxes)
    return BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)


__all__ = ["OverflowReport", "audit_layout_overflow"]
