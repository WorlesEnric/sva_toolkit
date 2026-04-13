"""Symbolic SVG renderer for scenario IR documents."""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from sva_toolkit.timing.core.scenario import ConstraintRegion, ExtractionStatus, ScenarioDocument
from sva_toolkit.timing.projection.scenario_view import ScenarioView, build_scenario_view


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

LEFT_GUTTER = 180
TOP_GUTTER = 92
RIGHT_GUTTER = 36
BOTTOM_GUTTER = 96
LANE_HEIGHT = 44
LANE_GAP = 12
ANCHOR_WIDTH = 88
WINDOW_WIDTH = 132
CUT_WIDTH = 34


@dataclass(frozen=True)
class ItemGeometry:
    """Resolved x geometry for one timeline item."""

    left: float
    right: float
    center: float


def render_symbolic_svg(document: ScenarioDocument) -> str:
    """Render a scenario document with symbolic windows and cuts."""

    view = build_scenario_view(document)
    item_boxes = _compute_item_geometry(view)
    lane_count = max(1, len(view.lanes))
    width = LEFT_GUTTER + (item_boxes[-1].right if item_boxes else 420) + RIGHT_GUTTER
    height = TOP_GUTTER + lane_count * (LANE_HEIGHT + LANE_GAP) + BOTTOM_GUTTER

    root = ET.Element(
        _svg("svg"),
        {
            "width": str(int(width)),
            "height": str(int(height)),
            "viewBox": f"0 0 {int(width)} {int(height)}",
        },
    )
    _append_styles(root)
    _append_header(root, document, width)
    _append_timeline_background(root, view, item_boxes, lane_count)
    _append_lanes(root, view, item_boxes)
    _append_footer(root, document, height - 54)
    return ET.tostring(root, encoding="unicode")


def _compute_item_geometry(view: ScenarioView) -> list[ItemGeometry]:
    current_x = 0.0
    geometry = []
    for item in view.timeline:
        width = ANCHOR_WIDTH if item.kind == "anchor" else WINDOW_WIDTH if item.kind == "window" else CUT_WIDTH
        geometry.append(ItemGeometry(left=current_x, right=current_x + width, center=current_x + width / 2.0))
        current_x += width
    if not geometry:
        geometry.append(ItemGeometry(left=0.0, right=420.0, center=210.0))
    return geometry


def _append_styles(root: ET.Element) -> None:
    style = ET.SubElement(root, _svg("style"))
    style.text = """
    .bg { fill: #f8fafc; }
    .header { font: 600 20px ui-sans-serif, system-ui, sans-serif; fill: #0f172a; }
    .subhead { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #475569; }
    .badge { font: 700 11px ui-sans-serif, system-ui, sans-serif; }
    .lane-label { font: 600 13px ui-sans-serif, system-ui, sans-serif; fill: #0f172a; }
    .lane-line { stroke: #cbd5e1; stroke-width: 2; }
    .anchor-line { stroke: #94a3b8; stroke-width: 1.5; stroke-dasharray: 4 4; }
    .anchor-name { font: 600 12px ui-sans-serif, system-ui, sans-serif; fill: #1e293b; }
    .window-box { fill: #e2e8f0; stroke: #94a3b8; stroke-width: 1; rx: 10; ry: 10; }
    .window-label { font: 600 12px ui-sans-serif, system-ui, sans-serif; fill: #1e293b; }
    .cut-line { stroke: #64748b; stroke-width: 2; fill: none; }
    .constraint-box { fill: #dbeafe; stroke: #2563eb; stroke-width: 1; rx: 8; ry: 8; }
    .constraint-lossy { fill: #fef3c7; stroke: #d97706; }
    .constraint-label { font: 600 11px ui-sans-serif, system-ui, sans-serif; fill: #1e293b; }
    .footer { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #334155; }
    .property-name { font-weight: 700; }
    """


def _append_header(root: ET.Element, document: ScenarioDocument, width: float) -> None:
    ET.SubElement(root, _svg("rect"), {"x": "0", "y": "0", "width": str(int(width)), "height": str(TOP_GUTTER), "class": "bg"})
    title = ET.SubElement(root, _svg("text"), {"x": "28", "y": "34", "class": "header"})
    title.text = document.name
    subtitle = ET.SubElement(root, _svg("text"), {"x": "28", "y": "56", "class": "subhead"})
    subtitle.text = f"@({document.clocking.edge} {document.clocking.signal})"
    if document.clocking.disable_iff:
        subtitle.text += f" disable iff ({document.clocking.disable_iff})"

    badge_fill = "#dcfce7" if document.effective_status == ExtractionStatus.EXACT else "#fef3c7" if document.effective_status == ExtractionStatus.LOSSY else "#fee2e2"
    ET.SubElement(
        root,
        _svg("rect"),
        {
            "x": str(int(width - 128)),
            "y": "18",
            "width": "92",
            "height": "26",
            "rx": "13",
            "ry": "13",
            "fill": badge_fill,
            "stroke": "none",
        },
    )
    badge_text = ET.SubElement(root, _svg("text"), {"x": str(int(width - 112)), "y": "36", "class": "badge"})
    badge_text.text = document.effective_status.value.upper()


def _append_timeline_background(root: ET.Element, view: ScenarioView, item_boxes: list[ItemGeometry], lane_count: int) -> None:
    bottom = TOP_GUTTER + lane_count * (LANE_HEIGHT + LANE_GAP)
    ET.SubElement(
        root,
        _svg("rect"),
        {
            "x": str(LEFT_GUTTER),
            "y": str(TOP_GUTTER - 24),
            "width": str(int(item_boxes[-1].right + RIGHT_GUTTER)),
            "height": str(int(bottom - TOP_GUTTER + 36)),
            "fill": "#ffffff",
            "stroke": "#e2e8f0",
        },
    )
    for item, box in zip(view.timeline, item_boxes):
        if item.kind == "anchor":
            ET.SubElement(
                root,
                _svg("line"),
                {
                    "x1": _fmt(LEFT_GUTTER + box.center),
                    "y1": str(TOP_GUTTER - 10),
                    "x2": _fmt(LEFT_GUTTER + box.center),
                    "y2": str(bottom),
                    "class": "anchor-line",
                },
            )
            text = ET.SubElement(
                root,
                _svg("text"),
                {
                    "x": _fmt(LEFT_GUTTER + box.center),
                    "y": str(TOP_GUTTER - 20),
                    "text-anchor": "middle",
                    "class": "anchor-name",
                },
            )
            text.text = item.label
        elif item.kind == "window":
            ET.SubElement(
                root,
                _svg("rect"),
                {
                    "x": _fmt(LEFT_GUTTER + box.left + 8),
                    "y": str(TOP_GUTTER - 38),
                    "width": _fmt(box.right - box.left - 16),
                    "height": "24",
                    "class": "window-box",
                },
            )
            text = ET.SubElement(
                root,
                _svg("text"),
                {
                    "x": _fmt(LEFT_GUTTER + box.center),
                    "y": str(TOP_GUTTER - 22),
                    "text-anchor": "middle",
                    "class": "window-label",
                },
            )
            text.text = item.label
        elif item.kind == "cut":
            x = LEFT_GUTTER + box.center
            ET.SubElement(
                root,
                _svg("path"),
                {
                    "d": f"M {_fmt(x)} {TOP_GUTTER - 8} C {_fmt(x - 6)} {TOP_GUTTER + 10}, {_fmt(x + 6)} {TOP_GUTTER + 26}, {_fmt(x)} {TOP_GUTTER + 44}",
                    "class": "cut-line",
                },
            )
            label = ET.SubElement(root, _svg("text"), {"x": _fmt(x), "y": str(TOP_GUTTER - 20), "text-anchor": "middle", "class": "subhead"})
            label.text = item.label


def _append_lanes(root: ET.Element, view: ScenarioView, item_boxes: list[ItemGeometry]) -> None:
    total_width = item_boxes[-1].right
    for index, lane in enumerate(view.lanes):
        top = TOP_GUTTER + index * (LANE_HEIGHT + LANE_GAP)
        center_y = top + LANE_HEIGHT / 2.0
        label = ET.SubElement(root, _svg("text"), {"x": str(LEFT_GUTTER - 18), "y": _fmt(center_y), "text-anchor": "end", "dominant-baseline": "middle", "class": "lane-label"})
        label.text = lane.signal_name
        ET.SubElement(
            root,
            _svg("line"),
            {
                "x1": str(LEFT_GUTTER),
                "y1": _fmt(center_y),
                "x2": _fmt(LEFT_GUTTER + total_width),
                "y2": _fmt(center_y),
                "class": "lane-line",
            },
        )
        for constraint in lane.constraints:
            left, right = _constraint_bounds(constraint, view, item_boxes)
            if left is None or right is None:
                continue
            css_class = "constraint-box"
            if view.document.effective_status == ExtractionStatus.LOSSY:
                css_class += " constraint-lossy"
            ET.SubElement(
                root,
                _svg("rect"),
                {
                    "x": _fmt(LEFT_GUTTER + left),
                    "y": _fmt(top + 8),
                    "width": _fmt(max(28.0, right - left)),
                    "height": _fmt(LANE_HEIGHT - 16),
                    "class": css_class,
                },
            )
            text = ET.SubElement(
                root,
                _svg("text"),
                {
                    "x": _fmt(LEFT_GUTTER + (left + right) / 2.0),
                    "y": _fmt(center_y),
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                    "class": "constraint-label",
                },
            )
            text.text = _constraint_label(constraint)


def _constraint_bounds(constraint, view: ScenarioView, item_boxes: list[ItemGeometry]) -> tuple[float | None, float | None]:
    anchor_positions = {item.anchor: box.center for item, box in zip(view.timeline, item_boxes) if item.anchor}
    window_positions = {item.window: (box.left, box.right) for item, box in zip(view.timeline, item_boxes) if item.window}
    if constraint.region == ConstraintRegion.AT:
        center = anchor_positions.get(constraint.anchor)
        if center is None:
            return None, None
        return center - 18, center + 18
    if constraint.region == ConstraintRegion.IN:
        return window_positions.get(constraint.window, (None, None))
    if constraint.region == ConstraintRegion.BEFORE:
        center = anchor_positions.get(constraint.anchor)
        if center is None:
            return None, None
        return 0.0, max(28.0, center - 12)
    if constraint.region == ConstraintRegion.AFTER:
        center = anchor_positions.get(constraint.anchor)
        if center is None:
            return None, None
        return center + 12, item_boxes[-1].right
    start = anchor_positions.get(constraint.start_anchor)
    end = anchor_positions.get(constraint.end_anchor)
    if start is None or end is None:
        return None, None
    return min(start, end), max(start, end)


def _constraint_label(constraint) -> str:
    if constraint.relation == "eq":
        return constraint.value or "="
    if constraint.relation == "neq":
        return f"!={constraint.value}"
    mapping = {
        "high": "1",
        "low": "0",
        "rise": "rise",
        "fall": "fall",
        "stable": "stable",
        "change": "change",
        "unknown": "unknown",
        "dontcare": "?",
        "raw": constraint.value or "expr",
    }
    return mapping.get(constraint.relation, constraint.relation)


def _append_footer(root: ET.Element, document: ScenarioDocument, top: float) -> None:
    y = top
    for prop in document.properties:
        line = ET.SubElement(root, _svg("text"), {"x": "28", "y": _fmt(y), "class": "footer"})
        line.text = f"{prop.name}: {prop.status.value} | {prop.body}"
        y += 18
        for note in prop.notes:
            note_line = ET.SubElement(root, _svg("text"), {"x": "44", "y": _fmt(y), "class": "footer"})
            note_line.text = note
            y += 16


def _svg(tag: str) -> str:
    return f"{{{SVG_NS}}}{tag}"


def _fmt(value: float) -> str:
    return f"{value:.1f}"
