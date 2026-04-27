from __future__ import annotations

import random

from sva_toolkit.timing.render2 import (
    BBox,
    DiagramLayout,
    LaneScene,
    LaneType,
    Point,
    RenderResult,
    SampleRun,
    TextPrimitive,
    TickModel,
    TimingScene,
    VisualVisibilityReport,
    sample_native_render_spec,
)
from sva_toolkit.timing.render2.audit.layout_overflow import audit_layout_overflow


def test_layout_overflow_passes_when_all_bboxes_fit() -> None:
    scene = _scene()
    result = _result(BBox(1, 2, 30, 20), width=100, height=80)

    report = audit_layout_overflow(scene, result)

    assert report.passed
    assert report.reason is None


def test_layout_overflow_fails_when_bbox_falls_outside_canvas() -> None:
    scene = _scene()
    result = _result(BBox(101, 0, 5, 5), width=100, height=80)

    report = audit_layout_overflow(scene, result)

    assert not report.passed
    assert report.reason == "layout_overflow"


def _result(box: BBox, *, width: float, height: float) -> RenderResult:
    spec = sample_native_render_spec(random.Random(1), profile="clean-native")
    text = TextPrimitive("clk", box, "lane_label", "visible_text")
    return RenderResult(
        svg_text="<svg />",
        png_bytes=None,
        layout=DiagramLayout(
            width=width,
            height=height,
            plot_origin=Point(0, 0),
            tick_width=10,
            lane_height=10,
            lane_pitch=12,
            bbox_by_role={"lane_label": (box,)},
        ),
        visibility=VisualVisibilityReport(
            rendered_text=(text,),
            target_tokens_visible=frozenset({"clk"}),
            nuisance_tokens=frozenset(),
            debug_overlay_tokens=frozenset(),
            leaked_tokens=frozenset(),
            occluded_lane_fractions={"clk": 0.0},
            minimum_contrast=1.0,
        ),
        render_spec=spec,
    )


def _scene() -> TimingScene:
    return TimingScene(
        name="overflow",
        clocking_edge="posedge",
        clocking_signal="clk",
        lanes=(
            LaneScene(
                name="clk",
                lane_type=LaneType.CLOCK,
                runs=(SampleRun(0, 0, "1"), SampleRun(1, 1, "0")),
            ),
        ),
        ticks=TickModel(total_ticks=2),
        cuts=(),
        events=(),
        constraints=(),
    )
