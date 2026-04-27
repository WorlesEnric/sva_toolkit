from __future__ import annotations

import random
from dataclasses import replace

from sva_toolkit.timing.render2 import (
    AnnotationPolicy,
    LaneScene,
    LaneType,
    NativeSvgRenderer,
    SampleRun,
    Stroke,
    TickModel,
    TimingScene,
    sample_native_render_spec,
)
from sva_toolkit.timing.render2.audit.contrast import audit_minimum_contrast, audit_occlusion


def test_white_stroke_on_white_background_fails_low_contrast() -> None:
    scene = _scene()
    spec = sample_native_render_spec(random.Random(51), profile="clean-native")
    spec = replace(
        spec,
        style=replace(
            spec.style,
            palette=("#ffffff", "#ffffff", "#ffffff", "#ffffff", "#ffffff"),
            waveform_stroke=Stroke(color="#ffffff", width=1.0),
        ),
        annotations=replace(spec.annotations, policy=AnnotationPolicy.NONE),
    )
    result = NativeSvgRenderer().render(scene, spec)

    report = audit_minimum_contrast(result)

    assert not report.passed
    assert report.reasons == ("low_contrast",)


def test_large_lane_occlusion_fails_occlusion_audit() -> None:
    scene = _scene()
    spec = sample_native_render_spec(random.Random(52), profile="clean-native")
    result = NativeSvgRenderer().render(scene, spec)
    visibility = replace(result.visibility, occluded_lane_fractions={"req": 0.5})
    result = replace(result, visibility=visibility)

    report = audit_occlusion(result)

    assert not report.passed
    assert report.reasons == ("required_bus_value_occluded",)


def _scene() -> TimingScene:
    return TimingScene(
        name="contrast",
        clocking_edge="posedge",
        clocking_signal="clk",
        lanes=(
            LaneScene("clk", LaneType.CLOCK, (SampleRun(0, 2, "1"),)),
            LaneScene("req", LaneType.BIT, (SampleRun(0, 0, "0"), SampleRun(1, 2, "1"))),
        ),
        ticks=TickModel(total_ticks=3),
        cuts=(),
        events=(),
        constraints=(),
    )
