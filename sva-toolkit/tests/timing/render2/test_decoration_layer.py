from __future__ import annotations

import random
from dataclasses import replace

import pytest

from sva_toolkit.timing.render2 import (
    AnnotationPolicy,
    Decoration,
    DecorationKind,
    DiagramLayout,
    LaneScene,
    LaneType,
    Point,
    SampleRun,
    TickModel,
    TimingScene,
    sample_native_render_spec,
)
from sva_toolkit.timing.render2.decoration_layer import render_decorations
from sva_toolkit.timing.render2.primitives import Text
from sva_toolkit.timing.visual import VisibilityClass


def test_decoration_layer_respects_annotation_policies() -> None:
    scene = _scene()
    layout = _layout()

    by_policy = {
        policy: render_decorations(scene, _spec(policy), layout)
        for policy in (
            AnnotationPolicy.NONE,
            AnnotationPolicy.NUISANCE_ONLY,
            AnnotationPolicy.GEOMETRIC_GUIDES,
            AnnotationPolicy.NATURAL_MEASUREMENTS,
            AnnotationPolicy.DEBUG_LEAKY,
        )
    }

    assert by_policy[AnnotationPolicy.NONE] == ()
    assert {primitive.role for primitive in by_policy[AnnotationPolicy.NUISANCE_ONLY]} <= {
        "nuisance_text",
        "hold_highlight",
    }
    assert "vertical_helper_line" in {primitive.role for primitive in by_policy[AnnotationPolicy.GEOMETRIC_GUIDES]}
    assert "measurement_bracket" in {primitive.role for primitive in by_policy[AnnotationPolicy.NATURAL_MEASUREMENTS]}
    assert "caption_text" in {primitive.role for primitive in by_policy[AnnotationPolicy.NATURAL_MEASUREMENTS]}
    assert "measurement_bracket" in {primitive.role for primitive in by_policy[AnnotationPolicy.DEBUG_LEAKY]}


def test_measurement_bracket_hidden_bound_emits_no_text() -> None:
    scene = replace(
        _scene(),
        decorations=(
            Decoration(
                kind=DecorationKind.MEASUREMENT_BRACKET,
                semantic=True,
                target_ref=None,
                text="MAX_LAT",
                visibility_class=VisibilityClass.HIDDEN_SEMANTIC,
                span=(1, 3),
            ),
        ),
    )

    primitives = render_decorations(scene, _spec(AnnotationPolicy.NATURAL_MEASUREMENTS), _layout())

    assert all(not isinstance(primitive, Text) for primitive in primitives)


def test_missing_decoration_target_ref_warns_and_drops() -> None:
    scene = replace(
        _scene(),
        decorations=(
            Decoration(
                kind=DecorationKind.VERTICAL_GUIDE,
                semantic=True,
                target_ref="anchor:missing",
                anchor_tick=1,
            ),
        ),
    )

    with pytest.warns(RuntimeWarning):
        primitives = render_decorations(scene, _spec(AnnotationPolicy.GEOMETRIC_GUIDES), _layout())

    assert primitives == ()


def _scene() -> TimingScene:
    return TimingScene(
        name="decor",
        clocking_edge="posedge",
        clocking_signal="clk",
        lanes=(
            LaneScene("clk", LaneType.CLOCK, (SampleRun(0, 1, "1"),)),
            LaneScene("req", LaneType.BIT, (SampleRun(0, 0, "0"), SampleRun(1, 3, "1"))),
        ),
        ticks=TickModel(total_ticks=4),
        cuts=(),
        events=(),
        constraints=(),
        decorations=(
            Decoration(DecorationKind.VERTICAL_GUIDE, True, None, anchor_tick=1),
            Decoration(
                DecorationKind.MEASUREMENT_BRACKET,
                True,
                None,
                text="1-3 cycles",
                visibility_class=VisibilityClass.VISIBLE_TEXT,
                span=(1, 3),
            ),
            Decoration(DecorationKind.HIGHLIGHT_REGION, False, None, span=(1, 2), lane_names=("req",)),
            Decoration(
                DecorationKind.CALLOUT_ARROW,
                True,
                None,
                text="tSU",
                visibility_class=VisibilityClass.VISIBLE_TEXT,
                anchor_tick=2,
            ),
            Decoration(DecorationKind.NUISANCE_TEXT, False, None, text="note 1"),
            Decoration(DecorationKind.CAPTION, True, None, text="Figure 2"),
        ),
    )


def _layout() -> DiagramLayout:
    return DiagramLayout(width=260, height=120, plot_origin=Point(60, 30), tick_width=40, lane_height=20, lane_pitch=28)


def _spec(policy: AnnotationPolicy):
    spec = sample_native_render_spec(random.Random(23), profile="clean-native")
    return replace(spec, annotations=replace(spec.annotations, policy=policy, semantic_guides_enabled=True))
