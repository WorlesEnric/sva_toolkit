from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import (
    AnnotationPolicy,
    DecorationKind,
    LaneType,
    NativeSvgRenderer,
    build_timing_scene,
    sample_native_render_spec,
)
from sva_toolkit.timing.visual import VisibilityClass, lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def test_policy_none_does_not_render_anchor_names_or_extra_target_tokens() -> None:
    scene = _scene()
    spec = _with_policy(sample_native_render_spec(random.Random(5), profile="clean-native"), AnnotationPolicy.NONE)

    result = NativeSvgRenderer().render(scene, spec)
    rendered = {text.text for text in result.visibility.rendered_text}
    allowed = {lane.name for lane in scene.lanes}
    allowed.update(
        str(run.value)
        for lane in scene.lanes
        if lane.lane_type == LaneType.BUS
        for run in lane.runs
        if str(run.value).lower() not in {"x", "z"}
    )

    assert {event.name for event in scene.events}.isdisjoint(rendered)
    assert result.visibility.target_tokens_visible <= allowed


def test_natural_measurements_render_visible_bound_bracket_text() -> None:
    scene = _scene()
    spec = _with_policy(
        sample_native_render_spec(random.Random(8), profile="datasheet-native"),
        AnnotationPolicy.NATURAL_MEASUREMENTS,
    )

    result = NativeSvgRenderer().render(scene, spec)
    visible_bound_texts = {
        decoration.text
        for decoration in scene.decorations
        if decoration.kind == DecorationKind.MEASUREMENT_BRACKET
        and decoration.visibility_class == VisibilityClass.VISIBLE_TEXT
        and decoration.text
    }

    assert visible_bound_texts
    assert visible_bound_texts & {text.text for text in result.visibility.rendered_text}


def test_debug_leaky_policy_can_render_anchor_names_for_inspection() -> None:
    scene = _scene()
    spec = _with_policy(sample_native_render_spec(random.Random(9), profile="clean-native"), AnnotationPolicy.DEBUG_LEAKY)

    result = NativeSvgRenderer().render(scene, spec)
    rendered = {text.text for text in result.visibility.rendered_text}

    assert {event.name for event in scene.events} <= rendered
    assert {event.name for event in scene.events} <= result.visibility.debug_overlay_tokens


def _with_policy(spec, policy: AnnotationPolicy):
    return replace(
        spec,
        annotations=replace(
            spec.annotations,
            policy=policy,
            semantic_guides_enabled=True,
            nuisance_text_count=max(1, spec.annotations.nuisance_text_count),
        ),
    )


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)
