from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import AnnotationPolicy, NativeSvgRenderer, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def test_geometric_guides_emit_event_vertical_lines_without_labels_or_nuisance_text() -> None:
    scene = _scene()
    spec = _with_policy(sample_native_render_spec(random.Random(20), profile="clean-native"), AnnotationPolicy.GEOMETRIC_GUIDES)

    result = NativeSvgRenderer().render(scene, spec)
    guide_boxes = result.layout.bbox_by_role.get("vertical_helper_line", ())
    guide_centers = {round(box.x + box.width / 2, 1) for box in guide_boxes}
    event_xs = {round(result.layout.plot_origin.x + event.tick * result.layout.tick_width, 1) for event in scene.events}

    assert "nuisance_text" not in result.layout.bbox_by_role
    assert guide_boxes
    assert guide_centers & event_xs
    assert {event.name for event in scene.events}.isdisjoint({text.text for text in result.visibility.rendered_text})
    assert all(fraction <= 0.15 for fraction in result.visibility.occluded_lane_fractions.values())


def test_nuisance_only_emits_nuisance_text_without_vertical_helper_lines() -> None:
    scene = _scene()
    spec = _with_policy(sample_native_render_spec(random.Random(21), profile="clean-native"), AnnotationPolicy.NUISANCE_ONLY)

    result = NativeSvgRenderer().render(scene, spec)

    assert result.layout.bbox_by_role.get("nuisance_text")
    assert "vertical_helper_line" not in result.layout.bbox_by_role
    assert result.visibility.nuisance_tokens
    assert all(fraction <= 0.15 for fraction in result.visibility.occluded_lane_fractions.values())


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
    return build_timing_scene(visual)
