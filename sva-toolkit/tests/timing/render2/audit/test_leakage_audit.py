from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import (
    AnnotationPolicy,
    BBox,
    DiagramLayout,
    NativeSvgRenderer,
    Point,
    RenderResult,
    TextPrimitive,
    VisualVisibilityReport,
    WaveDromAdapter,
    build_timing_scene,
    sample_native_render_spec,
)
from sva_toolkit.timing.render2.audit.leakage import audit_rendered_text
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def test_clean_wavedrom_render_passes_leakage_audit() -> None:
    document, scene = _document_and_scene()
    result = WaveDromAdapter().render(scene, _wavedrom_spec("clean-wavedrom", AnnotationPolicy.GEOMETRIC_GUIDES))

    report = audit_rendered_text(scene, result, target_dsl_text=emit_timing_dsl(document))

    assert report.passed
    assert not report.leaked_tokens


def test_debug_wavedrom_render_fails_leakage_audit() -> None:
    document, scene = _document_and_scene()
    result = WaveDromAdapter().render(scene, _wavedrom_spec("debug-current", AnnotationPolicy.GEOMETRIC_GUIDES))

    report = audit_rendered_text(scene, result, target_dsl_text=emit_timing_dsl(document))

    assert not report.passed
    assert report.debug_overlay_tokens


def test_native_none_policy_passes_leakage_audit() -> None:
    document, scene = _document_and_scene()
    spec = sample_native_render_spec(random.Random(31), profile="clean-native")
    spec = replace(spec, annotations=replace(spec.annotations, policy=AnnotationPolicy.NONE))
    result = NativeSvgRenderer().render(scene, spec)

    report = audit_rendered_text(scene, result, target_dsl_text=emit_timing_dsl(document))

    assert report.passed


def test_audit_catches_handcrafted_anchor_text_leak() -> None:
    _document, scene = _document_and_scene()
    spec = _wavedrom_spec("clean-wavedrom", AnnotationPolicy.NONE)
    result = RenderResult(
        svg_text="<svg><text>a0</text></svg>",
        png_bytes=None,
        layout=DiagramLayout(width=100, height=50, plot_origin=Point(0, 0), tick_width=10, lane_height=10, lane_pitch=12),
        visibility=VisualVisibilityReport(
            rendered_text=(TextPrimitive("a0", BBox(0, 0, 10, 10), "debug_overlay", "visible_text"),),
            target_tokens_visible=frozenset(),
            nuisance_tokens=frozenset(),
            debug_overlay_tokens=frozenset(),
            leaked_tokens=frozenset(),
            occluded_lane_fractions={},
            minimum_contrast=1.0,
        ),
        render_spec=spec,
    )

    report = audit_rendered_text(scene, result, target_dsl_text=emit_timing_dsl(scene.visible_target))

    assert not report.passed
    assert "a0" in report.leaked_tokens


def _document_and_scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return document, build_timing_scene(visual, semantic_document=document)


def _wavedrom_spec(profile: str, policy: AnnotationPolicy):
    spec = sample_native_render_spec(random.Random(29), profile="clean-native")
    return replace(
        spec,
        renderer_id="wavedrom",
        profile=profile,
        annotations=replace(spec.annotations, policy=policy, semantic_guides_enabled=True, nuisance_text_count=0),
    )
