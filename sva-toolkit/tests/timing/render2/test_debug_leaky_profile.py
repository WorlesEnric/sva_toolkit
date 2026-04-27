from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import AnnotationPolicy, WaveDromAdapter, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.render2.audit.leakage import audit_rendered_text
from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_debug_current_profile_preserves_legacy_leaky_overlay_text() -> None:
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    scene = build_timing_scene(visual, semantic_document=document)

    result = WaveDromAdapter().render(scene, _spec())
    report = audit_rendered_text(scene, result, target_dsl_text=emit_timing_dsl(document))

    assert "req_rise" in (result.svg_text or "")
    assert "[1:4]" in (result.svg_text or "")
    assert "RULES" in (result.svg_text or "")
    assert not report.passed


def _spec():
    spec = sample_native_render_spec(random.Random(71), profile="clean-native")
    return replace(
        spec,
        renderer_id="wavedrom",
        profile="debug-current",
        annotations=replace(spec.annotations, policy=AnnotationPolicy.GEOMETRIC_GUIDES, semantic_guides_enabled=True),
    )
