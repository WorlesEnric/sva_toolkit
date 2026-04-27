from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import AnnotationPolicy, build_timing_scene, render, sample_native_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_pipeline_render_returns_passing_outcome_for_clean_wavedrom() -> None:
    scene = _scene()

    outcome = render(scene, _spec(profile="clean-wavedrom", policy=AnnotationPolicy.GEOMETRIC_GUIDES))

    assert outcome.audits_passed
    assert outcome.rejection_reason is None
    assert outcome.leakage is not None
    assert outcome.target_visibility is not None
    assert outcome.contrast is not None
    assert outcome.occlusion is not None


def test_pipeline_can_disable_audit_enforcement_for_debug_leaks() -> None:
    scene = _scene()

    outcome = render(
        scene,
        _spec(profile="debug-current", policy=AnnotationPolicy.GEOMETRIC_GUIDES),
        enforce_audits=False,
    )

    assert outcome.audits_passed
    assert outcome.rejection_reason is None
    assert outcome.leakage is not None
    assert not outcome.leakage.passed


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)


def _spec(*, profile: str, policy: AnnotationPolicy):
    spec = sample_native_render_spec(random.Random(61), profile="clean-native")
    return replace(
        spec,
        renderer_id="wavedrom",
        profile=profile,
        annotations=replace(spec.annotations, policy=policy, semantic_guides_enabled=True, nuisance_text_count=0),
    )
