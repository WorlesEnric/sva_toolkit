from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import AnnotationPolicy, WaveDromAdapter, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.render2.audit.target_visibility import audit_target_visibility
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def test_target_visibility_passes_when_all_signals_are_labeled() -> None:
    scene = _scene()
    result = WaveDromAdapter().render(scene, _spec())

    report = audit_target_visibility(scene, result)

    assert report.passed
    assert not report.missing_signals


def test_target_visibility_fails_when_signal_label_is_missing() -> None:
    scene = _scene()
    result = WaveDromAdapter().render(scene, _spec())
    visibility = replace(
        result.visibility,
        rendered_text=tuple(text for text in result.visibility.rendered_text if text.text != "ack"),
    )
    result = replace(result, visibility=visibility)

    report = audit_target_visibility(scene, result)

    assert not report.passed
    assert report.reasons == ("target_not_visible",)
    assert "ack" in report.missing_signals


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)


def _spec():
    spec = sample_native_render_spec(random.Random(41), profile="clean-native")
    return replace(
        spec,
        renderer_id="wavedrom",
        profile="clean-wavedrom",
        annotations=replace(spec.annotations, policy=AnnotationPolicy.NONE),
    )
