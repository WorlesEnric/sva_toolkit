from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import AnnotationPolicy, WaveDromAdapter, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"
SVG_NS = "{http://www.w3.org/2000/svg}"


def test_wavedrom_adapter_clean_profile_has_no_target_leaking_overlay_text() -> None:
    scene = _scene()
    spec = _wavedrom_spec(profile="clean-wavedrom", policy=AnnotationPolicy.GEOMETRIC_GUIDES)

    result = WaveDromAdapter().render(scene, spec)
    root = ET.fromstring(result.svg_text or "")
    rendered = {text.text for text in result.visibility.rendered_text}

    assert root.tag == f"{SVG_NS}svg"
    assert "data-timing-" not in (result.svg_text or "")
    assert {"clk", "req", "ack"} <= rendered
    assert {"a0", "a1", "req_rise", "ack_rise", "[1:4]", "RULES"}.isdisjoint(rendered)
    assert "RULES" not in (result.svg_text or "")
    assert "req_rise" not in (result.svg_text or "")
    assert "handshake" not in (result.svg_text or "")


def test_wavedrom_adapter_debug_profile_keeps_legacy_overlays_for_inspection() -> None:
    scene = _scene()
    spec = _wavedrom_spec(profile="debug-current", policy=AnnotationPolicy.GEOMETRIC_GUIDES)

    result = WaveDromAdapter().render(scene, spec)

    assert "req_rise" in (result.svg_text or "")
    assert "RULES" in (result.svg_text or "")
    assert result.visibility.debug_overlay_tokens


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)


def _wavedrom_spec(*, profile: str, policy: AnnotationPolicy):
    spec = sample_native_render_spec(random.Random(17), profile="clean-native")
    return replace(
        spec,
        renderer_id="wavedrom",
        profile=profile,
        annotations=replace(spec.annotations, policy=policy, semantic_guides_enabled=True, nuisance_text_count=0),
    )
