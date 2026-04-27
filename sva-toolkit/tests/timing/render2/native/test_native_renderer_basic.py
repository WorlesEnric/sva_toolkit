from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import DEFAULT_REGISTRY, NativeSvgRenderer, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"
SVG_NS = "{http://www.w3.org/2000/svg}"


def test_native_renderer_renders_handshake_without_semantic_names(tmp_path: Path) -> None:
    scene = _scene("01_simple_handshake.td")
    spec = sample_native_render_spec(random.Random(11), profile="clean-native")

    result = NativeSvgRenderer().render(scene, spec)
    svg_path = tmp_path / "native_handshake.svg"
    svg_path.write_text(result.svg_text or "", encoding="utf-8")
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))

    assert root.tag == f"{SVG_NS}svg"
    assert "req" in result.svg_text
    assert "ack" in result.svg_text
    assert "req_rise" not in result.svg_text
    assert "ack_rise" not in result.svg_text
    assert "handshake" not in result.svg_text
    assert result.layout.width > 0
    assert result.layout.bbox_by_role["lane_label"]
    assert len(root.findall(f".//{SVG_NS}text")) == len(result.visibility.rendered_text)


def test_native_renderer_is_registered() -> None:
    assert isinstance(DEFAULT_REGISTRY.get("native_svg"), NativeSvgRenderer)


def _scene(name: str):
    document = parse_diagram((EXAMPLES_DIR / name).read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)
