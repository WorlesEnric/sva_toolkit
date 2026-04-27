from __future__ import annotations

import random
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import NativeSvgRenderer, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def test_same_scene_and_spec_render_identical_svg() -> None:
    scene = _scene()
    spec = sample_native_render_spec(random.Random(31), profile="clean-native")
    renderer = NativeSvgRenderer()

    assert renderer.render(scene, spec).svg_text == renderer.render(scene, spec).svg_text


def test_different_sampler_seeds_render_distinct_svgs() -> None:
    scene = _scene()
    renderer = NativeSvgRenderer()
    svgs = {
        renderer.render(scene, sample_native_render_spec(random.Random(seed), profile="clean-native")).svg_text
        for seed in range(10)
    }

    assert len(svgs) >= 9


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual)
