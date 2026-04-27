from __future__ import annotations

import hashlib
import random
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import NativeSvgRenderer, build_timing_scene, sample_native_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def test_clean_native_sampler_varies_style_axes_over_many_seeds() -> None:
    scene = _scene()
    renderer = NativeSvgRenderer()
    specs = [sample_native_render_spec(random.Random(seed), profile="clean-native") for seed in range(50)]
    hashes = {
        hashlib.sha256("".join((renderer.render(scene, spec).svg_text or "").split()).encode("utf-8")).hexdigest()
        for spec in specs
    }

    assert len(hashes) == 50
    assert len({spec.style.primary_font.family for spec in specs}) >= 5
    assert len({round(spec.layout.lane_height, 3) for spec in specs}) > 45
    assert len({spec.style.transition_shape for spec in specs}) > 1
    assert len({spec.style.color_mode for spec in specs}) > 1
    assert len({round(spec.layout.tick_width, 3) for spec in specs}) > 45


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual)
