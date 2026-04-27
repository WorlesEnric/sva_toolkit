from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import NativeSvgRenderer, build_timing_scene
from sva_toolkit.timing.render2.audit.reproducibility import audit_renderer_reproducibility
from sva_toolkit.timing.render2.profiles import PROFILE_NATIVE_RANDOM
from sva_toolkit.timing.render2.spec_sampler import sample_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def test_native_renderer_reproducibility_audit_passes() -> None:
    scene = _scene()
    spec = sample_render_spec(random.Random(51), profile=PROFILE_NATIVE_RANDOM, scene=scene)

    report = audit_renderer_reproducibility(NativeSvgRenderer(), scene, spec)

    assert report.passed
    assert report.reason is None


def test_reproducibility_audit_fails_when_adapter_output_changes() -> None:
    scene = _scene()
    spec = sample_render_spec(random.Random(52), profile=PROFILE_NATIVE_RANDOM, scene=scene)
    base = NativeSvgRenderer().render(scene, spec)
    adapter = _ChangingAdapter(base)

    report = audit_renderer_reproducibility(adapter, scene, spec)

    assert not report.passed
    assert report.reason == "non_reproducible_output"


class _ChangingAdapter:
    id = "native_svg"

    def __init__(self, base):
        self._base = base
        self._count = 0

    def render(self, scene, spec):
        del scene, spec
        self._count += 1
        return replace(self._base, svg_text=f"<svg>{self._count}</svg>")


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)
