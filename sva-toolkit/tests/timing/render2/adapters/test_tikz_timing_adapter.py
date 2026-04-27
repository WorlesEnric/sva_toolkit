from __future__ import annotations

import pytest

from sva_toolkit.timing.render2 import TikzTimingAdapter
from sva_toolkit.timing.render2.adapters.tikz_timing import dependency_status

from ._helpers import adapter_spec, assert_basic_render_result, assert_leakage_passes, document_and_scene, high_z_scene


pytestmark = pytest.mark.skipif(dependency_status() is not None, reason=dependency_status() or "tikz toolchain missing")


def test_tikz_timing_adapter_renders_png() -> None:
    _document, scene = document_and_scene()
    spec = adapter_spec("tikz_timing")
    adapter = TikzTimingAdapter()

    assert adapter.supports(scene, spec)
    result = adapter.render(scene, spec)

    assert result.png_bytes
    assert_basic_render_result(result)
    assert_leakage_passes(scene, result)


def test_tikz_timing_adapter_rejects_high_z_scene() -> None:
    adapter = TikzTimingAdapter()

    assert not adapter.supports(high_z_scene(), adapter_spec("tikz_timing"))
