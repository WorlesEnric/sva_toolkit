from __future__ import annotations

import importlib.util

import pytest

from sva_toolkit.timing.render2 import UndulateAdapter

from ._helpers import adapter_spec, assert_basic_render_result, assert_leakage_passes, document_and_scene, high_z_scene


pytestmark = pytest.mark.skipif(importlib.util.find_spec("undulate") is None, reason="undulate is not installed")


def test_undulate_adapter_renders_svg() -> None:
    _document, scene = document_and_scene()
    spec = adapter_spec("undulate")
    adapter = UndulateAdapter()

    assert adapter.supports(scene, spec)
    result = adapter.render(scene, spec)

    assert result.svg_text and "<svg" in result.svg_text
    assert_basic_render_result(result)
    assert_leakage_passes(scene, result)


def test_undulate_adapter_rejects_high_z_scene() -> None:
    adapter = UndulateAdapter()

    assert not adapter.supports(high_z_scene(), adapter_spec("undulate"))
