from __future__ import annotations

import pytest

from sva_toolkit.timing.render2 import PlantUMLAdapter
from sva_toolkit.timing.render2.adapters.plantuml import dependency_status

from ._helpers import adapter_spec, assert_basic_render_result, assert_leakage_passes, document_and_scene, high_z_scene


pytestmark = pytest.mark.skipif(dependency_status() is not None, reason=dependency_status() or "plantuml missing")


def test_plantuml_adapter_renders_png() -> None:
    _document, scene = document_and_scene()
    spec = adapter_spec("plantuml")
    adapter = PlantUMLAdapter()

    assert adapter.supports(scene, spec)
    result = adapter.render(scene, spec)

    assert result.png_bytes
    assert_basic_render_result(result)
    assert_leakage_passes(scene, result)


def test_plantuml_adapter_rejects_high_z_scene() -> None:
    adapter = PlantUMLAdapter()

    assert not adapter.supports(high_z_scene(), adapter_spec("plantuml"))
