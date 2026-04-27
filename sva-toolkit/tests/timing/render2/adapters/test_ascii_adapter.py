from __future__ import annotations

from sva_toolkit.timing.render2 import ASCIIAdapter

from ._helpers import adapter_spec, assert_basic_render_result, assert_leakage_passes, document_and_scene, high_z_scene


def test_ascii_adapter_renders_text_and_optional_png() -> None:
    document, scene = document_and_scene()
    spec = adapter_spec("ascii")
    adapter = ASCIIAdapter()

    assert adapter.supports(scene, spec)
    result = adapter.render(scene, spec)

    assert document.name == "simple_handshake"
    assert result.ascii_text
    assert "req" in result.ascii_text
    assert_basic_render_result(result)
    assert_leakage_passes(scene, result)


def test_ascii_adapter_rejects_high_z_scene() -> None:
    adapter = ASCIIAdapter()

    assert not adapter.supports(high_z_scene(), adapter_spec("ascii"))
