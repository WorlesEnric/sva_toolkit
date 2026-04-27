from __future__ import annotations

import builtins

import pytest

from sva_toolkit.timing.render2 import RasterSpec
from sva_toolkit.timing.render2.rasterize import rasterize_svg


def test_rasterize_svg_scales_with_dpi() -> None:
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="48"><rect width="96" height="48"/></svg>'

    image_96 = rasterize_svg(svg, RasterSpec(dpi=96, antialias=True, output_format="png"))
    image_300 = rasterize_svg(svg, RasterSpec(dpi=300, antialias=True, output_format="png"))

    assert image_96.size == (96, 48)
    assert image_300.size == (300, 150)


def test_rasterize_svg_falls_back_when_rasterizers_are_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__
    blocked = {"cairosvg", "resvg_py", "wand", "wand.image"}

    def _missing_rasterizers(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name in blocked:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_rasterizers)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10"><rect width="20" height="10"/></svg>'

    with pytest.warns(RuntimeWarning, match="falling back to synthetic SVG raster"):
        image = rasterize_svg(svg, RasterSpec(dpi=96, antialias=True, output_format="png"))

    assert image.size == (20, 10)
    assert image.mode == "RGBA"
