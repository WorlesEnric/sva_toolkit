from __future__ import annotations

from io import BytesIO
import random

import pytest

from sva_toolkit.timing.render2 import PageSpec, RasterSpec
from sva_toolkit.timing.render2.page_composer import NUISANCE_TEXTS, PageComposer


@pytest.mark.parametrize("crop_mode", ["tight", "loose", "off_center", "fragment"])
def test_page_composer_records_diagram_bbox_for_each_crop_mode(crop_mode: str) -> None:
    composer = PageComposer(_page(crop_mode), RasterSpec(dpi=96, antialias=True, output_format="png"))

    image, metadata = composer.compose(_diagram(), rng=random.Random(17))

    diagram_bbox = metadata["diagram_bbox"]
    assert image.width > 0
    assert image.height > 0
    assert diagram_bbox.width > 0
    assert diagram_bbox.height > 0
    assert any(element["role"] == "diagram" for element in metadata["elements"])


def test_page_composer_records_nuisance_text_for_leakage_audit() -> None:
    composer = PageComposer(_page("loose"), RasterSpec(dpi=96, antialias=True, output_format="png"))

    _image, metadata = composer.compose(_diagram(), rng=random.Random(5))

    nuisance = metadata["nuisance_text"]
    assert nuisance
    assert set(nuisance) <= set(NUISANCE_TEXTS)
    text_elements = {element["text"] for element in metadata["elements"] if "text" in element}
    assert set(nuisance) <= text_elements


def test_page_composer_is_deterministic_for_same_rng_seed() -> None:
    composer = PageComposer(_page("off_center"), RasterSpec(dpi=96, antialias=True, output_format="png"))

    first, _first_metadata = composer.compose(_diagram(), rng=random.Random(23))
    second, _second_metadata = composer.compose(_diagram(), rng=random.Random(23))

    assert _png_bytes(first) == _png_bytes(second)


def _page(crop_mode: str) -> PageSpec:
    return PageSpec(
        enabled=True,
        caption_above=True,
        caption_below=True,
        surrounding_paragraph=True,
        table_border=True,
        page_header=True,
        page_footer=True,
        crop_mode=crop_mode,
    )


def _diagram():
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (180, 90), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((8, 16), "req", fill=(0, 0, 0, 255))
    draw.line((44, 20, 160, 20), fill=(0, 0, 0, 255), width=2)
    draw.line((44, 58, 160, 58), fill=(0, 0, 0, 255), width=2)
    return image


def _png_bytes(image) -> bytes:  # type: ignore[no-untyped-def]
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
