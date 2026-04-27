from __future__ import annotations

from io import BytesIO
import random

import pytest

from sva_toolkit.timing.render2 import DegradationSpec
from sva_toolkit.timing.render2.degrade.pipeline import DegradationPipeline


@pytest.mark.parametrize("family", ["clean", "scan", "photocopy", "camera", "fax", "grayscale", "threshold", "noise"])
def test_degradation_pipeline_runs_each_family(family: str) -> None:
    image = _test_image()
    spec = _spec(family)

    output, chain = DegradationPipeline(spec).apply(image, rng=random.Random(19))

    assert output.width > 0
    assert output.height > 0
    assert chain


def test_degradation_pipeline_is_deterministic() -> None:
    image = _test_image()
    spec = _spec("scan")

    first, first_chain = DegradationPipeline(spec).apply(image, rng=random.Random(101))
    second, second_chain = DegradationPipeline(spec).apply(image, rng=random.Random(101))

    assert first_chain == second_chain
    assert _png_bytes(first) == _png_bytes(second)


def _spec(family: str) -> DegradationSpec:
    return DegradationSpec(
        family=family,
        blur_sigma=0.2,
        noise_sigma=0.02,
        contrast=0.92,
        brightness=1.04,
        rotation_deg=0.6 if family == "camera" else 0.0,
        perspective=0.015 if family == "camera" else 0.0,
        jpeg_quality=88,
        morphology="thin" if family == "fax" else "none",
    )


def _test_image():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (160, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "req", fill="black")
    draw.line((12, 42, 148, 42), fill="black", width=2)
    draw.rectangle((32, 54, 124, 72), outline="black", width=2)
    return image


def _png_bytes(image) -> bytes:  # type: ignore[no-untyped-def]
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
