"""Morphological degradation operations for strokes and linework."""

from __future__ import annotations

import importlib
from typing import Any

from sva_toolkit.timing.render2.spec import DegradationSpec


def apply_morphology(image: Any, spec: DegradationSpec, rng: Any) -> Any:
    image_filter = _require_image_filter()
    mode = spec.morphology
    if mode == "none" and spec.family == "fax":
        mode = "thin"
    result = image.convert("RGB")
    if mode in {"thin", "thinning"}:
        result = result.filter(image_filter.MaxFilter(size=3))
    elif mode in {"thick", "thicken", "thickening"}:
        result = result.filter(image_filter.MinFilter(size=3))
    elif mode in {"broken", "break", "gaps"}:
        result = broken_strokes(result, spec, rng)
    if spec.family == "fax" or mode in {"broken", "break", "gaps"}:
        result = broken_strokes(result, spec, rng)
    return result


def broken_strokes(image: Any, _spec: DegradationSpec, rng: Any) -> Any:
    image_draw = _require_image_draw()
    result = image.convert("RGB")
    draw = image_draw.Draw(result)
    count = max(6, result.width * result.height // 24000)
    for _ in range(count):
        x = rng.randrange(0, result.width)
        y = rng.randrange(0, result.height)
        width = rng.randrange(2, 7)
        height = rng.randrange(1, 4)
        draw.rectangle((x, y, x + width, y + height), fill=(255, 255, 255))
    return result


def _require_image_filter() -> Any:
    try:
        return importlib.import_module("PIL.ImageFilter")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for morphology degradation") from exc


def _require_image_draw() -> Any:
    try:
        return importlib.import_module("PIL.ImageDraw")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for morphology degradation") from exc


__all__ = ["apply_morphology", "broken_strokes"]
