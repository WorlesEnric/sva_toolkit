"""Color and tone degradation operations."""

from __future__ import annotations

import importlib
from typing import Any

from sva_toolkit.timing.render2.spec import DegradationSpec


def adjust_contrast_brightness(image: Any, spec: DegradationSpec, _rng: Any) -> Any:
    _image_ops, image_enhance = _require_pillow_modules()
    result = image.convert("RGB")
    if spec.contrast != 1.0:
        result = image_enhance.Contrast(result).enhance(float(spec.contrast))
    if spec.brightness != 1.0:
        result = image_enhance.Brightness(result).enhance(float(spec.brightness))
    return result


def grayscale(image: Any, _spec: DegradationSpec, _rng: Any) -> Any:
    image_ops, _image_enhance = _require_pillow_modules()
    return image_ops.grayscale(image).convert("RGB")


def monochrome_threshold(image: Any, _spec: DegradationSpec, rng: Any) -> Any:
    image_ops, _image_enhance = _require_pillow_modules()
    threshold = rng.randrange(132, 178)
    gray = image_ops.autocontrast(image_ops.grayscale(image))
    return gray.point(lambda value: 255 if value >= threshold else 0).convert("RGB")


def low_contrast(image: Any, spec: DegradationSpec, _rng: Any) -> Any:
    _image_ops, image_enhance = _require_pillow_modules()
    contrast = min(float(spec.contrast), 0.78) if spec.contrast != 1.0 else 0.78
    brightness = float(spec.brightness)
    result = image_enhance.Contrast(image.convert("RGB")).enhance(contrast)
    if brightness != 1.0:
        result = image_enhance.Brightness(result).enhance(brightness)
    return result


def inverted_dark_mode(image: Any, _spec: DegradationSpec, _rng: Any) -> Any:
    image_ops, _image_enhance = _require_pillow_modules()
    return image_ops.invert(image.convert("RGB"))


def _require_pillow_modules() -> tuple[Any, Any]:
    try:
        return importlib.import_module("PIL.ImageOps"), importlib.import_module("PIL.ImageEnhance")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for color degradation") from exc


__all__ = [
    "adjust_contrast_brightness",
    "grayscale",
    "inverted_dark_mode",
    "low_contrast",
    "monochrome_threshold",
]
