"""Compression and re-encoding degradation operations."""

from __future__ import annotations

from io import BytesIO
import importlib
from typing import Any

from sva_toolkit.timing.render2.spec import DegradationSpec


def reencode_image(image: Any, spec: DegradationSpec, _rng: Any, *, image_format: str = "jpeg") -> Any:
    image_module = _require_image_module()
    output = BytesIO()
    normalized = image_format.lower()
    if normalized in {"jpg", "jpeg"}:
        image.convert("RGB").save(output, format="JPEG", quality=int(spec.jpeg_quality), optimize=False)
    elif normalized == "webp":
        image.convert("RGB").save(output, format="WEBP", quality=int(spec.jpeg_quality), method=0)
    elif normalized == "png":
        image.save(output, format="PNG", optimize=False)
    else:
        raise ValueError(f"unsupported compression format: {image_format}")
    output.seek(0)
    result = image_module.open(output)
    result.load()
    return result.convert("RGB")


def _require_image_module() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for compression degradation") from exc


__all__ = ["reencode_image"]
