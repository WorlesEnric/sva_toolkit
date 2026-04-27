"""Ordered document degradation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib.util
from typing import Any

from sva_toolkit.timing.render2.degrade.color_ops import (
    adjust_contrast_brightness,
    grayscale,
    inverted_dark_mode,
    low_contrast,
    monochrome_threshold,
)
from sva_toolkit.timing.render2.degrade.compression_ops import reencode_image
from sva_toolkit.timing.render2.degrade.document_ops import apply_document_effects
from sva_toolkit.timing.render2.degrade.geometry_ops import crop_image, perspective_warp, rotate_image
from sva_toolkit.timing.render2.degrade.morphology_ops import apply_morphology
from sva_toolkit.timing.render2.spec import DegradationSpec


ImageOperation = Callable[[Any, DegradationSpec, Any], Any]


@dataclass(frozen=True)
class DegradationOperation:
    name: str
    function: ImageOperation

    def __call__(self, image: Any, spec: DegradationSpec, rng: Any) -> Any:
        return self.function(image, spec, rng)


class DegradationPipeline:
    """Apply degradation operations in the canonical render2 order."""

    def __init__(self, spec: DegradationSpec):
        self.spec = spec
        self.warnings: tuple[str, ...] = ()

    def apply(self, image: Any, *, rng: Any) -> tuple[Any, tuple[str, ...]]:
        warnings: list[str] = []
        if self.spec.augraphy_pipeline and importlib.util.find_spec("augraphy") is None:
            warnings.append("augraphy requested but not installed; using native document degradation")

        result = image
        names: list[str] = []
        for operation in self._operations():
            result = operation(result, self.spec, rng)
            names.append(operation.name)
        if not names:
            names.append(f"{self.spec.family or 'clean'}.identity")
            result = image.copy()
        self.warnings = tuple(warnings)
        return result, tuple(names)

    def __call__(self, image: Any, spec: DegradationSpec | None = None, rng: Any | None = None) -> tuple[Any, tuple[str, ...]]:
        if spec is not None and spec != self.spec:
            return DegradationPipeline(spec).apply(image, rng=rng)
        if rng is None:
            raise TypeError("rng is required")
        return self.apply(image, rng=rng)

    def _operations(self) -> tuple[DegradationOperation, ...]:
        family = self.spec.family
        operations: list[DegradationOperation] = []

        if abs(self.spec.rotation_deg) > 0.001 or family == "camera":
            operations.append(DegradationOperation("geometry.rotation", rotate_image))
        if self.spec.perspective > 0.0 or family == "camera":
            operations.append(DegradationOperation("geometry.perspective", perspective_warp))

        if family in {"scan", "photocopy", "camera", "fax", "noise", "native"} or self._has_pixel_effects():
            operations.append(DegradationOperation(f"document.{family}", apply_document_effects))

        if family == "grayscale":
            operations.append(DegradationOperation("color.grayscale", grayscale))
        elif family in {"threshold", "fax"}:
            operations.append(DegradationOperation("color.threshold", monochrome_threshold))
        elif family in {"low_contrast", "photocopy"}:
            operations.append(DegradationOperation("color.low_contrast", low_contrast))
        elif family in {"inverted", "dark", "dark_mode"}:
            operations.append(DegradationOperation("color.inverted", inverted_dark_mode))
        elif self.spec.contrast != 1.0 or self.spec.brightness != 1.0:
            operations.append(DegradationOperation("color.contrast_brightness", adjust_contrast_brightness))

        if self.spec.morphology != "none" or family == "fax":
            operations.append(DegradationOperation("morphology.stroke", apply_morphology))

        if self.spec.jpeg_quality < 100 or family in {"scan", "photocopy", "camera", "fax", "noise"}:
            operations.append(DegradationOperation("compression.jpeg", _jpeg_reencode))

        if family == "camera":
            operations.append(DegradationOperation("crop_resize.camera_crop", crop_image))
        return tuple(operations)

    def _has_pixel_effects(self) -> bool:
        return self.spec.blur_sigma > 0.0 or self.spec.noise_sigma > 0.0


def _jpeg_reencode(image: Any, spec: DegradationSpec, rng: Any) -> Any:
    return reencode_image(image, spec, rng, image_format="jpeg")


__all__ = ["DegradationOperation", "DegradationPipeline"]
