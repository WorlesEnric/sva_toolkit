"""Public post-render composition entry point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
import importlib
from types import MappingProxyType
from typing import Any
import warnings as warnings_module

from sva_toolkit.timing.render2.degrade.pipeline import DegradationPipeline
from sva_toolkit.timing.render2.page_composer import PageComposer
from sva_toolkit.timing.render2.primitives import BBox
from sva_toolkit.timing.render2.rasterize import rasterize_svg
from sva_toolkit.timing.render2.result import RenderResult, TextPrimitive
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec


@dataclass(frozen=True)
class ComposedRecord:
    image_bytes: bytes
    image_format: str
    image_width: int
    image_height: int
    intermediate_layers: Mapping[str, bytes] = field(default_factory=dict)
    crop_box: BBox | None = None
    page_metadata: Mapping[str, Any] = field(default_factory=dict)
    degradation_chain: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intermediate_layers", MappingProxyType(dict(self.intermediate_layers)))
        object.__setattr__(self, "page_metadata", MappingProxyType(dict(self.page_metadata)))
        object.__setattr__(self, "degradation_chain", tuple(self.degradation_chain))
        object.__setattr__(self, "warnings", tuple(self.warnings))


def compose_record(
    scene: TimingScene,
    spec: RenderSpec,
    result: RenderResult,
    *,
    rng: Any,
) -> ComposedRecord:
    """Rasterize, compose, degrade, resize, and encode one rendered scene."""

    del scene
    record_warnings = list(result.warnings)
    intermediate_layers: dict[str, bytes] = {}

    if result.png_bytes is not None:
        image = _image_from_bytes(result.png_bytes)
    elif result.svg_text is not None:
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always", RuntimeWarning)
            image = rasterize_svg(result.svg_text, spec.raster)
        record_warnings.extend(str(item.message) for item in caught)
    else:
        raise ValueError("RenderResult must include svg_text or png_bytes")

    image.load()
    image = image.convert("RGBA")
    _snapshot(intermediate_layers, "rasterized", image, spec)

    raster_scale_x = image.width / max(1.0, result.layout.width)
    raster_scale_y = image.height / max(1.0, result.layout.height)
    page_image, page_metadata = PageComposer(spec.page, spec.raster).compose(image, rng=rng)
    page_metadata = dict(page_metadata)
    page_metadata["pre_degradation_size"] = (page_image.width, page_image.height)
    _snapshot(intermediate_layers, "page", page_image, spec)

    pipeline = DegradationPipeline(spec.degradation)
    degraded_image, degradation_chain = pipeline.apply(page_image, rng=rng)
    record_warnings.extend(pipeline.warnings)
    page_metadata["post_degradation_size"] = (degraded_image.width, degraded_image.height)
    _snapshot(intermediate_layers, "degraded", degraded_image, spec)

    final_image, final_scale_x, final_scale_y = _resize_to_target(degraded_image, spec)
    page_metadata["final_size"] = (final_image.width, final_image.height)
    page_metadata["final_resize_scale_x"] = final_scale_x
    page_metadata["final_resize_scale_y"] = final_scale_y
    page_metadata["layout_transform"] = _layout_transform(
        page_metadata,
        raster_scale_x=raster_scale_x,
        raster_scale_y=raster_scale_y,
        final_scale_x=final_scale_x,
        final_scale_y=final_scale_y,
    )

    record_warnings.extend(_recoverability_warnings(result.visibility.rendered_text, final_image, page_metadata))
    image_bytes, image_format, output_warnings = _save_image(final_image, spec)
    record_warnings.extend(output_warnings)

    crop_box = page_metadata.get("crop_box")
    return ComposedRecord(
        image_bytes=image_bytes,
        image_format=image_format,
        image_width=final_image.width,
        image_height=final_image.height,
        intermediate_layers=intermediate_layers,
        crop_box=crop_box if isinstance(crop_box, BBox) else None,
        page_metadata=page_metadata,
        degradation_chain=degradation_chain,
        warnings=tuple(record_warnings),
    )


def _layout_transform(
    page_metadata: Mapping[str, Any],
    *,
    raster_scale_x: float,
    raster_scale_y: float,
    final_scale_x: float,
    final_scale_y: float,
) -> Mapping[str, float]:
    diagram_bbox = page_metadata.get("diagram_bbox")
    if not isinstance(diagram_bbox, BBox):
        diagram_bbox = BBox(0.0, 0.0, 0.0, 0.0)
    pre_width, pre_height = _size_tuple(page_metadata.get("pre_degradation_size"))
    post_width, post_height = _size_tuple(page_metadata.get("post_degradation_size"))
    rotation_offset_x = max(0.0, (post_width - pre_width) / 2.0)
    rotation_offset_y = max(0.0, (post_height - pre_height) / 2.0)
    diagram_scale_x = float(page_metadata.get("diagram_scale_x", 1.0))
    diagram_scale_y = float(page_metadata.get("diagram_scale_y", 1.0))
    return MappingProxyType(
        {
            "scale_x": raster_scale_x * diagram_scale_x * final_scale_x,
            "scale_y": raster_scale_y * diagram_scale_y * final_scale_y,
            "offset_x": (diagram_bbox.x + rotation_offset_x) * final_scale_x,
            "offset_y": (diagram_bbox.y + rotation_offset_y) * final_scale_y,
        }
    )


def _recoverability_warnings(rendered_text: tuple[TextPrimitive, ...], image: Any, page_metadata: Mapping[str, Any]) -> tuple[str, ...]:
    transform = page_metadata.get("layout_transform")
    if not isinstance(transform, Mapping):
        return ()
    warnings: list[str] = []
    for text in rendered_text:
        if text.role != "lane_label":
            continue
        bbox = _map_bbox(text.bbox, transform)
        padded = _pad_bbox(bbox, 5.0, image.width, image.height)
        if padded.width <= 0 or padded.height <= 0:
            warnings.append(f"lane label '{text.text}' maps outside final image")
            continue
        region = image.crop(_ltrb_from_bbox(padded)).convert("L")
        extrema = region.getextrema()
        if extrema[0] > 245:
            warnings.append(f"lane label '{text.text}' has low recoverability after degradation")
    return tuple(warnings)


def _map_bbox(bbox: BBox, transform: Mapping[str, float]) -> BBox:
    return BBox(
        x=bbox.x * float(transform["scale_x"]) + float(transform["offset_x"]),
        y=bbox.y * float(transform["scale_y"]) + float(transform["offset_y"]),
        width=bbox.width * float(transform["scale_x"]),
        height=bbox.height * float(transform["scale_y"]),
    )


def _pad_bbox(bbox: BBox, padding: float, width: int, height: int) -> BBox:
    left = max(0.0, bbox.x - padding)
    top = max(0.0, bbox.y - padding)
    right = min(float(width), bbox.x + bbox.width + padding)
    bottom = min(float(height), bbox.y + bbox.height + padding)
    return BBox(left, top, max(0.0, right - left), max(0.0, bottom - top))


def _resize_to_target(image: Any, spec: RenderSpec) -> tuple[Any, float, float]:
    image_module = _require_image_module()
    max_width = int(spec.extras.get("output_max_width", "1024"))
    max_height = int(spec.extras.get("output_max_height", "768"))
    resized = image.copy()
    if resized.width <= max_width and resized.height <= max_height:
        return resized, 1.0, 1.0
    original_width, original_height = resized.size
    resampling = getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")
    resized.thumbnail((max_width, max_height), resampling)
    return resized, resized.width / original_width, resized.height / original_height


def _save_image(image: Any, spec: RenderSpec) -> tuple[bytes, str, tuple[str, ...]]:
    requested = spec.raster.output_format.lower()
    image_format = "jpeg" if requested in {"jpg", "jpeg"} else requested
    output = BytesIO()
    warnings: list[str] = []
    try:
        if image_format == "png":
            image.save(output, format="PNG", optimize=False)
        elif image_format == "jpeg":
            image.convert("RGB").save(output, format="JPEG", quality=int(spec.raster.jpeg_quality), optimize=False)
        elif image_format == "webp":
            image.convert("RGB").save(output, format="WEBP", quality=int(spec.raster.jpeg_quality), method=0)
        else:
            raise ValueError(f"unsupported raster output format: {requested}")
    except Exception as exc:
        output = BytesIO()
        image.save(output, format="PNG", optimize=False)
        warnings.append(f"failed to save {image_format}; emitted png instead: {exc}")
        image_format = "png"
    return output.getvalue(), image_format, tuple(warnings)


def _snapshot(layers: dict[str, bytes], name: str, image: Any, spec: RenderSpec) -> None:
    if spec.extras.get("debug_intermediate_layers") != "1":
        return
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    layers[name] = output.getvalue()


def _image_from_bytes(data: bytes) -> Any:
    image_module = _require_image_module()
    image = image_module.open(BytesIO(data))
    image.load()
    return image


def _ltrb_from_bbox(bbox: BBox) -> tuple[int, int, int, int]:
    return (
        int(round(bbox.x)),
        int(round(bbox.y)),
        int(round(bbox.x + bbox.width)),
        int(round(bbox.y + bbox.height)),
    )


def _size_tuple(value: object) -> tuple[float, float]:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]), float(value[1])
    return 0.0, 0.0


def _require_image_module() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for record composition") from exc


__all__ = ["ComposedRecord", "compose_record"]
