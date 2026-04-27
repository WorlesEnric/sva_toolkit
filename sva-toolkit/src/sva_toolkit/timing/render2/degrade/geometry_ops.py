"""Geometric degradation operations with optional OpenCV acceleration."""

from __future__ import annotations

import importlib
import math
from typing import Any

from sva_toolkit.timing.render2.spec import DegradationSpec


def rotate_image(image: Any, spec: DegradationSpec, rng: Any) -> Any:
    image_module = _require_image_module()
    angle = float(spec.rotation_deg)
    if abs(angle) < 0.001 and spec.family == "camera":
        angle = rng.uniform(-1.2, 1.2)
    if abs(angle) < 0.001:
        return image.copy()
    resampling = getattr(getattr(image_module, "Resampling", image_module), "BICUBIC")
    return image.rotate(angle, resample=resampling, expand=True, fillcolor=_fillcolor(image))


def perspective_warp(image: Any, spec: DegradationSpec, rng: Any) -> Any:
    amount = float(spec.perspective)
    if amount <= 0.0 and spec.family == "camera":
        amount = rng.uniform(0.008, 0.025)
    if amount <= 0.0:
        return image.copy()

    width, height = image.size
    jitter_x = max(1.0, width * amount)
    jitter_y = max(1.0, height * amount)
    destination = (
        (rng.uniform(0.0, jitter_x), rng.uniform(0.0, jitter_y)),
        (width - rng.uniform(0.0, jitter_x), rng.uniform(0.0, jitter_y)),
        (width - rng.uniform(0.0, jitter_x), height - rng.uniform(0.0, jitter_y)),
        (rng.uniform(0.0, jitter_x), height - rng.uniform(0.0, jitter_y)),
    )
    source = ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))

    cv2_result = _try_cv2_perspective(image, source, destination)
    if cv2_result is not None:
        return cv2_result

    image_module = _require_image_module()
    coefficients = _perspective_coefficients(destination, source)
    transform_kind = getattr(getattr(image_module, "Transform", image_module), "PERSPECTIVE")
    resampling = getattr(getattr(image_module, "Resampling", image_module), "BICUBIC")
    return image.transform(
        (width, height),
        transform_kind,
        coefficients,
        resampling,
        fillcolor=_fillcolor(image),
    )


def crop_image(image: Any, spec: DegradationSpec, rng: Any) -> Any:
    if spec.family != "camera":
        return image.copy()
    width, height = image.size
    trim_x = int(width * rng.uniform(0.0, 0.012))
    trim_y = int(height * rng.uniform(0.0, 0.012))
    if trim_x <= 0 and trim_y <= 0:
        return image.copy()
    return image.crop((trim_x, trim_y, max(trim_x + 1, width - trim_x), max(trim_y + 1, height - trim_y)))


def resize_image(image: Any, _spec: DegradationSpec, _rng: Any, *, max_size: tuple[int, int] | None = None) -> Any:
    if max_size is None:
        return image.copy()
    image_module = _require_image_module()
    resized = image.copy()
    resampling = getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")
    resized.thumbnail(max_size, resampling)
    return resized


def _try_cv2_perspective(image: Any, source: tuple[tuple[float, float], ...], destination: tuple[tuple[float, float], ...]) -> Any | None:
    try:
        cv2 = importlib.import_module("cv2")
        np = importlib.import_module("numpy")
        image_module = _require_image_module()
    except ImportError:
        return None
    try:
        array = np.array(image.convert("RGB"))
        matrix = cv2.getPerspectiveTransform(np.float32(source), np.float32(destination))
        warped = cv2.warpPerspective(
            array,
            matrix,
            image.size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return image_module.fromarray(warped, "RGB").convert(image.mode)
    except Exception:
        return None


def _perspective_coefficients(
    source: tuple[tuple[float, float], ...],
    destination: tuple[tuple[float, float], ...],
) -> tuple[float, ...]:
    matrix: list[list[float]] = []
    vector: list[float] = []
    for (x_src, y_src), (x_dst, y_dst) in zip(source, destination, strict=True):
        matrix.append([x_src, y_src, 1.0, 0.0, 0.0, 0.0, -x_dst * x_src, -x_dst * y_src])
        matrix.append([0.0, 0.0, 0.0, x_src, y_src, 1.0, -y_dst * x_src, -y_dst * y_src])
        vector.extend([x_dst, y_dst])
    return tuple(_solve_linear_system(matrix, vector))


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row_index: abs(augmented[row_index][column]))
        if math.isclose(augmented[pivot][column], 0.0, abs_tol=1e-12):
            raise RuntimeError("singular perspective transform")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row_index in range(size):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            augmented[row_index] = [
                value - factor * pivot_value for value, pivot_value in zip(augmented[row_index], augmented[column], strict=True)
            ]
    return [row[-1] for row in augmented]


def _fillcolor(image: Any) -> tuple[int, ...]:
    if image.mode == "RGBA":
        return (255, 255, 255, 255)
    if image.mode == "L":
        return (255,)
    return (255, 255, 255)


def _require_image_module() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for geometric degradation") from exc


__all__ = ["crop_image", "perspective_warp", "resize_image", "rotate_image"]
