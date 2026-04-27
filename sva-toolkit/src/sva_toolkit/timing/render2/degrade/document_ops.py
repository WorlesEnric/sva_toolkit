"""Document-style degradation operations with native Pillow fallbacks."""

from __future__ import annotations

import importlib
from typing import Any

from sva_toolkit.timing.render2.spec import DegradationSpec


def apply_document_effects(image: Any, spec: DegradationSpec, rng: Any) -> Any:
    augraphy_image = _try_augraphy(image, spec, rng)
    if augraphy_image is not None:
        return augraphy_image
    return apply_native_document_effects(image, spec, rng)


def apply_native_document_effects(image: Any, spec: DegradationSpec, rng: Any) -> Any:
    image_filter, image_enhance = _require_pillow_modules()
    result = image.convert("RGB")
    family = spec.family

    blur_sigma = float(spec.blur_sigma)
    if family in {"scan", "fax"}:
        blur_sigma = max(blur_sigma, rng.uniform(0.25, 0.75))
    elif family == "camera":
        blur_sigma = max(blur_sigma, rng.uniform(0.10, 0.45))
    elif family == "photocopy":
        blur_sigma = max(blur_sigma, rng.uniform(0.08, 0.28))
    if blur_sigma > 0:
        result = result.filter(image_filter.GaussianBlur(radius=blur_sigma))

    if family == "photocopy":
        result = image_enhance.Contrast(result).enhance(min(spec.contrast, rng.uniform(0.70, 0.88)))
        result = _paper_texture(result, rng, strength=14)
        result = _dust_and_speckles(result, rng, count=max(24, result.width * result.height // 14000))
        result = _bleed_through(result, rng)
    elif family == "scan":
        result = _paper_texture(result, rng, strength=8)
        result = _dust_and_speckles(result, rng, count=max(16, result.width * result.height // 18000))
        result = _streaks(result, rng, count=4)
    elif family == "fax":
        result = _paper_texture(result, rng, strength=10)
        result = _streaks(result, rng, count=9)
        result = _dust_and_speckles(result, rng, count=max(32, result.width * result.height // 12000))
    elif family == "camera":
        result = _uneven_illumination(result, rng)

    if spec.noise_sigma > 0 or family in {"noise", "scan", "fax", "camera"}:
        sigma = max(float(spec.noise_sigma), 0.025 if family in {"scan", "fax", "noise"} else 0.012)
        result = add_noise(result, sigma=sigma, rng=rng)
    return result


def add_noise(image: Any, *, sigma: float, rng: Any) -> Any:
    try:
        np = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
    except ImportError:
        return _add_noise_pil(image, sigma=sigma, rng=rng)
    try:
        seed = rng.randrange(0, 2**32)
        generator = np.random.default_rng(seed)
        array = np.asarray(image.convert("RGB")).astype(np.float32)
        noise = generator.normal(0.0, max(0.0, sigma) * 255.0, array.shape)
        noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
        return image_module.fromarray(noisy, "RGB")
    except Exception:
        return _add_noise_pil(image, sigma=sigma, rng=rng)


def _try_augraphy(image: Any, spec: DegradationSpec, rng: Any) -> Any | None:
    if not spec.augraphy_pipeline:
        return None
    try:
        augraphy = importlib.import_module("augraphy")
        np = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
    except ImportError:
        return None
    try:
        pipeline_cls = getattr(augraphy, "default_augraphy_pipeline", None)
        pipeline = pipeline_cls() if callable(pipeline_cls) else None
        if pipeline is None:
            return None
        np.random.seed(rng.randrange(0, 2**32))
        augmented = pipeline(np.array(image.convert("RGB")))
        if isinstance(augmented, list):
            augmented = augmented[0]
        return image_module.fromarray(augmented).convert("RGB")
    except Exception:
        return None


def _paper_texture(image: Any, rng: Any, *, strength: int) -> Any:
    image_module, image_draw, _image_filter, _image_enhance = _require_all_pillow_modules()
    overlay = image_module.new("RGBA", image.size, (0, 0, 0, 0))
    draw = image_draw.Draw(overlay)
    for _ in range(max(40, image.width * image.height // 9000)):
        x = rng.randrange(0, image.width)
        y = rng.randrange(0, image.height)
        shade = rng.randrange(0, strength + 1)
        draw.point((x, y), fill=(shade, shade, shade, rng.randrange(12, 34)))
    return image_module.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _dust_and_speckles(image: Any, rng: Any, *, count: int) -> Any:
    image_draw = importlib.import_module("PIL.ImageDraw")
    result = image.copy()
    draw = image_draw.Draw(result)
    for _ in range(count):
        x = rng.randrange(0, result.width)
        y = rng.randrange(0, result.height)
        radius = rng.choice((1, 1, 1, 2))
        shade = rng.randrange(120, 238)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(shade, shade, shade))
    return result


def _streaks(image: Any, rng: Any, *, count: int) -> Any:
    image_draw = importlib.import_module("PIL.ImageDraw")
    result = image.copy()
    draw = image_draw.Draw(result)
    for _ in range(count):
        y = rng.randrange(0, result.height)
        shade = rng.randrange(176, 230)
        width = rng.choice((1, 1, 2))
        draw.line((0, y, result.width, y + rng.randrange(-1, 2)), fill=(shade, shade, shade), width=width)
    return result


def _bleed_through(image: Any, rng: Any) -> Any:
    image_module, image_draw, image_filter, _image_enhance = _require_all_pillow_modules()
    overlay = image_module.new("RGBA", image.size, (0, 0, 0, 0))
    draw = image_draw.Draw(overlay)
    for _ in range(12):
        x = rng.randrange(-image.width // 4, image.width)
        y = rng.randrange(0, image.height)
        draw.rectangle((x, y, x + rng.randrange(40, 160), y + rng.randrange(2, 7)), fill=(80, 80, 80, 12))
    overlay = overlay.filter(image_filter.GaussianBlur(radius=2.2))
    return image_module.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _uneven_illumination(image: Any, rng: Any) -> Any:
    image_module, image_draw, image_filter, _image_enhance = _require_all_pillow_modules()
    overlay = image_module.new("RGBA", image.size, (0, 0, 0, 0))
    draw = image_draw.Draw(overlay)
    corner = rng.choice(((0, 0), (image.width, 0), (0, image.height), (image.width, image.height)))
    radius = max(image.width, image.height)
    for step in range(18):
        alpha = max(0, 28 - step)
        left = corner[0] - radius + step * 18
        top = corner[1] - radius + step * 18
        right = corner[0] + radius - step * 18
        bottom = corner[1] + radius - step * 18
        draw.ellipse(
            (min(left, right), min(top, bottom), max(left, right), max(top, bottom)),
            fill=(255, 255, 255, alpha),
        )
    overlay = overlay.filter(image_filter.GaussianBlur(radius=18))
    return image_module.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _add_noise_pil(image: Any, *, sigma: float, rng: Any) -> Any:
    image_module = importlib.import_module("PIL.Image")
    result = image.convert("RGB")
    pixels = result.load()
    amplitude = max(1, int(round(sigma * 255)))
    for y in range(result.height):
        for x in range(result.width):
            delta = rng.randrange(-amplitude, amplitude + 1)
            red, green, blue = pixels[x, y]
            pixels[x, y] = (_clamp(red + delta), _clamp(green + delta), _clamp(blue + delta))
    return image_module.merge("RGB", result.split())


def _clamp(value: int) -> int:
    return max(0, min(255, value))


def _require_pillow_modules() -> tuple[Any, Any]:
    try:
        return importlib.import_module("PIL.ImageFilter"), importlib.import_module("PIL.ImageEnhance")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for document degradation") from exc


def _require_all_pillow_modules() -> tuple[Any, Any, Any, Any]:
    try:
        return (
            importlib.import_module("PIL.Image"),
            importlib.import_module("PIL.ImageDraw"),
            importlib.import_module("PIL.ImageFilter"),
            importlib.import_module("PIL.ImageEnhance"),
        )
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for document degradation") from exc


__all__ = ["add_noise", "apply_document_effects", "apply_native_document_effects"]
