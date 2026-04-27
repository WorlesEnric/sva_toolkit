"""SVG rasterization helpers for render2 post-processing."""

from __future__ import annotations

from io import BytesIO
import importlib
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Any

from sva_toolkit.timing.render2.spec import RasterSpec


_LENGTH_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)([a-zA-Z%]*)\s*$")
_CSS_DPI = 96.0


def rasterize_svg(svg_text: str, spec: RasterSpec) -> Any:
    """Rasterize SVG text into a Pillow image at the sampled raster DPI.

    CairoSVG is preferred when available. ``resvg-py`` and Wand are tried as
    secondary optional engines. If none can be imported or they fail to render,
    the function returns a synthetic blank Pillow image and emits a warning.
    """

    image_module, draw_module = _require_pillow()
    parent_width, parent_height = _svg_size(svg_text)
    output_width = max(1, int(round(parent_width * spec.dpi / _CSS_DPI)))
    output_height = max(1, int(round(parent_height * spec.dpi / _CSS_DPI)))

    failures: list[str] = []
    for renderer in (_render_with_cairosvg, _render_with_resvg, _render_with_wand):
        try:
            png_bytes = renderer(svg_text, spec, parent_width, parent_height, output_width, output_height)
        except Exception as exc:  # pragma: no cover - exercised when optional deps are missing/broken
            failures.append(f"{renderer.__name__}: {exc}")
            continue
        if png_bytes:
            image = image_module.open(BytesIO(png_bytes))
            image.load()
            return image.convert("RGBA")

    message = (
        "falling back to synthetic SVG raster because no optional rasterizer "
        f"could render this SVG ({'; '.join(failures) or 'no renderers tried'})"
    )
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    image = image_module.new("RGBA", (output_width, output_height), (255, 255, 255, 255))
    draw = draw_module.Draw(image)
    _draw_synthetic_svg(svg_text, draw, output_width / parent_width, output_height / parent_height)
    draw.rectangle((0, 0, output_width - 1, output_height - 1), outline=(220, 220, 220, 255))
    return image


def _render_with_cairosvg(
    svg_text: str,
    spec: RasterSpec,
    parent_width: float,
    parent_height: float,
    output_width: int,
    output_height: int,
) -> bytes:
    cairosvg = importlib.import_module("cairosvg")
    output = BytesIO()
    scale = output_width / parent_width if parent_width > 0 else spec.dpi / _CSS_DPI
    cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        write_to=output,
        parent_width=parent_width,
        parent_height=parent_height,
        output_width=output_width,
        output_height=output_height,
        scale=scale,
        dpi=spec.dpi,
    )
    return output.getvalue()


def _render_with_resvg(
    svg_text: str,
    _spec: RasterSpec,
    _parent_width: float,
    _parent_height: float,
    output_width: int,
    output_height: int,
) -> bytes:
    module = importlib.import_module("resvg_py")
    if hasattr(module, "svg_to_png"):
        rendered = module.svg_to_png(svg_text, width=output_width, height=output_height)
    elif hasattr(module, "render"):
        rendered = module.render(svg_text, width=output_width, height=output_height)
    else:
        raise RuntimeError("resvg_py exposes no supported render function")
    if isinstance(rendered, bytes):
        return rendered
    if hasattr(rendered, "as_png"):
        return rendered.as_png()
    raise RuntimeError("resvg_py returned an unsupported result")


def _render_with_wand(
    svg_text: str,
    spec: RasterSpec,
    _parent_width: float,
    _parent_height: float,
    output_width: int,
    output_height: int,
) -> bytes:
    wand_image = importlib.import_module("wand.image")
    with wand_image.Image(blob=svg_text.encode("utf-8"), format="svg", resolution=spec.dpi) as image:
        image.resize(output_width, output_height)
        image.format = "png"
        return image.make_blob()


def _svg_size(svg_text: str) -> tuple[float, float]:
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return 1.0, 1.0

    view_box = _view_box(root.attrib.get("viewBox"))
    width = _length_to_px(root.attrib.get("width"), fallback=view_box[2] if view_box else 1.0)
    height = _length_to_px(root.attrib.get("height"), fallback=view_box[3] if view_box else 1.0)
    return max(1.0, width), max(1.0, height)


def _view_box(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 4:
        return None
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _length_to_px(value: str | None, *, fallback: float) -> float:
    if value is None:
        return fallback
    match = _LENGTH_RE.match(value)
    if match is None:
        return fallback
    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"", "px"}:
        return number
    if unit == "in":
        return number * _CSS_DPI
    if unit == "cm":
        return number * _CSS_DPI / 2.54
    if unit == "mm":
        return number * _CSS_DPI / 25.4
    if unit == "pt":
        return number * _CSS_DPI / 72.0
    if unit == "pc":
        return number * 16.0
    return fallback


def _draw_synthetic_svg(svg_text: str, draw: Any, scale_x: float, scale_y: float) -> None:
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1]
        if name == "rect":
            _draw_rect(element, draw, scale_x, scale_y)
        elif name == "line":
            _draw_line(element, draw, scale_x, scale_y)
        elif name == "polyline":
            _draw_polyline(element, draw, scale_x, scale_y)
        elif name == "text":
            _draw_text(element, draw, scale_x, scale_y)


def _draw_rect(element: ET.Element, draw: Any, scale_x: float, scale_y: float) -> None:
    x = _attr_float(element, "x") * scale_x
    y = _attr_float(element, "y") * scale_y
    width = _attr_float(element, "width") * scale_x
    height = _attr_float(element, "height") * scale_y
    fill = _color(element.attrib.get("fill"), default=None)
    outline = _color(element.attrib.get("stroke"), default=None)
    draw.rectangle((x, y, x + width, y + height), fill=fill, outline=outline)


def _draw_line(element: ET.Element, draw: Any, scale_x: float, scale_y: float) -> None:
    fill = _color(element.attrib.get("stroke"), default=(0, 0, 0, 255))
    width = max(1, int(round(_attr_float(element, "stroke-width", 1.0) * max(scale_x, scale_y))))
    draw.line(
        (
            _attr_float(element, "x1") * scale_x,
            _attr_float(element, "y1") * scale_y,
            _attr_float(element, "x2") * scale_x,
            _attr_float(element, "y2") * scale_y,
        ),
        fill=fill,
        width=width,
    )


def _draw_polyline(element: ET.Element, draw: Any, scale_x: float, scale_y: float) -> None:
    points = []
    for pair in element.attrib.get("points", "").split():
        if "," not in pair:
            continue
        x_text, y_text = pair.split(",", 1)
        points.append((float(x_text) * scale_x, float(y_text) * scale_y))
    if len(points) < 2:
        return
    fill = _color(element.attrib.get("stroke"), default=(0, 0, 0, 255))
    width = max(1, int(round(_attr_float(element, "stroke-width", 1.0) * max(scale_x, scale_y))))
    draw.line(points, fill=fill, width=width, joint="curve")


def _draw_text(element: ET.Element, draw: Any, scale_x: float, scale_y: float) -> None:
    text = "".join(element.itertext())
    if not text:
        return
    x = _attr_float(element, "x") * scale_x
    font_size = _attr_float(element, "font-size", 12.0)
    y = (_attr_float(element, "y") - font_size * 0.78) * scale_y
    fill = _color(element.attrib.get("fill"), default=(0, 0, 0, 255))
    font = _fallback_font()
    try:
        left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
    except AttributeError:  # pragma: no cover - old Pillow compatibility
        text_width = draw.textlength(text, font=font)
    anchor = element.attrib.get("text-anchor", "start")
    if anchor == "middle":
        x -= text_width / 2
    elif anchor == "end":
        x -= text_width
    draw.text((x, y), text, fill=fill, font=font)


def _fallback_font() -> Any:
    try:
        return importlib.import_module("PIL.ImageFont").load_default()
    except Exception:
        return None


def _attr_float(element: ET.Element, name: str, default: float = 0.0) -> float:
    value = element.attrib.get(name)
    if value is None:
        return default
    try:
        return float(value.replace("px", ""))
    except ValueError:
        return default


def _color(value: str | None, *, default: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if value is None or value == "none":
        return default
    text = value.strip()
    if text.startswith("#") and len(text) == 7:
        return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16), 255)
    if text.lower() == "black":
        return (0, 0, 0, 255)
    if text.lower() == "white":
        return (255, 255, 255, 255)
    return default


def _require_pillow() -> tuple[Any, Any]:
    try:
        image_module = importlib.import_module("PIL.Image")
        draw_module = importlib.import_module("PIL.ImageDraw")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required to return rasterized SVG images") from exc
    return image_module, draw_module


__all__ = ["rasterize_svg"]
