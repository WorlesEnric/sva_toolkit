"""Synthetic document-page composition for render2 timing diagrams.

``NUISANCE_TEXTS`` is the deterministic source pool used for captions, page
headers, footers, and surrounding paragraph fragments. The entries are
hardware-spec flavored distractors intended to be recorded as nuisance text so
downstream leakage audits can ignore them rather than treating them as target
tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import random
from typing import Any

from sva_toolkit.timing.render2.primitives import BBox
from sva_toolkit.timing.render2.spec import PageSpec, RasterSpec


NUISANCE_TEXTS: tuple[str, ...] = (
    "Figure 7. Read transaction timing",
    "tCLK min = 10 ns",
    "valid sampled on rising edge",
    "see Table 12",
    "not to scale",
    "reserved",
    "all timing relative to clk",
    "setup path shown for reference",
    "hold interval excludes reset",
    "cycle count may vary by implementation",
    "write data accepted when ready is high",
    "read latency measured at boundary",
    "mask bits are ignored when idle",
    "response phase begins after grant",
    "address stable before enable",
    "burst length encoded as beats minus one",
    "do not sample during reset assertion",
    "valid deassertion may be delayed",
    "acknowledge pulse is one cycle",
    "see Section 4.3.2",
    "timing assumes nominal voltage",
    "falling edge path not shown",
    "command bus remains reserved",
    "lane skew less than 120 ps",
    "calibration updates are asynchronous",
    "minimum pulse width is two cycles",
    "idle cycles omitted for clarity",
    "transfer completes on final beat",
    "error response has priority",
    "clock gating not illustrated",
    "phase detector locks after reset",
    "write strobe follows data lane",
    "sample window centered on edge",
    "turnaround cycle required",
    "credit counter saturates at zero",
    "header parity checked in cycle N",
    "sideband signal synchronized locally",
    "receiver ignores duplicate token",
    "low power entry request",
    "implementation-defined delay",
)


@dataclass(frozen=True)
class PageComposer:
    """Wrap a rendered diagram image in deterministic synthetic page context."""

    page: PageSpec
    raster: RasterSpec | None = None

    def compose(self, image: Any, *, rng: random.Random) -> tuple[Any, Mapping[str, Any]]:
        image_module, draw_module, font_module = _require_pillow()
        base = image.convert("RGBA")
        mode = _normalized_mode(self.page.crop_mode)

        if not self.page.enabled:
            return base.copy(), {
                "page_enabled": False,
                "crop_mode": "tight",
                "diagram_bbox": BBox(0.0, 0.0, float(base.width), float(base.height)),
                "diagram_bbox_uncropped": BBox(0.0, 0.0, float(base.width), float(base.height)),
                "diagram_scale_x": 1.0,
                "diagram_scale_y": 1.0,
                "nuisance_text": (),
                "elements": (),
            }

        page_width, page_height = _page_size(base.width, base.height, self.raster.dpi if self.raster else 96)
        page_image = image_module.new("RGBA", (page_width, page_height), (253, 252, 248, 255))
        draw = draw_module.Draw(page_image)
        font = font_module.load_default()
        small_font = font_module.load_default()
        elements: list[dict[str, Any]] = []
        nuisance_text: list[str] = []

        _draw_page_background(draw, page_width, page_height, rng)
        if self.page.page_header:
            text = _sample_text(rng)
            _draw_text_element(
                draw,
                elements,
                nuisance_text,
                role="page_header",
                text=text,
                xy=(36, 26),
                font=small_font,
                fill=(94, 94, 88, 255),
            )
            draw.line((36, 44, page_width - 36, 44), fill=(210, 210, 204, 255), width=1)

        diagram = _resize_diagram(base, page_width, page_height, mode)
        diagram_x, diagram_y = _diagram_position(page_width, page_height, diagram.width, diagram.height, mode, rng)
        diagram_bbox = BBox(float(diagram_x), float(diagram_y), float(diagram.width), float(diagram.height))

        if self.page.table_border:
            pad = 12
            border = (
                diagram_x - pad,
                diagram_y - pad,
                diagram_x + diagram.width + pad,
                diagram_y + diagram.height + pad,
            )
            draw.rectangle(border, outline=(165, 165, 158, 255), width=1)
            elements.append({"role": "page_table_border", "bbox": _bbox_from_ltrb(border)})

        if self.page.caption_above:
            caption = "Figure 7. Read transaction timing"
            y = max(50, diagram_y - 28)
            _draw_text_element(
                draw,
                elements,
                nuisance_text,
                role="page_caption",
                text=caption,
                xy=(diagram_x, y),
                font=font,
                fill=(42, 42, 39, 255),
            )

        page_image.alpha_composite(diagram, (diagram_x, diagram_y))
        elements.append({"role": "diagram", "bbox": diagram_bbox})

        if self.page.caption_below:
            caption = _sample_text(rng)
            _draw_text_element(
                draw,
                elements,
                nuisance_text,
                role="page_caption",
                text=caption,
                xy=(diagram_x, min(page_height - 56, diagram_y + diagram.height + 14)),
                font=font,
                fill=(42, 42, 39, 255),
            )

        if self.page.surrounding_paragraph:
            _draw_paragraph_fragments(draw, elements, nuisance_text, page_width, page_height, diagram_bbox, mode, rng, font)

        if self.page.page_footer:
            footer = _sample_text(rng)
            bbox = _draw_text_element(
                draw,
                elements,
                nuisance_text,
                role="page_footer",
                text=footer,
                xy=(36, page_height - 34),
                font=small_font,
                fill=(94, 94, 88, 255),
            )
            draw.line((36, bbox.y - 8, page_width - 36, bbox.y - 8), fill=(210, 210, 204, 255), width=1)

        crop_box = _fragment_crop_box(page_width, page_height, diagram_bbox, rng) if mode == "fragment" else None
        if crop_box is not None:
            page_image = page_image.crop(_ltrb_from_bbox(crop_box))
            elements = [_adjust_element(element, crop_box) for element in elements]
            diagram_bbox = _clip_bbox(_offset_bbox(diagram_bbox, -crop_box.x, -crop_box.y), page_image.width, page_image.height)

        return page_image, {
            "page_enabled": True,
            "crop_mode": mode,
            "diagram_bbox": diagram_bbox,
            "diagram_bbox_uncropped": BBox(float(diagram_x), float(diagram_y), float(diagram.width), float(diagram.height)),
            "diagram_scale_x": diagram.width / max(1.0, base.width),
            "diagram_scale_y": diagram.height / max(1.0, base.height),
            "nuisance_text": tuple(nuisance_text),
            "elements": tuple(elements),
            "crop_box": crop_box,
            "page_width": page_width,
            "page_height": page_height,
            "raster_dpi": self.raster.dpi if self.raster else 96,
        }


def compose_page(image: Any, page: PageSpec, *, rng: random.Random, raster: RasterSpec | None = None) -> tuple[Any, Mapping[str, Any]]:
    return PageComposer(page, raster).compose(image, rng=rng)


def _page_size(image_width: int, image_height: int, dpi: int) -> tuple[int, int]:
    width = int(round(8.27 * dpi))
    height = int(round(11.69 * dpi))
    min_width = int(image_width * 1.08) + 96
    min_height = int(image_height * 1.18) + 128
    if width < min_width:
        width = min_width
        height = max(height, int(round(width * 11.69 / 8.27)))
    if height < min_height:
        height = min_height
        width = max(width, int(round(height * 8.27 / 11.69)))
    return max(120, width), max(120, height)


def _resize_diagram(image: Any, page_width: int, page_height: int, mode: str) -> Any:
    image_module, _, _ = _require_pillow()
    if mode == "tight":
        max_width, max_height = page_width * 0.90, page_height * 0.76
    elif mode == "off_center":
        max_width, max_height = page_width * 0.58, page_height * 0.48
    elif mode == "fragment":
        max_width, max_height = page_width * 0.88, page_height * 0.62
    else:
        max_width, max_height = page_width * 0.72, page_height * 0.46
    scale = min(max_width / max(1, image.width), max_height / max(1, image.height))
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    resampling = getattr(getattr(image_module, "Resampling", image_module), "LANCZOS")
    return image.resize((width, height), resampling)


def _diagram_position(
    page_width: int,
    page_height: int,
    diagram_width: int,
    diagram_height: int,
    mode: str,
    rng: random.Random,
) -> tuple[int, int]:
    margin = max(28, int(min(page_width, page_height) * 0.045))
    if mode == "tight":
        center_x = (page_width - diagram_width) // 2
        center_y = (page_height - diagram_height) // 2
        return (
            _clamp_int(center_x + rng.randrange(-margin // 2, margin // 2 + 1), margin, page_width - diagram_width - margin),
            _clamp_int(center_y + rng.randrange(-margin // 2, margin // 2 + 1), margin, page_height - diagram_height - margin),
        )
    if mode == "off_center":
        upper_left = rng.random() < 0.5
        if upper_left:
            return margin, max(70, margin + rng.randrange(0, margin + 1))
        return (
            max(margin, page_width - diagram_width - margin),
            max(70, page_height - diagram_height - margin * 3),
        )
    y = int(page_height * (0.26 if mode == "loose" else 0.22))
    y += rng.randrange(-margin, margin + 1)
    x = (page_width - diagram_width) // 2 + rng.randrange(-margin, margin + 1)
    return (
        _clamp_int(x, margin, page_width - diagram_width - margin),
        _clamp_int(y, 70, page_height - diagram_height - margin * 2),
    )


def _draw_page_background(draw: Any, width: int, height: int, rng: random.Random) -> None:
    for _ in range(80):
        x = rng.randrange(0, width)
        y = rng.randrange(0, height)
        shade = rng.randrange(230, 248)
        draw.point((x, y), fill=(shade, shade, shade - 3, 255))


def _draw_paragraph_fragments(
    draw: Any,
    elements: list[dict[str, Any]],
    nuisance_text: list[str],
    page_width: int,
    page_height: int,
    diagram_bbox: BBox,
    mode: str,
    rng: random.Random,
    font: Any,
) -> None:
    line_height = 15
    starts: list[tuple[int, int, int]]
    if mode == "off_center" and diagram_bbox.x < page_width * 0.45:
        starts = [(int(diagram_bbox.x + diagram_bbox.width + 28), int(diagram_bbox.y + 8), 4)]
    elif mode == "off_center":
        starts = [(36, int(diagram_bbox.y + 8), 4)]
    else:
        starts = [
            (36, max(58, int(diagram_bbox.y - 76)), 3),
            (36, min(page_height - 130, int(diagram_bbox.y + diagram_bbox.height + 48)), 3),
        ]
    for x, y, count in starts:
        for index in range(count):
            text = _sample_text(rng)
            _draw_text_element(
                draw,
                elements,
                nuisance_text,
                role="page_paragraph",
                text=text,
                xy=(x, y + index * line_height),
                font=font,
                fill=(96, 96, 90, 255),
            )


def _fragment_crop_box(page_width: int, page_height: int, diagram_bbox: BBox, rng: random.Random) -> BBox:
    context = rng.randrange(24, 64)
    left = max(0, int(diagram_bbox.x) - context)
    right = min(page_width, int(diagram_bbox.x + diagram_bbox.width) + context)
    top = max(0, int(diagram_bbox.y) - context - 26)
    bottom = min(page_height, int(diagram_bbox.y + diagram_bbox.height) + context + 42)
    return BBox(float(left), float(top), float(max(1, right - left)), float(max(1, bottom - top)))


def _draw_text_element(
    draw: Any,
    elements: list[dict[str, Any]],
    nuisance_text: list[str],
    *,
    role: str,
    text: str,
    xy: tuple[int, int],
    font: Any,
    fill: tuple[int, int, int, int],
) -> BBox:
    draw.text(xy, text, fill=fill, font=font)
    bbox = _text_bbox(draw, xy, text, font)
    elements.append({"role": role, "text": text, "bbox": bbox})
    nuisance_text.append(text)
    return bbox


def _text_bbox(draw: Any, xy: tuple[int, int], text: str, font: Any) -> BBox:
    try:
        left, top, right, bottom = draw.textbbox(xy, text, font=font)
    except AttributeError:  # pragma: no cover - old Pillow compatibility
        width, height = draw.textsize(text, font=font)
        left, top, right, bottom = xy[0], xy[1], xy[0] + width, xy[1] + height
    return BBox(float(left), float(top), float(right - left), float(bottom - top))


def _adjust_element(element: dict[str, Any], crop_box: BBox) -> dict[str, Any]:
    adjusted = dict(element)
    bbox = adjusted.get("bbox")
    if isinstance(bbox, BBox):
        adjusted["bbox"] = _offset_bbox(bbox, -crop_box.x, -crop_box.y)
    return adjusted


def _sample_text(rng: random.Random) -> str:
    return NUISANCE_TEXTS[rng.randrange(0, len(NUISANCE_TEXTS))]


def _normalized_mode(mode: str) -> str:
    if mode in {"fragment", "page_fragment"}:
        return "fragment"
    if mode in {"tight", "loose", "off_center"}:
        return mode
    return "tight"


def _bbox_from_ltrb(values: tuple[int, int, int, int]) -> BBox:
    left, top, right, bottom = values
    return BBox(float(left), float(top), float(max(0, right - left)), float(max(0, bottom - top)))


def _ltrb_from_bbox(bbox: BBox) -> tuple[int, int, int, int]:
    return (
        int(round(bbox.x)),
        int(round(bbox.y)),
        int(round(bbox.x + bbox.width)),
        int(round(bbox.y + bbox.height)),
    )


def _offset_bbox(bbox: BBox, dx: float, dy: float) -> BBox:
    return BBox(bbox.x + dx, bbox.y + dy, bbox.width, bbox.height)


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    left = min(max(0.0, bbox.x), float(width))
    top = min(max(0.0, bbox.y), float(height))
    right = min(max(left, bbox.x + bbox.width), float(width))
    bottom = min(max(top, bbox.y + bbox.height), float(height))
    return BBox(left, top, right - left, bottom - top)


def _clamp_int(value: int, lower: int, upper: int) -> int:
    if upper < lower:
        return lower
    return max(lower, min(upper, value))


def _require_pillow() -> tuple[Any, Any, Any]:
    try:
        image_module = importlib.import_module("PIL.Image")
        draw_module = importlib.import_module("PIL.ImageDraw")
        font_module = importlib.import_module("PIL.ImageFont")
    except ImportError as exc:  # pragma: no cover - Pillow is present in test env
        raise RuntimeError("Pillow is required for page composition") from exc
    return image_module, draw_module, font_module


__all__ = ["NUISANCE_TEXTS", "PageComposer", "compose_page"]
