"""Small deterministic text metrics for SVG layout decisions."""

from __future__ import annotations

from sva_toolkit.timing.render2.primitives import BBox, FontSpec, Point


_FAMILY_FACTORS = {
    "helvetica": 0.55,
    "arial": 0.55,
    "verdana": 0.58,
    "tahoma": 0.56,
    "times": 0.50,
    "georgia": 0.52,
    "courier": 0.62,
    "monaco": 0.60,
    "consolas": 0.60,
    "monospace": 0.62,
}


def family_factor(family: str) -> float:
    normalized = family.lower()
    for key, factor in _FAMILY_FACTORS.items():
        if key in normalized:
            return factor
    return 0.55


def estimate_text_width(text: str, font: FontSpec) -> float:
    """Estimate rendered text width in CSS pixels."""

    weight_factor = 1.06 if font.weight not in {"400", "normal"} else 1.0
    return len(text) * font.size_px * family_factor(font.family) * weight_factor


def estimate_text_bbox(
    text: str,
    font: FontSpec,
    anchor: Point = Point(0, 0),
    *,
    text_anchor: str = "start",
) -> BBox:
    """Estimate a tight text bbox from an SVG baseline anchor.

    The estimator intentionally stays simple; it only needs to be stable enough
    for role bbox bookkeeping and conservative occlusion checks.
    """

    width = estimate_text_width(text, font)
    height = font.size_px * 1.2
    if text_anchor == "middle":
        x = anchor.x - width / 2
    elif text_anchor == "end":
        x = anchor.x - width
    else:
        x = anchor.x
    y = anchor.y - font.size_px * 0.9
    return BBox(x=x, y=y, width=width, height=height)
