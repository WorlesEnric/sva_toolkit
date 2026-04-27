"""Renderer result metadata for timing diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from sva_toolkit.timing.render2.primitives import BBox, Point
from sva_toolkit.timing.render2.spec import RenderSpec


@dataclass(frozen=True)
class TextPrimitive:
    text: str
    bbox: BBox
    role: str
    visibility_class: str


@dataclass(frozen=True)
class VisualVisibilityReport:
    """Visibility audit produced by a renderer for downstream leakage checks."""

    rendered_text: tuple[TextPrimitive, ...]
    target_tokens_visible: frozenset[str]
    nuisance_tokens: frozenset[str]
    debug_overlay_tokens: frozenset[str]
    leaked_tokens: frozenset[str]
    occluded_lane_fractions: Mapping[str, float]
    minimum_contrast: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "rendered_text", tuple(self.rendered_text))
        object.__setattr__(self, "target_tokens_visible", frozenset(self.target_tokens_visible))
        object.__setattr__(self, "nuisance_tokens", frozenset(self.nuisance_tokens))
        object.__setattr__(self, "debug_overlay_tokens", frozenset(self.debug_overlay_tokens))
        object.__setattr__(self, "leaked_tokens", frozenset(self.leaked_tokens))
        object.__setattr__(self, "occluded_lane_fractions", MappingProxyType(dict(self.occluded_lane_fractions)))


VisibilityReport = VisualVisibilityReport


@dataclass(frozen=True)
class DiagramLayout:
    """Final layout returned by a renderer."""

    width: float
    height: float
    plot_origin: Point
    tick_width: float
    lane_height: float
    lane_pitch: float
    bbox_by_role: Mapping[str, tuple[BBox, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {role: tuple(boxes) for role, boxes in self.bbox_by_role.items()}
        object.__setattr__(self, "bbox_by_role", MappingProxyType(normalized))


@dataclass(frozen=True)
class RenderResult:
    svg_text: str | None
    png_bytes: bytes | None
    layout: DiagramLayout
    visibility: VisualVisibilityReport
    render_spec: RenderSpec
    warnings: tuple[str, ...] = ()
    ascii_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
