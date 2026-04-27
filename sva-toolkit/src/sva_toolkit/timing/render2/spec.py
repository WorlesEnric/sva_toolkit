"""Render specification dataclasses for timing renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from sva_toolkit.timing.render2.decorations import AnnotationPolicy
from sva_toolkit.timing.render2.primitives import BBox, FontSpec, Stroke


@dataclass(frozen=True)
class StyleSpec:
    family: str
    palette: tuple[str, ...]
    primary_font: FontSpec
    label_font: FontSpec
    waveform_stroke: Stroke
    grid_stroke: Stroke
    grid_mode: str
    bus_style: str
    unknown_style: str
    cut_style: str
    transition_shape: str
    color_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "palette", tuple(self.palette))


@dataclass(frozen=True)
class LayoutSpec:
    lane_height: float
    lane_pitch: float
    tick_width: float
    label_position: str
    label_alignment: str
    margin: BBox
    grouped_lanes: bool = False
    multiline_labels: bool = False

    def __post_init__(self) -> None:
        if self.lane_height <= 0 or self.lane_pitch <= 0 or self.tick_width <= 0:
            raise ValueError("layout dimensions must be positive")


@dataclass(frozen=True)
class AnnotationSpec:
    policy: AnnotationPolicy
    measurement_label_style: str
    helper_line_density: float
    nuisance_text_count: int
    semantic_guides_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", AnnotationPolicy(self.policy))
        if not 0.0 <= self.helper_line_density <= 1.0:
            raise ValueError("helper_line_density must be between 0 and 1")
        if self.nuisance_text_count < 0:
            raise ValueError("nuisance_text_count must be non-negative")


@dataclass(frozen=True)
class PageSpec:
    enabled: bool
    caption_above: bool
    caption_below: bool
    surrounding_paragraph: bool
    table_border: bool
    page_header: bool
    page_footer: bool
    crop_mode: str


@dataclass(frozen=True)
class RasterSpec:
    dpi: int
    antialias: bool
    output_format: str
    jpeg_quality: int = 85

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.output_format not in {"png", "jpg", "webp"}:
            raise ValueError(f"unsupported raster output_format: {self.output_format}")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True)
class DegradationSpec:
    family: str
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0
    contrast: float = 1.0
    brightness: float = 1.0
    rotation_deg: float = 0.0
    perspective: float = 0.0
    jpeg_quality: int = 95
    morphology: str = "none"
    augraphy_pipeline: str | None = None

    def __post_init__(self) -> None:
        if self.blur_sigma < 0 or self.noise_sigma < 0:
            raise ValueError("blur_sigma and noise_sigma must be non-negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True)
class RenderSpec:
    renderer_id: str
    style: StyleSpec
    layout: LayoutSpec
    annotations: AnnotationSpec
    page: PageSpec
    raster: RasterSpec
    degradation: DegradationSpec
    seed: int
    profile: str
    extras: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))
