"""Visual coverage tracking for render2 outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from sva_toolkit.timing.render2.decorations import AnnotationPolicy
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec


FONT_SIZE_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 10.0),
    (10.0, 12.0),
    (12.0, 14.0),
    (14.0, 16.0),
    (16.0, 18.0),
    (18.0, 9999.0),
)
STROKE_WIDTH_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.75),
    (0.75, 1.25),
    (1.25, 1.75),
    (1.75, 2.50),
    (2.50, 9999.0),
)
COUNT_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 2.0),
    (3.0, 5.0),
    (6.0, 10.0),
    (11.0, 9999.0),
)
DPI_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 96.0),
    (97.0, 150.0),
    (151.0, 220.0),
    (221.0, 9999.0),
)
BLUR_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.01),
    (0.01, 0.20),
    (0.20, 0.60),
    (0.60, 9999.0),
)
PERSPECTIVE_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.005),
    (0.005, 0.025),
    (0.025, 0.060),
    (0.060, 9999.0),
)
OCCLUSION_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 0.05),
    (0.05, 0.15),
    (0.15, 1.0),
)


class VisualCoverageTracker:
    AXES: tuple[str, ...] = (
        "renderer_id",
        "profile",
        "style_family",
        "font_family_bucket",
        "font_size_bucket",
        "stroke_width_bucket",
        "grid_mode",
        "tick_label_mode",
        "bus_style",
        "unknown_style",
        "cut_style",
        "annotation_policy",
        "helper_line_count_bucket",
        "nuisance_text_count_bucket",
        "page_context_mode",
        "color_mode",
        "raster_dpi_bucket",
        "compression_bucket",
        "blur_bucket",
        "perspective_bucket",
        "crop_bucket",
        "occlusion_bucket",
        "recoverability_class",
        "leakage_audit_status",
    )

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update(self, render_spec: RenderSpec, render_outcome: object | None, scene: TimingScene) -> None:
        for axis, bucket in _axis_buckets(render_spec, render_outcome, scene):
            self._counts[axis][bucket] += 1

    def buckets(self) -> Mapping[str, Mapping[str, int]]:
        return self.to_dict()

    def is_axis_deficient(self, axis: str, target_distinct: int) -> bool:
        if target_distinct <= 0:
            return False
        if axis not in self.AXES:
            raise ValueError(f"unknown visual coverage axis: {axis}")
        return len(self._counts.get(axis, {})) < target_distinct

    def deficient_axes(self, target_distinct: int) -> set[str]:
        if target_distinct <= 0:
            return set()
        return {axis for axis in self.AXES if self.is_axis_deficient(axis, target_distinct)}

    def to_dict(self) -> Mapping[str, Mapping[str, int]]:
        return {axis: dict(self._counts.get(axis, {})) for axis in self.AXES}


def _axis_buckets(
    spec: RenderSpec,
    render_outcome: object | None,
    scene: TimingScene,
) -> tuple[tuple[str, str], ...]:
    result = _render_result(render_outcome)
    helper_count = _helper_line_count(spec, result, scene)
    nuisance_count = _nuisance_text_count(spec, result)
    occlusion = _max_occlusion(result)
    leakage_status = _leakage_status(render_outcome, result)
    return (
        ("renderer_id", spec.renderer_id),
        ("profile", spec.profile),
        ("style_family", spec.style.family),
        ("font_family_bucket", _font_family_bucket(spec.style.primary_font.family)),
        ("font_size_bucket", _bucket_range(spec.style.primary_font.size_px, FONT_SIZE_BUCKETS)),
        ("stroke_width_bucket", _bucket_range(spec.style.waveform_stroke.width, STROKE_WIDTH_BUCKETS)),
        ("grid_mode", spec.style.grid_mode),
        ("tick_label_mode", spec.layout.label_position),
        ("bus_style", spec.style.bus_style),
        ("unknown_style", spec.style.unknown_style),
        ("cut_style", spec.style.cut_style),
        ("annotation_policy", spec.annotations.policy.value),
        ("helper_line_count_bucket", _bucket_range(helper_count, COUNT_BUCKETS)),
        ("nuisance_text_count_bucket", _bucket_range(nuisance_count, COUNT_BUCKETS)),
        ("page_context_mode", _page_context_mode(spec)),
        ("color_mode", spec.style.color_mode),
        ("raster_dpi_bucket", _bucket_range(spec.raster.dpi, DPI_BUCKETS)),
        ("compression_bucket", _compression_bucket(spec)),
        ("blur_bucket", _bucket_range(spec.degradation.blur_sigma, BLUR_BUCKETS)),
        ("perspective_bucket", _bucket_range(spec.degradation.perspective, PERSPECTIVE_BUCKETS)),
        ("crop_bucket", spec.page.crop_mode),
        ("occlusion_bucket", _bucket_range(occlusion, OCCLUSION_BUCKETS) if occlusion is not None else "not_rendered"),
        ("recoverability_class", _recoverability_class(spec, leakage_status, occlusion)),
        ("leakage_audit_status", leakage_status),
    )


def _render_result(render_outcome: object | None) -> RenderResult | None:
    if isinstance(render_outcome, RenderResult):
        return render_outcome
    result = getattr(render_outcome, "result", None)
    return result if isinstance(result, RenderResult) else None


def _helper_line_count(spec: RenderSpec, result: RenderResult | None, scene: TimingScene) -> float:
    if result is not None:
        roles = result.layout.bbox_by_role
        return float(len(roles.get("vertical_helper_line", ())) + len(roles.get("horizontal_helper_line", ())))
    if spec.annotations.policy == AnnotationPolicy.NONE:
        return 0.0
    return round(spec.annotations.helper_line_density * max(1, scene.ticks.total_ticks))


def _nuisance_text_count(spec: RenderSpec, result: RenderResult | None) -> float:
    if result is not None:
        role_count = len(result.layout.bbox_by_role.get("nuisance_text", ()))
        token_count = len(result.visibility.nuisance_tokens)
        return float(max(role_count, token_count))
    return float(spec.annotations.nuisance_text_count)


def _max_occlusion(result: RenderResult | None) -> float | None:
    if result is None:
        return None
    if not result.visibility.occluded_lane_fractions:
        return 0.0
    return max(result.visibility.occluded_lane_fractions.values())


def _leakage_status(render_outcome: object | None, result: RenderResult | None) -> str:
    leakage = getattr(render_outcome, "leakage", None)
    if leakage is not None:
        return "passed" if bool(getattr(leakage, "passed", False)) else "failed"
    rejection_reason = getattr(render_outcome, "rejection_reason", None)
    if rejection_reason == "render_text_leakage":
        return "failed"
    if result is not None:
        if result.visibility.leaked_tokens or result.visibility.debug_overlay_tokens:
            return "failed"
        return "metadata_clean"
    return "not_run"


def _font_family_bucket(family: str) -> str:
    lowered = family.lower()
    if any(token in lowered for token in ("mono", "courier", "consolas")):
        return "monospace"
    if any(token in lowered for token in ("times", "georgia", "serif")) and "sans" not in lowered:
        return "serif"
    if "narrow" in lowered:
        return "condensed_sans"
    if any(token in lowered for token in ("arial", "helvetica", "verdana", "tahoma", "sans")):
        return "sans"
    return "other"


def _page_context_mode(spec: RenderSpec) -> str:
    if not spec.page.enabled:
        return "none"
    parts = []
    if spec.page.caption_above or spec.page.caption_below:
        parts.append("caption")
    if spec.page.surrounding_paragraph:
        parts.append("paragraph")
    if spec.page.page_header or spec.page.page_footer:
        parts.append("header_footer")
    if spec.page.table_border:
        parts.append("table")
    return "+".join(parts) if parts else "page"


def _compression_bucket(spec: RenderSpec) -> str:
    quality = min(spec.raster.jpeg_quality, spec.degradation.jpeg_quality)
    if spec.raster.output_format == "png" and quality >= 95:
        return "lossless_or_high"
    if quality >= 90:
        return "high"
    if quality >= 75:
        return "medium"
    return "low"


def _recoverability_class(spec: RenderSpec, leakage_status: str, occlusion: float | None) -> str:
    if leakage_status == "failed":
        return "leaky"
    if occlusion is not None and occlusion > 0.15:
        return "occluded"
    if spec.annotations.policy == AnnotationPolicy.DEBUG_LEAKY:
        return "debug_only"
    if spec.degradation.family in {"ood", "screenshot"} or spec.degradation.blur_sigma >= 0.6:
        return "hard_ood"
    if spec.page.enabled or spec.degradation.family == "document":
        return "document_context"
    return "clean"


def _bucket_range(value: float, ranges: tuple[tuple[float, float], ...]) -> str:
    for low, high in ranges:
        if low <= value <= high or (low == 0.0 and abs(value) < 1e-9):
            if low == high:
                return _fmt(low)
            return f"{_fmt(low)}-{_fmt(high) if high < 9999 else 'plus'}"
    return _fmt(value)


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


__all__ = [
    "BLUR_BUCKETS",
    "COUNT_BUCKETS",
    "DPI_BUCKETS",
    "FONT_SIZE_BUCKETS",
    "OCCLUSION_BUCKETS",
    "PERSPECTIVE_BUCKETS",
    "STROKE_WIDTH_BUCKETS",
    "VisualCoverageTracker",
]
