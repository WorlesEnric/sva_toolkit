"""Contrast and occlusion audits for render2 results."""

from __future__ import annotations

from dataclasses import dataclass

from sva_toolkit.timing.render2.result import RenderResult


@dataclass(frozen=True)
class ContrastReport:
    minimum_contrast: float
    threshold: float
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OcclusionReport:
    occluded_lane_fractions: dict[str, float]
    max_lane_occlusion: float
    passed: bool
    reasons: tuple[str, ...]


def audit_minimum_contrast(render_result: RenderResult) -> ContrastReport:
    minimum = render_result.visibility.minimum_contrast
    threshold = 0.45
    passed = minimum >= threshold
    return ContrastReport(
        minimum_contrast=minimum,
        threshold=threshold,
        passed=passed,
        reasons=() if passed else ("low_contrast",),
    )


def audit_occlusion(render_result: RenderResult, *, max_lane_occlusion: float = 0.15) -> OcclusionReport:
    fractions = dict(render_result.visibility.occluded_lane_fractions)
    passed = all(fraction <= max_lane_occlusion for fraction in fractions.values())
    return OcclusionReport(
        occluded_lane_fractions=fractions,
        max_lane_occlusion=max_lane_occlusion,
        passed=passed,
        reasons=() if passed else ("required_bus_value_occluded",),
    )
