"""Audit-aware render2 convenience pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.render2.audit.contrast import (
    ContrastReport,
    OcclusionReport,
    audit_minimum_contrast,
    audit_occlusion,
)
from sva_toolkit.timing.render2.audit.leakage import LeakageAuditReport, audit_rendered_text
from sva_toolkit.timing.render2.audit.target_visibility import TargetVisibilityReport, audit_target_visibility
from sva_toolkit.timing.render2.protocol import DEFAULT_REGISTRY
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec


@dataclass(frozen=True)
class RenderOutcome:
    result: RenderResult
    leakage: LeakageAuditReport | None
    target_visibility: TargetVisibilityReport | None
    contrast: ContrastReport | None
    occlusion: OcclusionReport | None
    audits_passed: bool
    rejection_reason: str | None


def render(
    scene: TimingScene,
    spec: RenderSpec,
    *,
    target_dsl_text: str | None = None,
    enforce_audits: bool = True,
) -> RenderOutcome:
    """Render a scene with the registered renderer and run metadata audits."""

    renderer = DEFAULT_REGISTRY.get(spec.renderer_id)
    if not renderer.supports(scene, spec):
        raise RuntimeError(f"renderer '{spec.renderer_id}' does not support this scene/spec")

    result = renderer.render(scene, spec)
    target_text = target_dsl_text or _target_text_from_scene(scene)

    leakage = audit_rendered_text(scene, result, target_dsl_text=target_text) if target_text is not None else None
    target_visibility = audit_target_visibility(scene, result)
    contrast = audit_minimum_contrast(result)
    occlusion = audit_occlusion(result)

    rejection_reason = _rejection_reason(leakage, target_visibility, contrast, occlusion) if enforce_audits else None
    return RenderOutcome(
        result=result,
        leakage=leakage,
        target_visibility=target_visibility,
        contrast=contrast,
        occlusion=occlusion,
        audits_passed=True if not enforce_audits else rejection_reason is None,
        rejection_reason=rejection_reason,
    )


def _target_text_from_scene(scene: TimingScene) -> str | None:
    document = scene.visible_target or scene.semantic_document
    return emit_timing_dsl(document) if document is not None else None


def _rejection_reason(
    leakage: LeakageAuditReport | None,
    target_visibility: TargetVisibilityReport | None,
    contrast: ContrastReport | None,
    occlusion: OcclusionReport | None,
) -> str | None:
    if leakage is not None and not leakage.passed:
        return "render_text_leakage"
    if target_visibility is not None and not target_visibility.passed:
        return "target_not_visible"
    if contrast is not None and not contrast.passed:
        return "low_contrast"
    if occlusion is not None and not occlusion.passed:
        return "required_bus_value_occluded"
    return None
