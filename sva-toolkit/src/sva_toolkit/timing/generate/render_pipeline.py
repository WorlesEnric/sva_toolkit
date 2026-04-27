"""Dataset-facing render2 integration for one generated timing record."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import random
from typing import Any

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.core.scenario import ScenarioDocument
from sva_toolkit.timing.generate.model import GenerationError
from sva_toolkit.timing.render2.audit.layout_overflow import audit_layout_overflow
from sva_toolkit.timing.render2.audit.reproducibility import audit_renderer_reproducibility
from sva_toolkit.timing.render2.compose import ComposedRecord, compose_record
from sva_toolkit.timing.render2.pipeline import RenderOutcome, render
from sva_toolkit.timing.render2.profiles import PROFILE_BY_ID, PROFILE_SET_BY_ID, ProfileSet, RenderProfile
from sva_toolkit.timing.render2.protocol import DEFAULT_REGISTRY
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.scene_builder import build_timing_scene
from sva_toolkit.timing.render2.serialization import spec_to_dict
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.render2.spec_sampler import sample_profile, sample_render_spec
from sva_toolkit.timing.visual import TargetPolicy, VisibilityClass, VisibilityReport, lower_to_visual_document


@dataclass(frozen=True)
class DatasetRenderRecord:
    id: str
    seed: int
    semantic_dsl: str
    visual_dsl: str
    scene: TimingScene
    spec: RenderSpec
    outcome: RenderOutcome
    composed: ComposedRecord
    render_spec_payload: dict[str, Any]
    audits_payload: dict[str, Any]
    visibility: dict[str, Any]
    visual_features: dict[str, Any]
    difficulty: dict[str, Any]
    audit_status: dict[str, str]


def generate_one_record(
    *,
    item_id: str,
    seed: int,
    semantic_document: ScenarioDocument,
    rng: random.Random,
    render_profile: str | None = None,
    render_profile_set: str = "train_v2",
    target_policy: str = "visual",
    audit_strict: bool = True,
    style_holdout: Iterable[str] | str | None = None,
    degradation_holdout: Iterable[str] | str | None = None,
    annotation_holdout: Iterable[str] | str | None = None,
) -> DatasetRenderRecord:
    """Lower, render, audit, compose, and describe one generated semantic record."""

    policy = target_policy_from_name(target_policy)
    lowering = lower_to_visual_document(semantic_document, policy)
    visual_document = lowering.visual_document
    semantic_dsl = emit_timing_dsl(semantic_document)
    visual_dsl = emit_timing_dsl(visual_document)
    scene = build_timing_scene(visual_document, semantic_document=semantic_document)

    profile = _choose_profile(rng, render_profile=render_profile, render_profile_set=render_profile_set)
    _enforce_profile_holdouts(
        profile,
        style_holdout=style_holdout,
        degradation_holdout=degradation_holdout,
        annotation_holdout=annotation_holdout,
    )
    spec = sample_render_spec(rng, profile=profile, scene=scene)
    _enforce_spec_holdouts(
        spec,
        profile,
        style_holdout=style_holdout,
        degradation_holdout=degradation_holdout,
        annotation_holdout=annotation_holdout,
    )

    renderer = _renderer_for(scene, spec)
    try:
        outcome = render(scene, spec, target_dsl_text=visual_dsl, enforce_audits=audit_strict)
    except RuntimeError as exc:
        raise GenerationError(str(exc), reason="external_renderer_failed") from exc

    if audit_strict and not outcome.audits_passed:
        raise GenerationError(
            f"render audit failed: {outcome.rejection_reason}",
            reason=outcome.rejection_reason or "render_audit_failed",
        )

    overflow = audit_layout_overflow(scene, outcome.result)
    if audit_strict and not overflow.passed:
        raise GenerationError("render layout overflow", reason="layout_overflow")

    reproducibility = audit_renderer_reproducibility(renderer, scene, spec)
    if audit_strict and not reproducibility.passed:
        raise GenerationError(
            f"renderer reproducibility failed: {reproducibility.reason}",
            reason="external_renderer_failed",
        )

    try:
        composed = compose_record(scene, spec, outcome.result, rng=rng)
    except Exception as exc:
        raise GenerationError(f"record composition failed: {exc}", reason="external_renderer_failed") from exc

    audit_status = _audit_status(outcome, overflow=overflow, reproducibility=reproducibility)
    visibility = _visibility_summary(lowering.visibility, policy)
    visual_features = _visual_features(spec)
    difficulty = _difficulty(spec, outcome, composed)
    audits_payload = {
        "audit_status": audit_status,
        "leakage": _jsonable(outcome.leakage),
        "target_visibility": _jsonable(outcome.target_visibility),
        "contrast": _jsonable(outcome.contrast),
        "occlusion": _jsonable(outcome.occlusion),
        "layout_overflow": _jsonable(overflow),
        "reproducibility": _jsonable(reproducibility),
        "lowering_visibility": visibility_report_to_dict(lowering.visibility),
        "render_warnings": list(outcome.result.warnings),
        "composition_warnings": list(composed.warnings),
    }

    return DatasetRenderRecord(
        id=item_id,
        seed=seed,
        semantic_dsl=semantic_dsl,
        visual_dsl=visual_dsl,
        scene=scene,
        spec=spec,
        outcome=outcome,
        composed=composed,
        render_spec_payload=spec_to_dict(spec),
        audits_payload=audits_payload,
        visibility=visibility,
        visual_features=visual_features,
        difficulty=difficulty,
        audit_status=audit_status,
    )


def profile_set_by_id(profile_set_id: str) -> ProfileSet:
    key = profile_set_id.strip()
    if key in PROFILE_SET_BY_ID:
        return PROFILE_SET_BY_ID[key]
    if key == "test_synthetic_ood":
        return PROFILE_SET_BY_ID["test_ood"]
    raise ValueError(f"unknown render profile set: {profile_set_id}")


def profile_by_id(profile_id: str) -> RenderProfile:
    normalized = _normalized_key(profile_id)
    for profile in PROFILE_BY_ID.values():
        if normalized in {_normalized_key(profile.id), _normalized_key(profile.style_family)}:
            return profile
    raise ValueError(f"unknown render profile: {profile_id}")


def target_policy_from_name(name: str) -> TargetPolicy:
    if name == "visual":
        return TargetPolicy.visual()
    if name == "debug_keep_all":
        return TargetPolicy.debug_keep_all()
    raise ValueError("target_policy must be one of visual, debug_keep_all")


def visibility_report_to_dict(report: VisibilityReport) -> dict[str, Any]:
    return {
        "field_visibility": {
            key: {
                "name": value.name,
                "visibility_class": value.visibility_class.value,
                "rationale": value.rationale,
            }
            for key, value in sorted(report.field_visibility.items())
        },
        "dropped_fields": list(report.dropped_fields),
        "renames": dict(report.renames),
        "kept_bound_labels": dict(report.kept_bound_labels),
        "anchor_visibility": _enum_mapping(report.anchor_visibility),
        "window_visibility": _enum_mapping(report.window_visibility),
        "bound_visibility": _enum_mapping(report.bound_visibility),
        "constraint_visibility": _enum_mapping(report.constraint_visibility),
        "signal_visibility": _enum_mapping(report.signal_visibility),
        "dropped_property_names": list(report.dropped_property_names),
        "dropped_note_count": report.dropped_note_count,
        "notes": list(report.notes),
    }


def _choose_profile(
    rng: random.Random,
    *,
    render_profile: str | None,
    render_profile_set: str,
) -> RenderProfile:
    if render_profile:
        return profile_by_id(render_profile)
    return sample_profile(rng, profile_set_by_id(render_profile_set))


def _renderer_for(scene: TimingScene, spec: RenderSpec):
    try:
        renderer = DEFAULT_REGISTRY.get(spec.renderer_id)
    except KeyError as exc:
        raise GenerationError(
            f"renderer unavailable: {spec.renderer_id}",
            reason="external_renderer_unavailable",
        ) from exc
    if not renderer.supports(scene, spec):
        raise GenerationError(
            f"renderer {spec.renderer_id!r} does not support this scene/spec",
            reason="unsupported_scene_for_renderer",
        )
    return renderer


def _enforce_profile_holdouts(
    profile: RenderProfile,
    *,
    style_holdout: Iterable[str] | str | None,
    degradation_holdout: Iterable[str] | str | None,
    annotation_holdout: Iterable[str] | str | None,
) -> None:
    style_values = _normalized_values(style_holdout)
    if style_values and {_normalized_key(profile.id), _normalized_key(profile.style_family)} & style_values:
        raise GenerationError(f"profile held out: {profile.id}", reason="holdout_style")
    degradation_values = _normalized_values(degradation_holdout)
    if degradation_values and _normalized_key(profile.degradation_family) in degradation_values:
        raise GenerationError(f"degradation held out: {profile.degradation_family}", reason="holdout_degradation")
    annotation_values = _normalized_values(annotation_holdout)
    if annotation_values and _normalized_key(profile.annotation_policy.value) in annotation_values:
        raise GenerationError(f"annotation held out: {profile.annotation_policy.value}", reason="holdout_annotation")


def _enforce_spec_holdouts(
    spec: RenderSpec,
    profile: RenderProfile,
    *,
    style_holdout: Iterable[str] | str | None,
    degradation_holdout: Iterable[str] | str | None,
    annotation_holdout: Iterable[str] | str | None,
) -> None:
    style_values = _normalized_values(style_holdout)
    if style_values and {
        _normalized_key(spec.profile),
        _normalized_key(spec.style.family),
        _normalized_key(profile.id),
        _normalized_key(profile.style_family),
    } & style_values:
        raise GenerationError(f"style held out: {spec.style.family}", reason="holdout_style")
    degradation_values = _normalized_values(degradation_holdout)
    if degradation_values and _normalized_key(spec.degradation.family) in degradation_values:
        raise GenerationError(f"degradation held out: {spec.degradation.family}", reason="holdout_degradation")
    annotation_values = _normalized_values(annotation_holdout)
    if annotation_values and _normalized_key(spec.annotations.policy.value) in annotation_values:
        raise GenerationError(f"annotation held out: {spec.annotations.policy.value}", reason="holdout_annotation")


def _audit_status(outcome: RenderOutcome, *, overflow: object, reproducibility: object) -> dict[str, str]:
    return {
        "leakage": _pass_fail(outcome.leakage),
        "target_visibility": _pass_fail(outcome.target_visibility),
        "contrast": _pass_fail(outcome.contrast),
        "occlusion": _pass_fail(outcome.occlusion),
        "layout_overflow": _pass_fail(overflow),
        "reproducibility": _pass_fail(reproducibility),
    }


def _pass_fail(report: object | None) -> str:
    if report is None:
        return "not_run"
    return "pass" if bool(getattr(report, "passed", False)) else "fail"


def _visibility_summary(report: VisibilityReport, policy: TargetPolicy) -> dict[str, Any]:
    bound_classes = {value.value for value in report.bound_visibility.values()}
    if VisibilityClass.HIDDEN_SEMANTIC.value in bound_classes:
        bounds = VisibilityClass.HIDDEN_SEMANTIC.value
    elif VisibilityClass.VISIBLE_TEXT.value in bound_classes:
        bounds = VisibilityClass.VISIBLE_TEXT.value
    else:
        bounds = VisibilityClass.VISIBLE_GEOMETRY.value
    return {
        "anchor_names": policy.anchor_names.value,
        "bounds": bounds,
        "bus_values": VisibilityClass.VISIBLE_TEXT.value,
        "rule_summaries": "not_rendered" if policy.drop_property_paraphrase else VisibilityClass.DEBUG_OVERLAY.value,
    }


def _visual_features(spec: RenderSpec) -> dict[str, Any]:
    return {
        "renderer_id": spec.renderer_id,
        "profile": spec.profile,
        "style_family": spec.style.family,
        "annotation_policy": spec.annotations.policy.value,
        "degradation_profile": spec.degradation.family,
        "color_mode": spec.style.color_mode,
        "grid_mode": spec.style.grid_mode,
        "bus_style": spec.style.bus_style,
        "unknown_style": spec.style.unknown_style,
        "cut_style": spec.style.cut_style,
        "page_enabled": spec.page.enabled,
        "crop": spec.page.crop_mode,
        "dpi": spec.raster.dpi,
    }


def _difficulty(spec: RenderSpec, outcome: RenderOutcome, composed: ComposedRecord) -> dict[str, Any]:
    occlusion = 0.0
    if outcome.occlusion is not None:
        occlusion = float(outcome.occlusion.max_lane_occlusion)
    contrast = None
    if outcome.contrast is not None:
        contrast = float(outcome.contrast.minimum_contrast)
    return {
        "occlusion": occlusion,
        "contrast": contrast,
        "crop": spec.page.crop_mode if composed.crop_box is None else "cropped",
        "dpi": spec.raster.dpi,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return str(value)


def _enum_mapping(values: Mapping[str, VisibilityClass]) -> dict[str, str]:
    return {key: value.value for key, value in sorted(values.items())}


def _normalized_values(values: Iterable[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = []
        for value in values:
            raw.extend(str(value).split(","))
    return {_normalized_key(value) for value in raw if str(value).strip()}


def _normalized_key(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


__all__ = [
    "DatasetRenderRecord",
    "generate_one_record",
    "profile_by_id",
    "profile_set_by_id",
    "target_policy_from_name",
    "visibility_report_to_dict",
]
