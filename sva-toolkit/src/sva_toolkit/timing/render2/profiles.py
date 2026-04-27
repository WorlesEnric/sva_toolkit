"""Canonical render2 profile definitions and profile sets.

Profile set ids are:

``train_v2``
    Production training mix: native-random 0.50, clean-wavedrom 0.15,
    undulate-random 0.10, document-native 0.10, datasheet-native 0.07,
    tikz-datasheet 0.05, ascii-rfc 0.03. PlantUML and GTKWave are held out.
``val_seen_style``
    Same renderers and weights as ``train_v2``; callers provide different RNG
    seeds and use the semantic split mechanism for content holdouts.
``val_unseen_style``
    Held-out PlantUML and GTKWave styles only.
``test_ood``
    OOD mix covering tikz-datasheet, plantuml-ood, gtkwave-ood, ascii-rfc,
    and ood-native. The ood-native profile carries the harder native
    degradation/style settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from sva_toolkit.timing.render2.decorations import AnnotationPolicy


REGISTERED_RENDERER_IDS = frozenset(
    {
        "native_svg",
        "wavedrom",
        "undulate",
        "tikz_timing",
        "plantuml",
        "gtkwave",
        "ascii",
    }
)


@dataclass(frozen=True)
class RenderProfile:
    id: str
    description: str
    renderer_id: str
    style_family: str
    annotation_policy: AnnotationPolicy
    color_mode: str
    page_enabled: bool
    degradation_family: str
    style_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotation_policy", AnnotationPolicy(self.annotation_policy))
        object.__setattr__(self, "style_overrides", MappingProxyType(dict(self.style_overrides)))

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class ProfileSet:
    id: str
    profiles: tuple[RenderProfile, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))
        object.__setattr__(self, "weights", tuple(float(weight) for weight in self.weights))
        if len(self.profiles) != len(self.weights):
            raise ValueError("profiles and weights must have the same length")
        if not self.profiles:
            raise ValueError("profile set must contain at least one profile")
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("profile set weights must be non-negative")
        if sum(self.weights) <= 0.0:
            raise ValueError("profile set must have positive total weight")


PROFILE_DEBUG_CURRENT = RenderProfile(
    id="debug-current",
    description="Legacy WaveDrom debug rendering with target-leaky overlays for human inspection only.",
    renderer_id="wavedrom",
    style_family="debug_wavedrom",
    annotation_policy=AnnotationPolicy.DEBUG_LEAKY,
    color_mode="color",
    page_enabled=False,
    degradation_family="none",
    style_overrides={
        "family": "debug-current",
        "grid_mode": "major_only",
        "bus_style": "filled",
        "unknown_style": "x_hatch",
        "cut_style": "ellipsis",
    },
)

PROFILE_CLEAN_WAVEDROM = RenderProfile(
    id="clean-wavedrom",
    description="WaveDrom adapter without semantic overlays or target-leaking labels.",
    renderer_id="wavedrom",
    style_family="wavedrom_clean",
    annotation_policy=AnnotationPolicy.NONE,
    color_mode="color",
    page_enabled=False,
    degradation_family="clean",
    style_overrides={"family": "wavedrom-clean", "grid_mode": "major_only"},
)

PROFILE_NATIVE_RANDOM = RenderProfile(
    id="native-random",
    description="Native SVG randomized vector waveform style with geometric helper guides.",
    renderer_id="native_svg",
    style_family="native_random",
    annotation_policy=AnnotationPolicy.GEOMETRIC_GUIDES,
    color_mode="color",
    page_enabled=False,
    degradation_family="native",
    style_overrides={"family": "native-random"},
)

PROFILE_DATASHEET_NATIVE = RenderProfile(
    id="datasheet-native",
    description="Native SVG datasheet-like style with natural measurement annotations.",
    renderer_id="native_svg",
    style_family="native_datasheet",
    annotation_policy=AnnotationPolicy.NATURAL_MEASUREMENTS,
    color_mode="monochrome",
    page_enabled=False,
    degradation_family="datasheet",
    style_overrides={"family": "datasheet-native", "grid_mode": "major_minor"},
)

PROFILE_DOCUMENT_NATIVE = RenderProfile(
    id="document-native",
    description="Native SVG waveform composed into a document-like page fragment with degradation.",
    renderer_id="native_svg",
    style_family="native_document",
    annotation_policy=AnnotationPolicy.GEOMETRIC_GUIDES,
    color_mode="grayscale",
    page_enabled=True,
    degradation_family="document",
    style_overrides={"family": "document-native"},
)

PROFILE_OOD_NATIVE = RenderProfile(
    id="ood-native",
    description="Native SVG OOD style with exotic visual conventions and dense annotations.",
    renderer_id="native_svg",
    style_family="native_ood",
    annotation_policy=AnnotationPolicy.NATURAL_MEASUREMENTS,
    color_mode="low_contrast",
    page_enabled=False,
    degradation_family="ood",
    style_overrides={
        "family": "ood-native",
        "grid_mode": "dense",
        "bus_style": "hatched",
        "unknown_style": "orange_hatch",
        "cut_style": "double_slash",
        "transition_shape": "curved",
    },
)

PROFILE_UNDULATE_RANDOM = RenderProfile(
    id="undulate-random",
    description="Undulate renderer with randomized helper-line-rich geometric guides.",
    renderer_id="undulate",
    style_family="undulate_random",
    annotation_policy=AnnotationPolicy.GEOMETRIC_GUIDES,
    color_mode="color",
    page_enabled=False,
    degradation_family="clean",
    style_overrides={"family": "undulate-random", "grid_mode": "major_minor"},
)

PROFILE_TIKZ_DATASHEET = RenderProfile(
    id="tikz-datasheet",
    description="tikz-timing LaTeX/datasheet style with natural measurement semantics.",
    renderer_id="tikz_timing",
    style_family="tikz_datasheet",
    annotation_policy=AnnotationPolicy.NATURAL_MEASUREMENTS,
    color_mode="monochrome",
    page_enabled=False,
    degradation_family="datasheet",
    style_overrides={
        "family": "tikz-datasheet",
        "font_family": "Times New Roman, Times, serif",
        "label_font_family": "Times New Roman, Times, serif",
        "grid_mode": "major_minor",
        "bus_style": "empty",
    },
)

PROFILE_PLANTUML_OOD = RenderProfile(
    id="plantuml-ood",
    description="Held-out PlantUML timing style for OOD evaluation.",
    renderer_id="plantuml",
    style_family="plantuml_ood",
    annotation_policy=AnnotationPolicy.NONE,
    color_mode="color",
    page_enabled=False,
    degradation_family="ood",
    style_overrides={"family": "plantuml-ood", "grid_mode": "sparse", "bus_style": "boxed"},
)

PROFILE_GTKWAVE_OOD = RenderProfile(
    id="gtkwave-ood",
    description="Held-out GTKWave screenshot style for OOD evaluation.",
    renderer_id="gtkwave",
    style_family="gtkwave_ood",
    annotation_policy=AnnotationPolicy.NONE,
    color_mode="dark",
    page_enabled=False,
    degradation_family="screenshot",
    style_overrides={
        "family": "gtkwave-ood",
        "font_family": "Consolas, Courier New, monospace",
        "label_font_family": "Consolas, Courier New, monospace",
        "grid_mode": "dense",
        "bus_style": "inline_text",
        "unknown_style": "gray_block",
    },
)

PROFILE_ASCII_RFC = RenderProfile(
    id="ascii-rfc",
    description="Plain-text RFC/comment-style monospace waveform rendering.",
    renderer_id="ascii",
    style_family="ascii_rfc",
    annotation_policy=AnnotationPolicy.NONE,
    color_mode="monochrome",
    page_enabled=False,
    degradation_family="plain_text",
    style_overrides={
        "family": "ascii-rfc",
        "font_family": "Courier New, Courier, monospace",
        "label_font_family": "Courier New, Courier, monospace",
        "grid_mode": "none",
        "bus_style": "empty",
        "unknown_style": "dashed_outline",
    },
)


ALL_PROFILES = (
    PROFILE_DEBUG_CURRENT,
    PROFILE_CLEAN_WAVEDROM,
    PROFILE_NATIVE_RANDOM,
    PROFILE_DATASHEET_NATIVE,
    PROFILE_DOCUMENT_NATIVE,
    PROFILE_OOD_NATIVE,
    PROFILE_UNDULATE_RANDOM,
    PROFILE_TIKZ_DATASHEET,
    PROFILE_PLANTUML_OOD,
    PROFILE_GTKWAVE_OOD,
    PROFILE_ASCII_RFC,
)


PROFILE_SET_TRAIN_V2 = ProfileSet(
    id="train_v2",
    profiles=(
        PROFILE_NATIVE_RANDOM,
        PROFILE_CLEAN_WAVEDROM,
        PROFILE_UNDULATE_RANDOM,
        PROFILE_DOCUMENT_NATIVE,
        PROFILE_DATASHEET_NATIVE,
        PROFILE_TIKZ_DATASHEET,
        PROFILE_ASCII_RFC,
    ),
    weights=(0.50, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03),
)

PROFILE_SET_VAL_SEEN_STYLE = ProfileSet(
    id="val_seen_style",
    profiles=PROFILE_SET_TRAIN_V2.profiles,
    weights=PROFILE_SET_TRAIN_V2.weights,
)

PROFILE_SET_VAL_UNSEEN_STYLE = ProfileSet(
    id="val_unseen_style",
    profiles=(PROFILE_PLANTUML_OOD, PROFILE_GTKWAVE_OOD),
    weights=(0.50, 0.50),
)

PROFILE_SET_TEST_OOD = ProfileSet(
    id="test_ood",
    profiles=(
        PROFILE_TIKZ_DATASHEET,
        PROFILE_PLANTUML_OOD,
        PROFILE_GTKWAVE_OOD,
        PROFILE_ASCII_RFC,
        PROFILE_OOD_NATIVE,
    ),
    weights=(0.20, 0.20, 0.20, 0.15, 0.25),
)


PROFILE_BY_ID = {profile.id: profile for profile in ALL_PROFILES}
PROFILE_SET_BY_ID = {
    profile_set.id: profile_set
    for profile_set in (
        PROFILE_SET_TRAIN_V2,
        PROFILE_SET_VAL_SEEN_STYLE,
        PROFILE_SET_VAL_UNSEEN_STYLE,
        PROFILE_SET_TEST_OOD,
    )
}


__all__ = [
    "ALL_PROFILES",
    "PROFILE_ASCII_RFC",
    "PROFILE_BY_ID",
    "PROFILE_CLEAN_WAVEDROM",
    "PROFILE_DATASHEET_NATIVE",
    "PROFILE_DEBUG_CURRENT",
    "PROFILE_DOCUMENT_NATIVE",
    "PROFILE_GTKWAVE_OOD",
    "PROFILE_NATIVE_RANDOM",
    "PROFILE_OOD_NATIVE",
    "PROFILE_PLANTUML_OOD",
    "PROFILE_SET_BY_ID",
    "PROFILE_SET_TEST_OOD",
    "PROFILE_SET_TRAIN_V2",
    "PROFILE_SET_VAL_SEEN_STYLE",
    "PROFILE_SET_VAL_UNSEEN_STYLE",
    "PROFILE_TIKZ_DATASHEET",
    "PROFILE_UNDULATE_RANDOM",
    "REGISTERED_RENDERER_IDS",
    "ProfileSet",
    "RenderProfile",
]
