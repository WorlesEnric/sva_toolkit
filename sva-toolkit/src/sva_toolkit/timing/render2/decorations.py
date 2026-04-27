"""Annotation and decoration model for renderer-independent timing scenes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from sva_toolkit.timing.render2.primitives import Fill, FontSpec, Stroke
from sva_toolkit.timing.visual import VisibilityClass

if TYPE_CHECKING:
    from sva_toolkit.timing.render2.scene import TimingScene


class DecorationKind(str, Enum):
    VERTICAL_GUIDE = "vertical_guide"
    HORIZONTAL_GUIDE = "horizontal_guide"
    MEASUREMENT_BRACKET = "measurement_bracket"
    CALLOUT_ARROW = "callout_arrow"
    HIGHLIGHT_REGION = "highlight_region"
    CAPTION = "caption"
    NUISANCE_TEXT = "nuisance_text"
    HANDDRAWN_MARK = "handdrawn_mark"


class AnnotationPolicy(str, Enum):
    NONE = "none"
    NUISANCE_ONLY = "nuisance_only"
    GEOMETRIC_GUIDES = "geometric_guides"
    NATURAL_MEASUREMENTS = "natural_measurements"
    DEBUG_LEAKY = "debug_leaky"


@dataclass(frozen=True)
class DecorationStyle:
    stroke: Stroke | None = None
    fill: Fill | None = None
    font: FontSpec | None = None
    dashed: bool = False
    handdrawn: bool = False


@dataclass(frozen=True)
class Decoration:
    kind: DecorationKind
    semantic: bool
    target_ref: str | None
    text: str | None = None
    visibility_class: VisibilityClass = VisibilityClass.VISIBLE_GEOMETRY
    anchor_tick: int | None = None
    span: tuple[int, int] | None = None
    lane_names: tuple[str, ...] = ()
    style: DecorationStyle = DecorationStyle()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", DecorationKind(self.kind))
        object.__setattr__(self, "visibility_class", VisibilityClass(self.visibility_class))
        object.__setattr__(self, "lane_names", tuple(self.lane_names))
        if self.span is not None:
            object.__setattr__(self, "span", tuple(self.span))


_NUISANCE_KINDS = frozenset(
    {
        DecorationKind.NUISANCE_TEXT,
        DecorationKind.HANDDRAWN_MARK,
        DecorationKind.HIGHLIGHT_REGION,
    }
)
_GEOMETRIC_KINDS = frozenset(
    {
        DecorationKind.VERTICAL_GUIDE,
        DecorationKind.HORIZONTAL_GUIDE,
        DecorationKind.MEASUREMENT_BRACKET,
    }
)
_NATURAL_KINDS = frozenset(
    {
        DecorationKind.VERTICAL_GUIDE,
        DecorationKind.HORIZONTAL_GUIDE,
        DecorationKind.MEASUREMENT_BRACKET,
        DecorationKind.CALLOUT_ARROW,
        DecorationKind.HIGHLIGHT_REGION,
        DecorationKind.CAPTION,
    }
)


def select_decorations(scene: TimingScene, policy: AnnotationPolicy, rng: Any) -> tuple[Decoration, ...]:
    """Select decorations allowed by an annotation policy.

    This is intentionally a deterministic filter. Later renderer phases can use
    ``rng`` to sample from the returned safe candidate set.
    """

    del rng
    policy = AnnotationPolicy(policy)
    if policy == AnnotationPolicy.NONE:
        return ()
    if policy == AnnotationPolicy.DEBUG_LEAKY:
        return scene.decorations
    if policy == AnnotationPolicy.NUISANCE_ONLY:
        return tuple(
            decoration
            for decoration in scene.decorations
            if not decoration.semantic and decoration.kind in _NUISANCE_KINDS
        )
    if policy == AnnotationPolicy.GEOMETRIC_GUIDES:
        return tuple(
            decoration
            for decoration in scene.decorations
            if decoration.semantic and decoration.kind in _GEOMETRIC_KINDS
        )
    if policy == AnnotationPolicy.NATURAL_MEASUREMENTS:
        return tuple(
            decoration
            for decoration in scene.decorations
            if decoration.semantic and decoration.kind in _NATURAL_KINDS
        )
    return ()
