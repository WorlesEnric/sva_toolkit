"""Visibility classifications for the visual timing DSL contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class VisibilityClass(str, Enum):
    """Recoverability class for visual-target fields."""

    VISIBLE_GEOMETRY = "visible_geometry"
    VISIBLE_TEXT = "visible_text"
    VISIBLE_CONVENTION = "visible_convention"
    HIDDEN_SEMANTIC = "hidden_semantic"
    DEBUG_OVERLAY = "debug_overlay"


@dataclass(frozen=True)
class FieldVisibility:
    """Visibility rationale for one lowered field."""

    name: str
    visibility_class: VisibilityClass
    rationale: str


@dataclass(frozen=True)
class VisibilityReport:
    """Visibility metadata produced while lowering a semantic document."""

    field_visibility: Mapping[str, FieldVisibility] = field(default_factory=dict)
    dropped_fields: tuple[str, ...] = ()
    renames: Mapping[str, str] = field(default_factory=dict)
    kept_bound_labels: Mapping[str, str] = field(default_factory=dict)
    anchor_visibility: Mapping[str, VisibilityClass] = field(default_factory=dict)
    window_visibility: Mapping[str, VisibilityClass] = field(default_factory=dict)
    bound_visibility: Mapping[str, VisibilityClass] = field(default_factory=dict)
    constraint_visibility: Mapping[str, VisibilityClass] = field(default_factory=dict)
    signal_visibility: Mapping[str, VisibilityClass] = field(default_factory=dict)
    dropped_property_names: tuple[str, ...] = ()
    dropped_note_count: int = 0
    notes: tuple[str, ...] = ()

    @property
    def per_field(self) -> Mapping[str, FieldVisibility]:
        """Compatibility alias for the full per-field visibility map."""

        return self.field_visibility
