"""Policies for lowering semantic timing diagrams to visual targets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnchorNamePolicy(str, Enum):
    """How anchor names are represented in the visual target."""

    KEEP_ORIGINAL = "keep_original"
    CANONICAL_VISUAL = "canonical_visual"


class WindowNamePolicy(str, Enum):
    """How time-window names are represented in the visual target."""

    KEEP_ORIGINAL = "keep_original"
    CANONICAL_VISUAL = "canonical_visual"


class ConstraintNamePolicy(str, Enum):
    """How lane-constraint names are represented in the visual target."""

    KEEP_ORIGINAL = "keep_original"
    CANONICAL_VISUAL = "canonical_visual"


class BoundPolicy(str, Enum):
    """How timing bounds are retained in the visual target."""

    KEEP_ALL = "keep_all"
    DROP_PARAMETERIZED = "drop_parameterized"
    GEOMETRY_ONLY = "geometry_only"


@dataclass(frozen=True)
class TargetPolicy:
    """Lowering policy for image-recoverable timing DSL targets."""

    anchor_names: AnchorNamePolicy = AnchorNamePolicy.CANONICAL_VISUAL
    window_names: WindowNamePolicy = WindowNamePolicy.CANONICAL_VISUAL
    constraint_names: ConstraintNamePolicy = ConstraintNamePolicy.CANONICAL_VISUAL
    bounds: BoundPolicy = BoundPolicy.KEEP_ALL
    drop_property_paraphrase: bool = True
    drop_notes: bool = True
    drop_bundle_metadata: bool = True

    @classmethod
    def visual(cls) -> "TargetPolicy":
        """Default policy for supervised visual targets."""

        return cls()

    @classmethod
    def debug_keep_all(cls) -> "TargetPolicy":
        """Debug policy that preserves semantic/debug metadata and names."""

        return cls(
            anchor_names=AnchorNamePolicy.KEEP_ORIGINAL,
            window_names=WindowNamePolicy.KEEP_ORIGINAL,
            constraint_names=ConstraintNamePolicy.KEEP_ORIGINAL,
            bounds=BoundPolicy.KEEP_ALL,
            drop_property_paraphrase=False,
            drop_notes=False,
            drop_bundle_metadata=False,
        )

