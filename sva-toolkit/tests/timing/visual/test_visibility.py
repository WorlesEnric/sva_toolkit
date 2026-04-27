from __future__ import annotations


def test_public_api_imports() -> None:
    from sva_toolkit.timing.visual import (  # noqa: PLC0415
        AnchorNamePolicy,
        BoundPolicy,
        ConstraintNamePolicy,
        FieldVisibility,
        LoweringResult,
        TargetPolicy,
        VisibilityClass,
        VisibilityReport,
        WindowNamePolicy,
        lower_to_visual_document,
    )

    assert AnchorNamePolicy.CANONICAL_VISUAL.value == "canonical_visual"
    assert WindowNamePolicy.CANONICAL_VISUAL.value == "canonical_visual"
    assert ConstraintNamePolicy.CANONICAL_VISUAL.value == "canonical_visual"
    assert BoundPolicy.KEEP_ALL.value == "keep_all"
    assert TargetPolicy.visual().drop_property_paraphrase
    assert FieldVisibility.__name__ == "FieldVisibility"
    assert VisibilityClass.VISIBLE_GEOMETRY.value == "visible_geometry"
    assert VisibilityReport.__name__ == "VisibilityReport"
    assert LoweringResult.__name__ == "LoweringResult"
    assert callable(lower_to_visual_document)


def test_visibility_class_members_are_exact_contract() -> None:
    from sva_toolkit.timing.visual import VisibilityClass  # noqa: PLC0415

    assert tuple(member.value for member in VisibilityClass) == (
        "visible_geometry",
        "visible_text",
        "visible_convention",
        "hidden_semantic",
        "debug_overlay",
    )

