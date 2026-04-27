from __future__ import annotations

import random
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import AnnotationPolicy, DecorationKind, build_timing_scene, select_decorations
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_annotation_policy_none_returns_empty_tuple() -> None:
    scene = _scene()

    assert select_decorations(scene, AnnotationPolicy.NONE, random.Random(1)) == ()


def test_nuisance_only_policy_returns_only_nuisance_kinds() -> None:
    scene = _scene()
    allowed = {
        DecorationKind.NUISANCE_TEXT,
        DecorationKind.HANDDRAWN_MARK,
        DecorationKind.HIGHLIGHT_REGION,
    }

    selected = select_decorations(scene, AnnotationPolicy.NUISANCE_ONLY, random.Random(2))

    assert all(decoration.kind in allowed for decoration in selected)
    assert all(not decoration.semantic for decoration in selected)


def test_geometric_guides_policy_excludes_nuisance_text() -> None:
    scene = _scene()
    allowed = {
        DecorationKind.VERTICAL_GUIDE,
        DecorationKind.HORIZONTAL_GUIDE,
        DecorationKind.MEASUREMENT_BRACKET,
    }

    selected = select_decorations(scene, AnnotationPolicy.GEOMETRIC_GUIDES, random.Random(3))

    assert selected
    assert all(decoration.kind in allowed for decoration in selected)
    assert all(decoration.kind != DecorationKind.NUISANCE_TEXT for decoration in selected)


def test_decoration_selection_is_deterministic_for_same_seed() -> None:
    scene = _scene()

    left = select_decorations(scene, AnnotationPolicy.GEOMETRIC_GUIDES, random.Random(99))
    right = select_decorations(scene, AnnotationPolicy.GEOMETRIC_GUIDES, random.Random(99))

    assert left == right


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "07_symbolic_pipeline.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual)
