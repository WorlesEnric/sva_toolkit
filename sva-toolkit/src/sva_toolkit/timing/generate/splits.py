"""Canonical timing dataset split plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class SplitPlan:
    name: str
    profile_set_id: str
    semantic_holdouts: Mapping[str, frozenset[str]]
    style_holdouts: frozenset[str]
    degradation_holdouts: frozenset[str]
    annotation_holdouts: frozenset[str]


SPLIT_TRAIN_V2 = SplitPlan(
    name="train",
    profile_set_id="train_v2",
    semantic_holdouts={},
    style_holdouts=frozenset({"plantuml-ood", "gtkwave-ood"}),
    degradation_holdouts=frozenset(),
    annotation_holdouts=frozenset(),
)

SPLIT_VAL_SEEN_STYLE = SplitPlan(
    name="val_seen_style",
    profile_set_id="val_seen_style",
    semantic_holdouts={"topology": frozenset({"burst"}), "bound_kind": frozenset({"parameterized"})},
    style_holdouts=frozenset(),
    degradation_holdouts=frozenset(),
    annotation_holdouts=frozenset(),
)

SPLIT_VAL_UNSEEN_STYLE = SplitPlan(
    name="val_unseen_style",
    profile_set_id="val_unseen_style",
    semantic_holdouts={},
    style_holdouts=frozenset({"native-random", "clean-wavedrom", "document-native", "datasheet-native"}),
    degradation_holdouts=frozenset(),
    annotation_holdouts=frozenset(),
)

SPLIT_TEST_SYNTHETIC_OOD = SplitPlan(
    name="test_synthetic_ood",
    profile_set_id="test_ood",
    semantic_holdouts={"size": frozenset({"large"}), "rendering": frozenset({"symbolic"})},
    style_holdouts=frozenset(),
    degradation_holdouts=frozenset(),
    annotation_holdouts=frozenset(),
)

SPLIT_TEST_REAL = SplitPlan(
    name="test_real",
    profile_set_id="test_ood",
    semantic_holdouts={},
    style_holdouts=frozenset(),
    degradation_holdouts=frozenset(),
    annotation_holdouts=frozenset(),
)

SPLIT_PLANS: Mapping[str, SplitPlan] = {
    plan.name: plan
    for plan in (
        SPLIT_TRAIN_V2,
        SPLIT_VAL_SEEN_STYLE,
        SPLIT_VAL_UNSEEN_STYLE,
        SPLIT_TEST_SYNTHETIC_OOD,
        SPLIT_TEST_REAL,
    )
}


def load_real_split_directory(path: str | Path) -> tuple[Path, ...]:
    """Return manually annotated real-diagram records from a user-supplied directory."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"real split directory not found: {root}")
    return tuple(sorted(candidate for candidate in root.iterdir() if candidate.is_file()))


__all__ = [
    "SPLIT_PLANS",
    "SPLIT_TEST_REAL",
    "SPLIT_TEST_SYNTHETIC_OOD",
    "SPLIT_TRAIN_V2",
    "SPLIT_VAL_SEEN_STYLE",
    "SPLIT_VAL_UNSEEN_STYLE",
    "SplitPlan",
    "load_real_split_directory",
]
