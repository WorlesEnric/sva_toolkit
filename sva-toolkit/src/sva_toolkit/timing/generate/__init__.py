"""Procedural Image-DSL dataset generator for the timing diagram DSL."""

from __future__ import annotations

from sva_toolkit.timing.generate.coverage import CoverageTracker
from sva_toolkit.timing.generate.dataset import GenerationRng, generate_dataset
from sva_toolkit.timing.generate.model import (
    GeneratedItem,
    GenerationError,
    GenerationSpec,
)
from sva_toolkit.timing.generate.render_pipeline import DatasetRenderRecord, generate_one_record
from sva_toolkit.timing.generate.splits import (
    SPLIT_PLANS,
    SPLIT_TEST_REAL,
    SPLIT_TEST_SYNTHETIC_OOD,
    SPLIT_TRAIN_V2,
    SPLIT_VAL_SEEN_STYLE,
    SPLIT_VAL_UNSEEN_STYLE,
    SplitPlan,
)
from sva_toolkit.timing.generate.validate_dataset import validate_dataset


__all__ = [
    "CoverageTracker",
    "DatasetRenderRecord",
    "GeneratedItem",
    "GenerationError",
    "GenerationRng",
    "GenerationSpec",
    "SPLIT_PLANS",
    "SPLIT_TEST_REAL",
    "SPLIT_TEST_SYNTHETIC_OOD",
    "SPLIT_TRAIN_V2",
    "SPLIT_VAL_SEEN_STYLE",
    "SPLIT_VAL_UNSEEN_STYLE",
    "SplitPlan",
    "generate_dataset",
    "generate_one_record",
    "validate_dataset",
]
