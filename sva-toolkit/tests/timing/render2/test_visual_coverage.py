from __future__ import annotations

import json
import random
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import build_timing_scene
from sva_toolkit.timing.render2.profiles import PROFILE_SET_TRAIN_V2
from sva_toolkit.timing.render2.spec_sampler import sample_profile, sample_render_spec
from sva_toolkit.timing.render2.visual_coverage import VisualCoverageTracker
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_visual_coverage_populates_every_axis_after_train_v2_ingestions() -> None:
    rng = random.Random(2024)
    scene = _scene()
    tracker = VisualCoverageTracker()

    for _ in range(1000):
        profile = sample_profile(rng, PROFILE_SET_TRAIN_V2)
        spec = sample_render_spec(rng, profile=profile, scene=scene)
        tracker.update(spec, None, scene)

    buckets = tracker.buckets()
    assert all(buckets[axis] for axis in VisualCoverageTracker.AXES)
    assert not tracker.is_axis_deficient("renderer_id", 4)
    json.dumps(tracker.to_dict(), sort_keys=True)


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)
