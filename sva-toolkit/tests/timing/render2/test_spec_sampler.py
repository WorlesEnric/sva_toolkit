from __future__ import annotations

import math
import random
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import AnnotationPolicy, build_timing_scene
from sva_toolkit.timing.render2.profiles import PROFILE_NATIVE_RANDOM, PROFILE_SET_TRAIN_V2
from sva_toolkit.timing.render2.serialization import spec_to_dict
from sva_toolkit.timing.render2.spec_sampler import sample_profile, sample_render_spec
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_sample_profile_matches_train_v2_weights_within_binomial_bounds() -> None:
    draws = 200
    rng = random.Random(7321)
    counts = {profile.id: 0 for profile in PROFILE_SET_TRAIN_V2.profiles}

    for _ in range(draws):
        counts[sample_profile(rng, PROFILE_SET_TRAIN_V2).id] += 1

    for profile, weight in zip(PROFILE_SET_TRAIN_V2.profiles, PROFILE_SET_TRAIN_V2.weights, strict=True):
        expected = draws * weight
        sigma = math.sqrt(draws * weight * (1.0 - weight))
        tolerance = max(1.0, 3.0 * sigma)
        assert abs(counts[profile.id] - expected) <= tolerance


def test_sample_native_random_spec_has_profile_policy_and_renderer() -> None:
    spec = sample_render_spec(random.Random(13), profile=PROFILE_NATIVE_RANDOM, scene=_scene())

    assert spec.renderer_id == "native_svg"
    assert spec.profile == PROFILE_NATIVE_RANDOM.id
    assert spec.annotations.policy == AnnotationPolicy.GEOMETRIC_GUIDES


def test_sample_render_spec_is_deterministic_for_same_rng_seed() -> None:
    scene = _scene()

    left = sample_render_spec(random.Random(91), profile=PROFILE_NATIVE_RANDOM, scene=scene)
    right = sample_render_spec(random.Random(91), profile=PROFILE_NATIVE_RANDOM, scene=scene)

    assert spec_to_dict(left) == spec_to_dict(right)


def _scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return build_timing_scene(visual, semantic_document=document)
