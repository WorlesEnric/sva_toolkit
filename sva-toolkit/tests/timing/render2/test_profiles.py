from __future__ import annotations

from sva_toolkit.timing.render2 import DEFAULT_REGISTRY
from sva_toolkit.timing.render2.profiles import (
    ALL_PROFILES,
    PROFILE_ASCII_RFC,
    PROFILE_CLEAN_WAVEDROM,
    PROFILE_DATASHEET_NATIVE,
    PROFILE_DEBUG_CURRENT,
    PROFILE_DOCUMENT_NATIVE,
    PROFILE_GTKWAVE_OOD,
    PROFILE_NATIVE_RANDOM,
    PROFILE_OOD_NATIVE,
    PROFILE_PLANTUML_OOD,
    PROFILE_SET_TRAIN_V2,
    PROFILE_TIKZ_DATASHEET,
    PROFILE_UNDULATE_RANDOM,
    REGISTERED_RENDERER_IDS,
)


def test_all_profile_constants_exist_and_have_unique_ids() -> None:
    constants = (
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

    assert {profile.id for profile in constants} == {profile.id for profile in ALL_PROFILES}
    assert len({profile.id for profile in constants}) == len(constants)


def test_train_v2_weights_sum_to_one() -> None:
    assert abs(sum(PROFILE_SET_TRAIN_V2.weights) - 1.0) < 1e-6


def test_profile_renderers_are_known_or_registered() -> None:
    registered = {renderer.id for renderer in DEFAULT_REGISTRY.all()}

    for profile in ALL_PROFILES:
        assert profile.renderer_id in REGISTERED_RENDERER_IDS
        if profile.renderer_id in registered:
            assert DEFAULT_REGISTRY.get(profile.renderer_id).id == profile.renderer_id
