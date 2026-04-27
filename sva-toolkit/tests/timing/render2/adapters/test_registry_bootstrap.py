from __future__ import annotations

import importlib.util

import pytest

from sva_toolkit.timing.render2 import DEFAULT_REGISTRY, bootstrap_external_renderers


def test_registry_bootstrap_reports_every_external_adapter_and_keeps_core_renderers() -> None:
    before = tuple(renderer.id for renderer in DEFAULT_REGISTRY.all())

    statuses = bootstrap_external_renderers()
    after = tuple(renderer.id for renderer in DEFAULT_REGISTRY.all())
    statuses_again = bootstrap_external_renderers()

    assert set(statuses) == {"undulate", "tikz_timing", "plantuml", "gtkwave", "ascii"}
    assert all(
        status == "registered"
        or status.startswith("missing_dependency:")
        or status.startswith("missing_executable:")
        or status.startswith("failed:")
        for status in statuses.values()
    )
    assert DEFAULT_REGISTRY.get("native_svg").id == "native_svg"
    if importlib.util.find_spec("wavedrom") is not None:
        assert DEFAULT_REGISTRY.get("wavedrom").id == "wavedrom"
    else:
        with pytest.raises(KeyError):
            DEFAULT_REGISTRY.get("wavedrom")
    assert len(after) == len(set(after))
    assert tuple(renderer.id for renderer in DEFAULT_REGISTRY.all()) == after
    assert statuses_again["ascii"] == "registered"
    assert set(before) <= set(after)
