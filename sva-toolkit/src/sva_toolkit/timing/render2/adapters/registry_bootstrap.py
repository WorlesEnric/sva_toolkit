"""Bootstrap optional external render2 adapters."""

from __future__ import annotations

import importlib

from sva_toolkit.timing.render2.protocol import DEFAULT_REGISTRY, RendererRegistry


_ADAPTERS: tuple[tuple[str, str, str], ...] = (
    ("undulate", "sva_toolkit.timing.render2.adapters.undulate", "UndulateAdapter"),
    ("tikz_timing", "sva_toolkit.timing.render2.adapters.tikz_timing", "TikzTimingAdapter"),
    ("plantuml", "sva_toolkit.timing.render2.adapters.plantuml", "PlantUMLAdapter"),
    ("gtkwave", "sva_toolkit.timing.render2.adapters.gtkwave", "GTKWaveAdapter"),
    ("ascii", "sva_toolkit.timing.render2.adapters.ascii", "ASCIIAdapter"),
)


def bootstrap_external_renderers(registry: RendererRegistry = DEFAULT_REGISTRY) -> dict[str, str]:
    """Try to import and register every external adapter; return id-to-status."""

    statuses: dict[str, str] = {}
    for adapter_id, module_name, class_name in _ADAPTERS:
        try:
            module = importlib.import_module(module_name)
            dependency_status = getattr(module, "dependency_status", lambda: None)()
            if dependency_status is not None:
                statuses[adapter_id] = dependency_status
                continue
            try:
                registry.get(adapter_id)
            except KeyError:
                registry.register(getattr(module, class_name)())
            statuses[adapter_id] = "registered"
        except Exception as exc:
            statuses[adapter_id] = f"failed:{exc}"
    return statuses
