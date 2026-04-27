"""External renderer adapters for render2."""

from sva_toolkit.timing.render2.adapters.ascii import ASCIIAdapter
from sva_toolkit.timing.render2.adapters.gtkwave import GTKWaveAdapter
from sva_toolkit.timing.render2.adapters.plantuml import PlantUMLAdapter
from sva_toolkit.timing.render2.adapters.registry_bootstrap import bootstrap_external_renderers
from sva_toolkit.timing.render2.adapters.tikz_timing import TikzTimingAdapter
from sva_toolkit.timing.render2.adapters.undulate import UndulateAdapter
from sva_toolkit.timing.render2.adapters.wavedrom import WaveDromAdapter

__all__ = [
    "ASCIIAdapter",
    "GTKWaveAdapter",
    "PlantUMLAdapter",
    "TikzTimingAdapter",
    "UndulateAdapter",
    "WaveDromAdapter",
    "bootstrap_external_renderers",
]
