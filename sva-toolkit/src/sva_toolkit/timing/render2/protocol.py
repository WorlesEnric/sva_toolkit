"""Timing renderer protocol and registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sva_toolkit.timing.render2.decorations import AnnotationPolicy, DecorationKind
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import LaneType, TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec


KNOWN_CAPABILITIES = frozenset(
    {
        "bit",
        "bus",
        "clock",
        "cuts",
        "unknown",
        "high_z",
        "annotations",
        "style_overrides",
        "measurement_brackets",
        "page_composition",
        "raster",
        "vector_text",
        "vcd_input",
        "ascii_only",
        "latex_only",
        "wavejson_subset",
    }
)


@runtime_checkable
class TimingRenderer(Protocol):
    id: str
    capabilities: frozenset[str]

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool: ...

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult: ...


class RendererRegistry:
    def __init__(self) -> None:
        self._renderers: dict[str, TimingRenderer] = {}

    def register(self, renderer: TimingRenderer) -> None:
        capabilities = frozenset(renderer.capabilities)
        unknown = capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(f"unknown renderer capabilities: {', '.join(sorted(unknown))}")
        if renderer.id in self._renderers:
            raise ValueError(f"renderer already registered: {renderer.id}")
        self._renderers[renderer.id] = renderer

    def get(self, renderer_id: str) -> TimingRenderer:
        return self._renderers[renderer_id]

    def all(self) -> tuple[TimingRenderer, ...]:
        return tuple(self._renderers.values())

    def supporting(self, scene: TimingScene, spec: RenderSpec) -> tuple[TimingRenderer, ...]:
        required = _required_capabilities(scene, spec)
        return tuple(
            renderer
            for renderer in self._renderers.values()
            if required <= renderer.capabilities and renderer.supports(scene, spec)
        )


def _required_capabilities(scene: TimingScene, spec: RenderSpec) -> frozenset[str]:
    required: set[str] = set()
    for lane in scene.lanes:
        if lane.lane_type == LaneType.CLOCK:
            required.add("clock")
        elif lane.lane_type == LaneType.BIT:
            required.add("bit")
        elif lane.lane_type == LaneType.BUS:
            required.add("bus")
        elif lane.lane_type == LaneType.UNKNOWN:
            required.add("unknown")
        elif lane.lane_type == LaneType.HIGH_Z:
            required.add("high_z")
        if any(run.is_unknown for run in lane.runs):
            required.add("unknown")
        if any(run.is_high_z for run in lane.runs):
            required.add("high_z")
    if scene.cuts:
        required.add("cuts")
    if spec.annotations.policy != AnnotationPolicy.NONE:
        required.add("annotations")
    if spec.annotations.policy != AnnotationPolicy.NONE and any(
        decoration.kind == DecorationKind.MEASUREMENT_BRACKET for decoration in scene.decorations
    ):
        required.add("measurement_brackets")
    if spec.page.enabled:
        required.add("page_composition")
    return frozenset(required)


DEFAULT_REGISTRY = RendererRegistry()
