"""Undulate adapter.

Upstream tool: ``undulate`` by LudwigCRON, a Python WaveJSON-compatible timing
diagram renderer.
Install hint: ``python -m pip install git+https://github.com/LudwigCRON/undulate.git``.
Supported capabilities: bit, bus, clock, cuts, annotations, style_overrides,
vector_text.

The pinned call site is ``undulate.renderers.svgrenderer.SvgRenderer.draw``.
Undulate writes SVG to a filename, so this adapter renders into a temporary SVG
file and reads it back before applying render2's non-leaky decoration layer.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

from sva_toolkit.timing.render.wavedrom import _encode_bit_wave, _encode_bus_wave
from sva_toolkit.timing.render2.adapters._common import (
    add_decorations_to_svg,
    base_layout,
    bbox_by_role,
    collect_svg_text,
    lane_display_name,
    samples_from_runs,
    supports_scene,
    svg_size,
    visibility_report,
    with_bbox_by_role,
)
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import LaneScene, LaneType, TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec


class UndulateAdapter:
    id = "undulate"
    capabilities = frozenset({"bit", "bus", "clock", "cuts", "annotations", "style_overrides", "vector_text"})

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool:
        return supports_scene(self.id, self.capabilities, scene, spec, dependency_available=_undulate_available())

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult:
        if not _undulate_available():
            raise RuntimeError(
                "undulate is not installed; install with "
                "'python -m pip install git+https://github.com/LudwigCRON/undulate.git'"
            )
        if not self.supports(scene, spec):
            raise RuntimeError("undulate adapter does not support this scene/spec")

        svg_text = _render_undulate_svg(_build_undulate_source(scene), spec)
        root = ET.fromstring(svg_text)
        layout = base_layout(scene, spec)
        width, height = svg_size(root, layout)
        layout = base_layout(scene, spec, width=width or layout.width, height=height or layout.height)
        decorations = add_decorations_to_svg(root, scene, spec, layout)
        rendered_text = collect_svg_text(root, scene, spec)
        layout = with_bbox_by_role(layout, bbox_by_role(rendered_text, decorations))
        return RenderResult(
            svg_text=ET.tostring(root, encoding="unicode", short_empty_elements=True),
            png_bytes=None,
            layout=layout,
            visibility=visibility_report(scene, rendered_text),
            render_spec=replace(spec, renderer_id=self.id),
            warnings=(),
        )


def dependency_status() -> str | None:
    return None if _undulate_available() else "missing_dependency:undulate"


def _undulate_available() -> bool:
    return importlib.util.find_spec("undulate") is not None


def _build_undulate_source(scene: TimingScene) -> dict:
    return {
        "signal": [_build_lane_signal(lane, scene.ticks.total_ticks) for lane in scene.lanes],
        "head": {"tick": 0},
    }


def _build_lane_signal(lane: LaneScene, total_ticks: int) -> dict:
    samples = samples_from_runs(lane, total_ticks)
    signal = {"name": lane_display_name(lane)}
    if lane.lane_type in {LaneType.BIT, LaneType.CLOCK}:
        signal["wave"] = _encode_bit_wave(samples)
        return signal
    wave, data = _encode_bus_wave(samples)
    signal["wave"] = wave
    if data:
        signal["data"] = data
    return signal


def _render_undulate_svg(source: dict, spec: RenderSpec) -> str:
    _initialize_undulate_bricks()
    module = importlib.import_module("undulate.renderers.svgrenderer")
    renderer_cls = getattr(module, "SvgRenderer")
    renderer = renderer_cls()
    with tempfile.TemporaryDirectory(prefix="sva-undulate-") as tmpdir:
        output = Path(tmpdir) / "diagram.svg"
        renderer.draw(
            source,
            brick_height=max(12, int(spec.layout.lane_height)),
            brick_width=max(16, int(spec.layout.tick_width)),
            is_reg=False,
            filename=str(output),
        )
        return output.read_text(encoding="utf-8")


def _initialize_undulate_bricks() -> None:
    for module_name in (
        "undulate.bricks.analogue",
        "undulate.bricks.digital",
        "undulate.bricks.register",
        "undulate.bricks.shape",
    ):
        module = importlib.import_module(module_name)
        initialize = getattr(module, "initialize", None)
        if callable(initialize):
            initialize()
