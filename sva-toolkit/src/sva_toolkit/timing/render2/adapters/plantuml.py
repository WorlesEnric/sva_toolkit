"""PlantUML timing diagram adapter.

Upstream tool: PlantUML timing diagrams rendered by the ``plantuml`` command
or a jar path in ``PLANTUML_JAR``.
Install hint: install PlantUML on PATH, or set ``PLANTUML_JAR=/path/to/plantuml.jar``.
Supported capabilities: bit, bus, clock, ascii_only, wavejson_subset.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from sva_toolkit.timing.render2.adapters._common import (
    base_layout,
    bbox_by_role,
    lane_display_name,
    samples_from_runs,
    supports_scene,
    text_primitives_from_tokens,
    visibility_report,
    with_bbox_by_role,
)
from sva_toolkit.timing.render2.primitives import Point
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import LaneType, TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.visual import VisibilityClass


class PlantUMLAdapter:
    id = "plantuml"
    capabilities = frozenset({"bit", "bus", "clock", "ascii_only", "wavejson_subset"})

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool:
        return supports_scene(self.id, self.capabilities, scene, spec, dependency_available=_plantuml_command() is not None)

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult:
        command = _plantuml_command()
        if command is None:
            raise RuntimeError("plantuml is not installed; install plantuml on PATH or set PLANTUML_JAR")
        if not self.supports(scene, spec):
            raise RuntimeError("plantuml adapter does not support this scene/spec")

        source = _build_plantuml_source(scene)
        png_bytes = _render_plantuml_png(source, command)
        layout = base_layout(scene, spec)
        text = _visibility_text(scene, spec)
        layout = with_bbox_by_role(layout, bbox_by_role(text))
        return RenderResult(
            svg_text=None,
            png_bytes=png_bytes,
            layout=layout,
            visibility=visibility_report(scene, text),
            render_spec=replace(spec, renderer_id=self.id),
            warnings=(),
        )


def dependency_status() -> str | None:
    return None if _plantuml_command() is not None else "missing_executable:plantuml"


def _plantuml_command() -> list[str] | None:
    binary = shutil.which("plantuml")
    if binary:
        return [binary]
    jar = os.environ.get("PLANTUML_JAR")
    if jar and Path(jar).exists() and shutil.which("java"):
        return ["java", "-jar", jar]
    return None


def _build_plantuml_source(scene: TimingScene) -> str:
    aliases = {_lane_alias(lane.name): lane for lane in scene.lanes}
    lines = ["@startuml"]
    for alias, lane in aliases.items():
        label = _escape_label(lane_display_name(lane))
        if lane.lane_type == LaneType.CLOCK:
            lines.append(f'clock "{label}" as {alias} with period 2')
        elif lane.lane_type == LaneType.BUS:
            lines.append(f'concise "{label}" as {alias}')
        else:
            lines.append(f'binary "{label}" as {alias}')

    for tick in range(max(1, scene.ticks.total_ticks)):
        lines.append(f"@{tick}")
        for alias, lane in aliases.items():
            value = samples_from_runs(lane, scene.ticks.total_ticks)[tick]
            if lane.lane_type == LaneType.CLOCK:
                continue
            if lane.lane_type == LaneType.BUS:
                lines.append(f'{alias} is "{_escape_label(value)}"')
            else:
                lines.append(f"{alias} is {_plantuml_bit(value)}")
    lines.append("@enduml")
    return "\n".join(lines)


def _render_plantuml_png(source: str, command: list[str]) -> bytes:
    with tempfile.TemporaryDirectory(prefix="sva-plantuml-") as tmpdir:
        puml = Path(tmpdir) / "diagram.puml"
        png = Path(tmpdir) / "diagram.png"
        puml.write_text(source, encoding="utf-8")
        subprocess.run([*command, "-tpng", str(puml)], cwd=tmpdir, check=True, capture_output=True, timeout=45)
        return png.read_bytes()


def _visibility_text(scene: TimingScene, spec: RenderSpec):
    tokens = []
    for lane_index, lane in enumerate(scene.lanes):
        y = spec.layout.margin.y + lane_index * spec.layout.lane_pitch + spec.layout.lane_height * 0.7
        tokens.append((lane_display_name(lane), "lane_label", VisibilityClass.VISIBLE_TEXT.value, Point(0.0, y)))
        for run in lane.runs:
            if lane.lane_type == LaneType.BUS and str(run.value).lower() not in {"x", "z"}:
                x = spec.layout.margin.x + (run.start_tick + 0.5) * spec.layout.tick_width
                tokens.append((str(run.value), "bus_value_text", VisibilityClass.VISIBLE_TEXT.value, Point(x, y)))
    return text_primitives_from_tokens(tokens, spec)


def _plantuml_bit(value: str) -> str:
    text = str(value).lower()
    if text in {"1", "h", "high", "true"}:
        return "high"
    if text in {"0", "l", "low", "false"}:
        return "low"
    return "unknown"


def _lane_alias(name: str) -> str:
    return "s_" + "".join(char if char.isalnum() else "_" for char in name)


def _escape_label(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')
