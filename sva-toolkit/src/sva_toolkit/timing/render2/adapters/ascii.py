"""ASCII waveform adapter.

Upstream tool: optional ``asciiwave`` package for text waveforms.
Install hint: ``python -m pip install asciiwave pillow`` for the optional text
engine and PNG rasterization; without them this adapter uses its built-in text
renderer and may return text-only output.
Supported capabilities: ascii_only, bit, bus, clock, cuts, vector_text.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util

from sva_toolkit.timing.render2.adapters._common import (
    base_layout,
    bbox_by_role,
    lane_display_name,
    render_ascii_png,
    samples_from_runs,
    supports_scene,
    text_primitives_from_tokens,
    visibility_report,
    with_bbox_by_role,
)
from sva_toolkit.timing.render2.primitives import Point
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import LaneScene, LaneType, TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec
from sva_toolkit.timing.visual import VisibilityClass


class ASCIIAdapter:
    id = "ascii"
    capabilities = frozenset({"ascii_only", "bit", "bus", "clock", "cuts", "vector_text"})

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool:
        return supports_scene(self.id, self.capabilities, scene, spec)

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult:
        if not self.supports(scene, spec):
            raise RuntimeError("ascii adapter does not support this scene/spec")

        ascii_text = _render_ascii(scene)
        png_bytes = render_ascii_png(ascii_text, spec)
        layout = base_layout(scene, spec)
        text = _visibility_text(scene, spec, ascii_text)
        layout = with_bbox_by_role(layout, bbox_by_role(text))
        warnings = () if png_bytes is not None else ("Pillow is not installed; returning ASCII text only",)
        return RenderResult(
            svg_text=None,
            png_bytes=png_bytes,
            layout=layout,
            visibility=visibility_report(scene, text),
            render_spec=replace(spec, renderer_id=self.id),
            warnings=warnings,
            ascii_text=ascii_text,
        )


def dependency_status() -> str | None:
    return None


def has_optional_asciiwave() -> bool:
    return importlib.util.find_spec("asciiwave") is not None


def _render_ascii(scene: TimingScene) -> str:
    external = _render_with_asciiwave(scene)
    if external:
        return external
    label_width = max((len(lane_display_name(lane)) for lane in scene.lanes), default=0)
    rows = [_render_lane(lane, scene.ticks.total_ticks, label_width) for lane in scene.lanes]
    if scene.cuts:
        rows.append(_render_cuts(scene, label_width))
    return "\n".join(rows)


def _render_with_asciiwave(scene: TimingScene) -> str | None:
    try:
        import asciiwave
    except ImportError:
        return None

    source = [(lane_display_name(lane), samples_from_runs(lane, scene.ticks.total_ticks)) for lane in scene.lanes]
    for name in ("render", "draw", "wave"):
        candidate = getattr(asciiwave, name, None)
        if not callable(candidate):
            continue
        try:
            rendered = candidate(source)
        except Exception:
            continue
        if isinstance(rendered, str) and rendered.strip():
            return rendered
    return None


def _render_lane(lane: LaneScene, total_ticks: int, label_width: int) -> str:
    samples = samples_from_runs(lane, total_ticks)
    label = lane_display_name(lane).rjust(label_width)
    if lane.lane_type == LaneType.BUS:
        body = "".join(_bus_cell(value) for value in samples)
    else:
        body = _bit_wave(samples)
    return f"{label}  {body}"


def _bit_wave(samples: tuple[str, ...]) -> str:
    cells: list[str] = []
    previous = samples[0] if samples else "x"
    for index, value in enumerate(samples):
        level = _bit_level(value)
        if index == 0:
            cells.append("___" if level == "0" else "---" if level == "1" else "xxx")
        elif level == previous:
            cells.append("___" if level == "0" else "---" if level == "1" else "xxx")
        elif previous == "0" and level == "1":
            cells.append("/--")
        elif previous == "1" and level == "0":
            cells.append("\\__")
        else:
            cells.append("xxx")
        previous = level
    return "".join(cells)


def _bit_level(value: str) -> str:
    text = str(value).lower()
    if text in {"1", "h", "high", "true"}:
        return "1"
    if text in {"0", "l", "low", "false"}:
        return "0"
    return "x"


def _bus_cell(value: str) -> str:
    text = str(value)
    if text.lower() in {"x", "z"}:
        return "===x==="
    return f"=={text[:6]:^6}=="


def _render_cuts(scene: TimingScene, label_width: int) -> str:
    width = max(1, scene.ticks.total_ticks) * 3
    chars = [" "] * width
    for cut in scene.cuts:
        tick = max(0, min(scene.ticks.total_ticks - 1, cut.start_tick))
        index = min(width - 2, tick * 3)
        chars[index : index + 2] = ["/", "/"]
    return f"{'cuts'.rjust(label_width)}  {''.join(chars).rstrip()}"


def _visibility_text(scene: TimingScene, spec: RenderSpec, ascii_text: str):
    tokens = []
    x = 0.0
    for lane_index, lane in enumerate(scene.lanes):
        y = spec.layout.margin.y + lane_index * spec.layout.lane_pitch + spec.layout.lane_height * 0.7
        tokens.append((lane_display_name(lane), "lane_label", VisibilityClass.VISIBLE_TEXT.value, Point(x, y)))
        for run in lane.runs:
            if lane.lane_type == LaneType.BUS and str(run.value).lower() not in {"x", "z"}:
                bx = spec.layout.margin.x + (run.start_tick + 0.5) * spec.layout.tick_width
                tokens.append((str(run.value), "bus_value_text", VisibilityClass.VISIBLE_TEXT.value, Point(bx, y)))
    char_width = max(6.0, spec.style.primary_font.size_px * 0.55)
    line_height = max(12.0, spec.style.primary_font.size_px * 1.25)
    for row, line in enumerate(ascii_text.splitlines()):
        for column, char in enumerate(line):
            if char == " ":
                continue
            tokens.append(
                (
                    char,
                    "ascii_glyph",
                    VisibilityClass.VISIBLE_TEXT.value,
                    Point(column * char_width, spec.layout.margin.y + row * line_height),
                )
            )
    return text_primitives_from_tokens(tokens, spec)
