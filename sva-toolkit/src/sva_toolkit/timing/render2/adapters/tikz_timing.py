"""tikz-timing adapter.

Upstream tool: LaTeX ``tikz-timing`` package rendered by ``pdflatex`` or
``lualatex``.
Install hint: install TeX Live or MacTeX with tikz-timing, plus ``pdf2image``
or Poppler's ``pdftoppm`` for PNG conversion.
Supported capabilities: bit, bus, clock, latex_only, raster, vector_text.

This adapter chooses the reliable raster path by default: write a standalone
LaTeX document, run ``pdflatex``/``lualatex``, then convert the PDF to PNG via
``pdf2image`` when available or ``pdftoppm`` otherwise. If
``spec.extras["tikz_output"] == "svg"``, it tries ``dvisvgm`` and returns SVG.
"""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import importlib.util
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


class TikzTimingAdapter:
    id = "tikz_timing"
    capabilities = frozenset({"bit", "bus", "clock", "latex_only", "raster", "vector_text"})

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool:
        return supports_scene(self.id, self.capabilities, scene, spec, dependency_available=_toolchain_available(spec))

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult:
        latex = _latex_executable()
        if latex is None:
            raise RuntimeError("pdflatex is not installed; install TeX Live/MacTeX or provide lualatex on PATH")
        if _wants_svg(spec) and shutil.which("dvisvgm") is None:
            raise RuntimeError("dvisvgm is not installed; install dvisvgm or request PNG output")
        if not _wants_svg(spec) and not _png_converter_available():
            raise RuntimeError("pdftoppm is not installed; install Poppler or python -m pip install pdf2image")
        if not self.supports(scene, spec):
            raise RuntimeError("tikz-timing adapter does not support this scene/spec")

        source = _build_latex_source(scene)
        svg_text, png_bytes = _render_latex(source, latex, spec)
        layout = base_layout(scene, spec)
        text = _visibility_text(scene, spec)
        layout = with_bbox_by_role(layout, bbox_by_role(text))
        return RenderResult(
            svg_text=svg_text,
            png_bytes=png_bytes,
            layout=layout,
            visibility=visibility_report(scene, text),
            render_spec=replace(spec, renderer_id=self.id),
            warnings=(),
        )


def dependency_status() -> str | None:
    if _latex_executable() is None:
        return "missing_executable:pdflatex"
    if not _png_converter_available() and shutil.which("dvisvgm") is None:
        return "missing_executable:pdftoppm"
    return None


def _toolchain_available(spec: RenderSpec) -> bool:
    if _latex_executable() is None:
        return False
    if _wants_svg(spec):
        return shutil.which("dvisvgm") is not None
    return _png_converter_available()


def _latex_executable() -> str | None:
    return shutil.which("pdflatex") or shutil.which("lualatex")


def _png_converter_available() -> bool:
    return importlib.util.find_spec("pdf2image") is not None or shutil.which("pdftoppm") is not None


def _wants_svg(spec: RenderSpec) -> bool:
    return spec.extras.get("tikz_output") == "svg"


def _build_latex_source(scene: TimingScene) -> str:
    rows = "\n".join(f"{_latex_escape(lane_display_name(lane))} & {_tikz_wave(lane, scene.ticks.total_ticks)} \\\\" for lane in scene.lanes)
    return "\n".join(
        (
            r"\documentclass[tikz,border=2pt]{standalone}",
            r"\usepackage{tikz-timing}",
            r"\begin{document}",
            r"\begin{tikztimingtable}",
            rows,
            r"\end{tikztimingtable}",
            r"\end{document}",
            "",
        )
    )


def _tikz_wave(lane, total_ticks: int) -> str:
    samples = samples_from_runs(lane, total_ticks)
    if lane.lane_type == LaneType.BUS:
        return " ".join(f"D{{{_latex_escape(value)}}}" if value.lower() not in {"x", "z"} else "X" for value in samples)
    return " ".join(_tikz_bit(value) for value in samples)


def _tikz_bit(value: str) -> str:
    text = value.lower()
    if text in {"1", "h", "high", "true"}:
        return "H"
    if text in {"0", "l", "low", "false"}:
        return "L"
    return "X"


def _render_latex(source: str, latex: str, spec: RenderSpec) -> tuple[str | None, bytes | None]:
    with tempfile.TemporaryDirectory(prefix="sva-tikz-") as tmpdir:
        tex = Path(tmpdir) / "diagram.tex"
        tex.write_text(source, encoding="utf-8")
        subprocess.run(
            [latex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tmpdir,
            check=True,
            capture_output=True,
            timeout=45,
        )
        pdf = Path(tmpdir) / "diagram.pdf"
        if _wants_svg(spec):
            svg = Path(tmpdir) / "diagram.svg"
            subprocess.run(
                ["dvisvgm", "--pdf", "--no-fonts", "-o", str(svg), str(pdf)],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                timeout=45,
            )
            return svg.read_text(encoding="utf-8"), None
        return None, _pdf_to_png(pdf, spec)


def _pdf_to_png(pdf: Path, spec: RenderSpec) -> bytes:
    if importlib.util.find_spec("pdf2image") is not None:
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf), dpi=spec.raster.dpi, first_page=1, last_page=1)
        output = BytesIO()
        images[0].save(output, format="PNG")
        return output.getvalue()

    output_prefix = pdf.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", str(spec.raster.dpi), str(pdf), str(output_prefix)],
        check=True,
        capture_output=True,
        timeout=45,
    )
    return output_prefix.with_suffix(".png").read_bytes()


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


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))
