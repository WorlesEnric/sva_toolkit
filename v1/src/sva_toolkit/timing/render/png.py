"""Optional PNG export for rendered SVG timing diagrams."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from sva_toolkit.timing.core.model import DiagramSpec
from sva_toolkit.timing.core.scenario import ScenarioDocument
from sva_toolkit.timing.render.svg import render_diagram_svg


def render_diagram_png(diagram: DiagramSpec | ScenarioDocument, output_path: Union[str, Path]) -> None:
    """Render a diagram to PNG if cairosvg is available."""

    try:
        import cairosvg  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PNG export requires cairosvg. Install it as an optional dependency or use SVG output."
        ) from exc

    svg_text = render_diagram_svg(diagram)
    cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(output_path))
