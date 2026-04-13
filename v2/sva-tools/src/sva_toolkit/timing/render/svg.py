"""SVG rendering entrypoints for timing scenarios."""

from __future__ import annotations

from sva_toolkit.timing.core.scenario import ScenarioDocument
from sva_toolkit.timing.projection.wavedrom_view import build_wavedrom_view
from sva_toolkit.timing.render.wavedrom import render_wavedrom_svg


def render_diagram_svg(diagram: ScenarioDocument) -> str:
    """Render a timing scenario to SVG text."""

    view = build_wavedrom_view(diagram)
    return render_wavedrom_svg(view)
