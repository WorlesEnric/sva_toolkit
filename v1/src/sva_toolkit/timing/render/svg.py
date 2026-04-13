"""SVG rendering entrypoint for timing diagrams."""

from __future__ import annotations

from sva_toolkit.timing.core.model import DiagramSpec
from sva_toolkit.timing.core.scenario import ScenarioDocument
from sva_toolkit.timing.render.symbolic import render_symbolic_svg
from sva_toolkit.timing.render.wavedrom import render_diagram_wavedrom_svg


def render_diagram_svg(diagram: DiagramSpec | ScenarioDocument) -> str:
    """Render a timing diagram to SVG text."""

    if isinstance(diagram, ScenarioDocument):
        if diagram.legacy_diagram is not None and isinstance(diagram.legacy_diagram, DiagramSpec):
            return render_diagram_wavedrom_svg(diagram.legacy_diagram)
        return render_symbolic_svg(diagram)
    return render_diagram_wavedrom_svg(diagram)
