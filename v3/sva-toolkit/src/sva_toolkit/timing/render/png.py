"""PNG rendering helpers for timing scenarios."""

from __future__ import annotations


def render_diagram_png(diagram, output_path) -> None:
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError(
            "cairosvg is required to rasterize timing SVG output. "
            "Install sva-toolkit[timing-render] for PNG/WaveDrom support"
        ) from exc

    from sva_toolkit.timing.render.svg import render_diagram_svg

    svg_text = render_diagram_svg(diagram)
    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(output_path))
