"""CLI for timing diagram rendering and SVA emission."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
from sva_toolkit.timing.bridge.from_sva import bundle_sva_scenarios, extract_sva_scenario
from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.errors import TimingDslError
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render.png import render_diagram_png
from sva_toolkit.timing.render.svg import render_diagram_svg


@click.group()
def main() -> None:
    """Timing diagram tools for rendering and SVA emission."""


@main.command("render")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), help="Output file path")
@click.option("--format", "output_format", type=click.Choice(["svg", "png"]), default="svg")
def render_command(input_file: Path, output: Optional[Path], output_format: str) -> None:
    """Render a timing diagram DSL file to SVG or PNG."""

    diagram = _load_diagram(input_file)
    if output_format == "svg":
        svg = render_diagram_svg(diagram)
        if output:
            output.write_text(svg, encoding="utf-8")
        else:
            click.echo(svg)
        return

    if output is None:
        raise click.UsageError("--output is required for PNG rendering")
    render_diagram_png(diagram, output)
    click.echo(str(output))


@main.command("emit-sva")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), help="Output file path")
@click.option("--allow-lossy", is_flag=True, help="Allow heuristic lowering for lossy extracted scenarios")
def emit_sva_command(input_file: Path, output: Optional[Path], allow_lossy: bool) -> None:
    """Emit parameterized SVA from a timing diagram DSL file."""

    diagram = _load_diagram(input_file)
    sva_text = emit_parameterized_sva(diagram, allow_lossy=allow_lossy)
    if output:
        output.write_text(sva_text, encoding="utf-8")
    else:
        click.echo(sva_text)


@main.command("validate")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_command(input_file: Path) -> None:
    """Validate a timing diagram DSL file."""

    _load_diagram(input_file)
    click.echo("valid")


@main.command("extract-sva")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), help="Output DSL file path")
def extract_sva_command(input_file: Path, output: Optional[Path]) -> None:
    """Extract a timing scenario from an SVA file."""

    document = extract_sva_scenario(input_file.read_text(encoding="utf-8"), name=input_file.stem)
    dsl_text = emit_timing_dsl(document)
    if output:
        output.write_text(dsl_text, encoding="utf-8")
    else:
        click.echo(dsl_text)


@main.command("bundle-sva")
@click.argument("input_files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), help="Output DSL file path")
def bundle_sva_command(input_files: tuple[Path, ...], output: Optional[Path]) -> None:
    """Bundle related SVA properties into one or more grouped timing scenarios."""

    if not input_files:
        raise click.UsageError("at least one SVA input file is required")
    documents = [extract_sva_scenario(path.read_text(encoding="utf-8"), name=path.stem) for path in input_files]
    bundled = bundle_sva_scenarios(documents)
    payload = "\n\n".join(emit_timing_dsl(document) for document in bundled)
    if output:
        output.write_text(payload, encoding="utf-8")
    else:
        click.echo(payload)


def _load_diagram(path: Path):
    try:
        return parse_diagram(path.read_text(encoding="utf-8"))
    except TimingDslError as exc:
        raise click.ClickException(str(exc)) from exc
