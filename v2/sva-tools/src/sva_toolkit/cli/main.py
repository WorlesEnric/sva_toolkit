from pathlib import Path

import click


@click.group()
def main():
    """SVA Toolkit - formal verification, timing diagrams, and assertion workflows."""


@main.group()
def formal():
    """Formal verification commands."""


@main.group()
def timing():
    """Timing diagram commands."""


def _read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write_text_output(output: str | None, text: str) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
        return
    click.echo(text)


def _extract_sva_documents(input_file: str, *, depth: int = 32, timeout: int = 60):
    from sva_toolkit.formal.parse import split_property_texts
    from sva_toolkit.timing.bridge.ebmc_witness import EbmcWitnessSynthesizer
    from sva_toolkit.timing.bridge.from_sva import bundle_sva_scenarios, extract_sva_scenario

    # Thread depth/timeout into the synthesizer used inside extract/bundle
    _patch_witness_defaults(depth, timeout)

    property_texts = split_property_texts(_read_text_file(input_file))
    documents = tuple(extract_sva_scenario(property_text) for property_text in property_texts)
    if len(documents) <= 1:
        return documents
    return bundle_sva_scenarios(
        documents,
        overlap_threshold=0.0,
        max_signals=max(10, sum(len(document.signals) for document in documents)),
        max_properties=max(5, sum(len(document.properties) for document in documents)),
        max_chains=max(3, len(documents)),
    )


def _patch_witness_defaults(depth: int, timeout: int) -> None:
    """Override EbmcWitnessSynthesizer defaults for this invocation."""
    import sva_toolkit.timing.bridge.ebmc_witness as _ew
    _ew._DEFAULT_DEPTH = depth
    _ew._DEFAULT_TIMEOUT = timeout


def _build_formal_service(backend: str, timeout: int, depth: int, verbose: bool):
    from sva_toolkit.formal.service import FormalService

    return FormalService(backend=backend, timeout=timeout, depth=depth, verbose=verbose)


def _echo_check_result(result) -> None:
    click.echo(f"Result: {result.result.value}")
    click.echo(f"Message: {result.message}")
    if result.counterexample:
        click.echo(f"Counterexample: {result.counterexample}")


@formal.command("check")
@click.argument("antecedent")
@click.argument("consequent")
@click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto")
@click.option("--timeout", type=int, default=300)
@click.option("--depth", type=int, default=20)
@click.option("--verbose", is_flag=True)
def formal_check(antecedent, consequent, backend, timeout, depth, verbose):
    """Check if ANTECEDENT implies CONSEQUENT."""
    from sva_toolkit.formal.model import ImplicationResult

    service = _build_formal_service(backend=backend, timeout=timeout, depth=depth, verbose=verbose)
    result = service.check_implication(antecedent, consequent)
    _echo_check_result(result)
    if result.result != ImplicationResult.IMPLIES:
        raise SystemExit(1)


@formal.command("equivalent")
@click.argument("sva1")
@click.argument("sva2")
@click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto")
@click.option("--timeout", type=int, default=300)
@click.option("--depth", type=int, default=20)
@click.option("--verbose", is_flag=True)
def formal_equivalent(sva1, sva2, backend, timeout, depth, verbose):
    """Check if SVA1 and SVA2 are equivalent."""
    from sva_toolkit.formal.model import ImplicationResult

    service = _build_formal_service(backend=backend, timeout=timeout, depth=depth, verbose=verbose)
    result = service.check_equivalence(sva1, sva2)
    _echo_check_result(result)
    if result.result != ImplicationResult.EQUIVALENT:
        raise SystemExit(1)


@formal.command("relationship")
@click.argument("sva1")
@click.argument("sva2")
@click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto")
@click.option("--timeout", type=int, default=300)
@click.option("--depth", type=int, default=20)
@click.option("--verbose", is_flag=True)
def formal_relationship(sva1, sva2, backend, timeout, depth, verbose):
    """Determine implication relationship between SVA1 and SVA2."""
    service = _build_formal_service(backend=backend, timeout=timeout, depth=depth, verbose=verbose)
    forward, reverse = service.get_relationship(sva1, sva2)
    click.echo(f"SVA1 implies SVA2: {'yes' if forward else 'no'}")
    click.echo(f"SVA2 implies SVA1: {'yes' if reverse else 'no'}")


@timing.command("render")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False))
@click.option("--format", "output_format", type=click.Choice(["svg", "png"]), default="svg")
def timing_render(input_file, output, output_format):
    """Render a timing diagram DSL file to SVG or PNG."""
    if output_format == "png" and not output:
        raise click.ClickException("--output is required when --format=png")

    try:
        from sva_toolkit.timing.errors import TimingDslError
        from sva_toolkit.timing.frontend.parser import parse_diagram
        from sva_toolkit.timing.render.png import render_diagram_png
        from sva_toolkit.timing.render.svg import render_diagram_svg
        from sva_toolkit.timing.render.wavedrom import render_wavedrom_svg
        from sva_toolkit.timing.projection.wavedrom_view import build_wavedrom_view

        diagram = parse_diagram(_read_text_file(input_file))
        
        # Priority: Try WaveDrom-style first for publication-grade output
        view = build_wavedrom_view(diagram)
        svg_content = render_wavedrom_svg(view)
        if output_format == "png":
            # For PNG, we still use the basic renderer or a conversion
            render_diagram_png(diagram, Path(output))
            return
        _write_text_output(output, svg_content)
        return
    except TimingDslError as exc:
        raise click.ClickException(str(exc)) from exc


@timing.command("emit-sva")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False))
@click.option("--allow-lossy", is_flag=True)
def timing_emit_sva(input_file, output, allow_lossy):
    """Emit parameterized SVA from a timing diagram DSL file."""
    try:
        from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
        from sva_toolkit.timing.errors import TimingDslError
        from sva_toolkit.timing.frontend.parser import parse_diagram

        diagram = parse_diagram(_read_text_file(input_file))
        _write_text_output(output, emit_parameterized_sva(diagram, allow_lossy=allow_lossy))
    except TimingDslError as exc:
        raise click.ClickException(str(exc)) from exc


@timing.command("validate")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def timing_validate(input_file):
    """Validate a timing diagram DSL file."""
    try:
        from sva_toolkit.timing.errors import TimingDslError
        from sva_toolkit.timing.frontend.parser import parse_diagram

        parse_diagram(_read_text_file(input_file))
        click.echo("valid")
    except TimingDslError as exc:
        raise click.ClickException(str(exc)) from exc


@timing.command("extract-sva")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False))
@click.option("--depth", type=int, default=32, show_default=True, help="EBMC bound depth for witness synthesis.")
@click.option("--timeout", type=int, default=60, show_default=True, help="EBMC timeout in seconds for witness synthesis.")
def timing_extract_sva(input_file, output, depth, timeout):
    """Extract a timing scenario from an SVA file."""
    try:
        from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
        from sva_toolkit.timing.errors import TimingDslError

        documents = _extract_sva_documents(input_file, depth=depth, timeout=timeout)
        _write_text_output(output, "\n\n".join(emit_timing_dsl(document) for document in documents))
    except TimingDslError as exc:
        raise click.ClickException(str(exc)) from exc


@timing.command("bundle-sva")
@click.argument("input_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False))
def timing_bundle_sva(input_files, output):
    """Bundle related SVA properties into grouped timing scenarios."""
    if not input_files:
        raise click.ClickException("at least one input file is required")

    try:
        from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
        from sva_toolkit.timing.errors import TimingDslError

        scenarios = tuple(document for input_file in input_files for document in _extract_sva_documents(input_file))
        from sva_toolkit.timing.bridge.from_sva import bundle_sva_scenarios

        bundled = bundle_sva_scenarios(scenarios)
        _write_text_output(output, "\n\n".join(emit_timing_dsl(document) for document in bundled))
    except TimingDslError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
