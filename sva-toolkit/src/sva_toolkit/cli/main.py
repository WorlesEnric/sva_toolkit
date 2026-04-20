from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import wraps
import json
import os
from pathlib import Path
import pprint
import re
import traceback
from typing import Any, Callable

import click

from sva_toolkit import __version__
from sva_toolkit.cli.exit_codes import exit_code_for
from sva_toolkit.cli.formal_flags import register as _register_formal_flags
from sva_toolkit.cli.generate_flags import register as _register_generate_flags
from sva_toolkit.cli.timing_flags import record_report as _record_timing_report
from sva_toolkit.cli.timing_flags import register as _register_timing_flags
from sva_toolkit.runtime.atomic_io import atomic_write_text
from sva_toolkit.runtime.diagnostics import DEFAULT_DIAGNOSTICS, LOGGER, configure_cli_logging


def _handle_cli_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(function, "__sva_cli_errors_wrapped__", False):
        return function

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = click.get_current_context()
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            _emit_cli_error(exc)
            ctx.exit(int(exit_code_for(exc)))
        finally:
            _emit_diagnostics_summary()

    setattr(wrapper, "__sva_cli_errors_wrapped__", True)
    return wrapper


def _load_text_argument(value: str) -> str:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return value


def _write_text_output(output: str | None, text: str) -> None:
    if output:
        atomic_write_text(output, text)
        return
    click.echo(text)


def _write_json_output(output: str | None, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2)
    if output:
        atomic_write_text(output, rendered + "\n")
        return
    click.echo(rendered)


def _write_jsonl_output(output: str | None, rows: list[dict[str, Any]]) -> None:
    rendered = "\n".join(json.dumps(row) for row in rows)
    if output:
        atomic_write_text(output, rendered + ("\n" if rendered else ""))
        return
    if rendered:
        click.echo(rendered)


def _emit_cli_error(exc: BaseException) -> None:
    if _cli_verbose_enabled():
        click.echo("".join(traceback.format_exception(exc)), err=True, nl=False)
        return

    stream = click.get_text_stream("stderr")
    if isinstance(exc, click.ClickException):
        exc.show(file=stream)
        return
    if isinstance(exc, FileNotFoundError) and exc.filename:
        click.ClickException(f"File not found: {exc.filename}").show(file=stream)
        return

    message = str(exc) or exc.__class__.__name__
    click.ClickException(message).show(file=stream)


def _cli_verbose_enabled() -> bool:
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    root = ctx.find_root()
    return bool(root.obj and root.obj.get("verbose"))


def _emit_diagnostics_summary() -> None:
    summary = DEFAULT_DIAGNOSTICS.render_summary()
    if summary:
        LOGGER.warning(summary)


def _normalize_kind_name(name: str) -> str:
    kind = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    for suffix in ("_property", "_sequence", "_expr", "_event", "_spec"):
        if kind.endswith(suffix):
            kind = kind[: -len(suffix)]
            break
    return kind


def _to_json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        payload = {field.name: _to_json_compatible(getattr(value, field.name)) for field in fields(value)}
        payload["kind"] = _normalize_kind_name(type(value).__name__)
        return payload
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _format_structured_output(value: Any, output_format: str) -> str:
    payload = _to_json_compatible(value)
    if output_format == "json":
        return json.dumps(payload, indent=2)
    return pprint.pformat(payload, sort_dicts=False)


def _build_formal_service(backend: str, timeout: int, depth: int):
    from sva_toolkit.formal import FormalService

    return FormalService(backend=backend, timeout=timeout, depth=depth)


def _echo_check_result(result: Any) -> None:
    click.echo(f"Result: {result.result.value}")
    click.echo(f"Message: {result.message}")
    if result.counterexample:
        click.echo(f"Counterexample: {result.counterexample}")
    if result.log:
        click.echo(f"Log:\n{result.log}")


def _patch_witness_defaults(depth: int, timeout: int) -> None:
    import sva_toolkit.timing.bridge.ebmc_witness as witness_module

    witness_module._DEFAULT_DEPTH = depth
    witness_module._DEFAULT_TIMEOUT = timeout


def _extract_sva_documents(input_file: str, *, depth: int, timeout: int):
    from sva_toolkit.formal.parse import split_property_texts
    from sva_toolkit.timing.bridge.from_sva import bundle_sva_scenarios, extract_sva_scenarios
    from sva_toolkit.timing.bridge.status import merge_extraction_reports

    _patch_witness_defaults(depth, timeout)
    property_texts = split_property_texts(Path(input_file).read_text(encoding="utf-8"))
    documents, report = extract_sva_scenarios(property_texts)
    if len(documents) <= 1:
        return documents, report
    bundled_documents, bundle_report = bundle_sva_scenarios(
        documents,
        overlap_threshold=0.0,
        max_signals=max(10, sum(len(document.signals) for document in documents)),
        max_properties=max(5, sum(len(document.properties) for document in documents)),
        max_chains=max(3, len(documents)),
    )
    return bundled_documents, merge_extraction_reports(report, bundle_report)


def _build_dataset_builder(model: str | None, workers: int):
    from sva_toolkit.data import DatasetBuilder

    if model is None:
        return DatasetBuilder(num_workers=workers)

    from sva_toolkit.runtime.llm import LLMClient, LLMConfig

    return DatasetBuilder(
        llm_client=LLMClient(LLMConfig(model=model)),
        num_workers=workers,
    )


def _build_benchmark_runner(model: str, workers: int):
    from sva_toolkit.data import BenchmarkRunner
    from sva_toolkit.runtime.llm import LLMClient, LLMConfig

    return BenchmarkRunner(
        llm_clients=[LLMClient(LLMConfig(model=model))],
        num_workers=workers,
    )


@click.group(help="SVA Toolkit unified CLI for parsing, formal analysis, timing, generation, description, and data workflows.")
@click.version_option(__version__, prog_name="sva")
@click.option("--verbose", is_flag=True, help="Print full exception tracebacks, including chained causes.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Top-level `sva` command group."""
    DEFAULT_DIAGNOSTICS.reset()
    configure_cli_logging(1 if verbose else 0)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.group(help="Formal verification commands.")
def formal() -> None:
    """Formal verification commands."""


@main.group(help="Timing diagram commands.")
def timing() -> None:
    """Timing diagram commands."""


@main.group(help="Description commands.")
def describe() -> None:
    """Description commands."""


@main.group(help="Dataset and benchmark commands.")
def data() -> None:
    """Dataset and benchmark commands."""


@main.command("parse")
@click.argument("sva_code_or_file")
@click.option("--format", "output_format", type=click.Choice(["json", "text"]), default="text", show_default=True)
def parse_command(sva_code_or_file: str, output_format: str) -> None:
    """Parse a SystemVerilog assertion and display its structure."""
    from sva_toolkit.sva import parse_property_text

    parsed = parse_property_text(_load_text_argument(sva_code_or_file))
    click.echo(_format_structured_output(parsed, output_format))


@formal.command("check")
@click.argument("antecedent")
@click.argument("consequent")
@click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=300, show_default=True)
@click.option("--depth", type=int, default=20, show_default=True)
def formal_check(antecedent: str, consequent: str, backend: str, timeout: int, depth: int) -> None:
    """Check whether ANTECEDENT implies CONSEQUENT."""
    from sva_toolkit.formal.model import ImplicationResult

    result = _build_formal_service(backend, timeout, depth).check_implication(antecedent, consequent)
    _echo_check_result(result)
    if result.result is not ImplicationResult.IMPLIES:
        raise SystemExit(1)


@formal.command("equivalent")
@click.argument("sva1")
@click.argument("sva2")
@click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=300, show_default=True)
@click.option("--depth", type=int, default=20, show_default=True)
def formal_equivalent(sva1: str, sva2: str, backend: str, timeout: int, depth: int) -> None:
    """Check whether SVA1 and SVA2 are equivalent."""
    from sva_toolkit.formal.model import ImplicationResult

    result = _build_formal_service(backend, timeout, depth).check_equivalence(sva1, sva2)
    _echo_check_result(result)
    if result.result is not ImplicationResult.EQUIVALENT:
        raise SystemExit(1)


@formal.command("relationship")
@click.argument("sva1")
@click.argument("sva2")
@click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto", show_default=True)
@click.option("--timeout", type=int, default=300, show_default=True)
@click.option("--depth", type=int, default=20, show_default=True)
def formal_relationship(sva1: str, sva2: str, backend: str, timeout: int, depth: int) -> None:
    """Determine the implication relationship between SVA1 and SVA2."""
    forward, reverse = _build_formal_service(backend, timeout, depth).get_relationship(sva1, sva2)
    click.echo(f"SVA1 implies SVA2: {'yes' if forward else 'no'}")
    click.echo(f"SVA2 implies SVA1: {'yes' if reverse else 'no'}")


@timing.command("render")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False))
@click.option("--format", "output_format", type=click.Choice(["svg", "png"]), default="svg", show_default=True)
def timing_render(input_file: str, output: str | None, output_format: str) -> None:
    """Render a timing diagram file to SVG or PNG."""
    from sva_toolkit.timing.frontend.parser import parse_diagram
    from sva_toolkit.timing.render import render_diagram_png, render_diagram_svg

    if output_format == "png" and output is None:
        raise click.ClickException("--output is required when --format=png")

    diagram = parse_diagram(Path(input_file).read_text(encoding="utf-8"))
    if output_format == "png":
        render_diagram_png(diagram, Path(output))
        return
    _write_text_output(output, render_diagram_svg(diagram))


@timing.command("validate")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def timing_validate(input_file: str) -> None:
    """Validate a timing diagram file."""
    from sva_toolkit.timing.frontend.parser import parse_diagram

    parse_diagram(Path(input_file).read_text(encoding="utf-8"))
    click.echo("valid")


@timing.command("emit-sva")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False))
@click.option(
    "--allow-lossy", is_flag=True, help="Allow unsupported constructs to be emitted with lossy approximations."
)
def timing_emit_sva(input_file: str, output: str | None, allow_lossy: bool) -> None:
    """Emit parameterized SVA from a timing diagram file."""
    from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
    from sva_toolkit.timing.frontend.parser import parse_diagram

    diagram = parse_diagram(Path(input_file).read_text(encoding="utf-8"))
    _write_text_output(output, emit_parameterized_sva(diagram, allow_lossy=allow_lossy))


@timing.command("extract-sva")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False))
@click.option("--depth", type=int, default=32, show_default=True)
@click.option("--timeout", type=int, default=60, show_default=True)
def timing_extract_sva(input_file: str, output: str | None, depth: int, timeout: int) -> None:
    """Extract timing DSL scenarios from an SVA source file."""
    from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl

    documents, report = _extract_sva_documents(input_file, depth=depth, timeout=timeout)
    _record_timing_report(report)
    if report.worst_status().value != "exact":
        return
    rendered = "\n\n".join(emit_timing_dsl(document) for document in documents)
    _write_text_output(output, rendered)


@timing.command("bundle-sva")
@click.argument("input_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False))
def timing_bundle_sva(input_files: tuple[str, ...], output: str | None) -> None:
    """Bundle related SVA files into grouped timing scenarios."""
    from sva_toolkit.timing.bridge.from_sva import bundle_sva_scenarios
    from sva_toolkit.timing.bridge.status import merge_extraction_reports
    from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl

    if not input_files:
        raise click.ClickException("at least one input file is required")

    extracted = tuple(_extract_sva_documents(input_file, depth=32, timeout=60) for input_file in input_files)
    scenarios = tuple(document for documents, _ in extracted for document in documents)
    bundled_scenarios, bundle_report = bundle_sva_scenarios(scenarios)
    report = merge_extraction_reports(*(report for _, report in extracted), bundle_report)
    _record_timing_report(report)
    if report.worst_status().value != "exact":
        return
    rendered = "\n\n".join(emit_timing_dsl(document) for document in bundled_scenarios)
    _write_text_output(output, rendered)


@main.command("generate")
@click.option("--count", type=int, default=1, show_default=True)
@click.option("--mode", type=click.Choice(["random", "stratified"]), default="random", show_default=True)
@click.option("--validate", is_flag=True, help="Validate generated assertions with Verible when available.")
@click.option("--coverage", is_flag=True, help="Append coverage statistics for the generated assertions.")
def generate_command(count: int, mode: str, validate: bool, coverage: bool, seed: int | None = None) -> None:
    """Generate SVA properties using the V3 generator module."""
    from sva_toolkit.generate import (
        DEFAULT_SIGNALS,
        GenerationRng,
        SVASynthesizer,
        StratifiedGenerator,
        compute_coverage_statistics,
        generate_sv_module,
    )

    if count < 1:
        raise click.ClickException("--count must be at least 1")

    signals = list(DEFAULT_SIGNALS[:5]) if len(DEFAULT_SIGNALS) >= 5 else list(DEFAULT_SIGNALS)
    if not signals:
        signals = ["req", "ack", "gnt"]

    rng = GenerationRng(seed=seed)
    if seed is None:
        click.echo(f"Using generation seed: {rng.seed_value}", err=True)

    if mode == "stratified":
        generator = StratifiedGenerator(signals=signals, samples_per_construct=max(1, count), rng=rng)
        properties = generator.generate_stratified_dataset()[:count]
        module_code = generate_sv_module("generated_sva", signals, [prop.property_block for prop in properties])
        validator = generator.synth
    else:
        synthesizer = SVASynthesizer(signals=signals, max_depth=2, rng=rng)
        module_code, properties = synthesizer.generate_module("generated_sva", count)
        validator = synthesizer

    if validate:
        validation = validator.validate_properties(properties)
        if not validation.is_valid:
            raise click.ClickException(validation.error_message or "Generated assertions failed validation")

    click.echo(module_code.rstrip())
    if coverage:
        stats = compute_coverage_statistics([prop.sva_code for prop in properties])
        click.echo("")
        click.echo(
            "Coverage: "
            f"{stats['constructs_covered']}/{stats['constructs_total']} constructs "
            f"({stats['coverage_pct']:.1f}%)"
        )

@describe.command("svad")
@click.argument("sva_code_or_file")
@click.option(
    "--format", "output_format", type=click.Choice(["text", "json", "markdown"]), default="text", show_default=True
)
def describe_svad(sva_code_or_file: str, output_format: str) -> None:
    """Render an SVAD description for an assertion."""
    from sva_toolkit.describe import SVADTranslator

    rendered = SVADTranslator().translate(_load_text_argument(sva_code_or_file))
    if output_format == "json":
        click.echo(json.dumps({"svad": rendered}, indent=2))
        return
    click.echo(rendered)


@describe.command("cot")
@click.argument("sva_code_or_file")
@click.option(
    "--format", "output_format", type=click.Choice(["text", "json", "markdown"]), default="text", show_default=True
)
def describe_cot(sva_code_or_file: str, output_format: str) -> None:
    """Render chain-of-thought style reasoning for an assertion."""
    from sva_toolkit.describe import SVACoTBuilder

    rendered = SVACoTBuilder().build(_load_text_argument(sva_code_or_file))
    if output_format == "json":
        click.echo(json.dumps({"cot": rendered}, indent=2))
        return
    click.echo(rendered)


@data.command("build")
@click.argument("input_json", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False))
@click.option("--model", default=None, help="Optional LLM model name for SVAD generation.")
@click.option("--workers", type=int, default=4, show_default=True)
def data_build(input_json: str, output: str | None, model: str | None, workers: int) -> None:
    """Build a dataset JSONL file from SVA source records."""
    from sva_toolkit.data import DatasetEntry

    builder = _build_dataset_builder(model, workers)
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    entries = builder.build_dataset(
        payload,
        generate_svad=True,
        generate_cot=True,
        use_multiprocessing=workers > 1,
        rate_limit_delay=0,
    )
    rows = [entry.to_dict() if isinstance(entry, DatasetEntry) else entry for entry in entries]
    _write_jsonl_output(output, rows)


@data.command("benchmark")
@click.argument("dataset_json", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", default=None, help="LLM model name used to generate benchmark predictions.")
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("-o", "--output", type=click.Path(dir_okay=False))
def data_benchmark(dataset_json: str, model: str | None, workers: int, output: str | None) -> None:
    """Benchmark model-generated SVA against a reference dataset."""
    resolved_model = model or os.getenv("SVA_TOOLKIT_MODEL")
    if not resolved_model:
        raise click.ClickException("--model is required unless SVA_TOOLKIT_MODEL is set")

    runner = _build_benchmark_runner(resolved_model, workers)
    dataset = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    result = runner.run_benchmark(
        dataset,
        runner.llm_clients[0],
        use_multiprocessing=workers > 1,
        rate_limit_delay=0,
    )
    payload = result.to_dict() if hasattr(result, "to_dict") else _to_json_compatible(result)
    _write_json_output(output, payload)


_register_formal_flags(formal)
_register_timing_flags(timing)
_register_generate_flags(main)


def _install_cli_error_handlers(command: click.Command) -> None:
    if isinstance(command, click.Group):
        for child in command.commands.values():
            _install_cli_error_handlers(child)
        return
    if command.callback is not None:
        command.callback = _handle_cli_errors(command.callback)


_install_cli_error_handlers(main)


if __name__ == "__main__":
    main()
