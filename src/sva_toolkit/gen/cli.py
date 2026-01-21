"""
CLI for SVA Generator.

Provides command-line interface for generating SystemVerilog Assertions
with syntax validation using Verible.
"""

import click
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from typing import List, Optional

from sva_toolkit.gen.generator import SVASynthesizer, GenerationResult, ValidationResult, SVAProperty
from sva_toolkit.gen.stratified import StratifiedGenerator
from sva_toolkit.gen.utils import (
    DEFAULT_SIGNALS,
    HANDSHAKE_SIGNALS,
    FIFO_SIGNALS,
    AXI_SIGNALS,
)
from sva_toolkit.gen.coverage import compute_coverage_statistics

console = Console()

# Default verible path
DEFAULT_VERIBLE_PATH = (
    "/Users/wangqihang/wkspace/sva_toolkit/3rd_party/verible_bin/verible-verilog-syntax"
)

# Signal presets
SIGNAL_PRESETS = {
    "default": DEFAULT_SIGNALS,
    "handshake": HANDSHAKE_SIGNALS,
    "fifo": FIFO_SIGNALS,
    "axi": AXI_SIGNALS,
}


@click.group()
@click.option(
    "--verible-path",
    default=DEFAULT_VERIBLE_PATH,
    help="Path to verible-verilog-syntax binary",
    type=click.Path(),
)
@click.pass_context
def main(ctx: click.Context, verible_path: str) -> None:
    """SVA Generator - Generate syntactically legal SystemVerilog Assertions."""
    ctx.ensure_object(dict)
    ctx.obj["verible_path"] = verible_path


@main.command()
@click.option(
    "-n", "--num-assertions",
    default=10,
    type=int,
    help="Number of assertions to generate"
)
@click.option(
    "-d", "--max-depth",
    default=2,
    type=int,
    help="Maximum recursion depth for expression generation"
)
@click.option(
    "-m", "--module-name",
    default="sva_test_bench",
    help="Name of the generated module"
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="Output file path (stdout if not specified)"
)
@click.option(
    "-s", "--signals",
    multiple=True,
    help="Signal names to use (can be repeated)"
)
@click.option(
    "--preset",
    type=click.Choice(list(SIGNAL_PRESETS.keys())),
    default="default",
    help="Use predefined signal set"
)
@click.option(
    "--clock",
    default="clk",
    help="Clock signal name"
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate generated SVA syntax with Verible"
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output as JSON"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)
@click.option(
    "--mode",
    type=click.Choice(['random', 'stratified']),
    default='random',
    help="Generation mode: 'random' for probabilistic, 'stratified' for guaranteed coverage"
)
@click.option(
    "--samples-per-construct",
    type=int,
    default=50,
    help="For stratified mode: minimum samples per construct"
)
@click.pass_context
def generate(
    ctx: click.Context,
    num_assertions: int,
    max_depth: int,
    module_name: str,
    output: Optional[str],
    signals: tuple,
    preset: str,
    clock: str,
    validate: bool,
    json_output: bool,
    seed: Optional[int],
    mode: str,
    samples_per_construct: int
) -> None:
    """Generate SVA properties and wrap in a SystemVerilog module."""
    import random
    if seed is not None:
        random.seed(seed)
        if not json_output:
            console.print(f"[dim]Using random seed: {seed}[/dim]")

    # Determine signals to use
    if signals:
        signal_list = list(signals)
    else:
        signal_list = SIGNAL_PRESETS[preset]

    verible_path = ctx.obj["verible_path"]

    # Handle stratified mode
    if mode == 'stratified':
        if not json_output:
            console.print(f"[bold blue]SVA Generator - Stratified Mode[/bold blue]")
            console.print(f"  Mode: Guaranteed coverage of all constructs")
            console.print(f"  Samples per construct: {samples_per_construct}")
            console.print(f"  Signals: {', '.join(signal_list)}")
            console.print()

        generator = StratifiedGenerator(
            signals=signal_list,
            clock_signal=clock,
            max_depth=max_depth,
            samples_per_construct=samples_per_construct,
            verible_path=verible_path
        )

        properties = generator.generate_stratified_dataset()

        # Create result
        result = GenerationResult(
            properties=properties,
            module_code="",  # No module in stratified mode
            validation=None,
            valid_count=len(properties),
            invalid_count=0
        )

        _display_generation_result(result, json_output, output)
        return

    # Random mode (original behavior)
    if not json_output:
        console.print(f"[bold blue]SVA Generator - Random Mode[/bold blue]")
        console.print(f"  Target: {num_assertions} properties, Max Depth: {max_depth}")
        console.print(f"  Signals: {', '.join(signal_list)}")
        console.print(f"  Verible: {verible_path}")

    synthesizer = SVASynthesizer(
        signals=signal_list,
        max_depth=max_depth,
        clock_signal=clock,
        verible_path=verible_path
    )

    if validate:
        result = synthesizer.generate_validated(
            module_name=module_name,
            num_assertions=num_assertions
        )
        _display_generation_result(result, json_output, output)
    else:
        module_code, properties = synthesizer.generate_module(
            module_name=module_name,
            num_assertions=num_assertions
        )
        result = GenerationResult(
            properties=properties,
            module_code=module_code,
            validation=None,
            valid_count=len(properties),
            invalid_count=0
        )
        _display_generation_result(result, json_output, output)


@main.command()
@click.argument("sva_code")
@click.pass_context
def validate(ctx: click.Context, sva_code: str) -> None:
    """Validate SVA code syntax using Verible."""
    verible_path = ctx.obj["verible_path"]
    synthesizer = SVASynthesizer(
        signals=["dummy"],
        verible_path=verible_path
    )
    # Check if input is a file path
    if Path(sva_code).exists():
        with open(sva_code, "r") as f:
            code = f.read()
    else:
        code = sva_code
    result = synthesizer.validate_syntax(code)
    if result.is_valid:
        console.print("[bold green]✓ Syntax is valid[/bold green]")
    else:
        console.print("[bold red]✗ Syntax error[/bold red]")
        console.print(f"[red]{result.error_message}[/red]")
        sys.exit(1)


@main.command("validate-file")
@click.argument("filepath", type=click.Path(exists=True))
@click.pass_context
def validate_file(ctx: click.Context, filepath: str) -> None:
    """Validate SVA syntax in a SystemVerilog file."""
    verible_path = ctx.obj["verible_path"]
    synthesizer = SVASynthesizer(
        signals=["dummy"],
        verible_path=verible_path
    )
    with open(filepath, "r") as f:
        code = f.read()
    result = synthesizer.validate_syntax(code)
    if result.is_valid:
        console.print(f"[bold green]✓ {filepath}: Syntax is valid[/bold green]")
    else:
        console.print(f"[bold red]✗ {filepath}: Syntax error[/bold red]")
        console.print(f"[red]{result.error_message}[/red]")
        sys.exit(1)


@main.command()
@click.option(
    "-n", "--num-batches",
    default=10,
    type=int,
    help="Number of batches to generate"
)
@click.option(
    "-s", "--batch-size",
    default=10,
    type=int,
    help="Number of assertions per batch"
)
@click.option(
    "-d", "--max-depth",
    default=4,
    type=int,
    help="Maximum recursion depth"
)
@click.option(
    "--preset",
    type=click.Choice(list(SIGNAL_PRESETS.keys())),
    default="default",
    help="Use predefined signal set"
)
@click.pass_context
def stress_test(
    ctx: click.Context,
    num_batches: int,
    batch_size: int,
    max_depth: int,
    preset: str
) -> None:
    """Run stress test to find type system leaks."""
    verible_path = ctx.obj["verible_path"]
    signal_list = SIGNAL_PRESETS[preset]
    console.print("[bold blue]SVA Generator Stress Test[/bold blue]")
    console.print(f"  Batches: {num_batches}, Batch Size: {batch_size}")
    console.print(f"  Total Assertions: {num_batches * batch_size}")
    synthesizer = SVASynthesizer(
        signals=signal_list,
        max_depth=max_depth,
        verible_path=verible_path
    )
    total_valid = 0
    total_invalid = 0
    errors: List[str] = []
    with console.status("[bold green]Generating and validating...") as status:
        for batch_idx in range(num_batches):
            result = synthesizer.generate_validated(
                module_name=f"stress_test_{batch_idx}",
                num_assertions=batch_size
            )
            if result.validation and result.validation.is_valid:
                total_valid += batch_size
            else:
                total_invalid += batch_size
                if result.validation:
                    errors.append(result.validation.error_message)
            status.update(
                f"[bold green]Batch {batch_idx + 1}/{num_batches}: "
                f"Valid={total_valid}, Invalid={total_invalid}"
            )
    # Display results
    table = Table(title="Stress Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Assertions", str(num_batches * batch_size))
    table.add_row("Valid Batches", str(total_valid // batch_size))
    table.add_row("Invalid Batches", str(total_invalid // batch_size))
    table.add_row(
        "Success Rate",
        f"{100 * total_valid / (total_valid + total_invalid):.1f}%"
    )
    console.print(table)
    if errors:
        console.print("\n[bold red]Errors found (type system leaks):[/bold red]")
        for i, err in enumerate(errors[:5]):  # Show first 5 errors
            console.print(f"  {i + 1}. {err[:200]}...")


@main.command()
def list_presets() -> None:
    """List available signal presets."""
    table = Table(title="Signal Presets")
    table.add_column("Preset", style="cyan")
    table.add_column("Signals", style="green")
    for name, signals in SIGNAL_PRESETS.items():
        table.add_row(name, ", ".join(signals))
    console.print(table)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--top-n",
    default=None,
    type=int,
    help="Show only top N most frequent constructs"
)
def coverage(input_file: str, top_n: Optional[int]) -> None:
    """Analyze SVA construct coverage from a JSON dataset file.

    The input file should contain a JSON object with a 'properties' key
    containing a list of SVA properties with 'sva' field.
    """
    from sva_toolkit.gen.coverage import compute_coverage_statistics, CATEGORIES, SVA_CONSTRUCTS

    # Load JSON file
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON file: {e}[/red]")
        sys.exit(1)

    # Extract SVA strings
    if isinstance(data, dict) and 'properties' in data:
        properties = data['properties']
    elif isinstance(data, list):
        properties = data
    else:
        console.print("[red]Error: Expected JSON with 'properties' key or a list[/red]")
        sys.exit(1)

    # Extract SVA codes
    sva_codes = []
    for prop in properties:
        if isinstance(prop, dict) and 'sva' in prop:
            sva_codes.append(prop['sva'])
        elif isinstance(prop, str):
            sva_codes.append(prop)

    if not sva_codes:
        console.print("[yellow]Warning: No SVA properties found in input file[/yellow]")
        sys.exit(0)

    # Compute coverage
    stats = compute_coverage_statistics(sva_codes)

    # Display header
    console.print(Panel(
        f"[bold cyan]Coverage Analysis[/bold cyan]\n"
        f"Dataset: {input_file}\n"
        f"Properties: {stats['total_properties']}",
        border_style="cyan"
    ))

    # Summary statistics
    summary_table = Table(title="Coverage Summary", show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan", width=30)
    summary_table.add_column("Value", style="bold green")

    summary_table.add_row("Total Constructs", str(stats['constructs_total']))
    summary_table.add_row("Covered Constructs", str(stats['constructs_covered']))
    summary_table.add_row("Coverage Percentage", f"{stats['coverage_pct']:.1f}%")

    console.print(summary_table)
    console.print()

    # Category-wise coverage
    for category, constructs in stats['categories'].items():
        table = Table(title=category, show_lines=False)
        table.add_column("Construct", style="yellow", width=25)
        table.add_column("In Properties", style="cyan", justify="right", width=15)
        table.add_column("Coverage %", style="magenta", justify="right", width=12)
        table.add_column("Occurrences", style="green", justify="right", width=12)

        # Sort by coverage percentage (descending)
        constructs_sorted = sorted(constructs, key=lambda x: x['coverage_pct'], reverse=True)

        # Apply top_n filter if specified
        if top_n:
            constructs_sorted = constructs_sorted[:top_n]

        for construct_info in constructs_sorted:
            status = "✓" if construct_info['properties_with_construct'] > 0 else "✗"
            desc = construct_info['description']
            props_count = construct_info['properties_with_construct']
            coverage = construct_info['coverage_pct']
            occurrences = construct_info['occurrences']

            # Color coding based on coverage
            if coverage > 50:
                style = "green"
            elif coverage > 20:
                style = "yellow"
            elif coverage > 0:
                style = "dim yellow"
            else:
                style = "dim red"

            table.add_row(
                f"{status} {desc}",
                f"[{style}]{props_count}[/{style}]",
                f"[{style}]{coverage:.1f}%[/{style}]",
                f"[{style}]{occurrences}[/{style}]"
            )

        console.print(table)
        console.print()

    # Missing constructs (if any)
    if stats['missing_constructs']:
        console.print(f"[bold red]Missing Constructs ({len(stats['missing_constructs'])}):[/bold red]")
        missing_table = Table(show_header=False, box=None)
        missing_table.add_column("", style="dim red")

        for missing in stats['missing_constructs'][:10]:  # Show first 10
            missing_table.add_row(f"✗ {missing['description']} ({missing['key']})")

        console.print(missing_table)

        if len(stats['missing_constructs']) > 10:
            console.print(f"[dim]... and {len(stats['missing_constructs']) - 10} more[/dim]")
    else:
        console.print("[bold green]✓ All constructs covered![/bold green]")

    console.print()


def _display_generation_result(
    result: GenerationResult,
    json_output: bool,
    output_path: Optional[str]
) -> None:
    """Display generation result in appropriate format."""
    if json_output:
        # Convert SVAProperty objects to dictionaries for JSON serialization
        properties_list = [
            {
                "name": prop.name,
                "sva": prop.sva_code,
                "svad": prop.svad,
                "property_block": prop.property_block
            }
            for prop in result.properties
        ]

        coverage_metadata = compute_coverage_statistics(
            [prop["sva"] for prop in properties_list]
        )

        output_data = {
            "properties": properties_list,
            "metadata": {
                "coverage": coverage_metadata,
                "valid_count": result.valid_count,
                "invalid_count": result.invalid_count,
            }
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[dim]JSON output written to: {output_path}[/dim]")
            console.print(f"[green]✓[/green] Generated {len(result.properties)} SVA-SVAD pairs")
        else:
            print(json.dumps(output_data, indent=2))
    else:
        _write_output(result.module_code, output_path)
        if result.validation:
            if result.validation.is_valid:
                console.print(
                    f"[green]✓[/green] Generated {len(result.properties)} properties "
                    f"(syntax validated)"
                )
            else:
                console.print(
                    f"[yellow]![/yellow] Generated {len(result.properties)} properties "
                    f"(validation failed)"
                )
                console.print(f"[red]Error: {result.validation.error_message}[/red]")


def _write_output(content: str, output_path: Optional[str]) -> None:
    """Write content to file or stdout."""
    if output_path:
        with open(output_path, "w") as f:
            f.write(content)
        console.print(f"[dim]Output written to: {output_path}[/dim]")
    else:
        console.print(Syntax(content, "systemverilog", theme="monokai"))


if __name__ == "__main__":
    main()
