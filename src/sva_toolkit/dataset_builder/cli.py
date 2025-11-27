"""
CLI for SVA Dataset Builder.
"""

import click
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from sva_toolkit.dataset_builder.builder import DatasetBuilder, DatasetEntry
from sva_toolkit.utils.llm_client import LLMClient, LLMConfig

console = Console()


@click.group()
@click.pass_context
def main(ctx):
    """SVA Dataset Builder - Build training datasets with SVAD and CoT annotations."""
    ctx.ensure_object(dict)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--base-url', default="https://api.siliconflow.cn/v1", help='LLM API base URL')
@click.option('--model', default="Pro/deepseek-ai/DeepSeek-V3.2-Exp", help='LLM model name')
@click.option('--api-key', default="sk-anwluomxfwjhiwpoyjhjnmwnfqobzbdjaigihjwjcvncjehq", help='LLM API key')
@click.option('--no-svad', is_flag=True, help='Skip SVAD generation')
@click.option('--no-cot', is_flag=True, help='Skip CoT generation')
@click.option('--rate-limit', default=0.5, help='Delay between LLM calls (seconds)')
@click.option('--temperature', default=0.3, help='LLM temperature for SVAD generation')
def build(input_file, output_file, base_url, model, api_key, no_svad, no_cot, rate_limit, temperature):
    """Build dataset from input JSON file."""
    
    # Create LLM client
    llm_config = LLMConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
    llm_client = LLMClient(llm_config)
    
    # Create builder
    builder = DatasetBuilder(llm_client=llm_client)
    
    # Load input data
    console.print(f"[blue]Loading input file:[/blue] {input_file}")
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    total = len(input_data)
    console.print(f"[blue]Found {total} entries[/blue]")
    
    # Process with progress bar
    entries = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building dataset...", total=total)
        
        def progress_callback(current, total):
            progress.update(task, completed=current)
        
        entries = builder.build_dataset(
            input_data,
            generate_svad=not no_svad,
            generate_cot=not no_cot,
            progress_callback=progress_callback,
            rate_limit_delay=rate_limit,
        )
    
    # Save output
    output_data = [entry.to_dict() for entry in entries]
    with open(output_file, 'w') as f:
        json.dump(output_data, indent=2, fp=f)
    
    console.print(f"[green]Dataset saved to:[/green] {output_file}")
    
    # Show validation report
    report = builder.validate_dataset(entries)
    _display_validation_report(report)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def build_cot_only(input_file, output_file):
    """Build dataset with CoT only (no LLM required)."""
    
    builder = DatasetBuilder()
    
    console.print(f"[blue]Loading input file:[/blue] {input_file}")
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    total = len(input_data)
    console.print(f"[blue]Found {total} entries[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building CoT...", total=total)
        
        def progress_callback(current, total):
            progress.update(task, completed=current)
        
        entries = builder.build_dataset(
            input_data,
            generate_svad=False,
            generate_cot=True,
            progress_callback=progress_callback,
        )
    
    output_data = [entry.to_dict() for entry in entries]
    with open(output_file, 'w') as f:
        json.dump(output_data, indent=2, fp=f)
    
    console.print(f"[green]Dataset saved to:[/green] {output_file}")
    
    report = builder.validate_dataset(entries)
    _display_validation_report(report)


@main.command()
@click.argument('dataset_file', type=click.Path(exists=True))
def validate(dataset_file):
    """Validate an existing dataset file."""
    
    console.print(f"[blue]Loading dataset:[/blue] {dataset_file}")
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    entries = [DatasetEntry(**item) for item in data]
    
    builder = DatasetBuilder()
    report = builder.validate_dataset(entries)
    
    _display_validation_report(report)


@main.command()
@click.argument('sva_code')
@click.option('--base-url', required=True, help='LLM API base URL')
@click.option('--model', required=True, help='LLM model name')
@click.option('--api-key', required=True, help='LLM API key')
def generate_svad(sva_code, base_url, model, api_key):
    """Generate SVAD for a single SVA code snippet."""
    
    llm_config = LLMConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    llm_client = LLMClient(llm_config)
    builder = DatasetBuilder(llm_client=llm_client)
    
    console.print("[blue]Generating SVAD...[/blue]")
    svad = builder.generate_svad(sva_code)
    
    console.print("\n[bold]SVA Code:[/bold]")
    console.print(sva_code)
    console.print("\n[bold]Generated SVAD:[/bold]")
    console.print(svad)


@main.command()
@click.argument('sva_code')
def generate_cot(sva_code):
    """Generate CoT for a single SVA code snippet."""
    
    builder = DatasetBuilder()
    
    console.print("[blue]Generating CoT...[/blue]")
    cot = builder.generate_cot(sva_code)
    
    console.print("\n[bold]SVA Code:[/bold]")
    console.print(sva_code)
    console.print("\n[bold]Generated CoT:[/bold]")
    console.print(cot)


def _display_validation_report(report):
    """Display validation report as a table."""
    table = Table(title="Dataset Validation Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Entries", str(report["total_entries"]))
    table.add_row("Entries with SVAD", str(report["entries_with_svad"]))
    table.add_row("Entries with CoT", str(report["entries_with_cot"]))
    table.add_row("Entries with Errors", str(report["entries_with_errors"]))
    table.add_row("SVAD Coverage", f"{report['svad_coverage']*100:.1f}%")
    table.add_row("CoT Coverage", f"{report['cot_coverage']*100:.1f}%")
    
    console.print(table)


if __name__ == '__main__':
    main()
