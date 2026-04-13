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
@click.option('--rate-limit', default=0.5, help='Delay between LLM calls (seconds, single-process only)')
@click.option('--temperature', default=0.3, help='LLM temperature for SVAD generation')
@click.option('--workers', '-w', default=4, help='Number of worker processes for parallel processing')
@click.option('--cache-dir', type=click.Path(), help='Directory for caching progress (default: temp dir)')
@click.option('--no-cache', is_flag=True, help='Disable caching (start fresh)')
@click.option('--single-process', is_flag=True, help='Use single process mode (slower but simpler)')
@click.option('--system-prompt', type=str, help='Custom system prompt for SVAD generation')
@click.option('--system-prompt-file', type=click.Path(exists=True), help='Path to file containing custom system prompt (takes precedence over --system-prompt)')
def build(input_file, output_file, base_url, model, api_key, no_svad, no_cot, rate_limit, temperature, workers, cache_dir, no_cache, single_process, system_prompt, system_prompt_file):
    """Build dataset from input JSON file with multiprocessing support."""
    # Create LLM client
    llm_config = LLMConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
    llm_client = LLMClient(llm_config)
    # Handle cache directory
    effective_cache_dir = None if no_cache else cache_dir
    # Handle custom system prompt (file takes precedence)
    effective_system_prompt = None
    if system_prompt_file:
        with open(system_prompt_file, 'r') as f:
            effective_system_prompt = f.read().strip()
        console.print(f"[blue]Using custom system prompt from:[/blue] {system_prompt_file}")
    elif system_prompt:
        effective_system_prompt = system_prompt
        console.print("[blue]Using custom system prompt[/blue]")
    # Create builder with multiprocessing support
    builder = DatasetBuilder(
        llm_client=llm_client,
        num_workers=workers,
        cache_dir=effective_cache_dir,
        system_prompt=effective_system_prompt,
    )
    # Load input data
    console.print(f"[blue]Loading input file:[/blue] {input_file}")
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    total = len(input_data)
    console.print(f"[blue]Found {total} entries[/blue]")
    # Show cache info
    cache_stats = builder.get_cache_stats()
    if cache_stats["cached_items"] > 0:
        console.print(f"[green]Found {cache_stats['cached_items']} cached results[/green]")
    console.print(f"[dim]Cache directory: {builder.cache_dir}[/dim]")
    # Show execution mode
    use_multiprocessing = not single_process and workers > 1
    if use_multiprocessing:
        console.print(f"[blue]Running with {workers} worker processes[/blue]")
    else:
        console.print(f"[blue]Running in single-process mode[/blue]")
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
        def progress_callback(current: int, total: int) -> None:
            progress.update(task, completed=current)
        entries = builder.build_dataset(
            input_data,
            generate_svad=not no_svad,
            generate_cot=not no_cot,
            progress_callback=progress_callback,
            rate_limit_delay=rate_limit,
            use_multiprocessing=use_multiprocessing,
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
@click.option('--workers', '-w', default=4, help='Number of worker processes for parallel processing')
@click.option('--single-process', is_flag=True, help='Use single process mode')
def build_cot_only(input_file, output_file, workers, single_process):
    """Build dataset with CoT only (no LLM required) with multiprocessing support."""
    builder = DatasetBuilder(num_workers=workers)
    console.print(f"[blue]Loading input file:[/blue] {input_file}")
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    total = len(input_data)
    console.print(f"[blue]Found {total} entries[/blue]")
    # Show execution mode
    use_multiprocessing = not single_process and workers > 1
    if use_multiprocessing:
        console.print(f"[blue]Running with {workers} worker processes[/blue]")
    else:
        console.print(f"[blue]Running in single-process mode[/blue]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building CoT...", total=total)
        def progress_callback(current: int, total: int) -> None:
            progress.update(task, completed=current)
        entries = builder.build_dataset(
            input_data,
            generate_svad=False,
            generate_cot=True,
            progress_callback=progress_callback,
            use_multiprocessing=use_multiprocessing,
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
@click.option('--system-prompt', type=str, help='Custom system prompt for SVAD generation')
@click.option('--system-prompt-file', type=click.Path(exists=True), help='Path to file containing custom system prompt (takes precedence over --system-prompt)')
def generate_svad(sva_code, base_url, model, api_key, system_prompt, system_prompt_file):
    """Generate SVAD for a single SVA code snippet."""
    llm_config = LLMConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    llm_client = LLMClient(llm_config)
    # Handle custom system prompt (file takes precedence)
    effective_system_prompt = None
    if system_prompt_file:
        with open(system_prompt_file, 'r') as f:
            effective_system_prompt = f.read().strip()
        console.print(f"[blue]Using custom system prompt from:[/blue] {system_prompt_file}")
    elif system_prompt:
        effective_system_prompt = system_prompt
        console.print("[blue]Using custom system prompt[/blue]")
    builder = DatasetBuilder(llm_client=llm_client, system_prompt=effective_system_prompt)
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


@main.command('clear-cache')
@click.argument('cache_dir', type=click.Path(exists=True))
@click.option('--force', '-f', is_flag=True, help='Skip confirmation prompt')
def clear_cache(cache_dir: str, force: bool) -> None:
    """Clear dataset builder cache directory."""
    import os
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    if not cache_files:
        console.print(f"[yellow]No cache files found in {cache_dir}[/yellow]")
        return
    console.print(f"[blue]Found {len(cache_files)} cache files in {cache_dir}[/blue]")
    if not force:
        if not click.confirm("Are you sure you want to delete these cache files?"):
            console.print("[yellow]Aborted.[/yellow]")
            return
    deleted = 0
    for f in cache_files:
        try:
            os.remove(os.path.join(cache_dir, f))
            deleted += 1
        except OSError as e:
            console.print(f"[red]Failed to delete {f}: {e}[/red]")
    console.print(f"[green]Deleted {deleted} cache files.[/green]")


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
