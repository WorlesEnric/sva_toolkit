"""
CLI for SVA Benchmark Runner.
"""

import click
import json
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from sva_toolkit.benchmark.runner import BenchmarkRunner, BenchmarkResult, RelationshipType
from sva_toolkit.utils.llm_client import LLMClient, LLMConfig

console = Console()


@click.group()
@click.pass_context
def main(ctx):
    """SVA Benchmark - Evaluate LLM performance on SVA generation."""
    ctx.ensure_object(dict)


@main.command()
@click.argument('dataset_file', type=click.Path(exists=True))
@click.option('--base-url', required=True, help='LLM API base URL')
@click.option('--model', required=True, help='LLM model name')
@click.option('--api-key', required=True, help='LLM API key')
@click.option('--output', '-o', type=click.Path(), help='Output file for detailed results')
@click.option('--rate-limit', default=0.5, help='Delay between LLM calls (seconds)')
@click.option('--sby-path', default='sby', help='Path to sby binary')
@click.option('--depth', default=20, help='Proof depth for verification')
def run(dataset_file, base_url, model, api_key, output, rate_limit, sby_path, depth):
    """Run benchmark on a single LLM."""
    
    # Load dataset
    console.print(f"[blue]Loading dataset:[/blue] {dataset_file}")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    # Filter dataset to only include items with both SVAD and SVA
    valid_items = [item for item in dataset if item.get("SVAD") and item.get("SVA")]
    console.print(f"[blue]Found {len(valid_items)} valid items with SVAD and SVA[/blue]")
    
    if not valid_items:
        console.print("[red]No valid items found in dataset. Each item needs both 'SVAD' and 'SVA' fields.[/red]")
        return
    
    # Create benchmark runner
    llm_config = LLMConfig(base_url=base_url, model=model, api_key=api_key)
    llm_client = LLMClient(llm_config)
    
    from sva_toolkit.implication_checker import SVAImplicationChecker
    checker = SVAImplicationChecker(sby_path=sby_path, depth=depth)
    
    runner = BenchmarkRunner(llm_clients=[llm_client], implication_checker=checker)
    
    # Run benchmark with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Benchmarking {model}...", total=len(valid_items))
        
        def progress_callback(current, total):
            progress.update(task, completed=current)
        
        result = runner.run_benchmark(
            valid_items,
            llm_client,
            progress_callback=progress_callback,
            rate_limit_delay=rate_limit,
        )
    
    # Display results
    _display_benchmark_result(result)
    
    # Save detailed results if output specified
    if output:
        detailed_output = {
            "summary": result.to_dict(),
            "individual_results": [
                {
                    "svad": r.svad,
                    "reference_sva": r.reference_sva,
                    "generated_sva": r.generated_sva,
                    "relationship": r.relationship.value,
                    "error_message": r.error_message,
                    "generation_time": r.generation_time,
                    "verification_time": r.verification_time,
                }
                for r in result.individual_results
            ]
        }
        with open(output, 'w') as f:
            json.dump(detailed_output, indent=2, fp=f)
        console.print(f"[green]Detailed results saved to:[/green] {output}")


@main.command('run-multi')
@click.argument('dataset_file', type=click.Path(exists=True))
@click.option('--config-file', required=True, type=click.Path(exists=True),
              help='JSON file with LLM configurations')
@click.option('--output', '-o', type=click.Path(), help='Output file for detailed results')
@click.option('--rate-limit', default=0.5, help='Delay between LLM calls (seconds)')
def run_multi(dataset_file, config_file, output, rate_limit):
    """Run benchmark on multiple LLMs."""
    
    # Load dataset
    console.print(f"[blue]Loading dataset:[/blue] {dataset_file}")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    valid_items = [item for item in dataset if item.get("SVAD") and item.get("SVA")]
    console.print(f"[blue]Found {len(valid_items)} valid items[/blue]")
    
    # Load LLM configs
    console.print(f"[blue]Loading LLM configs:[/blue] {config_file}")
    with open(config_file, 'r') as f:
        llm_configs = json.load(f)
    
    console.print(f"[blue]Found {len(llm_configs)} LLM configurations[/blue]")
    
    # Create runner
    runner = BenchmarkRunner.from_configs(llm_configs)
    
    all_results = []
    
    for i, (llm_client, config) in enumerate(zip(runner.llm_clients, llm_configs)):
        model_name = config["model"]
        console.print(f"\n[bold blue]═══ Benchmarking Model {i+1}/{len(llm_configs)}: {model_name} ═══[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {model_name}...", total=len(valid_items))
            
            def progress_callback(current, total):
                progress.update(task, completed=current)
            
            result = runner.run_benchmark(
                valid_items,
                llm_client,
                progress_callback=progress_callback,
                rate_limit_delay=rate_limit,
            )
            all_results.append(result)
        
        _display_benchmark_result(result)
    
    # Display comparison
    console.print("\n[bold blue]═══ Comparison ═══[/bold blue]")
    _display_comparison(all_results)
    
    # Save results
    if output:
        output_data = {
            "comparison": BenchmarkRunner.compare_results(all_results),
            "results": [r.to_dict() for r in all_results],
        }
        with open(output, 'w') as f:
            json.dump(output_data, indent=2, fp=f)
        console.print(f"[green]Results saved to:[/green] {output}")


@main.command()
@click.argument('dataset_file', type=click.Path(exists=True))
def stats(dataset_file):
    """Show statistics about a benchmark dataset."""
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    
    total = len(dataset)
    has_sva = sum(1 for item in dataset if item.get("SVA"))
    has_svad = sum(1 for item in dataset if item.get("SVAD"))
    has_both = sum(1 for item in dataset if item.get("SVA") and item.get("SVAD"))
    has_cot = sum(1 for item in dataset if item.get("CoT"))
    
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="yellow")
    
    table.add_row("Total Entries", str(total), "100%")
    table.add_row("Has SVA", str(has_sva), f"{has_sva/total*100:.1f}%" if total > 0 else "N/A")
    table.add_row("Has SVAD", str(has_svad), f"{has_svad/total*100:.1f}%" if total > 0 else "N/A")
    table.add_row("Has Both (Benchmarkable)", str(has_both), f"{has_both/total*100:.1f}%" if total > 0 else "N/A")
    table.add_row("Has CoT", str(has_cot), f"{has_cot/total*100:.1f}%" if total > 0 else "N/A")
    
    console.print(table)


def _display_benchmark_result(result: BenchmarkResult):
    """Display a single benchmark result."""
    
    table = Table(title=f"Benchmark Results: {result.model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Items", str(result.total_items))
    table.add_row("Equivalent", f"{result.equivalent_count} ({result.equivalent_rate*100:.1f}%)")
    table.add_row("Generated → Reference", f"{result.generated_implies_reference_count}")
    table.add_row("Reference → Generated", f"{result.reference_implies_generated_count}")
    table.add_row("No Relationship", f"{result.no_relationship_count}")
    table.add_row("Errors", f"{result.error_count}")
    table.add_row("", "")
    table.add_row("Any Implication Rate", f"{result.any_implication_rate*100:.1f}%")
    table.add_row("Success Rate", f"{result.success_rate*100:.1f}%")
    table.add_row("Avg Generation Time", f"{result.avg_generation_time:.2f}s")
    table.add_row("Avg Verification Time", f"{result.avg_verification_time:.2f}s")
    
    console.print(table)
    
    # Visual breakdown
    console.print("\n[bold]Relationship Breakdown:[/bold]")
    total = result.total_items
    if total > 0:
        eq_bar = "█" * int(result.equivalent_rate * 40)
        gen_bar = "█" * int(result.generated_implies_reference_count / total * 40)
        ref_bar = "█" * int(result.reference_implies_generated_count / total * 40)
        no_bar = "█" * int(result.no_relationship_count / total * 40)
        err_bar = "█" * int(result.error_count / total * 40)
        
        console.print(f"[green]Equivalent:      {eq_bar}[/green] {result.equivalent_rate*100:.1f}%")
        console.print(f"[yellow]Gen→Ref:         {gen_bar}[/yellow] {result.generated_implies_reference_count/total*100:.1f}%")
        console.print(f"[blue]Ref→Gen:         {ref_bar}[/blue] {result.reference_implies_generated_count/total*100:.1f}%")
        console.print(f"[red]No Relationship: {no_bar}[/red] {result.no_relationship_count/total*100:.1f}%")
        console.print(f"[dim]Errors:          {err_bar}[/dim] {result.error_count/total*100:.1f}%")


def _display_comparison(results: List[BenchmarkResult]):
    """Display comparison of multiple benchmark results."""
    
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Equivalent", style="green")
    table.add_column("Any Impl.", style="yellow")
    table.add_column("Success", style="blue")
    table.add_column("Avg Gen Time", style="dim")
    
    best_equiv = max(r.equivalent_rate for r in results)
    best_impl = max(r.any_implication_rate for r in results)
    
    for result in results:
        equiv_str = f"{result.equivalent_rate*100:.1f}%"
        if result.equivalent_rate == best_equiv:
            equiv_str = f"[bold green]{equiv_str} ★[/bold green]"
        
        impl_str = f"{result.any_implication_rate*100:.1f}%"
        if result.any_implication_rate == best_impl:
            impl_str = f"[bold yellow]{impl_str} ★[/bold yellow]"
        
        table.add_row(
            result.model_name,
            equiv_str,
            impl_str,
            f"{result.success_rate*100:.1f}%",
            f"{result.avg_generation_time:.2f}s",
        )
    
    console.print(table)
    
    # Overall winner
    comparison = BenchmarkRunner.compare_results(results)
    console.print(f"\n[bold]Best Equivalent Rate:[/bold] {comparison['best_equivalent_model']}")
    console.print(f"[bold]Best Any Implication Rate:[/bold] {comparison['best_implication_model']}")


if __name__ == '__main__':
    main()
