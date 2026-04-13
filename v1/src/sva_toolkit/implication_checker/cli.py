"""
CLI for SVA Implication Checker.
"""

import click
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn

from sva_toolkit.implication_checker.checker import (
    SVAImplicationChecker,
    ImplicationResult,
)

console = Console()


@click.group()
@click.option('--ebmc-path', default=None, help='Path to ebmc binary')
@click.option('--depth', default=20, help='Proof depth for bounded model checking')
@click.option('--work-dir', default=None, help='Working directory for verification files')
@click.option('--keep-files', is_flag=True, help='Keep generated files after verification')
@click.pass_context
def main(ctx, ebmc_path, depth, work_dir, keep_files):
    """SVA Implication Checker - Verify implication relationships between SVA pairs."""
    ctx.ensure_object(dict)
    ctx.obj['checker'] = SVAImplicationChecker(
        ebmc_path=ebmc_path,
        depth=depth,
        work_dir=work_dir,
        keep_files=keep_files,
    )


@main.command()
@click.option('--antecedent', '-a', required=True, help='Antecedent SVA code')
@click.option('--consequent', '-c', required=True, help='Consequent SVA code')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed log')
@click.pass_context
def check(ctx, antecedent, consequent, json_output, verbose):
    """Check if antecedent implies consequent."""
    checker = ctx.obj['checker']
    
    try:
        result = checker.check_implication(antecedent, consequent)
        
        if json_output:
            output = {
                "result": result.result.value,
                "implies": result.result == ImplicationResult.IMPLIES,
                "message": result.message,
            }
            if result.counterexample:
                output["counterexample"] = result.counterexample
            console.print(json.dumps(output, indent=2), markup=False)
        else:
            _display_result(result, "Implication Check", verbose)
            
            # Return exit code based on result
            if result.result == ImplicationResult.IMPLIES:
                console.print("[green]✓ Result: TRUE (antecedent implies consequent)[/green]")
            else:
                console.print("[red]✗ Result: FALSE (antecedent does not imply consequent)[/red]")
                
    except Exception as e:
        console.print(f"[red]Error during verification:[/red] {escape(str(e))}")
        raise click.Abort()


@main.command()
@click.option('--sva1', required=True, help='First SVA code')
@click.option('--sva2', required=True, help='Second SVA code')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed log')
@click.pass_context
def equivalent(ctx, sva1, sva2, json_output, verbose):
    """Check if two SVAs are equivalent (bidirectional implication)."""
    checker = ctx.obj['checker']
    
    try:
        result = checker.check_equivalence(sva1, sva2)
        
        if json_output:
            output = {
                "result": result.result.value,
                "equivalent": result.result == ImplicationResult.EQUIVALENT,
                "message": result.message,
            }
            if result.counterexample:
                output["counterexample"] = result.counterexample
            console.print(json.dumps(output, indent=2), markup=False)
        else:
            _display_result(result, "Equivalence Check", verbose)
            
            if result.result == ImplicationResult.EQUIVALENT:
                console.print("[green]✓ Result: TRUE (SVAs are equivalent)[/green]")
            else:
                console.print("[red]✗ Result: FALSE (SVAs are not equivalent)[/red]")
                
    except Exception as e:
        console.print(f"[red]Error during verification:[/red] {escape(str(e))}")
        raise click.Abort()


@main.command()
@click.option('--sva1', required=True, help='First SVA code')
@click.option('--sva2', required=True, help='Second SVA code')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.pass_context
def relationship(ctx, sva1, sva2, json_output):
    """Determine the full implication relationship between two SVAs."""
    checker = ctx.obj['checker']
    
    try:
        sva1_implies_sva2, sva2_implies_sva1 = checker.get_implication_relationship(sva1, sva2)
        
        if json_output:
            output = {
                "sva1_implies_sva2": sva1_implies_sva2,
                "sva2_implies_sva1": sva2_implies_sva1,
                "equivalent": sva1_implies_sva2 and sva2_implies_sva1,
                "no_relationship": not sva1_implies_sva2 and not sva2_implies_sva1,
            }
            console.print(json.dumps(output, indent=2), markup=False)
        else:
            console.print("\n[bold]Implication Relationship Analysis[/bold]\n")
            
            console.print(f"SVA1 → SVA2: {'[green]✓ Yes[/green]' if sva1_implies_sva2 else '[red]✗ No[/red]'}")
            console.print(f"SVA2 → SVA1: {'[green]✓ Yes[/green]' if sva2_implies_sva1 else '[red]✗ No[/red]'}")
            
            console.print("\n[bold]Conclusion:[/bold]")
            if sva1_implies_sva2 and sva2_implies_sva1:
                console.print("[green]SVAs are EQUIVALENT (bidirectional implication)[/green]")
            elif sva1_implies_sva2:
                console.print("[yellow]SVA1 is STRONGER than SVA2 (SVA1 implies SVA2)[/yellow]")
            elif sva2_implies_sva1:
                console.print("[yellow]SVA2 is STRONGER than SVA1 (SVA2 implies SVA1)[/yellow]")
            else:
                console.print("[red]NO IMPLICATION RELATIONSHIP exists[/red]")
                
    except Exception as e:
        console.print(f"[red]Error during verification:[/red] {escape(str(e))}")
        raise click.Abort()


@main.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True), help='Input JSON file with list of SVA pairs')
@click.option('--output-file', '-o', default=None, type=click.Path(), help='Output JSON file (default: stdout)')
@click.option('--verbose', '-v', is_flag=True, help='Show progress and detailed logs')
@click.pass_context
def batch_equivalent(ctx, input_file, output_file, verbose):
    """Batch process equivalence checks from a JSON file containing a list of SVA pairs.
    
    Expected JSON format:
    [
        {"id": "id1", "sva1": "...", "sva2": "..."},
        {"id": "id2", "sva1": "...", "sva2": "..."},
        ...
    ]
    """
    checker = ctx.obj['checker']
    
    try:
        # Read input JSON file
        input_path = Path(input_file)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            console.print("[red]Error: Input JSON must be a list of dictionaries[/red]")
            raise click.Abort()
        
        if verbose:
            console.print(f"[bold]Processing {len(data)} SVA pairs from {input_file}[/bold]\n")
        
        results = []
        total = len(data)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=not verbose
        ) as progress:
            task = progress.add_task("Processing...", total=total)
            
            for idx, entry in enumerate(data, 1):
                if not isinstance(entry, dict):
                    console.print(f"[yellow]Warning: Entry {idx} is not a dictionary, skipping[/yellow]")
                    continue
                
                entry_id = entry.get('id', f'entry_{idx}')
                sva1 = entry.get('sva1')
                sva2 = entry.get('sva2')
                
                if not sva1 or not sva2:
                    console.print(f"[yellow]Warning: Entry {entry_id} missing sva1 or sva2, skipping[/yellow]")
                    continue
                
                if verbose:
                    progress.update(task, description=f"Processing {entry_id} ({idx}/{total})")
                
                try:
                    result = checker.check_equivalence(sva1, sva2)
                    
                    result_entry = {
                        "id": entry_id,
                        "result": result.result.value,
                        "equivalent": result.result == ImplicationResult.EQUIVALENT,
                        "message": result.message,
                    }
                    
                    if result.counterexample:
                        result_entry["counterexample"] = result.counterexample
                    
                    if verbose and result.log:
                        result_entry["log"] = result.log[:2000]  # Limit log length
                    
                    results.append(result_entry)
                    
                except Exception as e:
                    error_entry = {
                        "id": entry_id,
                        "result": "error",
                        "equivalent": False,
                        "message": f"Error during verification: {str(e)}",
                        "error": str(e),
                    }
                    results.append(error_entry)
                    
                    if verbose:
                        console.print(f"[red]Error processing {entry_id}:[/red] {escape(str(e))}")
                
                progress.update(task, advance=1)
        
        # Output results
        output_json = json.dumps(results, indent=2)
        
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_json)
            if verbose:
                console.print(f"\n[green]Results written to {output_file}[/green]")
        else:
            console.print(output_json, markup=False)
        
        # Summary
        if verbose:
            equivalent_count = sum(1 for r in results if r.get('equivalent', False))
            error_count = sum(1 for r in results if r.get('result') == 'error')
            console.print(f"\n[bold]Summary:[/bold] {equivalent_count}/{total} equivalent, {error_count} errors")
            
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON file:[/red] {escape(str(e))}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error during batch processing:[/red] {escape(str(e))}")
        raise click.Abort()


def _display_result(result, title, verbose):
    """Display a CheckResult in a formatted way."""
    color = "green" if result.result in [ImplicationResult.IMPLIES, ImplicationResult.EQUIVALENT] else "red"
    
    console.print(Panel(
        f"[bold]{escape(result.message)}[/bold]",
        title=title,
        border_style=color,
    ))
    
    if result.counterexample and verbose:
        console.print("\n[yellow]Counterexample:[/yellow]")
        console.print(escape(result.counterexample))
    
    if result.log and verbose:
        console.print("\n[dim]Verification Log:[/dim]")
        console.print(escape(result.log[:2000]))  # Limit log output


if __name__ == '__main__':
    main()