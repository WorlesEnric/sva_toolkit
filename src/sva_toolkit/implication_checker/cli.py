"""
CLI for SVA Implication Checker.
"""

import click
import json
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

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