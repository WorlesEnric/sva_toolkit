"""
CLI for SVA AST Parser.
"""

import click
import json
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

from sva_toolkit.ast_parser.parser import SVAASTParser

console = Console()


@click.group()
@click.option('--verible-path', default='3rd_party/verible_bin/verible-verilog-syntax',
              help='Path to verible-verilog-syntax binary')
@click.pass_context
def main(ctx, verible_path):
    """SVA AST Parser - Extract structured information from SVA code."""
    ctx.ensure_object(dict)
    ctx.obj['parser'] = SVAASTParser(verible_path=verible_path)


@main.command()
@click.argument('sva_code')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.pass_context
def parse(ctx, sva_code, json_output):
    """Parse SVA code and extract structure."""
    parser = ctx.obj['parser']
    
    try:
        structure = parser.parse(sva_code)
        
        if json_output:
            console.print(json.dumps(structure.to_dict(), indent=2))
        else:
            _display_structure(structure)
            
    except Exception as e:
        console.print(f"[red]Error parsing SVA:[/red] {e}")
        raise click.Abort()


@main.command('parse-file')
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.pass_context
def parse_file(ctx, filepath, json_output):
    """Parse SVA code from a file."""
    parser = ctx.obj['parser']
    
    try:
        structures = parser.parse_file(filepath)
        
        if json_output:
            output = [s.to_dict() for s in structures]
            console.print(json.dumps(output, indent=2))
        else:
            for i, structure in enumerate(structures):
                console.print(f"\n[bold blue]═══ SVA #{i+1} ═══[/bold blue]")
                _display_structure(structure)
                
    except Exception as e:
        console.print(f"[red]Error parsing file:[/red] {e}")
        raise click.Abort()


@main.command()
@click.argument('sva_code')
@click.pass_context
def signals(ctx, sva_code):
    """Extract all signals from SVA code."""
    parser = ctx.obj['parser']
    
    try:
        signal_names = parser.get_all_signals(sva_code)
        
        table = Table(title="Signals")
        table.add_column("Signal Name", style="cyan")
        
        for name in sorted(signal_names):
            table.add_row(name)
            
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error extracting signals:[/red] {e}")
        raise click.Abort()


def _display_structure(structure):
    """Display SVAStructure in a formatted way."""
    # Property info
    table = Table(title="SVA Structure", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim", width=20)
    table.add_column("Value", style="cyan")
    
    table.add_row("Property Name", structure.property_name or "N/A")
    table.add_row("Type", _get_type_str(structure))
    table.add_row("Clock", f"{structure.clock_edge}({structure.clock_signal})" 
                  if structure.clock_signal else "N/A")
    table.add_row("Reset", f"{structure.reset_signal} (Active {'Low' if structure.reset_active_low else 'High'})"
                  if structure.reset_signal else "N/A")
    table.add_row("Disable Condition", structure.disable_condition or "N/A")
    
    console.print(table)
    
    # Implication
    if structure.implication_type.value:
        impl_table = Table(title="Implication", show_header=True, header_style="bold magenta")
        impl_table.add_column("Component", style="dim", width=15)
        impl_table.add_column("Expression", style="green")
        
        impl_table.add_row("Type", structure.implication_type.value)
        impl_table.add_row("Antecedent", structure.antecedent or "N/A")
        impl_table.add_row("Consequent", structure.consequent or "N/A")
        
        console.print(impl_table)
    
    # Signals
    if structure.signals:
        sig_table = Table(title="Signals", show_header=True, header_style="bold magenta")
        sig_table.add_column("Name", style="cyan")
        sig_table.add_column("Clock?", style="yellow")
        sig_table.add_column("Reset?", style="red")
        
        for sig in sorted(structure.signals, key=lambda s: s.name):
            sig_table.add_row(
                sig.name,
                "✓" if sig.is_clock else "",
                "✓" if sig.is_reset else ""
            )
        
        console.print(sig_table)
    
    # Built-in functions
    if structure.builtin_functions:
        func_table = Table(title="Built-in Functions", show_header=True, header_style="bold magenta")
        func_table.add_column("Function", style="cyan")
        func_table.add_column("Arguments", style="green")
        
        for func in structure.builtin_functions:
            func_table.add_row(func.name, ", ".join(func.arguments))
        
        console.print(func_table)
    
    # Temporal operators
    if structure.temporal_operators:
        ops_table = Table(title="Temporal Operators", show_header=True, header_style="bold magenta")
        ops_table.add_column("Operator", style="cyan")
        
        for op in structure.temporal_operators:
            ops_table.add_row(op.value)
        
        console.print(ops_table)
    
    # Delays
    if structure.delays:
        delay_table = Table(title="Delays", show_header=True, header_style="bold magenta")
        delay_table.add_column("Min Cycles", style="cyan")
        delay_table.add_column("Max Cycles", style="green")
        delay_table.add_column("Unbounded?", style="yellow")
        
        for delay in structure.delays:
            delay_table.add_row(
                str(delay.min_cycles),
                str(delay.max_cycles) if delay.max_cycles else "N/A",
                "✓" if delay.is_unbounded else ""
            )
        
        console.print(delay_table)


def _get_type_str(structure):
    """Get string representation of assertion type."""
    types = []
    if structure.is_assertion:
        types.append("assertion")
    if structure.is_assumption:
        types.append("assumption")
    if structure.is_cover:
        types.append("cover")
    return ", ".join(types) if types else "unknown"


if __name__ == '__main__':
    main()
