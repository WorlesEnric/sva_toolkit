"""
CLI for SVA Chain-of-Thought Builder.
"""

import click
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from sva_toolkit.cot_builder.builder import SVACoTBuilder

console = Console()


@click.group()
@click.pass_context
def main(ctx):
    """SVA CoT Builder - Generate Chain-of-Thought reasoning from SVA code."""
    ctx.ensure_object(dict)
    ctx.obj['builder'] = SVACoTBuilder()


@main.command()
@click.argument('sva_code')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.option('--raw', '-r', is_flag=True, help='Output raw markdown without rendering')
@click.pass_context
def build(ctx, sva_code, output, json_output, raw):
    """Build Chain-of-Thought reasoning from SVA code."""
    builder = ctx.obj['builder']
    
    try:
        cot = builder.build(sva_code)
        
        if json_output:
            sections = builder.get_cot_sections(sva_code)
            output_data = {
                "sva_code": sva_code,
                "cot": cot,
                "sections": [{"title": s.title, "content": s.content} for s in sections]
            }
            result = json.dumps(output_data, indent=2)
            
            if output:
                with open(output, 'w') as f:
                    f.write(result)
                console.print(f"[green]CoT saved to {output}[/green]")
            else:
                console.print(result)
        else:
            if output:
                with open(output, 'w') as f:
                    f.write(cot)
                console.print(f"[green]CoT saved to {output}[/green]")
            elif raw:
                console.print(cot)
            else:
                # Render markdown
                md = Markdown(cot)
                console.print(md)
                
    except Exception as e:
        console.print(f"[red]Error building CoT:[/red] {e}")
        raise click.Abort()


@main.command('build-file')
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.pass_context
def build_file(ctx, filepath, output, json_output):
    """Build Chain-of-Thought from SVA code in a file."""
    builder = ctx.obj['builder']
    
    try:
        with open(filepath, 'r') as f:
            sva_code = f.read()
        
        cot = builder.build(sva_code)
        
        if json_output:
            sections = builder.get_cot_sections(sva_code)
            output_data = {
                "sva_code": sva_code,
                "cot": cot,
                "sections": [{"title": s.title, "content": s.content} for s in sections]
            }
            result = json.dumps(output_data, indent=2)
            
            if output:
                with open(output, 'w') as f:
                    f.write(result)
                console.print(f"[green]CoT saved to {output}[/green]")
            else:
                console.print(result)
        else:
            if output:
                with open(output, 'w') as f:
                    f.write(cot)
                console.print(f"[green]CoT saved to {output}[/green]")
            else:
                md = Markdown(cot)
                console.print(md)
                
    except Exception as e:
        console.print(f"[red]Error building CoT from file:[/red] {e}")
        raise click.Abort()


@main.command()
@click.argument('sva_code')
@click.pass_context
def sections(ctx, sva_code):
    """List the sections of the Chain-of-Thought."""
    builder = ctx.obj['builder']
    
    try:
        cot_sections = builder.get_cot_sections(sva_code)
        
        console.print("\n[bold]Chain-of-Thought Sections:[/bold]\n")
        
        for i, section in enumerate(cot_sections, 1):
            console.print(Panel(
                section.content,
                title=f"[bold blue]{i}. {section.title}[/bold blue]",
                border_style="blue",
            ))
            
    except Exception as e:
        console.print(f"[red]Error extracting sections:[/red] {e}")
        raise click.Abort()


if __name__ == '__main__':
    main()
