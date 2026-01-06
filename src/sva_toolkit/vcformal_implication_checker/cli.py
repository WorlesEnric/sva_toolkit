"""
CLI for VCFormal-based SVA Implication Checker.

Primary engine: VC Formal (vcf)
Optional cross-validation: EBMC
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from sva_toolkit.implication_checker.checker import ImplicationResult
from sva_toolkit.vcformal_implication_checker.checker import (
    CrossValidationEntry,
    CrossValidationSummary,
    VCFormalImplicationChecker,
)

console = Console()


@click.group()
@click.option("--vcf-path", default=None, help="Path to vcf binary (VC Formal)")
@click.option("--timeout", default=300, help="Timeout (seconds) per VCFormal run")
@click.option("--work-dir", default=None, help="Working directory for generated verification files")
@click.option("--keep-files", is_flag=True, help="Keep generated files after verification")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def main(
    ctx: click.Context,
    vcf_path: Optional[str],
    timeout: int,
    work_dir: Optional[str],
    keep_files: bool,
    verbose: bool,
) -> None:
    """SVA VCFormal Implication Checker - verify implication between SVA pairs."""
    ctx.ensure_object(dict)
    ctx.obj["checker"] = VCFormalImplicationChecker(
        vcf_path=vcf_path,
        timeout=timeout,
        work_dir=work_dir,
        keep_files=keep_files,
        verbose=verbose,
    )


@main.command()
@click.option("--antecedent", "-a", required=True, help="Antecedent SVA code")
@click.option("--consequent", "-c", required=True, help="Consequent SVA code")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--show-log", is_flag=True, help="Include truncated tool log in output")
@click.pass_context
def check(ctx: click.Context, antecedent: str, consequent: str, json_output: bool, show_log: bool) -> None:
    """Check if antecedent implies consequent using VCFormal."""
    checker: VCFormalImplicationChecker = ctx.obj["checker"]
    try:
        result = checker.check_implication(antecedent, consequent)
        if json_output:
            out: Dict[str, Any] = {
                "result": result.result.value,
                "implies": result.result == ImplicationResult.IMPLIES,
                "message": result.message,
            }
            if show_log and result.log:
                out["log"] = result.log[:4000]
            console.print(json.dumps(out, indent=2), markup=False)
            return
        _display_result(result.message, result.result, log=result.log if show_log else None)
    except Exception as e:
        console.print(f"[red]Error during verification:[/red] {escape(str(e))}")
        raise click.Abort()


@main.command()
@click.option("--sva1", required=True, help="First SVA code")
@click.option("--sva2", required=True, help="Second SVA code")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--show-log", is_flag=True, help="Include truncated tool log in output")
@click.pass_context
def equivalent(ctx: click.Context, sva1: str, sva2: str, json_output: bool, show_log: bool) -> None:
    """Check if two SVAs are equivalent (bidirectional implication) using VCFormal."""
    checker: VCFormalImplicationChecker = ctx.obj["checker"]
    try:
        result = checker.check_equivalence(sva1, sva2)
        if json_output:
            out: Dict[str, Any] = {
                "result": result.result.value,
                "equivalent": result.result == ImplicationResult.EQUIVALENT,
                "message": result.message,
            }
            if show_log and result.log:
                out["log"] = result.log[:4000]
            console.print(json.dumps(out, indent=2), markup=False)
            return
        _display_result(result.message, result.result, log=result.log if show_log else None)
    except Exception as e:
        console.print(f"[red]Error during verification:[/red] {escape(str(e))}")
        raise click.Abort()


@main.command()
@click.option("--sva1", required=True, help="First SVA code")
@click.option("--sva2", required=True, help="Second SVA code")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def relationship(ctx: click.Context, sva1: str, sva2: str, json_output: bool) -> None:
    """Determine the full implication relationship between two SVAs using VCFormal."""
    checker: VCFormalImplicationChecker = ctx.obj["checker"]
    try:
        sva1_implies_sva2, sva2_implies_sva1 = checker.get_implication_relationship(sva1, sva2)
        if json_output:
            out: Dict[str, Any] = {
                "sva1_implies_sva2": sva1_implies_sva2,
                "sva2_implies_sva1": sva2_implies_sva1,
                "equivalent": sva1_implies_sva2 and sva2_implies_sva1,
                "no_relationship": (not sva1_implies_sva2) and (not sva2_implies_sva1),
            }
            console.print(json.dumps(out, indent=2), markup=False)
            return
        console.print("\n[bold]Implication Relationship Analysis (VCFormal)[/bold]\n")
        console.print(f"SVA1 → SVA2: {'[green]✓ Yes[/green]' if sva1_implies_sva2 else '[red]✗ No[/red]'}")
        console.print(f"SVA2 → SVA1: {'[green]✓ Yes[/green]' if sva2_implies_sva1 else '[red]✗ No[/red]'}")
    except Exception as e:
        console.print(f"[red]Error during verification:[/red] {escape(str(e))}")
        raise click.Abort()


@main.command("cross-validate")
@click.option(
    "--input-file",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input JSON file with list of SVA pairs",
)
@click.option("--output-file", "-o", default=None, type=click.Path(), help="Output JSON (default: stdout)")
@click.option("--max-mismatches", default=50, help="Max mismatches to print to console")
@click.option("--ebmc-path", default=None, help="Path to ebmc binary (for cross-validation)")
@click.option("--ebmc-depth", default=20, help="EBMC proof depth (bound)")
@click.option("--ebmc-timeout", default=300, help="EBMC timeout (seconds)")
@click.option("--require-ebmc", is_flag=True, help="Fail if EBMC not available")
@click.pass_context
def cross_validate(
    ctx: click.Context,
    input_file: str,
    output_file: Optional[str],
    max_mismatches: int,
    ebmc_path: Optional[str],
    ebmc_depth: int,
    ebmc_timeout: int,
    require_ebmc: bool,
) -> None:
    """
    Run VCFormal implication checks and cross-validate against EBMC.

    Expected input JSON format:
    [
      {"id": "...", "antecedent": "...", "consequent": "..."},
      ...
    ]
    (Also accepts "sva1"/"sva2" as synonyms.)
    """
    checker: VCFormalImplicationChecker = ctx.obj["checker"]
    try:
        pairs: List[Dict[str, Any]] = json.loads(Path(input_file).read_text(encoding="utf-8"))
        if not isinstance(pairs, list):
            console.print("[red]Error: input JSON must be a list[/red]")
            raise click.Abort()
        if checker.verbose:
            console.print(f"[bold]Cross-validating {len(pairs)} implication checks[/bold]\n")
        if checker.verbose:
            entries: List[CrossValidationEntry] = []
            summary: CrossValidationSummary
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running VCFormal + EBMC...", total=len(pairs))
                def on_progress(current: int, total: int) -> None:
                    progress.update(task, completed=current)
                entries, summary = checker.cross_validate_with_ebmc(
                    pairs=pairs,
                    ebmc_path=ebmc_path,
                    ebmc_depth=ebmc_depth,
                    ebmc_timeout=ebmc_timeout,
                    require_ebmc=require_ebmc,
                    progress_callback=on_progress,
                )
        else:
            entries, summary = checker.cross_validate_with_ebmc(
                pairs=pairs,
                ebmc_path=ebmc_path,
                ebmc_depth=ebmc_depth,
                ebmc_timeout=ebmc_timeout,
                require_ebmc=require_ebmc,
            )
        mismatches: List[CrossValidationEntry] = [e for e in entries if (e.ebmc_result is not None and not e.aligned)]
        _display_summary(summary)
        _display_mismatches(mismatches[: max(0, max_mismatches)])
        output: Dict[str, Any] = {
            "summary": {
                "total": summary.total,
                "aligned": summary.aligned,
                "mismatched": summary.mismatched,
                "vcformal_counts": summary.vcformal_counts,
                "ebmc_counts": summary.ebmc_counts,
                "ebmc_skipped": summary.ebmc_skipped,
            },
            "mismatches": [
                {
                    "id": e.entry_id,
                    "vcformal_result": e.vcformal_result,
                    "ebmc_result": e.ebmc_result,
                    "vcformal_message": e.vcformal_message,
                    "ebmc_message": e.ebmc_message,
                    "antecedent": e.antecedent,
                    "consequent": e.consequent,
                }
                for e in mismatches
            ],
        }
        output_json: str = json.dumps(output, indent=2)
        if output_file:
            Path(output_file).write_text(output_json, encoding="utf-8")
            console.print(f"\n[green]Wrote output to {output_file}[/green]")
        else:
            console.print(output_json, markup=False)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON file:[/red] {escape(str(e))}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error during cross-validation:[/red] {escape(str(e))}")
        raise click.Abort()


def _display_result(message: str, result: ImplicationResult, log: Optional[str] = None) -> None:
    color: str = "green" if result in (ImplicationResult.IMPLIES, ImplicationResult.EQUIVALENT) else "red"
    console.print(Panel(f"[bold]{escape(message)}[/bold]", title="VCFormal Result", border_style=color))
    if log:
        console.print("\n[dim]Tool Log (truncated):[/dim]")
        console.print(escape(log[:4000]))


def _display_summary(summary: CrossValidationSummary) -> None:
    table = Table(title="Cross-validation Summary (VCFormal vs EBMC)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total", str(summary.total))
    table.add_row("Aligned", str(summary.aligned))
    table.add_row("Mismatched", str(summary.mismatched))
    table.add_row("EBMC Skipped", str(summary.ebmc_skipped))
    table.add_row("VCFormal Counts", json.dumps(summary.vcformal_counts, sort_keys=True))
    table.add_row("EBMC Counts", json.dumps(summary.ebmc_counts, sort_keys=True))
    console.print(table)


def _display_mismatches(mismatches: List[CrossValidationEntry]) -> None:
    if not mismatches:
        console.print("\n[green]No mismatches found.[/green]")
        return
    table = Table(title="Mismatched Cases (subset)")
    table.add_column("ID", style="cyan")
    table.add_column("VCFormal", style="yellow")
    table.add_column("EBMC", style="yellow")
    table.add_column("VC msg", style="dim")
    table.add_column("EBMC msg", style="dim")
    for e in mismatches:
        table.add_row(
            e.entry_id,
            e.vcformal_result,
            e.ebmc_result or "N/A",
            (e.vcformal_message or "")[:80],
            (e.ebmc_message or "")[:80] if e.ebmc_message else "",
        )
    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    # sva-vcformal-implication --vcf-path /path/to/vcf check -a "..." -c "..."
    # sva-vcformal-implication -v cross-validate -i pairs.json --ebmc-path /path/to/ebmc --ebmc-depth 20 --ebmc-timeout 300
    main()

