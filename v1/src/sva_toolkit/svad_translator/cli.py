"""
CLI for SVAD Translator.

Reads a JSON file of SVA properties and adds a markdown translation.
"""

import json
from typing import Any, Dict

import click
from rich.console import Console

from sva_toolkit.svad_translator.translator import SVADTranslator


console = Console()


@click.group()
def main() -> None:
    """SVAD Translator - Convert SVA into a markdown SVAD template."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--expression-style",
    type=click.Choice(["symbolic", "natural"]),
    default="symbolic",
    show_default=True,
    help="How to render expression definitions.",
)
def translate(input_file: str, output_file: str, expression_style: str) -> None:
    """
    Translate SVAs in a JSON file and add a "translation" field.
    """
    console.print(f"[blue]Loading input file:[/blue] {input_file}")
    with open(input_file, "r") as f:
        data: Dict[str, Any] = json.load(f)

    if "properties" not in data or not isinstance(data["properties"], list):
        raise click.ClickException(
            "Input JSON must include a 'properties' list."
        )

    translator = SVADTranslator(expression_style=expression_style)

    translated = 0
    for prop in data["properties"]:
        sva_code = prop.get("sva")
        if not isinstance(sva_code, str) or not sva_code.strip():
            prop["translation"] = ""
            continue
        prop["translation"] = translator.translate(sva_code)
        translated += 1

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    console.print(
        f"[green]Wrote translations for {translated} properties to:[/green] {output_file}"
    )
