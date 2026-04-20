"""CLI flag registration helpers for ``sva generate``."""

from __future__ import annotations

import click


def register(group: click.Group) -> None:
    """Attach the ``--seed`` option to the existing ``generate`` command."""
    command = group.commands.get("generate")
    if command is None:
        raise KeyError("generate command is not registered on the provided group")

    if any(getattr(param, "name", None) == "seed" for param in command.params):
        return

    seed_option = click.Option(
        ["--seed"],
        type=int,
        default=None,
        help="Seed for deterministic generation. When omitted, the chosen seed is echoed to stderr.",
    )

    insert_at = next(
        (
            index
            for index, param in enumerate(command.params)
            if getattr(param, "name", None) in {"validate", "coverage"}
        ),
        len(command.params),
    )
    command.params.insert(insert_at, seed_option)
