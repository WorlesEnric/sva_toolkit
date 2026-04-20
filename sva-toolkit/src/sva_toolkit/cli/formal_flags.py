"""CLI flag registration helpers for explicit formal clock/reset annotations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

from sva_toolkit.formal import FormalService
from sva_toolkit.formal.model import (
    CheckResult,
    ClockMismatchError,
    ImplicationResult,
    MissingClockingError,
    MissingResetError,
    ResetMismatchError,
    UnsupportedClockingError,
)
from sva_toolkit.formal.sanitize import IdentifierError


def register(formal_group: click.Group) -> None:
    """Attach explicit clock/reset flags to the existing formal subcommands."""

    for command_name in ("check", "equivalent", "relationship"):
        command = formal_group.commands.get(command_name)
        if command is None:
            raise KeyError(f"formal command {command_name!r} is not registered on the provided group")
        _ensure_annotation_options(command)
        if not getattr(command.callback, "__t08_clock_reset_wrapped__", False):
            command.callback = _wrap_formal_callback(command_name)


def _ensure_annotation_options(command: click.Command) -> None:
    existing = {getattr(param, "name", None) for param in command.params}
    option_specs = (
        click.Option(
            ["--clock"],
            default=None,
            help="Explicit clock signal used when the property text omits a clocking event.",
        ),
        click.Option(
            ["--clock-edge"],
            type=click.Choice(["posedge", "negedge"]),
            default=None,
            help="Explicit clock edge used with --clock when the property text omits clocking.",
        ),
        click.Option(
            ["--reset"],
            default=None,
            help="Explicit reset expression used when the property text omits disable iff (...).",
        ),
    )
    for option in reversed(option_specs):
        if option.name in existing:
            continue
        insert_at = next(
            (
                index
                for index, param in enumerate(command.params)
                if getattr(param, "name", None) in {"backend", "timeout", "depth"}
            ),
            len(command.params),
        )
        command.params.insert(insert_at, option)


def _wrap_formal_callback(command_name: str) -> Callable[..., None]:
    if command_name == "check":
        callback = _formal_check_callback
    elif command_name == "equivalent":
        callback = _formal_equivalent_callback
    elif command_name == "relationship":
        callback = _formal_relationship_callback
    else:
        raise KeyError(f"unsupported formal command wrapper: {command_name}")

    setattr(callback, "__t08_clock_reset_wrapped__", True)
    return callback


def _formal_check_callback(**kwargs: Any) -> None:
    service = FormalService(
        backend=kwargs["backend"],
        timeout=kwargs["timeout"],
        depth=kwargs["depth"],
    )
    result = _run_with_cli_errors(
        lambda: service.check_implication(
            kwargs["antecedent"],
            kwargs["consequent"],
            clock=kwargs["clock"],
            clock_edge=kwargs["clock_edge"],
            reset=kwargs["reset"],
        )
    )
    _echo_check_result(result)
    if result.result is not ImplicationResult.IMPLIES:
        raise SystemExit(1)


def _formal_equivalent_callback(**kwargs: Any) -> None:
    service = FormalService(
        backend=kwargs["backend"],
        timeout=kwargs["timeout"],
        depth=kwargs["depth"],
    )
    result = _run_with_cli_errors(
        lambda: service.check_equivalence(
            kwargs["sva1"],
            kwargs["sva2"],
            clock=kwargs["clock"],
            clock_edge=kwargs["clock_edge"],
            reset=kwargs["reset"],
        )
    )
    _echo_check_result(result)
    if result.result is not ImplicationResult.EQUIVALENT:
        raise SystemExit(1)


def _formal_relationship_callback(**kwargs: Any) -> None:
    service = FormalService(
        backend=kwargs["backend"],
        timeout=kwargs["timeout"],
        depth=kwargs["depth"],
    )
    forward, reverse = _run_with_cli_errors(
        lambda: service.get_relationship(
            kwargs["sva1"],
            kwargs["sva2"],
            clock=kwargs["clock"],
            clock_edge=kwargs["clock_edge"],
            reset=kwargs["reset"],
        )
    )
    click.echo(f"SVA1 implies SVA2: {'yes' if forward else 'no'}")
    click.echo(f"SVA2 implies SVA1: {'yes' if reverse else 'no'}")


def _run_with_cli_errors(function: Callable[[], Any]) -> Any:
    try:
        return function()
    except (MissingClockingError, MissingResetError) as exc:
        raise click.UsageError(str(exc)) from exc
    except (ClockMismatchError, ResetMismatchError, UnsupportedClockingError, IdentifierError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _echo_check_result(result: CheckResult) -> None:
    click.echo(f"Result: {result.result.value}")
    click.echo(f"Message: {result.message}")
    if result.counterexample:
        click.echo(f"Counterexample: {result.counterexample}")
    if result.log:
        click.echo(f"Log:\n{result.log}")
