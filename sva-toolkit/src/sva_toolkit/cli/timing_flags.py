"""CLI wrappers for timing extraction status surfacing."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import click

from sva_toolkit.timing.bridge.status import LossyExtractionError

_REPORT_META_KEY = "timing_extraction_report"


def register(timing_group: click.Group) -> None:
    """Attach exact-extraction enforcement to timing extraction commands."""

    for command_name in ("extract-sva", "bundle-sva"):
        command = timing_group.commands.get(command_name)
        if command is None:
            raise KeyError(f"timing command {command_name!r} is not registered on the provided group")
        callback = command.callback
        if callback is None or getattr(callback, "__t11_lossy_wrapped__", False):
            continue
        command.callback = _wrap_timing_callback(callback)


def record_report(report: object) -> None:
    """Expose the last extraction report to the timing CLI wrapper."""

    click.get_current_context().meta[_REPORT_META_KEY] = report


def _wrap_timing_callback(callback: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(callback)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        ctx = click.get_current_context()
        ctx.meta.pop(_REPORT_META_KEY, None)
        result = callback(*args, **kwargs)
        report = ctx.meta.pop(_REPORT_META_KEY, None)
        if report is not None and report.worst_status().value != "exact":
            raise LossyExtractionError(report)
        return result

    setattr(wrapped, "__t11_lossy_wrapped__", True)
    return wrapped


__all__ = ["record_report", "register"]
