"""Stable CLI exit-code mapping for typed command failures."""

from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import Iterator

import click

from sva_toolkit.runtime.errors import ToolMissingError
from sva_toolkit.runtime.retry import RetryExhaustedError
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.timing.bridge.status import LossyExtractionError
from sva_toolkit.timing.errors import TimingSyntaxError


class ExitCode(IntEnum):
    SUCCESS = 0
    GENERIC_ERROR = 1
    USAGE_ERROR = 2
    TOOL_MISSING = 3
    PARSE_ERROR = 4
    TIMEOUT = 5
    LOSSY_EXTRACTION = 6
    BACKEND_UNAVAILABLE = 7


class BackendUnavailableError(RuntimeError):
    """Raised when an optional backend dependency is present in the CLI surface but unusable."""


class ReportedParseError(RuntimeError):
    """Raised when a domain surface reports syntax failure without surfacing the original parser type."""


def exit_code_for(exc: BaseException) -> ExitCode:
    if _chain_contains(exc, click.UsageError):
        return ExitCode.USAGE_ERROR
    if _chain_contains(exc, ToolMissingError):
        return ExitCode.TOOL_MISSING
    if _chain_contains(exc, (SvaSyntaxError, TimingSyntaxError, ReportedParseError)):
        return ExitCode.PARSE_ERROR
    if _chain_contains(exc, TimeoutError):
        return ExitCode.TIMEOUT
    if _chain_contains(exc, LossyExtractionError):
        return ExitCode.LOSSY_EXTRACTION
    if _chain_contains(exc, BackendUnavailableError):
        return ExitCode.BACKEND_UNAVAILABLE
    if isinstance(exc, click.ClickException) and exc.exit_code == ExitCode.USAGE_ERROR:
        return ExitCode.USAGE_ERROR
    return ExitCode.GENERIC_ERROR


def _chain_contains(exc: BaseException, expected: type[BaseException] | tuple[type[BaseException], ...]) -> bool:
    return any(isinstance(item, expected) for item in iter_exception_chain(exc))


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    queue: deque[BaseException] = deque([exc])
    seen: set[int] = set()

    while queue:
        current = queue.popleft()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current

        if isinstance(current, RetryExhaustedError):
            queue.append(current.last_error)
        if current.__cause__ is not None:
            queue.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            queue.append(current.__context__)


__all__ = [
    "BackendUnavailableError",
    "ExitCode",
    "ReportedParseError",
    "exit_code_for",
    "iter_exception_chain",
]
