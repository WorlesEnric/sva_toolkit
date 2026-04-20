"""Timing-specific error types."""

from __future__ import annotations


class TimingDslError(ValueError):
    pass


class TimingSyntaxError(TimingDslError):
    """Source-aware syntax error with `line:column` diagnostics."""

    def __init__(self, position: int, message: str, source_text: str) -> None:
        super().__init__(message)
        self.position = position
        self.message = message
        self.source_text = source_text
        self.line, self.column = _line_column_from_position(source_text, position)

    def __str__(self) -> str:
        return f"{self.message} at {self.line}:{self.column}"


SvaTimingSyntaxError = TimingSyntaxError


def _line_column_from_position(source_text: str, position: int) -> tuple[int, int]:
    bounded = max(0, min(position, len(source_text)))
    line = source_text.count("\n", 0, bounded) + 1
    last_newline = source_text.rfind("\n", 0, bounded)
    if last_newline == -1:
        column = bounded + 1
    else:
        column = bounded - last_newline
    return (line, column)
