from __future__ import annotations


class SvaSyntaxError(Exception):
    def __init__(self, position: int, message: str, source_text: str) -> None:
        super().__init__(message)
        self.position = position
        self.message = message
        self.source_text = source_text

    def __str__(self) -> str:
        return f"{self.message} at position {self.position}"


class SvaEmitError(Exception):
    pass
