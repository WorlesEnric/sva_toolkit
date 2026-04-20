"""Typed runtime errors surfaced by subprocess execution helpers."""

from __future__ import annotations

import errno
from typing import Sequence


class ToolMissingError(FileNotFoundError):
    """Raised when a configured external tool binary cannot be executed."""

    def __init__(self, path: str | None, cmd: Sequence[str] | str) -> None:
        normalized_cmd: tuple[str, ...] | str
        if isinstance(cmd, str):
            normalized_cmd = cmd
        else:
            normalized_cmd = tuple(str(part) for part in cmd)

        resolved_path = path or _command_head(normalized_cmd)
        command_text = normalized_cmd if isinstance(normalized_cmd, str) else " ".join(normalized_cmd)

        self.path = resolved_path
        self.cmd = normalized_cmd

        super().__init__(
            errno.ENOENT,
            f"Required tool '{resolved_path}' is not available for command: {command_text}",
            resolved_path,
        )


def _command_head(cmd: tuple[str, ...] | str) -> str:
    if isinstance(cmd, str):
        return cmd
    if not cmd:
        return "<unknown>"
    return cmd[0]
