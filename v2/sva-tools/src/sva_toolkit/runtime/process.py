from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_tool(
    cmd: Sequence[str] | str,
    *,
    cwd: str | os.PathLike[str] | None = None,
    timeout: int = 300,
    input_text: str | None = None,
    capture: bool = True,
) -> RunResult:
    try:
        completed = subprocess.run(
            cmd,
            cwd=os.fspath(cwd) if cwd is not None else None,
            input=input_text,
            capture_output=capture,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return RunResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Failed to execute tool: {cmd}") from exc

    return RunResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        timed_out=False,
    )


def make_work_dir(prefix: str = "sva_") -> str:
    return tempfile.mkdtemp(prefix=prefix)
