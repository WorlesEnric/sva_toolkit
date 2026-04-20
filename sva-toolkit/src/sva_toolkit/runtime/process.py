"""Subprocess helpers for optional tool invocations.

On POSIX, each tool is launched in its own session so timeout handling can
terminate the entire process group and avoid orphaned EBMC/VCF helper
processes. On Windows, Python only gives us best-effort termination of the
direct child via ``terminate()`` followed by ``kill()``; helper processes may
still survive, and T15 should carry that caveat into ``docs/LIMITATIONS.md``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Sequence

from . import config as runtime_config
from .errors import ToolMissingError

_TERMINATE_GRACE_SECONDS = 1.0
_KILL_GRACE_SECONDS = 1.0


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
    resolved_cwd = os.fspath(cwd) if cwd is not None else None

    try:
        process = subprocess.Popen(
            cmd,
            cwd=resolved_cwd,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True,
            start_new_session=os.name != "nt",
        )
    except FileNotFoundError as exc:
        if resolved_cwd is not None and exc.filename == resolved_cwd:
            raise
        raise ToolMissingError(exc.filename or _command_path(cmd), cmd) from exc

    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_timed_out_process(process)
        stdout, stderr = _drain_process(process)
        return RunResult(
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )

    return RunResult(
        returncode=process.returncode or 0,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=False,
    )


def make_work_dir(prefix: str = "sva_") -> str:
    workdir_root = getattr(runtime_config, "workdir_root", None)
    base_dir = os.fspath(workdir_root()) if callable(workdir_root) else None
    work_dir = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
    if os.name != "nt":
        os.chmod(work_dir, 0o700)
    return work_dir


def _command_path(cmd: Sequence[str] | str) -> str:
    if isinstance(cmd, str):
        return cmd
    if not cmd:
        return "<unknown>"
    return str(cmd[0])


def _terminate_timed_out_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        _terminate_windows_process(process)
        return

    _terminate_posix_process_group(process)


def _terminate_posix_process_group(process: subprocess.Popen[str]) -> None:
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _terminate_windows_process(process: subprocess.Popen[str]) -> None:
    try:
        process.terminate()
    except OSError:
        return

    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        process.kill()
    except OSError:
        return

    try:
        process.wait(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _drain_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        stdout, stderr = process.communicate(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    return stdout or "", stderr or ""
