from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import sys

import pytest

import sva_toolkit.runtime.process as process_module
from sva_toolkit.runtime.errors import ToolMissingError
from sva_toolkit.runtime.process import RunResult, make_work_dir, run_tool


def test_make_work_dir_creates_directory() -> None:
    work_dir = make_work_dir(prefix="sva_toolkit_test_")

    try:
        assert Path(work_dir).is_dir()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_make_work_dir_uses_runtime_workdir_root_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(process_module.runtime_config, "workdir_root", lambda: tmp_path, raising=False)

    work_dir = make_work_dir(prefix="sva_toolkit_root_")

    try:
        assert Path(work_dir).parent == tmp_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not meaningful on Windows")
def test_make_work_dir_enforces_private_permissions() -> None:
    work_dir = make_work_dir(prefix="sva_toolkit_mode_")

    try:
        assert stat.S_IMODE(os.stat(work_dir).st_mode) == 0o700
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_run_result_dataclass_creation() -> None:
    result = RunResult(returncode=0, stdout="hello\n", stderr="", timed_out=False)

    assert result.returncode == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.timed_out is False


def test_run_tool_executes_simple_command() -> None:
    result = run_tool([sys.executable, "-c", "print('hello')"])

    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""
    assert result.timed_out is False


def test_run_tool_returns_timeout_result() -> None:
    result = run_tool(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"],
        timeout=1,
    )

    assert result.returncode == -1
    assert result.timed_out is True
    assert "started" in result.stdout


def test_run_tool_with_missing_command_raises_typed_error() -> None:
    with pytest.raises(ToolMissingError) as exc_info:
        run_tool(["definitely-not-a-real-command-xyz"])

    error = exc_info.value

    assert isinstance(error, FileNotFoundError)
    assert error.path == "definitely-not-a-real-command-xyz"
    assert error.cmd == ("definitely-not-a-real-command-xyz",)
