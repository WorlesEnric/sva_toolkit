from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from sva_toolkit.runtime.process import RunResult, make_work_dir, run_tool


def test_make_work_dir_creates_directory() -> None:
    work_dir = make_work_dir(prefix="sva_toolkit_test_")

    try:
        assert Path(work_dir).is_dir()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_run_result_dataclass_creation() -> None:
    result = RunResult(returncode=0, stdout="hello\n", stderr="", timed_out=False)

    assert result.returncode == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.timed_out is False


@pytest.mark.skipif(os.name == "nt", reason="uses a Unix-style echo command")
def test_run_tool_executes_simple_command() -> None:
    result = run_tool(["echo", "hello"])

    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""
    assert result.timed_out is False


def test_run_tool_with_missing_command_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="Failed to execute tool"):
        run_tool(["definitely-not-a-real-command-xyz"])
