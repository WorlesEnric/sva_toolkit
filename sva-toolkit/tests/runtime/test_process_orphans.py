"""POSIX orphan-reaping regression coverage for runtime process timeouts."""

from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from sva_toolkit.runtime.process import run_tool

pytestmark = pytest.mark.skipif(os.name == "nt", reason="requires POSIX process groups")


def test_run_tool_timeout_reaps_background_grandchild(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    script_path = tmp_path / "spawn_orphan.sh"
    script_path.write_text(
        "#!/bin/sh\n"
        "sleep 60 &\n"
        "printf '%s\\n' \"$!\" > \"$1\"\n"
        "sleep 60\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    result = run_tool(["sh", str(script_path), str(pid_file)], timeout=1)

    assert result.timed_out is True
    assert pid_file.exists()

    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    _assert_pid_gone(grandchild_pid)


def _assert_pid_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)

    pytest.fail(f"process {pid} still exists after timeout cleanup")
