from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from sva_toolkit.formal.backends.ebmc import EbmcBackend
from sva_toolkit.formal.model import FormalProperty, ImplicationResult


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.name == "nt", reason="requires POSIX process groups"),
]


def test_formal_timeout_reaps_script_and_grandchild_processes(tmp_path: Path) -> None:
    # R6 regression: timed-out formal runs must continue to reap the full process group on POSIX.
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    ebmc_path = tool_dir / "ebmc"
    grandchild_marker = f"t14-orphan-{tmp_path.name}"
    ebmc_path.write_text(
        "#!/bin/sh\n"
        f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)' {shlex.quote(grandchild_marker)} &\n"
        "wait\n",
        encoding="utf-8",
    )
    ebmc_path.chmod(0o755)

    antecedent = FormalProperty(
        body="req |-> ack",
        clock_edge="posedge",
        clock_name="clk",
        reset_expr="!rst_n",
        signals={"req", "ack"},
    )
    consequent = FormalProperty(
        body="req |-> ##1 ack",
        clock_edge="posedge",
        clock_name="clk",
        reset_expr="!rst_n",
        signals={"req", "ack"},
    )
    result = EbmcBackend(tool_path=str(ebmc_path), timeout=1).check_implication(antecedent, consequent)

    assert result.result is ImplicationResult.TIMEOUT
    _assert_process_gone(str(ebmc_path))
    _assert_process_gone(grandchild_marker)
    assert "timed out" in result.message.lower()


def _assert_process_gone(marker: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker not in subprocess.run(["ps", "aux"], capture_output=True, text=True, check=True).stdout:
            return
        time.sleep(0.05)
    pytest.fail(f"process marker {marker!r} still exists after timeout cleanup")
