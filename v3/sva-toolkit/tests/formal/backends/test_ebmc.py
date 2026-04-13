from __future__ import annotations

from pathlib import Path

from sva_toolkit.formal.backends.ebmc import EbmcBackend
from sva_toolkit.formal.model import FormalProperty, ImplicationResult
from sva_toolkit.runtime.process import RunResult


def test_ebmc_backend_runs_tool_and_writes_module(monkeypatch, tmp_path: Path) -> None:
    backend = EbmcBackend(tool_path="/tools/ebmc", depth=12, timeout=45)
    backend.keep_files = True
    calls: dict[str, object] = {}

    def _run_tool(cmd, *, cwd, timeout, input_text=None, capture=True):
        calls["cmd"] = list(cmd)
        calls["cwd"] = cwd
        calls["timeout"] = timeout
        calls["input_text"] = input_text
        calls["capture"] = capture
        return RunResult(returncode=0, stdout="VERIFICATION SUCCESSFUL", stderr="")

    monkeypatch.setattr("sva_toolkit.formal.backends.ebmc.make_work_dir", lambda prefix="sva_ebmc_": str(tmp_path))
    monkeypatch.setattr("sva_toolkit.formal.backends.ebmc.run_tool", _run_tool)

    result = backend.check_implication(
        FormalProperty(body="req |-> gnt", signals={"req", "gnt"}),
        FormalProperty(body="req |-> ##1 done", signals={"req", "done"}),
    )

    assert result.result is ImplicationResult.IMPLIES
    assert calls["cmd"] == [
        "/tools/ebmc",
        "--top",
        "sva_checker",
        "--bound",
        "12",
        str(tmp_path / "sva_checker.sv"),
    ]
    assert calls["cwd"] == tmp_path
    assert calls["timeout"] == 45
    module_text = (tmp_path / "sva_checker.sv").read_text(encoding="utf-8")
    assert "assume property" in module_text
    assert "assert property" in module_text
    assert "input wire req" in module_text
    assert "input wire gnt" in module_text
    assert "input wire done" in module_text


def test_ebmc_backend_reports_counterexample(monkeypatch, tmp_path: Path) -> None:
    backend = EbmcBackend(tool_path="/tools/ebmc")
    backend.keep_files = True

    monkeypatch.setattr("sva_toolkit.formal.backends.ebmc.make_work_dir", lambda prefix="sva_ebmc_": str(tmp_path))
    monkeypatch.setattr(
        "sva_toolkit.formal.backends.ebmc.run_tool",
        lambda *args, **kwargs: RunResult(returncode=10, stdout="counterexample\nstate 1", stderr=""),
    )

    result = backend.check_implication(
        FormalProperty(body="req |-> gnt", signals={"req", "gnt"}),
        FormalProperty(body="req |-> done", signals={"req", "done"}),
    )

    assert result.result is ImplicationResult.NOT_IMPLIES
    assert result.counterexample == "counterexample\nstate 1"
