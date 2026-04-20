from __future__ import annotations

from pathlib import Path

from sva_toolkit.formal.backends.vcformal import VcformalBackend
from sva_toolkit.formal.model import FormalProperty, ImplicationResult
from sva_toolkit.runtime.process import RunResult


def test_vcformal_backend_runs_tool_and_reads_report(monkeypatch, tmp_path: Path) -> None:
    backend = VcformalBackend(tool_path="/tools/vcf", timeout=90)
    backend.keep_files = True
    calls: dict[str, object] = {}

    def _run_tool(cmd, *, cwd, timeout, input_text=None, capture=True):
        calls["cmd"] = list(cmd)
        calls["cwd"] = cwd
        calls["timeout"] = timeout
        Path(cwd, "report.txt").write_text("[1] assert_consequent proven", encoding="utf-8")
        return RunResult(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr("sva_toolkit.formal.backends.vcformal.make_work_dir", lambda prefix="sva_vcf_": str(tmp_path))
    monkeypatch.setattr("sva_toolkit.formal.backends.vcformal.run_tool", _run_tool)

    result = backend.check_implication(
        FormalProperty(body="req |-> gnt", clock_name="clk", clock_edge="posedge", reset_expr="!rst_n", signals={"req", "gnt"}),
        FormalProperty(
            body="req |-> ##1 done",
            clock_name="clk",
            clock_edge="posedge",
            reset_expr="!rst_n",
            signals={"req", "done"},
        ),
    )

    assert result.result is ImplicationResult.IMPLIES
    assert calls["cmd"] == ["/tools/vcf", "-f", str(tmp_path / "run.tcl")]
    assert calls["cwd"] == tmp_path
    assert calls["timeout"] == 90
    assert "create_clock clk -period 1" in (tmp_path / "run.tcl").read_text(encoding="utf-8")
    assert "read_file -top sva_checker -format sverilog -sva" in (tmp_path / "run.tcl").read_text(encoding="utf-8")


def test_vcformal_backend_reports_falsified_result(monkeypatch, tmp_path: Path) -> None:
    backend = VcformalBackend(tool_path="/tools/vcf")
    backend.keep_files = True

    def _run_tool(cmd, *, cwd, timeout, input_text=None, capture=True):
        Path(cwd, "report.txt").write_text("[1] assert_consequent falsified", encoding="utf-8")
        return RunResult(returncode=0, stdout="counterexample trace", stderr="")

    monkeypatch.setattr("sva_toolkit.formal.backends.vcformal.make_work_dir", lambda prefix="sva_vcf_": str(tmp_path))
    monkeypatch.setattr("sva_toolkit.formal.backends.vcformal.run_tool", _run_tool)

    result = backend.check_implication(
        FormalProperty(body="req |-> gnt", clock_name="clk", clock_edge="posedge", reset_expr="!rst_n", signals={"req", "gnt"}),
        FormalProperty(body="req |-> done", clock_name="clk", clock_edge="posedge", reset_expr="!rst_n", signals={"req", "done"}),
    )

    assert result.result is ImplicationResult.NOT_IMPLIES
    assert result.counterexample is not None
    assert "counterexample trace" in result.counterexample
    assert "assert_consequent falsified" in result.counterexample
