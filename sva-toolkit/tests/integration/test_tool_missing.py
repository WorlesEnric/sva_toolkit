from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from sva_toolkit.cli.exit_codes import ExitCode
from sva_toolkit.cli.main import main
from sva_toolkit.formal.backends.ebmc import EbmcBackend
from sva_toolkit.formal.model import FormalProperty, ImplicationResult
from sva_toolkit.runtime.process import RunResult


pytestmark = pytest.mark.integration


def test_formal_cli_requires_explicit_annotations_when_properties_omit_them(runner) -> None:
    # R1 regression: formal checks must keep refusing implicit clock/reset defaults.
    result = runner.invoke(
        main,
        [
            "formal",
            "check",
            "disable iff (!rst_n) req |-> ack",
            "disable iff (!rst_n) req |-> ##1 ack",
        ],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.USAGE_ERROR
    assert "does not name a clocking event" in (result.output + result.stderr)


def test_formal_cli_maps_missing_backends_to_tool_missing_exit_code(monkeypatch: pytest.MonkeyPatch, runner, tmp_path: Path) -> None:
    # R17 regression: CI-visible exit codes must stay typed for the formal tool-missing path.
    empty_path = tmp_path / "bin"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))

    result = runner.invoke(
        main,
        [
            "formal",
            "check",
            "req |-> ack",
            "req |-> ##1 ack",
            "--clock",
            "clk",
            "--clock-edge",
            "posedge",
            "--reset",
            "!rst_n",
        ],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.TOOL_MISSING
    assert "No formal backend is available" in result.stderr


def test_formal_backend_rejects_reserved_identifiers_before_tool_execution() -> None:
    # R4 regression: hostile identifiers must continue to be rejected before template splicing or tool launch.
    antecedent = FormalProperty(
        body="req |-> ack",
        clock_edge="posedge",
        clock_name="module",
        reset_expr="!rst_n",
        signals={"req", "ack"},
    )
    consequent = FormalProperty(
        body="req |-> ##1 ack",
        clock_edge="posedge",
        clock_name="module",
        reset_expr="!rst_n",
        signals={"req", "ack"},
    )

    result = EbmcBackend(tool_path="/bin/true").check_implication(antecedent, consequent)

    assert result.result is ImplicationResult.SYNTAX_ERROR
    assert "reserved SystemVerilog keyword" in result.message


def test_ebmc_results_retain_counterexample_text_and_rendered_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # R13 regression: formal failures must continue to surface the backend counterexample text blob.
    # R18 regression: formal results must continue to carry a reproducible emitted-module artifact.
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

    monkeypatch.setattr(
        "sva_toolkit.formal.backends.ebmc.run_tool",
        lambda *args, **kwargs: RunResult(returncode=10, stdout="counterexample\nstate 1", stderr="EBMC 5.0"),
    )

    result = EbmcBackend(tool_path="ebmc").check_implication(antecedent, consequent)

    assert result.result is ImplicationResult.NOT_IMPLIES
    assert result.counterexample is not None
    assert "counterexample\nstate 1" in result.counterexample
    assert result.module is not None
    assert "module sva_checker" in result.module


def test_generate_validate_without_verible_reports_the_missing_binary(runner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_path = tmp_path / "bin"
    missing_path.mkdir()
    monkeypatch.setenv("PATH", str(missing_path))

    result = runner.invoke(main, ["generate", "--count", "1", "--seed", "7", "--validate"], prog_name="sva")

    assert result.exit_code == ExitCode.GENERIC_ERROR
    assert "verible-verilog-syntax" in result.stderr


def test_timing_render_png_without_cairosvg_reports_install_hint(
    runner,
    timing_diagram_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "diagram.png"
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cairosvg":
            raise ImportError("No module named 'cairosvg'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = runner.invoke(
        main,
        ["timing", "render", str(timing_diagram_path), "--format", "png", "-o", str(output_path)],
        prog_name="sva",
    )

    assert result.exit_code == ExitCode.GENERIC_ERROR
    assert "Install sva-toolkit[timing-render]" in result.stderr
