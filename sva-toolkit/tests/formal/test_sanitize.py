from __future__ import annotations

from pathlib import Path
import shutil
from string import Template
import subprocess

import pytest

from sva_toolkit.formal.backends.ebmc import EbmcBackend
from sva_toolkit.formal.backends.vcformal import VcformalBackend
from sva_toolkit.formal.model import FormalProperty, ImplicationResult
from sva_toolkit.formal.sanitize import IdentifierError, escape_body, validate_clock, validate_reset, validate_signal


@pytest.mark.parametrize(
    ("candidate", "accepted"),
    [
        ("req", True),
        ("ack", True),
        ("u_req_0", True),
        ("_hold", True),
        ("module", False),
        ("endmodule", False),
        ("wire", False),
        ("input", False),
        ("property", False),
        ("assert", False),
        ("u_dut.req", False),
        ("1req", False),
        ("req-ack", False),
        ("räst", False),
    ],
)
def test_validate_signal_curated_identifier_sweep(candidate: str, accepted: bool) -> None:
    if accepted:
        assert validate_signal(candidate) == candidate
        return

    with pytest.raises(IdentifierError):
        validate_signal(candidate)


def test_validate_signal_reports_hierarchical_identifier_clearly() -> None:
    with pytest.raises(IdentifierError, match="Hierarchical signal identifiers"):
        validate_signal("u_dut.req")


def test_validate_clock_accepts_simple_identifier() -> None:
    assert validate_clock("clk_i") == "clk_i"


@pytest.mark.parametrize(
    "expr",
    [
        "rst_n",
        "!rst_n",
        "~rst_n",
        "rst_n == 0",
        "1'b0 == rst_n",
        "(rst_n == 0)",
    ],
)
def test_validate_reset_accepts_simple_boolean_shapes(expr: str) -> None:
    assert validate_reset(expr)


@pytest.mark.parametrize("expr", ["u_dut.rst_n", "ready && rst_n", "module == 0"])
def test_validate_reset_rejects_invalid_shapes(expr: str) -> None:
    with pytest.raises(IdentifierError):
        validate_reset(expr)


def test_escape_body_round_trips_braces_and_system_functions() -> None:
    body = "{req, ack} == 2'b10 && $rose(done)"
    template = Template("assert property (${body});")

    rendered = template.safe_substitute(body=escape_body(body))

    assert rendered == f"assert property ({body});"


def test_ebmc_template_preserves_literal_braces(tmp_path: Path) -> None:
    backend = EbmcBackend()
    prop = FormalProperty(
        body="{req, ack} == 2'b10 && $rose(done)",
        clock_name="clk",
        reset_expr="rst_n == 0",
        signals={"req", "ack", "done"},
    )

    module_text = backend._build_module(prop, prop)

    assert "{req, ack} == 2'b10 && $rose(done)" in module_text
    assert "disable iff (rst_n == 0)" in module_text

    verible = shutil.which("verible-verilog-syntax")
    if verible is None:
        return

    sv_file = tmp_path / "sva_checker.sv"
    sv_file.write_text(module_text, encoding="utf-8")
    result = subprocess.run([verible, str(sv_file)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ebmc_backend_rejects_reserved_signal_name() -> None:
    backend = EbmcBackend()

    result = backend.check_implication(
        FormalProperty(body="req |-> module", signals={"req", "module"}),
        FormalProperty(body="req |-> ack", signals={"req", "ack"}),
    )

    assert result.result is ImplicationResult.SYNTAX_ERROR
    assert "reserved SystemVerilog keyword" in result.message


def test_vcformal_backend_rejects_hierarchical_clock_name() -> None:
    backend = VcformalBackend()

    result = backend.check_implication(
        FormalProperty(body="req |-> ack", clock_name="u_dut.clk", signals={"req", "ack"}),
        FormalProperty(body="req |-> ack", clock_name="u_dut.clk", signals={"req", "ack"}),
    )

    assert result.result is ImplicationResult.SYNTAX_ERROR
    assert "Hierarchical clock identifiers" in result.message
