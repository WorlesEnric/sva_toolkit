from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from sva_toolkit.cli.formal_flags import register
from sva_toolkit.formal import FormalService
from sva_toolkit.formal.model import CheckResult, ImplicationResult
from sva_toolkit.runtime.config import ToolConfig
from sva_toolkit.runtime.tools import ToolRegistry


def _build_cli() -> click.Group:
    @click.group()
    def cli() -> None:
        pass

    @cli.group()
    def formal() -> None:
        pass

    @formal.command("check")
    @click.argument("antecedent")
    @click.argument("consequent")
    @click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto", show_default=True)
    @click.option("--timeout", type=int, default=300, show_default=True)
    @click.option("--depth", type=int, default=20, show_default=True)
    def _check_placeholder(**_kwargs: object) -> None:
        raise AssertionError("formal_flags.register() did not replace the check callback")

    @formal.command("equivalent")
    @click.argument("sva1")
    @click.argument("sva2")
    @click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto", show_default=True)
    @click.option("--timeout", type=int, default=300, show_default=True)
    @click.option("--depth", type=int, default=20, show_default=True)
    def _equivalent_placeholder(**_kwargs: object) -> None:
        raise AssertionError("formal_flags.register() did not replace the equivalent callback")

    @formal.command("relationship")
    @click.argument("sva1")
    @click.argument("sva2")
    @click.option("--backend", type=click.Choice(["auto", "ebmc", "vcformal"]), default="auto", show_default=True)
    @click.option("--timeout", type=int, default=300, show_default=True)
    @click.option("--depth", type=int, default=20, show_default=True)
    def _relationship_placeholder(**_kwargs: object) -> None:
        raise AssertionError("formal_flags.register() did not replace the relationship callback")

    register(formal)
    return cli


def _registry(*, ebmc: str | None = None, vcf: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry._tools["ebmc"] = ToolConfig(name="ebmc", path=ebmc, available=ebmc is not None)
    registry._tools["vcf"] = ToolConfig(name="vcf", path=vcf, available=vcf is not None)
    return registry


def test_formal_check_missing_clocking_is_usage_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        _build_cli(),
        [
            "formal",
            "check",
            "disable iff (!rst_n) req |-> ack",
            "disable iff (!rst_n) req |-> ##1 ack",
        ],
        prog_name="sva",
    )

    assert result.exit_code == 2
    assert "does not name a clocking event" in result.output


def test_formal_check_missing_reset_is_usage_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        _build_cli(),
        [
            "formal",
            "check",
            "@(posedge clk) req |-> ack",
            "@(posedge clk) req |-> ##1 ack",
        ],
        prog_name="sva",
    )

    assert result.exit_code == 2
    assert "does not name a reset expression" in result.output


def test_formal_service_treats_active_low_reset_aliases_as_equivalent(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FormalService(registry=_registry(ebmc="/tools/ebmc"))
    captured: list[tuple[str | None, str | None]] = []

    class _FakeBackend:
        def check_implication(self, antecedent, consequent) -> CheckResult:
            captured.append((antecedent.reset_expr, consequent.reset_expr))
            return CheckResult(result=ImplicationResult.IMPLIES, message="proved")

    monkeypatch.setattr(service, "_select_backend", lambda: _FakeBackend())

    result = service.check_equivalence(
        "@(posedge clk) disable iff (!rst_n) req |-> ack",
        "@(posedge clk) disable iff (rst_n == 0) req |-> ack",
    )

    assert result.result is ImplicationResult.EQUIVALENT
    assert captured == [("rst_n == 0", "rst_n == 0"), ("rst_n == 0", "rst_n == 0")]


@pytest.mark.parametrize(
    ("args", "expected_backend", "expected_method"),
    [
        (
            [
                "formal",
                "check",
                "req |-> ack",
                "req |-> ##1 ack",
                "--backend",
                "ebmc",
                "--clock",
                "hclk",
                "--clock-edge",
                "posedge",
                "--reset",
                "!rst_n",
            ],
            "ebmc",
            "check_implication",
        ),
        (
            [
                "formal",
                "equivalent",
                "req |-> ack",
                "req |-> ack",
                "--backend",
                "vcformal",
                "--clock",
                "hclk",
                "--clock-edge",
                "posedge",
                "--reset",
                "!rst_n",
            ],
            "vcformal",
            "check_equivalence",
        ),
    ],
)
def test_formal_flags_plumb_clock_reset_kwargs_to_service(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    expected_backend: str,
    expected_method: str,
) -> None:
    captured: dict[str, object] = {}

    class _FakeFormalService:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        def check_implication(self, antecedent: str, consequent: str, **kwargs) -> CheckResult:
            captured["method"] = "check_implication"
            captured["pair"] = (antecedent, consequent)
            captured["kwargs"] = kwargs
            return CheckResult(result=ImplicationResult.IMPLIES, message="proved")

        def check_equivalence(self, sva1: str, sva2: str, **kwargs) -> CheckResult:
            captured["method"] = "check_equivalence"
            captured["pair"] = (sva1, sva2)
            captured["kwargs"] = kwargs
            return CheckResult(result=ImplicationResult.EQUIVALENT, message="equivalent")

        def get_relationship(self, sva1: str, sva2: str, **kwargs) -> tuple[bool, bool]:
            captured["method"] = "get_relationship"
            captured["pair"] = (sva1, sva2)
            captured["kwargs"] = kwargs
            return True, False

    monkeypatch.setattr("sva_toolkit.cli.formal_flags.FormalService", _FakeFormalService)

    result = CliRunner().invoke(_build_cli(), args, prog_name="sva")

    assert result.exit_code == 0
    assert captured["init"] == {"backend": expected_backend, "timeout": 300, "depth": 20}
    assert captured["method"] == expected_method
    assert captured["kwargs"] == {"clock": "hclk", "clock_edge": "posedge", "reset": "!rst_n"}
