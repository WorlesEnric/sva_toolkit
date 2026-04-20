from __future__ import annotations

import pytest

from sva_toolkit.cli.main import main
from sva_toolkit.formal.model import CheckResult, ImplicationResult


pytestmark = pytest.mark.integration


def test_formal_check_uses_mocked_service(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    captured: dict[str, object] = {}

    class _FakeFormalService:
        def __init__(self, **kwargs: object) -> None:
            captured["config"] = kwargs

        def check_implication(self, antecedent: str, consequent: str, **kwargs: object) -> CheckResult:
            captured["antecedent"] = antecedent
            captured["consequent"] = consequent
            captured["kwargs"] = kwargs
            return CheckResult(result=ImplicationResult.IMPLIES, message="proved")

    monkeypatch.setattr("sva_toolkit.cli.formal_flags.FormalService", _FakeFormalService)

    result = runner.invoke(
        main,
        [
            "formal",
            "check",
            "req |-> ack",
            "req |-> ##1 ack",
            "--backend",
            "ebmc",
            "--timeout",
            "15",
            "--depth",
            "5",
            "--clock",
            "clk",
            "--clock-edge",
            "posedge",
            "--reset",
            "!rst_n",
        ],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert captured["config"] == {"backend": "ebmc", "timeout": 15, "depth": 5}
    assert captured["antecedent"] == "req |-> ack"
    assert captured["consequent"] == "req |-> ##1 ack"
    assert captured["kwargs"] == {"clock": "clk", "clock_edge": "posedge", "reset": "!rst_n"}
    assert "Result: implies" in result.output
    assert "Message: proved" in result.output


def test_formal_equivalent_uses_mocked_service(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    class _FakeFormalService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def check_equivalence(self, sva1: str, sva2: str, **kwargs: object) -> CheckResult:
            assert sva1 == "req |-> ##1 ack"
            assert sva2 == "req |=> ack"
            assert kwargs == {"clock": "clk", "clock_edge": "posedge", "reset": "!rst_n"}
            return CheckResult(result=ImplicationResult.EQUIVALENT, message="equivalent")

    monkeypatch.setattr("sva_toolkit.cli.formal_flags.FormalService", _FakeFormalService)

    result = runner.invoke(
        main,
        [
            "formal",
            "equivalent",
            "req |-> ##1 ack",
            "req |=> ack",
            "--clock",
            "clk",
            "--clock-edge",
            "posedge",
            "--reset",
            "!rst_n",
        ],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert "Result: equivalent" in result.output
    assert "Message: equivalent" in result.output


def test_formal_relationship_uses_mocked_service(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    class _FakeFormalService:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def check_implication(self, sva1: str, sva2: str, **kwargs: object) -> CheckResult:
            assert kwargs == {"clock": "clk", "clock_edge": "posedge", "reset": "!rst_n"}
            if sva1 == "req |-> ack":
                assert sva2 == "req |-> ##1 ack"
                return CheckResult(result=ImplicationResult.IMPLIES, message="forward")
            assert sva1 == "req |-> ##1 ack"
            assert sva2 == "req |-> ack"
            return CheckResult(result=ImplicationResult.NOT_IMPLIES, message="reverse")

    monkeypatch.setattr("sva_toolkit.cli.formal_flags.FormalService", _FakeFormalService)

    result = runner.invoke(
        main,
        [
            "formal",
            "relationship",
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

    assert result.exit_code == 0
    assert "SVA1 implies SVA2: yes" in result.output
    assert "SVA2 implies SVA1: no" in result.output
