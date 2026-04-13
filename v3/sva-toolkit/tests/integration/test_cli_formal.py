from __future__ import annotations

import pytest

from sva_toolkit.cli.main import main
from sva_toolkit.formal.model import CheckResult, ImplicationResult


pytestmark = pytest.mark.integration


def test_formal_check_uses_mocked_service(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    captured: dict[str, object] = {}

    class _FakeFormalService:
        def check_implication(self, antecedent: str, consequent: str) -> CheckResult:
            captured["antecedent"] = antecedent
            captured["consequent"] = consequent
            return CheckResult(result=ImplicationResult.IMPLIES, message="proved")

    def _fake_build_formal_service(backend: str, timeout: int, depth: int) -> _FakeFormalService:
        captured["config"] = (backend, timeout, depth)
        return _FakeFormalService()

    monkeypatch.setattr("sva_toolkit.cli.main._build_formal_service", _fake_build_formal_service)

    result = runner.invoke(
        main,
        ["formal", "check", "req |-> ack", "req |-> ##1 ack", "--backend", "ebmc", "--timeout", "15", "--depth", "5"],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert captured["config"] == ("ebmc", 15, 5)
    assert captured["antecedent"] == "req |-> ack"
    assert captured["consequent"] == "req |-> ##1 ack"
    assert "Result: implies" in result.output
    assert "Message: proved" in result.output


def test_formal_equivalent_uses_mocked_service(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    class _FakeFormalService:
        def check_equivalence(self, sva1: str, sva2: str) -> CheckResult:
            assert sva1 == "req |-> ##1 ack"
            assert sva2 == "req |=> ack"
            return CheckResult(result=ImplicationResult.EQUIVALENT, message="equivalent")

    monkeypatch.setattr("sva_toolkit.cli.main._build_formal_service", lambda *args: _FakeFormalService())

    result = runner.invoke(
        main,
        ["formal", "equivalent", "req |-> ##1 ack", "req |=> ack"],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert "Result: equivalent" in result.output
    assert "Message: equivalent" in result.output


def test_formal_relationship_uses_mocked_service(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    class _FakeFormalService:
        def get_relationship(self, sva1: str, sva2: str) -> tuple[bool, bool]:
            assert sva1 == "req |-> ack"
            assert sva2 == "req |-> ##1 ack"
            return True, False

    monkeypatch.setattr("sva_toolkit.cli.main._build_formal_service", lambda *args: _FakeFormalService())

    result = runner.invoke(
        main,
        ["formal", "relationship", "req |-> ack", "req |-> ##1 ack"],
        prog_name="sva",
    )

    assert result.exit_code == 0
    assert "SVA1 implies SVA2: yes" in result.output
    assert "SVA2 implies SVA1: no" in result.output
