from __future__ import annotations

from pathlib import Path

import pytest

from sva_toolkit.cli.main import main
from sva_toolkit.formal.model import CheckResult, ImplicationResult


pytestmark = pytest.mark.integration


def test_timing_emit_sva_then_formal_check_pipeline(
    monkeypatch: pytest.MonkeyPatch, runner, timing_diagram_path: Path
) -> None:
    emit_result = runner.invoke(main, ["timing", "emit-sva", str(timing_diagram_path)], prog_name="sva")

    assert emit_result.exit_code == 0
    assert "property ready_window(int MAX_WAIT);" in emit_result.output

    captured: dict[str, str] = {}

    class _FakeFormalService:
        def check_implication(self, antecedent: str, consequent: str) -> CheckResult:
            captured["antecedent"] = antecedent
            captured["consequent"] = consequent
            return CheckResult(result=ImplicationResult.IMPLIES, message="mocked timing proof")

    monkeypatch.setattr("sva_toolkit.cli.main._build_formal_service", lambda *args: _FakeFormalService())

    formal_result = runner.invoke(
        main,
        ["formal", "check", emit_result.output, emit_result.output, "--backend", "ebmc"],
        prog_name="sva",
    )

    assert formal_result.exit_code == 0
    assert "property ready_window(int MAX_WAIT);" in captured["antecedent"]
    assert captured["antecedent"] == captured["consequent"]
    assert "Result: implies" in formal_result.output
    assert "mocked timing proof" in formal_result.output
