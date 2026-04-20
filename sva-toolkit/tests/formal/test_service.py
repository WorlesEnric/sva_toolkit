from __future__ import annotations

import pytest

from sva_toolkit.formal.model import CheckResult, ClockMismatchError, ImplicationResult, MissingClockingError
from sva_toolkit.runtime.config import ToolConfig
from sva_toolkit.runtime.tools import ToolRegistry


def _registry(*, ebmc: str | None = None, vcf: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry._tools["ebmc"] = ToolConfig(name="ebmc", path=ebmc, available=ebmc is not None)
    registry._tools["vcf"] = ToolConfig(name="vcf", path=vcf, available=vcf is not None)
    return registry


def test_formal_service_rejects_invalid_backend() -> None:
    from sva_toolkit.formal.service import FormalService

    with pytest.raises(ValueError, match="backend must be one of"):
        FormalService(backend="bad")


def test_formal_service_reports_parse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from sva_toolkit.formal.service import FormalService

    service = FormalService(registry=_registry(ebmc="/tools/ebmc"))
    monkeypatch.setattr(
        "sva_toolkit.formal.service.parse_property",
        lambda text, **_kwargs: (_ for _ in ()).throw(ValueError(text)),
    )

    result = service.check_implication(
        "bad antecedent",
        "bad consequent",
        clock="clk",
        clock_edge="posedge",
        reset="!rst_n",
    )

    assert result.result is ImplicationResult.SYNTAX_ERROR
    assert "Failed to parse property text" in result.message


def test_formal_service_requires_explicit_clocking() -> None:
    from sva_toolkit.formal.service import FormalService

    service = FormalService(registry=_registry(ebmc="/tools/ebmc"))

    with pytest.raises(MissingClockingError, match="does not name a clocking event"):
        service.check_implication(
            "disable iff (!rst_n) req |-> gnt",
            "disable iff (!rst_n) req |-> ##1 gnt",
        )


def test_formal_service_requires_matching_effective_clocking() -> None:
    from sva_toolkit.formal.service import FormalService

    service = FormalService(registry=_registry(ebmc="/tools/ebmc"))

    with pytest.raises(ClockMismatchError, match="Property clock mismatch"):
        service.check_implication(
            "@(posedge clk) disable iff (!rst_n) req |-> gnt",
            "@(negedge clk) disable iff (!rst_n) req |-> ##1 gnt",
        )


def test_formal_service_reports_missing_backend() -> None:
    from sva_toolkit.formal.service import FormalService

    service = FormalService(registry=_registry())

    result = service.check_implication(
        "req |-> gnt",
        "req |-> ##1 gnt",
        clock="clk",
        clock_edge="posedge",
        reset="!rst_n",
    )

    assert result.result is ImplicationResult.ERROR
    assert "No formal backend is available" in result.message


def test_formal_service_prefers_vcformal_in_auto_mode() -> None:
    from sva_toolkit.formal.service import FormalService

    service = FormalService(registry=_registry(ebmc="/tools/ebmc", vcf="/tools/vcf"))

    assert service._select_backend().__class__.__name__ == "VcformalBackend"


def test_formal_service_reports_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    from sva_toolkit.formal.service import FormalService

    service = FormalService(registry=_registry(ebmc="/tools/ebmc"))
    sequence = iter(
        [
            CheckResult(result=ImplicationResult.IMPLIES, message="forward", log="fwd"),
            CheckResult(result=ImplicationResult.IMPLIES, message="reverse", log="rev"),
        ]
    )
    monkeypatch.setattr(service, "check_implication", lambda *_args, **_kwargs: next(sequence))

    result = service.check_equivalence("a", "b", clock="clk", clock_edge="posedge", reset="!rst_n")

    assert result.result is ImplicationResult.EQUIVALENT
    assert result.log == "forward:\nfwd\n\nreverse:\nrev"
