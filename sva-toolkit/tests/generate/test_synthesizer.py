from __future__ import annotations

from typing import Any

import pytest

from sva_toolkit.generate import GenerationRng
from sva_toolkit.runtime.config import ToolConfig
from sva_toolkit.runtime.process import RunResult
from sva_toolkit.runtime.tools import ToolRegistry, create_default_registry


def _make_registry(*, path: str | None, available: bool) -> ToolRegistry:
    registry = ToolRegistry()
    registry._tools["verible-verilog-syntax"] = ToolConfig(  # noqa: SLF001
        name="verible-verilog-syntax",
        path=path,
        available=available,
    )
    return registry


def test_generate_module_returns_requested_property_count_and_coverage() -> None:
    from sva_toolkit.generate import SVASynthesizer, compute_coverage_statistics

    synthesizer = SVASynthesizer(signals=["req", "ack", "gnt"], max_depth=2, rng=GenerationRng(seed=7))

    module_code, properties = synthesizer.generate_module(
        module_name="demo_sva",
        num_assertions=4,
    )
    coverage = compute_coverage_statistics([prop.sva_code for prop in properties])

    assert len(properties) == 4
    assert "module demo_sva" in module_code
    assert "property p_gen_0;" in module_code
    assert coverage["total_properties"] == 4
    assert coverage["constructs_total"] >= coverage["constructs_covered"] >= 1


def test_validate_syntax_uses_runtime_registry_and_process_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sva_toolkit.generate.synthesizer import SVASynthesizer

    calls: dict[str, Any] = {}

    def fake_run_tool(cmd: list[str], **kwargs: Any) -> RunResult:
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return RunResult(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("sva_toolkit.generate.synthesizer.run_tool", fake_run_tool)

    synthesizer = SVASynthesizer(
        signals=["req", "ack"],
        tool_registry=_make_registry(path="/tmp/fake-verible", available=True),
    )

    result = synthesizer.validate_syntax("module demo; endmodule")

    assert result.is_valid is True
    assert calls["cmd"] == ["/tmp/fake-verible", "-"]
    assert calls["kwargs"]["input_text"] == "module demo; endmodule"


def test_validate_syntax_reports_missing_verible_tool() -> None:
    from sva_toolkit.generate.synthesizer import SVASynthesizer

    synthesizer = SVASynthesizer(
        signals=["req", "ack"],
        tool_registry=_make_registry(path=None, available=False),
    )

    result = synthesizer.validate_syntax("module demo; endmodule")

    assert result.is_valid is False
    assert "verible-verilog-syntax" in result.error_message


def test_generated_module_is_verible_valid_when_tool_is_available() -> None:
    from sva_toolkit.generate.synthesizer import SVASynthesizer

    registry = create_default_registry()
    if not registry.get("verible-verilog-syntax").available:
        pytest.skip("verible-verilog-syntax is not installed")

    synthesizer = SVASynthesizer(
        signals=["req", "ack", "gnt", "valid", "ready"],
        max_depth=2,
        rng=GenerationRng(seed=11),
        tool_registry=registry,
    )
    module_code, _properties = synthesizer.generate_module(
        module_name="verible_check",
        num_assertions=3,
    )

    result = synthesizer.validate_syntax(module_code)

    assert result.is_valid is True, result.error_message
