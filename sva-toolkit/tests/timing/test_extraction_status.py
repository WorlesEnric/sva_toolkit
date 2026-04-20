"""Extraction status reporting coverage for the timing SVA bridge."""

from __future__ import annotations

import sva_toolkit.timing.bridge.from_sva as from_sva_module
from sva_toolkit.timing.bridge.ebmc_witness import EbmcWitnessSynthesizer
from sva_toolkit.timing.bridge.from_sva import extract_sva_scenario
from sva_toolkit.timing.bridge.status import ExtractionReport, summarize_report
from sva_toolkit.timing.core.scenario import ExtractionStatus


def test_clean_input_yields_exact_report_without_reasons() -> None:
    document, report = extract_sva_scenario(
        "property p_clean; @(posedge clk) disable iff (!rst_n) req |-> ##1 ack; endproperty"
    )

    assert document.effective_status == ExtractionStatus.EXACT
    assert report.per_property == {"p_clean": ExtractionStatus.EXACT}
    assert report.worst_status() == ExtractionStatus.EXACT
    assert report.reasons == []


def test_unsupported_operator_yields_unsupported_report_with_reason() -> None:
    document, report = extract_sva_scenario(
        "property p_accept; @(posedge clk) disable iff (!rst_n) accept_on(abort) req |-> ack; endproperty"
    )

    assert document.effective_status == ExtractionStatus.UNSUPPORTED
    assert report.per_property["p_accept"] == ExtractionStatus.UNSUPPORTED
    assert report.worst_status() == ExtractionStatus.UNSUPPORTED
    assert report.reasons
    assert any("unsupported control wrapper" in reason for reason in report.reasons)


def test_dag_compilation_exception_is_recorded_on_report(monkeypatch) -> None:
    def raise_value_error(self) -> ExtractionStatus:
        raise ValueError("bad_delay")

    monkeypatch.setattr(from_sva_module.CEGSolver, "solve", raise_value_error)

    _, report = extract_sva_scenario(
        "property p_dag; @(posedge clk) disable iff (!rst_n) req |-> ##1 ack; endproperty"
    )

    assert report.per_property["p_dag"] == ExtractionStatus.UNSUPPORTED
    assert report.worst_status() == ExtractionStatus.UNSUPPORTED
    assert any("ValueError: bad_delay" in reason for reason in report.reasons)


def test_witness_exception_is_recorded_on_report(monkeypatch) -> None:
    def raise_runtime_error(self, formal_prop, *, signal_widths=None):
        raise RuntimeError("witness boom")

    monkeypatch.setattr(EbmcWitnessSynthesizer, "available", property(lambda self: True))
    monkeypatch.setattr(EbmcWitnessSynthesizer, "synthesize", raise_runtime_error)

    _, report = extract_sva_scenario(
        "property p_witness; @(posedge clk) disable iff (!rst_n) req |-> ##1 ack; endproperty"
    )

    assert report.per_property["p_witness"] == ExtractionStatus.LOSSY
    assert report.worst_status() == ExtractionStatus.LOSSY
    assert any("RuntimeError: witness boom" in reason for reason in report.reasons)


def test_summarize_report_renders_deterministic_lines() -> None:
    report = ExtractionReport()
    report.add_property("beta", ExtractionStatus.LOSSY, ["lossy branch"])
    report.add_property("alpha", ExtractionStatus.UNSUPPORTED, ["unsupported op"])

    assert summarize_report(report) == (
        "overall: unsupported\n"
        "property alpha: unsupported\n"
        "property beta: lossy\n"
        "reason: beta: lossy branch\n"
        "reason: alpha: unsupported op"
    )
