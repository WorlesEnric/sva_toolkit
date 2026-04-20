"""Extraction status reporting helpers for timing reverse-extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from sva_toolkit.runtime.diagnostics import DEFAULT_DIAGNOSTICS, LOGGER
from sva_toolkit.timing.core.scenario import ExtractionStatus, ScenarioDocument


def _merge_status(current: ExtractionStatus, incoming: ExtractionStatus) -> ExtractionStatus:
    if ExtractionStatus.UNSUPPORTED in (current, incoming):
        return ExtractionStatus.UNSUPPORTED
    if ExtractionStatus.LOSSY in (current, incoming):
        return ExtractionStatus.LOSSY
    return ExtractionStatus.EXACT


def _qualify_reason(property_name: str, reason: str) -> str:
    if reason.startswith(f"{property_name}: "):
        return reason
    return f"{property_name}: {reason}"


def _format_exception_reason(stage: str, exc: BaseException) -> str:
    detail = type(exc).__name__
    if str(exc):
        return f"{stage} failed with {detail}: {exc}"
    return f"{stage} failed with {detail}"


@dataclass
class ExtractionReport:
    """Aggregated extraction outcome for one or more properties."""

    per_property: dict[str, ExtractionStatus] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def worst_status(self) -> ExtractionStatus:
        status = ExtractionStatus.EXACT
        for item_status in self.per_property.values():
            status = _merge_status(status, item_status)
        return status

    def add_property(
        self,
        property_name: str,
        status: ExtractionStatus,
        reasons: Iterable[str] = (),
        *,
        record_diagnostic: bool = False,
    ) -> None:
        previous = self.per_property.get(property_name, ExtractionStatus.EXACT)
        merged = _merge_status(previous, status)
        self.per_property[property_name] = merged
        if record_diagnostic and previous == ExtractionStatus.EXACT and merged != ExtractionStatus.EXACT:
            DEFAULT_DIAGNOSTICS.record("lossy_extraction", detail=f"{property_name}:{merged.value}")
        for reason in reasons:
            qualified = _qualify_reason(property_name, reason)
            if qualified not in self.reasons:
                self.reasons.append(qualified)

    def record_exception(
        self,
        property_name: str,
        *,
        stage: str,
        exc: BaseException,
        status: ExtractionStatus,
    ) -> str:
        reason = _format_exception_reason(stage, exc)
        self.add_property(property_name, status, (reason,), record_diagnostic=True)
        LOGGER.warning("timing extraction %s for %s: %s", stage, property_name, reason)
        return reason

    def merge(self, other: "ExtractionReport") -> None:
        for property_name, status in other.per_property.items():
            self.add_property(property_name, status)
        for reason in other.reasons:
            if reason not in self.reasons:
                self.reasons.append(reason)

    @classmethod
    def from_documents(cls, documents: Iterable[ScenarioDocument]) -> "ExtractionReport":
        report = cls()
        for document in documents:
            if document.properties:
                for prop in document.properties:
                    report.add_property(prop.name, prop.status, prop.notes)
            else:
                report.add_property(document.name, document.effective_status, document.notes)
        return report


class LossyExtractionError(RuntimeError):
    """Raised when callers require exact extraction but receive a degraded report."""

    def __init__(self, report: ExtractionReport) -> None:
        self.report = report
        super().__init__(summarize_report(report))


def merge_extraction_reports(*reports: ExtractionReport) -> ExtractionReport:
    merged = ExtractionReport()
    for report in reports:
        merged.merge(report)
    return merged


def summarize_report(report: ExtractionReport) -> str:
    lines = [f"overall: {report.worst_status().value}"]
    for property_name in sorted(report.per_property):
        lines.append(f"property {property_name}: {report.per_property[property_name].value}")
    for reason in report.reasons:
        lines.append(f"reason: {reason}")
    return "\n".join(lines)


__all__ = [
    "ExtractionReport",
    "LossyExtractionError",
    "merge_extraction_reports",
    "summarize_report",
]
