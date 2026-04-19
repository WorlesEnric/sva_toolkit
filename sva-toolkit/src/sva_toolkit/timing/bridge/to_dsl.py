"""Scenario to timing DSL v2 emission."""

from __future__ import annotations

from sva_toolkit.timing.core.conditions import condition_to_dsl
from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    CutMeaning,
    CutPlacement,
    ExtractionStatus,
    ScenarioDocument,
    TimeBound,
    WindowBoundKind,
)


def emit_timing_dsl(document: ScenarioDocument) -> str:
    """Emit a scenario document using the v2 timing DSL surface."""

    lines = [f"diagram {document.name} {{", f"  clock {document.clocking.edge} {document.clocking.signal};"]
    if document.clocking.disable_iff:
        lines.append(f"  disable iff {document.clocking.disable_iff};")
    if document.ticks is not None:
        lines.append(f"  ticks {document.ticks};")
    if document.params:
        lines.append("")
        for param in document.params:
            if param.kind == "int":
                lines.append(f"  param {param.name};")
            else:
                lines.append(f"  param {param.name}: {param.kind};")
    if document.signals:
        lines.append("")
        for signal in document.signals:
            suffix = f": {signal.kind.value}"
            if signal.width and not (signal.kind.value == "bit" and signal.width == "1"):
                suffix += f"[{signal.width}]"
            if signal.samples:
                lines.append(f"  lane {signal.name}{suffix} = {' '.join(signal.samples)};")
            else:
                lines.append(f"  lane {signal.name}{suffix};")
    if document.anchors:
        lines.append("")
        for anchor in document.anchors:
            lines.append(f"  anchor {anchor.name} = {condition_to_dsl(anchor.condition)};")
    if document.windows:
        lines.append("")
        for window in document.windows:
            lines.append(
                f"  window {window.name} = between {window.start_anchor} and {window.end_anchor} {_bound_to_text(window.bound)};"
            )
    if document.cuts:
        lines.append("")
        for cut in document.cuts:
            label = f' label "{cut.label}"' if cut.label else ""
            meaning = _cut_meaning_to_text(cut.meaning)
            if cut.placement == CutPlacement.BETWEEN_WINDOWS:
                lines.append(
                    f"  cut {cut.name} = between {cut.left_window} and {cut.right_window} {meaning}{label};"
                )
            else:
                placement = "before" if cut.placement == CutPlacement.BEFORE_ANCHOR else "after"
                lines.append(f"  cut {cut.name} = {placement} {cut.anchor} {meaning}{label};")
    if document.lane_constraints:
        lines.append("")
        for constraint in document.lane_constraints:
            lines.append(f"  {_constraint_to_text(constraint)};")
    if document.properties:
        lines.append("")
        for prop in document.properties:
            status_suffix = "" if prop.status == ExtractionStatus.EXACT else f" [{prop.status.value}]"
            lines.append(f"  property {prop.name}{status_suffix}:")
            lines.append(f"    {prop.body};")
            for note in prop.notes:
                lines.append(f"  // {prop.name}: {note}")
    lines.append("}")
    return "\n".join(lines)


def _bound_to_text(bound: TimeBound) -> str:
    if bound.kind == WindowBoundKind.OMITTED:
        return "[0:$]"
    if bound.kind == WindowBoundKind.EXACT:
        if bound.min_delay == bound.max_delay:
            return f"[{bound.min_delay}:{bound.max_delay}]"
    if bound.kind == WindowBoundKind.RANGE:
        return f"[{bound.min_delay}:{bound.max_delay}]"
    if bound.kind == WindowBoundKind.UNBOUNDED:
        return f"[{bound.min_delay}:$]"
    return bound.label


def _cut_meaning_to_text(meaning: CutMeaning) -> str:
    if meaning == CutMeaning.OMITTED_HISTORY:
        return "omitted"
    if meaning == CutMeaning.OMITTED_FUTURE:
        return "omitted"
    if meaning == CutMeaning.LOOKBACK:
        return "lookback"
    return "compressed"


def _constraint_to_text(constraint) -> str:
    head = f"show {_constraint_head(constraint)}"
    if constraint.region == ConstraintRegion.AT:
        return f"{head} at {constraint.anchor}"
    if constraint.region == ConstraintRegion.IN:
        return f"{head} in {constraint.window}"
    if constraint.region == ConstraintRegion.BEFORE:
        return f"{head} before {constraint.anchor}"
    if constraint.region == ConstraintRegion.AFTER:
        return f"{head} after {constraint.anchor}"
    return f"{head} from {constraint.start_anchor} until {constraint.end_anchor}"


def _constraint_head(constraint) -> str:
    signal = constraint.signals[0] if len(constraint.signals) == 1 else "{" + ", ".join(constraint.signals) + "}"
    if constraint.relation == "eq":
        return f"{signal} = {constraint.value}"
    if constraint.relation == "neq":
        return f"neq({signal}, {constraint.value})"
    if constraint.relation in {"high", "low", "rise", "fall", "change", "stable", "unknown", "dontcare"}:
        if len(constraint.signals) > 1 and constraint.relation == "stable":
            return f"stable({signal})"
        return f"{constraint.relation}({signal})"
    return f"{constraint.relation}({signal})"
