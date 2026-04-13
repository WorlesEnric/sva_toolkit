"""Semantic validation for the canonical timing scenario model."""

from __future__ import annotations

import re

from sva_toolkit.timing.core.conditions import collect_signals
from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    CutMeaning,
    CutPlacement,
    ExtractionStatus,
    ScenarioDocument,
    SignalKind,
    WindowBoundKind,
)
from sva_toolkit.timing.errors import TimingDslError


def validate_diagram(diagram: ScenarioDocument) -> None:
    """Validate semantic consistency of a parsed or extracted scenario."""

    if diagram.ticks is not None and diagram.ticks <= 0:
        raise TimingDslError("ticks must be greater than zero")

    seen_names = set()
    for collection in (
        diagram.params,
        diagram.signals,
        diagram.anchors,
        diagram.windows,
        diagram.cuts,
        diagram.lane_constraints,
        diagram.properties,
    ):
        for item in collection:
            if item.name in seen_names:
                raise TimingDslError(f"duplicate identifier: {item.name}")
            seen_names.add(item.name)

    signal_map = diagram.signal_map
    anchor_names = set(diagram.anchor_map)
    window_names = set(diagram.window_map)

    sampled_lengths = {len(signal.samples) for signal in diagram.signals if signal.samples}
    if diagram.ticks is None and sampled_lengths:
        raise TimingDslError("ticks are required when any lane declares concrete samples")
    for signal in diagram.signals:
        if signal.kind == SignalKind.BIT and signal.width not in (None, "1"):
            raise TimingDslError(f"bit lane '{signal.name}' cannot declare width {signal.width}; use bus[...] instead")
        if diagram.ticks is not None and signal.samples and len(signal.samples) != diagram.ticks:
            raise TimingDslError(
                f"lane '{signal.name}' has {len(signal.samples)} samples but diagram declares {diagram.ticks} ticks"
            )
        if signal.kind == SignalKind.BIT:
            invalid_samples = [sample for sample in signal.samples if sample.lower() not in {"0", "1", "x", "z"}]
            if invalid_samples:
                raise TimingDslError(
                    f"bit lane '{signal.name}' uses unsupported sample '{invalid_samples[0]}'; "
                    "bit lanes support only 0, 1, x, and z"
                )

    for anchor in diagram.anchors:
        for signal_name in collect_signals(anchor.condition):
            if signal_name not in signal_map:
                raise TimingDslError(f"anchor '{anchor.name}' references unknown lane '{signal_name}'")

    for window in diagram.windows:
        if window.start_anchor not in anchor_names:
            raise TimingDslError(f"window '{window.name}' references unknown start anchor '{window.start_anchor}'")
        if window.end_anchor not in anchor_names:
            raise TimingDslError(f"window '{window.name}' references unknown end anchor '{window.end_anchor}'")
        _validate_bound_token(window.name, window.bound.min_delay, diagram)
        _validate_bound_token(window.name, window.bound.max_delay, diagram, allow_dollar=True)
        if window.bound.kind == WindowBoundKind.RANGE and _both_ints(window.bound.min_delay, window.bound.max_delay):
            if int(window.bound.min_delay) > int(window.bound.max_delay):
                raise TimingDslError(f"window '{window.name}' has min delay greater than max delay")

    for cut in diagram.cuts:
        if cut.placement == CutPlacement.BETWEEN_WINDOWS:
            if cut.left_window not in window_names or cut.right_window not in window_names:
                raise TimingDslError(f"cut '{cut.name}' references unknown windows")
        else:
            if cut.anchor not in anchor_names:
                raise TimingDslError(f"cut '{cut.name}' references unknown anchor '{cut.anchor}'")
        if cut.meaning == CutMeaning.OMITTED_HISTORY and cut.placement == CutPlacement.AFTER_ANCHOR:
            raise TimingDslError(f"cut '{cut.name}' cannot use omitted history after an anchor")

    for constraint in diagram.lane_constraints:
        for signal_name in constraint.signals:
            if signal_name not in signal_map:
                raise TimingDslError(f"constraint '{constraint.name}' references unknown lane '{signal_name}'")
        if constraint.region == ConstraintRegion.AT and constraint.anchor not in anchor_names:
            raise TimingDslError(f"constraint '{constraint.name}' references unknown anchor '{constraint.anchor}'")
        if constraint.region == ConstraintRegion.IN and constraint.window not in window_names:
            raise TimingDslError(f"constraint '{constraint.name}' references unknown window '{constraint.window}'")
        if constraint.region == ConstraintRegion.BEFORE and constraint.anchor not in anchor_names:
            raise TimingDslError(f"constraint '{constraint.name}' references unknown anchor '{constraint.anchor}'")
        if constraint.region == ConstraintRegion.AFTER and constraint.anchor not in anchor_names:
            raise TimingDslError(f"constraint '{constraint.name}' references unknown anchor '{constraint.anchor}'")
        if constraint.region == ConstraintRegion.FROM_UNTIL:
            if constraint.start_anchor not in anchor_names or constraint.end_anchor not in anchor_names:
                raise TimingDslError(f"constraint '{constraint.name}' references unknown anchors")
        if constraint.relation in {"high", "low", "rise", "fall"}:
            for signal_name in constraint.signals:
                if signal_map[signal_name].kind != SignalKind.BIT:
                    raise TimingDslError(
                        f"constraint '{constraint.name}' uses relation '{constraint.relation}' on non-bit lane '{signal_name}'"
                    )
        if constraint.relation in {"eq", "neq"} and constraint.value is None:
            raise TimingDslError(f"constraint '{constraint.name}' uses relation '{constraint.relation}' without a value")

    for prop in diagram.properties:
        if prop.status == ExtractionStatus.UNSUPPORTED and not prop.notes:
            raise TimingDslError(f"property '{prop.name}' is unsupported and must explain why in notes")
        for anchor_name in prop.related_anchors:
            if anchor_name not in anchor_names:
                raise TimingDslError(f"property '{prop.name}' references unknown anchor '{anchor_name}'")
        for window_name in prop.related_windows:
            if window_name not in window_names:
                raise TimingDslError(f"property '{prop.name}' references unknown window '{window_name}'")


def _validate_bound_token(owner: str, token: str | None, diagram: ScenarioDocument, allow_dollar: bool = False) -> None:
    if token is None:
        return
    if allow_dollar and token == "$":
        return
    if token.isdigit():
        return
    param_names = {param.name for param in diagram.params}
    if token in param_names:
        return
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
        raise TimingDslError(f"{owner} references undeclared parameter '{token}'")
    raise TimingDslError(f"{owner} uses invalid bound token '{token}'")


def _both_ints(left: str | None, right: str | None) -> bool:
    return bool(left and right and left.isdigit() and right.isdigit())
