"""Shared tick assignment and sample synthesis helpers for timing projections."""

from __future__ import annotations

from typing import Mapping, Sequence

from sva_toolkit.timing.core.conditions import Condition, Predicate
from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    CutMeaning,
    CutPlacement,
    ScenarioDocument,
    SignalKind,
)


Placement = str

_SUPPORTED_PREDICATES = {"high", "low", "rise", "fall", "change", "stable", "eq", "neq"}
_BOUNDARY_PREDICATES = {"rise", "fall", "change"}


def canonical_tick_assignment(document: ScenarioDocument) -> dict[str, int]:
    """Assign absolute ticks to anchors via topological longest-path over the window graph.

    Uses earliest-legal-tick policy: roots get the base offset, successors get
    predecessor_tick + min_delay (or max_delay when finite). Raises ValueError
    on unresolvable conflicts.
    """

    all_anchors = {anchor.name for anchor in document.anchors}
    offset = 0
    for cut in document.cuts:
        if cut.placement == CutPlacement.BEFORE_ANCHOR and cut.meaning == CutMeaning.OMITTED_HISTORY:
            offset = 1
            break

    successors: dict[str, list[tuple[str, int]]] = {name: [] for name in all_anchors}
    in_degree: dict[str, int] = {name: 0 for name in all_anchors}

    for window in document.windows:
        if window.start_anchor not in all_anchors or window.end_anchor not in all_anchors:
            continue
        delay_str = window.bound.max_delay if window.bound.max_delay and window.bound.max_delay != "$" else window.bound.min_delay
        try:
            delay = int(delay_str) if delay_str else 0
        except ValueError:
            delay = 1
        successors[window.start_anchor].append((window.end_anchor, delay))
        in_degree[window.end_anchor] = in_degree.get(window.end_anchor, 0) + 1

    ticks: dict[str, int] = {}
    queue = sorted(name for name in all_anchors if in_degree.get(name, 0) == 0)
    for root in queue:
        ticks[root] = offset

    pending = list(queue)
    while pending:
        pending.sort()
        node = pending.pop(0)
        for target, delay in successors.get(node, []):
            proposed = ticks[node] + delay
            if target in ticks:
                ticks[target] = max(ticks[target], proposed)
            else:
                ticks[target] = proposed
            in_degree[target] -= 1
            if in_degree[target] == 0:
                pending.append(target)

    for name in all_anchors:
        if name not in ticks:
            ticks[name] = offset

    return ticks


def has_concrete_samples(document: ScenarioDocument) -> bool:
    """Return whether all declared signals already carry concrete samples."""

    return document.ticks is not None and all(signal.samples for signal in document.signals)


def resolve_total_ticks(document: ScenarioDocument, anchor_ticks: Mapping[str, int] | None = None) -> int:
    """Resolve the sampled timeline length using the existing WaveDrom policy."""

    if document.ticks is not None:
        return document.ticks
    anchor_ticks = anchor_ticks or resolve_anchor_ticks(document, prefer_samples=False)
    if anchor_ticks:
        return max(anchor_ticks.values()) + 3
    return 8


def resolve_anchor_ticks(
    document: ScenarioDocument,
    *,
    samples: Mapping[str, Sequence[str]] | None = None,
    prefer_samples: bool = True,
) -> dict[str, int]:
    """Resolve one display tick per anchor.

    Explicit ``absolute_tick`` values win. For concrete sampled documents the
    first tick that satisfies the anchor condition is used. Otherwise this falls
    back to :func:`canonical_tick_assignment`, matching the historical
    WaveDrom-backed projection for symbolic documents.
    """

    solved = canonical_tick_assignment(document)
    signal_samples = samples
    if signal_samples is None and has_concrete_samples(document):
        signal_samples = {signal.name: signal.samples for signal in document.signals}

    total_ticks = document.ticks
    if total_ticks is None and signal_samples:
        total_ticks = max((len(values) for values in signal_samples.values()), default=0)

    resolved: dict[str, int] = {}
    for anchor in document.anchors:
        if anchor.absolute_tick is not None:
            resolved[anchor.name] = anchor.absolute_tick
            continue
        sample_tick: int | None = None
        if prefer_samples and signal_samples and total_ticks is not None:
            for tick in range(total_ticks):
                try:
                    if evaluate_condition(anchor.condition, signal_samples, tick):
                        sample_tick = tick
                        break
                except (KeyError, ValueError, IndexError):
                    break
        resolved[anchor.name] = sample_tick if sample_tick is not None else solved.get(anchor.name, 0)
    return resolved


def synthesize_signal_samples(
    document: ScenarioDocument,
    *,
    anchor_ticks: Mapping[str, int] | None = None,
    ticks: int | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return per-signal samples, synthesizing symbolic lanes when needed."""

    anchor_ticks = dict(anchor_ticks or resolve_anchor_ticks(document, prefer_samples=False))
    ticks = ticks if ticks is not None else resolve_total_ticks(document, anchor_ticks)
    signal_kinds = {signal.name: signal.kind for signal in document.signals}

    if has_concrete_samples(document):
        return {
            signal.name: _fit_samples(signal.samples, ticks, _default_value(signal.kind))
            for signal in document.signals
        }

    signals_mut: dict[str, list[str]] = {}
    for signal in document.signals:
        default = _default_value(signal.kind)
        signals_mut[signal.name] = list(_fit_samples(signal.samples, ticks, default)) if signal.samples else [default] * ticks

    for anchor in document.anchors:
        tick = anchor_ticks.get(anchor.name)
        if tick is not None and 0 <= tick < ticks:
            apply_anchor_condition(anchor.condition, signals_mut, tick, signal_kinds)

    infer_window_spans(document, signals_mut, anchor_ticks, signal_kinds, ticks)
    apply_span_semantics(document, signals_mut, anchor_ticks, signal_kinds)

    return {name: tuple(samples) for name, samples in signals_mut.items()}


def apply_anchor_condition(
    cond: Condition,
    signals: dict[str, list[str]],
    tick: int,
    signal_kinds: dict[str, SignalKind] | None = None,
) -> None:
    """Apply point-event anchor conditions at the given tick."""

    if cond.kind == "predicate" and cond.predicate is not None:
        predicate = cond.predicate
        if not predicate.signal or predicate.signal not in signals:
            return
        samples = signals[predicate.signal]
        if tick < 0 or tick >= len(samples):
            return
        if predicate.op == "high":
            samples[tick] = "1"
        elif predicate.op == "low":
            samples[tick] = "0"
        elif predicate.op == "rise":
            if tick > 0:
                samples[tick - 1] = "0"
            samples[tick] = "1"
            if tick + 1 < len(samples):
                samples[tick + 1] = "1"
        elif predicate.op == "fall":
            if tick > 0:
                samples[tick - 1] = "1"
            samples[tick] = "0"
            if tick + 1 < len(samples):
                samples[tick + 1] = "0"
        elif predicate.op == "eq":
            samples[tick] = predicate.value or "1"
        elif predicate.op == "stable":
            is_bus = (signal_kinds or {}).get(predicate.signal, SignalKind.BIT) == SignalKind.BUS
            if is_bus and samples[tick] == "x":
                samples[tick] = predicate.signal
    elif cond.kind == "all":
        for item in cond.items:
            apply_anchor_condition(item, signals, tick, signal_kinds)


def collect_condition_predicates(cond: Condition) -> list[Predicate]:
    """Collect all leaf predicates from a condition tree."""

    if cond.kind == "predicate" and cond.predicate is not None:
        return [cond.predicate]
    if cond.kind in ("all", "any", "not") and cond.items:
        return [predicate for item in cond.items for predicate in collect_condition_predicates(item)]
    return []


def infer_window_spans(
    document: ScenarioDocument,
    signals: dict[str, list[str]],
    anchor_ticks: Mapping[str, int],
    signal_kinds: Mapping[str, SignalKind],
    ticks: int,
) -> None:
    """Infer span-level fills from window endpoints when lane constraints are absent."""

    if document.lane_constraints:
        return

    anchor_map = {anchor.name: anchor for anchor in document.anchors}

    for window in document.windows:
        start_tick = anchor_ticks.get(window.start_anchor)
        end_tick = anchor_ticks.get(window.end_anchor)
        if start_tick is None or end_tick is None:
            continue

        lo = max(0, min(start_tick, end_tick))
        hi = min(max(start_tick, end_tick), ticks - 1)

        for anchor_name in (window.start_anchor, window.end_anchor):
            anchor = anchor_map.get(anchor_name)
            if anchor is None:
                continue
            for predicate in collect_condition_predicates(anchor.condition):
                if not predicate.signal or predicate.signal not in signals:
                    continue
                is_bus = signal_kinds.get(predicate.signal, SignalKind.BIT) == SignalKind.BUS
                samples = signals[predicate.signal]

                fill_value: str | None = None
                if predicate.op == "stable" and is_bus:
                    fill_value = predicate.signal
                elif predicate.op == "eq":
                    fill_value = predicate.value or predicate.signal
                elif predicate.op == "high":
                    fill_value = "1"
                elif predicate.op == "low":
                    fill_value = "0"

                if fill_value is None:
                    continue

                for tick in range(lo, hi + 1):
                    current = samples[tick]
                    if current in ("x", "0", "") or current == fill_value:
                        samples[tick] = fill_value


def apply_span_semantics(
    document: ScenarioDocument,
    signals: dict[str, list[str]],
    anchor_ticks: Mapping[str, int],
    signal_kinds: Mapping[str, SignalKind],
) -> None:
    """Enforce span-level semantics from lane constraints on signal samples."""

    for constraint in document.lane_constraints:
        start_tick: int | None = None
        end_tick: int | None = None

        if constraint.region == ConstraintRegion.FROM_UNTIL:
            start_tick = anchor_ticks.get(constraint.start_anchor) if constraint.start_anchor else None
            end_tick = anchor_ticks.get(constraint.end_anchor) if constraint.end_anchor else None
        elif constraint.region == ConstraintRegion.BEFORE:
            start_tick = 0
            end_tick = (anchor_ticks[constraint.anchor] - 1) if constraint.anchor and constraint.anchor in anchor_ticks else None
        elif constraint.region == ConstraintRegion.AT:
            at_tick = anchor_ticks.get(constraint.anchor) if constraint.anchor else None
            if at_tick is not None:
                start_tick = at_tick
                end_tick = at_tick

        if start_tick is None or end_tick is None:
            continue

        for signal_name in constraint.signals:
            if signal_name not in signals:
                continue
            samples = signals[signal_name]
            is_bus = signal_kinds.get(signal_name, SignalKind.BIT) == SignalKind.BUS

            for tick in range(max(0, start_tick), min(end_tick + 1, len(samples))):
                target_value: str | None = None
                if constraint.relation == "high":
                    target_value = "1"
                elif constraint.relation == "low":
                    target_value = "0"
                elif constraint.relation == "stable":
                    target_value = signal_name if is_bus else "1"
                elif constraint.relation == "eq":
                    target_value = constraint.value or ("x" if is_bus else "1")
                elif constraint.relation in ("rise", "fall", "change", "neq"):
                    continue

                if target_value is not None:
                    current = samples[tick]
                    if current not in ("0", "x", "") and current != target_value:
                        continue
                    samples[tick] = target_value


def evaluate_condition(condition: Condition, signals: Mapping[str, Sequence[str]], tick: int) -> bool:
    """Evaluate a supported condition tree against concrete samples."""

    if condition.kind == "predicate" and condition.predicate is not None:
        predicate = condition.predicate
        if predicate.signal is None:
            raise ValueError("predicate-backed conditions require a concrete signal")

        samples = signals[predicate.signal]
        current = samples[tick]
        previous = samples[tick - 1] if tick > 0 else None

        if predicate.op == "high":
            return current == "1"
        if predicate.op == "low":
            return current == "0"
        if predicate.op == "rise":
            return previous == "0" and current == "1"
        if predicate.op == "fall":
            return previous == "1" and current == "0"
        if predicate.op == "change":
            return previous is not None and current != previous
        if predicate.op == "stable":
            return previous is not None and current == previous
        if predicate.op == "eq":
            return current == predicate.value
        if predicate.op == "neq":
            return current != predicate.value
        raise ValueError(f"unsupported predicate op: {predicate.op}")

    if condition.kind == "all":
        return all(evaluate_condition(item, signals, tick) for item in condition.items)
    if condition.kind == "any":
        return any(evaluate_condition(item, signals, tick) for item in condition.items)
    if condition.kind == "not" and condition.items:
        return not evaluate_condition(condition.items[0], signals, tick)
    if condition.kind == "raw":
        text = (condition.text or "").strip().lower()
        if text in ("0", "1'b0", "false"):
            return False
        return True
    raise ValueError(f"unsupported condition kind: {condition.kind}")


def anchor_placement(condition: Condition) -> Placement:
    """Determine whether an anchor should snap to a boundary or sample center."""

    if all_predicates_in(condition, _BOUNDARY_PREDICATES):
        return "boundary"
    return "center"


def all_predicates_in(condition: Condition, allowed_ops: set[str]) -> bool:
    """Return whether every predicate in the tree belongs to the allowed set."""

    if condition.kind == "predicate" and condition.predicate is not None:
        return condition.predicate.op in allowed_ops
    if condition.kind in {"all", "any", "not"}:
        return bool(condition.items) and all(all_predicates_in(item, allowed_ops) for item in condition.items)
    return False


def _fit_samples(samples: Sequence[str], ticks: int, default: str) -> tuple[str, ...]:
    fitted = tuple(samples[:ticks])
    if len(fitted) >= ticks:
        return fitted
    return (*fitted, *((default,) * (ticks - len(fitted))))


def _default_value(kind: SignalKind) -> str:
    return "0" if kind == SignalKind.BIT else "x"


_canonical_tick_assignment = canonical_tick_assignment
_apply_anchor_condition = apply_anchor_condition
_collect_condition_predicates = collect_condition_predicates
_infer_window_spans = infer_window_spans
_apply_span_semantics = apply_span_semantics
