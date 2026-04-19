"""Parser for the dual-direction timing diagram DSL."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import List, Sequence

from sva_toolkit.timing.core.conditions import Condition, Predicate, parse_dsl_condition
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    ClockingSpec,
    ConstraintRegion,
    Cut,
    CutMeaning,
    CutPlacement,
    ExtractionStatus,
    LaneConstraint,
    ParameterDecl,
    PropertyOverlay,
    ScenarioDocument,
    SignalDecl,
    SignalKind,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
    normalize_signal_width,
)
from sva_toolkit.timing.errors import TimingDslError
from sva_toolkit.timing.frontend.validate import validate_diagram


IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_DIAGRAM_RE = re.compile(rf"^diagram\s+({IDENT})\s*\{{$")
_CLOCK_RE = re.compile(rf"^clock\s+(posedge|negedge)\s+({IDENT})\s*;$")
_DISABLE_RE = re.compile(r"^disable\s+iff\s+(.+)\s*;$")
_TICKS_RE = re.compile(r"^ticks\s+([0-9]+)\s*;$")
_PARAM_RE = re.compile(rf"^param\s+({IDENT})(?:\s*:\s*({IDENT}))?\s*;$")
_LANE_SAMPLES_RE = re.compile(
    rf"^lane\s+({IDENT})\s*:\s*(bit|bus)(?:\[\s*({IDENT}|[0-9]+)\s*\])?\s*=\s*(.+)\s*;$"
)
_LANE_DECL_RE = re.compile(rf"^lane\s+({IDENT})\s*:\s*(bit|bus)(?:\[\s*({IDENT}|[0-9]+)\s*\])?\s*;$")
_ANCHOR_RE = re.compile(rf"^(anchor|event)\s+({IDENT})\s*=\s*(.+)\s*;$")
_WINDOW_RE = re.compile(rf"^window\s+({IDENT})\s*=\s*between\s+({IDENT})\s+and\s+({IDENT})\s+(.+)\s*;$")
_CUT_ANCHOR_RE = re.compile(
    rf"^cut\s+({IDENT})\s*=\s*(before|after)\s+({IDENT})\s+(omitted|compressed|lookback)(?:\s+label\s+(.+))?\s*;$"
)
_CUT_WINDOW_RE = re.compile(
    rf"^cut\s+({IDENT})\s*=\s*between\s+({IDENT})\s+and\s+({IDENT})\s+(omitted|compressed|lookback)(?:\s+label\s+(.+))?\s*;$"
)
_SHOW_RE = re.compile(r"^show\s+(.+)\s*;$")
_PROPERTY_HEADER_RE = re.compile(rf"^property\s+({IDENT})(?:\s+\[(exact|lossy|unsupported)\])?\s*:\s*(.*)$")
_PROPERTY_NOTE_RE = re.compile(rf"^//\s*({IDENT})\s*:\s*(.+)$")
_RULE_HEADER_RE = re.compile(rf"^rule\s+({IDENT})\s*:\s*(.*)$")
_NOT_BEFORE_RE = re.compile(rf"^not\s+({IDENT})\s+before\s+({IDENT})$")
_RESPONSE_RE = re.compile(
    rf"^({IDENT})\s*->\s*after\s*\[\s*({IDENT}|[0-9]+)\s*:\s*({IDENT}|[0-9]+)\s*\]\s*({IDENT})$"
)
_HOLD_UNTIL_RE = re.compile(rf"^(.+)\s+from\s+({IDENT})\s+until\s+({IDENT})$")
_SIGNAL_ASSIGN_RE = re.compile(rf"^({IDENT})\s*=\s*(.+)$")


def parse_diagram(text: str) -> ScenarioDocument:
    """Parse timing DSL text into the canonical scenario model."""

    lines = _preprocess_lines(text)
    while lines and lines[0].startswith("//"):
        lines.pop(0)
    if not lines:
        raise TimingDslError("diagram text is empty")

    header_match = _DIAGRAM_RE.match(lines[0])
    if not header_match:
        raise TimingDslError("diagram must start with 'diagram <name> {'")

    name = header_match.group(1)
    ticks = None
    clocking = None
    params: List[ParameterDecl] = []
    signals: List[SignalDecl] = []
    anchors: List[Anchor] = []
    windows: List[TimeWindow] = []
    cuts: List[Cut] = []
    lane_constraints: List[LaneConstraint] = []
    properties: List[PropertyOverlay] = []

    index = 1
    while index < len(lines):
        line = lines[index]
        if line == "}":
            break

        if note_match := _PROPERTY_NOTE_RE.match(line):
            property_name, note = note_match.groups()
            for property_index in range(len(properties) - 1, -1, -1):
                if properties[property_index].name != property_name:
                    continue
                prop = properties[property_index]
                properties[property_index] = replace(prop, notes=prop.notes + (note.strip(),))
                break
            index += 1
            continue
        if line.startswith("//"):
            index += 1
            continue

        if clock_match := _CLOCK_RE.match(line):
            clocking = ClockingSpec(edge=clock_match.group(1), signal=clock_match.group(2))
            index += 1
            continue

        if disable_match := _DISABLE_RE.match(line):
            if clocking is None:
                raise TimingDslError("disable iff must appear after clock")
            clocking = replace(clocking, disable_iff=disable_match.group(1).strip())
            index += 1
            continue

        if ticks_match := _TICKS_RE.match(line):
            ticks = int(ticks_match.group(1))
            index += 1
            continue

        if line.startswith("params"):
            if line not in {"params {", "params{"}:
                raise TimingDslError("params block must use 'params {'")
            index += 1
            while index < len(lines) and lines[index] != "}":
                param_match = _PARAM_RE.match(lines[index]) or re.match(rf"^({IDENT})\s*;$", lines[index])
                if not param_match:
                    raise TimingDslError(f"invalid param declaration: {lines[index]}")
                param_name = param_match.group(1)
                param_kind = param_match.group(2) or "int"
                params.append(ParameterDecl(name=param_name, kind=param_kind))
                index += 1
            if index >= len(lines):
                raise TimingDslError("unterminated params block")
            index += 1
            continue

        if param_match := _PARAM_RE.match(line):
            param_name = param_match.group(1)
            param_kind = param_match.group(2) or "int"
            params.append(ParameterDecl(name=param_name, kind=param_kind))
            index += 1
            continue

        if lane_match := _LANE_SAMPLES_RE.match(line):
            signal_name, signal_kind, width, sample_text = lane_match.groups()
            normalized_kind = SignalKind(signal_kind)
            normalized_width = normalize_signal_width(normalized_kind, width)
            samples = tuple(sample_text.split())
            signals.append(
                SignalDecl(
                    name=signal_name,
                    kind=normalized_kind,
                    width=normalized_width,
                    samples=samples,
                )
            )
            index += 1
            continue

        if lane_match := _LANE_DECL_RE.match(line):
            signal_name, signal_kind, width = lane_match.groups()
            normalized_kind = SignalKind(signal_kind)
            signals.append(
                SignalDecl(
                    name=signal_name,
                    kind=normalized_kind,
                    width=normalize_signal_width(normalized_kind, width),
                )
            )
            index += 1
            continue

        if anchor_match := _ANCHOR_RE.match(line):
            keyword, anchor_name, expr_text = anchor_match.groups()
            expr_text = expr_text.removesuffix(" same_cycle").strip()
            condition = parse_dsl_condition(expr_text)
            anchors.append(
                Anchor(
                    name=anchor_name,
                    condition=condition,
                    role=AnchorRole.SYNTHETIC if keyword == "event" else AnchorRole.STATE,
                )
            )
            index += 1
            continue

        if window_match := _WINDOW_RE.match(line):
            window_name, start_anchor, end_anchor, bound_text = window_match.groups()
            windows.append(
                TimeWindow(
                    name=window_name,
                    start_anchor=start_anchor,
                    end_anchor=end_anchor,
                    bound=_parse_window_bound(bound_text.strip()),
                )
            )
            index += 1
            continue

        if cut_match := _CUT_ANCHOR_RE.match(line):
            cut_name, placement, anchor_name, meaning_text, label = cut_match.groups()
            cuts.append(
                Cut(
                    name=cut_name,
                    placement=CutPlacement.BEFORE_ANCHOR if placement == "before" else CutPlacement.AFTER_ANCHOR,
                    meaning=_parse_cut_meaning(meaning_text, placement=placement),
                    anchor=anchor_name,
                    label=_strip_optional_quotes(label),
                )
            )
            index += 1
            continue

        if cut_match := _CUT_WINDOW_RE.match(line):
            cut_name, left_window, right_window, meaning_text, label = cut_match.groups()
            cuts.append(
                Cut(
                    name=cut_name,
                    placement=CutPlacement.BETWEEN_WINDOWS,
                    meaning=_parse_cut_meaning(meaning_text, placement="between"),
                    left_window=left_window,
                    right_window=right_window,
                    label=_strip_optional_quotes(label),
                )
            )
            index += 1
            continue

        if show_match := _SHOW_RE.match(line):
            lane_constraints.extend(_parse_show_clause(show_match.group(1).strip(), len(lane_constraints)))
            index += 1
            continue

        if property_match := _PROPERTY_HEADER_RE.match(line):
            property_name, status_text, inline_body = property_match.groups()
            property_body, next_index = _consume_statement_body(lines, index, inline_body)
            properties.append(
                PropertyOverlay(
                    name=property_name,
                    body=property_body,
                    status=_parse_status(status_text),
                )
            )
            index = next_index
            continue

        if rule_match := _RULE_HEADER_RE.match(line):
            rule_name, inline_body = rule_match.groups()
            rule_body, next_index = _consume_statement_body(lines, index, inline_body)
            rule_windows, rule_constraints, property_overlay = _parse_legacy_rule(rule_name, rule_body)
            windows.extend(rule_windows)
            lane_constraints.extend(rule_constraints)
            properties.append(property_overlay)
            index = next_index
            continue

        raise TimingDslError(f"unrecognized statement: {line}")

    if index >= len(lines) or lines[index] != "}":
        raise TimingDslError("diagram is missing closing '}'")
    if clocking is None:
        raise TimingDslError("diagram is missing clock declaration")

    scenario = ScenarioDocument(
        name=name,
        clocking=clocking,
        params=tuple(params),
        signals=tuple(signals),
        anchors=tuple(anchors),
        windows=tuple(windows),
        cuts=tuple(cuts),
        lane_constraints=tuple(lane_constraints),
        properties=tuple(_link_property_references(properties, anchors, windows, lane_constraints)),
        ticks=ticks,
    )
    validate_diagram(scenario)
    return scenario


def _preprocess_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("//"):
            lines.append(stripped)
            continue
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        lines.append(line)
    return lines


def _consume_statement_body(lines: Sequence[str], start_index: int, inline_body: str) -> tuple[str, int]:
    body_parts: List[str] = []
    current_index = start_index

    if inline_body.strip():
        body_parts.append(inline_body.strip())
    else:
        current_index += 1

    while current_index < len(lines):
        if not body_parts:
            body_parts.append(lines[current_index].strip())
        if body_parts[-1].endswith(";"):
            body_text = " ".join(body_parts).strip()
            return body_text[:-1].strip(), current_index + 1
        current_index += 1
        if current_index >= len(lines):
            break
        body_parts.append(lines[current_index].strip())

    raise TimingDslError(f"statement body must end with ';' near line: {lines[start_index]}")


def _parse_window_bound(text: str) -> TimeBound:
    if re.fullmatch(rf"{IDENT}|[0-9]+", text):
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=text, max_delay=text)
    match = re.match(rf"^\[\s*({IDENT}|[0-9]+)\s*:\s*({IDENT}|[0-9]+|\$)\s*\]$", text)
    if not match:
        raise TimingDslError(f"invalid window bound: {text}")
    min_delay, max_delay = match.groups()
    if max_delay == "$":
        return TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay=min_delay, max_delay=max_delay)
    if min_delay == max_delay:
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=min_delay, max_delay=max_delay)
    return TimeBound(kind=WindowBoundKind.RANGE, min_delay=min_delay, max_delay=max_delay)


def _parse_cut_meaning(text: str, *, placement: str) -> CutMeaning:
    if text == "omitted":
        return CutMeaning.OMITTED_HISTORY if placement == "before" else CutMeaning.OMITTED_FUTURE
    if text == "compressed":
        return CutMeaning.SYMBOLIC_GAP
    return CutMeaning.LOOKBACK


def _parse_show_clause(body: str, sequence: int) -> List[LaneConstraint]:
    region_patterns = [
        (ConstraintRegion.FROM_UNTIL, re.compile(rf"^(.+?)\s+from\s+({IDENT})\s+until\s+({IDENT})$")),
        (ConstraintRegion.BEFORE, re.compile(rf"^(.+?)\s+before\s+({IDENT})$")),
        (ConstraintRegion.AFTER, re.compile(rf"^(.+?)\s+after\s+({IDENT})$")),
        (ConstraintRegion.IN, re.compile(rf"^(.+?)\s+in\s+({IDENT})$")),
        (ConstraintRegion.AT, re.compile(rf"^(.+?)\s+at\s+({IDENT})$")),
    ]
    for region, pattern in region_patterns:
        match = pattern.match(body)
        if not match:
            continue
        expr_text = match.group(1).strip()
        if region == ConstraintRegion.FROM_UNTIL:
            target = (match.group(2), match.group(3))
        else:
            target = match.group(2)
        return _build_constraints_from_show_expr(expr_text, region, target, sequence)
    raise TimingDslError(f"invalid show clause: {body}")


def _build_constraints_from_show_expr(
    expr_text: str, region: ConstraintRegion, target, sequence: int
) -> List[LaneConstraint]:
    constraints: List[LaneConstraint] = []

    assign_match = _SIGNAL_ASSIGN_RE.match(expr_text)
    if assign_match:
        signal_name, value = assign_match.groups()
        signals = (signal_name,)
        relation = "eq"
        value = value.strip()
    else:
        condition = parse_dsl_condition(expr_text)
        predicates = _flatten_condition_predicates(condition)
        if not predicates:
            raise TimingDslError(f"show clause must use typed predicates: {expr_text}")
        for offset, predicate in enumerate(predicates):
            constraints.append(_constraint_from_predicate(predicate, region, target, f"show_{sequence}_{offset}"))
        return constraints

    kwargs = _region_kwargs(region, target)
    return [
        LaneConstraint(
            name=f"show_{sequence}_0",
            signals=signals,
            relation=relation,
            value=value,
            region=region,
            **kwargs,
        )
    ]


def _constraint_from_predicate(
    predicate: Predicate, region: ConstraintRegion, target, name: str
) -> LaneConstraint:
    if predicate.signal is None:
        raise TimingDslError("predicate-backed constraints require a concrete signal")
    return LaneConstraint(
        name=name,
        signals=(predicate.signal,),
        relation=predicate.op,
        value=predicate.value,
        region=region,
        **_region_kwargs(region, target),
    )


def _region_kwargs(region: ConstraintRegion, target):
    if region == ConstraintRegion.FROM_UNTIL:
        return {"start_anchor": target[0], "end_anchor": target[1]}
    if region in {ConstraintRegion.BEFORE, ConstraintRegion.AFTER, ConstraintRegion.AT}:
        return {"anchor": target}
    if region == ConstraintRegion.IN:
        return {"window": target}
    return {}


def _parse_legacy_rule(
    rule_name: str,
    body_text: str,
) -> tuple[list[TimeWindow], list[LaneConstraint], PropertyOverlay]:
    if match := _NOT_BEFORE_RE.match(body_text):
        property_overlay = PropertyOverlay(
            name=rule_name,
            body=f"!{match.group(1)} until {match.group(2)}",
            related_anchors=(match.group(1), match.group(2)),
        )
        return ([], [], property_overlay)

    if match := _RESPONSE_RE.match(body_text):
        trigger_event, min_delay, max_delay, response_event = match.groups()
        window_name = f"{rule_name}__window"
        property_overlay = PropertyOverlay(
            name=rule_name,
            body=f"{trigger_event} |-> ##[{min_delay}:{max_delay}] {response_event}",
            related_anchors=(trigger_event, response_event),
            related_windows=(window_name,),
        )
        return (
            [
                TimeWindow(
                    name=window_name,
                    start_anchor=trigger_event,
                    end_anchor=response_event,
                    bound=TimeBound(
                        kind=WindowBoundKind.EXACT if min_delay == max_delay else WindowBoundKind.RANGE,
                        min_delay=min_delay,
                        max_delay=max_delay,
                    ),
                )
            ],
            [],
            property_overlay,
        )

    if match := _HOLD_UNTIL_RE.match(body_text):
        predicate_expr = _parse_event_expr(match.group(1))
        start_event = match.group(2)
        end_event = match.group(3)
        window_name = f"{rule_name}__window"
        constraints = [
            LaneConstraint(
                name=f"{rule_name}__constraint_{index}",
                signals=(predicate.signal,),
                relation=predicate.op,
                value=predicate.value,
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor=start_event,
                end_anchor=end_event,
            )
            for index, predicate in enumerate(predicate_expr)
            if predicate.signal is not None
        ]
        property_overlay = PropertyOverlay(
            name=rule_name,
            body=f"{start_event} |-> {_event_expr_to_sva(predicate_expr)} until_with {end_event}",
            related_anchors=(start_event, end_event),
            related_windows=(window_name,),
            related_constraints=tuple(constraint.name for constraint in constraints),
        )
        return (
            [
                TimeWindow(
                    name=window_name,
                    start_anchor=start_event,
                    end_anchor=end_event,
                    bound=TimeBound(kind=WindowBoundKind.OMITTED),
                )
            ],
            constraints,
            property_overlay,
        )

    raise TimingDslError(f"unsupported rule body: {body_text}")


def _condition_to_event_expr(condition: Condition) -> tuple[Predicate, ...]:
    predicates = tuple(_flatten_condition_predicates(condition))
    if not predicates:
        raise TimingDslError(f"event/anchor expression is not diagram-compatible: {condition}")
    return predicates


def _flatten_condition_predicates(condition: Condition) -> List[Predicate]:
    if condition.kind == "predicate" and condition.predicate is not None:
        predicate = condition.predicate
        if predicate.signal is None:
            return []
        if predicate.op == "past":
            return []
        return [predicate]
    if condition.kind == "all":
        predicates: List[Predicate] = []
        for item in condition.items:
            predicates.extend(_flatten_condition_predicates(item))
        return predicates
    return []


def _parse_event_expr(expr_text: str) -> tuple[Predicate, ...]:
    expr_text = expr_text.removesuffix(" same_cycle").strip()
    condition = parse_dsl_condition(expr_text)
    return _condition_to_event_expr(condition)


def _event_expr_to_sva(expr: Sequence[Predicate]) -> str:
    parts = []
    for predicate in expr:
        if predicate.op == "rise":
            parts.append(f"$rose({predicate.signal})")
        elif predicate.op == "fall":
            parts.append(f"$fell({predicate.signal})")
        elif predicate.op == "high":
            parts.append(predicate.signal or "")
        elif predicate.op == "low":
            parts.append(f"!{predicate.signal}")
        elif predicate.op == "stable":
            parts.append(f"$stable({predicate.signal})")
        elif predicate.op == "change":
            parts.append(f"$changed({predicate.signal})")
        elif predicate.op == "eq":
            parts.append(f"({predicate.signal} == {predicate.value})")
        elif predicate.op == "neq":
            parts.append(f"({predicate.signal} != {predicate.value})")
        else:
            raise TimingDslError(f"unsupported event predicate: {predicate.op}")
    return " && ".join(parts)


def _link_property_references(
    properties: Sequence[PropertyOverlay],
    anchors: Sequence[Anchor],
    windows: Sequence[TimeWindow],
    lane_constraints: Sequence[LaneConstraint],
) -> tuple[PropertyOverlay, ...]:
    anchor_names = {anchor.name for anchor in anchors}
    window_names = {window.name for window in windows}
    constraint_names = {constraint.name for constraint in lane_constraints}
    linked = []
    for property_overlay in properties:
        referenced_anchors = list(property_overlay.related_anchors)
        referenced_windows = list(property_overlay.related_windows)
        referenced_constraints = list(property_overlay.related_constraints)
        for anchor_name in sorted(anchor_names):
            if re.search(rf"\b{anchor_name}\b", property_overlay.body) and anchor_name not in referenced_anchors:
                referenced_anchors.append(anchor_name)
        for window_name in sorted(window_names):
            if re.search(rf"\b{window_name}\b", property_overlay.body) and window_name not in referenced_windows:
                referenced_windows.append(window_name)
        for constraint_name in sorted(constraint_names):
            if (
                re.search(rf"\b{constraint_name}\b", property_overlay.body)
                and constraint_name not in referenced_constraints
            ):
                referenced_constraints.append(constraint_name)
        linked.append(
            replace(
                property_overlay,
                related_anchors=tuple(referenced_anchors),
                related_windows=tuple(referenced_windows),
                related_constraints=tuple(referenced_constraints),
            )
        )
    return tuple(linked)


def _parse_status(text: str | None) -> ExtractionStatus:
    if text is None:
        return ExtractionStatus.EXACT
    return ExtractionStatus(text)


def _strip_optional_quotes(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text
