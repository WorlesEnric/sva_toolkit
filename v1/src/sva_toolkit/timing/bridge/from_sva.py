"""SVA to timing scenario extraction and bundling."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, List, Sequence, Tuple

from sva_toolkit.ast_parser import SVAASTParser
from sva_toolkit.ast_parser.parser import ImplicationType
from sva_toolkit.timing.core.conditions import (
    Condition,
    collect_signals,
    condition_to_sva,
    parse_sva_condition,
)
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
    merge_bundle_metadata,
    rebind_names,
)


_UNSUPPORTED_KEYWORDS = (
    "accept_on",
    "reject_on",
    "sync_accept_on",
    "sync_reject_on",
    "local",
    "var",
)


def extract_sva_scenario(sva_text: str, *, name: str | None = None, parser: SVAASTParser | None = None) -> ScenarioDocument:
    """Extract one SVA property into a scenario document."""

    parser = parser or SVAASTParser()
    structure = parser.parse(sva_text)
    property_body = parser._extract_property_body(sva_text).strip().strip(";")
    property_name = name or structure.property_name or "sva_property"
    normalized_body = _normalize_property_body(structure, property_body)

    clock_signal = structure.clock_signal or "clk"
    clocking = ClockingSpec(edge=structure.clock_edge, signal=clock_signal, disable_iff=structure.disable_condition)
    params = tuple(ParameterDecl(name=token) for token in _collect_parameters(normalized_body))
    signals = tuple(_build_signal_decls(structure, normalized_body, clock_signal))

    notes: List[str] = []
    status = ExtractionStatus.EXACT
    lower_body = normalized_body.lower()
    if any(keyword in lower_body for keyword in _UNSUPPORTED_KEYWORDS):
        status = ExtractionStatus.UNSUPPORTED
        notes.append("unsupported control wrapper or local sampled variable")
    elif "first_match" in lower_body:
        status = ExtractionStatus.UNSUPPORTED
        notes.append("first_match with branching is not diagrammed")

    anchors: List[Anchor] = []
    windows: List[TimeWindow] = []
    cuts: List[Cut] = []
    constraints: List[LaneConstraint] = []

    if status != ExtractionStatus.UNSUPPORTED:
        if structure.implication_type == ImplicationType.NONE:
            if structure.temporal_operators:
                synthetic_trigger = _make_anchor("root", parse_sva_condition("1'b1"), AnchorRole.SYNTHETIC)
                anchors.append(synthetic_trigger)
                sub_status, sub_notes = _extract_sequence(
                    property_name,
                    normalized_body,
                    synthetic_trigger.name,
                    anchors,
                    windows,
                    cuts,
                    constraints,
                )
                status = _merge_status(status, sub_status)
                notes.extend(sub_notes)
            else:
                status = ExtractionStatus.UNSUPPORTED
                notes.append("property has no implication or temporal skeleton to diagram")
        else:
            antecedent = structure.antecedent or "1'b1"
            consequent = structure.consequent or ""
            if structure.implication_type == ImplicationType.NON_OVERLAPPING:
                consequent = f"##1 {consequent}"
            trigger_anchor = _make_anchor("trigger", parse_sva_condition(antecedent), AnchorRole.TRIGGER)
            anchors.append(trigger_anchor)
            sub_status, sub_notes = _extract_sequence(
                property_name,
                consequent,
                trigger_anchor.name,
                anchors,
                windows,
                cuts,
                constraints,
            )
            status = _merge_status(status, sub_status)
            notes.extend(sub_notes)

    if not anchors and status == ExtractionStatus.UNSUPPORTED:
        anchors.append(_make_anchor("unsupported", parse_sva_condition("1'b1"), AnchorRole.SYNTHETIC))
    elif status != ExtractionStatus.UNSUPPORTED:
        _add_focus_cuts(property_name, anchors, windows, cuts)

    property_overlay = PropertyOverlay(
        name=property_name,
        body=normalized_body,
        status=status,
        source=sva_text.strip(),
        related_anchors=tuple(anchor.name for anchor in anchors),
        related_windows=tuple(window.name for window in windows),
        related_constraints=tuple(constraint.name for constraint in constraints),
        notes=tuple(notes),
    )

    return ScenarioDocument(
        name=property_name,
        clocking=clocking,
        params=params,
        signals=signals,
        anchors=tuple(anchors),
        windows=tuple(windows),
        cuts=tuple(cuts),
        lane_constraints=tuple(constraints),
        properties=(property_overlay,),
        extraction_status=status,
        notes=tuple(notes),
    )


def extract_sva_scenarios(sva_texts: Iterable[str], parser: SVAASTParser | None = None) -> Tuple[ScenarioDocument, ...]:
    """Extract multiple SVA properties into independent scenario documents."""

    parser = parser or SVAASTParser()
    return tuple(extract_sva_scenario(sva_text, parser=parser) for sva_text in sva_texts)


def bundle_sva_scenarios(
    documents: Sequence[ScenarioDocument],
    *,
    overlap_threshold: float = 0.35,
    max_signals: int = 10,
    max_properties: int = 5,
    max_chains: int = 3,
) -> Tuple[ScenarioDocument, ...]:
    """Group compatible scenario documents into deterministic bundles."""

    if not documents:
        return ()

    key_groups = {}
    for document in documents:
        key = (
            document.clocking.edge,
            document.clocking.signal,
            document.clocking.disable_iff or "",
        )
        key_groups.setdefault(key, []).append(document)

    bundled: List[ScenarioDocument] = []
    for group_documents in key_groups.values():
        ordered = sorted(group_documents, key=lambda document: document.name)
        while ordered:
            seed = ordered.pop(0)
            cluster = [seed]
            seed_signals = set(signal.name for signal in seed.signals)
            remaining = []
            for candidate in ordered:
                candidate_signals = set(signal.name for signal in candidate.signals)
                overlap = _signal_overlap(seed_signals, candidate_signals)
                if overlap >= overlap_threshold and _compatible_documents(cluster, candidate):
                    cluster.append(candidate)
                    seed_signals |= candidate_signals
                else:
                    remaining.append(candidate)
            ordered = remaining
            if len(cluster) == 1:
                bundled.append(cluster[0])
                continue
            if len(seed_signals) > max_signals or sum(len(doc.properties) for doc in cluster) > max_properties:
                bundled.extend(cluster)
                continue
            if _independent_chain_count(cluster) > max_chains:
                bundled.extend(cluster)
                continue
            bundled.append(_merge_cluster(tuple(cluster)))
    return tuple(bundled)


def _extract_sequence(
    property_name: str,
    expression: str,
    start_anchor_name: str,
    anchors: List[Anchor],
    windows: List[TimeWindow],
    cuts: List[Cut],
    constraints: List[LaneConstraint],
) -> Tuple[ExtractionStatus, List[str]]:
    expression = _strip_outer_parens(expression.strip())
    notes: List[str] = []
    status = ExtractionStatus.EXACT

    if not expression:
        return ExtractionStatus.UNSUPPORTED, ["empty consequent"]

    if expression.lower().startswith("if "):
        return ExtractionStatus.LOSSY, ["if/else property body preserved as lossy overlay"]

    for op in (" and ", " or "):
        parts = _split_top_level_word(expression, op.strip())
        if len(parts) > 1:
            child_statuses = []
            for index, part in enumerate(parts):
                child_status, child_notes = _extract_sequence(
                    property_name,
                    part,
                    start_anchor_name,
                    anchors,
                    windows,
                    cuts,
                    constraints,
                )
                child_statuses.append(child_status)
                notes.extend(child_notes)
                if op.strip() == "or":
                    notes.append(f"branch {index + 1} retained as one alternative")
            return _merge_status(ExtractionStatus.LOSSY, *child_statuses), notes

    until_split = _split_until(expression)
    if until_split is not None:
        left_text, op, right_text = until_split
        end_anchor = _append_condition_anchor(
            anchors,
            f"{property_name}__until_end",
            parse_sva_condition(right_text),
            AnchorRole.RESPONSE,
        )
        windows.append(
            TimeWindow(
                name=f"{property_name}__hold_window",
                start_anchor=start_anchor_name,
                end_anchor=end_anchor.name,
                bound=TimeBound(kind=WindowBoundKind.OMITTED),
            )
        )
        constraints.extend(
            _constraints_from_condition(
                parse_sva_condition(left_text),
                f"{property_name}__hold",
                ConstraintRegion.FROM_UNTIL,
                start_anchor=start_anchor_name,
                end_anchor=end_anchor.name,
            )
        )
        if op == "until":
            status = ExtractionStatus.LOSSY
            notes.append("until excludes the terminating cycle and is shown as a hold-until skeleton")
        return status, notes

    throughout_split = _split_top_level_word(expression, "throughout")
    if len(throughout_split) > 1:
        left_text = throughout_split[0]
        right_text = throughout_split[1]
        child_status, child_notes = _extract_sequence(
            property_name,
            right_text,
            start_anchor_name,
            anchors,
            windows,
            cuts,
            constraints,
        )
        notes.extend(child_notes)
        target_window = windows[-1].name if windows else None
        if target_window is None:
            return ExtractionStatus.UNSUPPORTED, ["throughout target did not produce a window"]
        constraints.extend(
            _constraints_from_condition(
                parse_sva_condition(left_text),
                f"{property_name}__throughout",
                ConstraintRegion.IN,
                window=target_window,
            )
        )
        return child_status, notes

    intersect_split = _split_top_level_word(expression, "intersect")
    if len(intersect_split) > 1:
        child_statuses = []
        notes.append("intersect can admit multiple interval layouts; diagram is lossy")
        for part in intersect_split:
            child_status, child_notes = _extract_sequence(
                property_name,
                part,
                start_anchor_name,
                anchors,
                windows,
                cuts,
                constraints,
            )
            child_statuses.append(child_status)
            notes.extend(child_notes)
        return _merge_status(ExtractionStatus.LOSSY, *child_statuses), notes

    delay_match = _match_leading_delay(expression)
    if delay_match is not None:
        delay_token, rest = delay_match
        end_anchor = _append_condition_anchor(
            anchors,
            f"{property_name}__response",
            _extract_anchor_condition(rest),
            AnchorRole.RESPONSE,
        )
        bound = _delay_to_bound(delay_token)
        window_name = f"{property_name}__delay_window"
        windows.append(TimeWindow(name=window_name, start_anchor=start_anchor_name, end_anchor=end_anchor.name, bound=bound))
        if bound.kind == WindowBoundKind.UNBOUNDED:
            cuts.append(
                Cut(
                    name=f"{property_name}__future_cut",
                    placement=CutPlacement.AFTER_ANCHOR,
                    meaning=CutMeaning.OMITTED_FUTURE,
                    anchor=end_anchor.name,
                    label=bound.label,
                )
            )
        repeat_status, repeat_notes = _extract_repeat_overlay(rest, property_name, start_anchor_name, end_anchor.name, constraints)
        notes.extend(repeat_notes)
        return _merge_status(status, repeat_status), notes

    repeat_status, repeat_notes = _extract_repeat_overlay(expression, property_name, start_anchor_name, None, constraints)
    if repeat_notes or repeat_status != ExtractionStatus.EXACT:
        notes.extend(repeat_notes)
        return repeat_status, notes

    response_anchor = _append_condition_anchor(
        anchors,
        f"{property_name}__response",
        _extract_anchor_condition(expression),
        AnchorRole.RESPONSE,
    )
    windows.append(
        TimeWindow(
            name=f"{property_name}__instant_window",
            start_anchor=start_anchor_name,
            end_anchor=response_anchor.name,
            bound=TimeBound(kind=WindowBoundKind.EXACT, min_delay="0", max_delay="0"),
        )
    )
    return status, notes


def _extract_repeat_overlay(
    expression: str,
    property_name: str,
    start_anchor_name: str,
    end_anchor_name: str | None,
    constraints: List[LaneConstraint],
) -> Tuple[ExtractionStatus, List[str]]:
    repeat_match = _match_repetition(expression)
    if repeat_match is None:
        return ExtractionStatus.EXACT, []

    base_text, op, count_text = repeat_match
    relation_constraints = _constraints_from_condition(
        parse_sva_condition(base_text),
        f"{property_name}__repeat",
        ConstraintRegion.FROM_UNTIL,
        start_anchor=start_anchor_name,
        end_anchor=end_anchor_name or start_anchor_name,
    )
    constraints.extend(relation_constraints)

    if op == "[*":
        return (ExtractionStatus.EXACT, [])
    if op == "[=":
        return (ExtractionStatus.LOSSY, [f"non-consecutive repetition {count_text} shown as a compressed hold region"])
    return (ExtractionStatus.LOSSY, [f"goto repetition {count_text} shown as a compressed reachability region"])


def _constraints_from_condition(condition: Condition, prefix: str, region: ConstraintRegion, **region_kwargs) -> List[LaneConstraint]:
    constraints = []
    predicates = _condition_predicates(condition)
    if not predicates:
        text = condition_to_sva(condition)
        if text:
            constraints.append(
                LaneConstraint(
                    name=f"{prefix}_raw",
                    signals=tuple(sorted(collect_signals(condition))),
                    relation="raw",
                    value=text,
                    region=region,
                    **region_kwargs,
                )
            )
        return constraints
    for index, predicate in enumerate(predicates):
        constraints.append(
            LaneConstraint(
                name=f"{prefix}_{index}",
                signals=(predicate.signal,),
                relation=predicate.op,
                value=predicate.value,
                region=region,
                **region_kwargs,
            )
        )
    return constraints


def _condition_predicates(condition: Condition):
    if condition.kind == "predicate" and condition.predicate is not None and condition.predicate.signal:
        return [condition.predicate]
    if condition.kind == "all":
        predicates = []
        for item in condition.items:
            predicates.extend(_condition_predicates(item))
        return predicates
    return []


def _extract_anchor_condition(expression: str) -> Condition:
    expression = _strip_outer_parens(expression.strip())
    repeat_match = _match_repetition(expression)
    if repeat_match is not None:
        expression = repeat_match[0]
    return parse_sva_condition(expression)


def _append_condition_anchor(anchors: List[Anchor], base_name: str, condition: Condition, role: AnchorRole) -> Anchor:
    existing_names = {anchor.name for anchor in anchors}
    candidate = base_name
    index = 2
    while candidate in existing_names:
        candidate = f"{base_name}_{index}"
        index += 1
    anchor = _make_anchor(candidate, condition, role)
    anchors.append(anchor)
    if _condition_uses_past(condition):
        anchors.append(_make_anchor(f"{candidate}__lookback", condition, AnchorRole.LOOKBACK))
    return anchor


def _condition_uses_past(condition: Condition) -> bool:
    if condition.kind == "predicate" and condition.predicate is not None:
        return condition.predicate.op == "past"
    return any(_condition_uses_past(item) for item in condition.items)


def _make_anchor(name: str, condition: Condition, role: AnchorRole) -> Anchor:
    return Anchor(name=name, condition=condition, role=role)


def _add_focus_cuts(property_name: str, anchors: List[Anchor], windows: List[TimeWindow], cuts: List[Cut]) -> None:
    if not windows:
        return

    anchor_map = {anchor.name: anchor for anchor in anchors}
    focus_start_name = windows[0].start_anchor
    focus_start = anchor_map.get(focus_start_name)
    if (
        focus_start is not None
        and not (
            focus_start.role == AnchorRole.SYNTHETIC
            and condition_to_sva(focus_start.condition).strip() == "1'b1"
        )
        and not any(cut.placement == CutPlacement.BEFORE_ANCHOR and cut.anchor == focus_start_name for cut in cuts)
    ):
        cuts.insert(
            0,
            Cut(
                name=f"{property_name}__history_cut",
                placement=CutPlacement.BEFORE_ANCHOR,
                meaning=CutMeaning.OMITTED_HISTORY,
                anchor=focus_start_name,
            ),
        )

    start_anchor_names = {window.start_anchor for window in windows}
    terminal_anchor_names = []
    for window in windows:
        if window.end_anchor not in start_anchor_names and window.end_anchor not in terminal_anchor_names:
            terminal_anchor_names.append(window.end_anchor)
    if not terminal_anchor_names:
        terminal_anchor_names.append(windows[-1].end_anchor)

    for index, anchor_name in enumerate(terminal_anchor_names):
        if any(cut.placement == CutPlacement.AFTER_ANCHOR and cut.anchor == anchor_name for cut in cuts):
            continue
        cut_name = f"{property_name}__future_cut" if len(terminal_anchor_names) == 1 else f"{anchor_name}__future_cut"
        cuts.append(
            Cut(
                name=cut_name,
                placement=CutPlacement.AFTER_ANCHOR,
                meaning=CutMeaning.OMITTED_FUTURE,
                anchor=anchor_name,
            )
        )


def _delay_to_bound(delay_token: str) -> TimeBound:
    token = delay_token.replace(" ", "")
    if re.fullmatch(r"##\d+", token):
        value = token[2:]
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=value, max_delay=value)
    match = re.fullmatch(r"##\[(.+):(.+)\]", token)
    if not match:
        return TimeBound(kind=WindowBoundKind.OMITTED)
    min_delay, max_delay = match.groups()
    if max_delay == "$":
        return TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay=min_delay, max_delay=max_delay)
    if min_delay == max_delay:
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=min_delay, max_delay=max_delay)
    return TimeBound(kind=WindowBoundKind.RANGE, min_delay=min_delay, max_delay=max_delay)


def _match_leading_delay(expression: str) -> Tuple[str, str] | None:
    if not expression.startswith("##"):
        return None
    match = re.match(r"^(##(?:\[\s*[^]]+\s*\]|\d+))\s+(.+)$", expression)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _match_repetition(expression: str) -> Tuple[str, str, str] | None:
    expression = _strip_outer_parens(expression)
    if not expression.endswith("]"):
        return None
    depth = 0
    for index in range(len(expression) - 1, -1, -1):
        char = expression[index]
        if char == "]":
            depth += 1
        elif char == "[":
            depth -= 1
            if depth == 0:
                content = expression[index + 1:-1].strip()
                if content.startswith("*"):
                    return expression[:index].strip(), "[*", content[1:].strip()
                if content.startswith("="):
                    return expression[:index].strip(), "[=", content[1:].strip()
                if content.startswith("->"):
                    return expression[:index].strip(), "[->", content[2:].strip()
                return None
    return None


def _split_until(expression: str) -> Tuple[str, str, str] | None:
    for op in ("until_with", "until"):
        parts = _split_top_level_word(expression, op)
        if len(parts) > 1:
            return parts[0], op, parts[1]
    return None


def _split_top_level_word(expression: str, word: str) -> Tuple[str, ...]:
    token = f" {word} "
    parts = []
    depth_paren = 0
    depth_brack = 0
    start = 0
    index = 0
    while index <= len(expression) - len(token):
        char = expression[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        if depth_paren == 0 and depth_brack == 0 and expression[index:index + len(token)] == token:
            parts.append(expression[start:index].strip())
            start = index + len(token)
            index = start
            continue
        index += 1
    if not parts:
        return (expression.strip(),)
    parts.append(expression[start:].strip())
    return tuple(part for part in parts if part)


def _strip_outer_parens(expression: str) -> str:
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        wrapped = True
        for index, char in enumerate(expression):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(expression) - 1:
                    wrapped = False
                    break
        if not wrapped:
            break
        expression = expression[1:-1].strip()
    return expression


def _collect_parameters(property_body: str) -> Tuple[str, ...]:
    keywords = {
        "and",
        "or",
        "throughout",
        "until",
        "until_with",
        "intersect",
        "first_match",
        "disable",
        "iff",
        "posedge",
        "negedge",
    }
    params = []
    for token in re.findall(r"\b[A-Z][A-Z0-9_]*\b", property_body):
        if token not in keywords and token not in params:
            params.append(token)
    return tuple(params)


def _normalize_property_body(structure, property_body: str) -> str:
    if structure.implication_type == ImplicationType.OVERLAPPING and structure.antecedent and structure.consequent:
        return f"{structure.antecedent} |-> {structure.consequent}"
    if structure.implication_type == ImplicationType.NON_OVERLAPPING and structure.antecedent and structure.consequent:
        return f"{structure.antecedent} |=> {structure.consequent}"
    normalized = re.sub(r"@\s*\([^)]+\)", "", property_body)
    normalized = re.sub(r"disable\s+iff\s*\([^)]+\)", "", normalized)
    return normalized.strip()


def _build_signal_decls(structure, property_body: str, clock_signal: str) -> List[SignalDecl]:
    bus_candidates = set()
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=)\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+'[bdhoBDHO][0-9A-Fa-f_xzXZ]+|\w+)", property_body):
        if match.group(3) not in {"0", "1"}:
            bus_candidates.add(match.group(1))
    signals = []
    seen = set()
    for signal in sorted(structure.signals, key=lambda item: item.name):
        if signal.name in {clock_signal, structure.reset_signal} or signal.name in seen:
            continue
        seen.add(signal.name)
        kind = SignalKind.BUS if signal.name in bus_candidates else SignalKind.BIT
        signals.append(SignalDecl(name=signal.name, kind=kind))
    return signals


def _merge_status(*statuses: ExtractionStatus) -> ExtractionStatus:
    if ExtractionStatus.UNSUPPORTED in statuses:
        return ExtractionStatus.UNSUPPORTED
    if ExtractionStatus.LOSSY in statuses:
        return ExtractionStatus.LOSSY
    return ExtractionStatus.EXACT


def _signal_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _compatible_documents(cluster: Sequence[ScenarioDocument], candidate: ScenarioDocument) -> bool:
    for existing in cluster:
        if existing.clocking != candidate.clocking:
            return False
        if _has_conflicting_constraints(existing, candidate):
            return False
    return True


def _has_conflicting_constraints(left: ScenarioDocument, right: ScenarioDocument) -> bool:
    left_facts = {_constraint_identity(constraint): constraint.value for constraint in left.lane_constraints if constraint.relation in {"eq", "high", "low"}}
    right_facts = {_constraint_identity(constraint): constraint.value for constraint in right.lane_constraints if constraint.relation in {"eq", "high", "low"}}
    for key, value in left_facts.items():
        if key in right_facts and right_facts[key] != value:
            return True
    return False


def _constraint_identity(constraint: LaneConstraint) -> Tuple:
    return (
        tuple(constraint.signals),
        constraint.region,
        constraint.anchor,
        constraint.window,
        constraint.start_anchor,
        constraint.end_anchor,
    )


def _independent_chain_count(documents: Sequence[ScenarioDocument]) -> int:
    chains = 0
    for document in documents:
        chains += max(1, len(document.properties))
    return chains


def _merge_cluster(documents: Tuple[ScenarioDocument, ...]) -> ScenarioDocument:
    primary = documents[0]
    rename_maps = []
    merged_signals = []
    merged_anchors = []
    merged_windows = []
    merged_cuts = []
    merged_constraints = []
    merged_properties = []
    signal_names = set()
    reserved_names = {param.name for document in documents for param in document.params}
    reserved_names.update(signal.name for document in documents for signal in document.signals)

    for document in documents:
        rename_map = {}
        for collection in (document.anchors, document.windows, document.cuts, document.lane_constraints, document.properties):
            for item in collection:
                rename_map[item.name] = _reserve_unique_name(
                    item.name,
                    document.name,
                    reserved_names,
                )
        rename_maps.append(rename_map)

    for document, rename_map in zip(documents, rename_maps):
        for signal in document.signals:
            if signal.name not in signal_names:
                merged_signals.append(signal)
                signal_names.add(signal.name)
        for anchor in document.anchors:
            merged_anchors.append(replace(anchor, name=rename_map.get(anchor.name, anchor.name)))
        for window in document.windows:
            merged_windows.append(
                replace(
                    window,
                    name=rename_map.get(window.name, window.name),
                    start_anchor=rename_map.get(window.start_anchor, window.start_anchor),
                    end_anchor=rename_map.get(window.end_anchor, window.end_anchor),
                )
            )
        for cut in document.cuts:
            merged_cuts.append(
                replace(
                    cut,
                    name=rename_map.get(cut.name, cut.name),
                    anchor=rename_map.get(cut.anchor, cut.anchor) if cut.anchor else None,
                    left_window=rename_map.get(cut.left_window, cut.left_window) if cut.left_window else None,
                    right_window=rename_map.get(cut.right_window, cut.right_window) if cut.right_window else None,
                )
            )
        for constraint in document.lane_constraints:
            merged_constraints.append(
                replace(
                    constraint,
                    name=rename_map.get(constraint.name, constraint.name),
                    anchor=rename_map.get(constraint.anchor, constraint.anchor) if constraint.anchor else None,
                    window=rename_map.get(constraint.window, constraint.window) if constraint.window else None,
                    start_anchor=rename_map.get(constraint.start_anchor, constraint.start_anchor) if constraint.start_anchor else None,
                    end_anchor=rename_map.get(constraint.end_anchor, constraint.end_anchor) if constraint.end_anchor else None,
                )
            )
        for prop in document.properties:
            merged_properties.append(
                replace(
                    prop,
                    name=rename_map.get(prop.name, prop.name),
                    body=rebind_names(rename_map, prop.body),
                    related_anchors=tuple(rename_map.get(anchor, anchor) for anchor in prop.related_anchors),
                    related_windows=tuple(rename_map.get(window, window) for window in prop.related_windows),
                    related_constraints=tuple(rename_map.get(item, item) for item in prop.related_constraints),
                )
            )

    overlap = 1.0
    if len(documents) > 1:
        overlap = _signal_overlap(set(signal.name for signal in documents[0].signals), set(signal.name for signal in documents[1].signals))

    return ScenarioDocument(
        name="__".join(document.name for document in documents),
        clocking=primary.clocking,
        params=tuple(dict.fromkeys(param for document in documents for param in document.params)),
        signals=tuple(merged_signals),
        anchors=tuple(merged_anchors),
        windows=tuple(merged_windows),
        cuts=tuple(merged_cuts),
        lane_constraints=tuple(merged_constraints),
        properties=tuple(merged_properties),
        extraction_status=_merge_status(*(document.effective_status for document in documents)),
        bundle=merge_bundle_metadata(documents, overlap),
        notes=tuple(note for document in documents for note in document.notes),
    )


def _reserve_unique_name(name: str, document_name: str, reserved_names: set[str]) -> str:
    if name not in reserved_names:
        reserved_names.add(name)
        return name

    base_name = f"{document_name}__{name}"
    candidate = base_name
    index = 2
    while candidate in reserved_names:
        candidate = f"{base_name}_{index}"
        index += 1
    reserved_names.add(candidate)
    return candidate
