"""Lower semantic timing scenarios to the visual-target contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Mapping, Sequence, TypeGuard, TypeVar, cast

from sva_toolkit.sva.ast import PropertyNode
from sva_toolkit.sva.emitter import emit_property_body
from sva_toolkit.sva.transforms import RenameIdentifiersTransformer
from sva_toolkit.timing.core.conditions import Condition, Predicate
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    BundleMetadata,
    ConstraintRegion,
    Cut,
    LaneConstraint,
    PropertyOverlay,
    ScenarioDocument,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.visual.policy import (
    AnchorNamePolicy,
    BoundPolicy,
    ConstraintNamePolicy,
    TargetPolicy,
    WindowNamePolicy,
)
from sva_toolkit.timing.visual.visibility import FieldVisibility, VisibilityClass, VisibilityReport


_PREDICATE_PRIORITY = {
    "rise": 0,
    "rose": 0,
    "fall": 1,
    "fell": 1,
    "change": 2,
    "changed": 2,
    "eq": 3,
    "neq": 4,
    "high": 5,
    "low": 6,
    "stable": 7,
}
_CANONICAL_NAME_RE = re.compile(r"^([A-Za-z_]+)([0-9]+)$")
_SMALL_INTEGER_BOUND_LIMIT = 64

T = TypeVar("T")


@dataclass(frozen=True)
class LoweringResult:
    """Result of lowering a semantic document to a visual target."""

    visual_document: ScenarioDocument
    visibility: VisibilityReport
    anchor_renames: Mapping[str, str]
    window_renames: Mapping[str, str]
    constraint_renames: Mapping[str, str]
    dropped_properties: tuple[str, ...]
    dropped_notes: tuple[str, ...]


def lower_to_visual_document(
    document: ScenarioDocument,
    policy: TargetPolicy | None = None,
) -> LoweringResult:
    """Lower a semantic scenario document to the configured visual-target contract."""

    policy = policy or TargetPolicy.visual()

    anchor_order, anchor_renames = _ordered_anchor_renames(document.anchors, policy)
    lowered_anchors = tuple(_lower_anchor(anchor, anchor_renames, policy) for anchor in anchor_order)
    anchor_index = {anchor.name: index for index, anchor in enumerate(lowered_anchors)}

    rewritten_windows: list[TimeWindow] = []
    bound_visibility_by_original_window: dict[str, VisibilityClass] = {}
    for window in document.windows:
        lowered_bound, visibility_class = _lower_bound(window.bound, policy)
        rewritten_windows.append(
            replace(
                window,
                start_anchor=_rename_required(window.start_anchor, anchor_renames),
                end_anchor=_rename_required(window.end_anchor, anchor_renames),
                bound=lowered_bound,
            )
        )
        bound_visibility_by_original_window[window.name] = visibility_class

    window_order, window_renames = _ordered_window_renames(rewritten_windows, policy, anchor_index)
    lowered_windows = tuple(replace(window, name=window_renames[window.name]) for window in window_order)

    rewritten_constraints = tuple(
        _rewrite_constraint_references(constraint, anchor_renames, window_renames)
        for constraint in document.lane_constraints
    )
    constraint_order, constraint_renames = _ordered_constraint_renames(rewritten_constraints, policy)
    lowered_constraints = tuple(
        replace(constraint, name=constraint_renames[constraint.name]) for constraint in constraint_order
    )

    combined_renames = {**anchor_renames, **window_renames, **constraint_renames}
    lowered_cuts = tuple(_rewrite_cut_references(cut, anchor_renames, window_renames) for cut in document.cuts)
    lowered_properties, dropped_properties, dropped_notes = _lower_properties(document.properties, combined_renames, policy)

    if policy.drop_notes and document.notes:
        dropped_notes = (*document.notes, *dropped_notes)
    document_notes = () if policy.drop_notes else document.notes

    visual_document = replace(
        document,
        anchors=lowered_anchors,
        windows=lowered_windows,
        cuts=lowered_cuts,
        lane_constraints=lowered_constraints,
        properties=lowered_properties,
        bundle=BundleMetadata() if policy.drop_bundle_metadata else document.bundle,
        notes=document_notes,
        ceg=None if _is_visual_policy(policy) else document.ceg,
    )

    visibility = _build_visibility_report(
        document=visual_document,
        anchor_renames=anchor_renames,
        window_renames=window_renames,
        constraint_renames=constraint_renames,
        bound_visibility_by_original_window=bound_visibility_by_original_window,
        dropped_properties=dropped_properties,
        dropped_notes=dropped_notes,
        policy=policy,
    )

    return LoweringResult(
        visual_document=visual_document,
        visibility=visibility,
        anchor_renames=anchor_renames,
        window_renames=window_renames,
        constraint_renames=constraint_renames,
        dropped_properties=dropped_properties,
        dropped_notes=dropped_notes,
    )


def _ordered_anchor_renames(
    anchors: Sequence[Anchor],
    policy: TargetPolicy,
) -> tuple[tuple[Anchor, ...], dict[str, str]]:
    if policy.anchor_names == AnchorNamePolicy.KEEP_ORIGINAL:
        ordered = tuple(anchors)
        return ordered, {anchor.name: anchor.name for anchor in ordered}

    if all(anchor.absolute_tick is None for anchor in anchors) and _has_complete_canonical_names(anchors, "a"):
        ordered = tuple(sorted(anchors, key=lambda anchor: _canonical_name_number(anchor.name)))
    else:
        ordered = tuple(
            anchor
            for _, anchor in sorted(
                enumerate(anchors),
                key=lambda item: (*_anchor_visual_sort_key(item[1]), item[0]),
            )
        )
    return ordered, {anchor.name: f"a{index}" for index, anchor in enumerate(ordered)}


def _ordered_window_renames(
    windows: Sequence[TimeWindow],
    policy: TargetPolicy,
    anchor_index: Mapping[str, int],
) -> tuple[tuple[TimeWindow, ...], dict[str, str]]:
    if policy.window_names == WindowNamePolicy.KEEP_ORIGINAL:
        ordered = tuple(windows)
        return ordered, {window.name: window.name for window in ordered}

    if _has_complete_canonical_names(windows, "w"):
        ordered = tuple(sorted(windows, key=lambda window: _canonical_name_number(window.name)))
    else:
        ordered = tuple(
            window
            for _, window in sorted(
                enumerate(windows),
                key=lambda item: (
                    anchor_index.get(item[1].start_anchor, len(anchor_index)),
                    anchor_index.get(item[1].end_anchor, len(anchor_index)),
                    item[1].bound.label,
                    item[0],
                ),
            )
        )
    return ordered, {window.name: f"w{index}" for index, window in enumerate(ordered)}


def _ordered_constraint_renames(
    constraints: Sequence[LaneConstraint],
    policy: TargetPolicy,
) -> tuple[tuple[LaneConstraint, ...], dict[str, str]]:
    if policy.constraint_names == ConstraintNamePolicy.KEEP_ORIGINAL:
        ordered = tuple(constraints)
        return ordered, {constraint.name: constraint.name for constraint in ordered}

    if _has_complete_canonical_names(constraints, "c"):
        ordered = tuple(sorted(constraints, key=lambda constraint: _canonical_name_number(constraint.name)))
    else:
        ordered = tuple(
            constraint
            for _, constraint in sorted(
                enumerate(constraints),
                key=lambda item: (*_constraint_visual_sort_key(item[1]), item[0]),
            )
        )
    return ordered, {constraint.name: f"c{index}" for index, constraint in enumerate(ordered)}


def _lower_anchor(anchor: Anchor, anchor_renames: Mapping[str, str], policy: TargetPolicy) -> Anchor:
    return replace(
        anchor,
        name=anchor_renames[anchor.name],
        derived_from=_rename_optional(anchor.derived_from, anchor_renames),
        role=AnchorRole.STATE if _is_visual_policy(policy) else anchor.role,
        role_metadata=None if _is_visual_policy(policy) else anchor.role_metadata,
    )


def _lower_bound(bound: TimeBound, policy: TargetPolicy) -> tuple[TimeBound, VisibilityClass]:
    visibility_class = _classify_bound_visibility(bound)
    if policy.bounds == BoundPolicy.KEEP_ALL:
        return bound, visibility_class

    if policy.bounds == BoundPolicy.DROP_PARAMETERIZED and _bound_has_symbolic_token(bound):
        return _drop_symbolic_bound(bound), VisibilityClass.HIDDEN_SEMANTIC

    if policy.bounds == BoundPolicy.GEOMETRY_ONLY:
        if visibility_class == VisibilityClass.VISIBLE_GEOMETRY:
            return bound, visibility_class
        if _bound_has_symbolic_token(bound):
            return _drop_symbolic_bound(bound), VisibilityClass.HIDDEN_SEMANTIC

    return bound, visibility_class


def _drop_symbolic_bound(bound: TimeBound) -> TimeBound:
    min_delay = bound.min_delay if _is_integer_literal(bound.min_delay) else "0"
    return TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay=min_delay, max_delay="$", inclusive=bound.inclusive)


def _classify_bound_visibility(bound: TimeBound) -> VisibilityClass:
    if _bound_has_symbolic_token(bound):
        return VisibilityClass.HIDDEN_SEMANTIC
    if (
        bound.kind == WindowBoundKind.EXACT
        and _is_integer_literal(bound.min_delay)
        and int(bound.min_delay) <= _SMALL_INTEGER_BOUND_LIMIT
    ):
        return VisibilityClass.VISIBLE_GEOMETRY
    if bound.kind == WindowBoundKind.RANGE and _is_integer_literal(bound.min_delay) and _is_integer_literal(
        bound.max_delay
    ):
        return VisibilityClass.VISIBLE_TEXT
    if bound.kind == WindowBoundKind.UNBOUNDED and _is_integer_literal(bound.min_delay) and bound.max_delay == "$":
        return VisibilityClass.VISIBLE_TEXT
    return VisibilityClass.HIDDEN_SEMANTIC


def _rewrite_constraint_references(
    constraint: LaneConstraint,
    anchor_renames: Mapping[str, str],
    window_renames: Mapping[str, str],
) -> LaneConstraint:
    return replace(
        constraint,
        anchor=_rename_optional(constraint.anchor, anchor_renames),
        window=_rename_optional(constraint.window, window_renames),
        start_anchor=_rename_optional(constraint.start_anchor, anchor_renames),
        end_anchor=_rename_optional(constraint.end_anchor, anchor_renames),
    )


def _rewrite_cut_references(
    cut: Cut,
    anchor_renames: Mapping[str, str],
    window_renames: Mapping[str, str],
) -> Cut:
    return replace(
        cut,
        anchor=_rename_optional(cut.anchor, anchor_renames),
        left_window=_rename_optional(cut.left_window, window_renames),
        right_window=_rename_optional(cut.right_window, window_renames),
    )


def _lower_properties(
    properties: Sequence[PropertyOverlay],
    renames: Mapping[str, str],
    policy: TargetPolicy,
) -> tuple[tuple[PropertyOverlay, ...], tuple[str, ...], tuple[str, ...]]:
    lowered: list[PropertyOverlay] = []
    dropped_properties: list[str] = []
    dropped_notes: list[str] = []

    for prop in properties:
        if policy.drop_property_paraphrase and prop.body_ast is None:
            dropped_properties.append(prop.name)
            if policy.drop_notes:
                dropped_notes.extend(_qualified_property_notes(prop))
            continue

        body_ast = prop.body_ast
        if body_ast is not None:
            body_ast = cast(PropertyNode, RenameIdentifiersTransformer(dict(renames)).visit(body_ast))
            body = emit_property_body(body_ast)
        else:
            body = _rebind_identifiers(prop.body, renames)

        notes = prop.notes
        if policy.drop_notes and prop.notes:
            dropped_notes.extend(_qualified_property_notes(prop))
            notes = ()

        lowered.append(
            replace(
                prop,
                body=body,
                source=None if _is_visual_policy(policy) else prop.source,
                related_anchors=tuple(_rename_required(anchor, renames) for anchor in prop.related_anchors),
                related_windows=tuple(_rename_required(window, renames) for window in prop.related_windows),
                related_constraints=tuple(_rename_required(constraint, renames) for constraint in prop.related_constraints),
                notes=notes,
                body_ast=body_ast,
            )
        )

    return tuple(lowered), tuple(dropped_properties), tuple(dropped_notes)


def _build_visibility_report(
    *,
    document: ScenarioDocument,
    anchor_renames: Mapping[str, str],
    window_renames: Mapping[str, str],
    constraint_renames: Mapping[str, str],
    bound_visibility_by_original_window: Mapping[str, VisibilityClass],
    dropped_properties: tuple[str, ...],
    dropped_notes: tuple[str, ...],
    policy: TargetPolicy,
) -> VisibilityReport:
    field_visibility: dict[str, FieldVisibility] = {}
    report_notes: list[str] = []

    signal_visibility = {
        signal.name: VisibilityClass.VISIBLE_TEXT
        for signal in document.signals
    }
    for signal in document.signals:
        field_visibility[f"signal:{signal.name}"] = FieldVisibility(
            name=signal.name,
            visibility_class=VisibilityClass.VISIBLE_TEXT,
            rationale="signal lane labels are natural visible text",
        )

    anchor_visibility = {anchor.name: VisibilityClass.VISIBLE_CONVENTION for anchor in document.anchors}
    for old_name, new_name in anchor_renames.items():
        field_visibility[f"anchor:{new_name}"] = FieldVisibility(
            name=new_name,
            visibility_class=VisibilityClass.VISIBLE_CONVENTION,
            rationale="anchor name is canonicalized by visual order",
        )
        if old_name != new_name:
            report_notes.append(f"renamed anchor {old_name} -> {new_name}")

    window_visibility = {window.name: VisibilityClass.VISIBLE_CONVENTION for window in document.windows}
    kept_bound_labels = {window.name: window.bound.label for window in document.windows}
    bound_visibility: dict[str, VisibilityClass] = {}
    for window in document.windows:
        original_name = _reverse_lookup(window_renames, window.name)
        visibility_class = bound_visibility_by_original_window.get(original_name, _classify_bound_visibility(window.bound))
        bound_visibility[window.name] = visibility_class
        field_visibility[f"window:{window.name}"] = FieldVisibility(
            name=window.name,
            visibility_class=VisibilityClass.VISIBLE_CONVENTION,
            rationale="window name is canonicalized by anchor order and bound label",
        )
        field_visibility[f"bound:{window.name}"] = FieldVisibility(
            name=window.name,
            visibility_class=visibility_class,
            rationale=_bound_visibility_rationale(visibility_class, window.bound),
        )

    for old_name, new_name in window_renames.items():
        if old_name != new_name:
            report_notes.append(f"renamed window {old_name} -> {new_name}")

    constraint_visibility = {
        constraint.name: VisibilityClass.VISIBLE_CONVENTION
        for constraint in document.lane_constraints
    }
    for old_name, new_name in constraint_renames.items():
        field_visibility[f"constraint:{new_name}"] = FieldVisibility(
            name=new_name,
            visibility_class=VisibilityClass.VISIBLE_CONVENTION,
            rationale="constraint name is canonicalized by visual relation order",
        )
        if old_name != new_name:
            report_notes.append(f"renamed constraint {old_name} -> {new_name}")

    dropped_fields: list[str] = [f"property:{name}" for name in dropped_properties]
    if policy.drop_notes and dropped_notes:
        dropped_fields.append("notes")
    if policy.drop_bundle_metadata:
        dropped_fields.append("bundle")
    for property_name in dropped_properties:
        report_notes.append(f"dropped property overlay {property_name}: body_ast is not available")
    if policy.drop_notes and dropped_notes:
        report_notes.append(f"dropped {len(dropped_notes)} note(s)")
    if policy.drop_bundle_metadata:
        report_notes.append("dropped bundle metadata")

    return VisibilityReport(
        field_visibility=field_visibility,
        dropped_fields=tuple(dropped_fields),
        renames={**anchor_renames, **window_renames, **constraint_renames},
        kept_bound_labels=kept_bound_labels,
        anchor_visibility=anchor_visibility,
        window_visibility=window_visibility,
        bound_visibility=bound_visibility,
        constraint_visibility=constraint_visibility,
        signal_visibility=signal_visibility,
        dropped_property_names=dropped_properties,
        dropped_note_count=len(dropped_notes),
        notes=tuple(report_notes),
    )


def _anchor_visual_sort_key(anchor: Anchor) -> tuple[int, int, int, str]:
    role_priority, primary_signal = _condition_visual_signature(anchor.condition)
    tick = anchor.absolute_tick if anchor.absolute_tick is not None else 0
    none_tick = 1 if anchor.absolute_tick is None else 0
    return (tick, none_tick, role_priority, primary_signal)


def _condition_visual_signature(condition: Condition) -> tuple[int, str]:
    predicates = list(_iter_predicates(condition))
    if not predicates:
        return (len(_PREDICATE_PRIORITY), condition.text or "")

    indexed = enumerate(predicates)
    _, selected = min(
        indexed,
        key=lambda item: (_PREDICATE_PRIORITY.get(item[1].op, len(_PREDICATE_PRIORITY)), _predicate_signal(item[1]), item[0]),
    )
    return (_PREDICATE_PRIORITY.get(selected.op, len(_PREDICATE_PRIORITY)), _predicate_signal(selected))


def _iter_predicates(condition: Condition) -> tuple[Predicate, ...]:
    if condition.kind == "predicate" and condition.predicate is not None:
        return (condition.predicate,)
    predicates: list[Predicate] = []
    for item in condition.items:
        predicates.extend(_iter_predicates(item))
    return tuple(predicates)


def _predicate_signal(predicate: Predicate) -> str:
    if predicate.signal:
        return predicate.signal
    for arg in predicate.args:
        if arg:
            return arg
    return predicate.text or ""


def _constraint_visual_sort_key(constraint: LaneConstraint) -> tuple[tuple[str, ...], str, tuple[str, ...], str, str]:
    target = _constraint_target(constraint)
    return (constraint.signals, constraint.region.value, target, constraint.relation, constraint.value or "")


def _constraint_target(constraint: LaneConstraint) -> tuple[str, ...]:
    if constraint.region in {ConstraintRegion.AT, ConstraintRegion.BEFORE, ConstraintRegion.AFTER}:
        return (constraint.anchor or "",)
    if constraint.region == ConstraintRegion.IN:
        return (constraint.window or "",)
    if constraint.region == ConstraintRegion.FROM_UNTIL:
        return (constraint.start_anchor or "", constraint.end_anchor or "")
    return ()


def _rename_optional(name: str | None, renames: Mapping[str, str]) -> str | None:
    if name is None:
        return None
    return renames.get(name, name)


def _rename_required(name: str, renames: Mapping[str, str]) -> str:
    return renames.get(name, name)


def _has_complete_canonical_names(items: Sequence[T], prefix: str) -> bool:
    names = [getattr(item, "name") for item in items]
    return set(names) == {f"{prefix}{index}" for index in range(len(names))}


def _canonical_name_number(name: str) -> int:
    match = _CANONICAL_NAME_RE.match(name)
    if match is None:
        return 0
    return int(match.group(2))


def _is_visual_policy(policy: TargetPolicy) -> bool:
    return policy.drop_property_paraphrase or policy.drop_notes or policy.drop_bundle_metadata


def _is_integer_literal(token: str | None) -> TypeGuard[str]:
    return bool(token and token.isdigit())


def _bound_has_symbolic_token(bound: TimeBound) -> bool:
    return _is_symbolic_bound_token(bound.min_delay) or _is_symbolic_bound_token(bound.max_delay)


def _is_symbolic_bound_token(token: str | None) -> bool:
    return token is not None and token != "$" and not token.isdigit()


def _qualified_property_notes(prop: PropertyOverlay) -> tuple[str, ...]:
    return tuple(f"{prop.name}: {note}" for note in prop.notes)


def _rebind_identifiers(text: str, renames: Mapping[str, str]) -> str:
    active_renames = {old: new for old, new in renames.items() if old != new}
    if not active_renames:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(active_renames, key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda match: active_renames[match.group(1)], text)


def _reverse_lookup(mapping: Mapping[str, str], value: str) -> str:
    for old_name, new_name in mapping.items():
        if new_name == value:
            return old_name
    return value


def _bound_visibility_rationale(visibility_class: VisibilityClass, bound: TimeBound) -> str:
    if visibility_class == VisibilityClass.VISIBLE_GEOMETRY:
        return f"exact integer bound {bound.label} can be inferred from tick spacing"
    if visibility_class == VisibilityClass.VISIBLE_TEXT:
        return f"bound {bound.label} is recoverable only if rendered as measurement text"
    return f"bound {bound.label} contains hidden or non-geometric timing semantics"
