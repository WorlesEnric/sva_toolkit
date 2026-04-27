from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.core.conditions import parse_dsl_condition
from sva_toolkit.timing.core.scenario import (
    Anchor,
    ClockingSpec,
    ConstraintRegion,
    Cut,
    CutMeaning,
    CutPlacement,
    LaneConstraint,
    ParameterDecl,
    PropertyOverlay,
    ScenarioDocument,
    SignalDecl,
    SignalKind,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.visual import BoundPolicy, TargetPolicy, VisibilityClass, lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_example_handshake_lowers_anchor_and_window_names_by_visual_order() -> None:
    document = _handshake_with_ticks_and_extra_anchor()

    result = lower_to_visual_document(document)

    assert tuple(anchor.name for anchor in result.visual_document.anchors) == ("a0", "a1", "a2")
    assert result.anchor_renames == {"req_rise": "a0", "ack_rise": "a1", "handshake": "a2"}
    assert tuple(window.name for window in result.visual_document.windows) == ("w0",)
    assert result.visual_document.windows[0].start_anchor == "a0"
    assert result.visual_document.windows[0].end_anchor == "a1"


def test_lowering_is_deterministic() -> None:
    document = _document_with_references()

    first = lower_to_visual_document(document)
    second = lower_to_visual_document(document)

    assert asdict(first) == asdict(second)


def test_lowering_is_idempotent() -> None:
    document = _document_with_references()
    first = lower_to_visual_document(document)

    second = lower_to_visual_document(first.visual_document)

    assert second.visual_document == first.visual_document
    assert asdict(second.visual_document) == asdict(first.visual_document)


def test_lowered_dsl_round_trips_canonically() -> None:
    lowered = lower_to_visual_document(_document_with_references()).visual_document

    parsed_once = parse_diagram(emit_timing_dsl(lowered))
    parsed_twice = parse_diagram(emit_timing_dsl(parsed_once))

    assert parsed_twice == parsed_once


def test_property_overlays_without_body_ast_drop_under_visual_and_keep_under_debug() -> None:
    document = replace(
        _handshake_with_ticks_and_extra_anchor(),
        properties=(
            PropertyOverlay(
                name="response_hint",
                body="ack rises after req",
                related_anchors=("req_rise", "ack_rise"),
                related_windows=("response_time",),
            ),
        ),
    )

    visual_result = lower_to_visual_document(document)
    debug_result = lower_to_visual_document(document, TargetPolicy.debug_keep_all())

    assert visual_result.visual_document.properties == ()
    assert visual_result.dropped_properties == ("response_hint",)
    assert visual_result.visibility.dropped_property_names == ("response_hint",)
    assert tuple(prop.name for prop in debug_result.visual_document.properties) == ("response_hint",)


def test_notes_are_dropped_under_default_policy() -> None:
    document = replace(
        _handshake_with_ticks_and_extra_anchor(),
        notes=("generated from semantic idiom",),
        properties=(
            PropertyOverlay(
                name="debug_hint",
                body="plain English hint",
                notes=("renderer-only note",),
            ),
        ),
    )

    result = lower_to_visual_document(document)

    assert result.visual_document.notes == ()
    assert result.dropped_notes == ("generated from semantic idiom", "debug_hint: renderer-only note")
    assert result.visibility.dropped_note_count == 2


def test_anchor_references_inside_window_constraint_and_cut_are_rewritten() -> None:
    document = _document_with_references()

    result = lower_to_visual_document(document)
    visual = result.visual_document

    assert visual.windows[0].start_anchor == "a0"
    assert visual.windows[0].end_anchor == "a1"
    assert visual.lane_constraints[0].start_anchor == "a0"
    assert visual.lane_constraints[0].end_anchor == "a2"
    assert visual.cuts[0].anchor == "a1"


def test_documents_that_only_differ_by_anchor_name_lower_to_same_visual_document() -> None:
    left = parse_diagram(
        """
        diagram same_visual {
          clock posedge clk;
          lane req: bit;
          lane ack: bit;
          anchor req_rise = rise(req);
          anchor ack_rise = rise(ack);
          window response = between req_rise and ack_rise [1:4];
        }
        """
    )
    right = parse_diagram(
        """
        diagram same_visual {
          clock posedge clk;
          lane req: bit;
          lane ack: bit;
          anchor t0 = rise(req);
          anchor ack_rise = rise(ack);
          window response = between t0 and ack_rise [1:4];
        }
        """
    )

    assert lower_to_visual_document(left).visual_document == lower_to_visual_document(right).visual_document


def test_same_tick_anchors_use_role_and_signal_tie_breaks() -> None:
    document = ScenarioDocument(
        name="same_tick",
        clocking=ClockingSpec(edge="posedge", signal="clk"),
        signals=(
            SignalDecl(name="req", kind=SignalKind.BIT),
            SignalDecl(name="ack", kind=SignalKind.BIT),
            SignalDecl(name="mode", kind=SignalKind.BUS, width="2"),
        ),
        anchors=(
            Anchor(name="is_high", condition=parse_dsl_condition("high(req)"), absolute_tick=2),
            Anchor(name="changed", condition=parse_dsl_condition("change(req)"), absolute_tick=2),
            Anchor(name="is_low", condition=parse_dsl_condition("low(ack)"), absolute_tick=2),
            Anchor(name="equals", condition=parse_dsl_condition("eq(mode, 2'b10)"), absolute_tick=2),
            Anchor(name="rose", condition=parse_dsl_condition("rise(ack)"), absolute_tick=2),
            Anchor(name="stable_mode", condition=parse_dsl_condition("stable(mode)"), absolute_tick=2),
            Anchor(name="fell", condition=parse_dsl_condition("fall(req)"), absolute_tick=2),
            Anchor(name="not_equal", condition=parse_dsl_condition("neq(mode, 2'b00)"), absolute_tick=2),
        ),
    )

    result = lower_to_visual_document(document)

    assert result.anchor_renames == {
        "rose": "a0",
        "fell": "a1",
        "changed": "a2",
        "equals": "a3",
        "not_equal": "a4",
        "is_high": "a5",
        "is_low": "a6",
        "stable_mode": "a7",
    }


def test_geometry_only_bound_policy_drops_parameterized_upper_bound() -> None:
    document = ScenarioDocument(
        name="parameterized_bound",
        clocking=ClockingSpec(edge="posedge", signal="clk"),
        params=(ParameterDecl(name="MAX_LAT"),),
        signals=(
            SignalDecl(name="req", kind=SignalKind.BIT),
            SignalDecl(name="ack", kind=SignalKind.BIT),
        ),
        anchors=(
            Anchor(name="req_rise", condition=parse_dsl_condition("rise(req)"), absolute_tick=1),
            Anchor(name="ack_rise", condition=parse_dsl_condition("rise(ack)"), absolute_tick=3),
        ),
        windows=(
            TimeWindow(
                name="response",
                start_anchor="req_rise",
                end_anchor="ack_rise",
                bound=TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay="MAX_LAT"),
            ),
        ),
    )
    policy = replace(TargetPolicy.visual(), bounds=BoundPolicy.GEOMETRY_ONLY)

    result = lower_to_visual_document(document, policy)

    assert result.visual_document.windows[0].bound == TimeBound(
        kind=WindowBoundKind.UNBOUNDED,
        min_delay="1",
        max_delay="$",
    )
    assert result.visibility.bound_visibility == {"w0": VisibilityClass.HIDDEN_SEMANTIC}


def _handshake_with_ticks_and_extra_anchor() -> ScenarioDocument:
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    anchor_ticks = {"req_rise": 2, "ack_rise": 4}
    anchors = tuple(
        replace(anchor, absolute_tick=anchor_ticks[anchor.name])
        for anchor in document.anchors
    ) + (
        Anchor(
            name="handshake",
            condition=parse_dsl_condition("high(req) and high(ack)"),
            absolute_tick=4,
        ),
    )
    return replace(document, anchors=anchors)


def _document_with_references() -> ScenarioDocument:
    document = _handshake_with_ticks_and_extra_anchor()
    return replace(
        document,
        lane_constraints=(
            LaneConstraint(
                name="hold_req",
                signals=("req",),
                relation="high",
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor="req_rise",
                end_anchor="handshake",
            ),
        ),
        cuts=(
            Cut(
                name="skip_after_ack",
                placement=CutPlacement.AFTER_ANCHOR,
                meaning=CutMeaning.SYMBOLIC_GAP,
                anchor="ack_rise",
            ),
        ),
    )

