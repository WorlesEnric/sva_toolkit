from __future__ import annotations

from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

import sva_toolkit.timing as timing
import sva_toolkit.timing.render as timing_render
from sva_toolkit.timing.core.conditions import parse_dsl_condition
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
    PropertyOverlay,
    ScenarioDocument,
    SignalDecl,
    SignalKind,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.projection import WaveDromScenarioView, build_wavedrom_view, can_render_with_wavedrom
from sva_toolkit.timing.projection.wavedrom_view import evaluate_condition
import sva_toolkit.timing.render.svg as svg_backend
from sva_toolkit.timing.render.svg import render_diagram_svg


def test_render_package_exports_svg_and_png_renderers() -> None:
    assert callable(timing_render.render_diagram_svg)
    assert callable(timing_render.render_diagram_png)


def test_timing_package_exports_svg_and_png_renderers() -> None:
    assert callable(timing.render_diagram_svg)
    assert callable(timing.render_diagram_png)


def test_render_diagram_svg_wraps_output_in_svg_element() -> None:
    svg = render_diagram_svg(_empty_document())

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")


def test_render_diagram_svg_includes_document_name() -> None:
    svg = render_diagram_svg(_empty_document(name="arbiter_flow"))

    assert "arbiter_flow" in svg


def test_render_diagram_svg_includes_anchor_names() -> None:
    svg = render_diagram_svg(_sampleless_wavedrom_document())

    assert "start" in svg
    assert "grant" in svg
    assert "settle" in svg


def test_render_diagram_svg_includes_rule_labels() -> None:
    svg = render_diagram_svg(_sampleless_wavedrom_document())

    assert "[1:2]" in svg


def test_render_diagram_svg_includes_rule_summary_lines() -> None:
    svg = render_diagram_svg(_sampleless_wavedrom_document())

    assert "RULES" in svg
    # Summary lines now show descriptions, not auto-generated names
    assert "grant" in svg
    assert "settle" in svg


def test_render_diagram_svg_includes_cut_labels() -> None:
    svg = render_diagram_svg(_sampleless_wavedrom_document())

    assert "gap" in svg


def test_render_diagram_svg_handles_empty_document() -> None:
    svg = render_diagram_svg(_empty_document())

    assert svg.startswith("<svg")
    assert "empty" in svg
    assert svg.endswith("</svg>")


def test_render_diagram_svg_synthesizes_samples_for_sampleless_documents() -> None:
    svg = render_diagram_svg(_sampleless_wavedrom_document())

    assert svg.startswith("<svg")
    assert "sampleless_wave" in svg
    assert "start" in svg
    assert "grant" in svg
    assert "settle" in svg
    assert "[1:2]" in svg
    assert "gap" in svg
    assert "3'b101" in svg
    # Summary descriptions appear instead of auto-generated property names
    assert "from grant" in svg or "grant" in svg
    assert "settle" in svg
    assert svg.endswith("</svg>")


def test_can_render_with_wavedrom_returns_true_without_ticks() -> None:
    document = replace(_concrete_wavedrom_document(), ticks=None)

    assert can_render_with_wavedrom(document) is True


def test_can_render_with_wavedrom_returns_true_with_cuts() -> None:
    document = replace(
        _concrete_wavedrom_document(),
        cuts=(
            Cut(
                name="gap",
                placement=CutPlacement.AFTER_ANCHOR,
                meaning=CutMeaning.SYMBOLIC_GAP,
                anchor="start",
            ),
        ),
    )

    assert can_render_with_wavedrom(document) is True


def test_can_render_with_wavedrom_returns_true_for_fully_sampled_documents() -> None:
    assert can_render_with_wavedrom(_concrete_wavedrom_document()) is True


@pytest.mark.parametrize(
    ("expression", "tick", "expected"),
    [
        ("high(req)", 1, True),
        ("low(req)", 0, True),
        ("rise(req)", 1, True),
        ("fall(req)", 4, True),
        ("stable(req)", 2, True),
        ("change(gnt)", 2, True),
        ("eq(state, 3'b101)", 3, True),
        ("neq(state, 3'b000)", 3, True),
        ("high(req) and high(gnt)", 2, True),
        ("high(req) and not high(gnt)", 1, True),
        ("high(req) or high(gnt)", 0, False),
    ],
)
def test_evaluate_condition_supports_concrete_predicates(expression: str, tick: int, expected: bool) -> None:
    document = _concrete_wavedrom_document()
    signals = {signal.name: signal.samples for signal in document.signals}

    assert evaluate_condition(parse_dsl_condition(expression), signals, tick) is expected


def test_build_wavedrom_view_produces_anchor_occurrences() -> None:
    view = build_wavedrom_view(_concrete_wavedrom_document())

    assert isinstance(view, WaveDromScenarioView)
    assert [(occ.anchor_name, occ.tick, occ.placement) for occ in view.anchor_occurrences] == [
        ("start", 1, "boundary"),
        ("grant", 2, "center"),
        ("grant", 3, "center"),
        ("settle", 3, "center"),
        ("settle", 4, "center"),
    ]


def test_build_wavedrom_view_produces_response_spans() -> None:
    view = build_wavedrom_view(_concrete_wavedrom_document())

    assert [(span.name, span.trigger_tick, span.response_tick, span.delay_text) for span in view.response_spans] == [
        ("grant_response", 1, 2, "[1:2]"),
    ]


def test_render_diagram_svg_renders_wavedrom_shell_for_concrete_documents() -> None:
    svg = render_diagram_svg(_concrete_wavedrom_document())
    root = ET.fromstring(svg)
    title = root.find("{http://www.w3.org/2000/svg}text[@class='timing-title']")
    clocking = root.find("{http://www.w3.org/2000/svg}text[@class='timing-meta']")

    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "<path" in svg or "<polyline" in svg
    assert title is not None
    assert title.text == "concrete_wave"
    assert clocking is not None
    assert clocking.text is not None
    assert "@(posedge clk) disable iff (!rst_n)" in clocking.text


def test_render_diagram_svg_renders_clock_signal_first() -> None:
    svg = render_diagram_svg(_concrete_wavedrom_document())
    root = ET.fromstring(svg)
    labels = [
        element.text.strip()
        for element in root.iter()
        if element.text and element.text.strip() in {"clk", "req", "gnt", "state[3]"}
    ]

    assert labels[:4] == ["clk", "req", "gnt", "state[3]"]


def test_render_diagram_svg_includes_bus_values_as_text() -> None:
    svg = render_diagram_svg(_concrete_wavedrom_document())

    assert "3'b000" in svg
    assert "3'b001" in svg
    assert "3'b101" in svg


def test_render_diagram_svg_builds_and_renders_wavedrom(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = "<svg>wavedrom-sentinel</svg>"
    document = _empty_document()

    def _build(diagram: ScenarioDocument) -> str:
        assert diagram is document
        return "view"

    def _render(view: str) -> str:
        assert view == "view"
        return sentinel

    monkeypatch.setattr(svg_backend, "build_wavedrom_view", _build)
    monkeypatch.setattr(svg_backend, "render_wavedrom_svg", _render)

    assert svg_backend.render_diagram_svg(document) == sentinel


def _clocking() -> ClockingSpec:
    return ClockingSpec(edge="posedge", signal="clk")


def _empty_document(*, name: str = "empty") -> ScenarioDocument:
    return ScenarioDocument(name=name, clocking=_clocking())


def _sampleless_wavedrom_document() -> ScenarioDocument:
    return ScenarioDocument(
        name="sampleless_wave",
        clocking=ClockingSpec(edge="posedge", signal="clk", disable_iff="!rst_n"),
        signals=(
            SignalDecl(name="req", kind=SignalKind.BIT),
            SignalDecl(name="gnt", kind=SignalKind.BIT),
            SignalDecl(name="state", kind=SignalKind.BUS, width="3"),
        ),
        anchors=(
            Anchor(name="start", condition=parse_dsl_condition("rise(req)"), role=AnchorRole.TRIGGER),
            Anchor(name="grant", condition=parse_dsl_condition("high(gnt)"), role=AnchorRole.RESPONSE),
            Anchor(name="settle", condition=parse_dsl_condition("eq(state, 3'b101)"), role=AnchorRole.STATE),
        ),
        windows=(
            TimeWindow(
                name="grant_window",
                start_anchor="start",
                end_anchor="grant",
                bound=TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay="2"),
            ),
            TimeWindow(
                name="settle_window",
                start_anchor="grant",
                end_anchor="settle",
                bound=TimeBound(kind=WindowBoundKind.OMITTED),
            ),
        ),
        cuts=(
            Cut(
                name="symbolic_gap",
                placement=CutPlacement.AFTER_ANCHOR,
                meaning=CutMeaning.SYMBOLIC_GAP,
                anchor="grant",
                label="gap",
            ),
        ),
        lane_constraints=(
            LaneConstraint(
                name="state_hold",
                signals=("state",),
                relation="eq",
                value="3'b101",
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor="grant",
                end_anchor="settle",
            ),
        ),
        properties=(
            PropertyOverlay(
                name="grant_response",
                body="start |-> ##[1:2] grant",
                related_anchors=("start", "grant"),
                related_windows=("grant_window",),
            ),
            PropertyOverlay(
                name="settle_hold",
                body="grant |-> state until_with settle",
                status=ExtractionStatus.LOSSY,
                related_anchors=("grant", "settle"),
                related_windows=("settle_window",),
                related_constraints=("state_hold",),
            ),
        ),
    )


def _concrete_wavedrom_document() -> ScenarioDocument:
    return ScenarioDocument(
        name="concrete_wave",
        clocking=ClockingSpec(edge="posedge", signal="clk", disable_iff="!rst_n"),
        signals=(
            SignalDecl(name="req", kind=SignalKind.BIT, samples=("0", "1", "1", "1", "0")),
            SignalDecl(name="gnt", kind=SignalKind.BIT, samples=("0", "0", "1", "1", "0")),
            SignalDecl(
                name="state",
                kind=SignalKind.BUS,
                width="3",
                samples=("3'b000", "3'b000", "3'b001", "3'b101", "3'b101"),
            ),
        ),
        anchors=(
            Anchor(name="start", condition=parse_dsl_condition("rise(req)"), role=AnchorRole.TRIGGER),
            Anchor(name="grant", condition=parse_dsl_condition("high(gnt)"), role=AnchorRole.RESPONSE),
            Anchor(name="settle", condition=parse_dsl_condition("eq(state, 3'b101)"), role=AnchorRole.STATE),
        ),
        windows=(
            TimeWindow(
                name="grant_window",
                start_anchor="start",
                end_anchor="grant",
                bound=TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay="2"),
            ),
            TimeWindow(
                name="settle_window",
                start_anchor="grant",
                end_anchor="settle",
                bound=TimeBound(kind=WindowBoundKind.OMITTED),
            ),
        ),
        lane_constraints=(
            LaneConstraint(
                name="state_hold_constraint",
                signals=("state",),
                relation="eq",
                value="3'b101",
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor="grant",
                end_anchor="settle",
            ),
        ),
        properties=(
            PropertyOverlay(
                name="grant_response",
                body="start |-> ##[1:2] grant",
                related_anchors=("start", "grant"),
                related_windows=("grant_window",),
            ),
            PropertyOverlay(
                name="state_hold",
                body="grant |-> eq(state, 3'b101) until_with settle",
                related_anchors=("grant", "settle"),
                related_windows=("settle_window",),
                related_constraints=("state_hold_constraint",),
            ),
            PropertyOverlay(
                name="not_before_start",
                body="!grant until start",
                related_anchors=("grant", "start"),
            ),
        ),
        ticks=5,
    )
