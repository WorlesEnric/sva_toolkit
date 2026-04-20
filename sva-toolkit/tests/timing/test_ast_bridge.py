from __future__ import annotations

from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
from sva_toolkit.timing.bridge.from_sva import extract_sva_scenario
from sva_toolkit.timing.core.conditions import parse_dsl_condition
from sva_toolkit.timing.core.scenario import (
    Anchor,
    ClockingSpec,
    ParameterDecl,
    PropertyOverlay,
    ScenarioDocument,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.frontend.parser import parse_diagram


def test_extract_sva_scenario_populates_ast_fields_and_ast_driven_metadata() -> None:
    document, report = extract_sva_scenario(
        "property req_to_mode; @(posedge clk) disable iff (!rst_n) req |-> (state == MODE); endproperty"
    )

    assert document.name == "req_to_mode"
    assert report.worst_status().value == "exact"
    assert tuple(param.name for param in document.params) == ("MODE",)
    assert {signal.name for signal in document.signals} == {"req", "state"}
    assert document.clocking.disable_iff == "!rst_n"
    assert document.clocking.disable_iff_ast is not None
    assert document.properties[0].body == "req |-> state == MODE"
    assert document.properties[0].body_ast is not None


def test_extract_sva_scenario_keeps_uppercase_signal_names_as_signals() -> None:
    document, report = extract_sva_scenario(
        "property aw_ready; @(posedge clk) $rose(AWVALID) |-> ##[0:7] (AWVALID && AWREADY); endproperty"
    )

    assert report.worst_status().value == "exact"
    assert document.params == ()
    assert {signal.name for signal in document.signals} == {"AWREADY", "AWVALID"}


def test_emit_parameterized_sva_ignores_explicit_property_blocks() -> None:
    document = ScenarioDocument(
        name="ast_emit",
        clocking=ClockingSpec(
            edge="posedge",
            signal="clk",
            disable_iff="!rst_n",
        ),
        params=(ParameterDecl(name="MODE"),),
        anchors=(
            Anchor(name="start", condition=parse_dsl_condition("high(req)")),
            Anchor(name="done", condition=parse_dsl_condition("eq(state, MODE)")),
        ),
        windows=(
            TimeWindow(
                name="req_to_done",
                start_anchor="start",
                end_anchor="done",
                bound=TimeBound(
                    kind=WindowBoundKind.EXACT,
                    min_delay="1",
                    max_delay="1",
                ),
            ),
        ),
        properties=(
            PropertyOverlay(
                name="p_ast",
                body="start |-> ##1 done",
            ),
        ),
    )

    emitted = emit_parameterized_sva(document)

    assert "property p_ast" not in emitted
    assert "disable iff (!rst_n)" in emitted
    assert "property req_to_done(int MODE);" in emitted
    assert "req |-> ##1 state == MODE;" in emitted


def test_emit_parameterized_sva_derives_window_and_hold_properties_from_show_constraints() -> None:
    document = parse_diagram(
        """
        diagram hold_until_ready {
          clock posedge clk;
          param MAX_WAIT;

          lane valid: bit;
          lane ready: bit;
          lane data: bus[8];

          anchor asserted = rise(valid);
          anchor handshake = high(valid) and high(ready);

          window ready_window = between asserted and handshake [0:MAX_WAIT];

          show high(valid) from asserted until handshake;
          show stable(data) from asserted until handshake;
        }
        """
    )

    emitted = emit_parameterized_sva(document)

    assert "property ready_window(int MAX_WAIT);" in emitted
    assert "$rose(valid) |-> ##[0:MAX_WAIT] (valid && ready);" in emitted
    assert "high_valid_from_asserted_until_handshake" in emitted
    assert "$rose(valid) |-> valid until_with (valid && ready);" in emitted
    assert "stable_data_from_asserted_until_handshake" in emitted
    assert "$rose(valid) |-> $stable(data) until_with (valid && ready);" in emitted


def test_emit_parameterized_sva_derives_before_constraint_property() -> None:
    document = parse_diagram(
        """
        diagram ordering_guard {
          clock posedge clk;

          lane awvalid: bit;
          lane awready: bit;
          lane wvalid: bit;

          anchor aw_handshake = high(awvalid) and high(awready);

          show low(wvalid) before aw_handshake;
        }
        """
    )

    emitted = emit_parameterized_sva(document)

    assert "low_wvalid_before_aw_handshake" in emitted
    assert "!wvalid until (awvalid && awready);" in emitted
    assert "property ordering_guard;" not in emitted
