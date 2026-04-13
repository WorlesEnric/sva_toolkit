"""Tests for timing DSL parsing and scenario validation."""

from textwrap import dedent

import pytest

from sva_toolkit.timing import TimingDslError, emit_parameterized_sva, parse_diagram
from sva_toolkit.timing.core.scenario import ConstraintRegion, ExtractionStatus, WindowBoundKind


SAMPLE_DIAGRAM = dedent(
    """
    diagram req_ack {
      clock posedge clk;
      disable iff !rst_n;
      ticks 5;

      param LAT_MIN;
      param LAT_MAX;

      lane req: bit = 0 1 1 1 0;
      lane ack: bit = 0 0 0 1 0;
      lane state: bus[2] = IDLE WAIT WAIT DONE IDLE;

      event req_start = rise(req);
      event ack_seen = rise(ack);
      event transfer = high(req) and high(ack) same_cycle;

      rule no_early_ack:
        not ack_seen before req_start;

      rule ack_after_req:
        req_start -> after [LAT_MIN:LAT_MAX] ack_seen;

      rule req_hold_until_ack:
        high(req) from req_start until ack_seen;
    }
    """
)


SYMBOLIC_DIAGRAM = dedent(
    """
    diagram axi_wait {
      clock posedge ACLK;
      disable iff !ARESETn;

      param READY_WAIT_MAX;
      param RESP_OK;

      lane valid: bit;
      lane ready: bit;
      lane addr: bus[32];
      lane id: bus[4];
      lane resp: bus[2];

      anchor wait_start = high(valid) and low(ready);
      anchor fire = high(valid) and high(ready);
      anchor resp_ok = eq(resp, RESP_OK);

      window wait_gap = between wait_start and fire [0:READY_WAIT_MAX];
      cut prefix = before wait_start omitted;
      cut suffix = after fire compressed label "future";

      show valid = 0 before wait_start;
      show valid = 1 from wait_start until fire;
      show ready = 0 in wait_gap;
      show ready = 1 at fire;
      show stable({addr, id}) in wait_gap;
      show eq(resp, RESP_OK) at resp_ok;

      property ready_after_wait:
        wait_start |-> ##[0:READY_WAIT_MAX] fire;

      property ok_resp_after_fire:
        fire |-> ##0 resp_ok;
    }
    """
)


LOSSY_DIAGRAM = dedent(
    """
    diagram lossy_case {
      clock posedge clk;
      lane a: bit;
      lane b: bit;
      anchor t = high(a);
      anchor u = high(b);
      property p [lossy]:
        t |-> (u or ##[1:3] u);
    }
    """
)


def test_parse_v1_diagram_returns_scenario_with_legacy_adapter():
    diagram = parse_diagram(SAMPLE_DIAGRAM)

    assert diagram.name == "req_ack"
    assert diagram.clocking.edge == "posedge"
    assert diagram.clocking.signal == "clk"
    assert diagram.clocking.disable_iff == "!rst_n"
    assert diagram.ticks == 5
    assert [param.name for param in diagram.params] == ["LAT_MIN", "LAT_MAX"]
    assert [signal.name for signal in diagram.signals] == ["req", "ack", "state"]
    assert diagram.signal_map["state"].width == "2"
    assert diagram.signal_map["state"].samples[3] == "DONE"
    assert [anchor.name for anchor in diagram.anchors] == ["req_start", "ack_seen", "transfer"]
    assert diagram.legacy_diagram is not None
    assert {prop.name for prop in diagram.properties} == {"no_early_ack", "ack_after_req", "req_hold_until_ack"}


def test_parse_symbolic_v2_diagram_materializes_windows_cuts_and_show_constraints():
    diagram = parse_diagram(SYMBOLIC_DIAGRAM)

    assert diagram.legacy_diagram is None
    assert diagram.ticks is None
    assert diagram.window_map["wait_gap"].bound.kind == WindowBoundKind.RANGE
    assert diagram.window_map["wait_gap"].bound.max_delay == "READY_WAIT_MAX"
    assert {cut.name for cut in diagram.cuts} == {"prefix", "suffix"}
    assert any(constraint.region == ConstraintRegion.IN for constraint in diagram.lane_constraints)
    assert any(constraint.relation == "stable" and constraint.signals == ("addr",) for constraint in diagram.lane_constraints)
    assert any(constraint.relation == "stable" and constraint.signals == ("id",) for constraint in diagram.lane_constraints)


def test_emit_sva_for_symbolic_exact_diagram_expands_anchor_conditions():
    diagram = parse_diagram(SYMBOLIC_DIAGRAM)

    output = emit_parameterized_sva(diagram)

    assert "@(posedge ACLK) disable iff (!ARESETn)" in output
    assert "property ready_after_wait(int READY_WAIT_MAX);" in output
    assert "(valid && !ready) |-> ##[0:READY_WAIT_MAX] (valid && ready);" in output
    assert "property ok_resp_after_fire(int RESP_OK);" in output
    assert "(valid && ready) |-> ##0 (resp == RESP_OK);" in output


def test_lossy_property_requires_explicit_opt_in_for_sva_lowering():
    diagram = parse_diagram(LOSSY_DIAGRAM)

    with pytest.raises(TimingDslError, match="lossy"):
        emit_parameterized_sva(diagram)

    lowered = emit_parameterized_sva(diagram, allow_lossy=True)
    assert "lossy lowering" in lowered


def test_symbolic_window_validation_requires_declared_anchor():
    with pytest.raises(TimingDslError, match="unknown end anchor 'done'"):
        parse_diagram(
            dedent(
                """
                diagram bad {
                  clock posedge clk;
                  lane req: bit;
                  anchor start = high(req);
                  window gap = between start and done [0:3];
                  property p:
                    start |-> ##[0:3] done;
                }
                """
            )
        )


def test_invalid_unsupported_property_status_requires_note():
    with pytest.raises(TimingDslError, match="unsupported"):
        parse_diagram(
            dedent(
                """
                diagram bad_status {
                  clock posedge clk;
                  lane req: bit;
                  anchor a = high(req);
                  property p [unsupported]:
                    a;
                }
                """
            )
        )
