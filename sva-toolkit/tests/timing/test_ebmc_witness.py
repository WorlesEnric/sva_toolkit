"""Unit and integration tests for EBMC witness synthesis.

Unit tests (no EBMC binary required) cover the trace parser, condition
evaluator, and document refiner.  The integration test is skipped when
``ebmc`` is not on PATH.
"""

from __future__ import annotations

import shutil

import pytest

from sva_toolkit.timing.bridge.ebmc_witness import (
    EbmcWitnessSynthesizer,
    WitnessTrace,
    _extract_cover_sequence,
    _is_high,
    _is_low,
    _parse_witness_text,
    _topological_sort,
    eval_condition,
    find_anchor_tick,
    refine_document_from_witness,
)
from sva_toolkit.timing.core.conditions import (
    Condition,
    all_of,
    predicate_condition,
)
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    ClockingSpec,
    ScenarioDocument,
    SignalDecl,
    SignalKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(signal_values: dict[str, list[str]]) -> WitnessTrace:
    ticks = max(len(v) for v in signal_values.values()) if signal_values else 0
    # Pad shorter lists with "0"
    padded = {k: (v + ["0"] * (ticks - len(v)))[:ticks] for k, v in signal_values.items()}
    return WitnessTrace(ticks=ticks, signal_values=padded, raw_states=ticks * 2)


_SAMPLE_EBMC_OUTPUT = """\
Trace:

Transition system state 0
----------------------------------------------------
  sva_witness.clk = 0
  sva_witness.rst_n = 1
  sva_witness.req = 0
  sva_witness.ack = 0

Transition system state 1
----------------------------------------------------
  sva_witness.clk = 1
  sva_witness.rst_n = 1
  sva_witness.req = 1
  sva_witness.ack = 0

Transition system state 2
----------------------------------------------------
  sva_witness.clk = 0
  sva_witness.rst_n = 1
  sva_witness.req = 1
  sva_witness.ack = 1

** Covered **
"""


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_witness_text_basic():
    trace = _parse_witness_text(_SAMPLE_EBMC_OUTPUT, module_prefix="sva_witness")
    assert trace is not None
    # 3 Transition system state blocks → 3 ticks
    assert trace.ticks == 3
    assert trace.signal_values["req"] == ["0", "1", "1"]
    assert trace.signal_values["ack"] == ["0", "0", "1"]


def test_parse_witness_text_empty():
    assert _parse_witness_text("", "sva_witness") is None
    assert _parse_witness_text("no state blocks here", "sva_witness") is None


def test_parse_witness_text_single_state():
    output = "Transition system state 0\n  sva_witness.clk=1  sva_witness.sig=1\n"
    trace = _parse_witness_text(output, module_prefix="sva_witness")
    assert trace is not None
    assert trace.ticks == 1


# ---------------------------------------------------------------------------
# Condition evaluator tests
# ---------------------------------------------------------------------------

def test_eval_condition_high():
    trace = _make_trace({"sig": ["0", "1", "1"]})
    cond = predicate_condition("high", "sig")
    assert not eval_condition(cond, trace, 0)
    assert eval_condition(cond, trace, 1)
    assert eval_condition(cond, trace, 2)


def test_eval_condition_low():
    trace = _make_trace({"sig": ["0", "1"]})
    cond = predicate_condition("low", "sig")
    assert eval_condition(cond, trace, 0)
    assert not eval_condition(cond, trace, 1)


def test_eval_condition_rise():
    trace = _make_trace({"sig": ["0", "1", "1"]})
    cond = predicate_condition("rise", "sig")
    assert not eval_condition(cond, trace, 0)   # no previous tick
    assert eval_condition(cond, trace, 1)        # 0 → 1
    assert not eval_condition(cond, trace, 2)    # 1 → 1 (stable)


def test_eval_condition_fall():
    trace = _make_trace({"sig": ["1", "0", "0"]})
    cond = predicate_condition("fall", "sig")
    assert not eval_condition(cond, trace, 0)
    assert eval_condition(cond, trace, 1)
    assert not eval_condition(cond, trace, 2)


def test_eval_condition_stable():
    trace = _make_trace({"sig": ["1", "1", "0"]})
    cond = predicate_condition("stable", "sig")
    # tick 0: prev defaults to "0", curr is "1" → not stable
    assert not eval_condition(cond, trace, 0)
    # tick 1: prev="1", curr="1" → stable
    assert eval_condition(cond, trace, 1)
    # tick 2: prev="1", curr="0" → not stable
    assert not eval_condition(cond, trace, 2)


def test_eval_condition_all():
    trace = _make_trace({"a": ["1", "1"], "b": ["0", "1"]})
    cond = all_of([predicate_condition("high", "a"), predicate_condition("high", "b")])
    assert not eval_condition(cond, trace, 0)  # b is low at tick 0
    assert eval_condition(cond, trace, 1)       # both high at tick 1


def test_eval_condition_unknown_signal():
    trace = _make_trace({"sig": ["1"]})
    cond = predicate_condition("high", "nonexistent")
    # Unknown signal → treated as satisfied (don't filter)
    assert eval_condition(cond, trace, 0)


def test_eval_condition_raw():
    trace = _make_trace({"sig": ["0"]})
    cond = Condition(kind="raw", text="1'b1")
    assert eval_condition(cond, trace, 0)


# ---------------------------------------------------------------------------
# find_anchor_tick
# ---------------------------------------------------------------------------

def test_find_anchor_tick_found():
    trace = _make_trace({"req": ["0", "0", "1"]})
    anchor = Anchor(
        name="a_req",
        condition=predicate_condition("high", "req"),
        role=AnchorRole.TRIGGER,
    )
    assert find_anchor_tick(anchor, trace) == 2


def test_find_anchor_tick_not_found():
    trace = _make_trace({"req": ["0", "0", "0"]})
    anchor = Anchor(
        name="a_req",
        condition=predicate_condition("high", "req"),
        role=AnchorRole.TRIGGER,
    )
    assert find_anchor_tick(anchor, trace) is None


# ---------------------------------------------------------------------------
# refine_document_from_witness
# ---------------------------------------------------------------------------

def _simple_document() -> ScenarioDocument:
    clocking = ClockingSpec(edge="posedge", signal="clk", disable_iff="!rst_n")
    anchors = (
        Anchor(name="a_req", condition=predicate_condition("high", "req"), role=AnchorRole.TRIGGER),
        Anchor(name="a_ack", condition=predicate_condition("high", "ack"), role=AnchorRole.RESPONSE),
    )
    signals = (
        SignalDecl(name="clk", kind=SignalKind.BIT),
        SignalDecl(name="req", kind=SignalKind.BIT),
        SignalDecl(name="ack", kind=SignalKind.BIT),
    )
    return ScenarioDocument(name="test", clocking=clocking, anchors=anchors, signals=signals)


def test_refine_document_from_witness_ticks():
    doc = _simple_document()
    trace = _make_trace({"clk": ["0", "1", "1"], "req": ["0", "1", "1"], "ack": ["0", "0", "1"]})
    refined = refine_document_from_witness(doc, trace)
    assert refined.ticks == 3


def test_refine_document_from_witness_anchor_ticks():
    doc = _simple_document()
    trace = _make_trace({"clk": ["0", "1", "1"], "req": ["0", "1", "1"], "ack": ["0", "0", "1"]})
    refined = refine_document_from_witness(doc, trace)
    anchor_map = {a.name: a.absolute_tick for a in refined.anchors}
    assert anchor_map["a_req"] == 1
    assert anchor_map["a_ack"] == 2


def test_refine_document_from_witness_signal_samples():
    doc = _simple_document()
    trace = _make_trace({"clk": ["0", "1", "1"], "req": ["0", "1", "1"], "ack": ["0", "0", "1"]})
    refined = refine_document_from_witness(doc, trace)
    sig_map = {s.name: s.samples for s in refined.signals}
    assert sig_map["req"] == ("0", "1", "1")
    assert sig_map["ack"] == ("0", "0", "1")
    # Clock is rebuilt as alternating 0 1 0 1...
    assert sig_map["clk"] == ("0", "1", "0")


# ---------------------------------------------------------------------------
# Helpers unit tests
# ---------------------------------------------------------------------------

def test_extract_cover_sequence_until():
    assert _extract_cover_sequence("A until B") == "##[0:$] (B)"
    assert _extract_cover_sequence("A until_with B") == "##[0:$] (B)"
    assert _extract_cover_sequence("trigger |-> A until B") == "(trigger) ##[0:$] (B)"
    assert _extract_cover_sequence("trigger |-> A until_with B") == "(trigger) ##[0:$] (B)"

def test_extract_cover_sequence_symbolic_ranges():
    assert _extract_cover_sequence("A |-> ##PARAM B") == "(A) ##$ (B)"
    assert _extract_cover_sequence("A |-> ##[1:PARAM] B") == "(A) ##[1:$] (B)"
    assert _extract_cover_sequence("A |-> ##[MIN:MAX] B") == "(A) ##[$:$] (B)"
    assert _extract_cover_sequence("A |-> ##[0:8] B") == "(A) ##[0:8] (B)"

def test_build_module_symbolic_ports():
    from sva_toolkit.formal.model import FormalProperty
    prop = FormalProperty(
        body="$rose(VALID) |-> ##[0:READY_MAX] (VALID && READY)",
        clock_name="clk",
        clock_edge="posedge",
        reset_expr="!rst_n",
        signals=frozenset({"VALID", "READY"}),
    )
    synth = EbmcWitnessSynthesizer()
    module_text = synth._build_module([prop], {})
    # Check that READY_MAX is included as a port despite not being in signals
    assert "input wire [7:0] READY_MAX" in module_text
    assert "input wire [7:0] VALID" in module_text
    assert "input wire [7:0] READY" in module_text


def test_topological_sort_chain():
    order = _topological_sort(3, [(0, 1), (1, 2)])
    assert order == [0, 1, 2]


def test_topological_sort_no_edges():
    order = _topological_sort(3, [])
    assert set(order) == {0, 1, 2}


def test_is_high_low():
    assert _is_high("1")
    assert _is_high("1'b1")
    assert _is_high("0x1")
    assert not _is_high("0")
    assert not _is_high("1'b0")
    assert _is_low("0")
    assert not _is_low("1")


# ---------------------------------------------------------------------------
# Integration test (requires ebmc on PATH)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("ebmc") is None, reason="ebmc not on PATH")
def test_synthesize_simple_handshake():
    """End-to-end: req |-> ##1 ack produces a witness where ack follows req."""
    from sva_toolkit.formal.model import FormalProperty

    prop = FormalProperty(
        body="req |-> ##1 ack",
        clock_name="clk",
        clock_edge="posedge",
        reset_expr="!rst_n",
        signals=frozenset({"req", "ack"}),
    )
    synth = EbmcWitnessSynthesizer(depth=8, timeout=30)
    trace = synth.synthesize(prop)
    assert trace is not None, "EBMC should find a witness for req |-> ##1 ack"
    assert trace.ticks > 0
    assert "req" in trace.signal_values
    assert "ack" in trace.signal_values
    req_vals = trace.signal_values["req"]
    ack_vals = trace.signal_values["ack"]
    # Find tick where req is high — ack should be high at the next tick
    found = False
    for t, r in enumerate(req_vals):
        if _is_high(r) and t + 1 < len(ack_vals) and _is_high(ack_vals[t + 1]):
            found = True
            break
    assert found, f"No req→ack sequence found in trace: req={req_vals} ack={ack_vals}"
