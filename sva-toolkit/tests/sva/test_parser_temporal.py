from __future__ import annotations

import pytest

from sva_toolkit.sva import (
    Always,
    BinaryOperator,
    BinaryExpr,
    Eventually,
    Nexttime,
    PropertyBinary,
    PropertyBinaryOperator,
    Strong,
    Weak,
    emit_expr,
    emit_property_body,
    emit_property_text,
    parse_expr,
    parse_property_body,
    parse_property_text,
)
from sva_toolkit.sva.ast import ImplicationProperty, MultiEventClocking
from sva_toolkit.sva.diagnostics import ParserDiagnostics
from sva_toolkit.sva.errors import SvaSyntaxError


@pytest.fixture(autouse=True)
def reset_parser_diagnostics() -> None:
    ParserDiagnostics.reset()
    yield
    ParserDiagnostics.reset()


@pytest.mark.parametrize(
    ("text", "expected_type", "strong", "range_text"),
    [
        ("@(posedge clk or negedge rst_n) req |-> nexttime[2] ack", Nexttime, False, "2"),
        ("@(posedge clk) req |-> s_nexttime[1:2] ack", Nexttime, True, "1:2"),
        ("@(posedge clk) req |-> always[1:2] ack", Always, False, "1:2"),
        ("@(posedge clk) req |-> s_always[1:$] ack", Always, True, "1:$"),
        ("@(posedge clk) req |-> eventually[0:3] ack", Eventually, False, "0:3"),
        ("@(posedge clk) req |-> s_eventually[1:2] ack", Eventually, True, "1:2"),
    ],
)
def test_temporal_property_operators_round_trip(
    text: str,
    expected_type: type[Nexttime | Always | Eventually],
    strong: bool,
    range_text: str,
) -> None:
    spec = parse_property_text(text)

    assert isinstance(spec.body, ImplicationProperty)
    assert isinstance(spec.body.consequent, expected_type)
    assert spec.body.consequent.strong is strong
    assert emit_property_text(spec) == text
    assert ParserDiagnostics.opaque_count() == 0

    cycle_range = getattr(spec.body.consequent, "cycle_delay", None) or getattr(spec.body.consequent, "cycle_range")
    if ":" in range_text:
        minimum, maximum = range_text.split(":", 1)
        assert cycle_range.minimum.text == minimum
        if maximum == "$":
            assert cycle_range.maximum is None
            assert cycle_range.unbounded is True
        else:
            assert cycle_range.maximum is not None
            assert cycle_range.maximum.text == maximum
            assert cycle_range.unbounded is False
    else:
        assert cycle_range.minimum.text == range_text
        assert cycle_range.maximum is None
        assert cycle_range.unbounded is False


def test_multi_edge_clocking_parses_to_multi_event_node() -> None:
    spec = parse_property_text("@(posedge clk or negedge rst_n) req |-> ack")

    assert isinstance(spec.clocking, MultiEventClocking)
    assert [event.edge.value for event in spec.clocking.events] == ["posedge", "negedge"]
    assert [event.signal.name for event in spec.clocking.events] == ["clk", "rst_n"]
    assert emit_property_text(spec) == "@(posedge clk or negedge rst_n) req |-> ack"
    assert ParserDiagnostics.snapshot()["opaque_property"] == 0


@pytest.mark.parametrize(
    ("text", "expected_op"),
    [
        ("req implies ack", PropertyBinaryOperator.IMPLIES),
        ("req iff ack", PropertyBinaryOperator.IFF),
        ("req s_until ack", PropertyBinaryOperator.S_UNTIL),
        ("req s_until_with ack", PropertyBinaryOperator.S_UNTIL_WITH),
    ],
)
def test_property_binary_variants_round_trip(text: str, expected_op: PropertyBinaryOperator) -> None:
    node = parse_property_body(text)

    assert isinstance(node, PropertyBinary)
    assert node.op is expected_op
    assert emit_property_body(node) == text
    assert ParserDiagnostics.opaque_count() == 0


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("strong(req)", Strong),
        ("weak(req or ack)", Weak),
    ],
)
def test_strength_wrappers_round_trip(text: str, expected_type: type[Strong | Weak]) -> None:
    node = parse_property_body(text)

    assert isinstance(node, expected_type)
    assert emit_property_body(node) == text
    assert ParserDiagnostics.opaque_count() == 0


def test_inside_and_dist_round_trip() -> None:
    inside = parse_expr("a inside {1, 2}")
    dist = parse_expr("a dist {1 := 2, 3 :/ 4}")

    assert emit_expr(inside) == "a inside {1, 2}"
    assert emit_expr(dist) == "a dist {1 := 2, 3 :/ 4}"
    assert ParserDiagnostics.opaque_count() == 0


def test_expression_implies_and_iff_round_trip() -> None:
    expr = parse_expr("a -> b <-> c")

    assert isinstance(expr, BinaryExpr)
    assert expr.op is BinaryOperator.IFF
    assert emit_expr(expr) == "a -> b <-> c"


def test_always_without_clocking_raises_syntax_error() -> None:
    with pytest.raises(SvaSyntaxError, match="requires a clocking event"):
        parse_property_body("always ack")
