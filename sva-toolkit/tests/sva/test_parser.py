from __future__ import annotations

import pytest

from sva_toolkit.sva import (
    BinaryExpr,
    BinaryOperator,
    CallExpr,
    DelaySequence,
    ExprSequence,
    FirstMatchSequence,
    Identifier,
    IfElseProperty,
    ImplicationOperator,
    ImplicationProperty,
    Literal,
    OpaqueProperty,
    PropertyBinary,
    PropertyBinaryOperator,
    RepeatOperator,
    RepeatSequence,
    SequenceBinary,
    SequenceBinaryOperator,
    UnaryExpr,
    UnaryProperty,
    UnaryOperator,
    parse_expr,
    parse_property_body,
    parse_property_text,
    parse_sequence,
)


def test_parse_expr_handles_identifiers_and_literals() -> None:
    identifier = parse_expr("req")
    literal = parse_expr("4'b1010")

    assert isinstance(identifier, Identifier)
    assert identifier.name == "req"
    assert isinstance(literal, Literal)
    assert literal.text == "4'b1010"


def test_parse_expr_handles_system_function_calls() -> None:
    rose = parse_expr("$rose(req)")
    fell = parse_expr("$fell(ack)")
    past = parse_expr("$past(data, 2)")

    assert isinstance(rose, CallExpr)
    assert rose.name == "$rose"
    assert len(rose.args) == 1
    assert isinstance(rose.args[0], Identifier)
    assert rose.args[0].name == "req"

    assert isinstance(fell, CallExpr)
    assert fell.name == "$fell"
    assert len(fell.args) == 1
    assert isinstance(fell.args[0], Identifier)
    assert fell.args[0].name == "ack"

    assert isinstance(past, CallExpr)
    assert past.name == "$past"
    assert len(past.args) == 2
    assert isinstance(past.args[0], Identifier)
    assert past.args[0].name == "data"
    assert isinstance(past.args[1], Literal)
    assert past.args[1].text == "2"


def test_parse_expr_respects_binary_precedence() -> None:
    node = parse_expr("a + b * c == d && e || f")

    assert isinstance(node, BinaryExpr)
    assert node.op is BinaryOperator.LOGICAL_OR
    assert isinstance(node.right, Identifier)
    assert node.right.name == "f"

    assert isinstance(node.left, BinaryExpr)
    assert node.left.op is BinaryOperator.LOGICAL_AND
    assert isinstance(node.left.right, Identifier)
    assert node.left.right.name == "e"

    equality = node.left.left
    assert isinstance(equality, BinaryExpr)
    assert equality.op is BinaryOperator.EQ

    additive = equality.left
    assert isinstance(additive, BinaryExpr)
    assert additive.op is BinaryOperator.ADD
    assert isinstance(additive.left, Identifier)
    assert additive.left.name == "a"

    multiplicative = additive.right
    assert isinstance(multiplicative, BinaryExpr)
    assert multiplicative.op is BinaryOperator.MUL
    assert isinstance(multiplicative.left, Identifier)
    assert multiplicative.left.name == "b"
    assert isinstance(multiplicative.right, Identifier)
    assert multiplicative.right.name == "c"


def test_parse_expr_handles_ternary_expressions() -> None:
    node = parse_expr("sel ? a : b")

    assert node.condition.name == "sel"
    assert node.when_true.name == "a"
    assert node.when_false.name == "b"


def test_parse_sequence_handles_simple_delay() -> None:
    node = parse_sequence("req ##1 ack")

    assert isinstance(node, DelaySequence)
    assert isinstance(node.left, ExprSequence)
    assert isinstance(node.left.expr, Identifier)
    assert node.left.expr.name == "req"
    assert node.delay.minimum.text == "1"
    assert node.delay.maximum is None
    assert node.delay.unbounded is False
    assert isinstance(node.right, ExprSequence)
    assert isinstance(node.right.expr, Identifier)
    assert node.right.expr.name == "ack"


def test_parse_sequence_handles_range_delay() -> None:
    node = parse_sequence("req ##[1:3] ack")

    assert isinstance(node, DelaySequence)
    assert node.delay.minimum.text == "1"
    assert node.delay.maximum.text == "3"
    assert node.delay.unbounded is False


def test_parse_sequence_handles_unbounded_delay() -> None:
    node = parse_sequence("req ##[1:$] ack")

    assert isinstance(node, DelaySequence)
    assert node.delay.minimum.text == "1"
    assert node.delay.maximum is None
    assert node.delay.unbounded is True


@pytest.mark.parametrize(
    ("text", "expected_op", "minimum", "maximum", "unbounded"),
    [
        ("req [*3]", RepeatOperator.CONSECUTIVE, "3", None, False),
        ("ack [=1:5]", RepeatOperator.NON_CONSECUTIVE, "1", "5", False),
        ("data [->2]", RepeatOperator.GOTO, "2", None, False),
    ],
)
def test_parse_sequence_handles_repetitions(
    text: str,
    expected_op: RepeatOperator,
    minimum: str,
    maximum: str | None,
    unbounded: bool,
) -> None:
    node = parse_sequence(text)

    assert isinstance(node, RepeatSequence)
    assert node.op is expected_op
    assert node.count.minimum.text == minimum
    assert (node.count.maximum.text if node.count.maximum is not None else None) == maximum
    assert node.count.unbounded is unbounded


def test_parse_sequence_handles_binary_operators() -> None:
    intersect = parse_sequence("seq1 intersect seq2")
    throughout = parse_sequence("req throughout ack")

    assert isinstance(intersect, SequenceBinary)
    assert intersect.op is SequenceBinaryOperator.INTERSECT
    assert isinstance(intersect.left, ExprSequence)
    assert intersect.left.expr.name == "seq1"
    assert isinstance(intersect.right, ExprSequence)
    assert intersect.right.expr.name == "seq2"

    assert isinstance(throughout, SequenceBinary)
    assert throughout.op is SequenceBinaryOperator.THROUGHOUT
    assert isinstance(throughout.left, ExprSequence)
    assert throughout.left.expr.name == "req"
    assert isinstance(throughout.right, ExprSequence)
    assert throughout.right.expr.name == "ack"


def test_parse_sequence_handles_first_match() -> None:
    node = parse_sequence("first_match(req ##1 ack)")

    assert isinstance(node, FirstMatchSequence)
    assert isinstance(node.body, DelaySequence)


def test_parse_property_body_handles_implications() -> None:
    overlapped = parse_property_body("req |-> ack")
    non_overlapped = parse_property_body("req |=> ##1 ack")

    assert isinstance(overlapped, ImplicationProperty)
    assert overlapped.op is ImplicationOperator.OVERLAPPED
    assert isinstance(overlapped.antecedent, ExprSequence)
    assert overlapped.antecedent.expr.name == "req"
    assert isinstance(non_overlapped, ImplicationProperty)
    assert non_overlapped.op is ImplicationOperator.NON_OVERLAPPED
    assert isinstance(non_overlapped.consequent, DelaySequence)


def test_parse_property_body_handles_property_binary_operators() -> None:
    conjunction = parse_property_body("prop1 and prop2")
    disjunction = parse_property_body("prop1 or prop2")

    assert isinstance(conjunction, PropertyBinary)
    assert conjunction.op is PropertyBinaryOperator.AND
    assert isinstance(conjunction.left, ExprSequence)
    assert conjunction.left.expr.name == "prop1"
    assert isinstance(conjunction.right, ExprSequence)
    assert conjunction.right.expr.name == "prop2"

    assert isinstance(disjunction, PropertyBinary)
    assert disjunction.op is PropertyBinaryOperator.OR


def test_parse_property_body_handles_until_variants() -> None:
    until_node = parse_property_body("prop1 until prop2")
    until_with_node = parse_property_body("prop1 until_with prop2")

    assert isinstance(until_node, PropertyBinary)
    assert until_node.op is PropertyBinaryOperator.UNTIL
    assert isinstance(until_with_node, PropertyBinary)
    assert until_with_node.op is PropertyBinaryOperator.UNTIL_WITH


def test_parse_property_body_handles_not_property() -> None:
    node = parse_property_body("not req")

    assert isinstance(node, UnaryProperty)
    assert isinstance(node.operand, ExprSequence)
    assert node.operand.expr.name == "req"


def test_parse_property_body_handles_if_else() -> None:
    node = parse_property_body("if (sel) req else ack")

    assert isinstance(node, IfElseProperty)
    assert isinstance(node.condition, Identifier)
    assert node.condition.name == "sel"
    assert isinstance(node.when_true, ExprSequence)
    assert node.when_true.expr.name == "req"
    assert isinstance(node.when_false, ExprSequence)
    assert node.when_false.expr.name == "ack"


def test_parse_property_text_handles_disable_iff() -> None:
    spec = parse_property_text("disable iff (!rst_n) req")

    assert spec.name is None
    assert spec.clocking is None
    assert isinstance(spec.disable_iff, UnaryExpr)
    assert spec.disable_iff.op is UnaryOperator.LOGICAL_NOT
    assert isinstance(spec.disable_iff.operand, Identifier)
    assert spec.disable_iff.operand.name == "rst_n"
    assert isinstance(spec.body, ExprSequence)
    assert spec.body.expr.name == "req"


def test_parse_property_text_handles_clocking_event() -> None:
    spec = parse_property_text("@(posedge clk) req")

    assert spec.clocking is not None
    assert spec.clocking.edge.value == "posedge"
    assert spec.clocking.signal.name == "clk"
    assert isinstance(spec.body, ExprSequence)
    assert spec.body.expr.name == "req"


def test_parse_property_text_handles_full_named_property() -> None:
    spec = parse_property_text(
        "property p; @(posedge clk) disable iff (!rst_n) req |-> ##[1:3] ack; endproperty"
    )

    assert spec.name == "p"
    assert spec.clocking is not None
    assert spec.clocking.signal.name == "clk"
    assert spec.disable_iff is not None
    assert isinstance(spec.body, ImplicationProperty)
    assert spec.body.op is ImplicationOperator.OVERLAPPED
    assert isinstance(spec.body.consequent, DelaySequence)
    assert spec.body.consequent.delay.minimum.text == "1"
    assert spec.body.consequent.delay.maximum.text == "3"


def test_parse_property_text_handles_assert_property_statement() -> None:
    spec = parse_property_text("assert property(req |-> ack)")

    assert spec.statement_kind is not None
    assert spec.statement_kind.value == "assert"
    assert isinstance(spec.body, ImplicationProperty)
    assert spec.body.op is ImplicationOperator.OVERLAPPED


def test_parse_property_body_recover_returns_opaque_fallback() -> None:
    node = parse_property_body("req |->", recover=True)

    assert isinstance(node, OpaqueProperty)
    assert node.text == "req |->"

