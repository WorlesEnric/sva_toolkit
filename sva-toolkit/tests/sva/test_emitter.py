from __future__ import annotations

from sva_toolkit.sva import (
    BinaryExpr,
    BinaryOperator,
    CallExpr,
    ClockEdge,
    ClockingEvent,
    ControlOperator,
    ControlProperty,
    CycleRange,
    DelaySequence,
    ExprSequence,
    FirstMatchSequence,
    Identifier,
    IfElseProperty,
    ImplicationOperator,
    ImplicationProperty,
    Literal,
    LocalVarDecl,
    PropertyBinary,
    PropertyBinaryOperator,
    PropertyFormal,
    PropertySpec,
    RepeatOperator,
    RepeatSequence,
    SequenceBinary,
    SequenceBinaryOperator,
    SequenceEndedExpr,
    TernaryExpr,
    UnaryExpr,
    UnaryOperator,
    emit_expr,
    emit_property_body,
    emit_property_text,
    emit_sequence,
    parse_property_body,
)


def test_emit_expr_covers_expression_nodes() -> None:
    ended = SequenceEndedExpr(sequence=ExprSequence(expr=Identifier(name="req")))
    condition = BinaryExpr(
        left=UnaryExpr(op=UnaryOperator.LOGICAL_NOT, operand=Identifier(name="req")),
        op=BinaryOperator.LOGICAL_AND,
        right=CallExpr(name="$rose", args=(Identifier(name="ack"),)),
    )
    ternary = TernaryExpr(
        condition=condition,
        when_true=CallExpr(name="$past", args=(Identifier(name="data"), Literal(text="2"))),
        when_false=ended,
    )

    assert emit_expr(ternary) == "!req && $rose(ack) ? $past(data, 2) : req.ended"


def test_emit_sequence_covers_sequence_nodes() -> None:
    repeated = RepeatSequence(
        body=ExprSequence(expr=Identifier(name="req")),
        op=RepeatOperator.CONSECUTIVE,
        count=CycleRange(minimum=Literal(text="3")),
    )
    delayed = DelaySequence(
        left=repeated,
        delay=CycleRange(minimum=Literal(text="1"), maximum=Literal(text="3")),
        right=SequenceBinary(
            left=ExprSequence(expr=Identifier(name="ack")),
            op=SequenceBinaryOperator.INTERSECT,
            right=ExprSequence(expr=Identifier(name="done")),
        ),
    )
    node = FirstMatchSequence(body=delayed)

    assert emit_sequence(node) == "first_match(req [*3] ##[1:3] (ack intersect done))"


def test_emit_property_body_covers_property_nodes() -> None:
    implication = ImplicationProperty(
        antecedent=ExprSequence(expr=Identifier(name="req")),
        op=ImplicationOperator.NON_OVERLAPPED,
        consequent=PropertyBinary(
            left=ExprSequence(expr=Identifier(name="ack")),
            op=PropertyBinaryOperator.AND,
            right=ControlProperty(
                op=ControlOperator.ACCEPT_ON,
                condition=Identifier(name="sample_ok"),
                operand=ExprSequence(expr=Identifier(name="done")),
            ),
        ),
    )
    node = IfElseProperty(
        condition=Identifier(name="sel"),
        when_true=implication,
        when_false=ExprSequence(expr=Identifier(name="err")),
    )

    assert emit_property_body(node) == "if (sel) req |=> ack and accept_on(sample_ok) done else err"


def test_emit_property_text_covers_surface_wrappers_and_declarations() -> None:
    spec = PropertySpec(
        name="p",
        formals=(PropertyFormal(name="depth", type_text="int", direction="input", default=Literal(text="2")),),
        local_vars=(
            LocalVarDecl(
                name="seen",
                type_text="logic",
                qualifiers=("local", "var"),
                initializer=Literal(text="1'b0"),
            ),
        ),
        clocking=ClockingEvent(edge=ClockEdge.POSEDGE, signal=Identifier(name="clk")),
        disable_iff=UnaryExpr(op=UnaryOperator.LOGICAL_NOT, operand=Identifier(name="rst_n")),
        body=ImplicationProperty(
            antecedent=ExprSequence(expr=Identifier(name="req")),
            op=ImplicationOperator.OVERLAPPED,
            consequent=DelaySequence(
                left=ExprSequence(expr=Literal(text="1'b1")),
                delay=CycleRange(minimum=Literal(text="1"), maximum=Literal(text="3")),
                right=ExprSequence(expr=Identifier(name="ack")),
            ),
        ),
    )

    assert (
        emit_property_text(spec)
        == "property p(input int depth = 2); local var logic seen = 1'b0; @(posedge clk) disable iff (!rst_n) req |-> ##[1:3] ack; endproperty"
    )


def test_emit_parenthesizes_by_precedence() -> None:
    expr = BinaryExpr(
        left=BinaryExpr(
            left=Identifier(name="a"),
            op=BinaryOperator.LOGICAL_OR,
            right=Identifier(name="b"),
        ),
        op=BinaryOperator.LOGICAL_AND,
        right=Identifier(name="c"),
    )
    prop = parse_property_body("(a or b) |-> c and d")

    assert emit_expr(expr) == "(a || b) && c"
    assert emit_property_body(prop) == "(a or b) |-> c and d"


def test_emit_round_trip_preserves_property_semantics() -> None:
    text = "if (sel) req |=> ##1 ack else not done"

    assert emit_property_body(parse_property_body(text)) == text
