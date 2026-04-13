from __future__ import annotations

import pytest

from sva_toolkit.sva import (
    BinaryExpr,
    BinaryOperator,
    CallExpr,
    Identifier,
    Literal,
    UnaryExpr,
    UnaryOperator,
    emit_expr,
)
from sva_toolkit.sva.lowerings.conditions import condition_to_expr, expr_to_condition
from sva_toolkit.timing.core.conditions import Condition, Predicate


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (Identifier(name="req"), Condition(kind="predicate", predicate=Predicate(op="high", signal="req"))),
        (
            UnaryExpr(op=UnaryOperator.LOGICAL_NOT, operand=Identifier(name="req")),
            Condition(kind="predicate", predicate=Predicate(op="low", signal="req")),
        ),
        (
            CallExpr(name="$rose", args=(Identifier(name="req"),)),
            Condition(kind="predicate", predicate=Predicate(op="rise", signal="req")),
        ),
        (
            CallExpr(name="$fell", args=(Identifier(name="req"),)),
            Condition(kind="predicate", predicate=Predicate(op="fall", signal="req")),
        ),
        (
            CallExpr(name="$stable", args=(Identifier(name="req"),)),
            Condition(kind="predicate", predicate=Predicate(op="stable", signal="req")),
        ),
        (
            CallExpr(name="$changed", args=(Identifier(name="req"),)),
            Condition(kind="predicate", predicate=Predicate(op="change", signal="req")),
        ),
        (
            CallExpr(name="$isunknown", args=(Identifier(name="req"),)),
            Condition(kind="predicate", predicate=Predicate(op="unknown", signal="req")),
        ),
        (
            BinaryExpr(left=Identifier(name="a"), op=BinaryOperator.EQ, right=Literal(text="3")),
            Condition(kind="predicate", predicate=Predicate(op="eq", signal="a", value="3")),
        ),
        (
            BinaryExpr(left=Identifier(name="a"), op=BinaryOperator.NE, right=Literal(text="3")),
            Condition(kind="predicate", predicate=Predicate(op="neq", signal="a", value="3")),
        ),
    ],
)
def test_expr_to_condition_handles_direct_mappings(expr, expected: Condition) -> None:
    assert expr_to_condition(expr) == expected


def test_expr_to_condition_handles_past_and_boolean_composition() -> None:
    past_expr = CallExpr(name="$past", args=(Identifier(name="data"), Literal(text="2")))
    boolean_expr = BinaryExpr(
        left=Identifier(name="req"),
        op=BinaryOperator.LOGICAL_AND,
        right=BinaryExpr(
            left=CallExpr(name="$rose", args=(Identifier(name="ack"),)),
            op=BinaryOperator.LOGICAL_OR,
            right=Literal(text="1'b0"),
        ),
    )

    past_condition = expr_to_condition(past_expr)
    boolean_condition = expr_to_condition(boolean_expr)

    assert past_condition.kind == "predicate"
    assert past_condition.predicate == Predicate(op="past", signal="data", args=("data", "2"))
    assert boolean_condition.kind == "all"
    assert boolean_condition.items[0] == Condition(kind="predicate", predicate=Predicate(op="high", signal="req"))
    assert boolean_condition.items[1].kind == "any"
    assert boolean_condition.items[1].items[0] == Condition(
        kind="predicate",
        predicate=Predicate(op="rise", signal="ack"),
    )
    assert boolean_condition.items[1].items[1] == Condition(kind="raw", text="1'b0")


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (Condition(kind="predicate", predicate=Predicate(op="high", signal="req")), "req"),
        (Condition(kind="predicate", predicate=Predicate(op="low", signal="req")), "!req"),
        (Condition(kind="predicate", predicate=Predicate(op="rise", signal="req")), "$rose(req)"),
        (Condition(kind="predicate", predicate=Predicate(op="fall", signal="req")), "$fell(req)"),
        (Condition(kind="predicate", predicate=Predicate(op="stable", signal="req")), "$stable(req)"),
        (Condition(kind="predicate", predicate=Predicate(op="change", signal="req")), "$changed(req)"),
        (Condition(kind="predicate", predicate=Predicate(op="unknown", signal="req")), "$isunknown(req)"),
        (Condition(kind="predicate", predicate=Predicate(op="eq", signal="a", value="3")), "a == 3"),
        (Condition(kind="predicate", predicate=Predicate(op="neq", signal="a", value="3")), "a != 3"),
        (Condition(kind="predicate", predicate=Predicate(op="past", signal="data", args=("data", "2"))), "$past(data, 2)"),
        (
            Condition(
                kind="all",
                items=(
                    Condition(kind="predicate", predicate=Predicate(op="high", signal="req")),
                    Condition(kind="predicate", predicate=Predicate(op="rise", signal="ack")),
                ),
            ),
            "req && $rose(ack)",
        ),
        (
            Condition(
                kind="any",
                items=(
                    Condition(kind="predicate", predicate=Predicate(op="high", signal="req")),
                    Condition(kind="predicate", predicate=Predicate(op="rise", signal="ack")),
                ),
            ),
            "req || $rose(ack)",
        ),
        (Condition(kind="raw", text="custom_expr"), "custom_expr"),
    ],
)
def test_condition_to_expr_handles_direct_mappings(condition: Condition, expected: str) -> None:
    assert emit_expr(condition_to_expr(condition)) == expected


def test_condition_lowering_round_trip_is_stable() -> None:
    expr = BinaryExpr(
        left=Identifier(name="req"),
        op=BinaryOperator.LOGICAL_OR,
        right=CallExpr(name="$past", args=(Identifier(name="data"), Literal(text="2"))),
    )

    round_tripped = condition_to_expr(expr_to_condition(expr))

    assert emit_expr(round_tripped) == "req || $past(data, 2)"
