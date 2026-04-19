from __future__ import annotations

from sva_toolkit.sva.ast import (
    BinaryExpr,
    BinaryOperator,
    CallExpr,
    ExprNode,
    Identifier,
    Literal,
    OpaqueExpr,
    UnaryExpr,
    UnaryOperator,
)
from sva_toolkit.sva.emitter import emit_expr
from sva_toolkit.sva.parser import parse_expr
from sva_toolkit.timing.core.conditions import Condition, Predicate


_CALL_TO_OP = {
    "$rose": "rise",
    "$fell": "fall",
    "$stable": "stable",
    "$changed": "change",
    "$isunknown": "unknown",
}

_OP_TO_CALL = {value: key for key, value in _CALL_TO_OP.items()}
_OP_TO_CALL.update({"rose": "$rose", "fell": "$fell"})


def expr_to_condition(expr: ExprNode) -> Condition:
    if isinstance(expr, Identifier):
        return Condition(kind="predicate", predicate=Predicate(op="high", signal=expr.name))
    if isinstance(expr, UnaryExpr):
        if expr.op is UnaryOperator.LOGICAL_NOT and isinstance(expr.operand, Identifier):
            return Condition(kind="predicate", predicate=Predicate(op="low", signal=expr.operand.name))
        if expr.op is UnaryOperator.LOGICAL_NOT:
            return Condition(kind="not", items=(expr_to_condition(expr.operand),))
    if isinstance(expr, CallExpr):
        if expr.name in _CALL_TO_OP and len(expr.args) == 1 and isinstance(expr.args[0], Identifier):
            return Condition(kind="predicate", predicate=Predicate(op=_CALL_TO_OP[expr.name], signal=expr.args[0].name))
        if expr.name == "$past" and expr.args:
            signal = emit_expr(expr.args[0])
            args = tuple(emit_expr(arg) for arg in expr.args)
            return Condition(kind="predicate", predicate=Predicate(op="past", signal=signal, args=args))
    if isinstance(expr, BinaryExpr):
        if expr.op is BinaryOperator.LOGICAL_AND:
            return _flatten_condition("all", expr)
        if expr.op is BinaryOperator.LOGICAL_OR:
            return _flatten_condition("any", expr)
        if isinstance(expr.left, Identifier) and expr.op in {BinaryOperator.EQ, BinaryOperator.NE}:
            return Condition(
                kind="predicate",
                predicate=Predicate(
                    op="eq" if expr.op is BinaryOperator.EQ else "neq",
                    signal=expr.left.name,
                    value=emit_expr(expr.right),
                ),
            )
    return Condition(kind="raw", text=emit_expr(expr))


def condition_to_expr(condition: Condition) -> ExprNode:
    if condition.kind == "predicate" and condition.predicate is not None:
        predicate = condition.predicate
        if predicate.op == "high" and predicate.signal is not None:
            return Identifier(name=predicate.signal)
        if predicate.op == "low" and predicate.signal is not None:
            return UnaryExpr(op=UnaryOperator.LOGICAL_NOT, operand=Identifier(name=predicate.signal))
        if predicate.op in _OP_TO_CALL and predicate.signal is not None:
            return CallExpr(name=_OP_TO_CALL[predicate.op], args=(Identifier(name=predicate.signal),))
        if predicate.op in {"eq", "neq"} and predicate.signal is not None and predicate.value is not None:
            return BinaryExpr(
                left=Identifier(name=predicate.signal),
                op=BinaryOperator.EQ if predicate.op == "eq" else BinaryOperator.NE,
                right=_parse_text_as_expr(predicate.value),
            )
        if predicate.op == "past":
            arg_texts = predicate.args or ((predicate.signal or ""),)
            return CallExpr(name="$past", args=tuple(_parse_text_as_expr(arg) for arg in arg_texts))
        return OpaqueExpr(text=predicate.text or "")
    if condition.kind == "all":
        return _combine_boolean(condition.items, BinaryOperator.LOGICAL_AND)
    if condition.kind == "any":
        return _combine_boolean(condition.items, BinaryOperator.LOGICAL_OR)
    if condition.kind == "not" and condition.items:
        return UnaryExpr(op=UnaryOperator.LOGICAL_NOT, operand=condition_to_expr(condition.items[0]))
    return OpaqueExpr(text=condition.text or "")


def _flatten_condition(kind: str, expr: BinaryExpr) -> Condition:
    items: list[Condition] = []
    op = BinaryOperator.LOGICAL_AND if kind == "all" else BinaryOperator.LOGICAL_OR

    def collect(node: ExprNode) -> None:
        if isinstance(node, BinaryExpr) and node.op is op:
            collect(node.left)
            collect(node.right)
            return
        items.append(expr_to_condition(node))

    collect(expr)
    return Condition(kind=kind, items=tuple(items))


def _combine_boolean(items: tuple[Condition, ...], op: BinaryOperator) -> ExprNode:
    if not items:
        return Literal(text="1'b1" if op is BinaryOperator.LOGICAL_AND else "1'b0")
    expr = condition_to_expr(items[0])
    for item in items[1:]:
        expr = BinaryExpr(left=expr, op=op, right=condition_to_expr(item))
    return expr


def _parse_text_as_expr(text: str) -> ExprNode:
    return parse_expr(text, recover=True)


__all__ = ["condition_to_expr", "expr_to_condition", "Condition", "Predicate"]
