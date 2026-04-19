from __future__ import annotations

from sva_toolkit.sva.ast import (
    BinaryExpr,
    CallExpr,
    ClockingEvent,
    ClockingSequence,
    ControlProperty,
    CycleRange,
    DelaySequence,
    ExprNode,
    ExprSequence,
    FirstMatchSequence,
    Identifier,
    IfElseProperty,
    ImplicationProperty,
    Literal,
    OpaqueExpr,
    OpaqueProperty,
    OpaqueSequence,
    PropertyBinary,
    PropertyBinaryOperator,
    PropertyFormal,
    PropertyNode,
    PropertySpec,
    RepeatSequence,
    SequenceBinary,
    SequenceBinaryOperator,
    SequenceEndedExpr,
    SequenceMatch,
    SequenceNode,
    TernaryExpr,
    UnaryExpr,
    UnaryProperty,
)
from sva_toolkit.sva.errors import SvaEmitError


_EXPR_PRECEDENCE = {
    TernaryExpr: 1,
    BinaryExpr: {
        "||": 2,
        "&&": 3,
        "|": 4,
        "^": 5,
        "^~": 5,
        "~^": 5,
        "&": 6,
        "==": 7,
        "!=": 7,
        "===": 7,
        "!==": 7,
        "<": 8,
        "<=": 8,
        ">": 8,
        ">=": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    },
    UnaryExpr: 11,
    Identifier: 12,
    Literal: 12,
    CallExpr: 12,
    SequenceEndedExpr: 12,
    OpaqueExpr: 12,
}

_SEQUENCE_PRECEDENCE = {
    SequenceBinaryOperator.OR: 1,
    SequenceBinaryOperator.AND: 2,
    SequenceBinaryOperator.INTERSECT: 3,
    SequenceBinaryOperator.THROUGHOUT: 4,
    DelaySequence: 5,
    RepeatSequence: 6,
    SequenceMatch: 7,
    ClockingSequence: 7,
    FirstMatchSequence: 7,
    ExprSequence: 8,
    OpaqueSequence: 8,
}

_PROPERTY_PRECEDENCE = {
    IfElseProperty: 1,
    ImplicationProperty: 2,
    PropertyBinaryOperator.OR: 3,
    PropertyBinaryOperator.UNTIL: 4,
    PropertyBinaryOperator.UNTIL_WITH: 4,
    PropertyBinaryOperator.AND: 5,
    ControlProperty: 6,
    UnaryProperty: 7,
}


def emit_property_text(spec: PropertySpec) -> str:
    surface = emit_property_surface(spec)
    if spec.name is not None:
        header = f"property {spec.name}"
        if spec.formals:
            header = f"{header}({', '.join(_emit_formal(formal) for formal in spec.formals)})"
        locals_text = " ".join(_emit_local_var(local_var) for local_var in spec.local_vars)
        pieces = [f"{header};"]
        if locals_text:
            pieces.append(locals_text)
        pieces.append(f"{surface};")
        pieces.append("endproperty")
        return " ".join(piece for piece in pieces if piece)
    if spec.statement_kind is not None:
        return f"{spec.statement_kind.value} property({surface})"
    return surface


def emit_property_surface(spec: PropertySpec) -> str:
    pieces: list[str] = []
    if spec.clocking is not None:
        pieces.append(_emit_clocking(spec.clocking))
    if spec.disable_iff is not None:
        pieces.append(f"disable iff ({emit_expr(spec.disable_iff)})")
    pieces.append(emit_property_body(spec.body))
    return " ".join(pieces)


def emit_property_body(node: PropertyNode) -> str:
    return _emit_property(node, 0)


def emit_sequence(node: SequenceNode) -> str:
    return _emit_sequence(node, 0)


def emit_expr(node: ExprNode) -> str:
    return _emit_expr(node, 0)


def _emit_expr(node: ExprNode, parent_precedence: int, *, is_right: bool = False) -> str:
    if isinstance(node, Identifier):
        rendered = node.name
        precedence = _expr_precedence(node)
    elif isinstance(node, Literal):
        rendered = node.text
        precedence = _expr_precedence(node)
    elif isinstance(node, OpaqueExpr):
        rendered = node.text
        precedence = _expr_precedence(node)
    elif isinstance(node, CallExpr):
        rendered = f"{node.name}({', '.join(_emit_expr(arg, 0) for arg in node.args)})"
        precedence = _expr_precedence(node)
    elif isinstance(node, SequenceEndedExpr):
        sequence_text = emit_sequence(node.sequence)
        if not (
            isinstance(node.sequence, ExprSequence)
            and isinstance(node.sequence.expr, (Identifier, Literal, CallExpr, OpaqueExpr))
        ):
            sequence_text = f"({sequence_text})"
        rendered = f"{sequence_text}.ended"
        precedence = _expr_precedence(node)
    elif isinstance(node, UnaryExpr):
        precedence = _expr_precedence(node)
        rendered = f"{node.op.value}{_emit_expr(node.operand, precedence)}"
    elif isinstance(node, BinaryExpr):
        precedence = _expr_precedence(node)
        left = _emit_expr(node.left, precedence)
        right = _emit_expr(node.right, precedence, is_right=True)
        rendered = f"{left} {node.op.value} {right}"
    elif isinstance(node, TernaryExpr):
        precedence = _expr_precedence(node)
        rendered = (
            f"{_emit_expr(node.condition, precedence)} ? "
            f"{_emit_expr(node.when_true, precedence)} : "
            f"{_emit_expr(node.when_false, precedence, is_right=True)}"
        )
    else:
        raise SvaEmitError(f"unsupported expression node: {type(node).__name__}")

    if precedence < parent_precedence or (is_right and precedence == parent_precedence and isinstance(node, BinaryExpr)):
        return f"({rendered})"
    return rendered


def _emit_sequence(node: SequenceNode, parent_precedence: int, *, is_right: bool = False) -> str:
    if isinstance(node, ExprSequence):
        rendered = emit_expr(node.expr)
        precedence = _sequence_precedence(node)
    elif isinstance(node, OpaqueSequence):
        rendered = node.text
        precedence = _sequence_precedence(node)
    elif isinstance(node, FirstMatchSequence):
        precedence = _sequence_precedence(node)
        rendered = f"first_match({_emit_sequence(node.body, 0)})"
    elif isinstance(node, ClockingSequence):
        precedence = _sequence_precedence(node)
        rendered = f"{_emit_clocking(node.clocking)} {_emit_sequence(node.body, precedence)}"
    elif isinstance(node, SequenceMatch):
        precedence = _sequence_precedence(node)
        body_text = _emit_sequence(node.body, 0)
        items_text = ", ".join(f"{item.lvalue.name} = {emit_expr(item.rvalue)}" for item in node.items)
        rendered = f"({body_text}, {items_text})"
    elif isinstance(node, RepeatSequence):
        precedence = _sequence_precedence(node)
        rendered = f"{_emit_sequence(node.body, precedence)} {_repeat_prefix(node)}{_emit_inner_cycle_range(node.count)}]"
    elif isinstance(node, DelaySequence):
        precedence = _sequence_precedence(node)
        right = _emit_sequence(node.right, precedence, is_right=True)
        if _is_trivial_true_sequence(node.left):
            rendered = f"{_emit_delay_range(node.delay)} {right}"
        else:
            left = _emit_sequence(node.left, precedence)
            rendered = f"{left} {_emit_delay_range(node.delay)} {right}"
    elif isinstance(node, SequenceBinary):
        precedence = _sequence_precedence(node)
        left = _emit_sequence(node.left, precedence)
        right = _emit_sequence(node.right, precedence, is_right=True)
        rendered = f"{left} {node.op.value} {right}"
    else:
        raise SvaEmitError(f"unsupported sequence node: {type(node).__name__}")

    if precedence < parent_precedence or (is_right and precedence == parent_precedence and isinstance(node, SequenceBinary)):
        return f"({rendered})"
    return rendered


def _emit_property(node: PropertyNode, parent_precedence: int, *, is_right: bool = False) -> str:
    if isinstance(node, (ExprSequence, DelaySequence, RepeatSequence, SequenceBinary, FirstMatchSequence, OpaqueSequence)):
        rendered = _emit_sequence(node, 0)
        precedence = 8
    elif isinstance(node, OpaqueProperty):
        rendered = node.text
        precedence = 8
    elif isinstance(node, UnaryProperty):
        precedence = _property_precedence(node)
        rendered = f"{node.op.value} {_emit_property(node.operand, precedence)}"
    elif isinstance(node, ImplicationProperty):
        precedence = _property_precedence(node)
        antecedent = _emit_sequence(node.antecedent, 0)
        if isinstance(node.antecedent, SequenceBinary) and node.antecedent.op in {
            SequenceBinaryOperator.OR,
            SequenceBinaryOperator.AND,
        }:
            antecedent = f"({antecedent})"
        rendered = f"{antecedent} {node.op.value} {_emit_property(node.consequent, precedence, is_right=True)}"
    elif isinstance(node, PropertyBinary):
        precedence = _property_precedence(node)
        left = _emit_property(node.left, precedence)
        right = _emit_property(node.right, precedence, is_right=True)
        rendered = f"{left} {node.op.value} {right}"
    elif isinstance(node, IfElseProperty):
        precedence = _property_precedence(node)
        rendered = f"if ({emit_expr(node.condition)}) {_emit_property(node.when_true, precedence)}"
        if node.when_false is not None:
            rendered = f"{rendered} else {_emit_property(node.when_false, precedence, is_right=True)}"
    elif isinstance(node, ControlProperty):
        precedence = _property_precedence(node)
        rendered = f"{node.op.value}({emit_expr(node.condition)}) {_emit_property(node.operand, precedence)}"
    else:
        raise SvaEmitError(f"unsupported property node: {type(node).__name__}")

    if precedence < parent_precedence or (is_right and precedence == parent_precedence and isinstance(node, PropertyBinary)):
        return f"({rendered})"
    return rendered


def _emit_formal(formal: PropertyFormal) -> str:
    pieces = []
    if formal.direction is not None:
        pieces.append(formal.direction)
    pieces.append(formal.type_text)
    pieces.append(formal.name)
    if formal.default is not None:
        pieces.append(f"= {emit_expr(formal.default)}")
    return " ".join(piece for piece in pieces if piece)


def _emit_local_var(local_var) -> str:
    prefix = " ".join(local_var.qualifiers)
    rendered = f"{prefix} {local_var.type_text} {local_var.name}".strip()
    if local_var.initializer is not None:
        rendered = f"{rendered} = {emit_expr(local_var.initializer)}"
    return f"{rendered};"


def _emit_clocking(clocking: ClockingEvent) -> str:
    return f"@({clocking.edge.value} {clocking.signal.name})"


def _emit_delay_range(delay: CycleRange) -> str:
    if delay.maximum is None and not delay.unbounded:
        return f"##{emit_expr(delay.minimum)}"
    return f"##[{_emit_inner_cycle_range(delay)}]"


def _emit_inner_cycle_range(cycle_range: CycleRange) -> str:
    if cycle_range.maximum is None and not cycle_range.unbounded:
        return emit_expr(cycle_range.minimum)
    if cycle_range.unbounded:
        return f"{emit_expr(cycle_range.minimum)}:$"
    return f"{emit_expr(cycle_range.minimum)}:{emit_expr(cycle_range.maximum)}"


def _repeat_prefix(node: RepeatSequence) -> str:
    return {
        "[*]": "[*",
        "[=]": "[=",
        "[->]": "[->",
    }[node.op.value]


def _expr_precedence(node: ExprNode) -> int:
    if isinstance(node, BinaryExpr):
        return _EXPR_PRECEDENCE[BinaryExpr][node.op.value]
    return _EXPR_PRECEDENCE[type(node)]


def _sequence_precedence(node: SequenceNode) -> int:
    if isinstance(node, SequenceBinary):
        return _SEQUENCE_PRECEDENCE[node.op]
    return _SEQUENCE_PRECEDENCE[type(node)]


def _property_precedence(node: PropertyNode) -> int:
    if isinstance(node, PropertyBinary):
        return _PROPERTY_PRECEDENCE[node.op]
    return _PROPERTY_PRECEDENCE.get(type(node), 8)


def _is_trivial_true_sequence(node: SequenceNode) -> bool:
    return isinstance(node, ExprSequence) and isinstance(node.expr, Literal) and node.expr.text.strip().lower() in {
        "1",
        "1'b1",
    }


__all__ = ["emit_expr", "emit_property_body", "emit_property_text", "emit_sequence"]
