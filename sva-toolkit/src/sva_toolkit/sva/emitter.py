from __future__ import annotations

from sva_toolkit.sva.ast import (
    Always,
    BinaryExpr,
    Bind,
    CallExpr,
    CheckerDecl,
    ClockingDecl,
    ClockingEvent,
    ClockingSequence,
    ControlProperty,
    CycleRange,
    DelaySequence,
    Dist,
    DistItem,
    Ended,
    Eventually,
    Expect,
    ExprNode,
    ExprSequence,
    FirstMatchSequence,
    Identifier,
    IfElseProperty,
    ImplicationProperty,
    Inside,
    LetDecl,
    Literal,
    LocalVarDecl,
    Matched,
    MultiEventClocking,
    Nexttime,
    Node,
    OpaqueExpr,
    OpaqueProperty,
    OpaqueSequence,
    PropertyBinary,
    PropertyBinaryOperator,
    PropertyFormal,
    PropertyNode,
    PropertySpec,
    RepeatOperator,
    RepeatSequence,
    Restrict,
    SequenceBinary,
    SequenceBinaryOperator,
    SequenceDecl,
    SequenceEndedExpr,
    SequenceMatch,
    SequenceNode,
    Strong,
    TernaryExpr,
    UnaryExpr,
    UnaryProperty,
    Weak,
    Within,
)
from sva_toolkit.sva.errors import SvaEmitError


_EXPR_PRECEDENCE = {
    TernaryExpr: 1,
    BinaryExpr: {
        "<->": 2,
        "->": 3,
        "||": 4,
        "&&": 5,
        "|": 6,
        "^": 7,
        "^~": 7,
        "~^": 7,
        "&": 8,
        "==": 9,
        "!=": 9,
        "===": 9,
        "!==": 9,
        "<": 10,
        "<=": 10,
        ">": 10,
        ">=": 10,
        "+": 11,
        "-": 11,
        "*": 12,
        "/": 12,
        "%": 12,
    },
    Inside: 10,
    Dist: 10,
    UnaryExpr: 13,
    Identifier: 14,
    Literal: 14,
    CallExpr: 14,
    SequenceEndedExpr: 14,
    OpaqueExpr: 14,
}

_SEQUENCE_PRECEDENCE = {
    Within: 1,
    SequenceBinaryOperator.OR: 2,
    SequenceBinaryOperator.AND: 3,
    SequenceBinaryOperator.INTERSECT: 4,
    SequenceBinaryOperator.THROUGHOUT: 5,
    DelaySequence: 6,
    RepeatSequence: 7,
    SequenceMatch: 8,
    ClockingSequence: 8,
    FirstMatchSequence: 8,
    Matched: 8,
    Ended: 8,
    ExprSequence: 9,
    OpaqueSequence: 9,
}

_PROPERTY_PRECEDENCE = {
    IfElseProperty: 1,
    ImplicationProperty: 2,
    PropertyBinaryOperator.IFF: 3,
    PropertyBinaryOperator.IMPLIES: 4,
    PropertyBinaryOperator.OR: 5,
    PropertyBinaryOperator.UNTIL: 6,
    PropertyBinaryOperator.UNTIL_WITH: 6,
    PropertyBinaryOperator.S_UNTIL: 6,
    PropertyBinaryOperator.S_UNTIL_WITH: 6,
    PropertyBinaryOperator.AND: 7,
    ControlProperty: 8,
    UnaryProperty: 9,
    Restrict: 9,
    Expect: 9,
    Strong: 9,
    Weak: 9,
    Nexttime: 9,
    Always: 9,
    Eventually: 9,
}

_SEQUENCE_NODE_TYPES = (
    ExprSequence,
    DelaySequence,
    RepeatSequence,
    Within,
    Matched,
    Ended,
    SequenceBinary,
    SequenceMatch,
    ClockingSequence,
    FirstMatchSequence,
    OpaqueSequence,
)


def emit_property_text(spec: PropertySpec) -> str:
    surface = emit_property_surface(spec)
    if spec.name is not None:
        header = f"property {spec.name}"
        if spec.formals:
            header = f"{header}({', '.join(_emit_formal(formal) for formal in spec.formals)})"
        local_text = " ".join(_emit_local_var(local_var) for local_var in spec.local_vars)
        pieces = [f"{header};"]
        if local_text:
            pieces.append(local_text)
        pieces.append(f"{surface};")
        pieces.append("endproperty")
        return " ".join(piece for piece in pieces if piece)
    if spec.statement_kind is not None:
        if spec.statement_kind.value == "expect":
            return f"expect ({surface})"
        if spec.statement_kind.value == "restrict":
            return f"restrict property({surface})"
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


def emit_declaration(node: Node) -> str:
    if isinstance(node, PropertySpec):
        return emit_property_text(node)
    if isinstance(node, SequenceDecl):
        header = f"sequence {node.name}"
        if node.formals:
            header = f"{header}({', '.join(_emit_formal(formal) for formal in node.formals)})"
        local_text = " ".join(_emit_local_var(local_var) for local_var in node.local_vars)
        body_text = emit_sequence(node.body) if node.body is not None else ""
        pieces = [f"{header};"]
        if local_text:
            pieces.append(local_text)
        if body_text:
            pieces.append(f"{body_text};")
        pieces.append("endsequence")
        return " ".join(piece for piece in pieces if piece)
    if isinstance(node, CheckerDecl):
        header = f"checker {node.name}"
        if node.formals:
            header = f"{header}({', '.join(_emit_formal(formal) for formal in node.formals)})"
        items = " ".join(emit_declaration(item) for item in node.items)
        pieces = [f"{header};"]
        if items:
            pieces.append(items)
        pieces.append("endchecker")
        return " ".join(piece for piece in pieces if piece)
    if isinstance(node, ClockingDecl):
        prefix = "default clocking" if node.default else "clocking"
        if node.name is not None:
            prefix = f"{prefix} {node.name}"
        if node.event is not None:
            prefix = f"{prefix} {_emit_clocking(node.event)}"
        return f"{prefix}; endclocking"
    if isinstance(node, LetDecl):
        header = f"let {node.name}"
        if node.formals:
            header = f"{header}({', '.join(_emit_formal(formal) for formal in node.formals)})"
        if node.body is None:
            return f"{header};"
        return f"{header} = {_emit_embedded_body(node.body)};"
    if isinstance(node, Bind):
        pieces = ["bind"]
        if node.target is not None:
            pieces.append(emit_expr(node.target))
        if node.instance_name is not None:
            pieces.append(node.instance_name)
        if node.checker_name is not None:
            args = ", ".join(emit_expr(arg) for arg in node.args)
            pieces.append(f"{node.checker_name}({args})")
        return " ".join(pieces) + ";"
    raise SvaEmitError(f"unsupported declaration node: {type(node).__name__}")


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
        rendered = (
            f"{_emit_expr(node.left, precedence)} {node.op.value} "
            f"{_emit_expr(node.right, precedence, is_right=True)}"
        )
    elif isinstance(node, Inside):
        precedence = _expr_precedence(node)
        rendered = f"{emit_expr(node.expr)} inside {{{', '.join(emit_expr(item) for item in node.items)}}}"
    elif isinstance(node, Dist):
        precedence = _expr_precedence(node)
        rendered = f"{emit_expr(node.expr)} dist {{{', '.join(_emit_dist_item(item) for item in node.items)}}}"
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
    elif isinstance(node, Matched):
        precedence = _sequence_precedence(node)
        rendered = f"matched({_emit_sequence(node.sequence, 0)})"
    elif isinstance(node, Ended):
        precedence = _sequence_precedence(node)
        rendered = f"ended({_emit_sequence(node.sequence, 0)})"
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
        rendered = f"{_emit_sequence(node.body, precedence)} {_emit_repeat(node)}"
    elif isinstance(node, DelaySequence):
        precedence = _sequence_precedence(node)
        right = _emit_sequence(node.right, precedence, is_right=True)
        if _is_trivial_true_sequence(node.left):
            rendered = f"{_emit_delay_range(node.delay)} {right}"
        else:
            left = _emit_sequence(node.left, precedence)
            rendered = f"{left} {_emit_delay_range(node.delay)} {right}"
    elif isinstance(node, Within):
        precedence = _sequence_precedence(node)
        rendered = f"{_emit_sequence(node.left, precedence)} within {_emit_sequence(node.right, precedence, is_right=True)}"
    elif isinstance(node, SequenceBinary):
        precedence = _sequence_precedence(node)
        rendered = (
            f"{_emit_sequence(node.left, precedence)} {node.op.value} "
            f"{_emit_sequence(node.right, precedence, is_right=True)}"
        )
    else:
        raise SvaEmitError(f"unsupported sequence node: {type(node).__name__}")

    if precedence < parent_precedence or (is_right and precedence == parent_precedence and isinstance(node, (SequenceBinary, Within))):
        return f"({rendered})"
    return rendered


def _emit_property(node: PropertyNode, parent_precedence: int, *, is_right: bool = False) -> str:
    if isinstance(node, _SEQUENCE_NODE_TYPES):
        rendered = _emit_sequence(node, 0)
        precedence = 10
    elif isinstance(node, OpaqueProperty):
        rendered = node.text
        precedence = 10
    elif isinstance(node, UnaryProperty):
        precedence = _property_precedence(node)
        rendered = f"{node.op.value} {_emit_property(node.operand, precedence)}"
    elif isinstance(node, Restrict):
        precedence = _property_precedence(node)
        rendered = f"restrict property({_emit_property(node.operand, 0)})"
    elif isinstance(node, Expect):
        precedence = _property_precedence(node)
        rendered = f"expect ({_emit_property(node.operand, 0)})"
    elif isinstance(node, Strong):
        precedence = _property_precedence(node)
        rendered = f"strong({_emit_property_or_sequence(node.operand)})"
    elif isinstance(node, Weak):
        precedence = _property_precedence(node)
        rendered = f"weak({_emit_property_or_sequence(node.operand)})"
    elif isinstance(node, Nexttime):
        precedence = _property_precedence(node)
        keyword = "s_nexttime" if node.strong else "nexttime"
        rendered = f"{keyword}{_emit_operator_range(node.cycle_delay)} {_emit_property(node.operand, precedence)}".strip()
    elif isinstance(node, Always):
        precedence = _property_precedence(node)
        keyword = "s_always" if node.strong else "always"
        rendered = f"{keyword}{_emit_operator_range(node.cycle_range)} {_emit_property(node.operand, precedence)}".strip()
    elif isinstance(node, Eventually):
        precedence = _property_precedence(node)
        keyword = "s_eventually" if node.strong else "eventually"
        rendered = f"{keyword}{_emit_operator_range(node.cycle_range)} {_emit_property(node.operand, precedence)}".strip()
    elif isinstance(node, ImplicationProperty):
        precedence = _property_precedence(node)
        antecedent = _emit_sequence(node.antecedent, 0)
        if isinstance(node.antecedent, (SequenceBinary, Within)):
            antecedent = f"({antecedent})"
        rendered = f"{antecedent} {node.op.value} {_emit_property(node.consequent, precedence, is_right=True)}"
    elif isinstance(node, PropertyBinary):
        precedence = _property_precedence(node)
        rendered = (
            f"{_emit_property(node.left, precedence)} {node.op.value} "
            f"{_emit_property(node.right, precedence, is_right=True)}"
        )
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


def _emit_local_var(local_var: LocalVarDecl) -> str:
    prefix = " ".join(local_var.qualifiers)
    rendered = f"{prefix} {local_var.type_text} {local_var.name}".strip()
    if local_var.initializer is not None:
        rendered = f"{rendered} = {emit_expr(local_var.initializer)}"
    return f"{rendered};"


def _emit_clocking(clocking: ClockingEvent | MultiEventClocking) -> str:
    if isinstance(clocking, MultiEventClocking):
        return f"@({' or '.join(_emit_clock_event(event) for event in clocking.events)})"
    return f"@({_emit_clock_event(clocking)})"


def _emit_clock_event(event: ClockingEvent) -> str:
    return f"{event.edge.value} {event.signal.name}"


def _emit_delay_range(delay: CycleRange) -> str:
    if delay.maximum is None and not delay.unbounded:
        return f"##{emit_expr(delay.minimum)}"
    return f"##[{_emit_inner_cycle_range(delay)}]"


def _emit_operator_range(cycle_range: CycleRange | None) -> str:
    if cycle_range is None:
        return ""
    return f"[{_emit_inner_cycle_range(cycle_range)}]"


def _emit_inner_cycle_range(cycle_range: CycleRange) -> str:
    if cycle_range.maximum is None and not cycle_range.unbounded:
        return emit_expr(cycle_range.minimum)
    if cycle_range.unbounded:
        return f"{emit_expr(cycle_range.minimum)}:$"
    return f"{emit_expr(cycle_range.minimum)}:{emit_expr(cycle_range.maximum)}"


def _emit_repeat(node: RepeatSequence) -> str:
    if node.op is RepeatOperator.ONE_OR_MORE:
        return "[+]"
    if node.op is RepeatOperator.CONSECUTIVE and node.count.maximum is None and node.count.unbounded and _is_zero_literal(node.count.minimum):
        return "[*]"
    prefix = {
        RepeatOperator.CONSECUTIVE: "[*",
        RepeatOperator.NON_CONSECUTIVE: "[=",
        RepeatOperator.GOTO: "[->",
    }[node.op]
    return f"{prefix}{_emit_inner_cycle_range(node.count)}]"


def _emit_dist_item(item: DistItem) -> str:
    rendered = emit_expr(item.value)
    if item.weight is None:
        return rendered
    operator = ":/" if item.per_item else ":="
    return f"{rendered} {operator} {emit_expr(item.weight)}"


def _emit_property_or_sequence(node: SequenceNode | PropertyNode) -> str:
    if isinstance(node, _SEQUENCE_NODE_TYPES):
        return emit_sequence(node)
    return emit_property_body(node)


def _emit_embedded_body(node: ExprNode | SequenceNode | PropertyNode) -> str:
    if isinstance(node, _SEQUENCE_NODE_TYPES):
        return emit_sequence(node)
    if isinstance(
        node,
        (
            OpaqueProperty,
            UnaryProperty,
            Restrict,
            Expect,
            Strong,
            Weak,
            Nexttime,
            Always,
            Eventually,
            ImplicationProperty,
            PropertyBinary,
            IfElseProperty,
            ControlProperty,
        ),
    ):
        return emit_property_body(node)
    return emit_expr(node)


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
    return _PROPERTY_PRECEDENCE.get(type(node), 10)


def _is_trivial_true_sequence(node: SequenceNode) -> bool:
    return isinstance(node, ExprSequence) and isinstance(node.expr, Literal) and node.expr.text.strip().lower() in {
        "1",
        "1'b1",
    }


def _is_zero_literal(node: ExprNode) -> bool:
    return isinstance(node, Literal) and node.text.strip() == "0"


__all__ = [
    "emit_declaration",
    "emit_expr",
    "emit_property_body",
    "emit_property_surface",
    "emit_property_text",
    "emit_sequence",
]
