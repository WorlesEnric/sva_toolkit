from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    span: SourceSpan | None = None


class StatementKind(str, Enum):
    ASSERT = "assert"
    ASSUME = "assume"
    COVER = "cover"


class ClockEdge(str, Enum):
    POSEDGE = "posedge"
    NEGEDGE = "negedge"


class UnaryOperator(str, Enum):
    LOGICAL_NOT = "!"
    BITWISE_NOT = "~"
    UNARY_PLUS = "+"
    UNARY_MINUS = "-"


class BinaryOperator(str, Enum):
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"
    BITWISE_AND = "&"
    BITWISE_OR = "|"
    BITWISE_XOR = "^"
    BITWISE_XNOR = "^~"
    BITWISE_XNOR_ALT = "~^"
    EQ = "=="
    NE = "!="
    CASE_EQ = "==="
    CASE_NE = "!=="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"


class ImplicationOperator(str, Enum):
    OVERLAPPED = "|->"
    NON_OVERLAPPED = "|=>"


class SequenceBinaryOperator(str, Enum):
    AND = "and"
    OR = "or"
    INTERSECT = "intersect"
    THROUGHOUT = "throughout"


class PropertyBinaryOperator(str, Enum):
    AND = "and"
    OR = "or"
    UNTIL = "until"
    UNTIL_WITH = "until_with"


class RepeatOperator(str, Enum):
    CONSECUTIVE = "[*]"
    NON_CONSECUTIVE = "[=]"
    GOTO = "[->]"


class ControlOperator(str, Enum):
    ACCEPT_ON = "accept_on"
    REJECT_ON = "reject_on"
    SYNC_ACCEPT_ON = "sync_accept_on"
    SYNC_REJECT_ON = "sync_reject_on"


class PropertyUnaryOperator(str, Enum):
    NOT = "not"


@dataclass(frozen=True, slots=True)
class Identifier(Node):
    name: str


@dataclass(frozen=True, slots=True)
class Literal(Node):
    text: str


@dataclass(frozen=True, slots=True)
class OpaqueExpr(Node):
    text: str


@dataclass(frozen=True, slots=True)
class UnaryExpr(Node):
    op: UnaryOperator
    operand: "ExprNode"


@dataclass(frozen=True, slots=True)
class BinaryExpr(Node):
    left: "ExprNode"
    op: BinaryOperator
    right: "ExprNode"


@dataclass(frozen=True, slots=True)
class TernaryExpr(Node):
    condition: "ExprNode"
    when_true: "ExprNode"
    when_false: "ExprNode"


@dataclass(frozen=True, slots=True)
class CallExpr(Node):
    name: str
    args: tuple["ExprNode", ...] = ()


@dataclass(frozen=True, slots=True)
class CycleRange(Node):
    minimum: "ExprNode"
    maximum: "ExprNode | None" = None
    unbounded: bool = False


@dataclass(frozen=True, slots=True)
class SequenceEndedExpr(Node):
    sequence: "SequenceNode"


ExprNode: TypeAlias = (
    Identifier
    | Literal
    | OpaqueExpr
    | UnaryExpr
    | BinaryExpr
    | TernaryExpr
    | CallExpr
    | SequenceEndedExpr
)


@dataclass(frozen=True, slots=True)
class ExprSequence(Node):
    expr: ExprNode


@dataclass(frozen=True, slots=True)
class DelaySequence(Node):
    left: "SequenceNode"
    delay: CycleRange
    right: "SequenceNode"


@dataclass(frozen=True, slots=True)
class RepeatSequence(Node):
    body: "SequenceNode"
    op: RepeatOperator
    count: CycleRange


@dataclass(frozen=True, slots=True)
class SequenceBinary(Node):
    left: "SequenceNode"
    op: SequenceBinaryOperator
    right: "SequenceNode"


@dataclass(frozen=True, slots=True)
class SequenceMatch(Node):
    body: "SequenceNode"
    items: tuple["SequenceMatchItem", ...]


@dataclass(frozen=True, slots=True)
class SequenceMatchItem(Node):
    lvalue: Identifier
    rvalue: ExprNode


@dataclass(frozen=True, slots=True)
class ClockingSequence(Node):
    clocking: "ClockingEvent"
    body: "SequenceNode"


@dataclass(frozen=True, slots=True)
class FirstMatchSequence(Node):
    body: "SequenceNode"


@dataclass(frozen=True, slots=True)
class OpaqueSequence(Node):
    text: str


SequenceNode: TypeAlias = (
    ExprSequence
    | DelaySequence
    | RepeatSequence
    | SequenceBinary
    | SequenceMatch
    | ClockingSequence
    | FirstMatchSequence
    | OpaqueSequence
)


@dataclass(frozen=True, slots=True)
class UnaryProperty(Node):
    op: PropertyUnaryOperator
    operand: "PropertyNode"


@dataclass(frozen=True, slots=True)
class ImplicationProperty(Node):
    antecedent: SequenceNode
    op: ImplicationOperator
    consequent: "PropertyNode"


@dataclass(frozen=True, slots=True)
class PropertyBinary(Node):
    left: "PropertyNode"
    op: PropertyBinaryOperator
    right: "PropertyNode"


@dataclass(frozen=True, slots=True)
class IfElseProperty(Node):
    condition: ExprNode
    when_true: "PropertyNode"
    when_false: "PropertyNode | None" = None


@dataclass(frozen=True, slots=True)
class ControlProperty(Node):
    op: ControlOperator
    condition: ExprNode
    operand: "PropertyNode"


@dataclass(frozen=True, slots=True)
class OpaqueProperty(Node):
    text: str


PropertyNode: TypeAlias = (
    SequenceNode
    | UnaryProperty
    | ImplicationProperty
    | PropertyBinary
    | IfElseProperty
    | ControlProperty
    | OpaqueProperty
)


@dataclass(frozen=True, slots=True)
class PropertyFormal(Node):
    name: str
    type_text: str = "int"
    direction: str | None = None
    default: ExprNode | None = None


@dataclass(frozen=True, slots=True)
class LocalVarDecl(Node):
    name: str
    type_text: str
    qualifiers: tuple[str, ...] = ()
    initializer: ExprNode | None = None


@dataclass(frozen=True, slots=True)
class ClockingEvent(Node):
    edge: ClockEdge
    signal: Identifier


@dataclass(frozen=True, slots=True)
class PropertySpec(Node):
    body: PropertyNode
    name: str | None = None
    statement_kind: StatementKind | None = None
    formals: tuple[PropertyFormal, ...] = ()
    local_vars: tuple[LocalVarDecl, ...] = ()
    clocking: ClockingEvent | None = None
    disable_iff: ExprNode | None = None


__all__ = [
    "BinaryExpr",
    "BinaryOperator",
    "CallExpr",
    "ClockEdge",
    "ClockingEvent",
    "ControlOperator",
    "ControlProperty",
    "CycleRange",
    "DelaySequence",
    "ExprNode",
    "ExprSequence",
    "FirstMatchSequence",
    "Identifier",
    "IfElseProperty",
    "ImplicationOperator",
    "ImplicationProperty",
    "Literal",
    "LocalVarDecl",
    "Node",
    "OpaqueExpr",
    "OpaqueProperty",
    "OpaqueSequence",
    "PropertyBinary",
    "PropertyBinaryOperator",
    "PropertyFormal",
    "PropertyNode",
    "PropertySpec",
    "PropertyUnaryOperator",
    "RepeatOperator",
    "RepeatSequence",
    "SequenceBinary",
    "SequenceBinaryOperator",
    "SequenceEndedExpr",
    "SequenceNode",
    "SourceSpan",
    "StatementKind",
    "TernaryExpr",
    "UnaryExpr",
    "UnaryOperator",
    "UnaryProperty",
]
