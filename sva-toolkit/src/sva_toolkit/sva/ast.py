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
    RESTRICT = "restrict"
    EXPECT = "expect"


class ClockEdge(str, Enum):
    POSEDGE = "posedge"
    NEGEDGE = "negedge"
    EDGE = "edge"


class UnaryOperator(str, Enum):
    LOGICAL_NOT = "!"
    BITWISE_NOT = "~"
    UNARY_PLUS = "+"
    UNARY_MINUS = "-"


class BinaryOperator(str, Enum):
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"
    IMPLIES = "->"
    IFF = "<->"
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
    S_UNTIL = "s_until"
    S_UNTIL_WITH = "s_until_with"
    IMPLIES = "implies"
    IFF = "iff"


class RepeatOperator(str, Enum):
    CONSECUTIVE = "[*]"
    NON_CONSECUTIVE = "[=]"
    GOTO = "[->]"
    ONE_OR_MORE = "[+]"


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
class Inside(Node):
    expr: "ExprNode"
    items: tuple["ExprNode", ...] = ()


@dataclass(frozen=True, slots=True)
class DistItem(Node):
    value: "ExprNode"
    weight: "ExprNode | None" = None
    per_item: bool = False


@dataclass(frozen=True, slots=True)
class Dist(Node):
    expr: "ExprNode"
    items: tuple[DistItem, ...] = ()


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
    | Inside
    | Dist
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
class Within(Node):
    left: "SequenceNode"
    right: "SequenceNode"


@dataclass(frozen=True, slots=True)
class Matched(Node):
    sequence: "SequenceNode"


@dataclass(frozen=True, slots=True)
class Ended(Node):
    sequence: "SequenceNode"


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
class MultiEventClocking(Node):
    events: tuple["ClockingEvent", ...] = ()


@dataclass(frozen=True, slots=True)
class ClockingSequence(Node):
    clocking: "ClockingEvent | MultiEventClocking"
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
    | Within
    | Matched
    | Ended
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
class Nexttime(Node):
    operand: "PropertyNode"
    cycle_delay: CycleRange | None = None
    strong: bool = False


@dataclass(frozen=True, slots=True)
class Always(Node):
    operand: "PropertyNode"
    cycle_range: CycleRange | None = None
    strong: bool = False


@dataclass(frozen=True, slots=True)
class Eventually(Node):
    operand: "PropertyNode"
    cycle_range: CycleRange | None = None
    strong: bool = False


@dataclass(frozen=True, slots=True)
class Strong(Node):
    operand: "SequenceNode | PropertyNode"


@dataclass(frozen=True, slots=True)
class Weak(Node):
    operand: "SequenceNode | PropertyNode"


@dataclass(frozen=True, slots=True)
class Restrict(Node):
    operand: "PropertyNode"


@dataclass(frozen=True, slots=True)
class Expect(Node):
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
    | Nexttime
    | Always
    | Eventually
    | Strong
    | Weak
    | Restrict
    | Expect
    | ImplicationProperty
    | PropertyBinary
    | IfElseProperty
    | ControlProperty
    | OpaqueProperty
)


@dataclass(frozen=True, slots=True)
class SequenceDecl(Node):
    name: str
    formals: tuple["PropertyFormal", ...] = ()
    local_vars: tuple["LocalVarDecl", ...] = ()
    body: "SequenceNode | None" = None


@dataclass(frozen=True, slots=True)
class CheckerDecl(Node):
    name: str
    formals: tuple["PropertyFormal", ...] = ()
    items: tuple[Node, ...] = ()


@dataclass(frozen=True, slots=True)
class Bind(Node):
    target: ExprNode | None = None
    instance_name: str | None = None
    checker_name: str | None = None
    args: tuple[ExprNode, ...] = ()


@dataclass(frozen=True, slots=True)
class ClockingDecl(Node):
    name: str | None = None
    default: bool = False
    event: "ClockingEvent | MultiEventClocking | None" = None


@dataclass(frozen=True, slots=True)
class LetDecl(Node):
    name: str
    formals: tuple["PropertyFormal", ...] = ()
    body: "ExprNode | SequenceNode | PropertyNode | None" = None


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
    clocking: ClockingEvent | MultiEventClocking | None = None
    disable_iff: ExprNode | None = None


__all__ = [
    "Always",
    "BinaryExpr",
    "BinaryOperator",
    "Bind",
    "CallExpr",
    "CheckerDecl",
    "ClockEdge",
    "ClockingDecl",
    "ClockingEvent",
    "ControlOperator",
    "ControlProperty",
    "CycleRange",
    "DelaySequence",
    "Dist",
    "DistItem",
    "Ended",
    "Eventually",
    "Expect",
    "ExprNode",
    "ExprSequence",
    "FirstMatchSequence",
    "Identifier",
    "IfElseProperty",
    "ImplicationOperator",
    "ImplicationProperty",
    "Inside",
    "LetDecl",
    "Literal",
    "LocalVarDecl",
    "Matched",
    "MultiEventClocking",
    "Nexttime",
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
    "Restrict",
    "SequenceBinary",
    "SequenceBinaryOperator",
    "SequenceDecl",
    "SequenceEndedExpr",
    "SequenceNode",
    "SourceSpan",
    "StatementKind",
    "Strong",
    "TernaryExpr",
    "UnaryExpr",
    "UnaryOperator",
    "UnaryProperty",
    "Weak",
    "Within",
]
