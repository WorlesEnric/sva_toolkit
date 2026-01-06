"""
SVA Type System - Type-directed nodes for SystemVerilog Assertions.

This module defines the core type system for SVA generation, ensuring
that temporal logic, boolean expressions, and property implications
are always correctly nested following the SVA promotion hierarchy.
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class SVAType(Enum):
    """
    @brief SVA type categories following the promotion hierarchy.
    
    The hierarchy is:
    EXPR -> BOOL -> SEQUENCE -> PROPERTY
    """
    EXPR = "EXPR"
    BOOL = "BOOL"
    SEQUENCE = "SEQUENCE"
    PROPERTY = "PROPERTY"


# Type Constants for backward compatibility
TYPE_EXPR: str = SVAType.EXPR.value
TYPE_BOOL: str = SVAType.BOOL.value
TYPE_SEQUENCE: str = SVAType.SEQUENCE.value
TYPE_PROPERTY: str = SVAType.PROPERTY.value


class SVANode:
    """
    @brief Base class for all SVA Syntax Tree nodes.
    
    Each node has a return type that determines where it can be used
    in the SVA hierarchy.
    """

    def __init__(self, return_type: str) -> None:
        """
        @brief Initialize SVA node with return type.
        @param return_type The type this node evaluates to (EXPR, BOOL, SEQUENCE, PROPERTY)
        """
        self.return_type: str = return_type

    def __str__(self) -> str:
        """
        @brief Convert node to SVA code string.
        @return String representation of the SVA node
        """
        raise NotImplementedError("Subclasses must implement __str__")


# --- BOOLEAN / EXPRESSION LAYER ---


class Signal(SVANode):
    """
    @brief Represents a signal reference in SVA.
    
    Signals are the leaf nodes of the expression tree.
    """

    def __init__(self, name: str) -> None:
        """
        @brief Initialize signal node.
        @param name The signal name
        """
        super().__init__(TYPE_EXPR)
        self.name: str = name

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Signal name
        """
        return self.name


class BinaryOp(SVANode):
    """
    @brief Represents a binary operation in SVA.
    
    Can be used for arithmetic, bitwise, comparison, or logical operations.
    """

    def __init__(
        self,
        left: SVANode,
        op: str,
        right: SVANode,
        ret_type: str = TYPE_BOOL
    ) -> None:
        """
        @brief Initialize binary operation node.
        @param left Left operand
        @param op Operator string (e.g., "+", "&&", "==")
        @param right Right operand
        @param ret_type Return type of this operation
        """
        super().__init__(ret_type)
        self.left: SVANode = left
        self.op: str = op
        self.right: SVANode = right

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Parenthesized binary operation
        """
        return f"({self.left} {self.op} {self.right})"


class UnarySysFunction(SVANode):
    """
    @brief Represents a unary system function call in SVA.
    
    Examples: $rose, $fell, $stable
    """

    def __init__(self, func: str, arg: SVANode) -> None:
        """
        @brief Initialize system function node.
        @param func Function name (e.g., "$rose", "$fell", "$stable")
        @param arg Function argument
        """
        super().__init__(TYPE_BOOL)
        self.func: str = func
        self.arg: SVANode = arg

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return System function call
        """
        return f"{self.func}({self.arg})"


# --- SEQUENCE LAYER ---


class SequenceDelay(SVANode):
    """
    @brief Represents a sequence delay operation in SVA.
    
    Examples: seq ##1 seq, seq ##[1:5] seq
    """

    def __init__(self, left: SVANode, delay_range: str, right: SVANode) -> None:
        """
        @brief Initialize sequence delay node.
        @param left Left sequence
        @param delay_range Delay specification (e.g., "##1", "##[1:5]")
        @param right Right sequence
        """
        super().__init__(TYPE_SEQUENCE)
        self.left: SVANode = left
        self.delay: str = delay_range
        self.right: SVANode = right

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Sequence with delay
        """
        return f"({self.left} {self.delay} {self.right})"


class SequenceRepeat(SVANode):
    """
    @brief Represents a sequence repetition in SVA.
    
    Examples: seq [*3], seq [=1:5], seq [->2]
    """

    def __init__(self, expr: SVANode, op: str, count: str) -> None:
        """
        @brief Initialize sequence repeat node.
        @param expr Expression to repeat
        @param op Repetition operator ([*, [=, [->)
        @param count Repetition count (e.g., "3", "1:5")
        """
        super().__init__(TYPE_SEQUENCE)
        self.expr: SVANode = expr
        self.op: str = op
        self.count: str = count

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Sequence with repetition
        """
        return f"({self.expr}{self.op}{self.count})"


class SequenceBinary(SVANode):
    """
    @brief Represents a binary sequence operation in SVA.
    
    Examples: seq1 intersect seq2, seq1 within seq2
    """

    def __init__(self, left: SVANode, op: str, right: SVANode) -> None:
        """
        @brief Initialize binary sequence operation.
        @param left Left sequence
        @param op Sequence operator (intersect, within, throughout, and, or)
        @param right Right sequence
        """
        super().__init__(TYPE_SEQUENCE)
        self.left: SVANode = left
        self.op: str = op
        self.right: SVANode = right

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Binary sequence operation
        """
        return f"({self.left} {self.op} {self.right})"


# --- PROPERTY LAYER ---


class Implication(SVANode):
    """
    @brief Represents an implication in SVA.
    
    Examples: ante |-> cons, ante |=> cons
    """

    def __init__(self, ante: SVANode, op: str, cons: SVANode) -> None:
        """
        @brief Initialize implication node.
        @param ante Antecedent (left side of implication)
        @param op Implication operator (|-> or |=>)
        @param cons Consequent (right side of implication)
        """
        super().__init__(TYPE_PROPERTY)
        self.ante: SVANode = ante
        self.op: str = op
        self.cons: SVANode = cons

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Implication expression
        """
        return f"{self.ante} {self.op} {self.cons}"


class DisableIff(SVANode):
    """
    @brief Represents a disable iff clause in SVA.
    
    Example: disable iff (reset) (property)
    """

    def __init__(self, reset_expr: SVANode, prop: SVANode) -> None:
        """
        @brief Initialize disable iff node.
        @param reset_expr Reset/disable expression
        @param prop Property to disable
        """
        super().__init__(TYPE_PROPERTY)
        self.reset: SVANode = reset_expr
        self.prop: SVANode = prop

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Disable iff clause
        """
        return f"disable iff ({self.reset}) ({self.prop})"


class NotProperty(SVANode):
    """
    @brief Represents a negated property in SVA.
    
    Example: not (property)
    """

    def __init__(self, prop: SVANode) -> None:
        """
        @brief Initialize not property node.
        @param prop Property to negate
        """
        super().__init__(TYPE_PROPERTY)
        self.prop: SVANode = prop

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Negated property
        """
        return f"not ({self.prop})"
