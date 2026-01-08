"""
SVA Type System - Type-directed nodes for SystemVerilog Assertions.

This module defines the core type system for SVA generation, ensuring
that temporal logic, boolean expressions, and property implications
are always correctly nested following the SVA promotion hierarchy.
"""

from typing import Optional, List, Union
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


class UnaryOperator(Enum):
    """Unary operators for expressions and boolean logic."""
    LOGICAL_NOT = "!"       # Logical NOT
    BITWISE_NOT = "~"       # Bitwise NOT
    UNARY_MINUS = "-"       # Unary minus
    UNARY_PLUS = "+"        # Unary plus


class BinaryOperator(Enum):
    """Binary operators categorized by type."""
    # Logical operators (result: 1-bit boolean)
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"

    # Bitwise operators (result: same width as operands)
    BITWISE_AND = "&"
    BITWISE_OR = "|"
    BITWISE_XOR = "^"
    BITWISE_XNOR = "^~"
    BITWISE_XNOR_ALT = "~^"

    # Relational operators
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    # Equality operators
    EQ = "=="               # 2-value equality (may output X)
    NE = "!="               # 2-value inequality (may output X)
    CASE_EQ = "==="         # 4-value equality (treats X/Z as comparable)
    CASE_NE = "!=="         # 4-value inequality

    # Arithmetic operators
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"


class SystemFunction(Enum):
    """SVA system functions."""
    ROSE = "$rose"           # Signal rose from 0 to 1
    FELL = "$fell"           # Signal fell from 1 to 0
    STABLE = "$stable"       # Signal value unchanged
    CHANGED = "$changed"     # Signal value changed
    PAST = "$past"           # Past value with optional depth
    ONEHOT = "$onehot"       # Exactly one bit high
    ONEHOT0 = "$onehot0"     # At most one bit high
    ISUNKNOWN = "$isunknown" # Check for X or Z
    COUNTONES = "$countones" # Count bits equal to 1


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

    def to_natural_language(self) -> str:
        """
        @brief Convert node to natural language description using SVATrans.
        @return Natural language description of the SVA node
        """
        from sva_toolkit.gen.nl import sva_to_english
        return sva_to_english(self)


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

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        return f"signal '{self.name}'"


class UnaryOp(SVANode):
    """
    @brief Represents a unary operation in SVA.

    Supports logical NOT, bitwise NOT, unary minus/plus.
    """

    def __init__(
        self,
        op: Union[UnaryOperator, str],
        operand: SVANode,
        ret_type: Optional[str] = None
    ) -> None:
        """
        @brief Initialize unary operation node.
        @param op Unary operator (enum or string)
        @param operand Operand expression
        @param ret_type Return type (auto-determined if None)
        """
        # Auto-determine return type based on operator
        if ret_type is None:
            op_value = op.value if isinstance(op, UnaryOperator) else op
            if op_value == "!":
                ret_type = TYPE_BOOL
            elif op_value in ["~", "-", "+"]:
                ret_type = TYPE_EXPR
            else:
                ret_type = TYPE_EXPR

        super().__init__(ret_type)
        self.op: str = op.value if isinstance(op, UnaryOperator) else op
        self.operand: SVANode = operand

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Unary operation
        """
        return f"({self.op}{self.operand})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        templates = {
            "!": "NOT {operand}",
            "~": "bitwise NOT of {operand}",
            "-": "negative of {operand}",
            "+": "positive of {operand}",
        }
        template = templates.get(self.op, f"{self.op} {{operand}}")
        return template.format(operand=self.operand.to_natural_language())


class BinaryOp(SVANode):
    """
    @brief Represents a binary operation in SVA.

    Can be used for arithmetic, bitwise, comparison, or logical operations.
    """

    def __init__(
        self,
        left: SVANode,
        op: Union[BinaryOperator, str],
        right: SVANode,
        ret_type: Optional[str] = None
    ) -> None:
        """
        @brief Initialize binary operation node.
        @param left Left operand
        @param op Operator (enum or string)
        @param right Right operand
        @param ret_type Return type (auto-determined if None)
        """
        # Auto-determine return type based on operator
        if ret_type is None:
            op_value = op.value if isinstance(op, BinaryOperator) else op
            if op_value in ["&&", "||", "<", "<=", ">", ">=", "==", "!=", "===", "!=="]:
                ret_type = TYPE_BOOL
            else:
                ret_type = TYPE_EXPR

        super().__init__(ret_type)
        self.left: SVANode = left
        self.op: str = op.value if isinstance(op, BinaryOperator) else op
        self.right: SVANode = right

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Parenthesized binary operation
        """
        return f"({self.left} {self.op} {self.right})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        templates = {
            "==": "{left} equals {right}",
            "!=": "{left} does not equal {right}",
            "===": "{left} equals {right} (including X/Z)",
            "!==": "{left} does not equal {right} (including X/Z)",
            "&&": "{left} AND {right}",
            "||": "{left} OR {right}",
            "&": "{left} bitwise-AND {right}",
            "|": "{left} bitwise-OR {right}",
            "^": "{left} bitwise-XOR {right}",
            "^~": "{left} bitwise-XNOR {right}",
            "~^": "{left} bitwise-XNOR {right}",
            ">": "{left} is greater than {right}",
            "<": "{left} is less than {right}",
            ">=": "{left} is greater than or equal to {right}",
            "<=": "{left} is less than or equal to {right}",
            "+": "{left} plus {right}",
            "-": "{left} minus {right}",
            "*": "{left} multiplied by {right}",
            "/": "{left} divided by {right}",
            "%": "{left} modulo {right}",
        }
        template = templates.get(self.op, f"{{left}} {self.op} {{right}}")
        return template.format(
            left=self.left.to_natural_language(),
            right=self.right.to_natural_language()
        )


class TernaryOp(SVANode):
    """
    @brief Represents a ternary conditional operation (? :).

    Syntax: condition ? true_expr : false_expr
    """

    def __init__(
        self,
        condition: SVANode,
        true_expr: SVANode,
        false_expr: SVANode,
        ret_type: str = TYPE_EXPR
    ) -> None:
        """
        @brief Initialize ternary operation node.
        @param condition Condition expression
        @param true_expr Expression when condition is true
        @param false_expr Expression when condition is false
        @param ret_type Return type
        """
        super().__init__(ret_type)
        self.condition: SVANode = condition
        self.true_expr: SVANode = true_expr
        self.false_expr: SVANode = false_expr

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Ternary operation
        """
        return f"({self.condition} ? {self.true_expr} : {self.false_expr})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        return (f"if {self.condition.to_natural_language()} then "
                f"{self.true_expr.to_natural_language()} else "
                f"{self.false_expr.to_natural_language()}")


class UnarySysFunction(SVANode):
    """
    @brief Represents a unary system function call in SVA.

    Examples: $rose, $fell, $stable, $changed, $onehot, $isunknown, $countones
    """

    def __init__(
        self,
        func: Union[SystemFunction, str],
        arg: SVANode
    ) -> None:
        """
        @brief Initialize system function node.
        @param func Function name (enum or string)
        @param arg Function argument
        """
        super().__init__(TYPE_BOOL)
        self.func: str = func.value if isinstance(func, SystemFunction) else func
        self.arg: SVANode = arg

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return System function call
        """
        return f"{self.func}({self.arg})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        templates = {
            "$rose": "{arg} rises from 0 to 1",
            "$fell": "{arg} falls from 1 to 0",
            "$stable": "{arg} remains stable",
            "$changed": "{arg} changes value",
            "$onehot": "exactly one bit is high in {arg}",
            "$onehot0": "at most one bit is high in {arg}",
            "$isunknown": "{arg} is unknown (X or Z)",
            "$countones": "count of ones in {arg}",
        }
        template = templates.get(self.func, f"{self.func}({{arg}})")
        return template.format(arg=self.arg.to_natural_language())


class PastFunction(SVANode):
    """
    @brief Represents the $past system function with optional depth.

    Syntax: $past(signal) or $past(signal, depth)
    """

    def __init__(
        self,
        signal: SVANode,
        depth: Optional[int] = None
    ) -> None:
        """
        @brief Initialize $past function node.
        @param signal Signal to get past value of
        @param depth Number of clock cycles back (default: 1)
        """
        super().__init__(TYPE_EXPR)
        self.signal: SVANode = signal
        self.depth: Optional[int] = depth

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return $past function call
        """
        if self.depth is None:
            return f"$past({self.signal})"
        return f"$past({self.signal}, {self.depth})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        if self.depth is None or self.depth == 1:
            return f"the previous value of {self.signal.to_natural_language()}"
        return f"the value of {self.signal.to_natural_language()} {self.depth} cycles ago"


# --- SEQUENCE LAYER ---


class SequenceDelay(SVANode):
    """
    @brief Represents a sequence delay operation in SVA.

    Examples: seq ##1 seq, seq ##[1:5] seq, seq ##[0:$] seq
    """

    def __init__(
        self,
        left: SVANode,
        delay_range: str,
        right: SVANode
    ) -> None:
        """
        @brief Initialize sequence delay node.
        @param left Left sequence
        @param delay_range Delay specification (e.g., "##1", "##[1:5]", "##[0:$]")
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

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        delay_desc = self._parse_delay_description(self.delay)
        left_desc = self.left.to_natural_language()
        right_desc = self.right.to_natural_language()
        return f"({left_desc}), then {delay_desc}, ({right_desc})"

    def _parse_delay_description(self, delay: str) -> str:
        """Parse delay string into natural language."""
        if delay == "##0":
            return "immediately"
        elif delay == "##1":
            return "1 cycle later"
        elif delay.startswith("##[") and ":" in delay:
            # Extract min:max from ##[min:max]
            range_part = delay[3:-1]  # Remove ##[ and ]
            if "$" in range_part:
                parts = range_part.split(":")
                min_val = parts[0]
                if min_val == "0":
                    return "eventually"
                return f"at least {min_val} cycles later"
            else:
                min_val, max_val = range_part.split(":")
                if min_val == max_val:
                    return f"{min_val} cycles later"
                return f"between {min_val} and {max_val} cycles later"
        else:
            # ##N format
            cycles = delay[2:]
            return f"{cycles} cycles later"


class SequenceRepeat(SVANode):
    """
    @brief Represents a sequence repetition in SVA.

    Examples: seq [*3], seq [=1:5], seq [->2], seq [*0:$]
    """

    def __init__(
        self,
        expr: SVANode,
        op: str,
        count: str
    ) -> None:
        """
        @brief Initialize sequence repeat node.
        @param expr Expression to repeat
        @param op Repetition operator ([*, [=, [->)
        @param count Repetition count (e.g., "3", "1:5", "0:$")
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

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        count_desc = self._parse_count_description(self.count)
        op_desc = {
            "[*": "consecutively",
            "[=": "non-consecutively",
            "[->": "with goto"
        }.get(self.op, "")

        return f"{self.expr.to_natural_language()} occurs {count_desc} {op_desc}"

    def _parse_count_description(self, count: str) -> str:
        """Parse count string into natural language."""
        count = count.rstrip("]")
        if "$" in count:
            parts = count.split(":")
            min_val = parts[0]
            if min_val == "0":
                return "zero or more times"
            return f"at least {min_val} times"
        elif ":" in count:
            min_val, max_val = count.split(":")
            if min_val == max_val:
                return f"{min_val} times"
            return f"between {min_val} and {max_val} times"
        else:
            return f"{count} times"


class SequenceBinary(SVANode):
    """
    @brief Represents a binary sequence operation in SVA.

    Examples: seq1 intersect seq2, seq1 within seq2, expr throughout seq
    """

    def __init__(
        self,
        left: SVANode,
        op: str,
        right: SVANode
    ) -> None:
        """
        @brief Initialize binary sequence operation.
        @param left Left sequence
        @param op Sequence operator (intersect, throughout, and, or)
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

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        templates = {
            "intersect": "{left} intersects with {right}",
            "throughout": "{left} holds throughout {right}",
            "and": "{left} and {right}",
            "or": "{left} or {right}",
            "within": "{left} occurs within {right}",
        }
        template = templates.get(self.op, f"{{left}} {self.op} {{right}}")
        return template.format(
            left=self.left.to_natural_language(),
            right=self.right.to_natural_language()
        )


class SequenceFirstMatch(SVANode):
    """
    @brief Represents first_match operator for sequences.

    Syntax: first_match(sequence)
    """

    def __init__(self, sequence: SVANode) -> None:
        """
        @brief Initialize first_match node.
        @param sequence Sequence to match
        """
        super().__init__(TYPE_SEQUENCE)
        self.sequence: SVANode = sequence

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return first_match expression
        """
        return f"first_match({self.sequence})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        return f"the first match of {self.sequence.to_natural_language()}"


class SequenceEnded(SVANode):
    """
    @brief Represents sequence.ended construct.

    Syntax: sequence.ended
    """

    def __init__(self, sequence: SVANode) -> None:
        """
        @brief Initialize sequence.ended node.
        @param sequence Sequence to check for ending
        """
        super().__init__(TYPE_BOOL)
        self.sequence: SVANode = sequence

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return sequence.ended expression
        """
        return f"({self.sequence}).ended"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        return f"{self.sequence.to_natural_language()} has ended"


# --- PROPERTY LAYER ---


class Implication(SVANode):
    """
    @brief Represents an implication in SVA.

    Examples: ante |-> cons, ante |=> cons
    """

    def __init__(
        self,
        ante: SVANode,
        op: str,
        cons: SVANode
    ) -> None:
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

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        ante_desc = self.ante.to_natural_language()
        cons_desc = self.cons.to_natural_language()

        if self.op == "|->":
            return f"When {ante_desc}, then in the same cycle: {cons_desc}."
        else:  # |=>
            return f"When {ante_desc}, then in the next cycle: {cons_desc}."


class DisableIff(SVANode):
    """
    @brief Represents a disable iff clause in SVA.

    Example: disable iff (reset) (property)
    """

    def __init__(
        self,
        reset_expr: SVANode,
        prop: SVANode
    ) -> None:
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

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        prop_desc = self.prop.to_natural_language()
        reset_desc = self.reset.to_natural_language()
        # Remove trailing period if present
        if prop_desc.endswith('.'):
            prop_desc = prop_desc[:-1]
        return f"{prop_desc} (disabled when {reset_desc})."


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

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        return f"NOT ({self.prop.to_natural_language()})"


class PropertyIfElse(SVANode):
    """
    @brief Represents an if-else conditional property.

    Syntax: if (condition) property1 else property2
    """

    def __init__(
        self,
        condition: SVANode,
        true_prop: SVANode,
        false_prop: Optional[SVANode] = None
    ) -> None:
        """
        @brief Initialize if-else property node.
        @param condition Boolean condition
        @param true_prop Property when condition is true
        @param false_prop Property when condition is false (optional)
        """
        super().__init__(TYPE_PROPERTY)
        self.condition: SVANode = condition
        self.true_prop: SVANode = true_prop
        self.false_prop: Optional[SVANode] = false_prop

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return If-else property
        """
        if self.false_prop is None:
            return f"if ({self.condition}) {self.true_prop}"
        return f"if ({self.condition}) {self.true_prop} else {self.false_prop}"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        if self.false_prop is None:
            return (f"if {self.condition.to_natural_language()}, "
                   f"then {self.true_prop.to_natural_language()}")
        return (f"if {self.condition.to_natural_language()}, "
               f"then {self.true_prop.to_natural_language()}, "
               f"otherwise {self.false_prop.to_natural_language()}")


class PropertyUntil(SVANode):
    """
    @brief Represents temporal until operators.

    Syntax: property1 until property2
    Syntax: property1 until_with property2
    """

    def __init__(
        self,
        left: SVANode,
        op: str,
        right: SVANode
    ) -> None:
        """
        @brief Initialize until property node.
        @param left Left property/sequence
        @param op Until operator ("until" or "until_with")
        @param right Right property/sequence
        """
        super().__init__(TYPE_PROPERTY)
        self.left: SVANode = left
        self.op: str = op
        self.right: SVANode = right

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Until property
        """
        return f"({self.left} {self.op} {self.right})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        if self.op == "until":
            return f"{self.left.to_natural_language()} until {self.right.to_natural_language()}"
        else:  # until_with
            return f"{self.left.to_natural_language()} until (and including when) {self.right.to_natural_language()}"


class PropertyBinary(SVANode):
    """
    @brief Represents binary property operations (and, or).

    Syntax: property1 and property2, property1 or property2
    """

    def __init__(
        self,
        left: SVANode,
        op: str,
        right: SVANode
    ) -> None:
        """
        @brief Initialize binary property operation.
        @param left Left property
        @param op Binary operator ("and" or "or")
        @param right Right property
        """
        super().__init__(TYPE_PROPERTY)
        self.left: SVANode = left
        self.op: str = op
        self.right: SVANode = right

    def __str__(self) -> str:
        """
        @brief Convert to SVA code.
        @return Binary property operation
        """
        return f"({self.left} {self.op} {self.right})"

    def to_natural_language(self) -> str:
        """
        @brief Convert to natural language.
        @return Natural language description
        """
        left_desc = self.left.to_natural_language()
        right_desc = self.right.to_natural_language()

        # Remove trailing periods for cleaner conjunction
        if left_desc.endswith('.'):
            left_desc = left_desc[:-1]
        if right_desc.endswith('.'):
            right_desc = right_desc[:-1]

        if self.op == "and":
            return f"({left_desc}) AND ({right_desc})."
        else:  # or
            return f"({left_desc}) OR ({right_desc})."
