"""
Semantic Extraction from SVA AST.

This module provides the extractor that converts SVANode trees
into SVASemantics IR.
"""

import re
from typing import Optional, Any
from sva_toolkit.gen.nl.ir import SVASemantics, TimingSpec, TemporalType, ImplicationType


class SemanticExtractor:
    """Extracts semantic meaning from SVA AST nodes."""

    def __init__(self):
        self.signal_formatter = SignalFormatter()
        self.comparison_formatter = ComparisonFormatter()
        self.temporal_formatter = TemporalFormatter()

    def extract(self, node: Any) -> SVASemantics:
        """
        Main entry point for extraction.

        Dispatches to specific extract_X() methods based on node type.
        """
        node_type_name = type(node).__name__
        method_name = f"extract_{node_type_name}"
        method = getattr(self, method_name, self.extract_default)
        return method(node)

    def extract_default(self, node: Any) -> SVASemantics:
        """Default extractor for unknown node types."""
        return SVASemantics(
            description=str(node),
            node_type=type(node).__name__
        )

    # ===== Expression Layer =====

    def extract_Signal(self, node: Any) -> SVASemantics:
        """Extract semantics from Signal node."""
        return SVASemantics(
            description=self.signal_formatter.format(node.name),
            node_type="Signal"
        )

    def extract_UnaryOp(self, node: Any) -> SVASemantics:
        """Extract semantics from UnaryOp node."""
        operand = self.extract(node.operand)

        if node.op == "!":
            desc = f"NOT ({operand.description})"
        elif node.op == "~":
            desc = f"bitwise NOT of {operand.description}"
        elif node.op == "-":
            desc = f"negative of {operand.description}"
        elif node.op == "+":
            desc = operand.description
        else:
            desc = f"{node.op}({operand.description})"

        return SVASemantics(
            description=desc,
            negated=(node.op == "!"),
            node_type="UnaryOp"
        )

    def extract_BinaryOp(self, node: Any) -> SVASemantics:
        """Extract semantics from BinaryOp node."""
        left = self.extract(node.left)
        right = self.extract(node.right)

        desc = self.comparison_formatter.format(
            left.description,
            node.op,
            right.description
        )

        # For logical AND/OR, preserve as components
        if node.op in ["&&", "||"]:
            logical_op = "and" if node.op == "&&" else "or"
            return SVASemantics(
                description=desc,
                logical_op=logical_op,
                components=[left, right],
                node_type="BinaryOp"
            )

        return SVASemantics(
            description=desc,
            node_type="BinaryOp"
        )

    def extract_TernaryOp(self, node: Any) -> SVASemantics:
        """Extract semantics from TernaryOp node (? :)."""
        cond = self.extract(node.cond)
        true_val = self.extract(node.true_val)
        false_val = self.extract(node.false_val)

        return SVASemantics(
            description=f"if {cond.description} then {true_val.description} else {false_val.description}",
            conditional_on=cond.description,
            outcome=true_val.description,
            conditional_else=false_val.description,
            node_type="TernaryOp"
        )

    def extract_UnarySysFunction(self, node: Any) -> SVASemantics:
        """Extract semantics from UnarySysFunction node."""
        arg = self.extract(node.arg)
        signal_name = arg.description

        func_descriptions = {
            "$rose": f"{signal_name} rises from low to high",
            "$fell": f"{signal_name} falls from high to low",
            "$stable": f"{signal_name} remains stable",
            "$changed": f"{signal_name} changes value",
            "$onehot": f"exactly one bit is high in {signal_name}",
            "$onehot0": f"at most one bit is high in {signal_name}",
            "$isunknown": f"{signal_name} is unknown (X or Z)",
            "$countones": f"the count of high bits in {signal_name}",
        }

        desc = func_descriptions.get(node.func, f"{node.func}({signal_name})")

        return SVASemantics(
            description=desc,
            node_type="UnarySysFunction"
        )

    def extract_PastFunction(self, node: Any) -> SVASemantics:
        """Extract semantics from PastFunction node ($past)."""
        arg = self.extract(node.arg)
        depth = node.depth

        if depth == 1:
            desc = f"the value of {arg.description} from the previous cycle"
        else:
            desc = f"the value of {arg.description} from {depth} cycles ago"

        return SVASemantics(
            description=desc,
            node_type="PastFunction"
        )

    # ===== Sequence Layer =====

    def extract_SequenceDelay(self, node: Any) -> SVASemantics:
        """Extract semantics from SequenceDelay node (##)."""
        left = self.extract(node.left)
        right = self.extract(node.right)
        timing = self.temporal_formatter.parse_delay(node.delay)

        return SVASemantics(
            trigger=left.description,
            outcome=right.description,
            timing=timing,
            description=f"{left.description}, then {timing.to_natural_language()}, {right.description}",
            complexity="moderate",
            node_type="SequenceDelay"
        )

    def extract_SequenceRepeat(self, node: Any) -> SVASemantics:
        """Extract semantics from SequenceRepeat node ([*, [=, [->)."""
        expr = self.extract(node.expr)
        repeat_op = node.op  # The operator: [*, [=, [->
        count_str = node.count  # The count: [3], [1:5], [0:$], etc.

        if repeat_op == "[*":
            repeat_desc = "consecutively"
        elif repeat_op == "[=":
            repeat_desc = "non-consecutively"
        elif repeat_op == "[->":
            repeat_desc = "with goto"
        else:
            repeat_desc = ""

        # Parse count string to get count description
        # Strip brackets if present
        count_clean = count_str.strip("[]")

        if ":" in count_clean:
            parts = count_clean.split(":")
            min_count = parts[0]
            max_count = parts[1]
            if max_count == "$":
                count_desc = f"at least {min_count} times"
            else:
                count_desc = f"between {min_count} and {max_count} times"
        elif count_clean == "*":
            count_desc = "zero or more times"
        else:
            count = int(count_clean)
            if count == 1:
                count_desc = "once"
            else:
                count_desc = f"{count} times"

        desc = f"{expr.description} occurs {count_desc} {repeat_desc}".strip()

        return SVASemantics(
            description=desc,
            complexity="moderate",
            node_type="SequenceRepeat"
        )

    def extract_SequenceBinary(self, node: Any) -> SVASemantics:
        """Extract semantics from SequenceBinary node (intersect, throughout, etc.)."""
        left = self.extract(node.left)
        right = self.extract(node.right)

        if node.op == "intersect":
            desc = f"{left.description} intersects with {right.description}"
        elif node.op == "throughout":
            desc = f"{left.description} holds continuously while {right.description}"
        elif node.op == "and":
            desc = f"{left.description} and {right.description}"
        elif node.op == "or":
            desc = f"{left.description} or {right.description}"
        else:
            desc = f"{left.description} {node.op} {right.description}"

        logical_op = None
        components = []
        if node.op in ["and", "or"]:
            logical_op = node.op
            components = [left, right]

        return SVASemantics(
            description=desc,
            logical_op=logical_op,
            components=components,
            complexity="moderate",
            node_type="SequenceBinary"
        )

    def extract_SequenceFirstMatch(self, node: Any) -> SVASemantics:
        """Extract semantics from SequenceFirstMatch node."""
        seq = self.extract(node.seq)
        return SVASemantics(
            description=f"the first match of {seq.description}",
            complexity="moderate",
            node_type="SequenceFirstMatch"
        )

    def extract_SequenceEnded(self, node: Any) -> SVASemantics:
        """Extract semantics from SequenceEnded node."""
        seq = self.extract(node.seq)
        return SVASemantics(
            description=f"{seq.description} has ended",
            node_type="SequenceEnded"
        )

    # ===== Property Layer =====

    def extract_Implication(self, node: Any) -> SVASemantics:
        """Extract semantics from Implication node."""
        ante = self.extract(node.ante)
        cons = self.extract(node.cons)

        implication_type = (
            ImplicationType.OVERLAPPING if node.op == "|->"
            else ImplicationType.NON_OVERLAPPING
        )

        timing = TimingSpec(
            TemporalType.IMMEDIATE if node.op == "|->"
            else TemporalType.NEXT_CYCLE
        )

        # Determine complexity based on the antecedent and consequent
        complexity = "simple"
        if ante.complexity != "simple" or cons.complexity != "simple":
            complexity = "moderate"
        if ante.is_compound() or cons.is_compound():
            complexity = "complex"

        return SVASemantics(
            trigger=ante.description,
            outcome=cons.description,
            implication=implication_type,
            timing=timing,
            complexity=complexity,
            node_type="Implication"
        )

    def extract_DisableIff(self, node: Any) -> SVASemantics:
        """Extract semantics from DisableIff node."""
        prop = self.extract(node.prop)
        cond = self.extract(node.reset)

        return SVASemantics(
            description=prop.description,
            trigger=prop.trigger,
            outcome=prop.outcome,
            implication=prop.implication,
            timing=prop.timing,
            disabled_when=cond.description,
            complexity=prop.complexity,
            node_type="DisableIff"
        )

    def extract_NotProperty(self, node: Any) -> SVASemantics:
        """Extract semantics from NotProperty node."""
        prop = self.extract(node.prop)

        return SVASemantics(
            description=f"NOT ({prop.description})",
            trigger=prop.trigger,
            outcome=prop.outcome,
            negated=True,
            complexity=prop.complexity,
            node_type="NotProperty"
        )

    def extract_PropertyIfElse(self, node: Any) -> SVASemantics:
        """Extract semantics from PropertyIfElse node."""
        cond = self.extract(node.cond)
        true_prop = self.extract(node.true_prop)
        false_prop = self.extract(node.false_prop) if node.false_prop else None

        return SVASemantics(
            conditional_on=cond.description,
            outcome=true_prop.description,
            conditional_else=false_prop.description if false_prop else None,
            complexity="moderate",
            node_type="PropertyIfElse"
        )

    def extract_PropertyUntil(self, node: Any) -> SVASemantics:
        """Extract semantics from PropertyUntil node."""
        left = self.extract(node.left)
        right = self.extract(node.right)

        if node.op == "until":
            desc = f"{left.description} until {right.description}"
        elif node.op == "until_with":
            desc = f"{left.description} until and including {right.description}"
        else:
            desc = f"{left.description} {node.op} {right.description}"

        return SVASemantics(
            description=desc,
            complexity="moderate",
            node_type="PropertyUntil"
        )

    def extract_PropertyBinary(self, node: Any) -> SVASemantics:
        """Extract semantics from PropertyBinary node (and, or)."""
        left = self.extract(node.left)
        right = self.extract(node.right)

        return SVASemantics(
            description=f"{left.description} {node.op} {right.description}",
            logical_op=node.op,
            components=[left, right],
            complexity="complex" if left.is_compound() or right.is_compound() else "moderate",
            node_type="PropertyBinary"
        )


class SignalFormatter:
    """Formats signal names naturally for use in English."""

    # Common signal name expansions
    SIGNAL_EXPANSIONS = {
        "req": "request",
        "ack": "acknowledge",
        "gnt": "grant",
        "clk": "clock",
        "rst": "reset",
        "en": "enable",
    }

    def format(self, signal_name: str, as_subject: bool = True) -> str:
        """Format signal name for natural language."""
        # Check if signal has a known expansion
        base_name = signal_name.lower()
        if base_name in self.SIGNAL_EXPANSIONS:
            expanded = self.SIGNAL_EXPANSIONS[base_name]
            return f"the {expanded} signal"

        # Default: "the {signal} signal"
        return f"the {signal_name} signal"


class ComparisonFormatter:
    """Formats comparison expressions naturally."""

    OPERATOR_MAP = {
        "==": "equals",
        "!=": "does not equal",
        "===": "is exactly equal to",
        "!==": "is not exactly equal to",
        ">": "is greater than",
        "<": "is less than",
        ">=": "is at least",
        "<=": "is at most",
        "&&": "AND",
        "||": "OR",
        "&": "bitwise AND",
        "|": "bitwise OR",
        "^": "bitwise XOR",
        "^~": "bitwise XNOR",
        "~^": "bitwise XNOR",
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "divided by",
        "%": "modulo",
        "<<": "left-shifted by",
        ">>": "right-shifted by",
    }

    def format(self, left: str, op: str, right: str) -> str:
        """Format comparison expression."""
        op_text = self.OPERATOR_MAP.get(op, op)
        return f"{left} {op_text} {right}"


class TemporalFormatter:
    """Formats temporal specifications naturally."""

    def parse_delay(self, delay_str: str) -> TimingSpec:
        """
        Parse delay string like '##1', '##[1:5]', '##[0:$]' into TimingSpec.
        """
        # Handle ##0 (immediate)
        if delay_str == "##0":
            return TimingSpec(TemporalType.IMMEDIATE)

        # Handle ##N (fixed delay)
        if delay_str.startswith("##") and "[" not in delay_str:
            cycles = int(delay_str[2:])
            if cycles == 1:
                return TimingSpec(TemporalType.NEXT_CYCLE)
            return TimingSpec(TemporalType.FIXED, cycles, cycles)

        # Handle ##[min:max] (range)
        if delay_str.startswith("##[") and ":" in delay_str:
            range_part = delay_str[3:-1]  # Remove ##[ and ]
            min_str, max_str = range_part.split(":")

            min_cycles = int(min_str)

            if max_str == "$":
                # Unbounded: ##[n:$]
                return TimingSpec(TemporalType.UNBOUNDED, min_cycles, None)
            else:
                max_cycles = int(max_str)
                if min_cycles == max_cycles:
                    return TimingSpec(TemporalType.FIXED, min_cycles, max_cycles)
                return TimingSpec(TemporalType.RANGE, min_cycles, max_cycles)

        # Fallback
        return TimingSpec(TemporalType.IMMEDIATE)
