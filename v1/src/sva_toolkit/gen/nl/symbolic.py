"""
Symbolic SVAD Generator.

This module implements the "Symbolic SVAD" template which breaks down constraints into:
1. Scope (disable iff)
2. Logic (High-level symbolic formula)
3. Definitions (Recursive breakdown of symbols)
"""

from typing import Dict, List, Optional, Any, Tuple
import re
from dataclasses import dataclass, field
from sva_toolkit.gen.types_sva import (
    SVANode, Signal, UnaryOp, BinaryOp, TernaryOp,
    UnarySysFunction, PastFunction, SequenceDelay, SequenceRepeat,
    SequenceBinary, SequenceFirstMatch, SequenceEnded,
    Implication, DisableIff, NotProperty, PropertyIfElse,
    PropertyUntil, PropertyBinary
)
from sva_toolkit.gen.nl.extractor import SemanticExtractor, SignalFormatter, ComparisonFormatter, TemporalFormatter
from sva_toolkit.gen.nl.ir import TimingSpec, TemporalType, ImplicationType


@dataclass
class SymbolDefinition:
    symbol_id: str
    description: str
    sub_definitions: List['SymbolDefinition'] = field(default_factory=list)


class SymbolTable:
    """
    Tracks defined symbols to ensure deduplication.
    """
    def __init__(self):
        self.definitions: Dict[str, str] = {}  # content_hash -> symbol_id
        self.symbol_map: Dict[str, SymbolDefinition] = {} # symbol_id -> Definition
        self.counter = 0

    def get_or_create(self, content_key: str, description: str) -> str:
        """
        Returns existing symbol ID if content matches, otherwise creates new one.
        """
        if content_key in self.definitions:
            return self.definitions[content_key]
        
        # Create new symbol
        # Use alphabet for first few (A, B, C...) then append numbers if needed?
        # User example uses "Expression A", "Scenario 1".
        # Let's stick to a simple generic "Expression_N" or similar for now, 
        # but the user example had "Expression A", "Scenario 1".
        # Let's try to infer type? 
        # Actually user said: "Assign it a Symbol ID (e.g., Exp_0)"
        
        symbol_id = f"Sym_{self.counter}"
        self.counter += 1
        
        self.definitions[content_key] = symbol_id
        self.symbol_map[symbol_id] = SymbolDefinition(symbol_id, description)
        return symbol_id

    def add_definition(self, symbol_id: str, desc: str):
        if symbol_id not in self.symbol_map:
             self.symbol_map[symbol_id] = SymbolDefinition(symbol_id, desc)
        else:
            # Update description if it was a placeholder?
            # In this design, we create with description.
            pass


class SymbolicSVADGenerator:
    """
    Generates Symbolic SVAD descriptions.
    """
    def __init__(self):
        self.extractor = SemanticExtractor()
        self.symbol_table = SymbolTable()
        self.signal_formatter = SignalFormatter()
        self.comparison_formatter = ComparisonFormatter()
        self.temporal_formatter = TemporalFormatter()
        
        # Counters for specific types to match user style "Expression A", "Sequence B" could be nice
        # but "Sym_0" is safer for automation.
        self.id_counter = 0

    def generate(self, node: SVANode) -> str:
        self.symbol_table = SymbolTable() # Reset for new generation
        self.id_counter = 0
        
        # 1. Identify Scope (Disable Iff)
        scope_text = ""
        root_logic = node
        
        if isinstance(node, DisableIff):
            scope_text = f"Scope: This property is active unless {self._format_simple(node.reset)} is asserted.\n"
            root_logic = node.prop
            
        # 2. Extract Logic (The core implication or formula)
        logic_text = self._process_root_logic(root_logic)
        
        # 3. Format Definitions
        definitions_text = "Definitions:\n"
        # We need to print them in order of appearance or logical depth?
        # The symbol table has them.
        # User example:
        # * Expression A: ...
        # * Scenario 1: ...
        
        ordered_defs = self._ordered_definitions()

        for sym_id, definition in ordered_defs:
            definitions_text += f"* {sym_id}: {definition.description}\n"

        return f"{scope_text}Logic: {logic_text}\n{definitions_text}"

    def _get_symbol_id(self, prefix: str = "Exp") -> str:
        symbol_id = f"Exp_{self.id_counter}"
        self.id_counter += 1
        return symbol_id

    def _register_complex_node(self, desc: str, node_key: str, prefix: str = "Exp") -> str:
        # Simple deduplication check
        if node_key in self.symbol_table.definitions:
            return self.symbol_table.definitions[node_key]
            
        symbol_id = self._get_symbol_id(prefix)
        self.symbol_table.definitions[node_key] = symbol_id
        self.symbol_table.symbol_map[symbol_id] = SymbolDefinition(symbol_id, desc)
        return symbol_id

    def _process_root_logic(self, node: SVANode) -> str:
        """
        Process the top-level node. Expected to be Implication usually.
        Returns the "Logic: ..." string.
        """
        if isinstance(node, Implication):
            # Special handling for trigger/result split
            # "When {Left} occurs, then {Timing}, {Right} must hold."

            # Left side is Trigger
            trigger_ref = self._root_ref(node.ante, is_trigger=True)

            # Right side is Requirement
            req_ref = self._root_ref(node.cons, is_trigger=False)

            timing_str = "in the same cycle"
            if node.op == "|=>":
                timing_str = "in the next cycle"

            if self._is_sequence_node(node.ante):
                return (
                    f"When {trigger_ref} occurs, then {req_ref} must hold "
                    f"{timing_str}."
                )
            return (
                f"If {trigger_ref}, then {req_ref} must hold "
                f"{timing_str}."
            )
            
        return self._process_node(node, is_root=True)

    def _process_node(self, node: SVANode, is_root: bool = False) -> str:
        """
        Recursive processing.
        If node is complex, registers it and returns Symbol ID.
        If node is simple, returns text directly.
        """
        method_name = f"_handle_{type(node).__name__}"
        handler = getattr(self, method_name, self._handle_default)
        return handler(node, is_root)

    def _is_simple(self, node: SVANode) -> bool:
        """
        Determines if a node is 'simple' enough to be inlined.
        Simple: short signal/expr forms with no deep nesting.
        """
        if isinstance(node, Signal):
            return True
        if isinstance(node, UnaryOp) and self._is_simple(node.operand):
            return True
        if isinstance(node, UnarySysFunction) and self._is_simple(node.arg):
            return True
        if isinstance(node, PastFunction) and self._is_simple(node.signal):
            return True
        if isinstance(node, BinaryOp) and self._is_simple(node.left) and self._is_simple(node.right):
            return True
        if isinstance(node, TernaryOp):
            return (
                self._is_simple(node.condition)
                and self._is_simple(node.true_expr)
                and self._is_simple(node.false_expr)
            )
        if isinstance(node, SequenceEnded) and self._is_simple(node.sequence):
            return True
        if isinstance(node, SequenceDelay) and node.delay == "##0":
            return self._is_simple(node.left) and self._is_simple(node.right)
        return False

    def _format_simple(self, node: SVANode) -> str:
        """
        Returns string for simple nodes without creating symbols.
        """
        # We can implement a mini-recursive simple formatter here
        # or reuse parts of the old extractor but that one assumes flat strings.
        # Let's make a dedicated simple formatter.
        if isinstance(node, Signal):
            return self.signal_formatter.format(node.name)
        
        if isinstance(node, UnaryOp):
            inner = self._format_simple(node.operand)
            return self._format_unary_text(node.op, inner)

        if isinstance(node, BinaryOp):
            l = self._format_simple(node.left)
            r = self._format_simple(node.right)
            return self._format_binary_text(l, node.op, r)

        return str(node)

    def _is_sequence_node(self, node: SVANode) -> bool:
        return isinstance(
            node,
            (SequenceDelay, SequenceRepeat, SequenceBinary, SequenceFirstMatch),
        )

    def _is_symbol_ref(self, text: str) -> bool:
        return bool(re.fullmatch(r"Exp_\d+", text.strip()))

    def _ordered_definitions(self) -> List[Tuple[str, SymbolDefinition]]:
        symbols = self.symbol_table.symbol_map
        if not symbols:
            return []

        def symbol_index(symbol_id: str) -> int:
            try:
                return int(symbol_id.split("_", 1)[1])
            except (IndexError, ValueError):
                return 0

        ref_pattern = re.compile(r"\bExp_\d+\b")
        deps: Dict[str, set] = {sid: set() for sid in symbols}
        indegree: Dict[str, int] = {sid: 0 for sid in symbols}

        for sid, definition in symbols.items():
            for ref in ref_pattern.findall(definition.description):
                if ref in symbols and ref != sid and ref not in deps[sid]:
                    deps[sid].add(ref)
                    indegree[ref] += 1

        ready = sorted(
            [sid for sid, deg in indegree.items() if deg == 0],
            key=symbol_index
        )
        ordered: List[str] = []

        while ready:
            sid = ready.pop(0)
            ordered.append(sid)
            for child in sorted(deps[sid], key=symbol_index):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort(key=symbol_index)

        if len(ordered) != len(symbols):
            ordered = sorted(symbols.keys(), key=symbol_index)

        return [(sid, symbols[sid]) for sid in ordered]

    def _needs_asserted_suffix(self, text: str) -> bool:
        verb_markers = (
            " is ", " equals ", " does ", " rises ", " falls ", " remains ",
            " changes ", " holds ", " contains ", " occurs ", " greater ",
            " less ", " at most ", " at least ", " bitwise ", " plus ",
            " minus ", " times ", " divided ", " modulo ", " not ",
        )
        return " signal" in text and not any(marker in text for marker in verb_markers)

    def _condition_text(self, text: str) -> str:
        if self._is_symbol_ref(text):
            return f"{text} holds"
        if self._needs_asserted_suffix(text):
            return f"{text} is asserted"
        return text

    def _condition_ref(self, node: SVANode, text: str) -> str:
        if self._is_sequence_node(node):
            return text
        return self._condition_text(text)

    def _root_ref(self, node: SVANode, is_trigger: bool) -> str:
        if self._is_simple(node):
            ref = self._process_node(node, is_root=True)
        else:
            ref = self._process_node(node, is_root=False)
        if not is_trigger:
            return ref
        if self._is_sequence_node(node):
            return ref
        return self._condition_text(ref)

    def _format_unary_text(self, op: str, operand: str) -> str:
        if op == "!":
            return f"not ({operand})"
        if op == "~":
            return f"bitwise NOT of {operand}"
        if op == "-":
            return f"negative of {operand}"
        if op == "+":
            return operand
        return f"{op}({operand})"

    def _format_binary_text(self, left: str, op: str, right: str) -> str:
        if op == "&&":
            return f"{left} and {right}"
        if op == "||":
            return f"{left} or {right}"
        return self.comparison_formatter.format(left, op, right)

    # --- Handlers ---

    def _handle_Signal(self, node: Signal, is_root: bool) -> str:
        return self.signal_formatter.format(node.name)

    def _handle_UnaryOp(self, node: UnaryOp, is_root: bool) -> str:
        inner = self._process_node(node.operand)
        desc = self._format_unary_text(node.op, inner)
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Exp")

    def _handle_SequenceDelay(self, node: SequenceDelay, is_root: bool) -> str:
        # {Sub-A} followed by {Sub-B} exactly n cycles later
        
        # For sequence delay, we generally want to define it as a Sequence symbol 
        # unless it's very simple.
        
        # If we are inside a definition, we return text.
        # If we are at root, we might not need a symbol if it's the only thing.
        # BUT, the request says: "Break it into definitions".
        
        # Let's act recursively.
        sub_a = self._process_node(node.left)
        sub_b = self._process_node(node.right)
        
        timing = self.temporal_formatter.parse_delay(node.delay).to_natural_language()
        
        desc = f"{sub_a} followed by {sub_b} {timing}"
        
        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Seq")

    def _handle_SequenceRepeat(self, node: SequenceRepeat, is_root: bool) -> str:
        # {Sub} remains true for n consecutive cycles
        sub = self._process_node(node.expr)

        count_clean = node.count.rstrip("]")
        count_desc = self._describe_repeat_count(count_clean)

        if node.op == "[*":
            desc = f"{sub} remains true {count_desc} consecutively"
        elif node.op == "[=":
            desc = f"{sub} occurs {count_desc} non-consecutively before the sequence continues"
        elif node.op == "[->":
            desc = f"wait for the {self._ordinal_from_count(count_clean)} occurrence of {sub}"
        else:
            desc = f"{sub} {node.op} {count_desc}"

        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Seq")

    def _handle_SequenceBinary(self, node: SequenceBinary, is_root: bool) -> str:
        l = self._process_node(node.left)
        r = self._process_node(node.right)
        
        desc = ""
        if node.op == "intersect":
            desc = f"Sequence {l} and Sequence {r} start and end at the exact same time"
        elif node.op == "throughout":
            desc = f"{l} holds throughout {r}"
        elif node.op == "and":
            desc = f"{l} and {r} both occur"
        elif node.op == "or":
            desc = f"{l} or {r} occurs"
        else:
            desc = f"{l} {node.op} {r}"
            
        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Seq")

    def _handle_BinaryOp(self, node: BinaryOp, is_root: bool) -> str:
        l = self._process_node(node.left)
        r = self._process_node(node.right)
        desc = self._format_binary_text(l, node.op, r)
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Exp")

    def _handle_TernaryOp(self, node: TernaryOp, is_root: bool) -> str:
        cond = self._process_node(node.condition)
        t = self._process_node(node.true_expr)
        f = self._process_node(node.false_expr)
        desc = f"if {cond} then {t} else {f}"
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Exp")

    def _handle_SequenceFirstMatch(self, node: SequenceFirstMatch, is_root: bool) -> str:
        seq = self._process_node(node.sequence)
        desc = f"the first match of {seq}"
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Seq")

    def _handle_SequenceEnded(self, node: SequenceEnded, is_root: bool) -> str:
        seq = self._process_node(node.sequence)
        desc = f"the sequence {seq} has ended"
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Seq")

    def _handle_UnarySysFunction(self, node: UnarySysFunction, is_root: bool) -> str:
        arg_desc = self._process_node(node.arg)
        func_desc = {
            "$rose": f"{arg_desc} rises from low to high",
            "$fell": f"{arg_desc} falls from high to low",
            "$stable": f"{arg_desc} remains stable",
            "$changed": f"{arg_desc} changes value",
            "$onehot": f"exactly one bit of {arg_desc} is high",
            "$onehot0": f"at most one bit of {arg_desc} is high",
            "$isunknown": f"{arg_desc} is unknown (X or Z)",
            "$countones": f"the count of high bits in {arg_desc}",
        }
        desc = func_desc.get(node.func, f"{node.func} applied to {arg_desc}")
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Sys")

    def _handle_PastFunction(self, node: PastFunction, is_root: bool) -> str:
        signal_desc = self._process_node(node.signal)
        depth = node.depth if node.depth is not None else 1
        if depth == 1:
            desc = f"the previous value of {signal_desc}"
        else:
            desc = f"the value of {signal_desc} from {depth} cycles ago"
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Past")

    def _handle_NotProperty(self, node: NotProperty, is_root: bool) -> str:
        prop_desc = self._process_node(node.prop)
        desc = f"it is not the case that {prop_desc}"
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Prop")

    def _handle_DisableIff(self, node: DisableIff, is_root: bool) -> str:
        prop_desc = self._process_node(node.prop)
        reset_desc = self._process_node(node.reset)
        desc = f"{prop_desc} (disabled when {reset_desc})"
        if is_root:
            return desc
        return self._register_complex_node(desc, str(node), "Prop")

    def _handle_Implication(self, node: Implication, is_root: bool) -> str:
        # Nested implication? 
        return self._process_root_logic(node)

    def _handle_PropertyIfElse(self, node: PropertyIfElse, is_root: bool) -> str:
        cond_raw = self._process_node(node.condition)
        cond = self._condition_ref(node.condition, cond_raw)
        t = self._process_node(node.true_prop)
        f = self._process_node(node.false_prop) if node.false_prop else None
        
        if f:
            desc = f"If {cond}, then {t}, otherwise {f}"
        else:
            desc = f"If {cond}, then {t}"
            
        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Prop")

    def _handle_PropertyUntil(self, node: PropertyUntil, is_root: bool) -> str:
        l_raw = self._process_node(node.left)
        r_raw = self._process_node(node.right)
        l = self._condition_ref(node.left, l_raw)
        if self._is_sequence_node(node.right):
            r = f"{r_raw} occurs"
        else:
            r = self._condition_text(r_raw)
        
        op_text = "until and including" if node.op == "until_with" else "until"
        desc = f"{l} {op_text} {r}"
        
        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Prop")

    def _handle_PropertyBinary(self, node: PropertyBinary, is_root: bool) -> str:
        l = self._process_node(node.left)
        r = self._process_node(node.right)
        
        # Similar to SequenceBinary/BinaryOp
        op_text = node.op.lower() 
        desc = f"{l} {op_text} {r}"
        
        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Prop")

    def _handle_default(self, node: SVANode, is_root: bool) -> str:
        return self._format_simple(node)

    def _describe_repeat_count(self, count: str) -> str:
        """Convert raw repeat count tokens into English."""
        if ":" in count:
            min_part, max_part = count.split(":", 1)
            if max_part == "$":
                return f"at least {min_part} times"
            if min_part == max_part:
                return f"{min_part} times"
            return f"between {min_part} and {max_part} times"

        if count == "$":
            return "an unbounded number of times"

        return f"{count} times"

    def _ordinal_from_count(self, count: str) -> str:
        """Convert count string to ordinal text (1 -> 1st)."""
        try:
            num = int(count.split(":")[0])
        except ValueError:
            return f"{count}-th"

        suffix = "th"
        if 10 <= num % 100 <= 20:
            suffix = "th"
        else:
            if num % 10 == 1:
                suffix = "st"
            elif num % 10 == 2:
                suffix = "nd"
            elif num % 10 == 3:
                suffix = "rd"

        return f"{num}{suffix}"
