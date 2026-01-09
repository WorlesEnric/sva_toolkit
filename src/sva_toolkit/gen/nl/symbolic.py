"""
Symbolic SVAD Generator.

This module implements the "Symbolic SVAD" template which breaks down constraints into:
1. Scope (disable iff)
2. Logic (High-level symbolic formula)
3. Definitions (Recursive breakdown of symbols)
"""

from typing import Dict, List, Optional, Any, Tuple
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
        
        # Sort by ID number (Sym_0, Sym_1...)
        sorted_syms = sorted(self.symbol_table.symbol_map.items(), key=lambda x: int(x[0].split('_')[1]))
        
        for sym_id, definition in sorted_syms:
            definitions_text += f"* {sym_id}: {definition.description}\n"

        return f"{scope_text}Logic: {logic_text}\n{definitions_text}"

    def _get_symbol_id(self, prefix: str = "Exp") -> str:
        symbol_id = f"{prefix}_{self.id_counter}"
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
            trigger_desc = self._process_node(node.ante, is_root=True)
            
            # Right side is Requirement
            req_desc = self._process_node(node.cons, is_root=True)
            
            timing_str = "in the same cycle"
            if node.op == "|=>":
                timing_str = "in the next cycle"
                
            return f"When {trigger_desc} occurs, then {timing_str}, {req_desc} must hold."
            
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
        Simple: Signal, UnaryOp(Signal), BinaryOp(Signal, Signal)
        """
        if isinstance(node, Signal):
            return True
        if isinstance(node, UnaryOp) and self._is_simple(node.operand):
            return True
        if isinstance(node, BinaryOp) and self._is_simple(node.left) and self._is_simple(node.right):
            return True
        if isinstance(node, SequenceDelay) and node.delay == "##0":
             # "A ##0 B" -> "A and B". Simple enough? 
             # Maybe if A and B are simple.
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
            return f"{node.op}{inner}"
            
        if isinstance(node, BinaryOp):
            l = self._format_simple(node.left)
            r = self._format_simple(node.right)
            # Map op to text? User wanted symbolic formulas though: "A => (B or C)"
            # So keeping operators is fine? Or partial text?
            # User example: "Expression A: Signal busy is high." -> "busy"
            # User example: "$A \implies (B \text{ or } C)$"
            return f"({l} {node.op} {r})"
            
        return str(node)

    # --- Handlers ---

    def _handle_Signal(self, node: Signal, is_root: bool) -> str:
        return self.signal_formatter.format(node.name)

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
        
        count_clean = node.count.strip("[]")
        count_desc = count_clean
        
        if ":" in count_clean:
             count_desc = f"between {count_clean.replace(':', ' and ')} times"
        else:
             count_desc = f"{count_clean} times"
        
        if node.op == "[*":
             desc = f"{sub} remains true for {count_desc} consecutive cycles"
        elif node.op == "[=":
             # if count is "4 times", make it "4 times"
             # if count is "between 4 and 5 times", make it "between 4 and 5 times"
             desc = f"{sub} occurs {count_desc} (non-consecutively) before sequence continues"
        elif node.op == "[->":
             desc = f"the {count_desc} occurrence of {sub} (goto)"
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
            desc = f"{l} holds true throughout the execution of {r}"
        elif node.op == "and":
            desc = f"{l} and {r} both occur"
        elif node.op == "or":
            desc = f"{l} or {r} occurs"
        else:
            desc = f"{l} {node.op} {r}"
            
        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Seq")

    def _handle_BinaryOp(self, node: BinaryOp, is_root: bool) -> str:
        # (A or B), (A and B) etc.
        # If logical (&&, ||), split.
        if node.op in ["&&", "||", "and", "or"]:
            l = self._process_node(node.left)
            r = self._process_node(node.right)
            op_text = "or" if node.op in ["||", "or"] else "and"
            desc = f"{l} {op_text} {r}"
            
            if is_root: return desc
            return self._register_complex_node(desc, str(node), "Exp")
        
        # If arithmetic/comparison, treat as leaf/simple usually?
        # "((valid % $past(req, 2)) ^ (busy % ready))" -> "The result of calculation..."
        
        # If it's complex arithmetic, make it a symbol.
        if not self._is_simple(node):
            desc = f"The result of: {str(node)}"
            return self._register_complex_node(desc, str(node), "Calc")
            
        return self._format_simple(node)

    def _handle_Implication(self, node: Implication, is_root: bool) -> str:
        # Nested implication? 
        return self._process_root_logic(node)

    def _handle_PropertyIfElse(self, node: PropertyIfElse, is_root: bool) -> str:
        cond = self._process_node(node.condition)
        t = self._process_node(node.true_prop)
        f = self._process_node(node.false_prop) if node.false_prop else None
        
        if f:
            desc = f"If {cond} holds, then {t}, otherwise {f}"
        else:
            desc = f"If {cond} holds, then {t}"
            
        if is_root: return desc
        return self._register_complex_node(desc, str(node), "Prop")

    def _handle_PropertyUntil(self, node: PropertyUntil, is_root: bool) -> str:
        l = self._process_node(node.left)
        r = self._process_node(node.right)
        
        op_text = "until and including" if node.op == "until_with" else "until"
        desc = f"{l} holds {op_text} {r} occurs"
        
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
