"""
SVA Chain-of-Thought Builder - Generate CoT reasoning from SVA AST.

Uses template matching to construct structured reasoning chains.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import re

from sva_toolkit.ast_parser import SVAASTParser, SVAStructure
from sva_toolkit.ast_parser.parser import (
    ImplicationType,
    TemporalOperator,
    BuiltinFunction,
    Signal,
    DelayRange,
)


@dataclass
class CoTSection:
    """A section of the Chain-of-Thought."""
    title: str
    content: str


class SVACoTBuilder:
    """
    Builder for generating Chain-of-Thought reasoning from SVA code.
    
    Uses template matching to construct structured reasoning chains
    based on the SVA AST structure.
    """

    # Built-in function descriptions
    BUILTIN_DESCRIPTIONS = {
        "$rose": "Signal transitions from 0 to 1 (rising edge)",
        "$fell": "Signal transitions from 1 to 0 (falling edge)", 
        "$stable": "Signal value remains unchanged from previous cycle",
        "$changed": "Signal value changes from previous cycle",
        "$past": "Reference to signal value from previous cycles",
        "$onehot": "Exactly one bit is set in the signal",
        "$onehot0": "At most one bit is set (all zeros allowed)",
        "$isunknown": "Signal contains unknown (X) values",
        "$countones": "Count the number of 1 bits in the signal",
        "$sampled": "Sampled value of signal in the current clock cycle",
        "$bits": "Number of bits in the signal",
    }

    # Temporal operator descriptions
    TEMPORAL_DESCRIPTIONS = {
        TemporalOperator.DELAY: "Cycle delay operator",
        TemporalOperator.REPETITION_CONSECUTIVE: "Consecutive repetition",
        TemporalOperator.REPETITION_GOTO: "Goto repetition (non-consecutive with final match)",
        TemporalOperator.REPETITION_NON_CONSECUTIVE: "Non-consecutive repetition",
        TemporalOperator.THROUGHOUT: "Condition holds throughout sequence",
        TemporalOperator.WITHIN: "Sequence completes within another",
        TemporalOperator.INTERSECT: "Sequences start and end together",
        TemporalOperator.AND: "Both sequences must hold",
        TemporalOperator.OR: "At least one sequence must hold",
        TemporalOperator.NOT: "Negation of sequence/property",
        TemporalOperator.FIRST_MATCH: "First successful match of sequence",
        TemporalOperator.IF_ELSE: "Conditional sequence selection",
    }

    def __init__(self, parser: Optional[SVAASTParser] = None):
        """
        Initialize the CoT builder.
        
        Args:
            parser: Optional SVAASTParser instance
        """
        self.parser = parser or SVAASTParser()

    def build(self, sva_code: str) -> str:
        """
        Build Chain-of-Thought reasoning from SVA code.
        
        Args:
            sva_code: SVA property/assertion code
            
        Returns:
            Markdown-formatted Chain-of-Thought
        """
        # Parse the SVA
        structure = self.parser.parse(sva_code)
        
        # Build each section
        sections = [
            self._build_header(structure),
            self._build_step1_interface(structure),
            self._build_step2_semantic(structure),
            self._build_step3_sequence(structure),
            self._build_step4_property(structure),
            self._build_step5_final(structure),
        ]
        
        return "\n\n".join(sections)

    def build_from_structure(self, structure: SVAStructure) -> str:
        """
        Build Chain-of-Thought reasoning from pre-parsed SVAStructure.
        
        Args:
            structure: Parsed SVA structure
            
        Returns:
            Markdown-formatted Chain-of-Thought
        """
        sections = [
            self._build_header(structure),
            self._build_step1_interface(structure),
            self._build_step2_semantic(structure),
            self._build_step3_sequence(structure),
            self._build_step4_property(structure),
            self._build_step5_final(structure),
        ]
        
        return "\n\n".join(sections)

    def _build_header(self, structure: SVAStructure) -> str:
        """Build the header section."""
        prop_type = "Assertion"
        if structure.is_assumption:
            prop_type = "Assumption"
        elif structure.is_cover:
            prop_type = "Cover"
        
        name = structure.property_name or "Unnamed Property"
        
        return f"# SVA Generation Chain-of-Thought\n\n**Property:** {name}\n**Type:** {prop_type}"

    def _build_step1_interface(self, structure: SVAStructure) -> str:
        """Build Step 1: Interface & Clock Domain Analysis."""
        lines = ["## Step 1: Interface & Clock Domain Analysis"]
        
        # Signals classification
        lines.append("\n* **Signals:**")
        
        clock_signals = [s for s in structure.signals if s.is_clock]
        reset_signals = [s for s in structure.signals if s.is_reset]
        other_signals = [s for s in structure.signals if not s.is_clock and not s.is_reset]
        
        if other_signals:
            signal_names = ", ".join(f"`{s.name}`" for s in sorted(other_signals, key=lambda x: x.name))
            lines.append(f"    * Design Signals: {signal_names}")
        
        # Clock and reset
        lines.append("\n* **Clocks & Resets:**")
        
        if structure.clock_signal:
            lines.append(f"    * Primary Clock: `{structure.clock_signal}` ({structure.clock_edge})")
        else:
            lines.append("    * Primary Clock: Not specified (synchronous design assumed)")
        
        if structure.reset_signal:
            active = "Active Low" if structure.reset_active_low else "Active High"
            lines.append(f"    * Reset Signal: `{structure.reset_signal}` ({active})")
        elif structure.disable_condition:
            lines.append(f"    * Disable Condition: `{structure.disable_condition}`")
        
        # Cross-domain check
        lines.append("\n* **Cross-Domain Check:** ")
        if len(clock_signals) > 1:
            lines.append("    * Multiple clock domains detected - synchronization may be needed")
        else:
            lines.append("    * Single clock domain - no CDC concerns")
        
        return "\n".join(lines)

    def _build_step2_semantic(self, structure: SVAStructure) -> str:
        """Build Step 2: Semantic Mapping."""
        lines = ["## Step 2: Semantic Mapping (Primitives & Built-ins)"]
        
        # Boolean conditions
        lines.append("\n* **Boolean Conditions:**")
        if structure.antecedent:
            lines.append(f"    * Trigger condition: `{structure.antecedent}`")
        if structure.consequent:
            lines.append(f"    * Response condition: `{structure.consequent}`")
        
        # Built-in functions
        if structure.builtin_functions:
            lines.append("\n* **Edge/Change Detection & Built-in Functions:**")
            for func in structure.builtin_functions:
                desc = self.BUILTIN_DESCRIPTIONS.get(func.name, "Custom function")
                args_str = ", ".join(func.arguments) if func.arguments else ""
                lines.append(f"    * `{func.name}({args_str})`: {desc}")
        
        # Past values
        past_funcs = [f for f in structure.builtin_functions if f.name == "$past"]
        if past_funcs:
            lines.append("\n* **Past Values:**")
            for func in past_funcs:
                if len(func.arguments) >= 2:
                    lines.append(f"    * Reference to `{func.arguments[0]}` from {func.arguments[1]} cycles ago")
                elif func.arguments:
                    lines.append(f"    * Reference to previous value of `{func.arguments[0]}`")
        
        return "\n".join(lines)

    def _build_step3_sequence(self, structure: SVAStructure) -> str:
        """Build Step 3: Sequence Construction."""
        lines = ["## Step 3: Sequence Construction (Bottom-Up)"]
        
        # Antecedent sequence
        if structure.antecedent:
            lines.append("\n* **Sequence A (Trigger/Antecedent):**")
            lines.append(f"    * Description: The triggering condition that initiates property evaluation")
            lines.append(f"    * Logic: `{structure.antecedent}`")
            
            # Analyze antecedent for temporal constructs
            ant_delays = self._extract_delays_from_expr(structure.antecedent)
            if ant_delays:
                lines.append(f"    * Timing: {self._describe_delays(ant_delays)}")
        
        # Consequent sequence
        if structure.consequent:
            lines.append("\n* **Sequence B (Response/Consequent):**")
            lines.append(f"    * Description: The expected behavior when trigger occurs")
            lines.append(f"    * Logic: `{structure.consequent}`")
            
            # Analyze consequent for temporal constructs
            cons_delays = self._extract_delays_from_expr(structure.consequent)
            if cons_delays:
                lines.append(f"    * Timing: {self._describe_delays(cons_delays)}")
            
            # Check for nested logic
            if self._has_nested_logic(structure.consequent):
                lines.append("    * **Handling Nesting:** Complex nested temporal logic detected")
        
        # Temporal operators used
        if structure.temporal_operators:
            lines.append("\n* **Temporal Operators Used:**")
            for op in structure.temporal_operators:
                desc = self.TEMPORAL_DESCRIPTIONS.get(op, "Unknown operator")
                lines.append(f"    * `{op.value}`: {desc}")
        
        return "\n".join(lines)

    def _build_step4_property(self, structure: SVAStructure) -> str:
        """Build Step 4: Property Assembly."""
        lines = ["## Step 4: Property Assembly"]
        
        # Implication type
        lines.append("\n* **Implication Type:**")
        if structure.implication_type == ImplicationType.OVERLAPPING:
            lines.append("    * Overlapping (`|->`) - consequent evaluation starts in the same cycle as antecedent match")
        elif structure.implication_type == ImplicationType.NON_OVERLAPPING:
            lines.append("    * Non-overlapping (`|=>`) - consequent evaluation starts one cycle after antecedent match")
        else:
            lines.append("    * No implication - simple property expression")
        
        # Disable condition
        lines.append("\n* **Disable Condition:**")
        if structure.disable_condition:
            lines.append(f"    * `disable iff ({structure.disable_condition})`")
            if structure.reset_signal:
                active = "low" if structure.reset_active_low else "high"
                lines.append(f"    * Property is disabled when reset `{structure.reset_signal}` is active {active}")
        else:
            lines.append("    * None specified - property always active")
        
        # Assertion structure
        lines.append("\n* **Assertion Structure:**")
        if structure.clock_signal and structure.implication_type.value:
            lines.append(f"    * `@({structure.clock_edge} {structure.clock_signal}) "
                        f"{structure.antecedent} {structure.implication_type.value} {structure.consequent}`")
        elif structure.clock_signal:
            # Simple property without implication
            body = structure.antecedent or structure.consequent or "property_expression"
            lines.append(f"    * `@({structure.clock_edge} {structure.clock_signal}) {body}`")
        
        return "\n".join(lines)

    def _build_step5_final(self, structure: SVAStructure) -> str:
        """Build Step 5: Final SVA Code."""
        lines = ["## Step 5: Final SVA Code"]
        
        lines.append("\n```systemverilog")
        lines.append(structure.raw_code.strip())
        lines.append("```")
        
        # Add summary
        lines.append("\n**Summary:**")
        
        summary_parts = []
        if structure.property_name:
            summary_parts.append(f"Property `{structure.property_name}`")
        
        if structure.implication_type == ImplicationType.OVERLAPPING:
            summary_parts.append("uses overlapping implication")
        elif structure.implication_type == ImplicationType.NON_OVERLAPPING:
            summary_parts.append("uses non-overlapping implication")
        
        if structure.clock_signal:
            summary_parts.append(f"synchronized to `{structure.clock_signal}`")
        
        if structure.builtin_functions:
            func_names = set(f.name for f in structure.builtin_functions)
            summary_parts.append(f"using built-in functions: {', '.join(func_names)}")
        
        if summary_parts:
            lines.append(" ".join(summary_parts) + ".")
        
        return "\n".join(lines)

    def _extract_delays_from_expr(self, expr: str) -> List[DelayRange]:
        """Extract delay specifications from an expression."""
        delays = []
        
        # Exact delay
        for match in re.finditer(r'##(\d+)', expr):
            n = int(match.group(1))
            delays.append(DelayRange(min_cycles=n, max_cycles=n))
        
        # Range delay
        for match in re.finditer(r'##\[(\d+):(\d+|\$)\]', expr):
            min_cycles = int(match.group(1))
            max_str = match.group(2)
            if max_str == '$':
                delays.append(DelayRange(min_cycles=min_cycles, is_unbounded=True))
            else:
                delays.append(DelayRange(min_cycles=min_cycles, max_cycles=int(max_str)))
        
        return delays

    def _describe_delays(self, delays: List[DelayRange]) -> str:
        """Generate human-readable delay description."""
        descriptions = []
        for delay in delays:
            if delay.is_unbounded:
                descriptions.append(f"{delay.min_cycles} to unbounded cycles")
            elif delay.max_cycles and delay.max_cycles != delay.min_cycles:
                descriptions.append(f"{delay.min_cycles} to {delay.max_cycles} cycles")
            else:
                descriptions.append(f"{delay.min_cycles} cycle(s)")
        return ", ".join(descriptions)

    def _has_nested_logic(self, expr: str) -> bool:
        """Check if expression has nested temporal logic."""
        # Check for nested parentheses with temporal operators
        nesting_indicators = [
            r'\([^)]*\[\*',  # Repetition inside parens
            r'\([^)]*##',    # Delay inside parens
            r'throughout',
            r'within',
            r'intersect',
        ]
        for pattern in nesting_indicators:
            if re.search(pattern, expr):
                return True
        return False

    def get_cot_sections(self, sva_code: str) -> List[CoTSection]:
        """
        Get CoT as a list of sections for programmatic access.
        
        Args:
            sva_code: SVA code
            
        Returns:
            List of CoTSection objects
        """
        structure = self.parser.parse(sva_code)
        
        return [
            CoTSection("Header", self._build_header(structure)),
            CoTSection("Interface & Clock Domain Analysis", self._build_step1_interface(structure)),
            CoTSection("Semantic Mapping", self._build_step2_semantic(structure)),
            CoTSection("Sequence Construction", self._build_step3_sequence(structure)),
            CoTSection("Property Assembly", self._build_step4_property(structure)),
            CoTSection("Final SVA Code", self._build_step5_final(structure)),
        ]
