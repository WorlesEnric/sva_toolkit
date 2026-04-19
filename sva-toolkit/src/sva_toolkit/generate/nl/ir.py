"""
Intermediate Representation for SVA Semantics.

This module defines the IR that sits between SVA AST and natural language.
The IR captures the semantic meaning of SVA properties in a way that's
easier to convert to natural language than the raw AST.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ImplicationType(Enum):
    """Types of implications in SVA."""
    OVERLAPPING = "|->"  # Same cycle
    NON_OVERLAPPING = "|=>"  # Next cycle
    NONE = None


class TemporalType(Enum):
    """Types of temporal delays."""
    IMMEDIATE = "immediate"
    NEXT_CYCLE = "next"
    FIXED = "fixed"
    RANGE = "range"
    UNBOUNDED = "unbounded"


@dataclass
class TimingSpec:
    """Specification of temporal timing."""
    delay_type: TemporalType
    min_cycles: Optional[int] = None
    max_cycles: Optional[int] = None

    def to_natural_language(self) -> str:
        """Convert timing specification to natural language."""
        if self.delay_type == TemporalType.IMMEDIATE:
            return "immediately"
        if self.delay_type == TemporalType.NEXT_CYCLE:
            return "in the next cycle"
        if self.delay_type == TemporalType.FIXED:
            if self.min_cycles == 1:
                return "one cycle later"
            return f"{self.min_cycles} cycles later"
        if self.delay_type == TemporalType.RANGE:
            if self.min_cycles == self.max_cycles:
                return f"exactly {self.min_cycles} cycles later"
            if self.max_cycles is None or self.max_cycles == float('inf'):
                return f"at least {self.min_cycles} cycles later"
            return f"between {self.min_cycles} and {self.max_cycles} cycles later"
        if self.delay_type == TemporalType.UNBOUNDED:
            if self.min_cycles == 0 or self.min_cycles is None:
                return "eventually"
            return f"at least {self.min_cycles} cycles later"
        return "at some point"


@dataclass
class SVASemantics:
    """
    Intermediate representation of SVA meaning.

    This captures the semantic intent of an SVA property in a way
    that's easier to convert to natural language than the raw AST.
    """

    # Core semantic roles
    description: str = ""  # Primary description
    context: Optional[str] = None  # Precondition/execution context
    trigger: Optional[str] = None  # Triggering event
    outcome: Optional[str] = None  # Required outcome

    # Temporal
    timing: Optional[TimingSpec] = None
    implication: ImplicationType = ImplicationType.NONE

    # Logical structure
    logical_op: Optional[str] = None  # "and", "or"
    components: List['SVASemantics'] = field(default_factory=list)

    # Modifiers
    negated: bool = False
    disabled_when: Optional[str] = None
    conditional_on: Optional[str] = None
    conditional_else: Optional[str] = None

    # Metadata
    complexity: str = "simple"  # "simple", "moderate", "complex"
    node_type: Optional[str] = None  # For debugging

    def is_compound(self) -> bool:
        """Check if this is a compound property with multiple components."""
        return len(self.components) > 0

    def has_implication(self) -> bool:
        """Check if this property involves an implication."""
        return self.implication != ImplicationType.NONE

    def has_timing(self) -> bool:
        """Check if this property has explicit timing constraints."""
        return self.timing is not None
