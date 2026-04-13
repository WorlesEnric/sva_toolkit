"""
Natural Language Generation for SVA (SVATrans).

This package provides tools to convert SystemVerilog Assertions (SVA)
into natural, readable English descriptions.

Main interfaces:
- sva_to_english(): Convenience function to convert SVA nodes to English
- NaturalLanguageRealizer: Main class for SVA → English conversion
- SemanticExtractor: Extracts semantic IR from SVA AST
- TemplateRegistry: Applies narrative templates to generate text

Example:
    from sva_toolkit.gen.nl import sva_to_english
    from sva_toolkit.gen.types_sva import Signal, Implication

    req = Signal("req")
    ack = Signal("ack")
    prop = Implication(req, "|->", ack)

    description = sva_to_english(prop)
    # Output: "When the req signal is asserted, the ack signal must be asserted in the same cycle."
"""

from sva_toolkit.gen.nl.realizer import NaturalLanguageRealizer, sva_to_english
from sva_toolkit.gen.nl.extractor import SemanticExtractor
from sva_toolkit.gen.nl.templates import TemplateRegistry
from sva_toolkit.gen.nl.ir import (
    SVASemantics,
    TimingSpec,
    TemporalType,
    ImplicationType,
)

__all__ = [
    # Main interfaces
    "sva_to_english",
    "NaturalLanguageRealizer",

    # Core classes
    "SemanticExtractor",
    "TemplateRegistry",

    # IR types
    "SVASemantics",
    "TimingSpec",
    "TemporalType",
    "ImplicationType",
]

__version__ = "0.1.0"
