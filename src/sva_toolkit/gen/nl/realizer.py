"""
Natural Language Realizer.

Top-level interface for converting SVA AST to natural language.
"""

from typing import Any
from sva_toolkit.gen.nl.extractor import SemanticExtractor
from sva_toolkit.gen.nl.templates import TemplateRegistry
from sva_toolkit.gen.nl.ir import SVASemantics


class NaturalLanguageRealizer:
    """Main interface for SVA → Natural Language conversion."""

    def __init__(self):
        self.extractor = SemanticExtractor()
        self.templates = TemplateRegistry()

    def realize(self, node: Any) -> str:
        """
        Convert SVA AST node to natural language.

        This is the main entry point for the SVATrans system.
        It performs a three-stage transformation:

        1. Extract semantics from AST (node → IR)
        2. Select and apply appropriate template (IR → text)
        3. Post-process for fluency (text → polished text)

        Args:
            node: SVA AST node (SVANode subclass)

        Returns:
            Natural language description as a string
        """
        # Stage 1: Extract semantics
        semantics = self.extractor.extract(node)

        # Stage 2: Realize using templates
        text = self.templates.realize(semantics)

        # Stage 3: Post-process
        text = self._post_process(text)

        return text

    def _post_process(self, text: str) -> str:
        """
        Apply final polishing to generated text.

        - Ensures proper capitalization
        - Cleans up punctuation
        - Removes extra whitespace
        """
        if not text:
            return text

        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 0 else text

        # Ensure single period at end
        text = text.rstrip('.')
        text += '.'

        # Clean up extra whitespace
        text = ' '.join(text.split())

        # Fix spacing around punctuation
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace('( ', '(')
        text = text.replace(' )', ')')

        return text


# Convenience function for direct use
def sva_to_english(node: Any) -> str:
    """
    Convert SVA node to natural English (convenience function).

    Args:
        node: SVA AST node (SVANode subclass)

    Returns:
        Natural language description

    Example:
        >>> from sva_toolkit.gen.types_sva import Signal, Implication
        >>> req = Signal("req")
        >>> ack = Signal("ack")
        >>> prop = Implication(req, "|->", ack)
        >>> print(sva_to_english(prop))
        When the req signal is asserted, the ack signal must be asserted in the same cycle.
    """
    realizer = NaturalLanguageRealizer()
    return realizer.realize(node)
