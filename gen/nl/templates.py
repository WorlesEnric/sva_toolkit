"""
Scenario-based templates for natural language generation.

This module defines narrative templates that convert SVA IR to natural language.
"""

from typing import Optional
from sva_toolkit.gen.nl.ir import SVASemantics, ImplicationType


class TemplateRegistry:
    """Registry of narrative templates for different SVA patterns."""

    def realize(self, semantics: SVASemantics) -> str:
        """
        Convert semantics to natural language using templates.

        Selects the appropriate template based on the IR structure
        and generates fluent English text.
        """
        # Handle conditional properties
        if semantics.conditional_on and semantics.conditional_else:
            return self._realize_conditional(semantics)

        # Handle implications
        if semantics.has_implication():
            return self._realize_implication(semantics)

        # Handle compound logical properties
        if semantics.is_compound() and semantics.logical_op:
            return self._realize_logical(semantics)

        # Handle negated properties
        if semantics.negated:
            return self._realize_negated(semantics)

        # Default: simple description
        return self._realize_simple(semantics)

    def _realize_implication(self, sem: SVASemantics) -> str:
        """
        Template: When {trigger}, {outcome} must occur {timing}.

        Handles both overlapping (|->) and non-overlapping (|=>) implications.
        """
        trigger = sem.trigger or "the condition is met"
        outcome = sem.outcome or "the property holds"

        # Build the base sentence
        parts = []

        # Add trigger clause
        parts.append(f"When {trigger}")

        # Add outcome with timing
        if sem.implication == ImplicationType.OVERLAPPING:
            if sem.timing and sem.timing.to_natural_language() == "immediately":
                parts.append(f"{outcome} must occur in the same cycle")
            else:
                parts.append(f"{outcome} must be asserted in the same cycle")
        elif sem.implication == ImplicationType.NON_OVERLAPPING:
            parts.append(f"{outcome} must occur in the next cycle")
        else:
            # Fallback
            timing_str = sem.timing.to_natural_language() if sem.timing else ""
            parts.append(f"{outcome} must occur{' ' + timing_str if timing_str else ''}")

        sentence = ", ".join(parts) + "."

        # Add disable clause if present
        if sem.disabled_when:
            sentence = sentence[:-1]  # Remove period
            sentence += f" (disabled when {sem.disabled_when})."

        return sentence

    def _realize_logical(self, sem: SVASemantics) -> str:
        """
        Template: {A} and/or {B}.

        Handles compound properties with logical operators.
        """
        if not sem.components:
            return sem.description + "."

        # Realize each component
        component_texts = []
        for comp in sem.components:
            comp_text = self.realize(comp)
            # Remove trailing period for joining
            if comp_text.endswith('.'):
                comp_text = comp_text[:-1]
            component_texts.append(comp_text)

        # Join with logical operator
        if sem.logical_op == "and":
            result = " and ".join(component_texts)
        elif sem.logical_op == "or":
            result = " or ".join(component_texts)
        else:
            result = f" {sem.logical_op} ".join(component_texts)

        return f"({result})."

    def _realize_conditional(self, sem: SVASemantics) -> str:
        """
        Template: If {condition}, then {outcome}; otherwise {else}.

        Handles conditional properties (if-else).
        """
        parts = [f"If {sem.conditional_on}, then {sem.outcome}"]

        if sem.conditional_else:
            parts.append(f"otherwise {sem.conditional_else}")

        return "; ".join(parts) + "."

    def _realize_negated(self, sem: SVASemantics) -> str:
        """
        Template: NOT ({property}).

        Handles negated properties.
        """
        if sem.description and "NOT (" not in sem.description:
            return f"NOT ({sem.description})."
        return sem.description + ("." if not sem.description.endswith(".") else "")

    def _realize_simple(self, sem: SVASemantics) -> str:
        """
        Template: {description}.

        Handles simple properties without special structure.
        """
        desc = sem.description

        if not desc:
            desc = "the property holds"

        # Add disable clause if present
        if sem.disabled_when:
            if desc.endswith('.'):
                desc = desc[:-1]
            desc += f" (disabled when {sem.disabled_when})."

        # Ensure proper punctuation
        if not desc.endswith('.'):
            desc += "."

        return desc


class SentenceBuilder:
    """Helper class for building grammatically correct sentences."""

    @staticmethod
    def capitalize_first(text: str) -> str:
        """Capitalize the first letter of the text."""
        if not text:
            return text
        return text[0].upper() + text[1:]

    @staticmethod
    def ensure_period(text: str) -> str:
        """Ensure text ends with a period."""
        text = text.rstrip()
        if not text.endswith('.'):
            text += '.'
        return text

    @staticmethod
    def join_clauses(clauses: list, connector: str = ",") -> str:
        """Join multiple clauses with proper punctuation."""
        if not clauses:
            return ""
        return f"{connector} ".join(clauses)

    @staticmethod
    def add_parenthetical(text: str, parenthetical: str) -> str:
        """Add a parenthetical remark to text."""
        if text.endswith('.'):
            return f"{text[:-1]} ({parenthetical})."
        return f"{text} ({parenthetical})."
