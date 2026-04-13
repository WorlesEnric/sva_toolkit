"""
Coverage utilities for SVA datasets.

This module mirrors the statistics produced by examples/coverage_analysis.py
so that other components (e.g., CLI exports) can annotate generated datasets
with syntax coverage metadata.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from collections import Counter, defaultdict

# Define all SVA constructs we want to track
SVA_CONSTRUCTS: Dict[str, str] = {
    # Implications
    "|->": "Overlapping implication",
    "|=>": "Non-overlapping implication",

    # Property operators
    " and ": "Property AND",
    " or ": "Property OR",
    "until": "Until operator",
    "until_with": "Until with operator",
    "not ": "NOT property",
    "disable iff": "Disable iff",
    "if ": "If-else property",

    # Sequence operators
    "##": "Delay operator",
    "[*": "Consecutive repetition",
    "[=": "Non-consecutive repetition",
    "[->": "Goto repetition",
    "intersect": "Intersect operator",
    "throughout": "Throughout operator",
    "first_match": "First match",
    ".ended": "Sequence ended",

    # System functions
    "$rose": "Rose function",
    "$fell": "Fell function",
    "$stable": "Stable function",
    "$changed": "Changed function",
    "$past": "Past function",
    "$onehot": "Onehot function",
    "$onehot0": "Onehot0 function",
    "$isunknown": "Is unknown function",
    "$countones": "Count ones function",

    # Boolean operators
    "&&": "Logical AND",
    "||": "Logical OR",
    "!": "Logical NOT",

    # Comparison operators
    "==": "Equality",
    "!=": "Inequality",
    "===": "Case equality",
    "!==": "Case inequality",
    ">": "Greater than",
    "<": "Less than",
    ">=": "Greater or equal",
    "<=": "Less or equal",

    # Arithmetic operators
    " + ": "Addition",
    " - ": "Subtraction",
    " * ": "Multiplication",
    " / ": "Division",
    " % ": "Modulo",

    # Bitwise operators
    " & ": "Bitwise AND",
    " | ": "Bitwise OR",
    " ^ ": "Bitwise XOR",
    "^~": "Bitwise XNOR",
    "~^": "Bitwise XNOR alt",
    "~": "Bitwise NOT",
}

CATEGORIES: Dict[str, List[str]] = {
    "Property Operators": ["|->", "|=>", " and ", " or ", "until", "until_with", "not ", "disable iff", "if "],
    "Sequence Operators": ["##", "[*", "[=", "[->", "intersect", "throughout", "first_match", ".ended"],
    "System Functions": ["$rose", "$fell", "$stable", "$changed", "$past", "$onehot", "$onehot0", "$isunknown", "$countones"],
    "Boolean Operators": ["&&", "||", "!"],
    "Comparison Operators": ["==", "!=", "===", "!==", ">", "<", ">=", "<="],
    "Arithmetic Operators": [" + ", " - ", " * ", " / ", " % "],
    "Bitwise Operators": [" & ", " | ", " ^ ", "^~", "~^", "~"],
}


def _count_constructs(sva_code: str) -> Counter:
    """Count occurrences of each construct in SVA code."""
    counts = Counter()

    for construct in SVA_CONSTRUCTS.keys():
        if construct in sva_code:
            counts[construct] = sva_code.count(construct)

    return counts


def compute_coverage_statistics(properties: List[str]) -> Dict[str, object]:
    """
    Compute coverage metadata for a dataset of SVA strings.

    Returns:
        Dict containing summary stats, per-category detail, and missing constructs.
    """
    total_counts: Counter = Counter()
    properties_with_construct = defaultdict(int)

    for sva in properties:
        prop_counts = _count_constructs(sva)
        for construct, count in prop_counts.items():
            total_counts[construct] += count
            properties_with_construct[construct] += 1

    total_properties = len(properties)
    covered_constructs = sum(
        1 for construct in SVA_CONSTRUCTS.keys()
        if properties_with_construct.get(construct, 0) > 0
    )

    coverage_pct = (
        (covered_constructs / len(SVA_CONSTRUCTS)) * 100
        if SVA_CONSTRUCTS else 0.0
    )

    missing_constructs = [
        {
            "key": construct,
            "description": SVA_CONSTRUCTS[construct]
        }
        for construct in SVA_CONSTRUCTS.keys()
        if properties_with_construct.get(construct, 0) == 0
    ]

    category_details = {
        category: [
            {
                "key": construct,
                "description": SVA_CONSTRUCTS.get(construct, construct),
                "properties_with_construct": properties_with_construct.get(construct, 0),
                "occurrences": total_counts.get(construct, 0),
                "coverage_pct": (
                    (properties_with_construct.get(construct, 0) / total_properties) * 100
                    if total_properties else 0.0
                )
            }
            for construct in constructs
        ]
        for category, constructs in CATEGORIES.items()
    }

    return {
        "total_properties": total_properties,
        "constructs_total": len(SVA_CONSTRUCTS),
        "constructs_covered": covered_constructs,
        "coverage_pct": coverage_pct,
        "missing_constructs": missing_constructs,
        "categories": category_details,
    }
