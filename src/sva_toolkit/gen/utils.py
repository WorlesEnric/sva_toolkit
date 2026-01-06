"""
SVA Generation Utilities - Helper functions for random SVA generation.

This module provides utility functions for weighted random choices
and generating random delay/repetition specifications.
"""

import random
from typing import Dict, TypeVar, List

T = TypeVar('T')


def weighted_choice(choices_dict: Dict[T, float]) -> T:
    """
    @brief Select a choice based on weighted probability.
    @param choices_dict Dictionary mapping choices to their weights
    @return Selected choice based on probability distribution
    """
    total = sum(choices_dict.values())
    r = random.uniform(0, total)
    upto = 0.0
    for key, weight in choices_dict.items():
        if upto + weight >= r:
            return key
        upto += weight
    return list(choices_dict.keys())[0]


def get_random_delay(min_val: int = 0, max_val: int = 10) -> str:
    """
    @brief Generate a random SVA delay string.
    @param min_val Minimum delay value
    @param max_val Maximum delay value
    @return SVA delay string (e.g., "##1" or "##[1:5]")
    """
    d1 = random.randint(min_val, max_val)
    if random.random() > 0.7:  # 30% chance for a range delay
        d2 = d1 + random.randint(1, 5)
        return f"##[{d1}:{d2}]"
    return f"##{d1}"


def get_random_repeat_count(max_val: int = 5) -> str:
    """
    @brief Generate a random SVA repeat count.
    @param max_val Maximum repeat value
    @return SVA repeat count string (e.g., "3" or "1:5")
    """
    c1 = random.randint(1, max_val)
    if random.random() > 0.8:  # 20% chance for a range
        c2 = c1 + random.randint(1, 3)
        return f"{c1}:{c2}"
    return str(c1)


def generate_signal_list(
    base_names: List[str],
    count: int,
    prefix: str = ""
) -> List[str]:
    """
    @brief Generate a list of signal names.
    @param base_names Base signal names to use
    @param count Number of signals to generate
    @param prefix Optional prefix to add to each signal
    @return List of signal names
    """
    signals: List[str] = []
    for i in range(count):
        base = base_names[i % len(base_names)]
        if prefix:
            signals.append(f"{prefix}_{base}_{i}")
        else:
            signals.append(f"{base}_{i}" if count > len(base_names) else base)
    return signals


# Default signal sets for different domains
DEFAULT_SIGNALS: List[str] = [
    "req", "ack", "gnt", "valid", "ready", "data_en", "busy"
]

HANDSHAKE_SIGNALS: List[str] = [
    "req", "ack", "valid", "ready", "grant", "request"
]

FIFO_SIGNALS: List[str] = [
    "push", "pop", "full", "empty", "almost_full", "almost_empty"
]

AXI_SIGNALS: List[str] = [
    "awvalid", "awready", "wvalid", "wready", "bvalid", "bready",
    "arvalid", "arready", "rvalid", "rready"
]
