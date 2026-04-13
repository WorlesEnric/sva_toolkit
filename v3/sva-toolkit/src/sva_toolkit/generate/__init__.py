"""Public exports for the SVA generator package."""

from sva_toolkit.generate.coverage import compute_coverage_statistics
from sva_toolkit.generate.signal_presets import (
    AXI_SIGNALS,
    DEFAULT_SIGNALS,
    FIFO_SIGNALS,
    HANDSHAKE_SIGNALS,
)
from sva_toolkit.generate.stratified import StratifiedGenerator
from sva_toolkit.generate.synthesizer import GenerationResult, SVAProperty, SVASynthesizer, ValidationResult
from sva_toolkit.generate.templates import (
    generate_assertion_only,
    generate_assume_property,
    generate_cover_property,
    generate_minimal_wrapper,
    generate_sv_module,
)
from sva_toolkit.generate.types import (
    BinaryOp,
    DisableIff,
    Implication,
    NotProperty,
    SVAType,
    SVANode,
    SequenceBinary,
    SequenceDelay,
    SequenceRepeat,
    Signal,
    TYPE_BOOL,
    TYPE_EXPR,
    TYPE_PROPERTY,
    TYPE_SEQUENCE,
    UnarySysFunction,
)
from sva_toolkit.generate.utils import generate_signal_list, get_random_delay, get_random_repeat_count, weighted_choice

__all__ = [
    "AXI_SIGNALS",
    "BinaryOp",
    "DEFAULT_SIGNALS",
    "DisableIff",
    "FIFO_SIGNALS",
    "GenerationResult",
    "HANDSHAKE_SIGNALS",
    "Implication",
    "NotProperty",
    "SVAProperty",
    "SVANode",
    "SVASynthesizer",
    "SVAType",
    "SequenceBinary",
    "SequenceDelay",
    "SequenceRepeat",
    "Signal",
    "StratifiedGenerator",
    "TYPE_BOOL",
    "TYPE_EXPR",
    "TYPE_PROPERTY",
    "TYPE_SEQUENCE",
    "UnarySysFunction",
    "ValidationResult",
    "compute_coverage_statistics",
    "generate_assertion_only",
    "generate_assume_property",
    "generate_cover_property",
    "generate_minimal_wrapper",
    "generate_signal_list",
    "generate_sv_module",
    "get_random_delay",
    "get_random_repeat_count",
    "weighted_choice",
]
