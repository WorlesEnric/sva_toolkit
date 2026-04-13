"""
SVA Generator module.

Provides type-directed synthesis of syntactically legal SystemVerilog Assertions.
"""

from sva_toolkit.gen.generator import (
    SVASynthesizer,
    GenerationResult,
    ValidationResult,
    SVAProperty,
)
from sva_toolkit.gen.stratified import (
    StratifiedGenerator,
)
from sva_toolkit.gen.types_sva import (
    SVANode,
    SVAType,
    Signal,
    BinaryOp,
    UnarySysFunction,
    SequenceDelay,
    SequenceRepeat,
    SequenceBinary,
    Implication,
    DisableIff,
    NotProperty,
    TYPE_EXPR,
    TYPE_BOOL,
    TYPE_SEQUENCE,
    TYPE_PROPERTY,
)
from sva_toolkit.gen.templates import (
    generate_sv_module,
    generate_minimal_wrapper,
    generate_assertion_only,
    generate_cover_property,
    generate_assume_property,
)
from sva_toolkit.gen.utils import (
    weighted_choice,
    get_random_delay,
    get_random_repeat_count,
    generate_signal_list,
    DEFAULT_SIGNALS,
    HANDSHAKE_SIGNALS,
    FIFO_SIGNALS,
    AXI_SIGNALS,
)

__all__ = [
    # Main classes
    "SVASynthesizer",
    "GenerationResult",
    "ValidationResult",
    "SVAProperty",
    "StratifiedGenerator",
    # Type nodes
    "SVANode",
    "SVAType",
    "Signal",
    "BinaryOp",
    "UnarySysFunction",
    "SequenceDelay",
    "SequenceRepeat",
    "SequenceBinary",
    "Implication",
    "DisableIff",
    "NotProperty",
    # Type constants
    "TYPE_EXPR",
    "TYPE_BOOL",
    "TYPE_SEQUENCE",
    "TYPE_PROPERTY",
    # Templates
    "generate_sv_module",
    "generate_minimal_wrapper",
    "generate_assertion_only",
    "generate_cover_property",
    "generate_assume_property",
    # Utilities
    "weighted_choice",
    "get_random_delay",
    "get_random_repeat_count",
    "generate_signal_list",
    # Signal presets
    "DEFAULT_SIGNALS",
    "HANDSHAKE_SIGNALS",
    "FIFO_SIGNALS",
    "AXI_SIGNALS",
]

