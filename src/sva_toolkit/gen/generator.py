"""
SVA Generator - Type-directed synthesis engine for SystemVerilog Assertions.

This module provides the SVASynthesizer class that generates syntactically
legal SVA properties using type-directed synthesis.
"""

import random
import subprocess
import tempfile
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from sva_toolkit.gen.types_sva import (
    SVANode,
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


@dataclass
class ValidationResult:
    """
    @brief Result of SVA syntax validation.
    """
    is_valid: bool
    error_message: str = ""
    error_line: Optional[int] = None


@dataclass
class GenerationResult:
    """
    @brief Result of SVA generation including validation status.
    """
    properties: List[str] = field(default_factory=list)
    module_code: str = ""
    validation: Optional[ValidationResult] = None
    valid_count: int = 0
    invalid_count: int = 0


class SVASynthesizer:
    """
    @brief Type-directed synthesis engine for SVA generation.
    
    This class generates syntactically legal SystemVerilog Assertions
    by following the SVA promotion hierarchy:
    Expressions -> Booleans -> Sequences -> Properties
    """

    # Default operators
    BOOL_OPS: List[str] = ["==", "!=", "&&", "||"]
    REL_OPS: List[str] = [">", "<", ">=", "<="]
    SYS_FUNCS: List[str] = ["$rose", "$fell", "$stable"]
    SEQ_BIN_OPS: List[str] = ["intersect", "within", "throughout", "and", "or"]
    IMPLICATIONS: List[str] = ["|->", "|=>"]
    REPEAT_OPS: List[str] = ["[*", "[=", "[->"]
    ARITH_OPS: List[str] = ["+", "-", "&", "|"]

    def __init__(
        self,
        signals: List[str],
        max_depth: int = 3,
        clock_signal: str = "clk",
        verible_path: Optional[str] = None
    ) -> None:
        """
        @brief Initialize the SVA synthesizer.
        @param signals List of signal names to use in generated assertions
        @param max_depth Maximum recursion depth for expression generation
        @param clock_signal Clock signal name for property clocking
        @param verible_path Path to verible-verilog-syntax binary for validation
        """
        self.signals: List[str] = signals
        self.max_depth: int = max_depth
        self.clock_signal: str = clock_signal
        self.verible_path: Optional[str] = verible_path
        if self.verible_path is None:
            import pathlib
            project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            self.verible_path = str(
                project_root / "3rd_party" / "verible_bin" / "verible-verilog-syntax"
            )

    def _get_random_signal(self) -> Signal:
        """
        @brief Get a random signal from the available signals.
        @return Signal node
        """
        return Signal(random.choice(self.signals))

    def generate_expr(self, depth: int) -> SVANode:
        """
        @brief Generate an expression (TYPE_EXPR).
        @param depth Current recursion depth
        @return SVANode of type EXPR (Signal or Binary Arithmetic/Bitwise)
        """
        if depth >= self.max_depth or random.random() > 0.6:
            return self._get_random_signal()
        op = random.choice(self.ARITH_OPS)
        return BinaryOp(
            self.generate_expr(depth + 1),
            op,
            self.generate_expr(depth + 1),
            TYPE_EXPR
        )

    def generate_bool(self, depth: int) -> SVANode:
        """
        @brief Generate a boolean expression (TYPE_BOOL).
        @param depth Current recursion depth
        @return SVANode of type BOOL (Logical comparisons or system functions)
        """
        choice = random.random()
        if depth >= self.max_depth or choice < 0.3:
            # Leaf: A signal or a system function call
            if random.random() < 0.5:
                return UnarySysFunction(
                    random.choice(self.SYS_FUNCS),
                    self._get_random_signal()
                )
            return self._get_random_signal()
        elif choice < 0.7:
            # Binary comparison
            op = random.choice(self.BOOL_OPS + self.REL_OPS)
            return BinaryOp(
                self.generate_expr(depth + 1),
                op,
                self.generate_expr(depth + 1),
                TYPE_BOOL
            )
        else:
            # Nested logical
            op = random.choice(["&&", "||"])
            return BinaryOp(
                self.generate_bool(depth + 1),
                op,
                self.generate_bool(depth + 1),
                TYPE_BOOL
            )

    def generate_sequence(self, depth: int) -> SVANode:
        """
        @brief Generate a sequence (TYPE_SEQUENCE).
        @param depth Current recursion depth
        @return SVANode of type SEQUENCE (Temporal chains)
        """
        choice = random.random()
        if depth >= self.max_depth or choice < 0.3:
            # Base case: A sequence can be a simple boolean
            return self.generate_bool(depth + 1)
        elif choice < 0.6:
            # Sequence Delay: seq ##[n:m] seq
            d_min = random.randint(0, 2)
            d_max = d_min + random.randint(0, 3)
            delay = f"##{d_min}" if d_min == d_max else f"##[{d_min}:{d_max}]"
            return SequenceDelay(
                self.generate_sequence(depth + 1),
                delay,
                self.generate_sequence(depth + 1)
            )
        elif choice < 0.8:
            # Sequence Repeat: seq [* n]
            op = random.choice(self.REPEAT_OPS)
            count = str(random.randint(1, 5))
            return SequenceRepeat(
                self.generate_sequence(depth + 1),
                op,
                count + "]"
            )
        else:
            # Sequence Binary Op: seq intersect seq
            op = random.choice(self.SEQ_BIN_OPS)
            return SequenceBinary(
                self.generate_sequence(depth + 1),
                op,
                self.generate_sequence(depth + 1)
            )

    def generate_property(self, depth: int = 0) -> SVANode:
        """
        @brief Generate a property (TYPE_PROPERTY).
        @param depth Current recursion depth
        @return SVANode of type PROPERTY (The top level)
        """
        choice = random.random()
        # Most rich SVA use implications
        if choice < 0.8:
            ante = self.generate_sequence(depth + 1)
            cons = self.generate_sequence(depth + 1)
            op = random.choice(self.IMPLICATIONS)
            prop = Implication(ante, op, cons)
        else:
            # Direct sequence as property or "not" property
            prop = self.generate_sequence(depth + 1)
            if random.random() < 0.5:
                prop = NotProperty(prop)
        # 20% chance to wrap in a disable iff
        if random.random() < 0.2:
            reset = self._get_random_signal()
            prop = DisableIff(reset, prop)
        return prop

    def synthesize(self, name: str = "p_gen") -> str:
        """
        @brief Generate a complete property block.
        @param name Property name
        @return Complete SVA property string block
        """
        prop_logic = self.generate_property()
        return (
            f"property {name};\n"
            f"  @(posedge {self.clock_signal}) {prop_logic};\n"
            f"endproperty"
        )

    def generate_module(
        self,
        module_name: str,
        num_assertions: int,
        include_assertions: bool = True
    ) -> Tuple[str, List[str]]:
        """
        @brief Generate a complete SystemVerilog module with properties.
        @param module_name Name of the generated module
        @param num_assertions Number of assertions to generate
        @param include_assertions Whether to include assert property statements
        @return Tuple of (complete module code, list of property strings)
        """
        properties: List[str] = []
        for i in range(num_assertions):
            prop_str = self.synthesize(name=f"p_gen_{i}")
            properties.append(prop_str)
        signal_decls = "\n  ".join([f"logic {sig};" for sig in self.signals])
        properties_block = "\n\n  ".join(properties)
        if include_assertions:
            assertions = "\n  ".join([
                f"assert_p{i}: assert property (p_gen_{i});"
                for i in range(num_assertions)
            ])
        else:
            assertions = ""
        module_code = f"""module {module_name} (
    input logic {self.clock_signal},
    input logic rst_n
);

  // Signal Declarations
  {signal_decls}

  // Generated Properties
  {properties_block}

  // Assertion Instances
  {assertions}

endmodule
"""
        return module_code, properties

    def validate_syntax(self, code: str) -> ValidationResult:
        """
        @brief Validate SVA syntax using Verible.
        @param code SystemVerilog code to validate
        @return ValidationResult with is_valid flag and any error messages
        """
        if not self.verible_path or not os.path.exists(self.verible_path):
            return ValidationResult(
                is_valid=False,
                error_message=f"Verible not found at: {self.verible_path}"
            )
        try:
            result = subprocess.run(
                [self.verible_path, "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return ValidationResult(is_valid=True)
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                return ValidationResult(is_valid=False, error_message=error_msg)
        except subprocess.TimeoutExpired:
            return ValidationResult(
                is_valid=False,
                error_message="Verible validation timed out"
            )
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                error_message=f"Verible not found at: {self.verible_path}"
            )

    def validate_properties(self, properties: List[str]) -> ValidationResult:
        """
        @brief Validate a list of SVA properties by wrapping in a dummy module.
        @param properties List of property strings to validate
        @return ValidationResult with is_valid flag and any error messages
        """
        # Create a dummy module wrapping all properties
        properties_block = "\n".join(properties)
        dummy_module = f"""module sva_validation_wrapper (
    input logic {self.clock_signal}
);
  // Dummy signals
  logic dummy_sig;
{properties_block}
endmodule
"""
        return self.validate_syntax(dummy_module)

    def generate_validated(
        self,
        module_name: str,
        num_assertions: int,
        max_retries: int = 3
    ) -> GenerationResult:
        """
        @brief Generate SVA module with syntax validation.
        @param module_name Name of the generated module
        @param num_assertions Number of assertions to generate
        @param max_retries Maximum retries on validation failure
        @return GenerationResult with properties, module code, and validation status
        """
        for attempt in range(max_retries):
            module_code, properties = self.generate_module(
                module_name, num_assertions
            )
            validation = self.validate_syntax(module_code)
            if validation.is_valid:
                return GenerationResult(
                    properties=properties,
                    module_code=module_code,
                    validation=validation,
                    valid_count=num_assertions,
                    invalid_count=0
                )
        # Return last attempt even if invalid
        return GenerationResult(
            properties=properties,
            module_code=module_code,
            validation=validation,
            valid_count=0,
            invalid_count=num_assertions
        )
