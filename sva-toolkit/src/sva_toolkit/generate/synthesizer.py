"""Type-directed synthesis for SystemVerilog Assertions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sva_toolkit.generate.rng import GenerationRng
from sva_toolkit.generate.types import (
    SVANode,
    Signal,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    UnarySysFunction,
    PastFunction,
    SequenceDelay,
    SequenceRepeat,
    SequenceBinary,
    SequenceFirstMatch,
    SequenceEnded,
    Implication,
    DisableIff,
    NotProperty,
    PropertyIfElse,
    PropertyUntil,
    PropertyBinary,
    TYPE_EXPR,
    TYPE_BOOL,
)
from sva_toolkit.generate.utils import weighted_choice
from sva_toolkit.runtime.process import run_tool
from sva_toolkit.runtime.tools import ToolRegistry, create_default_registry


@dataclass
class ValidationResult:
    """
    @brief Result of SVA syntax validation.
    """
    is_valid: bool
    error_message: str = ""
    error_line: int | None = None


@dataclass
class SVAProperty:
    """
    @brief Represents an SVA property with its natural language description.
    """
    name: str
    sva_code: str
    svad: str  # SVA Description (natural language)
    property_block: str  # Full property ... endproperty block


@dataclass
class GenerationResult:
    """
    @brief Result of SVA generation including validation status.
    """
    properties: list[SVAProperty] = field(default_factory=list)
    module_code: str = ""
    validation: ValidationResult | None = None
    valid_count: int = 0
    invalid_count: int = 0


class SVASynthesizer:
    """
    @brief Type-directed synthesis engine for SVA generation.

    This class generates syntactically legal SystemVerilog Assertions
    by following the SVA promotion hierarchy:
    Expressions -> Booleans -> Sequences -> Properties
    """

    # Operator lists
    UNARY_OPS: list[str] = ["!", "~"]

    LOGICAL_OPS: list[str] = ["&&", "||"]

    RELATIONAL_OPS: list[str] = [">", "<", ">=", "<="]

    EQUALITY_OPS: list[str] = ["==", "!=", "===", "!=="]

    BITWISE_OPS: list[str] = ["&", "|", "^", "^~", "~^"]

    ARITH_OPS: list[str] = ["+", "-", "*", "/", "%"]

    # System functions
    UNARY_SYS_FUNCS: list[str] = [
        "$rose", "$fell", "$stable", "$changed",
        "$onehot", "$onehot0", "$isunknown", "$countones"
    ]

    # Sequence operators
    SEQ_BIN_OPS: list[str] = ["intersect", "throughout", "and", "or"]

    # Implication operators
    IMPLICATIONS: list[str] = ["|->", "|=>"]

    # Repetition operators
    REPEAT_OPS: list[str] = ["[*", "[=", "[->"]

    # Property operators
    PROPERTY_BIN_OPS: list[str] = ["and", "or"]
    UNTIL_OPS: list[str] = ["until", "until_with"]

    def __init__(
        self,
        signals: list[str],
        max_depth: int = 2,
        clock_signal: str = "clk",
        verible_path: str | None = None,
        enable_advanced_features: bool = True,
        rng: GenerationRng | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """
        @brief Initialize the SVA synthesizer.
        @param signals List of signal names to use in generated assertions
        @param max_depth Maximum recursion depth for expression generation
        @param clock_signal Clock signal name for property clocking
        @param verible_path Path to verible-verilog-syntax binary for validation
        @param enable_advanced_features Enable advanced features (ternary, until, if-else, etc.)
        """
        self.signals: list[str] = signals
        self.max_depth: int = max_depth
        self.clock_signal: str = clock_signal
        self.verible_path: str | None = verible_path
        self.enable_advanced_features: bool = enable_advanced_features
        self.rng = rng or GenerationRng()
        self.tool_registry = tool_registry or create_default_registry()
        self._operator_weights = self._load_operator_weights()
        self._max_ops_per_property = self._parse_max_ops_per_property(self._operator_weights)
        self._remaining_ops_budget: int | None = None
        self._expr_op_weights = self._filtered_op_weights(
            self.ARITH_OPS + self.BITWISE_OPS
        )
        self._comparison_op_weights = self._filtered_op_weights(
            self.RELATIONAL_OPS + self.EQUALITY_OPS
        )
        self._logical_op_weights = self._filtered_op_weights(self.LOGICAL_OPS)

    def _get_random_signal(self) -> Signal:
        """
        @brief Get a random signal from the available signals.
        @return Signal node
        """
        return Signal(self.rng.choice(self.signals))

    def _get_random_delay(self) -> str:
        """
        @brief Generate a random delay specification.
        @return Delay string (e.g., "##1", "##[1:5]", "##[0:$]")
        """
        if self.rng.random() < 0.1:  # 10% chance for unbounded
            min_val = self.rng.randint(0, 2)
            return f"##[{min_val}:$]"

        d_min = self.rng.randint(0, 2)
        d_max = d_min + self.rng.randint(0, 3)

        if d_min == d_max:
            return f"##{d_min}"
        return f"##[{d_min}:{d_max}]"

    def _get_random_repeat_count(
        self,
        allow_range: bool = True,
        allow_unbounded: bool = True
    ) -> str:
        """
        @brief Generate a random repetition count.
        @return Count string (e.g., "3", "1:5", "0:$")
        """
        if allow_unbounded and self.rng.random() < 0.1:  # 10% chance for unbounded
            min_val = self.rng.randint(0, 2)
            return f"{min_val}:$]"

        c_min = self.rng.randint(1, 5)

        if not allow_range:
            return f"{c_min}]"

        c_max = c_min + self.rng.randint(0, 3)
        if c_min == c_max:
            return f"{c_min}]"
        return f"{c_min}:{c_max}]"

    def _load_operator_weights(self) -> dict:
        weights_path = Path(__file__).resolve().parent / "arith_weight.json"
        try:
            with weights_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        return data

    def _parse_max_ops_per_property(self, data: dict) -> int | None:
        value = data.get("max_ops_per_property")
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
        return None

    def _filtered_op_weights(self, ops: list[str]) -> dict[str, float]:
        weights = {op: 1.0 for op in ops}
        for op in ops:
            weight = self._operator_weights.get(op)
            if isinstance(weight, (int, float)) and weight >= 0:
                weights[op] = float(weight)

        if sum(weights.values()) <= 0:
            return {op: 1.0 for op in ops}

        return weights

    def _reset_ops_budget(self) -> None:
        if self._max_ops_per_property is None:
            self._remaining_ops_budget = None
        else:
            self._remaining_ops_budget = self._max_ops_per_property

    def _clear_ops_budget(self) -> None:
        self._remaining_ops_budget = None

    def _consume_ops_budget(self) -> bool:
        if self._remaining_ops_budget is None:
            return True
        if self._remaining_ops_budget <= 0:
            return False
        self._remaining_ops_budget -= 1
        return True

    def _generate_expr_leaf(self) -> SVANode:
        if self.rng.random() < 0.2:
            past_depth = self.rng.randint(1, 3)
            return PastFunction(self._get_random_signal(), past_depth)
        return self._get_random_signal()

    def _generate_arith_operand(self) -> SVANode:
        return self._get_random_signal()

    def _generate_bool_leaf(self, depth: int) -> SVANode:
        leaf_choice = self.rng.random()

        if leaf_choice < 0.4:
            func = self.rng.choice(self.UNARY_SYS_FUNCS)
            return UnarySysFunction(func, self._get_random_signal())
        elif leaf_choice < 0.5 and self.enable_advanced_features:
            seq = self.generate_sequence(depth + 1)
            return SequenceEnded(seq)
        return self._get_random_signal()

    def _generate_sequence_or_bool(self, depth: int) -> SVANode:
        """Mix in boolean expressions with sequences for richer coverage."""
        # Boolean expressions introduce arithmetic/bitwise/system function usage
        if self.rng.random() < 0.35:
            bool_depth = max(0, min(depth, self.max_depth - 1))
            return self.generate_bool(bool_depth)
        return self.generate_sequence(depth)

    def generate_expr(self, depth: int) -> SVANode:
        """
        @brief Generate an expression (TYPE_EXPR).
        @param depth Current recursion depth
        @return SVANode of type EXPR
        """
        if depth >= self.max_depth or self.rng.random() > 0.7:
            # Leaf: signal or $past
            return self._generate_expr_leaf()

        choice = self.rng.random()

        # Unary operation
        if choice < 0.15:
            op = self.rng.choice(["~", "-", "+"])
            return UnaryOp(op, self.generate_expr(depth + 1))

        # Ternary operation (conditional)
        elif choice < 0.25 and self.enable_advanced_features:
            cond = self.generate_bool(depth + 1)
            true_expr = self.generate_expr(depth + 1)
            false_expr = self.generate_expr(depth + 1)
            return TernaryOp(cond, true_expr, false_expr, TYPE_EXPR)

        # Binary operation
        else:
            if not self._consume_ops_budget():
                return self._generate_expr_leaf()
            op = weighted_choice(self._expr_op_weights, rng=self.rng)
            if op in self.ARITH_OPS:
                left = self._generate_arith_operand()
                right = self._generate_arith_operand()
            else:
                left = self.generate_expr(depth + 1)
                right = self.generate_expr(depth + 1)
            return BinaryOp(
                left,
                op,
                right,
                TYPE_EXPR
            )

    def generate_bool(self, depth: int) -> SVANode:
        """
        @brief Generate a boolean expression (TYPE_BOOL).
        @param depth Current recursion depth
        @return SVANode of type BOOL
        """
        choice = self.rng.random()

        if depth >= self.max_depth or choice < 0.25:
            # Leaf: signal, system function, or sequence.ended
            return self._generate_bool_leaf(depth)

        elif choice < 0.4:
            # Unary logical NOT
            return UnaryOp("!", self.generate_bool(depth + 1))

        elif choice < 0.7:
            # Binary comparison (relational or equality)
            if not self._consume_ops_budget():
                return self._generate_bool_leaf(depth)
            op = weighted_choice(self._comparison_op_weights, rng=self.rng)
            return BinaryOp(
                self.generate_expr(0),
                op,
                self.generate_expr(0),
                TYPE_BOOL
            )

        else:
            # Logical operation (&&, ||)
            if not self._consume_ops_budget():
                return self._generate_bool_leaf(depth)
            op = weighted_choice(self._logical_op_weights, rng=self.rng)
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
        @return SVANode of type SEQUENCE
        """
        choice = self.rng.random()

        # Increased probability of base case for simpler properties
        if depth >= self.max_depth or choice < 0.45:  # Increased from 0.3
            # Base case: simple signal (valid sequence base)
            # A signal by itself is a valid sequence expression
            return self._get_random_signal()

        elif choice < 0.65:  # Sequence delay - most common temporal operator
            # Sequence delay
            delay = self._get_random_delay()
            return SequenceDelay(
                self.generate_sequence(depth + 1),
                delay,
                self.generate_sequence(depth + 1)
            )

        elif choice < 0.8:  # Sequence repeat
            # Sequence repeat
            op = self.rng.choice(self.REPEAT_OPS)
            count = self._get_random_repeat_count(
                allow_range=(op != "[->"),
                allow_unbounded=(op != "[->")
            )
            if op in ("[=", "[->"):
                expr = self.generate_bool(depth + 1)
            else:
                expr = self._generate_sequence_or_bool(depth + 1)
            return SequenceRepeat(
                expr,
                op,
                count
            )

        elif choice < 0.9:  # Binary sequence operation - reduced
            # Binary sequence operation
            op = self.rng.choice(self.SEQ_BIN_OPS)
            if op == "throughout":
                left = self.generate_bool(depth + 1)
                right = self.generate_sequence(depth + 1)
            else:
                left = self.generate_sequence(depth + 1)
                right = self.generate_sequence(depth + 1)
            return SequenceBinary(
                left,
                op,
                right
            )

        else:
            # first_match (advanced feature) - reduced probability
            if self.enable_advanced_features and self.rng.random() < 0.3:
                return SequenceFirstMatch(self.generate_sequence(depth + 1))
            else:
                # Fallback to simple signal
                return self._get_random_signal()

    def generate_property(self, depth: int = 0) -> SVANode:
        """
        @brief Generate a property (TYPE_PROPERTY).
        @param depth Current recursion depth
        @return SVANode of type PROPERTY
        """
        reset_budget = False
        if depth == 0 and self._remaining_ops_budget is None:
            self._reset_ops_budget()
            reset_budget = True

        choice = self.rng.random()

        # Implication (most common - increased probability for readability)
        if choice < 0.75:  # Increased from 0.6
            ante = self._generate_sequence_or_bool(depth + 1)
            cons = self._generate_sequence_or_bool(depth + 1)
            op = self.rng.choice(self.IMPLICATIONS)
            prop = Implication(ante, op, cons)

        # Property binary operation (and, or) - reduced probability
        elif choice < 0.82 and self.enable_advanced_features and depth < self.max_depth - 1:
            op = self.rng.choice(self.PROPERTY_BIN_OPS)
            # Avoid deep nesting - use sequences instead of properties
            left_prop = self._generate_sequence_or_bool(depth + 1)
            right_prop = self._generate_sequence_or_bool(depth + 1)
            prop = PropertyBinary(left_prop, op, right_prop)

        # Until operators - reduced probability
        elif choice < 0.88 and self.enable_advanced_features:
            op = self.rng.choice(self.UNTIL_OPS)
            left = self._generate_sequence_or_bool(depth + 1)
            right = self._generate_sequence_or_bool(depth + 1)
            prop = PropertyUntil(left, op, right)

        # If-else property - reduced probability
        elif choice < 0.92 and self.enable_advanced_features:
            condition = self.generate_bool(depth + 1)
            true_prop = self._generate_sequence_or_bool(depth + 1)
            false_prop = (
                self._generate_sequence_or_bool(depth + 1)
                if self.rng.random() < 0.5 else None
            )
            prop = PropertyIfElse(condition, true_prop, false_prop)

        else:
            # Direct sequence as property or "not" property
            prop = self._generate_sequence_or_bool(depth + 1)
            if self.rng.random() < 0.2:  # Reduced from 0.3
                prop = NotProperty(prop)

        # Reduced chance to wrap in disable iff for simpler properties
        if self.rng.random() < 0.1 and depth == 0:  # Only at top level
            reset = self._get_random_signal()
            prop = DisableIff(reset, prop)

        if reset_budget:
            self._clear_ops_budget()

        return prop

    def synthesize(self, name: str = "p_gen") -> SVAProperty:
        """
        @brief Generate a complete property with natural language description.
        @param name Property name
        @return SVAProperty object containing SVA code and description
        """
        prop_logic = self.generate_property()
        sva_code = str(prop_logic)
        svad = prop_logic.to_natural_language()

        property_block = (
            f"property {name};\n"
            f"  @(posedge {self.clock_signal}) {sva_code};\n"
            f"endproperty"
        )

        return SVAProperty(
            name=name,
            sva_code=sva_code,
            svad=svad,
            property_block=property_block
        )

    def generate_module(
        self,
        module_name: str,
        num_assertions: int,
        include_assertions: bool = True
    ) -> tuple[str, list[SVAProperty]]:
        """
        @brief Generate a complete SystemVerilog module with properties.
        @param module_name Name of the generated module
        @param num_assertions Number of assertions to generate
        @param include_assertions Whether to include assert property statements
        @return Tuple of (complete module code, list of SVAProperty objects)
        """
        properties: list[SVAProperty] = []
        for i in range(num_assertions):
            prop = self.synthesize(name=f"p_gen_{i}")
            properties.append(prop)

        signal_decls = "\n  ".join([f"logic {sig};" for sig in self.signals])
        properties_block = "\n\n  ".join([p.property_block for p in properties])

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

    def _resolve_verible_path(self) -> str | None:
        if self.verible_path:
            return self.verible_path

        try:
            tool = self.tool_registry.get("verible-verilog-syntax")
        except KeyError:
            return None
        return tool.path if tool.available else None

    def validate_syntax(self, code: str) -> ValidationResult:
        """
        @brief Validate SVA syntax using Verible.
        @param code SystemVerilog code to validate
        @return ValidationResult with is_valid flag and any error messages
        """
        verible_path = self._resolve_verible_path()
        if not verible_path:
            return ValidationResult(
                is_valid=False,
                error_message="Required tool 'verible-verilog-syntax' is not available"
            )

        result = run_tool(
            [verible_path, "-"],
            input_text=code,
            timeout=30,
        )
        if result.timed_out:
            return ValidationResult(
                is_valid=False,
                error_message="Verible validation timed out"
            )
        if result.returncode == 0:
            return ValidationResult(is_valid=True)

        error_msg = result.stderr.strip() or result.stdout.strip()
        return ValidationResult(is_valid=False, error_message=error_msg)

    def validate_properties(self, properties: list[SVAProperty]) -> ValidationResult:
        """
        @brief Validate a list of SVA properties by wrapping in a dummy module.
        @param properties List of SVAProperty objects to validate
        @return ValidationResult with is_valid flag and any error messages
        """
        # Create a dummy module wrapping all properties
        properties_block = "\n".join([p.property_block for p in properties])
        dummy_module = f"""module sva_validation_wrapper (
  input logic {self.clock_signal}
);
  // Dummy signals
  logic dummy_sig;
{properties_block}
endmodule
"""
        return self.validate_syntax(dummy_module)

    def validate_single_property(self, prop: SVAProperty) -> ValidationResult:
        """
        @brief Validate a single SVA property by wrapping it in its own module.
        @param prop SVAProperty object to validate
        @return ValidationResult with is_valid flag and any error messages
        """
        # Generate signal declarations
        signal_decls = "\n  ".join([f"logic {sig};" for sig in self.signals])

        # Create a module wrapping just this property
        dummy_module = f"""module sva_validation_{prop.name} (
  input logic {self.clock_signal},
  input logic rst_n
);
  // Signal Declarations
  {signal_decls}

  // Property to validate
  {prop.property_block}

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
        @brief Generate SVA module with individual property validation.
        Each property is validated separately, and only valid ones are included.
        @param module_name Name of the generated module
        @param num_assertions Number of assertions to generate
        @param max_retries Maximum retries per property on validation failure
        @return GenerationResult with only valid properties
        """
        valid_properties: list[SVAProperty] = []
        invalid_count = 0

        for i in range(num_assertions):
            prop_name = f"p_gen_{i}"
            validated = False

            # Try to generate a valid property with retries
            for attempt in range(max_retries):
                prop = self.synthesize(name=prop_name)
                validation = self.validate_single_property(prop)

                if validation.is_valid:
                    valid_properties.append(prop)
                    validated = True
                    break

            if not validated:
                invalid_count += 1

        # Build module with only valid properties
        if valid_properties:
            signal_decls = "\n  ".join([f"logic {sig};" for sig in self.signals])
            properties_block = "\n\n  ".join([p.property_block for p in valid_properties])
            assertions = "\n  ".join([
                f"assert_{p.name}: assert property ({p.name});"
                for p in valid_properties
            ])

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
        else:
            module_code = ""

        return GenerationResult(
            properties=valid_properties,
            module_code=module_code,
            validation=ValidationResult(is_valid=True) if valid_properties else ValidationResult(is_valid=False, error_message="No valid properties generated"),
            valid_count=len(valid_properties),
            invalid_count=invalid_count
        )
