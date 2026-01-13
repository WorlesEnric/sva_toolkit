"""
Stratified Sampling for SVA Generation

Provides guaranteed coverage of all SVA constructs by using constraint-based generation.
"""

from typing import List, Optional, Dict, Set
import random

from sva_toolkit.gen.generator import SVASynthesizer, SVAProperty
from sva_toolkit.gen.types_sva import *


class StratifiedGenerator:
    """
    Stratified sampling generator that guarantees coverage of all SVA constructs.

    This generator ensures that each SVA construct appears at least a minimum
    number of times in the generated dataset.
    """

    # Define all constructs organized by category
    PROPERTY_OPERATORS = [
        ('|->', 'overlapping_implication'),
        ('|=>', 'non_overlapping_implication'),
        ('and', 'property_and'),
        ('or', 'property_or'),
        ('until', 'until_operator'),
        ('until_with', 'until_with_operator'),
        ('not', 'not_property'),
        ('disable iff', 'disable_iff'),
        ('if', 'if_else'),
    ]

    SEQUENCE_OPERATORS = [
        ('##', 'delay'),
        ('[*', 'consecutive_repeat'),
        ('[=', 'non_consecutive_repeat'),
        ('[->', 'goto_repeat'),
        ('intersect', 'intersect'),
        ('throughout', 'throughout'),
        ('first_match', 'first_match'),
        ('.ended', 'sequence_ended'),
    ]

    SYSTEM_FUNCTIONS = [
        ('$rose', 'rose'),
        ('$fell', 'fell'),
        ('$stable', 'stable'),
        ('$changed', 'changed'),
        ('$past', 'past'),
        ('$onehot', 'onehot'),
        ('$onehot0', 'onehot0'),
        ('$isunknown', 'isunknown'),
        ('$countones', 'countones'),
    ]

    BOOLEAN_OPERATORS = [
        ('&&', 'logical_and'),
        ('||', 'logical_or'),
        ('!', 'logical_not'),
    ]

    COMPARISON_OPERATORS = [
        ('==', 'equality'),
        ('!=', 'inequality'),
        ('===', 'case_equality'),
        ('!==', 'case_inequality'),
        ('>', 'greater_than'),
        ('<', 'less_than'),
        ('>=', 'greater_equal'),
        ('<=', 'less_equal'),
    ]

    ARITHMETIC_OPERATORS = [
        ('+', 'addition'),
        ('-', 'subtraction'),
        ('*', 'multiplication'),
        ('/', 'division'),
        ('%', 'modulo'),
    ]

    BITWISE_OPERATORS = [
        ('&', 'bitwise_and'),
        ('|', 'bitwise_or'),
        ('^', 'bitwise_xor'),
        ('^~', 'bitwise_xnor'),
        ('~', 'bitwise_not'),
    ]

    def __init__(
        self,
        signals: List[str],
        clock_signal: str = "clk",
        max_depth: int = 2,
        samples_per_construct: int = 50,
        verible_path: Optional[str] = None
    ):
        """
        Initialize stratified generator.

        Args:
            signals: List of signal names
            clock_signal: Clock signal name
            max_depth: Maximum recursion depth
            samples_per_construct: Minimum samples per construct
            verible_path: Path to verible for validation
        """
        self.synth = SVASynthesizer(
            signals=signals,
            max_depth=max_depth,
            clock_signal=clock_signal,
            verible_path=verible_path,
            enable_advanced_features=True
        )
        self.signals = signals
        self.samples_per_construct = samples_per_construct
        self.property_counter = 0

    def _get_random_signal(self) -> Signal:
        """Get a random signal."""
        return Signal(random.choice(self.signals))

    def _get_random_delay(self) -> str:
        """Get a random delay specification."""
        return self.synth._get_random_delay()

    def _get_random_count(
        self,
        allow_range: bool = True,
        allow_unbounded: bool = True
    ) -> str:
        """Get a random repetition count."""
        return self.synth._get_random_repeat_count(
            allow_range=allow_range,
            allow_unbounded=allow_unbounded
        )

    # ========================================================================
    # Property-level constraint generators
    # ========================================================================

    def generate_overlapping_implication(self) -> SVANode:
        """Generate property with overlapping implication |->"""
        ante = self.synth.generate_sequence(0)
        cons = self.synth.generate_sequence(0)
        return Implication(ante, "|->", cons)

    def generate_non_overlapping_implication(self) -> SVANode:
        """Generate property with non-overlapping implication |=>"""
        ante = self.synth.generate_sequence(0)
        cons = self.synth.generate_sequence(0)
        return Implication(ante, "|=>", cons)

    def generate_property_and(self) -> SVANode:
        """Generate property with AND operator."""
        left = self.synth.generate_sequence(0)
        right = self.synth.generate_sequence(0)
        return PropertyBinary(left, "and", right)

    def generate_property_or(self) -> SVANode:
        """Generate property with OR operator."""
        left = self.synth.generate_sequence(0)
        right = self.synth.generate_sequence(0)
        return PropertyBinary(left, "or", right)

    def generate_until_operator(self) -> SVANode:
        """Generate property with until operator."""
        left = self.synth.generate_sequence(0)
        right = self.synth.generate_sequence(0)
        return PropertyUntil(left, "until", right)

    def generate_until_with_operator(self) -> SVANode:
        """Generate property with until_with operator."""
        left = self.synth.generate_sequence(0)
        right = self.synth.generate_sequence(0)
        return PropertyUntil(left, "until_with", right)

    def generate_not_property(self) -> SVANode:
        """Generate NOT property."""
        prop = self.synth.generate_sequence(0)
        return NotProperty(prop)

    def generate_disable_iff(self) -> SVANode:
        """Generate property with disable iff."""
        reset = self._get_random_signal()
        prop = Implication(
            self.synth.generate_sequence(0),
            random.choice(["|->", "|=>"]),
            self.synth.generate_sequence(0)
        )
        return DisableIff(reset, prop)

    def generate_if_else(self) -> SVANode:
        """Generate if-else property."""
        cond = self.synth.generate_bool(0)
        true_prop = self.synth.generate_sequence(0)
        false_prop = self.synth.generate_sequence(0) if random.random() < 0.7 else None
        return PropertyIfElse(cond, true_prop, false_prop)

    # ========================================================================
    # Sequence-level constraint generators
    # ========================================================================

    def generate_delay(self) -> SVANode:
        """Generate sequence with delay operator ##."""
        delay = self._get_random_delay()
        left = self.synth.generate_bool(0)
        right = self.synth.generate_bool(0)
        seq = SequenceDelay(left, delay, right)
        # Wrap in implication
        return Implication(seq, "|->", self.synth.generate_sequence(0))

    def generate_consecutive_repeat(self) -> SVANode:
        """Generate sequence with [* repetition."""
        expr = self.synth.generate_bool(0)
        count = self._get_random_count()
        seq = SequenceRepeat(expr, "[*", count)
        return Implication(seq, "|->", self._get_random_signal())

    def generate_non_consecutive_repeat(self) -> SVANode:
        """Generate sequence with [= repetition."""
        expr = self.synth.generate_bool(0)
        count = self._get_random_count()
        seq = SequenceRepeat(expr, "[=", count)
        return Implication(seq, "|->", self._get_random_signal())

    def generate_goto_repeat(self) -> SVANode:
        """Generate sequence with [-> repetition."""
        expr = self.synth.generate_bool(0)
        count = self._get_random_count(
            allow_range=False,
            allow_unbounded=False
        )
        seq = SequenceRepeat(expr, "[->", count)
        return Implication(seq, "|->", self._get_random_signal())

    def generate_intersect(self) -> SVANode:
        """Generate sequence with intersect."""
        left = self.synth.generate_bool(0)
        right = self.synth.generate_bool(0)
        seq = SequenceBinary(left, "intersect", right)
        return Implication(seq, "|->", self._get_random_signal())

    def generate_throughout(self) -> SVANode:
        """Generate sequence with throughout."""
        left = self.synth.generate_bool(0)
        right = self.synth.generate_bool(0)
        seq = SequenceBinary(left, "throughout", right)
        return Implication(seq, "|->", self._get_random_signal())

    def generate_first_match(self) -> SVANode:
        """Generate sequence with first_match."""
        inner_seq = self.synth.generate_sequence(0)
        seq = SequenceFirstMatch(inner_seq)
        return Implication(self._get_random_signal(), "|->", seq)

    def generate_sequence_ended(self) -> SVANode:
        """Generate boolean with .ended."""
        seq = self.synth.generate_sequence(0)
        ended = SequenceEnded(seq)
        return Implication(ended, "|->", self._get_random_signal())

    # ========================================================================
    # System function constraint generators
    # ========================================================================

    def _generate_with_system_function(self, func_name: str) -> SVANode:
        """Generate property containing specific system function."""
        sig = self._get_random_signal()
        func = UnarySysFunction(func_name, sig)
        cons = self.synth.generate_sequence(0)
        return Implication(func, "|->", cons)

    def generate_rose(self) -> SVANode:
        return self._generate_with_system_function("$rose")

    def generate_fell(self) -> SVANode:
        return self._generate_with_system_function("$fell")

    def generate_stable(self) -> SVANode:
        return self._generate_with_system_function("$stable")

    def generate_changed(self) -> SVANode:
        return self._generate_with_system_function("$changed")

    def generate_past(self) -> SVANode:
        """Generate property with $past function."""
        sig = self._get_random_signal()
        depth = random.randint(1, 3)
        past = PastFunction(sig, depth)
        other_sig = self._get_random_signal()
        comparison = BinaryOp(past, "==", other_sig)
        return Implication(self._get_random_signal(), "|->", comparison)

    def generate_onehot(self) -> SVANode:
        return self._generate_with_system_function("$onehot")

    def generate_onehot0(self) -> SVANode:
        return self._generate_with_system_function("$onehot0")

    def generate_isunknown(self) -> SVANode:
        return self._generate_with_system_function("$isunknown")

    def generate_countones(self) -> SVANode:
        return self._generate_with_system_function("$countones")

    # ========================================================================
    # Boolean/Expression constraint generators
    # ========================================================================

    def _generate_with_boolean_op(self, op: str) -> SVANode:
        """Generate property with specific boolean operator."""
        left = self.synth.generate_bool(1)
        right = self.synth.generate_bool(1)
        bool_expr = BinaryOp(left, op, right, TYPE_BOOL)
        return Implication(bool_expr, "|->", self._get_random_signal())

    def generate_logical_and(self) -> SVANode:
        return self._generate_with_boolean_op("&&")

    def generate_logical_or(self) -> SVANode:
        return self._generate_with_boolean_op("||")

    def generate_logical_not(self) -> SVANode:
        """Generate property with logical NOT."""
        operand = self.synth.generate_bool(1)
        not_expr = UnaryOp("!", operand)
        return Implication(not_expr, "|->", self._get_random_signal())

    def _generate_with_comparison(self, op: str) -> SVANode:
        """Generate property with specific comparison operator."""
        left = self.synth.generate_expr(1)
        right = self.synth.generate_expr(1)
        comparison = BinaryOp(left, op, right, TYPE_BOOL)
        return Implication(comparison, "|->", self._get_random_signal())

    def generate_equality(self) -> SVANode:
        return self._generate_with_comparison("==")

    def generate_inequality(self) -> SVANode:
        return self._generate_with_comparison("!=")

    def generate_case_equality(self) -> SVANode:
        return self._generate_with_comparison("===")

    def generate_case_inequality(self) -> SVANode:
        return self._generate_with_comparison("!==")

    def generate_greater_than(self) -> SVANode:
        return self._generate_with_comparison(">")

    def generate_less_than(self) -> SVANode:
        return self._generate_with_comparison("<")

    def generate_greater_equal(self) -> SVANode:
        return self._generate_with_comparison(">=")

    def generate_less_equal(self) -> SVANode:
        return self._generate_with_comparison("<=")

    def _generate_with_arithmetic(self, op: str) -> SVANode:
        """Generate property with specific arithmetic operator."""
        left = self.synth.generate_expr(1)
        right = self.synth.generate_expr(1)
        arith = BinaryOp(left, op, right, TYPE_EXPR)
        other = self.synth.generate_expr(1)
        comparison = BinaryOp(arith, ">", other, TYPE_BOOL)
        return Implication(comparison, "|->", self._get_random_signal())

    def generate_addition(self) -> SVANode:
        return self._generate_with_arithmetic("+")

    def generate_subtraction(self) -> SVANode:
        return self._generate_with_arithmetic("-")

    def generate_multiplication(self) -> SVANode:
        return self._generate_with_arithmetic("*")

    def generate_division(self) -> SVANode:
        return self._generate_with_arithmetic("/")

    def generate_modulo(self) -> SVANode:
        return self._generate_with_arithmetic("%")

    def _generate_with_bitwise(self, op: str) -> SVANode:
        """Generate property with specific bitwise operator."""
        left = self.synth.generate_expr(1)
        right = self.synth.generate_expr(1)
        bitwise = BinaryOp(left, op, right, TYPE_EXPR)
        other = self.synth.generate_expr(1)
        comparison = BinaryOp(bitwise, "==", other, TYPE_BOOL)
        return Implication(comparison, "|->", self._get_random_signal())

    def generate_bitwise_and(self) -> SVANode:
        return self._generate_with_bitwise("&")

    def generate_bitwise_or(self) -> SVANode:
        return self._generate_with_bitwise("|")

    def generate_bitwise_xor(self) -> SVANode:
        return self._generate_with_bitwise("^")

    def generate_bitwise_xnor(self) -> SVANode:
        return self._generate_with_bitwise("^~")

    def generate_bitwise_not(self) -> SVANode:
        """Generate property with bitwise NOT."""
        operand = self.synth.generate_expr(1)
        not_expr = UnaryOp("~", operand)
        other = self.synth.generate_expr(1)
        comparison = BinaryOp(not_expr, "==", other, TYPE_BOOL)
        return Implication(comparison, "|->", self._get_random_signal())

    # ========================================================================
    # Main generation method
    # ========================================================================

    def generate_stratified_dataset(self) -> List[SVAProperty]:
        """
        Generate a stratified dataset with guaranteed coverage.
        Each property is individually validated, and only valid ones are included.

        Returns:
            List of SVAProperty objects covering all constructs (only valid ones)
        """
        properties = []
        invalid_count = 0
        max_retries_per_sample = 3

        # Collect all construct generators
        all_constructs = (
            self.PROPERTY_OPERATORS +
            self.SEQUENCE_OPERATORS +
            self.SYSTEM_FUNCTIONS +
            self.BOOLEAN_OPERATORS +
            self.COMPARISON_OPERATORS +
            self.ARITHMETIC_OPERATORS +
            self.BITWISE_OPERATORS
        )

        print(f"Generating stratified dataset:")
        print(f"  Total constructs: {len(all_constructs)}")
        print(f"  Samples per construct: {self.samples_per_construct}")
        print(f"  Target samples: {len(all_constructs) * self.samples_per_construct}")
        print()

        for keyword, method_name in all_constructs:
            # Get the generator method
            generator_method = getattr(self, f'generate_{method_name}')

            print(f"Generating {self.samples_per_construct} samples for '{keyword}'... ", end='', flush=True)

            valid_samples = 0
            for i in range(self.samples_per_construct):
                validated = False

                # Try to generate a valid property with retries
                for attempt in range(max_retries_per_sample):
                    try:
                        # Generate property with this construct
                        prop_node = generator_method()

                        # Convert to SVAProperty
                        sva_code = str(prop_node)
                        svad = prop_node.to_natural_language()
                        name = f"p_{self.property_counter}"

                        property_block = (
                            f"property {name};\n"
                            f"  @(posedge {self.synth.clock_signal}) {sva_code};\n"
                            f"endproperty"
                        )

                        prop = SVAProperty(
                            name=name,
                            sva_code=sva_code,
                            svad=svad,
                            property_block=property_block
                        )

                        # Validate the property individually
                        validation = self.synth.validate_single_property(prop)

                        if validation.is_valid:
                            properties.append(prop)
                            self.property_counter += 1
                            valid_samples += 1
                            validated = True
                            break

                    except Exception as e:
                        # Continue to retry on exceptions
                        continue

                if not validated:
                    invalid_count += 1

            print(f"Done ({valid_samples} valid)")

        # Shuffle to mix constructs
        random.shuffle(properties)

        print(f"\n✓ Generated {len(properties)} valid properties (dropped {invalid_count} invalid)")
        return properties
