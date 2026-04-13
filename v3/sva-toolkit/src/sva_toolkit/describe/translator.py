"""
SVAD Translator - Convert SVA code into a markdown SVAD template.

This module parses SVA code, builds a lightweight symbolic summary, and
renders it into the requested markdown template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Dict, List, Optional, Tuple

from sva_toolkit.sva import emit_expr, emit_property_body, emit_sequence, parse_property_text
from sva_toolkit.sva.ast import (
    BinaryExpr,
    CallExpr,
    ClockEdge,
    ClockingEvent,
    ControlProperty,
    CycleRange,
    DelaySequence,
    ExprNode,
    ExprSequence,
    FirstMatchSequence,
    Identifier,
    IfElseProperty,
    ImplicationOperator as V3ImplicationOperator,
    ImplicationProperty,
    Literal,
    OpaqueExpr,
    OpaqueProperty,
    OpaqueSequence,
    PropertyBinary,
    PropertyBinaryOperator,
    PropertyNode,
    PropertySpec,
    PropertyUnaryOperator,
    RepeatOperator,
    RepeatSequence,
    SequenceBinary,
    SequenceBinaryOperator,
    SequenceEndedExpr,
    SequenceMatch,
    SequenceMatchItem,
    SequenceNode,
    TernaryExpr,
    UnaryExpr,
    UnaryProperty,
)


class ImplicationType(Enum):
    """Compatibility implication types matching the V1 describe module."""

    OVERLAPPING = "|->"
    NON_OVERLAPPING = "|=>"
    NONE = None


class TemporalOperator(Enum):
    """Compatibility temporal operator names consumed by the V1 CoT templates."""

    DELAY = "##"
    REPETITION_CONSECUTIVE = "[*"
    REPETITION_GOTO = "[->"
    REPETITION_NON_CONSECUTIVE = "[="
    THROUGHOUT = "throughout"
    WITHIN = "within"
    INTERSECT = "intersect"
    AND = "and"
    OR = "or"
    NOT = "not"
    FIRST_MATCH = "first_match"
    IF_ELSE = "if"


@dataclass(frozen=True)
class Signal:
    """Compatibility signal model used by the V1 describe templates."""

    name: str
    is_clock: bool = False
    is_reset: bool = False
    width: int | None = None


@dataclass(frozen=True)
class BuiltinFunction:
    """Compatibility built-in function descriptor."""

    name: str
    arguments: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DelayRange:
    """Compatibility delay range used by the V1 CoT builder."""

    min_cycles: int
    max_cycles: int | None = None
    is_unbounded: bool = False


@dataclass
class SVAStructure:
    """Compatibility structure extracted from the V3 parser AST."""

    raw_code: str
    property_name: str | None = None
    is_assertion: bool = True
    is_assumption: bool = False
    is_cover: bool = False
    clock_signal: str | None = None
    clock_edge: str = "posedge"
    reset_signal: str | None = None
    reset_active_low: bool = True
    disable_condition: str | None = None
    implication_type: ImplicationType = ImplicationType.NONE
    antecedent: str | None = None
    consequent: str | None = None
    signals: List[Signal] = field(default_factory=list)
    builtin_functions: List[BuiltinFunction] = field(default_factory=list)
    delays: List[DelayRange] = field(default_factory=list)
    temporal_operators: List[TemporalOperator] = field(default_factory=list)


@dataclass(frozen=True)
class TimingSpec:
    """Minimal timing description used by the translator's natural-language templates."""

    minimum: int | None
    maximum: int | None
    unbounded: bool = False

    def to_natural_language(self) -> str:
        if self.minimum == 0 and self.maximum in {None, 0} and not self.unbounded:
            return "immediately"
        if self.minimum == 1 and self.maximum is None and not self.unbounded:
            return "in the next cycle"
        if self.unbounded:
            if self.minimum in {None, 0}:
                return "eventually"
            return f"at least {self.minimum} cycles later"
        if self.maximum is None:
            if self.minimum == 1:
                return "one cycle later"
            return f"{self.minimum} cycles later"
        if self.minimum == self.maximum:
            return f"exactly {self.minimum} cycles later"
        return f"between {self.minimum} and {self.maximum} cycles later"


class SignalFormatter:
    """Small local copy of the V1 natural-language signal formatter."""

    SIGNAL_EXPANSIONS = {
        "req": "request (req)",
        "ack": "acknowledge (ack)",
        "gnt": "grant (gnt)",
        "clk": "clock (clk)",
        "rst": "reset (rst)",
        "en": "enable (en)",
        "data_en": "data enable (data_en)",
        "addr": "address (addr)",
        "we": "write enable (we)",
        "re": "read enable (re)",
    }

    def format(self, signal_name: str, as_subject: bool = True) -> str:
        del as_subject
        expanded = self.SIGNAL_EXPANSIONS.get(signal_name.lower())
        if expanded is not None:
            return f"the {expanded} signal"
        return f"the {signal_name} signal"


class TemporalFormatter:
    """Small local copy of the V1 delay-to-natural-language helper."""

    def parse_delay(self, delay_str: str) -> TimingSpec:
        if delay_str == "##0":
            return TimingSpec(0, 0)
        if delay_str.startswith("##") and "[" not in delay_str:
            cycles = int(delay_str[2:])
            return TimingSpec(cycles, None)
        if delay_str.startswith("##[") and ":" in delay_str:
            min_text, max_text = delay_str[3:-1].split(":", 1)
            minimum = int(min_text)
            if max_text == "$":
                return TimingSpec(minimum, None, unbounded=True)
            return TimingSpec(minimum, int(max_text))
        return TimingSpec(0, 0)


class SVAASTParser:
    """Compatibility parser that extracts V1-style structure from the V3 AST."""

    _CLOCK_PATTERNS = re.compile(r"\b(clk|clock)\w*\b", re.IGNORECASE)
    _RESET_PATTERNS = re.compile(r"\b(rst|reset)\w*\b", re.IGNORECASE)

    def parse(self, sva_code: str) -> SVAStructure:
        spec = parse_property_text(sva_code)
        collector = _StructureCollector()
        collector.collect_spec(spec)

        implication_type, antecedent, consequent = self._extract_implication(spec.body)
        disable_condition = emit_expr(spec.disable_iff) if spec.disable_iff is not None else None
        reset_signal, reset_active_low = self._infer_reset(spec.disable_iff, disable_condition)

        if spec.clocking is not None:
            collector.mark_signal(spec.clocking.signal.name, is_clock=True)

        if reset_signal is not None:
            collector.mark_signal(reset_signal, is_reset=True)

        is_assumption = spec.statement_kind is not None and spec.statement_kind.value == "assume"
        is_cover = spec.statement_kind is not None and spec.statement_kind.value == "cover"

        return SVAStructure(
            raw_code=sva_code.strip(),
            property_name=spec.name,
            is_assertion=not is_assumption and not is_cover,
            is_assumption=is_assumption,
            is_cover=is_cover,
            clock_signal=spec.clocking.signal.name if spec.clocking is not None else None,
            clock_edge=(spec.clocking.edge.value if spec.clocking is not None else ClockEdge.POSEDGE.value),
            reset_signal=reset_signal,
            reset_active_low=reset_active_low,
            disable_condition=disable_condition,
            implication_type=implication_type,
            antecedent=antecedent,
            consequent=consequent,
            signals=collector.signals,
            builtin_functions=collector.builtin_functions,
            delays=collector.delays,
            temporal_operators=collector.temporal_operators,
        )

    def _extract_property_body(self, sva_code: str) -> str:
        return emit_property_body(parse_property_text(sva_code).body)

    def _extract_implication(self, body: PropertyNode) -> tuple[ImplicationType, str | None, str | None]:
        if isinstance(body, ImplicationProperty):
            implication_type = (
                ImplicationType.OVERLAPPING
                if body.op is V3ImplicationOperator.OVERLAPPED
                else ImplicationType.NON_OVERLAPPING
            )
            return implication_type, emit_sequence(body.antecedent), emit_property_body(body.consequent)
        return ImplicationType.NONE, emit_property_body(body), None

    def _infer_reset(self, disable_iff: ExprNode | None, disable_text: str | None) -> tuple[str | None, bool]:
        if disable_iff is None or disable_text is None:
            return None, True

        identifiers: list[str] = []
        _collect_identifiers_from_expr(disable_iff, identifiers)
        reset_signal = next((name for name in identifiers if self._RESET_PATTERNS.search(name)), None)
        if reset_signal is None:
            reset_signal = next(iter(identifiers), None)
        if reset_signal is None:
            return None, True

        active_low = reset_signal.lower().endswith("_n")
        compact = disable_text.replace(" ", "")
        if compact.startswith("!"):
            active_low = True
        elif f"{reset_signal}==0" in compact or f"{reset_signal}===0" in compact or f"{reset_signal}==1'b0" in compact:
            active_low = True
        elif f"{reset_signal}==1" in compact or f"{reset_signal}===1" in compact or f"{reset_signal}==1'b1" in compact:
            active_low = False

        return reset_signal, active_low


class _StructureCollector:
    """Walk the V3 AST and collect the compatibility data expected by V1 describe code."""

    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}
        self.builtin_functions: list[BuiltinFunction] = []
        self.delays: list[DelayRange] = []
        self.temporal_operators: list[TemporalOperator] = []
        self._seen_temporal_ops: set[TemporalOperator] = set()

    @property
    def signals(self) -> list[Signal]:
        return sorted(self._signals.values(), key=lambda signal: signal.name)

    def mark_signal(self, name: str, *, is_clock: bool = False, is_reset: bool = False) -> None:
        current = self._signals.get(name)
        if current is None:
            self._signals[name] = Signal(name=name, is_clock=is_clock, is_reset=is_reset)
            return
        self._signals[name] = Signal(
            name=name,
            is_clock=current.is_clock or is_clock,
            is_reset=current.is_reset or is_reset,
            width=current.width,
        )

    def collect_spec(self, spec: PropertySpec) -> None:
        if spec.clocking is not None:
            self.collect_clocking(spec.clocking)
        if spec.disable_iff is not None:
            self.collect_expr(spec.disable_iff)
        self.collect_property(spec.body)

    def collect_clocking(self, node: ClockingEvent) -> None:
        self.mark_signal(node.signal.name, is_clock=True)

    def collect_property(self, node: PropertyNode) -> None:
        if isinstance(node, ImplicationProperty):
            self.collect_sequence(node.antecedent)
            self.collect_property(node.consequent)
            return
        if isinstance(node, UnaryProperty):
            if node.op is PropertyUnaryOperator.NOT:
                self._add_temporal_operator(TemporalOperator.NOT)
            self.collect_property(node.operand)
            return
        if isinstance(node, PropertyBinary):
            if node.op is PropertyBinaryOperator.AND:
                self._add_temporal_operator(TemporalOperator.AND)
            elif node.op is PropertyBinaryOperator.OR:
                self._add_temporal_operator(TemporalOperator.OR)
            self.collect_property(node.left)
            self.collect_property(node.right)
            return
        if isinstance(node, IfElseProperty):
            self._add_temporal_operator(TemporalOperator.IF_ELSE)
            self.collect_expr(node.condition)
            self.collect_property(node.when_true)
            if node.when_false is not None:
                self.collect_property(node.when_false)
            return
        if isinstance(node, ControlProperty):
            self.collect_expr(node.condition)
            self.collect_property(node.operand)
            return
        if isinstance(node, OpaqueProperty):
            self.collect_text(node.text)
            return
        self.collect_sequence(node)

    def collect_sequence(self, node: SequenceNode) -> None:
        if isinstance(node, ExprSequence):
            self.collect_expr(node.expr)
            return
        if isinstance(node, DelaySequence):
            self._add_temporal_operator(TemporalOperator.DELAY)
            delay_range = _delay_range_from_cycle_range(node.delay)
            if delay_range is not None:
                self.delays.append(delay_range)
            self.collect_sequence(node.left)
            self.collect_sequence(node.right)
            return
        if isinstance(node, RepeatSequence):
            if node.op is RepeatOperator.CONSECUTIVE:
                self._add_temporal_operator(TemporalOperator.REPETITION_CONSECUTIVE)
            elif node.op is RepeatOperator.NON_CONSECUTIVE:
                self._add_temporal_operator(TemporalOperator.REPETITION_NON_CONSECUTIVE)
            elif node.op is RepeatOperator.GOTO:
                self._add_temporal_operator(TemporalOperator.REPETITION_GOTO)
            self.collect_sequence(node.body)
            return
        if isinstance(node, SequenceBinary):
            operator = {
                SequenceBinaryOperator.AND: TemporalOperator.AND,
                SequenceBinaryOperator.OR: TemporalOperator.OR,
                SequenceBinaryOperator.INTERSECT: TemporalOperator.INTERSECT,
                SequenceBinaryOperator.THROUGHOUT: TemporalOperator.THROUGHOUT,
            }.get(node.op)
            if operator is not None:
                self._add_temporal_operator(operator)
            self.collect_sequence(node.left)
            self.collect_sequence(node.right)
            return
        if isinstance(node, SequenceMatch):
            self.collect_sequence(node.body)
            for item in node.items:
                self.collect_match_item(item)
            return
        if isinstance(node, FirstMatchSequence):
            self._add_temporal_operator(TemporalOperator.FIRST_MATCH)
            self.collect_sequence(node.body)
            return
        if isinstance(node, OpaqueSequence):
            self.collect_text(node.text)
            return

    def collect_match_item(self, node: SequenceMatchItem) -> None:
        self.mark_signal(node.lvalue.name)
        self.collect_expr(node.rvalue)

    def collect_expr(self, node: ExprNode) -> None:
        if isinstance(node, Identifier):
            self.mark_signal(node.name)
            return
        if isinstance(node, Literal):
            return
        if isinstance(node, OpaqueExpr):
            self.collect_text(node.text)
            return
        if isinstance(node, UnaryExpr):
            self.collect_expr(node.operand)
            return
        if isinstance(node, BinaryExpr):
            self.collect_expr(node.left)
            self.collect_expr(node.right)
            return
        if isinstance(node, TernaryExpr):
            self.collect_expr(node.condition)
            self.collect_expr(node.when_true)
            self.collect_expr(node.when_false)
            return
        if isinstance(node, CallExpr):
            arguments = [emit_expr(argument) for argument in node.args]
            if node.name.startswith("$"):
                self.builtin_functions.append(BuiltinFunction(name=node.name, arguments=arguments))
            for argument in node.args:
                self.collect_expr(argument)
            return
        if isinstance(node, SequenceEndedExpr):
            self.collect_sequence(node.sequence)

    def collect_text(self, text: str) -> None:
        for name in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text):
            if name in {"posedge", "negedge", "disable", "iff", "and", "or", "not", "if", "else"}:
                continue
            self.mark_signal(name)
        for match in re.finditer(r"(\$[a-zA-Z_][a-zA-Z0-9_]*)\(([^()]*)\)", text):
            args = [part.strip() for part in match.group(2).split(",") if part.strip()]
            self.builtin_functions.append(BuiltinFunction(name=match.group(1), arguments=args))

    def _add_temporal_operator(self, operator: TemporalOperator) -> None:
        if operator in self._seen_temporal_ops:
            return
        self._seen_temporal_ops.add(operator)
        self.temporal_operators.append(operator)


def _collect_identifiers_from_expr(node: ExprNode, out: list[str]) -> None:
    if isinstance(node, Identifier):
        out.append(node.name)
    elif isinstance(node, UnaryExpr):
        _collect_identifiers_from_expr(node.operand, out)
    elif isinstance(node, BinaryExpr):
        _collect_identifiers_from_expr(node.left, out)
        _collect_identifiers_from_expr(node.right, out)
    elif isinstance(node, TernaryExpr):
        _collect_identifiers_from_expr(node.condition, out)
        _collect_identifiers_from_expr(node.when_true, out)
        _collect_identifiers_from_expr(node.when_false, out)
    elif isinstance(node, CallExpr):
        for argument in node.args:
            _collect_identifiers_from_expr(argument, out)
    elif isinstance(node, SequenceEndedExpr):
        _collect_identifiers_from_expr(Identifier(name=emit_sequence(node.sequence)), out)


def _delay_range_from_cycle_range(node: CycleRange) -> DelayRange | None:
    minimum = _integer_text(node.minimum)
    if minimum is None:
        return None
    if node.unbounded:
        return DelayRange(min_cycles=minimum, is_unbounded=True)
    maximum = _integer_text(node.maximum) if node.maximum is not None else None
    if maximum == minimum:
        maximum = None
    return DelayRange(min_cycles=minimum, max_cycles=maximum)


def _integer_text(node: ExprNode | None) -> int | None:
    if isinstance(node, Literal) and node.text.isdigit():
        return int(node.text)
    if isinstance(node, Identifier) and node.name.isdigit():
        return int(node.name)
    return None


@dataclass
class SymbolicSVAD:
    """Lightweight symbolic SVAD summary."""
    scope: Optional[str]
    logic: str
    definitions: List[str]
    trigger_symbol: Optional[str]
    outcome_symbol: Optional[str]


@dataclass
class _SeqBase:
    text: str


@dataclass
class _SeqRepeat:
    expr: object
    op: str
    count: str


@dataclass
class _SeqDelay:
    left: object
    delay: str
    right: object


@dataclass
class _SeqBinary:
    left: object
    op: str
    right: object


@dataclass
class _SeqFirstMatch:
    sequence: object


@dataclass
class _SeqEnded:
    sequence: object


class _SafeSVAASTParser(SVAASTParser):
    """SVAASTParser that tolerates non-executable Verible binaries."""

    def _check_verible(self) -> bool:
        try:
            return super()._check_verible()
        except OSError:
            return False


class SVADTranslator:
    """
    Translate SVA code into a markdown SVAD template.
    """

    _SYS_FUNC_TEMPLATES = {
        "$rose": "{sig} rises from low to high",
        "$fell": "{sig} falls from high to low",
        "$stable": "{sig} remains stable",
        "$changed": "{sig} changes value",
        "$onehot": "exactly one bit of {sig} is high",
        "$onehot0": "at most one bit of {sig} is high",
        "$isunknown": "{sig} is unknown (X or Z)",
        "$countones": "the count of high bits in {sig}",
    }

    def __init__(
        self,
        parser: Optional[SVAASTParser] = None,
        expression_style: str = "symbolic",
    ) -> None:
        self.parser = parser or _SafeSVAASTParser()
        self.signal_formatter = SignalFormatter()
        self.temporal_formatter = TemporalFormatter()
        self.expression_style = expression_style
        if self.expression_style not in {"symbolic", "natural"}:
            raise ValueError(
                "expression_style must be 'symbolic' or 'natural'"
            )

    def translate(self, sva_code: str) -> str:
        """
        Translate SVA code into markdown SVAD.
        """
        structure = self.parser.parse(sva_code)
        symbolic = self._build_symbolic_svad(structure)
        return self._render_markdown(structure, symbolic)

    def _build_symbolic_svad(self, structure: SVAStructure) -> SymbolicSVAD:
        scope = None
        if structure.disable_condition:
            scope = (
                "This property is active unless "
                f"{structure.disable_condition} is asserted."
            )

        self._seq_registry: Dict[str, str] = {}
        self._seq_definitions: List[Tuple[str, str]] = []
        self._seq_counter = 0

        self._calc_registry: Dict[str, str] = {}
        self._calc_definitions: List[Tuple[str, str]] = []
        self._calc_counter = 0

        exp_defs: List[Tuple[str, str]] = []
        sys_defs: List[Tuple[str, str]] = []
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str] = {}

        sys_index = 0
        seen_funcs = set()
        for func in structure.builtin_functions:
            func_key = (func.name, tuple(func.arguments))
            if func_key in seen_funcs:
                continue
            seen_funcs.add(func_key)
            desc = self._describe_builtin(func)
            if not desc:
                continue
            sys_symbol = f"Sys_{sys_index}"
            sys_index += 1
            sys_defs.append((sys_symbol, desc))
            sys_symbol_by_key[func_key] = sys_symbol

        exp_index = 0
        trigger_symbol = None
        outcome_symbol = None

        if structure.antecedent:
            trigger_symbol = f"Exp_{exp_index}"
            exp_desc = self._render_expression(
                structure.antecedent,
                sys_symbol_by_key,
                structure.signals,
            )
            exp_defs.append((trigger_symbol, exp_desc))
            exp_index += 1

        if structure.consequent:
            outcome_symbol = f"Exp_{exp_index}"
            exp_desc = self._render_expression(
                structure.consequent,
                sys_symbol_by_key,
                structure.signals,
            )
            exp_defs.append((outcome_symbol, exp_desc))
            exp_index += 1

        logic = self._build_logic(structure, trigger_symbol, outcome_symbol)

        definitions = [
            f"{sym}: {desc}"
            for sym, desc in (
                self._seq_definitions +
                self._calc_definitions +
                exp_defs +
                sys_defs
            )
        ]

        return SymbolicSVAD(
            scope=scope,
            logic=logic,
            definitions=definitions,
            trigger_symbol=trigger_symbol,
            outcome_symbol=outcome_symbol,
        )

    def _build_logic(
        self,
        structure: SVAStructure,
        trigger_symbol: Optional[str],
        outcome_symbol: Optional[str],
    ) -> str:
        if structure.implication_type == ImplicationType.NONE:
            property_body = self.parser._extract_property_body(structure.raw_code)
            return property_body or structure.raw_code

        timing = "in the same cycle"
        if structure.implication_type == ImplicationType.NON_OVERLAPPING:
            timing = "in the next cycle"

        trigger_ref = trigger_symbol or structure.antecedent or "the trigger condition"
        outcome_ref = outcome_symbol or structure.consequent or "the expected condition"

        return (
            f"When {trigger_ref} occurs, then {timing}, {outcome_ref} must hold."
        )

    def _describe_builtin(self, func: BuiltinFunction) -> str:
        args = func.arguments
        if not args:
            return ""

        signal_desc = self.signal_formatter.format(args[0])

        if func.name == "$past":
            if len(args) >= 2:
                return f"the value of {signal_desc} from {args[1]} cycles ago"
            return f"the previous value of {signal_desc}"

        template = self._SYS_FUNC_TEMPLATES.get(func.name)
        if template:
            return template.format(sig=signal_desc)

        args_text = ", ".join(args)
        return f"{func.name} applied to {args_text}"

    def _render_expression(
        self,
        expr: str,
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str],
        signals: List,
    ) -> str:
        if self.expression_style == "symbolic":
            return self._render_expression_symbolic(
                expr,
                sys_symbol_by_key,
                signals,
            )
        return self._render_expression_natural(
            expr,
            sys_symbol_by_key,
            signals,
        )

    def _render_expression_symbolic(
        self,
        expr: str,
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str],
        signals: List,
    ) -> str:
        expr = expr.strip()
        seq_node = self._parse_sequence_expr(expr)
        if not isinstance(seq_node, _SeqBase):
            return self._describe_sequence(
                seq_node,
                sys_symbol_by_key,
                signals,
                is_root=True,
            )

        logical = self._split_logical(expr)
        if logical is not None:
            left, op, right = logical
            left_desc = self._render_expression_symbolic(
                left,
                sys_symbol_by_key,
                signals,
            )
            right_desc = self._render_expression_symbolic(
                right,
                sys_symbol_by_key,
                signals,
            )
            op_text = "and" if op == "&&" else "or"
            return f"{left_desc} {op_text} {right_desc}"

        if self._should_register_calc(expr):
            return self._register_calc(expr)

        rendered = self._replace_builtin_calls(expr, sys_symbol_by_key)
        rendered = self._replace_signals(rendered, signals)
        rendered = rendered.replace("&&", " and ").replace("||", " or ")
        return self._cleanup_text(rendered)

    def _render_expression_natural(
        self,
        expr: str,
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str],
        signals: List,
    ) -> str:
        rendered = self._replace_builtin_calls(expr, sys_symbol_by_key)
        rendered = self._replace_operators(rendered)
        rendered = self._replace_signals(rendered, signals)
        return self._cleanup_text(rendered)

    def _describe_sequence(
        self,
        node: object,
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str],
        signals: List,
        is_root: bool,
    ) -> str:
        if not is_root and isinstance(
            node,
            (_SeqRepeat, _SeqDelay, _SeqBinary, _SeqFirstMatch, _SeqEnded),
        ):
            return self._register_sequence(node, sys_symbol_by_key, signals)

        if isinstance(node, _SeqBase):
            return self._render_expression_symbolic_base(
                node.text,
                sys_symbol_by_key,
                signals,
            )

        if isinstance(node, _SeqRepeat):
            sub = self._describe_sequence(
                node.expr,
                sys_symbol_by_key,
                signals,
                is_root=False,
            )
            count_clean = node.count.rstrip("]")
            count_desc = self._describe_repeat_count(count_clean)
            if node.op == "[*":
                desc = f"{sub} remains true {count_desc} consecutively"
            elif node.op == "[=":
                desc = (
                    f"{sub} occurs {count_desc} non-consecutively "
                    "before the sequence continues"
                )
            elif node.op == "[->":
                desc = (
                    f"wait for the {self._ordinal_from_count(count_clean)} "
                    f"occurrence of {sub}"
                )
            else:
                desc = f"{sub} {node.op} {count_desc}"
            return desc

        if isinstance(node, _SeqDelay):
            left = self._describe_sequence(
                node.left,
                sys_symbol_by_key,
                signals,
                is_root=False,
            )
            right = self._describe_sequence(
                node.right,
                sys_symbol_by_key,
                signals,
                is_root=False,
            )
            timing = self.temporal_formatter.parse_delay(node.delay).to_natural_language()
            return f"{left} followed by {right} {timing}"

        if isinstance(node, _SeqBinary):
            left = self._describe_sequence(
                node.left,
                sys_symbol_by_key,
                signals,
                is_root=False,
            )
            right = self._describe_sequence(
                node.right,
                sys_symbol_by_key,
                signals,
                is_root=False,
            )
            if node.op == "intersect":
                return f"Sequence {left} and Sequence {right} start and end at the exact same time"
            if node.op == "throughout":
                return f"{left} holds true throughout the execution of {right}"
            if node.op == "and":
                return f"{left} and {right} both occur"
            if node.op == "or":
                return f"{left} or {right} occurs"
            return f"{left} {node.op} {right}"

        if isinstance(node, _SeqFirstMatch):
            sub = self._describe_sequence(
                node.sequence,
                sys_symbol_by_key,
                signals,
                is_root=False,
            )
            return f"the first match of {sub}"

        if isinstance(node, _SeqEnded):
            sub = self._describe_sequence(
                node.sequence,
                sys_symbol_by_key,
                signals,
                is_root=False,
            )
            return f"{sub} has ended"

        return str(node)

    def _register_sequence(
        self,
        node: object,
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str],
        signals: List,
    ) -> str:
        key = repr(node)
        if key in self._seq_registry:
            return self._seq_registry[key]

        symbol = f"Seq_{self._seq_counter}"
        self._seq_counter += 1
        self._seq_registry[key] = symbol

        desc = self._describe_sequence(
            node,
            sys_symbol_by_key,
            signals,
            is_root=True,
        )
        self._seq_definitions.append((symbol, desc))
        return symbol

    def _register_calc(self, expr: str) -> str:
        key = expr.strip()
        if key in self._calc_registry:
            return self._calc_registry[key]

        symbol = f"Calc_{self._calc_counter}"
        self._calc_counter += 1
        self._calc_registry[key] = symbol
        desc = f"The result of: {key}"
        self._calc_definitions.append((symbol, desc))
        return symbol

    def _render_expression_symbolic_base(
        self,
        expr: str,
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str],
        signals: List,
    ) -> str:
        rendered = self._replace_builtin_calls(expr, sys_symbol_by_key)
        rendered = self._replace_signals(rendered, signals)
        rendered = rendered.replace("&&", " and ").replace("||", " or ")
        return self._cleanup_text(rendered)

    def _parse_sequence_expr(self, expr: str) -> object:
        expr = self._strip_outer_parens(expr.strip())

        call = self._extract_wrapped_call(expr, "first_match")
        if call is not None:
            return _SeqFirstMatch(self._parse_sequence_expr(call))

        call = self._extract_wrapped_call(expr, "ended")
        if call is not None:
            return _SeqEnded(self._parse_sequence_expr(call))

        rep = self._split_repetition(expr)
        if rep is not None:
            base, op, count = rep
            return _SeqRepeat(self._parse_sequence_expr(base), op, count)

        delay = self._split_delay(expr)
        if delay is not None:
            left, delay_token, right = delay
            return _SeqDelay(
                self._parse_sequence_expr(left),
                delay_token,
                self._parse_sequence_expr(right),
            )

        binary = self._split_sequence_binary(expr)
        if binary is not None:
            left, op, right = binary
            return _SeqBinary(
                self._parse_sequence_expr(left),
                op,
                self._parse_sequence_expr(right),
            )

        return _SeqBase(expr)

    def _strip_outer_parens(self, expr: str) -> str:
        while expr.startswith("(") and expr.endswith(")"):
            if not self._outer_parens_wrap(expr):
                break
            expr = expr[1:-1].strip()
        return expr

    def _outer_parens_wrap(self, expr: str) -> bool:
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    return False
        return depth == 0

    def _extract_wrapped_call(self, expr: str, name: str) -> Optional[str]:
        prefix = f"{name}("
        if not expr.startswith(prefix) or not expr.endswith(")"):
            return None
        inner = expr[len(prefix):-1].strip()
        if self._outer_parens_wrap(f"({inner})"):
            return inner
        return None

    def _split_repetition(self, expr: str) -> Optional[Tuple[str, str, str]]:
        if not expr.endswith("]"):
            return None

        depth = 0
        for i in range(len(expr) - 1, -1, -1):
            ch = expr[i]
            if ch == "]":
                depth += 1
            elif ch == "[":
                depth -= 1
                if depth == 0:
                    content = expr[i + 1:-1].strip()
                    if content.startswith("->"):
                        op = "[->"
                        count = content[2:].strip()
                    elif content.startswith("*"):
                        op = "[*"
                        count = content[1:].strip()
                    elif content.startswith("="):
                        op = "[="
                        count = content[1:].strip()
                    else:
                        return None
                    if not count:
                        return None
                    base = expr[:i].strip()
                    if not base:
                        return None
                    return base, op, f"{count}]"
        return None

    def _split_delay(self, expr: str) -> Optional[Tuple[str, str, str]]:
        depth_paren = 0
        depth_brack = 0
        i = 0
        while i < len(expr) - 1:
            ch = expr[i]
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack = max(0, depth_brack - 1)

            if expr[i:i + 2] == "##" and depth_paren == 0 and depth_brack == 0:
                match = re.match(
                    r"##\s*(\[\d+\s*:\s*\d+\s*\]|\[\d+\s*:\s*\$\s*\]|\d+)",
                    expr[i:],
                )
                if not match:
                    return None
                token = "##" + match.group(1).replace(" ", "")
                left = expr[:i].strip()
                right = expr[i + len(token):].strip()
                if not left or not right:
                    return None
                return left, token, right
            i += 1
        return None

    def _split_sequence_binary(self, expr: str) -> Optional[Tuple[str, str, str]]:
        depth_paren = 0
        depth_brack = 0
        token = []
        for i, ch in enumerate(expr):
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack = max(0, depth_brack - 1)

            if depth_paren == 0 and depth_brack == 0:
                if ch.isalnum() or ch == "_":
                    token.append(ch)
                else:
                    if token:
                        word = "".join(token)
                        if word in {"intersect", "throughout", "and", "or"}:
                            start = i - len(word)
                            left = expr[:start].strip()
                            right = expr[i:].strip()
                            if left and right:
                                return left, word, right
                        token = []
            else:
                if token:
                    token = []
        return None

    def _split_logical(self, expr: str) -> Optional[Tuple[str, str, str]]:
        depth_paren = 0
        depth_brack = 0
        i = 0
        while i < len(expr) - 1:
            ch = expr[i]
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "[":
                depth_brack += 1
            elif ch == "]":
                depth_brack = max(0, depth_brack - 1)

            if depth_paren == 0 and depth_brack == 0:
                token = expr[i:i + 2]
                if token in {"&&", "||"}:
                    left = expr[:i].strip()
                    right = expr[i + 2:].strip()
                    if left and right:
                        return left, token, right
            i += 1
        return None

    def _should_register_calc(self, expr: str) -> bool:
        expr = self._strip_outer_parens(expr.strip())
        if not expr:
            return False

        simple_id = r"[a-zA-Z_][a-zA-Z0-9_]*"

        if re.fullmatch(simple_id, expr):
            return False

        if re.fullmatch(rf"[!~+-]\s*{simple_id}", expr):
            return False

        binary_ops = (
            r"===", r"!==", r"==", r"!=", r"<=", r">=", r"<", r">",
            r"\^~", r"~\^", r"\^", r"&", r"\|", r"\+", r"-",
            r"\*", r"/", r"%"
        )
        binary_pattern = rf"{simple_id}\s*({'|'.join(binary_ops)})\s*{simple_id}"
        if re.fullmatch(binary_pattern, expr):
            return False

        if "##" in expr or "[*" in expr or "[=" in expr or "[->" in expr:
            return False

        op_tokens = [
            "===", "!==", "==", "!=", "<=", ">=", "<", ">",
            "^~", "~^", "^", "&", "|", "+", "-", "*", "/", "%",
        ]
        for token in op_tokens:
            if token in expr:
                return True

        return False

    def _describe_repeat_count(self, count: str) -> str:
        if ":" in count:
            min_part, max_part = count.split(":", 1)
            if max_part == "$":
                return f"at least {min_part} times"
            if min_part == max_part:
                return f"{min_part} times"
            return f"between {min_part} and {max_part} times"

        if count == "$":
            return "an unbounded number of times"

        return f"{count} times"

    def _ordinal_from_count(self, count: str) -> str:
        try:
            num = int(count.split(":")[0])
        except ValueError:
            return f"{count}-th"

        suffix = "th"
        if 10 <= num % 100 <= 20:
            suffix = "th"
        else:
            if num % 10 == 1:
                suffix = "st"
            elif num % 10 == 2:
                suffix = "nd"
            elif num % 10 == 3:
                suffix = "rd"

        return f"{num}{suffix}"

    def _replace_builtin_calls(
        self,
        expr: str,
        sys_symbol_by_key: Dict[Tuple[str, Tuple[str, ...]], str],
    ) -> str:
        i = 0
        out = []
        length = len(expr)

        while i < length:
            ch = expr[i]
            if ch == "$" and i + 1 < length and expr[i + 1].isalpha():
                name_start = i
                i += 1
                while i < length and (expr[i].isalnum() or expr[i] == "_"):
                    i += 1
                func_name = expr[name_start:i]

                j = i
                while j < length and expr[j].isspace():
                    j += 1

                if j < length and expr[j] == "(":
                    args_start = j + 1
                    depth = 1
                    k = args_start
                    while k < length and depth > 0:
                        if expr[k] == "(":
                            depth += 1
                        elif expr[k] == ")":
                            depth -= 1
                        k += 1

                    if depth == 0:
                        args_str = expr[args_start:k - 1]
                        args = self._split_args(args_str)
                        key = (func_name, tuple(args))
                        replacement = sys_symbol_by_key.get(key)
                        if replacement is None:
                            replacement = self._describe_builtin(
                                BuiltinFunction(name=func_name, arguments=args)
                            )
                        out.append(replacement)
                        i = k
                        continue

                out.append(func_name)
                i = j
                continue

            out.append(ch)
            i += 1

        return "".join(out)

    def _split_args(self, args_str: str) -> List[str]:
        args = []
        current = []
        depth = 0
        i = 0
        length = len(args_str)
        while i < length:
            ch = args_str[i]
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth = max(0, depth - 1)
                current.append(ch)
            elif ch == "," and depth == 0:
                arg = "".join(current).strip()
                if arg:
                    args.append(arg)
                current = []
            else:
                current.append(ch)
            i += 1
        last = "".join(current).strip()
        if last:
            args.append(last)
        return args

    def _replace_operators(self, expr: str) -> str:
        expr = self._strip_unary_plus(expr)
        replacements = [
            ("|=>", " implies in the next cycle "),
            ("|->", " implies in the same cycle "),
            ("!==", " is not exactly equal to "),
            ("===", " is exactly equal to "),
            ("==", " equals "),
            ("!=", " does not equal "),
            (">=", " is at least "),
            ("<=", " is at most "),
            (">", " is greater than "),
            ("<", " is less than "),
            ("&&", " and "),
            ("||", " or "),
            ("^~", " bitwise XNOR "),
            ("~^", " bitwise XNOR "),
            ("^", " bitwise XOR "),
            ("&", " bitwise AND "),
            ("|", " bitwise OR "),
            ("+", " plus "),
            ("-", " minus "),
            ("*", " times "),
            ("/", " divided by "),
            ("%", " modulo "),
            ("?", " then "),
            (":", " else "),
        ]

        out = expr
        for op, text in replacements:
            out = out.replace(op, text)

        out = re.sub(r'!\s*', 'not ', out)
        out = re.sub(r'~\s*', 'bitwise not ', out)
        return out

    def _strip_unary_plus(self, expr: str) -> str:
        out = []
        prev_nonspace = None
        unary_context = set("([{:?,=<>!~&|^+-*/%")

        for ch in expr:
            if ch == "+":
                if prev_nonspace is None or prev_nonspace in unary_context:
                    continue
            out.append(ch)
            if not ch.isspace():
                prev_nonspace = ch
        return "".join(out)

    def _replace_signals(self, expr: str, signals: List) -> str:
        if not signals:
            return expr

        signal_names = sorted(
            {s.name for s in signals},
            key=len,
            reverse=True,
        )

        out = expr
        for name in signal_names:
            formatted = self.signal_formatter.format(name)
            out = re.sub(rf"\b{re.escape(name)}\b", formatted, out)
        return out

    def _cleanup_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = text.replace(" ,", ",")
        return text.strip()

    def _render_markdown(self, structure: SVAStructure, symbolic: SymbolicSVAD) -> str:
        lines: List[str] = []

        lines.append("**1. Relevant Signals:**")
        lines.extend(self._format_relevant_signals(structure, symbolic))

        lines.append("")
        lines.append("2. Check Condition:")
        lines.append(self._format_check_condition(structure, symbolic))

        lines.append("")
        lines.append("3. Expected Results:")
        lines.append(self._format_expected_results(structure, symbolic))

        if symbolic.definitions:
            lines.append("")
            lines.append("Definitions:")
            for definition in symbolic.definitions:
                lines.append(f"- {definition}")

        return "\n".join(lines).strip()

    def _format_relevant_signals(
        self,
        structure: SVAStructure,
        symbolic: SymbolicSVAD,
    ) -> List[str]:
        items: List[str] = []

        if structure.clock_signal:
            edge_desc = self._clock_edge_desc(structure.clock_edge)
            items.append(
                f"- Clock name: `{structure.clock_signal}`, "
                f"trigger at every {edge_desc} of `{structure.clock_signal}`"
            )
        else:
            items.append(
                "- Clock not specified; sampling assumed at each evaluation cycle"
            )

        if structure.reset_signal:
            active = "active low" if structure.reset_active_low else "active high"
            items.append(f"- Reset signal: `{structure.reset_signal}` ({active})")
        elif structure.disable_condition:
            items.append(f"- Disable condition: `{structure.disable_condition}`")
        else:
            items.append("- Reset not specified in the description, so none assumed")

        if symbolic.scope:
            items.append(f"- Scope: {symbolic.scope}")

        other_signals = sorted(
            s.name for s in structure.signals
            if not s.is_clock and not s.is_reset
        )
        if other_signals:
            signal_list = ", ".join(f"`{name}`" for name in other_signals)
            items.append(f"- Other relevant signals: {signal_list}")
        else:
            items.append("- Other relevant signals: (none detected)")

        return items

    def _format_check_condition(
        self,
        structure: SVAStructure,
        symbolic: SymbolicSVAD,
    ) -> str:
        if structure.implication_type == ImplicationType.NONE:
            return f"Evaluate the property: {symbolic.logic}"

        clock_clause = self._clock_clause(structure)
        trigger_ref = symbolic.trigger_symbol or structure.antecedent or "the trigger"
        return f"{clock_clause}, the target condition is that {trigger_ref} holds."

    def _format_expected_results(
        self,
        structure: SVAStructure,
        symbolic: SymbolicSVAD,
    ) -> str:
        if structure.implication_type == ImplicationType.NONE:
            return "No explicit implication; the property is the condition itself."

        outcome_ref = symbolic.outcome_symbol or structure.consequent or "the outcome"
        if structure.implication_type == ImplicationType.OVERLAPPING:
            prefix = self._expected_timing_clause(structure, same_cycle=True)
        else:
            prefix = self._expected_timing_clause(structure, same_cycle=False)
        return f"{prefix}, {outcome_ref} must hold."

    def _clock_edge_desc(self, edge: str) -> str:
        return "rising edge" if edge == "posedge" else "falling edge"

    def _clock_clause(self, structure: SVAStructure) -> str:
        if structure.clock_signal:
            edge_desc = self._clock_edge_desc(structure.clock_edge)
            return f"At a {edge_desc} of `{structure.clock_signal}`"
        return "At each evaluation cycle"

    def _expected_timing_clause(self, structure: SVAStructure, same_cycle: bool) -> str:
        if structure.clock_signal:
            edge_desc = self._clock_edge_desc(structure.clock_edge)
            if same_cycle:
                return f"At that same {edge_desc} of `{structure.clock_signal}`"
            return f"On the next {edge_desc} of `{structure.clock_signal}`"
        if same_cycle:
            return "In the same cycle"
        return "In the next cycle"
