"""
SVAD Translator - Convert SVA code into a markdown SVAD template.

This module parses SVA code, builds a lightweight symbolic summary, and
renders it into the requested markdown template.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

from sva_toolkit.ast_parser import SVAASTParser, SVAStructure
from sva_toolkit.ast_parser.parser import BuiltinFunction, ImplicationType
from sva_toolkit.gen.nl.extractor import SignalFormatter, TemporalFormatter


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
