"""Formal property models, annotation errors, and compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import re

from sva_toolkit.sva.analysis import CollectIdentifiersVisitor
from sva_toolkit.sva.ast import ClockingEvent, MultiEventClocking, PropertySpec
from sva_toolkit.sva.emitter import emit_expr, emit_property_body
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.lexer import TokenKind, tokenize


class ImplicationResult(str, Enum):
    """Normalized result states for implication/equivalence checks."""

    IMPLIES = "implies"
    NOT_IMPLIES = "not_implies"
    EQUIVALENT = "equivalent"
    TIMEOUT = "timeout"
    ERROR = "error"
    SYNTAX_ERROR = "syntax_error"


@dataclass(frozen=True)
class CheckResult:
    """Structured result returned by formal backends/services."""

    result: ImplicationResult
    message: str
    counterexample: str | None = None
    log: str | None = None
    module: str | None = None


class FormalPropertyError(Exception):
    """Base class for formal-property annotation/configuration errors."""


class MissingClockingError(FormalPropertyError):
    """Raised when a property has no explicit clocking annotation."""


class MissingResetError(FormalPropertyError):
    """Raised when a property has no explicit reset annotation."""


class ClockMismatchError(FormalPropertyError):
    """Raised when two properties disagree on their effective clocking."""


class ResetMismatchError(FormalPropertyError):
    """Raised when two properties disagree on their effective reset."""


class UnsupportedClockingError(FormalPropertyError):
    """Raised when a property uses clocking that formal normalization cannot reduce."""


NormalizedResetStream = tuple[tuple[str, str], ...]


def _derived_signals_from_ast(spec: PropertySpec) -> frozenset[str]:
    names = CollectIdentifiersVisitor().visit(spec.body)
    bound_names = {formal.name for formal in spec.formals}
    bound_names.update(local.name for local in spec.local_vars)
    if spec.clocking is not None:
        bound_names.update(_clock_signal_names(spec.clocking))
    if spec.disable_iff is not None:
        bound_names.update(CollectIdentifiersVisitor().visit(spec.disable_iff))
    return frozenset(names - bound_names)


def _clock_signal_names(clocking: ClockingEvent | MultiEventClocking) -> frozenset[str]:
    if isinstance(clocking, MultiEventClocking):
        return frozenset(event.signal.name for event in clocking.events)
    return frozenset({clocking.signal.name})


def _clocking_from_ast(spec: PropertySpec) -> tuple[str | None, str | None]:
    if spec.clocking is None:
        return None, None
    if isinstance(spec.clocking, MultiEventClocking):
        return None, None
    return spec.clocking.edge.value, spec.clocking.signal.name


def normalize_clock_edge(clock_edge: str) -> str:
    normalized = clock_edge.strip().lower()
    if not normalized:
        raise ValueError("clock edge must not be empty")
    return normalized


@dataclass(frozen=True)
class FormalProperty:
    """Thin normalized property model used by timing and formal workflows."""

    body: str
    clock_edge: str | None = None
    clock_name: str | None = None
    reset_expr: str | None = None
    signals: frozenset[str] = field(default_factory=frozenset)
    name: str | None = None
    has_explicit_reset: bool = False
    ast: PropertySpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", frozenset(self.signals))

    @classmethod
    def from_ast(cls, spec: PropertySpec) -> "FormalProperty":
        """Build a normalized property model from a parsed AST."""

        clock_edge, clock_name = _clocking_from_ast(spec)
        return cls(
            body=emit_property_body(spec.body),
            clock_edge=clock_edge,
            clock_name=clock_name,
            reset_expr=emit_expr(spec.disable_iff) if spec.disable_iff is not None else None,
            signals=_derived_signals_from_ast(spec),
            name=spec.name,
            has_explicit_reset=spec.disable_iff is not None,
            ast=spec,
        )

    @property
    def reset_name(self) -> str:
        if self.reset_expr is None:
            raise MissingResetError(
                "Explicit reset annotation is required because the property text does not name a reset expression."
            )
        interpreted = _interpret_reset_stream(normalize_reset_token_stream(self.reset_expr))
        if interpreted is not None:
            return interpreted[0]
        match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", self.reset_expr)
        if match:
            return match.group(0)
        raise ResetMismatchError(f"Unable to derive a reset signal name from {self.reset_expr!r}.")

    @property
    def reset_sense(self) -> str:
        if self.reset_expr is None:
            raise MissingResetError(
                "Explicit reset annotation is required because the property text does not name a reset expression."
            )
        interpreted = _interpret_reset_stream(normalize_reset_token_stream(self.reset_expr))
        if interpreted is not None:
            return interpreted[1]
        expr = self.reset_expr.strip()
        if expr.startswith(("!", "~")):
            return "low"
        if re.search(r"==\s*0\b", expr):
            return "low"
        return "high"

    def model_copy(self, *, update: dict[str, object] | None = None) -> "FormalProperty":
        return replace(self, **(update or {}))


def normalize_reset_token_stream(expr: str) -> NormalizedResetStream:
    """Tokenize a reset expression and normalize simple semantic aliases."""

    stripped = expr.strip()
    if not stripped:
        return ()

    try:
        tokens = [token for token in tokenize(stripped) if token.kind is not TokenKind.EOF]
    except SvaSyntaxError:
        return (("RAW", " ".join(stripped.split())),)

    tokens = _strip_wrapping_parens(tokens)
    canonical = _canonicalize_simple_reset(tokens)
    if canonical is not None:
        return canonical
    return tuple((token.kind.name, token.text) for token in tokens)


def reset_exprs_equivalent(lhs: str, rhs: str) -> bool:
    """Return True when two reset expressions normalize to the same token stream."""

    return normalize_reset_token_stream(lhs) == normalize_reset_token_stream(rhs)


def canonicalize_reset_expr(expr: str) -> str:
    """Render a normalized reset expression suitable for backend comparison."""

    stream = normalize_reset_token_stream(expr)
    interpreted = _interpret_reset_stream(stream)
    if interpreted is not None:
        ident, sense = interpreted
        level = "0" if sense == "low" else "1"
        return f"{ident} == {level}"
    if stream == ():
        return ""
    if stream[0][0] == "RAW":
        return stream[0][1]
    return " ".join(text for _, text in stream)


def harmonize_property_pair(
    antecedent: FormalProperty,
    consequent: FormalProperty,
) -> tuple[FormalProperty, FormalProperty]:
    """Normalize two properties and require their effective annotations to match."""

    antecedent_edge, antecedent_clock = _normalized_clocking(antecedent)
    consequent_edge, consequent_clock = _normalized_clocking(consequent)
    if (antecedent_edge, antecedent_clock) != (consequent_edge, consequent_clock):
        raise ClockMismatchError(
            "Property clock mismatch: "
            f"{antecedent_edge} {antecedent_clock} vs {consequent_edge} {consequent_clock}."
        )

    antecedent_reset = _normalized_reset_expr(antecedent)
    consequent_reset = _normalized_reset_expr(consequent)
    if not reset_exprs_equivalent(antecedent_reset, consequent_reset):
        raise ResetMismatchError(f"Property reset mismatch: {antecedent_reset} vs {consequent_reset}.")

    canonical_reset = canonicalize_reset_expr(antecedent_reset)
    update = {
        "clock_edge": antecedent_edge,
        "clock_name": antecedent_clock,
        "reset_expr": canonical_reset,
    }
    return antecedent.model_copy(update=update), consequent.model_copy(update=update)


def _normalized_clocking(prop: FormalProperty) -> tuple[str, str]:
    if prop.clock_edge is None or prop.clock_name is None:
        raise MissingClockingError(
            "Explicit clocking is required because the property text does not name a clocking event."
        )
    return normalize_clock_edge(prop.clock_edge), prop.clock_name.strip()


def _normalized_reset_expr(prop: FormalProperty) -> str:
    if prop.reset_expr is None:
        raise MissingResetError(
            "Explicit reset annotation is required because the property text does not name a reset expression."
        )
    return prop.reset_expr.strip()


def _strip_wrapping_parens(tokens: list) -> list:
    result = list(tokens)
    while len(result) >= 2 and result[0].kind is TokenKind.LPAREN and result[-1].kind is TokenKind.RPAREN:
        depth = 0
        wrapped = True
        for index, token in enumerate(result):
            if token.kind is TokenKind.LPAREN:
                depth += 1
            elif token.kind is TokenKind.RPAREN:
                depth -= 1
                if depth == 0 and index != len(result) - 1:
                    wrapped = False
                    break
                if depth < 0:
                    wrapped = False
                    break
        if not wrapped or depth != 0:
            break
        result = result[1:-1]
    return result


def _canonicalize_simple_reset(tokens: list) -> NormalizedResetStream | None:
    if len(tokens) == 1 and tokens[0].kind is TokenKind.IDENT:
        return (
            ("IDENT", tokens[0].text),
            ("EQ_EQ", "=="),
            ("LITERAL", "1"),
        )
    if len(tokens) == 2 and tokens[0].kind in {TokenKind.BANG, TokenKind.TILDE} and tokens[1].kind is TokenKind.IDENT:
        return (
            ("IDENT", tokens[1].text),
            ("EQ_EQ", "=="),
            ("LITERAL", "0"),
        )
    if len(tokens) != 3:
        return None

    left, op, right = tokens
    if op.kind not in {TokenKind.EQ_EQ, TokenKind.BANG_EQ, TokenKind.EQ_EQ_EQ, TokenKind.BANG_EQ_EQ}:
        return None

    if left.kind is TokenKind.IDENT:
        ident = left.text
        literal = _literal_to_bool(right.text)
    elif right.kind is TokenKind.IDENT:
        ident = right.text
        literal = _literal_to_bool(left.text)
    else:
        return None
    if literal is None:
        return None

    if op.kind in {TokenKind.EQ_EQ, TokenKind.EQ_EQ_EQ}:
        active_level = literal
    else:
        active_level = 1 - literal
    return (
        ("IDENT", ident),
        ("EQ_EQ", "=="),
        ("LITERAL", str(active_level)),
    )


def _literal_to_bool(text: str) -> int | None:
    normalized = text.replace("_", "").lower()
    if normalized in {"0", "1'b0"}:
        return 0
    if normalized in {"1", "1'b1"}:
        return 1
    return None


def _interpret_reset_stream(stream: NormalizedResetStream) -> tuple[str, str] | None:
    if len(stream) != 3:
        return None
    if stream[0][0] != "IDENT" or stream[1] != ("EQ_EQ", "==") or stream[2][0] != "LITERAL":
        return None
    if stream[2][1] == "0":
        return stream[0][1], "low"
    if stream[2][1] == "1":
        return stream[0][1], "high"
    return None
