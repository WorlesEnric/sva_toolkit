"""Minimal formal property and result models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import re

from sva_toolkit.sva.analysis import CollectIdentifiersVisitor
from sva_toolkit.sva.ast import PropertySpec
from sva_toolkit.sva.emitter import emit_expr, emit_property_body


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


def _derived_signals_from_ast(spec: PropertySpec) -> frozenset[str]:
    names = CollectIdentifiersVisitor().visit(spec.body)
    bound_names = {formal.name for formal in spec.formals}
    bound_names.update(local.name for local in spec.local_vars)
    if spec.clocking is not None:
        bound_names.add(spec.clocking.signal.name)
    if spec.disable_iff is not None:
        bound_names.update(CollectIdentifiersVisitor().visit(spec.disable_iff))
    return frozenset(names - bound_names)


def _compat_fields_from_ast(spec: PropertySpec) -> dict[str, object]:
    return {
        "body": emit_property_body(spec.body),
        "clock_edge": spec.clocking.edge.value if spec.clocking is not None else "posedge",
        "clock_name": spec.clocking.signal.name if spec.clocking is not None else "clk",
        "reset_expr": emit_expr(spec.disable_iff) if spec.disable_iff is not None else "!rst_n",
        "signals": _derived_signals_from_ast(spec),
        "has_explicit_reset": spec.disable_iff is not None,
    }


@dataclass(frozen=True)
class FormalProperty:
    """Thin normalized property model used by timing and formal workflows."""

    body: str
    clock_edge: str = "posedge"
    clock_name: str = "clk"
    reset_expr: str = "!rst_n"
    signals: frozenset[str] = field(default_factory=frozenset)
    name: str | None = None
    has_explicit_reset: bool = False
    ast: PropertySpec | None = None

    def __post_init__(self) -> None:
        if self.ast is not None:
            if self.ast.name is not None:
                object.__setattr__(self, "name", self.ast.name)
            for field_name, value in _compat_fields_from_ast(self.ast).items():
                object.__setattr__(self, field_name, value)
        object.__setattr__(self, "signals", frozenset(self.signals))

    @property
    def reset_name(self) -> str:
        match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", self.reset_expr)
        if match:
            return match.group(0)
        return "rst_n"

    @property
    def reset_sense(self) -> str:
        expr = self.reset_expr.strip()
        if expr.startswith(("!", "~")):
            return "low"
        if re.search(r"==\s*0\b", expr):
            return "low"
        return "high"

    def model_copy(self, *, update: dict[str, object] | None = None) -> "FormalProperty":
        return replace(self, **(update or {}))
