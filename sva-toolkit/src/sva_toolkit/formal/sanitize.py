"""Template-sanitization helpers shared by formal backends."""

from __future__ import annotations

from dataclasses import dataclass
import re

SV_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SV_RESERVED = frozenset(
    {
        "accept_on",
        "always",
        "and",
        "assert",
        "assume",
        "begin",
        "bind",
        "bit",
        "checker",
        "clocking",
        "cover",
        "default",
        "disable",
        "else",
        "end",
        "endchecker",
        "endclocking",
        "endmodule",
        "endproperty",
        "endsequence",
        "eventually",
        "expect",
        "first_match",
        "genvar",
        "if",
        "iff",
        "initial",
        "input",
        "int",
        "integer",
        "intersect",
        "let",
        "local",
        "localparam",
        "logic",
        "module",
        "negedge",
        "not",
        "or",
        "output",
        "parameter",
        "posedge",
        "property",
        "reg",
        "reject_on",
        "restrict",
        "sequence",
        "strong",
        "sync_accept_on",
        "sync_reject_on",
        "throughout",
        "until",
        "until_with",
        "var",
        "weak",
        "wire",
        "within",
    }
)

_RESET_LEVEL_PATTERN = r"(?:0|1|1'[bB][01])"
_RESET_IDENTIFIER_PATTERN = r"(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
_RESET_SHAPES = (
    re.compile(rf"^{_RESET_IDENTIFIER_PATTERN}$"),
    re.compile(rf"^[!~]\s*{_RESET_IDENTIFIER_PATTERN}$"),
    re.compile(rf"^{_RESET_IDENTIFIER_PATTERN}\s*(?:==|!=|===|!==)\s*{_RESET_LEVEL_PATTERN}$"),
    re.compile(rf"^{_RESET_LEVEL_PATTERN}\s*(?:==|!=|===|!==)\s*{_RESET_IDENTIFIER_PATTERN}$"),
)


@dataclass(frozen=True, slots=True)
class IdentifierError(ValueError):
    """Raised when a value cannot be safely used as a SystemVerilog identifier."""

    reason: str
    value: str

    def __str__(self) -> str:
        return f"{self.reason}: {self.value!r}"


def validate_signal(name: str) -> str:
    """Validate a signal identifier for generated formal templates."""

    return _validate_identifier(name, kind="Signal")


def validate_clock(name: str) -> str:
    """Validate a clock identifier for generated formal templates."""

    return _validate_identifier(name, kind="Clock")


def validate_reset(expr: str) -> str:
    """Validate a simple reset expression used by generated formal templates."""

    text = _strip_wrapping_parens(expr.strip())
    if not text:
        raise IdentifierError("Reset expression must not be empty", expr)
    if "." in text:
        raise IdentifierError(
            "Hierarchical reset identifiers are not supported in formal checker templates",
            expr,
        )

    for pattern in _RESET_SHAPES:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        _validate_identifier(match.group("ident"), kind="Reset")
        return text

    raise IdentifierError(
        "Reset expression must be a single identifier or a simple form like '!rst_n' or 'rst_n == 0'",
        expr,
    )


def escape_body(text: str) -> str:
    """Return body text unchanged at the explicit template-splice boundary."""

    return text


def _validate_identifier(name: str, *, kind: str) -> str:
    value = name.strip()
    if not value:
        raise IdentifierError(f"{kind} identifier must not be empty", name)
    if "." in value:
        raise IdentifierError(
            f"Hierarchical {kind.lower()} identifiers are not supported in formal checker templates",
            name,
        )
    if not SV_IDENTIFIER_PATTERN.fullmatch(value):
        raise IdentifierError(
            f"{kind} identifier must match {SV_IDENTIFIER_PATTERN.pattern}",
            name,
        )
    if value.lower() in SV_RESERVED:
        raise IdentifierError(
            f"{kind} identifier is a reserved SystemVerilog keyword",
            name,
        )
    return value


def _strip_wrapping_parens(text: str) -> str:
    result = text
    while result.startswith("(") and result.endswith(")") and _outer_parens_wrap_entire_text(result):
        result = result[1:-1].strip()
    return result


def _outer_parens_wrap_entire_text(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0


__all__ = [
    "IdentifierError",
    "SV_IDENTIFIER_PATTERN",
    "SV_RESERVED",
    "escape_body",
    "validate_clock",
    "validate_reset",
    "validate_signal",
]
