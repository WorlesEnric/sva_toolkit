"""Practical SVA property parsing helpers with explicit clock/reset handling."""

from __future__ import annotations

import re

from sva_toolkit.formal.model import (
    ClockMismatchError,
    FormalProperty,
    MissingClockingError,
    MissingResetError,
    ResetMismatchError,
    UnsupportedClockingError,
    normalize_clock_edge,
    reset_exprs_equivalent,
)
from sva_toolkit.formal.sanitize import validate_clock, validate_reset
from sva_toolkit.sva.ast import MultiEventClocking
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.lexer import TokenKind, tokenize
from sva_toolkit.sva.parser import parse_property_text


IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_CLOCK_RE = re.compile(rf"@\s*\(\s*(posedge|negedge|edge)\s+({IDENT})\s*\)", re.IGNORECASE)
_HEADER_RE = re.compile(rf"\bproperty\s+({IDENT})\b", re.IGNORECASE)


def parse_property(
    text: str,
    *,
    clock: str | None = None,
    clock_edge: str | None = None,
    reset: str | None = None,
    require_clocking: bool = True,
    require_reset: bool = True,
) -> FormalProperty:
    """Parse an SVA property/assertion and require explicit clock/reset annotations."""

    stripped = text.strip()
    try:
        spec = parse_property_text(stripped)
    except SvaSyntaxError:
        prop = _parse_property_fallback(stripped)
    else:
        prop = FormalProperty.from_ast(spec)
    return _require_explicit_annotations(
        prop,
        clock=clock,
        clock_edge=clock_edge,
        reset=reset,
        require_clocking=require_clocking,
        require_reset=require_reset,
    )


def split_property_texts(text: str) -> tuple[str, ...]:
    """Split an SVA source file into standalone property/assertion surfaces."""

    stripped = text.strip()
    if not stripped:
        return ()

    try:
        tokens = tokenize(stripped)
    except SvaSyntaxError:
        return (stripped,)

    blocks: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.kind == TokenKind.PROPERTY and index + 1 < len(tokens) and tokens[index + 1].kind == TokenKind.IDENT:
            end_index = index + 1
            while end_index < len(tokens) and tokens[end_index].kind != TokenKind.ENDPROPERTY:
                end_index += 1
            if end_index < len(tokens):
                blocks.append(stripped[token.span.start:tokens[end_index].span.end].strip())
                index = end_index + 1
                continue

        if token.kind in {TokenKind.ASSERT, TokenKind.ASSUME, TokenKind.COVER}:
            end_index = _find_assertion_statement_end(tokens, index + 1)
            if end_index is not None:
                blocks.append(stripped[token.span.start:tokens[end_index].span.end].strip())
                index = end_index + 1
                continue

        index += 1

    return tuple(blocks) if blocks else (stripped,)


def _find_assertion_statement_end(tokens, start_index: int) -> int | None:
    if start_index >= len(tokens) or tokens[start_index].kind != TokenKind.PROPERTY:
        return None

    paren_depth = 0
    seen_lparen = False
    index = start_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token.kind == TokenKind.LPAREN:
            paren_depth += 1
            seen_lparen = True
        elif token.kind == TokenKind.RPAREN:
            paren_depth = max(0, paren_depth - 1)
        elif token.kind == TokenKind.SEMI and seen_lparen and paren_depth == 0:
            return index
        index += 1
    return None


def _parse_property_fallback(text: str) -> FormalProperty:
    name, body = _extract_property_surface(text)
    clock_edge, clock_name = _extract_clocking(body or text)
    reset_expr = _extract_disable_iff(body or text)
    normalized_body = _normalize_body(body)
    return FormalProperty(
        name=name,
        body=normalized_body,
        clock_edge=clock_edge,
        clock_name=clock_name,
        reset_expr=reset_expr,
        signals=_extract_signals(normalized_body),
        has_explicit_reset=reset_expr is not None,
    )


def _extract_property_surface(text: str) -> tuple[str | None, str]:
    lower = text.lower()
    if "endproperty" in lower and (header_match := _HEADER_RE.search(text)):
        name = header_match.group(1)
        header_end = text.find(";", header_match.end())
        if header_end == -1:
            raise ValueError("named property is missing ';' after the header")
        end_index = lower.rfind("endproperty")
        body = text[header_end + 1:end_index].strip()
        return name, body

    property_index = lower.find("property")
    if property_index != -1:
        paren_start = text.find("(", property_index)
        if paren_start != -1:
            body, _ = _extract_parenthesized(text, paren_start)
            return None, body

    return None, text.rstrip(";").strip()


def _extract_clocking(text: str) -> tuple[str | None, str | None]:
    if match := _CLOCK_RE.search(text):
        return match.group(1).lower(), match.group(2)
    return None, None


def _extract_disable_iff(text: str) -> str | None:
    lower = text.lower()
    marker = "disable iff"
    start = lower.find(marker)
    if start == -1:
        return None
    paren_start = text.find("(", start + len(marker))
    if paren_start == -1:
        return None
    value, _ = _extract_parenthesized(text, paren_start)
    return value.strip()


def _normalize_body(body: str) -> str:
    normalized = body.strip().rstrip(";")
    normalized = _CLOCK_RE.sub("", normalized)
    normalized = _remove_disable_iff(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return _strip_outer_parens(normalized)


def _remove_disable_iff(text: str) -> str:
    lower = text.lower()
    marker = "disable iff"
    start = lower.find(marker)
    if start == -1:
        return text
    paren_start = text.find("(", start + len(marker))
    if paren_start == -1:
        return text
    _, paren_end = _extract_parenthesized(text, paren_start)
    return (text[:start] + text[paren_end + 1:]).strip()


def _extract_parenthesized(text: str, start_index: int) -> tuple[str, int]:
    if start_index >= len(text) or text[start_index] != "(":
        raise ValueError("parenthesized extraction must start at '('")
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start_index + 1:index], index
    raise ValueError("unbalanced parentheses in property text")


def _strip_outer_parens(text: str) -> str:
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        wrapped = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    wrapped = False
                    break
        if not wrapped:
            break
        text = text[1:-1].strip()
    return text


def _extract_signals(body: str) -> tuple[str, ...]:
    reserved = {
        "accept_on",
        "and",
        "assert",
        "assume",
        "begin",
        "bit",
        "cover",
        "disable",
        "else",
        "end",
        "endproperty",
        "endsequence",
        "iff",
        "if",
        "input",
        "intersect",
        "logic",
        "module",
        "negedge",
        "not",
        "or",
        "output",
        "posedge",
        "property",
        "reg",
        "reject_on",
        "sequence",
        "sync_accept_on",
        "sync_reject_on",
        "throughout",
        "until",
        "until_with",
        "var",
        "wire",
    }
    names = []
    for token in re.findall(rf"\b{IDENT}\b", body):
        if token in reserved or token.isupper():
            continue
        if token not in names:
            names.append(token)
    return tuple(names)


def _require_explicit_annotations(
    prop: FormalProperty,
    *,
    clock: str | None,
    clock_edge: str | None,
    reset: str | None,
    require_clocking: bool,
    require_reset: bool,
) -> FormalProperty:
    if prop.ast is not None and isinstance(prop.ast.clocking, MultiEventClocking):
        raise UnsupportedClockingError(
            "Only single-event @(posedge clk) or @(negedge clk) clocking is supported for formal normalization."
        )

    supplied_clock_name, supplied_clock_edge = _normalize_supplied_clocking(clock=clock, clock_edge=clock_edge)
    effective_clock_name = prop.clock_name.strip() if prop.clock_name is not None else None
    effective_clock_edge = normalize_clock_edge(prop.clock_edge) if prop.clock_edge is not None else None

    if effective_clock_name is None or effective_clock_edge is None:
        if supplied_clock_name is None or supplied_clock_edge is None:
            if require_clocking:
                raise MissingClockingError(
                    "Explicit clocking is required because the property text does not name a clocking event; "
                    "provide `@(posedge clk)` in the property or pass both `clock`/`clock_edge` "
                    "(`--clock`/`--clock-edge` in the CLI)."
                )
        effective_clock_name = supplied_clock_name
        effective_clock_edge = supplied_clock_edge
    elif supplied_clock_name is not None and supplied_clock_edge is not None:
        if (effective_clock_edge, effective_clock_name) != (supplied_clock_edge, supplied_clock_name):
            raise ClockMismatchError(
                "Explicit clocking override does not match the property text: "
                f"{effective_clock_edge} {effective_clock_name} vs {supplied_clock_edge} {supplied_clock_name}."
            )

    supplied_reset = validate_reset(reset) if reset is not None else None
    effective_reset = prop.reset_expr.strip() if prop.reset_expr is not None else None
    if effective_reset is None:
        if supplied_reset is None:
            if require_reset:
                raise MissingResetError(
                    "Explicit reset annotation is required because the property text does not name a reset expression; "
                    "provide `disable iff (...)` in the property or pass `reset` (`--reset` in the CLI)."
                )
        effective_reset = supplied_reset
    elif supplied_reset is not None and not reset_exprs_equivalent(effective_reset, supplied_reset):
        raise ResetMismatchError(
            "Explicit reset override does not match the property text: "
            f"{effective_reset} vs {supplied_reset}."
        )

    return prop.model_copy(
        update={
            "clock_name": effective_clock_name,
            "clock_edge": effective_clock_edge,
            "reset_expr": effective_reset,
        }
    )


def _normalize_supplied_clocking(*, clock: str | None, clock_edge: str | None) -> tuple[str | None, str | None]:
    if clock is None and clock_edge is None:
        return None, None
    if clock is None or clock_edge is None:
        raise MissingClockingError(
            "Provide both `clock` and `clock_edge` together when the property text does not already name clocking."
        )
    return validate_clock(clock), normalize_clock_edge(clock_edge)
