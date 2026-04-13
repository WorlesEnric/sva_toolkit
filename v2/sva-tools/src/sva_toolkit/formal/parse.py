"""Practical SVA property parsing helpers."""

from __future__ import annotations

import re

from sva_toolkit.formal.model import FormalProperty
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.lexer import TokenKind, tokenize
from sva_toolkit.sva.parser import parse_property_text


IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_CLOCK_RE = re.compile(rf"@\s*\(\s*(posedge|negedge)\s+({IDENT})\s*\)", re.IGNORECASE)
_HEADER_RE = re.compile(rf"\bproperty\s+({IDENT})\b", re.IGNORECASE)


def parse_property(text: str) -> FormalProperty:
    """Parse a practical SVA property/assertion into a thin normalized model."""

    stripped = text.strip()
    try:
        spec = parse_property_text(stripped)
    except SvaSyntaxError:
        return _parse_property_fallback(stripped)
    return FormalProperty(name=spec.name, body="", ast=spec)


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
        clock_name=clock_name or "clk",
        reset_expr=reset_expr or "!rst_n",
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


def _extract_clocking(text: str) -> tuple[str, str | None]:
    if match := _CLOCK_RE.search(text):
        return match.group(1).lower(), match.group(2)
    return "posedge", None


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
