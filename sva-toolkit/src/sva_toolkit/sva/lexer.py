from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sva_toolkit.sva.ast import SourceSpan
from sva_toolkit.sva.errors import SvaSyntaxError


class TokenKind(str, Enum):
    IDENT = "IDENT"
    LITERAL = "LITERAL"
    DOLLAR_IDENT = "DOLLAR_IDENT"
    PROPERTY = "property"
    ENDPROPERTY = "endproperty"
    ASSERT = "assert"
    ASSUME = "assume"
    COVER = "cover"
    DISABLE = "disable"
    IFF = "iff"
    IF = "if"
    ELSE = "else"
    AND = "and"
    OR = "or"
    NOT = "not"
    INTERSECT = "intersect"
    THROUGHOUT = "throughout"
    UNTIL = "until"
    UNTIL_WITH = "until_with"
    FIRST_MATCH = "first_match"
    ACCEPT_ON = "accept_on"
    REJECT_ON = "reject_on"
    SYNC_ACCEPT_ON = "sync_accept_on"
    SYNC_REJECT_ON = "sync_reject_on"
    POSEDGE = "posedge"
    NEGEDGE = "negedge"
    LOCAL = "local"
    VAR = "var"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    SEMI = ";"
    COMMA = ","
    DOT = "."
    COLON = ":"
    QUESTION = "?"
    AT = "@"
    HASH_HASH = "##"
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    BANG = "!"
    TILDE = "~"
    AMP = "&"
    AMP_AMP = "&&"
    PIPE = "|"
    PIPE_PIPE = "||"
    CARET = "^"
    CARET_TILDE = "^~"
    TILDE_CARET = "~^"
    ASSIGN = "="
    EQ_EQ = "=="
    BANG_EQ = "!="
    EQ_EQ_EQ = "==="
    BANG_EQ_EQ = "!=="
    LT = "<"
    LT_EQ = "<="
    GT = ">"
    GT_EQ = ">="
    BAR_ARROW = "|->"
    BAR_FAT_ARROW = "|=>"
    EOF = "EOF"


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    text: str
    span: SourceSpan


_KEYWORDS = {
    "property": TokenKind.PROPERTY,
    "endproperty": TokenKind.ENDPROPERTY,
    "assert": TokenKind.ASSERT,
    "assume": TokenKind.ASSUME,
    "cover": TokenKind.COVER,
    "disable": TokenKind.DISABLE,
    "iff": TokenKind.IFF,
    "if": TokenKind.IF,
    "else": TokenKind.ELSE,
    "and": TokenKind.AND,
    "or": TokenKind.OR,
    "not": TokenKind.NOT,
    "intersect": TokenKind.INTERSECT,
    "throughout": TokenKind.THROUGHOUT,
    "until": TokenKind.UNTIL,
    "until_with": TokenKind.UNTIL_WITH,
    "first_match": TokenKind.FIRST_MATCH,
    "accept_on": TokenKind.ACCEPT_ON,
    "reject_on": TokenKind.REJECT_ON,
    "sync_accept_on": TokenKind.SYNC_ACCEPT_ON,
    "sync_reject_on": TokenKind.SYNC_REJECT_ON,
    "posedge": TokenKind.POSEDGE,
    "negedge": TokenKind.NEGEDGE,
    "local": TokenKind.LOCAL,
    "var": TokenKind.VAR,
}

_MULTI_CHAR_TOKENS = (
    ("|=>", TokenKind.BAR_FAT_ARROW),
    ("|->", TokenKind.BAR_ARROW),
    ("===", TokenKind.EQ_EQ_EQ),
    ("!==", TokenKind.BANG_EQ_EQ),
    ("==", TokenKind.EQ_EQ),
    ("!=", TokenKind.BANG_EQ),
    ("<=", TokenKind.LT_EQ),
    (">=", TokenKind.GT_EQ),
    ("&&", TokenKind.AMP_AMP),
    ("||", TokenKind.PIPE_PIPE),
    ("^~", TokenKind.CARET_TILDE),
    ("~^", TokenKind.TILDE_CARET),
    ("##", TokenKind.HASH_HASH),
)

_SINGLE_CHAR_TOKENS = {
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    ";": TokenKind.SEMI,
    ",": TokenKind.COMMA,
    ".": TokenKind.DOT,
    ":": TokenKind.COLON,
    "?": TokenKind.QUESTION,
    "@": TokenKind.AT,
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "/": TokenKind.SLASH,
    "%": TokenKind.PERCENT,
    "!": TokenKind.BANG,
    "~": TokenKind.TILDE,
    "&": TokenKind.AMP,
    "|": TokenKind.PIPE,
    "^": TokenKind.CARET,
    "=": TokenKind.ASSIGN,
    "<": TokenKind.LT,
    ">": TokenKind.GT,
}


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue

        start = index

        if char.isdigit():
            index = _consume_numeric_literal(text, index)
            tokens.append(Token(TokenKind.LITERAL, text[start:index], SourceSpan(start, index)))
            continue

        if char == "$":
            if index + 1 < length and (text[index + 1].isalpha() or text[index + 1] == "_"):
                index += 2
                while index < length and (text[index].isalnum() or text[index] in {"_", "$"}):
                    index += 1
                tokens.append(Token(TokenKind.DOLLAR_IDENT, text[start:index], SourceSpan(start, index)))
                continue
            index += 1
            tokens.append(Token(TokenKind.LITERAL, text[start:index], SourceSpan(start, index)))
            continue

        if char.isalpha() or char == "_":
            index += 1
            while index < length and (text[index].isalnum() or text[index] in {"_", "$"}):
                index += 1
            token_text = text[start:index]
            kind = _KEYWORDS.get(token_text.lower(), TokenKind.IDENT)
            tokens.append(Token(kind, token_text, SourceSpan(start, index)))
            continue

        matched = False
        for operator, kind in _MULTI_CHAR_TOKENS:
            if text.startswith(operator, index):
                end = index + len(operator)
                tokens.append(Token(kind, operator, SourceSpan(index, end)))
                index = end
                matched = True
                break
        if matched:
            continue

        if char in _SINGLE_CHAR_TOKENS:
            index += 1
            tokens.append(Token(_SINGLE_CHAR_TOKENS[char], char, SourceSpan(start, index)))
            continue

        raise SvaSyntaxError(start, f"unexpected character {char!r}", text)

    eof_span = SourceSpan(length, length)
    tokens.append(Token(TokenKind.EOF, "", eof_span))
    return tokens


def _consume_numeric_literal(text: str, index: int) -> int:
    length = len(text)
    while index < length and (text[index].isdigit() or text[index] == "_"):
        index += 1
    if index >= length or text[index] != "'":
        return index

    quote_index = index
    index += 1
    if index < length and text[index] in {"s", "S"}:
        index += 1
    if index >= length or text[index] not in {"b", "B", "d", "D", "h", "H", "o", "O"}:
        raise SvaSyntaxError(quote_index, "invalid sized literal", text)
    index += 1
    while index < length and (text[index].isalnum() or text[index] in {"_", "?", "x", "X", "z", "Z"}):
        index += 1
    return index


__all__ = ["Token", "TokenKind", "tokenize"]
