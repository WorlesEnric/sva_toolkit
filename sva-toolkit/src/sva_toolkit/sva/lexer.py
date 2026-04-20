from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sva_toolkit.sva.ast import SourceSpan
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.preprocessor import preprocess
from sva_toolkit.sva.trivia import Trivia, collect_trivia, consume_string_literal


class TokenKind(str, Enum):
    IDENT = "IDENT"
    LITERAL = "LITERAL"
    DOLLAR = "LITERAL"
    STRING = "STRING"
    DOLLAR_IDENT = "DOLLAR_IDENT"
    PROPERTY = "property"
    ENDPROPERTY = "endproperty"
    ASSERT = "assert"
    ASSUME = "assume"
    COVER = "cover"
    RESTRICT = "restrict"
    EXPECT = "expect"
    SEQUENCE = "sequence"
    ENDSEQUENCE = "endsequence"
    CHECKER = "checker"
    ENDCHECKER = "endchecker"
    BIND = "bind"
    CLOCKING = "clocking"
    ENDCLOCKING = "endclocking"
    LET = "let"
    DISABLE = "disable"
    IFF = "iff"
    IF = "if"
    ELSE = "else"
    AND = "and"
    OR = "or"
    NOT = "not"
    NEXTTIME = "nexttime"
    S_NEXTTIME = "s_nexttime"
    ALWAYS = "always"
    S_ALWAYS = "s_always"
    EVENTUALLY = "eventually"
    S_EVENTUALLY = "s_eventually"
    STRONG = "strong"
    WEAK = "weak"
    INTERSECT = "intersect"
    THROUGHOUT = "throughout"
    WITHIN = "within"
    MATCHED = "matched"
    INSIDE = "inside"
    DIST = "dist"
    UNTIL = "until"
    UNTIL_WITH = "until_with"
    S_UNTIL = "s_until"
    S_UNTIL_WITH = "s_until_with"
    IMPLIES = "implies"
    FIRST_MATCH = "first_match"
    ACCEPT_ON = "accept_on"
    REJECT_ON = "reject_on"
    SYNC_ACCEPT_ON = "sync_accept_on"
    SYNC_REJECT_ON = "sync_reject_on"
    POSEDGE = "posedge"
    NEGEDGE = "negedge"
    EDGE = "edge"
    LOCAL = "local"
    VAR = "var"
    BIT = "bit"
    LOGIC = "logic"
    REG = "reg"
    WIRE = "wire"
    INPUT = "input"
    OUTPUT = "output"
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
    LT_MINUS_GT = "<->"
    EQ_EQ_GT = "==>"
    MINUS_GT = "->"
    LBRACKET_PLUS_RBRACKET = "[+]"
    LBRACKET_STAR_RBRACKET = "[*]"
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
    "restrict": TokenKind.RESTRICT,
    "expect": TokenKind.EXPECT,
    "sequence": TokenKind.SEQUENCE,
    "endsequence": TokenKind.ENDSEQUENCE,
    "checker": TokenKind.CHECKER,
    "endchecker": TokenKind.ENDCHECKER,
    "bind": TokenKind.BIND,
    "clocking": TokenKind.CLOCKING,
    "endclocking": TokenKind.ENDCLOCKING,
    "let": TokenKind.LET,
    "disable": TokenKind.DISABLE,
    "iff": TokenKind.IFF,
    "if": TokenKind.IF,
    "else": TokenKind.ELSE,
    "and": TokenKind.AND,
    "or": TokenKind.OR,
    "not": TokenKind.NOT,
    "nexttime": TokenKind.NEXTTIME,
    "s_nexttime": TokenKind.S_NEXTTIME,
    "always": TokenKind.ALWAYS,
    "s_always": TokenKind.S_ALWAYS,
    "eventually": TokenKind.EVENTUALLY,
    "s_eventually": TokenKind.S_EVENTUALLY,
    "strong": TokenKind.STRONG,
    "weak": TokenKind.WEAK,
    "intersect": TokenKind.INTERSECT,
    "throughout": TokenKind.THROUGHOUT,
    "within": TokenKind.WITHIN,
    "matched": TokenKind.MATCHED,
    "inside": TokenKind.INSIDE,
    "dist": TokenKind.DIST,
    "until": TokenKind.UNTIL,
    "until_with": TokenKind.UNTIL_WITH,
    "s_until": TokenKind.S_UNTIL,
    "s_until_with": TokenKind.S_UNTIL_WITH,
    "implies": TokenKind.IMPLIES,
    "first_match": TokenKind.FIRST_MATCH,
    "accept_on": TokenKind.ACCEPT_ON,
    "reject_on": TokenKind.REJECT_ON,
    "sync_accept_on": TokenKind.SYNC_ACCEPT_ON,
    "sync_reject_on": TokenKind.SYNC_REJECT_ON,
    "posedge": TokenKind.POSEDGE,
    "negedge": TokenKind.NEGEDGE,
    "edge": TokenKind.EDGE,
    "local": TokenKind.LOCAL,
    "var": TokenKind.VAR,
    "bit": TokenKind.BIT,
    "logic": TokenKind.LOGIC,
    "reg": TokenKind.REG,
    "wire": TokenKind.WIRE,
    "input": TokenKind.INPUT,
    "output": TokenKind.OUTPUT,
}

_MULTI_CHAR_TOKENS = (
    ("<->", TokenKind.LT_MINUS_GT),
    ("==>", TokenKind.EQ_EQ_GT),
    ("|=>", TokenKind.BAR_FAT_ARROW),
    ("|->", TokenKind.BAR_ARROW),
    ("[+]", TokenKind.LBRACKET_PLUS_RBRACKET),
    ("[*]", TokenKind.LBRACKET_STAR_RBRACKET),
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
    tokens, _ = tokenize_with_trivia(text)
    return tokens


def tokenize_with_trivia(text: str) -> tuple[list[Token], list[Trivia]]:
    preprocessed = preprocess(text)
    for directive in preprocessed.directives:
        if directive.name == "protect":
            raise SvaSyntaxError(directive.span.start, "encrypted preprocessor regions are not supported", text)

    tokens: list[Token] = []
    trivia: list[Trivia] = []
    skipped = iter(preprocessed.trivia)
    pending = next(skipped, None)
    index = 0
    length = len(text)

    while index < length:
        if pending is not None and pending.span.start == index:
            trivia.append(pending)
            index = pending.span.end
            pending = next(skipped, None)
            continue

        if piece := collect_trivia(text, index):
            trivia.append(piece)
            index = piece.span.end
            continue

        token, index = _consume_token(text, index)
        tokens.append(token)

    eof_span = SourceSpan(length, length)
    tokens.append(Token(TokenKind.EOF, "", eof_span))
    return tokens, trivia


def _consume_token(text: str, index: int) -> tuple[Token, int]:
    char = text[index]
    start = index

    if char.isdigit():
        end = _consume_numeric_literal(text, index)
        return Token(TokenKind.LITERAL, text[start:end], SourceSpan(start, end)), end

    if char == '"':
        end = consume_string_literal(text, index)
        return Token(TokenKind.STRING, text[start:end], SourceSpan(start, end)), end

    if char == "$":
        if index + 1 < len(text) and (text[index + 1].isalpha() or text[index + 1] == "_"):
            index += 2
            while index < len(text) and (text[index].isalnum() or text[index] in {"_", "$"}):
                index += 1
            return Token(TokenKind.DOLLAR_IDENT, text[start:index], SourceSpan(start, index)), index
        index += 1
        return Token(TokenKind.DOLLAR, text[start:index], SourceSpan(start, index)), index

    if char == "\\":
        end = _consume_escaped_identifier(text, index)
        return Token(TokenKind.IDENT, text[start:end], SourceSpan(start, end)), end

    if char.isalpha() or char == "_":
        index += 1
        while index < len(text) and (text[index].isalnum() or text[index] in {"_", "$"}):
            index += 1
        token_text = text[start:index]
        kind = _KEYWORDS.get(token_text.lower(), TokenKind.IDENT)
        return Token(kind, token_text, SourceSpan(start, index)), index

    if text.startswith("->", index) and not _is_goto_repeat_operator(text, index):
        end = index + 2
        return Token(TokenKind.MINUS_GT, text[start:end], SourceSpan(start, end)), end

    for operator, kind in _MULTI_CHAR_TOKENS:
        if text.startswith(operator, index):
            end = index + len(operator)
            return Token(kind, operator, SourceSpan(start, end)), end

    if char in _SINGLE_CHAR_TOKENS:
        index += 1
        return Token(_SINGLE_CHAR_TOKENS[char], char, SourceSpan(start, index)), index

    raise SvaSyntaxError(start, f"unexpected character {char!r}", text)


def _consume_escaped_identifier(text: str, index: int) -> int:
    start = index
    index += 1
    if index >= len(text) or text[index].isspace():
        raise SvaSyntaxError(start, "invalid escaped identifier", text)
    while index < len(text) and not text[index].isspace():
        index += 1
    return index


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


def _is_goto_repeat_operator(text: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    return previous >= 0 and text[previous] == "["


__all__ = ["Token", "TokenKind", "tokenize", "tokenize_with_trivia"]
