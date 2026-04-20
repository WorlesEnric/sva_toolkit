from __future__ import annotations

import pytest

from sva_toolkit.sva.lexer import TokenKind, tokenize


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("nexttime", TokenKind.NEXTTIME),
        ("s_nexttime", TokenKind.S_NEXTTIME),
        ("always", TokenKind.ALWAYS),
        ("s_always", TokenKind.S_ALWAYS),
        ("eventually", TokenKind.EVENTUALLY),
        ("s_eventually", TokenKind.S_EVENTUALLY),
        ("strong", TokenKind.STRONG),
        ("weak", TokenKind.WEAK),
        ("restrict", TokenKind.RESTRICT),
        ("expect", TokenKind.EXPECT),
        ("sequence", TokenKind.SEQUENCE),
        ("endsequence", TokenKind.ENDSEQUENCE),
        ("checker", TokenKind.CHECKER),
        ("endchecker", TokenKind.ENDCHECKER),
        ("bind", TokenKind.BIND),
        ("clocking", TokenKind.CLOCKING),
        ("endclocking", TokenKind.ENDCLOCKING),
        ("let", TokenKind.LET),
        ("within", TokenKind.WITHIN),
        ("matched", TokenKind.MATCHED),
        ("inside", TokenKind.INSIDE),
        ("dist", TokenKind.DIST),
        ("s_until", TokenKind.S_UNTIL),
        ("s_until_with", TokenKind.S_UNTIL_WITH),
        ("implies", TokenKind.IMPLIES),
        ("edge", TokenKind.EDGE),
        ("bit", TokenKind.BIT),
        ("logic", TokenKind.LOGIC),
        ("reg", TokenKind.REG),
        ("wire", TokenKind.WIRE),
        ("input", TokenKind.INPUT),
        ("output", TokenKind.OUTPUT),
    ],
)
def test_tokenize_recognizes_new_keywords(text: str, kind: TokenKind) -> None:
    tokens = tokenize(text)

    assert [token.kind for token in tokens] == [kind, TokenKind.EOF]


def test_tokenize_recognizes_new_operator_tokens_and_bare_dollar() -> None:
    tokens = tokenize("lhs -> rhs <-> eq ==> prop [*] [+] $")

    assert [token.kind for token in tokens[:-1]] == [
        TokenKind.IDENT,
        TokenKind.MINUS_GT,
        TokenKind.IDENT,
        TokenKind.LT_MINUS_GT,
        TokenKind.IDENT,
        TokenKind.EQ_EQ_GT,
        TokenKind.IDENT,
        TokenKind.LBRACKET_STAR_RBRACKET,
        TokenKind.LBRACKET_PLUS_RBRACKET,
        TokenKind.DOLLAR,
    ]


def test_tokenize_keeps_goto_repeat_parser_compatible() -> None:
    tokens = tokenize("data [->2]")

    assert [token.kind for token in tokens[:-1]] == [
        TokenKind.IDENT,
        TokenKind.LBRACKET,
        TokenKind.MINUS,
        TokenKind.GT,
        TokenKind.LITERAL,
        TokenKind.RBRACKET,
    ]
