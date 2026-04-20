from __future__ import annotations

from pathlib import Path

import pytest

from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.lexer import TokenKind, tokenize, tokenize_with_trivia
from sva_toolkit.sva.trivia import TriviaKind


_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "sva"


def test_tokenize_with_trivia_skips_comments_and_line_continuations() -> None:
    source = "// header\nfoo\\\nbar /* block */ baz"

    tokens, trivia = tokenize_with_trivia(source)

    assert [token.text for token in tokens[:-1]] == ["foo", "bar", "baz"]
    assert [item.kind for item in trivia] == [
        TriviaKind.COMMENT_LINE,
        TriviaKind.WHITESPACE,
        TriviaKind.LINE_CONTINUATION,
        TriviaKind.WHITESPACE,
        TriviaKind.COMMENT_BLOCK,
        TriviaKind.WHITESPACE,
    ]


def test_tokenize_emits_string_tokens_and_escaped_identifiers() -> None:
    source = r'\foo-bar == "a\\n\"b\t"'

    tokens = tokenize(source)

    assert tokens[0].kind is TokenKind.IDENT
    assert tokens[0].text == r"\foo-bar"
    assert tokens[1].kind is TokenKind.EQ_EQ
    assert tokens[2].kind is TokenKind.STRING
    assert tokens[2].text == r'"a\\n\"b\t"'


def test_tokenize_with_trivia_preserves_source_order_and_text() -> None:
    source = " \n// one\n(* attr = \"v\" *) foo /* two */\n"

    _, trivia = tokenize_with_trivia(source)

    assert [item.kind for item in trivia] == [
        TriviaKind.WHITESPACE,
        TriviaKind.COMMENT_LINE,
        TriviaKind.WHITESPACE,
        TriviaKind.ATTRIBUTE,
        TriviaKind.WHITESPACE,
        TriviaKind.WHITESPACE,
        TriviaKind.COMMENT_BLOCK,
        TriviaKind.WHITESPACE,
    ]
    assert [item.text for item in trivia] == [source[item.span.start:item.span.end] for item in trivia]
    assert all(first.span.end <= second.span.start for first, second in zip(trivia, trivia[1:]))


def test_block_comments_stop_at_first_terminator() -> None:
    source = "/* outer /* inner */ property p;"

    tokens, trivia = tokenize_with_trivia(source)

    assert trivia[0].kind is TriviaKind.COMMENT_BLOCK
    assert trivia[0].text == "/* outer /* inner */"
    assert [token.text for token in tokens[:-1]] == ["property", "p", ";"]


def test_unterminated_block_comment_raises_precise_error() -> None:
    source = "assert /* missing"

    with pytest.raises(SvaSyntaxError) as exc_info:
        tokenize(source)

    assert exc_info.value.position == 7
    assert exc_info.value.message == "unterminated block comment"


def test_unterminated_string_raises_precise_error() -> None:
    source = 'assert "missing'

    with pytest.raises(SvaSyntaxError) as exc_info:
        tokenize(source)

    assert exc_info.value.position == 7
    assert exc_info.value.message == "unterminated string literal"


@pytest.mark.parametrize("example_path", sorted(_EXAMPLES_DIR.glob("*.sv")))
def test_examples_tokenize_with_injected_comments(example_path: Path) -> None:
    original = example_path.read_text(encoding="utf-8")
    mutated = "// header\n" + original.replace("\n", " /* block */\n", 1)

    tokens, trivia = tokenize_with_trivia(mutated)

    assert tokens[-1].kind is TokenKind.EOF
    assert any(item.kind is TriviaKind.COMMENT_LINE for item in trivia)
    assert any(item.kind is TriviaKind.COMMENT_BLOCK for item in trivia)
