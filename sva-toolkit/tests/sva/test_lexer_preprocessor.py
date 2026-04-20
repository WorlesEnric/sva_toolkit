from __future__ import annotations

import pytest

from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.lexer import TokenKind, tokenize, tokenize_with_trivia
from sva_toolkit.sva.preprocessor import preprocess
from sva_toolkit.sva.trivia import TriviaKind


def test_preprocess_records_define_include_and_timescale_directives() -> None:
    source = '`define WIDTH(x) x + \\\n 1\n`include "defs.svh"\n`timescale 1ns/1ps\nassert property (@(posedge clk) req);\n'

    result = preprocess(source)
    tokens, trivia = tokenize_with_trivia(source)

    assert [directive.name for directive in result.directives] == ["define", "include", "timescale"]
    assert result.directives[0].text.startswith("`define WIDTH")
    assert len(result.stripped_text) == len(source)
    assert all(item.kind is TriviaKind.DIRECTIVE for item in result.trivia)
    assert tokens[0].kind is TokenKind.ASSERT
    assert [item.kind for item in trivia[:3]] == [TriviaKind.DIRECTIVE, TriviaKind.DIRECTIVE, TriviaKind.DIRECTIVE]


def test_preprocess_tracks_ifdef_blocks_without_expanding_them() -> None:
    source = "`ifdef SIM\nassert property (@(posedge clk) req);\n`endif\n"

    result = preprocess(source)
    tokens, trivia = tokenize_with_trivia(source)

    assert [directive.name for directive in result.directives] == ["ifdef", "endif"]
    assert tokens[0].kind is TokenKind.ASSERT
    assert [item.kind for item in trivia if item.kind is TriviaKind.DIRECTIVE] == [
        TriviaKind.DIRECTIVE,
        TriviaKind.DIRECTIVE,
    ]


def test_preprocess_strips_attribute_instances_to_trivia() -> None:
    source = '(* attr = "val" *) assert property (@(posedge clk) req);'

    result = preprocess(source)
    tokens, trivia = tokenize_with_trivia(source)

    assert result.trivia[0].kind is TriviaKind.ATTRIBUTE
    assert result.trivia[0].text == '(* attr = "val" *)'
    assert tokens[0].kind is TokenKind.ASSERT
    assert trivia[0].kind is TriviaKind.ATTRIBUTE


def test_preprocess_rejects_unterminated_ifdef_blocks() -> None:
    source = "`ifdef SIM\nassert property (@(posedge clk) req);\n"

    with pytest.raises(SvaSyntaxError) as exc_info:
        preprocess(source)

    assert exc_info.value.position == 0
    assert exc_info.value.message == "unterminated `ifdef` block"


def test_tokenize_surfaces_encrypted_protect_regions() -> None:
    source = "`protect begin_protected\npayload\n`endprotect\nassert property (@(posedge clk) req);\n"

    result = preprocess(source)

    assert result.directives[0].name == "protect"
    with pytest.raises(SvaSyntaxError) as exc_info:
        tokenize(source)

    assert exc_info.value.position == 0
    assert exc_info.value.message == "encrypted preprocessor regions are not supported"
