from __future__ import annotations

import json
from pathlib import Path

import pytest

from sva_toolkit.cli.exit_codes import ExitCode
from sva_toolkit.cli.main import main
from sva_toolkit.describe import SVACoTBuilder, SVADTranslator
from sva_toolkit.describe.translator import SVAASTParser
from sva_toolkit.sva.ast import (
    ClockEdge,
    ClockingEvent,
    ExprSequence,
    Identifier,
    ImplicationOperator,
    ImplicationProperty,
    OpaqueSequence,
    PropertySpec,
    SourceSpan,
)
from sva_toolkit.sva.lexer import TokenKind, tokenize, tokenize_with_trivia
from sva_toolkit.sva.trivia import TriviaKind


pytestmark = pytest.mark.integration


class _StaticStructureParser:
    def __init__(self, structure) -> None:
        self._structure = structure

    def parse(self, sva_code: str):
        del sva_code
        return self._structure


def _opaque_inner_sequence_case() -> tuple[str, object]:
    span = SourceSpan(0, 0)
    raw_code = "assert property (@(posedge clk) req |=> (ack and));"
    spec = PropertySpec(
        clocking=ClockingEvent(
            edge=ClockEdge.POSEDGE,
            signal=Identifier(name="clk", span=span),
            span=span,
        ),
        body=ImplicationProperty(
            antecedent=ExprSequence(expr=Identifier(name="req", span=span), span=span),
            op=ImplicationOperator.NON_OVERLAPPED,
            consequent=OpaqueSequence(text="ack and", span=span),
            span=span,
        ),
        span=span,
    )
    structure = SVAASTParser()._build_structure(spec, raw_code)
    return raw_code, structure


def _json_payload(text: str) -> dict[str, object]:
    return json.loads(text[text.index("{") :])


def test_adversarial_corpus_files_surface_real_rtl_edges_cleanly(sva_corpus_file, runner) -> None:
    # R3 regression: real-world comments, strings, and attributes must stay parseable from disk.
    # R10 regression: supported preprocessor directives must remain trivia instead of requiring hand preprocessing.
    # R14 regression: encrypted-IP markers must fail with a clear parse error instead of crashing the CLI.
    expected = {
        "with_line_comments.sv": ExitCode.SUCCESS,
        "with_block_comments.sv": ExitCode.SUCCESS,
        "with_backtick_directives.sv": ExitCode.SUCCESS,
        "with_string_literals.sv": ExitCode.SUCCESS,
        "with_attributes.sv": ExitCode.SUCCESS,
        "with_encrypted_ip.sv": ExitCode.PARSE_ERROR,
    }

    for fixture_name, expected_code in expected.items():
        path = sva_corpus_file(fixture_name)
        result = runner.invoke(main, ["parse", str(path), "--format", "json"], prog_name="sva")
        assert result.exit_code == expected_code, fixture_name

        if expected_code is ExitCode.SUCCESS:
            payload = _json_payload(result.output)
            assert payload["kind"] == "property"
            assert payload["clocking"]["signal"]["name"] == "clk"
        else:
            assert "encrypted preprocessor regions are not supported" in result.stderr


def test_adversarial_corpus_exercises_required_token_and_trivia_kinds(sva_corpus_file) -> None:
    # R3 regression: the lexer must keep accepting comment- and string-heavy assertion files.
    # R10 regression: directive-heavy files must continue to tokenize after preprocessing.
    line_tokens, line_trivia = tokenize_with_trivia(sva_corpus_file("with_line_comments.sv").read_text(encoding="utf-8"))
    block_tokens, block_trivia = tokenize_with_trivia(sva_corpus_file("with_block_comments.sv").read_text(encoding="utf-8"))
    directive_tokens, directive_trivia = tokenize_with_trivia(
        sva_corpus_file("with_backtick_directives.sv").read_text(encoding="utf-8")
    )
    string_tokens = tokenize(sva_corpus_file("with_string_literals.sv").read_text(encoding="utf-8"))
    _, attribute_trivia = tokenize_with_trivia(sva_corpus_file("with_attributes.sv").read_text(encoding="utf-8"))

    assert any(item.kind is TriviaKind.COMMENT_LINE for item in line_trivia)
    assert any(item.kind is TriviaKind.COMMENT_BLOCK for item in block_trivia)
    assert any(item.kind is TriviaKind.DIRECTIVE for item in directive_trivia)
    assert any(token.kind is TokenKind.STRING for token in string_tokens)
    assert any(item.kind is TriviaKind.ATTRIBUTE for item in attribute_trivia)
    assert directive_tokens[0].kind is TokenKind.ASSERT
    assert line_tokens[-1].kind is TokenKind.EOF


def test_cli_parse_logs_recover_warning_for_opaque_defaults(runner) -> None:
    # R2 regression: recover=True parser downgrades must remain visible to CLI users.
    property_text = (
        "property p(int DELAY = (foo and)); "
        "@(posedge clk) disable iff (!rst_n) req |-> ##1 ack; "
        "endproperty"
    )

    result = runner.invoke(main, ["parse", property_text, "--format", "json"], prog_name="sva")

    assert result.exit_code == ExitCode.SUCCESS
    payload = _json_payload(result.output)
    assert payload["formals"][0]["default"]["kind"] == "opaque"
    assert "parser recover=True downgraded to opaque_expr" in result.stderr


def test_describe_layers_mark_unverified_fragments_when_opaque_nodes_reach_them() -> None:
    # R2 regression: once an opaque node reaches describe, both SVAD and CoT must surface uncertainty explicitly.
    raw_code, structure = _opaque_inner_sequence_case()

    svad = SVADTranslator(parser=_StaticStructureParser(structure)).translate(raw_code)
    cot = SVACoTBuilder(parser=_StaticStructureParser(structure)).build(raw_code)

    assert "[unverified]" in svad
    assert "opaque parser fallback nodes" in svad
    assert "[unverified]" in cot
    assert "low-confidence passthrough text" in cot


def test_timing_extract_surfaces_unsupported_status_via_exit_code_and_summary(runner, tmp_path: Path) -> None:
    # R9 regression: lossy or unsupported timing extraction must stay visible at the CLI boundary.
    input_file = tmp_path / "unsupported.sv"
    input_file.write_text(
        "property p_accept; @(posedge clk) disable iff (!rst_n) accept_on(abort) req |-> ack; endproperty",
        encoding="utf-8",
    )

    result = runner.invoke(main, ["timing", "extract-sva", str(input_file)], prog_name="sva")

    assert result.exit_code == ExitCode.LOSSY_EXTRACTION
    assert result.stdout == ""
    assert "overall: unsupported" in result.stderr
    assert "Diagnostics summary: lossy_extraction=1" in result.stderr


def test_vendor_extensions_and_uvm_macros_fail_clearly(runner) -> None:
    # R11 regression: vendor-specific assertion syntax must fail clearly instead of silently parsing as supported SVA.
    vendor_result = runner.invoke(
        main,
        ["parse", "cover property (@(posedge clk) disable iff (!rst_n) req |-> ack) option.weight = 2;"],
        prog_name="sva",
    )
    assert vendor_result.exit_code == ExitCode.PARSE_ERROR
    assert "expected EOF" in vendor_result.stderr

    # R12 regression: UVM/OVL-style macros remain out of scope, but the failure must stay explicit and typed.
    uvm_result = runner.invoke(
        main,
        ["parse", '`uvm_info("tag", "msg", UVM_LOW)\nassert property (@(posedge clk) req |-> ack);'],
        prog_name="sva",
    )
    assert uvm_result.exit_code == ExitCode.PARSE_ERROR
    assert "unexpected character '`'" in uvm_result.stderr
