from __future__ import annotations

import pytest

from sva_toolkit.sva import (
    Bind,
    CheckerDecl,
    ClockingDecl,
    Expect,
    LetDecl,
    Restrict,
    SequenceDecl,
    emit_property_body,
    emit_property_text,
    emit_sequence,
    parse_property_body,
    parse_property_text,
    parse_sequence,
)
from sva_toolkit.sva.ast import ClockingEvent, Ended, Matched, MultiEventClocking, RepeatOperator, Within
from sva_toolkit.sva.diagnostics import ParserDiagnostics
from sva_toolkit.sva.emitter import emit_declaration
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.parser import parse_declaration_text


@pytest.fixture(autouse=True)
def reset_parser_diagnostics() -> None:
    ParserDiagnostics.reset()
    yield
    ParserDiagnostics.reset()


def test_sequence_declaration_round_trip() -> None:
    node = parse_declaration_text("sequence s; req within ack; endsequence")

    assert isinstance(node, SequenceDecl)
    assert isinstance(node.body, Within)
    assert emit_declaration(node) == "sequence s; req within ack; endsequence"
    assert ParserDiagnostics.opaque_count() == 0


def test_checker_declaration_round_trip() -> None:
    node = parse_declaration_text("checker c; sequence s; req [*]; endsequence endchecker")

    assert isinstance(node, CheckerDecl)
    assert len(node.items) == 1
    assert isinstance(node.items[0], SequenceDecl)
    assert emit_declaration(node) == "checker c; sequence s; req [*]; endsequence endchecker"
    assert ParserDiagnostics.opaque_count() == 0


def test_let_and_bind_round_trip() -> None:
    let_node = parse_declaration_text("let sample(input logic req) = req inside {1, 2};")
    bind_node = parse_declaration_text("bind dut chk my_checker(req, ack);")

    assert isinstance(let_node, LetDecl)
    assert emit_declaration(let_node) == "let sample(input logic req) = req inside {1, 2};"

    assert isinstance(bind_node, Bind)
    assert bind_node.instance_name == "chk"
    assert bind_node.checker_name == "my_checker"
    assert emit_declaration(bind_node) == "bind dut chk my_checker(req, ack);"


def test_clocking_declarations_round_trip() -> None:
    node = parse_declaration_text("clocking cb @(posedge clk or negedge rst_n); endclocking")
    default_node = parse_declaration_text("default clocking @(posedge clk); endclocking")

    assert isinstance(node, ClockingDecl)
    assert isinstance(node.event, MultiEventClocking)
    assert emit_declaration(node) == "clocking cb @(posedge clk or negedge rst_n); endclocking"

    assert isinstance(default_node, ClockingDecl)
    assert default_node.default is True
    assert isinstance(default_node.event, ClockingEvent)
    assert emit_declaration(default_node) == "default clocking @(posedge clk); endclocking"


def test_restrict_and_expect_round_trip() -> None:
    restrict_spec = parse_property_text("restrict property(@(posedge clk) req |-> ack)")
    expect_spec = parse_property_text("expect (@(posedge clk) req |-> ack)")
    restrict_body = parse_property_body("restrict property(req)")
    expect_body = parse_property_body("expect (req)")

    assert restrict_spec.statement_kind is not None
    assert restrict_spec.statement_kind.value == "restrict"
    assert emit_property_text(restrict_spec) == "restrict property(@(posedge clk) req |-> ack)"

    assert expect_spec.statement_kind is not None
    assert expect_spec.statement_kind.value == "expect"
    assert emit_property_text(expect_spec) == "expect (@(posedge clk) req |-> ack)"

    assert isinstance(restrict_body, Restrict)
    assert emit_property_body(restrict_body) == "restrict property(req)"
    assert isinstance(expect_body, Expect)
    assert emit_property_body(expect_body) == "expect (req)"


def test_sequence_helpers_and_repetition_round_trip() -> None:
    matched = parse_sequence("matched(req ##1 ack) within ended(done)")
    repeated = parse_sequence("req [+] ##[1:$] ack")

    assert isinstance(matched, Within)
    assert isinstance(matched.left, Matched)
    assert isinstance(matched.right, Ended)
    assert emit_sequence(matched) == "matched(req ##1 ack) within ended(done)"

    assert emit_sequence(repeated) == "req [+] ##[1:$] ack"
    assert repeated.left.op is RepeatOperator.ONE_OR_MORE
    assert repeated.delay.unbounded is True


def test_property_local_variable_types_are_preserved() -> None:
    spec = parse_property_text(
        "property p; local var logic seen = 1'b0; local var state_t state; @(posedge clk) req |-> ack; endproperty"
    )

    assert [local.type_text for local in spec.local_vars] == ["logic", "state_t"]
    assert emit_property_text(spec) == (
        "property p; local var logic seen = 1'b0; local var state_t state; @(posedge clk) req |-> ack; endproperty"
    )


def test_property_local_variable_requires_explicit_type() -> None:
    with pytest.raises(SvaSyntaxError, match="requires an explicit type"):
        parse_property_text("property p; local var seen = 1'b0; @(posedge clk) req; endproperty")
