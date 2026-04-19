from __future__ import annotations

import pytest

from sva_toolkit.sva import (
    emit_expr,
    emit_property_body,
    emit_property_text,
    emit_sequence,
    parse_expr,
    parse_property_body,
    parse_property_text,
    parse_sequence,
)


@pytest.mark.parametrize(
    ("parser", "emitter", "text"),
    [
        (parse_expr, emit_expr, "req"),
        (parse_expr, emit_expr, "$past(data, 2) == 3"),
        (parse_expr, emit_expr, "a + b * c == d && e || f"),
        (parse_expr, emit_expr, "sel ? a : b"),
        (parse_sequence, emit_sequence, "req ##1 ack"),
        (parse_sequence, emit_sequence, "req ##[1:3] ack"),
        (parse_sequence, emit_sequence, "req [*3]"),
        (parse_sequence, emit_sequence, "first_match(req ##1 ack)"),
        (parse_sequence, emit_sequence, "req intersect ack"),
        (parse_property_body, emit_property_body, "req |-> ack"),
        (parse_property_body, emit_property_body, "req |=> ##1 ack"),
        (parse_property_body, emit_property_body, "prop1 and prop2"),
        (parse_property_body, emit_property_body, "prop1 until_with prop2"),
        (parse_property_body, emit_property_body, "if (cond) req else ack"),
        (parse_property_body, emit_property_body, "not req"),
        (parse_property_text, emit_property_text, "@(posedge clk) disable iff (!rst_n) req |-> ack"),
        (
            parse_property_text,
            emit_property_text,
            "property p; @(posedge clk) disable iff (!rst_n) req |-> ##[1:3] ack; endproperty",
        ),
        (parse_property_text, emit_property_text, "assert property(req |-> ack)"),
    ],
)
def test_parse_emit_parse_emit_is_stable(parser, emitter, text: str) -> None:
    first = emitter(parser(text))
    second = emitter(parser(first))

    assert second == first
