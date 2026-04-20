from __future__ import annotations

import pytest

from sva_toolkit.formal import FormalProperty, MissingClockingError, MissingResetError, normalize_property, parse_property
from sva_toolkit.formal.parse import split_property_texts
from sva_toolkit.sva.ast import PropertySpec
from sva_toolkit.sva.errors import SvaSyntaxError


def test_parse_property_requires_clocking_when_source_and_override_absent() -> None:
    with pytest.raises(MissingClockingError, match="does not name a clocking event"):
        parse_property("disable iff (!rst_n) req |-> ##1 gnt")


def test_parse_property_requires_reset_when_source_and_override_absent() -> None:
    with pytest.raises(MissingResetError, match="does not name a reset expression"):
        parse_property("@(posedge clk) req |-> gnt")


def test_parse_property_accepts_explicit_clocking_and_reset_unchanged() -> None:
    prop = parse_property("@(posedge clk) disable iff (!rst_n) req |-> ##1 gnt")

    assert prop.body == "req |-> ##1 gnt"
    assert prop.clock_edge == "posedge"
    assert prop.clock_name == "clk"
    assert prop.reset_expr == "!rst_n"
    assert prop.has_explicit_reset is True


def test_parse_property_extracts_negedge_clock() -> None:
    prop = parse_property("@(negedge mclk) disable iff (!rst_n) a |-> b")

    assert prop.body == "a |-> b"
    assert prop.clock_edge == "negedge"
    assert prop.clock_name == "mclk"
    assert prop.reset_expr == "!rst_n"


def test_parse_property_uses_explicit_overrides_when_source_omits_annotations() -> None:
    prop = parse_property(
        "req |-> gnt",
        clock="hclk",
        clock_edge="posedge",
        reset="rst_n == 0",
    )

    assert prop.body == "req |-> gnt"
    assert prop.clock_edge == "posedge"
    assert prop.clock_name == "hclk"
    assert prop.reset_expr == "rst_n == 0"
    assert prop.has_explicit_reset is False


def test_parse_property_handles_property_wrapper() -> None:
    prop = parse_property(
        """
        property req_to_gnt;
          @(posedge clk) disable iff (!rst_n) req |-> gnt;
        endproperty
        """
    )

    assert prop.name == "req_to_gnt"
    assert prop.body == "req |-> gnt"
    assert prop.clock_edge == "posedge"
    assert prop.clock_name == "clk"
    assert prop.reset_expr == "!rst_n"


def test_parse_property_handles_assert_property_wrapper() -> None:
    prop = parse_property("assert property (@(posedge clk) disable iff (!rst_n) req |-> gnt);")

    assert prop.name is None
    assert prop.body == "req |-> gnt"
    assert prop.clock_edge == "posedge"
    assert prop.clock_name == "clk"
    assert prop.reset_expr == "!rst_n"


def test_parse_property_collects_signals_without_reserved_keywords() -> None:
    prop = parse_property("@(posedge clk) disable iff (!rst_n) req and gnt or done")

    assert prop.signals == frozenset({"req", "gnt", "done"})


def test_parse_property_populates_ast_on_primary_parser_path() -> None:
    prop = parse_property("property req_to_gnt; @(posedge clk) disable iff (!rst_n) req |-> gnt; endproperty")

    assert isinstance(prop.ast, PropertySpec)
    assert prop.name == "req_to_gnt"
    assert prop.body == "req |-> gnt"
    assert prop.signals == frozenset({"req", "gnt"})


def test_parse_property_falls_back_to_regex_parser_when_ast_parse_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(text: str, *, recover: bool = False) -> PropertySpec:
        raise SvaSyntaxError(0, "boom", text)

    monkeypatch.setattr("sva_toolkit.formal.parse.parse_property_text", _fail)

    prop = parse_property(
        "@(posedge clk) disable iff (!rst_n) req and gnt or done",
    )

    assert prop.ast is None
    assert prop.body == "req and gnt or done"
    assert prop.clock_edge == "posedge"
    assert prop.clock_name == "clk"
    assert prop.reset_expr == "!rst_n"
    assert prop.signals == frozenset({"req", "gnt", "done"})
    assert prop.has_explicit_reset is True


def test_split_property_texts_splits_named_property_blocks() -> None:
    blocks = split_property_texts(
        """
        property first_prop;
          @(posedge clk) req |-> ack;
        endproperty

        property second_prop;
          @(posedge clk) ack |-> done;
        endproperty
        """
    )

    assert len(blocks) == 2
    assert blocks[0].startswith("property first_prop;")
    assert blocks[1].startswith("property second_prop;")


def test_split_property_texts_splits_assertion_statements() -> None:
    blocks = split_property_texts(
        """
        assert property (@(posedge clk) req |-> ack);
        cover property (@(posedge clk) ack |-> done);
        """
    )

    assert blocks == (
        "assert property (@(posedge clk) req |-> ack);",
        "cover property (@(posedge clk) ack |-> done);",
    )


def test_normalize_property_collapses_whitespace() -> None:
    prop = FormalProperty(body="  req   |->  ##1   gnt ;  ", signals={"req", "gnt"})

    normalized = normalize_property(prop)

    assert normalized.body == "req |-> ##1 gnt"
    assert normalized.signals == frozenset({"req", "gnt"})
    assert normalized.clock_edge is None
    assert normalized.clock_name is None


def test_normalize_property_re_emits_body_from_ast() -> None:
    spec = parse_property("@(posedge clk) disable iff (!rst_n) if (sel) req |=> ##1 ack else not done").ast
    assert spec is not None
    prop = FormalProperty.from_ast(spec).model_copy(update={"body": "(stale text)"})

    normalized = normalize_property(prop)

    assert normalized.body == "if (sel) req |=> ##1 ack else not done"
