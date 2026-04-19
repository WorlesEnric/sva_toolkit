from __future__ import annotations

import pytest

from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.timing.core.conditions import (
    Condition,
    Predicate,
    condition_to_dsl,
    condition_to_sva,
    parse_dsl_condition,
    parse_sva_condition,
)


def test_parse_dsl_condition_parses_simple_predicates() -> None:
    rise = parse_dsl_condition("rise(req)")
    high = parse_dsl_condition("high(valid)")

    assert rise == Condition(kind="predicate", predicate=Predicate(op="rise", signal="req"))
    assert high == Condition(kind="predicate", predicate=Predicate(op="high", signal="valid"))


def test_condition_to_dsl_roundtrip() -> None:
    condition = parse_dsl_condition("high(valid)")

    assert condition_to_dsl(condition) == "high(valid)"


def test_condition_to_sva_emits_rose_function() -> None:
    condition = parse_dsl_condition("rise(req)")

    assert condition_to_sva(condition) == "$rose(req)"


def test_parse_sva_condition_parses_sva_syntax() -> None:
    condition = parse_sva_condition("$rose(req)")

    assert condition.kind == "predicate"
    assert condition.predicate == Predicate(op="rose", signal="req")


def test_parse_sva_condition_falls_back_to_regex_parser_on_ast_syntax_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(text: str, *, recover: bool = False):
        raise SvaSyntaxError(0, "boom", text)

    monkeypatch.setattr("sva_toolkit.sva.parser.parse_expr", _fail)

    condition = parse_sva_condition("$rose(req)")

    assert condition.kind == "predicate"
    assert condition.predicate == Predicate(op="rose", signal="req")


def test_condition_to_sva_supports_legacy_sva_predicate_aliases() -> None:
    condition = Condition(kind="predicate", predicate=Predicate(op="rose", signal="req"))

    assert condition_to_sva(condition) == "$rose(req)"
