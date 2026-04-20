from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sva_toolkit.formal.model import (
    CheckResult,
    FormalProperty,
    ImplicationResult,
    MissingResetError,
    canonicalize_reset_expr,
    reset_exprs_equivalent,
)
from sva_toolkit.sva import parse_property_text


def test_formal_property_is_frozen_and_normalizes_signals() -> None:
    prop = FormalProperty(body="req |-> gnt", signals={"req", "gnt"})

    assert prop.signals == frozenset({"req", "gnt"})
    assert prop.clock_name is None
    assert prop.clock_edge is None
    assert prop.reset_expr is None

    with pytest.raises(FrozenInstanceError):
        prop.body = "a |-> b"  # type: ignore[misc]


def test_formal_property_from_ast_derives_compatibility_fields() -> None:
    spec = parse_property_text(
        "property p(input int depth = 2); @(posedge clk) disable iff (!rst_n) req |-> depth && gnt; endproperty"
    )

    prop = FormalProperty.from_ast(spec)

    assert prop.name == "p"
    assert prop.body == "req |-> depth && gnt"
    assert prop.clock_edge == "posedge"
    assert prop.clock_name == "clk"
    assert prop.reset_expr == "!rst_n"
    assert prop.signals == frozenset({"req", "gnt"})
    assert prop.has_explicit_reset is True
    assert prop.reset_name == "rst_n"
    assert prop.reset_sense == "low"


def test_reset_properties_require_an_explicit_reset_expression() -> None:
    prop = FormalProperty(body="req |-> gnt")

    with pytest.raises(MissingResetError):
        _ = prop.reset_name

    with pytest.raises(MissingResetError):
        _ = prop.reset_sense


def test_reset_expression_semantic_comparator_handles_active_low_aliases() -> None:
    assert reset_exprs_equivalent("!rst_n", "rst_n == 0")
    assert reset_exprs_equivalent("!rst_n", "0 == rst_n")
    assert canonicalize_reset_expr("!rst_n") == "rst_n == 0"
    assert canonicalize_reset_expr("rst_n != 1") == "rst_n == 0"


def test_implication_result_enum_values() -> None:
    assert ImplicationResult.IMPLIES.value == "implies"
    assert ImplicationResult.NOT_IMPLIES.value == "not_implies"
    assert ImplicationResult.EQUIVALENT.value == "equivalent"
    assert ImplicationResult.TIMEOUT.value == "timeout"
    assert ImplicationResult.ERROR.value == "error"
    assert ImplicationResult.SYNTAX_ERROR.value == "syntax_error"


def test_check_result_creation() -> None:
    result = CheckResult(
        result=ImplicationResult.IMPLIES,
        message="proved",
        counterexample=None,
        log="formal log",
        module="top",
    )

    assert result.result is ImplicationResult.IMPLIES
    assert result.message == "proved"
    assert result.counterexample is None
    assert result.log == "formal log"
    assert result.module == "top"
