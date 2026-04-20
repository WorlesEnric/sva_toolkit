from __future__ import annotations

from pathlib import Path
import logging
import re

import pytest

from sva_toolkit.sva import OpaqueExpr, OpaqueProperty, OpaqueSequence, parse_expr, parse_property_body, parse_property_text, parse_sequence
from sva_toolkit.sva.diagnostics import ParserDiagnostics
from sva_toolkit.sva.errors import SvaSyntaxError


PROPERTY_BLOCK_RE = re.compile(r"\bproperty\b.*?\bendproperty\b", re.DOTALL)
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "sva"


@pytest.fixture(autouse=True)
def reset_parser_diagnostics() -> None:
    ParserDiagnostics.reset()
    yield
    ParserDiagnostics.reset()


def test_recover_property_logs_warning_and_counts(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sva_toolkit"):
        node = parse_property_body("req |->", recover=True)

    assert isinstance(node, OpaqueProperty)
    assert ParserDiagnostics.snapshot()["opaque_property"] == 1
    assert ParserDiagnostics.snapshot()["fallback_recover_used"] == 1
    assert any("parser recover=True downgraded to opaque_property" in record.message for record in caplog.records)


def test_recover_sequence_and_expr_bump_their_own_counters(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sva_toolkit"):
        sequence = parse_sequence("req [*", recover=True)
        expr = parse_expr("a inside {", recover=True)

    assert isinstance(sequence, OpaqueSequence)
    assert isinstance(expr, OpaqueExpr)
    snapshot = ParserDiagnostics.snapshot()
    assert snapshot["opaque_sequence"] == 1
    assert snapshot["opaque_expr"] == 1
    assert snapshot["fallback_recover_used"] == 2
    assert len(caplog.records) == 2


def test_recover_false_still_raises() -> None:
    with pytest.raises(SvaSyntaxError):
        parse_property_body("req |->", recover=False)


def test_clean_parse_leaves_counters_at_zero(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="sva_toolkit"):
        spec = parse_property_text("@(posedge clk) req |-> ack")

    assert spec.body is not None
    assert ParserDiagnostics.opaque_count() == 0
    assert ParserDiagnostics.snapshot()["fallback_recover_used"] == 0
    assert caplog.records == []


def test_examples_corpus_parses_without_opaque_fallbacks() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.sv")):
        text = path.read_text(encoding="utf-8")
        blocks = PROPERTY_BLOCK_RE.findall(text)
        assert blocks, f"no property blocks found in {path}"
        for block in blocks:
            parse_property_text(block)

    assert ParserDiagnostics.opaque_count() == 0
