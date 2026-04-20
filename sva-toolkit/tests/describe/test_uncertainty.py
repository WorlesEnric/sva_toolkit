"""Regression tests for uncertainty surfacing in the describe engine."""

from __future__ import annotations

from pathlib import Path

from sva_toolkit.describe import SVACoTBuilder, SVADTranslator
from sva_toolkit.describe.translator import SUPPORTED_SYSTEM_FUNCTIONS, SVAASTParser
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
from sva_toolkit.sva.lexer import TokenKind, tokenize


class _StaticStructureParser:
    def __init__(self, structure) -> None:
        self._structure = structure

    def parse(self, sva_code: str):
        del sva_code
        return self._structure


def _opaque_inner_sequence_case():
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


def test_translator_marks_opaque_inner_sequences_as_unverified() -> None:
    raw_code, structure = _opaque_inner_sequence_case()
    translator = SVADTranslator(parser=_StaticStructureParser(structure))

    rendered = translator.translate(raw_code)

    assert "[unverified]" in rendered
    assert "- Exp_1: [unverified] ack and" in rendered
    assert "opaque parser fallback nodes" in rendered


def test_cot_builder_adds_low_confidence_paragraph_for_opaque_nodes() -> None:
    raw_code, structure = _opaque_inner_sequence_case()
    builder = SVACoTBuilder(parser=_StaticStructureParser(structure))

    rendered = builder.build(raw_code)
    sections = builder.get_cot_sections(raw_code)

    assert "[unverified]" in rendered
    assert "low-confidence passthrough text" in rendered
    assert any(section.title == "Confidence & Uncertainty" for section in sections)


def test_clean_descriptions_do_not_emit_unverified_marker() -> None:
    property_text = "assert property (@(posedge clk) req |=> ##1 ack);"

    svad = SVADTranslator().translate(property_text)
    cot = SVACoTBuilder().build(property_text)

    assert "[unverified]" not in svad
    assert "[unverified]" not in cot


def test_examples_tree_dollar_identifiers_have_templates() -> None:
    examples_root = Path(__file__).resolve().parents[2] / "examples"
    seen = set()

    for path in examples_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tokens = tokenize(text)
        except Exception:
            continue
        for token in tokens:
            if token.kind is TokenKind.DOLLAR_IDENT:
                seen.add(token.text)

    assert seen
    assert seen <= SUPPORTED_SYSTEM_FUNCTIONS
