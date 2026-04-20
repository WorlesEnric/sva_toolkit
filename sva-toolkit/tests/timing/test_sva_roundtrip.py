from __future__ import annotations

import re

import pytest

from sva_toolkit.sva.emitter import emit_property_surface
from sva_toolkit.sva.parser import parse_property_text
from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
from sva_toolkit.timing.bridge.from_sva import extract_sva_scenario
from sva_toolkit.timing.core.scenario import ExtractionStatus


_PROPERTY_BLOCK_RE = re.compile(r"property\b.*?endproperty", re.DOTALL)


def _normalize_sva_text(text: str) -> str:
    return emit_property_surface(parse_property_text(text.strip()))


def _normalized_emitted_properties(text: str) -> tuple[str, ...]:
    blocks = tuple(match.group(0) for match in _PROPERTY_BLOCK_RE.finditer(text))
    assert blocks, f"expected at least one emitted property block, got: {text!r}"
    return tuple(_normalize_sva_text(block) for block in blocks)


@pytest.mark.parametrize(
    "source_sva",
    [
        pytest.param(
            "property p_delay; @(posedge clk) disable iff (!rst_n) req |-> ##[M:N] ack; endproperty",
            id="implication-range",
        ),
        pytest.param(
            "property p_hold; @(posedge clk) disable iff (!rst_n) trigger |-> hold until_with response; endproperty",
            id="hold-until-with",
        ),
        pytest.param(
            "property p_stable; @(posedge clk) disable iff (!rst_n) trigger |-> $stable(data) until_with done; endproperty",
            id="stable-until-with",
        ),
    ],
)
def test_extract_emit_roundtrip_preserves_exact_property_surface(source_sva: str) -> None:
    document, report = extract_sva_scenario(source_sva)

    assert document.effective_status == ExtractionStatus.EXACT
    assert report.worst_status() == ExtractionStatus.EXACT

    emitted = emit_parameterized_sva(document)

    assert _normalized_emitted_properties(emitted) == (_normalize_sva_text(source_sva),)


@pytest.mark.parametrize(
    ("source_sva", "expected_approximation"),
    [
        pytest.param(
            "property p_until; @(posedge clk) disable iff (!rst_n) trigger |-> hold until response; endproperty",
            (
                "@(posedge clk) disable iff (!rst_n) trigger |-> response",
                "@(posedge clk) disable iff (!rst_n) 1'b1 |-> hold until_with response",
            ),
            id="until-lossy-approximation",
        ),
    ],
)
def test_extract_emit_roundtrip_uses_conservative_approximation_for_lossy_cases(
    source_sva: str,
    expected_approximation: tuple[str, ...],
) -> None:
    document, report = extract_sva_scenario(source_sva)

    assert document.effective_status == ExtractionStatus.LOSSY
    assert report.worst_status() == ExtractionStatus.LOSSY

    emitted = emit_parameterized_sva(document, allow_lossy=True)

    assert _normalized_emitted_properties(emitted) == tuple(
        _normalize_sva_text(surface) for surface in expected_approximation
    )
    assert _normalized_emitted_properties(emitted) != (_normalize_sva_text(source_sva),)
