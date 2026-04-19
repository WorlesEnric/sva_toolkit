from __future__ import annotations


def test_default_signal_presets_are_curated_and_unique() -> None:
    from sva_toolkit.generate.signal_presets import (
        AXI_SIGNALS,
        DEFAULT_SIGNALS,
        FIFO_SIGNALS,
        HANDSHAKE_SIGNALS,
    )

    assert len(DEFAULT_SIGNALS) >= 150
    assert len(DEFAULT_SIGNALS) == len(set(DEFAULT_SIGNALS))
    assert {"req", "ack", "valid", "ready"}.issubset(DEFAULT_SIGNALS)
    assert AXI_SIGNALS
    assert FIFO_SIGNALS
    assert HANDSHAKE_SIGNALS


def test_sva_nodes_can_render_symbolic_natural_language() -> None:
    from sva_toolkit.generate.types import Implication, Signal

    property_node = Implication(Signal("req"), "|->", Signal("ack"))
    description = property_node.to_natural_language()

    assert "req" in description.lower()
    assert "ack" in description.lower()
